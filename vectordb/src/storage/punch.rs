//! Giving back the disk behind bytes nothing points at any more.
//!
//! Both row files -- `vectors/<name>.f32` and `records/<name>.rec` -- are append-only. An
//! update appends a new row and repoints the id at it; the old row becomes dead space. Nothing
//! reused it and nothing rewrote the file, so **a table's files grew with the number of writes
//! rather than with the amount of data**. Measured in a soak at 200 rows/s over a key space of
//! 400,000 rows: after 2.4 hours the vector file held 5.2 GB of which 1.2 GB was live, and the
//! rate -- about 52 GB a day -- did not depend on the table's size at all.
//!
//! The module comment in `vector_file.rs` used to say a rebuild rewrites the file and reclaims
//! the dead rows. No code did that, and the sweep that rebuilds an index only fires when a
//! table's size calls for a *different algorithm*, which for a table of constant size never
//! happens. So on the one workload where dead space accumulates fastest -- steady-state
//! upserts -- nothing ever reclaimed any of it.
//!
//! ## Why a hole and not a rewrite
//!
//! Rewriting the file compacts it properly: dead rows disappear and the file shrinks. It also
//! renumbers every row, and row numbers are held in three places (the record store's
//! `vector_rows`, the table's in-memory map, and the `.rows` sidecar the id map is rebuilt
//! from at startup) plus the record log that replays them. Getting that wrong loses the
//! connection between a record and its vector, and the commit has to be atomic across four
//! files or a crash mid-rename leaves a log pointing into the wrong file.
//!
//! `FALLOC_FL_PUNCH_HOLE` releases the blocks behind a byte range and leaves the file's length
//! alone. Every row keeps its number, so nothing above this module changes, there is nothing
//! to commit atomically, and a crash halfway through leaves some dead ranges punched and some
//! not -- both of which are correct, because the bytes were dead either way.
//!
//! What it does not do is shrink the file's apparent length: a table that has taken a
//! terabyte of writes has a terabyte-long file holding a gigabyte of blocks. `du` tells the
//! truth and `ls -l` does not. That is worth knowing but is not what fills a disk, and a
//! sparse file costs nothing but address space. A real rewrite is still the eventual answer
//! for the length; this is the answer for the space.
//!
//! ## What is safe to punch
//!
//! Only whole filesystem blocks that lie entirely inside dead ranges: a block shared with a
//! live row keeps its data. Rows are never reused once dead, so a range that is dead when the
//! caller reads the live set cannot become live again while the punch runs -- the race that
//! would matter does not exist in this direction.
//!
//! The files are read through an `mmap`, and punching a hole under a live mapping makes the
//! range read as zeros rather than faulting. Nothing reads it: no live id points there. A
//! stale row number would read zeros instead of a stale vector, which is the better of the two
//! wrong answers.

/// A half-open byte range of a file that holds nothing live.
pub type DeadRange = (u64, u64);

/// Punch every whole block covered by `dead`, and return the bytes released.
///
/// `file` must be open for writing. Ranges may be given in any order; each is aligned inward
/// to `blksize` so a block shared with live data is left alone.
#[cfg(target_os = "linux")]
pub fn punch_holes(file: &std::fs::File, dead: &[DeadRange]) -> crate::Result<u64> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::MetadataExt;

    let blksize = file.metadata()?.blksize().max(1);
    let mut released = 0u64;
    for &(start, end) in dead {
        // Inward alignment: the first whole block at or after `start`, up to the last whole
        // block ending at or before `end`.
        let aligned_start = start.div_ceil(blksize) * blksize;
        let aligned_end = (end / blksize) * blksize;
        if aligned_end <= aligned_start {
            continue;
        }
        let len = aligned_end - aligned_start;
        // SAFETY: a libc call on a file descriptor this process owns, with a length and offset
        // inside the file. PUNCH_HOLE with KEEP_SIZE does not change the file's length, so an
        // existing mapping stays valid and in range.
        let rc = unsafe {
            libc::fallocate(
                file.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                aligned_start as libc::off_t,
                len as libc::off_t,
            )
        };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            // EOPNOTSUPP means this filesystem cannot punch holes (tmpfs on old kernels, some
            // network filesystems). That is a reason to stop trying, not a reason to fail a
            // maintenance sweep: the data is untouched and the space is merely still in use.
            if err.raw_os_error() == Some(libc::EOPNOTSUPP) {
                tracing::debug!("punch_holes: filesystem does not support hole punching");
                return Ok(released);
            }
            return Err(crate::Error::Io(err));
        }
        released += len;
    }
    Ok(released)
}

/// Non-Linux: nothing is released and nothing is wrong.
#[cfg(not(target_os = "linux"))]
pub fn punch_holes(_file: &std::fs::File, _dead: &[DeadRange]) -> crate::Result<u64> {
    Ok(0)
}

/// Blocks actually allocated to `path`, in bytes -- what the filesystem is holding, which is
/// what a punched file reports less of. `metadata().len()` reports the apparent length and
/// does not change when a hole is punched.
#[cfg(unix)]
pub fn allocated_bytes(path: &std::path::Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path)
        .map(|m| m.blocks() * 512)
        .unwrap_or(0)
}

/// Nowhere to ask for blocks: the apparent length is the best answer available, and without
/// hole punching it is also the right one.
#[cfg(not(unix))]
pub fn allocated_bytes(path: &std::path::Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Fold a sorted-by-start list of dead ranges into maximal runs, so one `fallocate` covers a
/// run of consecutive dead rows rather than one call per row.
pub fn coalesce(mut ranges: Vec<DeadRange>) -> Vec<DeadRange> {
    if ranges.is_empty() {
        return ranges;
    }
    ranges.sort_unstable();
    let mut out: Vec<DeadRange> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        match out.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => out.push((start, end)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};
    #[cfg(target_os = "linux")]
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn coalescing_merges_touching_and_overlapping_runs() {
        assert_eq!(
            coalesce(vec![(0, 10), (10, 20), (40, 50), (45, 60)]),
            vec![(0, 20), (40, 60)]
        );
        assert_eq!(coalesce(vec![]), Vec::<DeadRange>::new());
        assert_eq!(coalesce(vec![(30, 40), (0, 5)]), vec![(0, 5), (30, 40)]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_punched_range_reads_as_zeros_and_the_length_is_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rows");
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .expect("open");
        let block = file.metadata().expect("meta").blksize() as usize;
        file.write_all(&vec![7u8; block * 4]).expect("write");
        file.sync_all().expect("sync");

        // The middle two blocks are dead; the range given is deliberately unaligned at both
        // ends, so the test also proves the first and last live bytes survive.
        let released = punch_holes(&file, &[(block as u64 - 1, block as u64 * 3 + 1)])
            .expect("punch should work on a tmpdir");
        assert_eq!(released, block as u64 * 2, "two whole blocks");

        let mut back = Vec::new();
        file.seek(SeekFrom::Start(0)).expect("seek");
        file.read_to_end(&mut back).expect("read");
        assert_eq!(back.len(), block * 4, "the length does not change");
        assert!(back[..block].iter().all(|&b| b == 7), "first block is live");
        assert!(
            back[block..block * 3].iter().all(|&b| b == 0),
            "the punched blocks read as zeros"
        );
        assert!(
            back[block * 3..].iter().all(|&b| b == 7),
            "last block is live"
        );
        assert!(
            allocated_bytes(&path) < block as u64 * 4,
            "the filesystem gave the blocks back"
        );
    }
}
