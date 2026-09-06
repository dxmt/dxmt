//! Full vectors on disk, addressed by row.
//!
//! Phase 2 of the disk index (design/DISK_INDEX.md section 4): once a table's graph walks on
//! PQ codes, the only readers of a full vector are the exact rerank of a search's candidates,
//! `with_vector` and `get_record`. Those can read from a file. This is that file: fixed-size
//! rows of `dim` little-endian `f32`s, appended in insertion order, read through a memory map.
//!
//! A rerank wants a few hundred rows scattered across the file. Warm, they are borrowed
//! straight from the mapping, no copy and no syscall. Cold, a page fault per row serialises
//! the disk, so the rows are read with `pread` across the rayon pool instead; [`IoMode`]
//! says how the two are chosen.
//!
//! Beside it, `<name>.rows` records `(id, row)` for every appended row, in the same order and
//! flushed with it, so a restart rebuilds the id-to-row map without the snapshot having to
//! carry it. A row whose id is later re-appended (update) or dropped (delete) becomes dead
//! space; a rebuild rewrites the file in graph order and reclaims it.
//!
//! Durability order per batch: rows appended, `fdatasync`, then the WAL entry is written and
//! the caller acknowledged -- the same "apply, then log, then ack" the record map uses. The
//! WAL entry keeps the vector, so replay can re-append a row the file is missing.

use memmap2::{Mmap, MmapOptions};
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// How a rerank's rows are read.
///
/// Measured on 1M rows (3 GB) after a restart with the page cache dropped (design doc
/// section 4, phase 2): touching the rows through the mapping cost a p99 of 326 ms with the
/// kernel's own fault-around and readahead, and 80 ms with a `madvise(WILLNEED)` per row --
/// the advised reads did not overlap, 500 pages came in at about 80 us each. A page fault is
/// one read at a time; a rerank wants a few hundred at once. So a cold query reads its rows
/// with `pread` across the rayon pool, and a warm one keeps the zero-copy mapping.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IoMode {
    /// Always through the mapping.
    Mmap,
    /// Always parallel `pread`.
    Parallel,
    /// Mapping while it does not fault to disk; parallel reads once it does, re-probing the
    /// mapping every [`PROBE_EVERY`] queries. `getrusage` major-fault counts are the signal:
    /// one syscall per mapped query, exact, and free of guesswork about the page cache.
    Adaptive,
}

/// In adaptive mode, how many parallel-read queries pass between probes of the mapping.
const PROBE_EVERY: u32 = 32;

/// Rows per parallel read task. 500 rows over 16-row tasks is about one task per core.
const PARALLEL_CHUNK: usize = 16;

fn io_mode() -> IoMode {
    static MODE: std::sync::OnceLock<IoMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("DB_VECTOR_IO").ok().as_deref() {
        Some("mmap") => IoMode::Mmap,
        Some("parallel") => IoMode::Parallel,
        _ => IoMode::Adaptive,
    })
}

/// Major page faults taken by the calling thread so far, or `None` where that is not
/// available (adaptive mode then behaves as `Mmap`).
fn major_faults() -> Option<i64> {
    #[cfg(target_os = "linux")]
    {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: `ru` is a valid, writable rusage; RUSAGE_THREAD is Linux-specific.
        if unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut ru) } == 0 {
            return Some(ru.ru_majflt);
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// `DB_VECTOR_PREFETCH=1` issues a `madvise(WILLNEED)` per row before touching the mapping.
/// Off by default: on 1M rows it cost 13% of warm QPS in syscalls and, cold, the advised
/// reads still arrived one at a time (p99 80 ms). The parallel path is what serves cold rows.
fn prefetch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DB_VECTOR_PREFETCH").ok().as_deref() == Some("1"))
}

pub struct VectorFile {
    path: PathBuf,
    dim: usize,
    row_bytes: usize,
    /// Appends go through the file handle; the map is re-created to cover them.
    writer: Mutex<Writer>,
    /// Covers `mapped_rows` rows; rows past it need a remap.
    map: RwLock<Mapped>,
    /// A read-only handle for positional reads, shared by the parallel path.
    reader: File,
    /// Adaptive mode: the last mapped query faulted to disk.
    cold: AtomicBool,
    /// Adaptive mode: parallel queries since the mapping was last probed.
    since_probe: AtomicU32,
}

struct Writer {
    file: File,
    rows_file: File,
    /// Rows written to `file` (counting any that came before this process opened it).
    rows: u64,
}

struct Mapped {
    mmap: Option<Mmap>,
    rows: u64,
}

impl VectorFile {
    /// Open or create `<path>` and `<path>.rows`. Returns the file and the `(id, row)` pairs
    /// it already holds, oldest first, for the caller to fold into its id map.
    pub fn open(path: &Path, dim: usize) -> crate::Result<(Self, Vec<(String, u64)>)> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)?;
        let row_bytes = dim * std::mem::size_of::<f32>();
        let len = file.metadata()?.len();
        if row_bytes > 0 && len % row_bytes as u64 != 0 {
            return Err(crate::Error::Serialization(format!(
                "{}: {} bytes is not a whole number of {}-byte rows",
                path.display(),
                len,
                row_bytes
            )));
        }
        let rows = if row_bytes > 0 {
            len / row_bytes as u64
        } else {
            0
        };

        let rows_path = Self::rows_path(path);
        let rows_file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&rows_path)?;
        let mut ids = Vec::new();
        for line in BufReader::new(File::open(&rows_path)?).lines() {
            let line = line?;
            let Some((row, id)) = line.split_once('\t') else {
                continue;
            };
            let row: u64 = row.parse().map_err(|_| {
                crate::Error::Serialization(format!(
                    "{}: bad row line {line:?}",
                    rows_path.display()
                ))
            })?;
            // A row the vector file does not have is a torn write: the rows line was
            // appended after the data, so this can only be a truncated data file. Refuse,
            // as an unreadable snapshot is refused, rather than serve a record with no vector.
            if row >= rows {
                return Err(crate::Error::Serialization(format!(
                    "{}: names row {row} but {} holds only {rows} rows; the vector file is \
                     behind its row list",
                    rows_path.display(),
                    path.display()
                )));
            }
            ids.push((id.to_string(), row));
        }

        let this = Self {
            path: path.to_path_buf(),
            dim,
            row_bytes,
            writer: Mutex::new(Writer {
                file,
                rows_file,
                rows,
            }),
            map: RwLock::new(Mapped {
                mmap: None,
                rows: 0,
            }),
            reader: File::open(path)?,
            cold: AtomicBool::new(false),
            since_probe: AtomicU32::new(0),
        };
        this.remap(rows)?;
        Ok((this, ids))
    }

    /// The companion `(id, row)` list beside a data file.
    pub fn rows_path(path: &Path) -> PathBuf {
        let mut p = path.as_os_str().to_owned();
        p.push(".rows");
        PathBuf::from(p)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Rows appended so far, dead ones included.
    pub fn rows(&self) -> u64 {
        self.writer.lock().rows
    }

    /// Append one row per vector and return their row numbers, in order. Durable on return.
    pub fn append(&self, items: &[(&str, &[f32])]) -> crate::Result<Vec<u64>> {
        let mut w = self.writer.lock();
        let first = w.rows;
        let mut data = Vec::with_capacity(items.len() * self.row_bytes);
        let mut lines = String::new();
        for (i, (id, v)) in items.iter().enumerate() {
            if v.len() != self.dim {
                return Err(crate::Error::DimensionMismatch {
                    expected: self.dim,
                    actual: v.len(),
                });
            }
            for x in *v {
                data.extend_from_slice(&x.to_le_bytes());
            }
            lines.push_str(&format!("{}\t{}\n", first + i as u64, id));
        }
        w.file.write_all(&data)?;
        w.file.sync_data()?;
        w.rows_file.write_all(lines.as_bytes())?;
        w.rows_file.sync_data()?;
        w.rows += items.len() as u64;
        let rows = w.rows;
        drop(w);
        self.remap(rows)?;
        Ok((first..first + items.len() as u64).collect())
    }

    fn remap(&self, rows: u64) -> crate::Result<()> {
        let mut m = self.map.write();
        if rows == 0 {
            m.mmap = None;
            m.rows = 0;
            return Ok(());
        }
        let file = File::open(&self.path)?;
        let len = rows as usize * self.row_bytes;
        // SAFETY: the file is append-only and never truncated while mapped; rows already
        // written are never modified.
        let mmap = unsafe { MmapOptions::new().len(len).map(&file)? };
        m.mmap = Some(mmap);
        m.rows = rows;
        Ok(())
    }

    /// One row, copied out. `None` past the end.
    pub fn read(&self, row: u64) -> Option<Vec<f32>> {
        let m = self.map.read();
        let mmap = m.mmap.as_ref()?;
        if row >= m.rows {
            return None;
        }
        let start = row as usize * self.row_bytes;
        Some(Self::decode(&mmap[start..start + self.row_bytes]))
    }

    /// Many rows, prefetched together, returned in the order asked. A row past the end is
    /// `None`.
    pub fn read_rows(&self, rows: &[u64]) -> Vec<Option<Vec<f32>>> {
        let m = self.map.read();
        let Some(mmap) = m.mmap.as_ref() else {
            return vec![None; rows.len()];
        };
        // Sorted, so the advice and the touches walk the file forward.
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_unstable_by_key(|&i| rows[i]);
        if prefetch_enabled() {
            self.advise(mmap, &order, rows, m.rows);
        }
        let mut out = vec![None; rows.len()];
        for &i in &order {
            let row = rows[i];
            if row < m.rows {
                let start = row as usize * self.row_bytes;
                out[i] = Some(Self::decode(&mmap[start..start + self.row_bytes]));
            }
        }
        out
    }

    /// Visit many rows without copying them: `f(i, Some(row_i))` for each requested row, in
    /// the order asked, with the floats borrowed straight from the mapping. The prefetch is
    /// the same as `read_rows`; the copy and the allocation per row are not. A rerank of 500
    /// candidates made 500 `Vec<f32>`s of 3 KB per query through `read_rows`, which is what
    /// this exists to avoid.
    ///
    /// `DB_VECTOR_PREFETCH=0` skips the `madvise` pass, for measuring what it costs.
    pub fn with_rows(&self, rows: &[u64], f: impl FnMut(usize, Option<&[f32]>)) {
        let mode = io_mode();
        let parallel = match mode {
            IoMode::Mmap => false,
            IoMode::Parallel => true,
            IoMode::Adaptive => {
                self.cold.load(Ordering::Relaxed)
                    && !self
                        .since_probe
                        .fetch_add(1, Ordering::Relaxed)
                        .is_multiple_of(PROBE_EVERY)
            }
        };
        if parallel {
            self.with_rows_parallel(rows, f);
        } else {
            let before = major_faults();
            self.with_rows_mapped(rows, f);
            if mode == IoMode::Adaptive {
                if let (Some(before), Some(after)) = (before, major_faults()) {
                    self.cold.store(after > before, Ordering::Relaxed);
                    self.since_probe.store(0, Ordering::Relaxed);
                }
            }
        }
    }

    /// Whether adaptive mode currently reads in parallel. For tests and diagnostics.
    pub fn is_cold(&self) -> bool {
        self.cold.load(Ordering::Relaxed)
    }

    /// The rows read with positional reads spread over the rayon pool, then handed to `f`
    /// on the calling thread in the order asked. Allocates a buffer per row; this is the
    /// cold path, where the disk is the cost.
    fn with_rows_parallel(&self, rows: &[u64], mut f: impl FnMut(usize, Option<&[f32]>)) {
        use std::os::unix::fs::FileExt;
        let mapped_rows = self.map.read().rows;
        let bufs: Vec<Option<Vec<f32>>> = rows
            .par_chunks(PARALLEL_CHUNK)
            .flat_map_iter(|chunk| {
                let mut out = Vec::with_capacity(chunk.len());
                let mut buf = vec![0u8; self.row_bytes];
                for &row in chunk {
                    if row < mapped_rows
                        && self
                            .reader
                            .read_exact_at(&mut buf, row * self.row_bytes as u64)
                            .is_ok()
                    {
                        out.push(Some(Self::decode(&buf)));
                    } else {
                        out.push(None);
                    }
                }
                out
            })
            .collect();
        for (i, v) in bufs.iter().enumerate() {
            f(i, v.as_deref());
        }
    }

    fn with_rows_mapped(&self, rows: &[u64], mut f: impl FnMut(usize, Option<&[f32]>)) {
        let m = self.map.read();
        let Some(mmap) = m.mmap.as_ref() else {
            for i in 0..rows.len() {
                f(i, None);
            }
            return;
        };
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_unstable_by_key(|&i| rows[i]);
        if prefetch_enabled() {
            self.advise(mmap, &order, rows, m.rows);
        }
        for &i in &order {
            let row = rows[i];
            if row < m.rows {
                let start = row as usize * self.row_bytes;
                let bytes = &mmap[start..start + self.row_bytes];
                // SAFETY: the mapping is page-aligned and every row starts at a multiple of
                // `row_bytes`, itself a multiple of 4, so the slice is 4-byte aligned; it lies
                // inside the mapping, and the file is never modified in place.
                let floats =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.dim) };
                f(i, Some(floats));
            } else {
                f(i, None);
            }
        }
    }

    #[cfg(unix)]
    fn advise(&self, mmap: &Mmap, order: &[usize], rows: &[u64], mapped_rows: u64) {
        for &i in order {
            let row = rows[i];
            if row < mapped_rows {
                let start = row as usize * self.row_bytes;
                // SAFETY: the range lies inside the mapping; advice does not alter memory.
                unsafe {
                    libc::madvise(
                        mmap.as_ptr().add(start & !4095) as *mut libc::c_void,
                        self.row_bytes + (start & 4095),
                        libc::MADV_WILLNEED,
                    );
                }
            }
        }
    }

    #[cfg(not(unix))]
    fn advise(&self, _mmap: &Mmap, _order: &[usize], _rows: &[u64], _mapped_rows: u64) {}

    fn decode(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(seed: u64, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|d| (seed * 31 + d as u64) as f32 * 0.5)
            .collect()
    }

    #[test]
    fn rows_round_trip_and_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.f32");
        let (file, ids) = VectorFile::open(&path, 8).unwrap();
        assert!(ids.is_empty());
        assert_eq!(file.rows(), 0);
        assert_eq!(file.read(0), None);

        let a = v(1, 8);
        let b = v(2, 8);
        let rows = file
            .append(&[("a", a.as_slice()), ("b", b.as_slice())])
            .unwrap();
        assert_eq!(rows, vec![0, 1]);
        assert_eq!(file.read(1).unwrap(), b);
        let c = v(3, 8);
        assert_eq!(file.append(&[("c", c.as_slice())]).unwrap(), vec![2]);
        let got = file.read_rows(&[2, 0, 7]);
        assert_eq!(got[0].as_ref().unwrap(), &c);
        assert_eq!(got[1].as_ref().unwrap(), &a);
        assert!(got[2].is_none(), "a row past the end is None, not garbage");
        drop(file);

        let (file, ids) = VectorFile::open(&path, 8).unwrap();
        assert_eq!(
            ids,
            vec![
                ("a".to_string(), 0),
                ("b".to_string(), 1),
                ("c".to_string(), 2)
            ]
        );
        assert_eq!(file.rows(), 3);
        assert_eq!(file.read(2).unwrap(), c);
    }

    #[test]
    fn with_rows_borrows_what_read_rows_copies() {
        let dir = tempfile::tempdir().unwrap();
        let (file, _) = VectorFile::open(&dir.path().join("t.f32"), 8).unwrap();
        let a = v(1, 8);
        let b = v(2, 8);
        file.append(&[("a", a.as_slice()), ("b", b.as_slice())])
            .unwrap();
        let mut seen = Vec::new();
        file.with_rows(&[1, 5, 0], |i, row| seen.push((i, row.map(|r| r.to_vec()))));
        assert_eq!(seen.len(), 3);
        let by_i: std::collections::HashMap<usize, Option<Vec<f32>>> = seen.into_iter().collect();
        assert_eq!(by_i[&0].as_ref().unwrap(), &b);
        assert_eq!(by_i[&1], None);
        assert_eq!(by_i[&2].as_ref().unwrap(), &a);
    }

    #[test]
    fn the_parallel_path_reads_what_the_mapping_reads() {
        let dir = tempfile::tempdir().unwrap();
        let (file, _) = VectorFile::open(&dir.path().join("t.f32"), 8).unwrap();
        let items: Vec<Vec<f32>> = (0..100).map(|i| v(i, 8)).collect();
        let ids: Vec<String> = (0..100).map(|i| i.to_string()).collect();
        let refs: Vec<(&str, &[f32])> = ids
            .iter()
            .zip(&items)
            .map(|(id, v)| (id.as_str(), v.as_slice()))
            .collect();
        file.append(&refs).unwrap();
        let rows: Vec<u64> = vec![99, 3, 250, 0, 42, 42, 7];
        let mut mapped = Vec::new();
        file.with_rows_mapped(&rows, |i, r| mapped.push((i, r.map(|r| r.to_vec()))));
        let mut parallel = Vec::new();
        file.with_rows_parallel(&rows, |i, r| parallel.push((i, r.map(|r| r.to_vec()))));
        mapped.sort_by_key(|(i, _)| *i);
        parallel.sort_by_key(|(i, _)| *i);
        assert_eq!(mapped, parallel);
        assert_eq!(parallel[2].1, None, "row 250 is past the end");
        assert_eq!(parallel[0].1.as_ref().unwrap(), &items[99]);
    }

    #[test]
    fn a_wrong_dimension_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let (file, _) = VectorFile::open(&dir.path().join("t.f32"), 4).unwrap();
        assert!(file.append(&[("x", &[1.0, 2.0][..])]).is_err());
        assert_eq!(file.rows(), 0);
    }

    #[test]
    fn a_data_file_behind_its_row_list_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.f32");
        let (file, _) = VectorFile::open(&path, 4).unwrap();
        file.append(&[("a", &[1.0, 2.0, 3.0, 4.0][..])]).unwrap();
        drop(file);
        // Truncate the data file: the rows list still names row 0.
        std::fs::write(&path, b"").unwrap();
        let err = VectorFile::open(&path, 4).err().expect("must refuse");
        assert!(err.to_string().contains("behind its row list"), "{err}");
    }
}
