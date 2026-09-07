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

/// Rows a probe touches through the mapping. A probe that mapped the whole query paid up to
/// 500 readahead-inflated faults when it guessed wrong -- on NVMe a cold p99 of 29 ms against
/// 7.5 ms for plain parallel reads, and a worst query of 629 ms. Eight faults bound that.
const PROBE_ROWS: usize = 8;

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
/// Whether mapping a file of `bytes` can pay for itself, which needs it to fit in the page
/// cache with room to spare for everything else the process holds.
///
/// Half of the memory the process may use (the host's, or a cgroup limit). The measured gap
/// is wide enough that the exact fraction does not matter: at 10M the rows are 15.4 GB
/// against 121 GB of RAM and mapping is the faster path; at 100M they are 153.6 GB against
/// the same 121 GB, and mapping is 7x the cold tail of parallel reads and still worse warm.
/// Anything between a fifth and the whole of memory separates those two.
///
/// Unknown memory (not Linux) keeps the previous behaviour rather than guessing.
fn mapping_can_pay(bytes: u64) -> bool {
    match crate::storage::memory_guard::total_bytes() {
        Some(total) => bytes <= total / 2,
        None => true,
    }
}

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

/// How a row is stored: four bytes a dimension, or two.
///
/// Half precision halves the file and doubles what the page cache covers. At 100M rows
/// the file is 307 GB in f32 against a ~90 GB cache -- every rerank misses on most of
/// its 500 rows -- and 154 GB in f16. The rerank distances are computed in f32 after
/// decoding; whether the 11-bit mantissa moves recall is measured, not assumed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VectorFormat {
    F32,
    F16,
}

impl VectorFormat {
    fn bytes_per_dim(self) -> usize {
        match self {
            VectorFormat::F32 => 4,
            VectorFormat::F16 => 2,
        }
    }

    /// The file extension a format is stored under.
    pub fn extension(self) -> &'static str {
        match self {
            VectorFormat::F32 => "f32",
            VectorFormat::F16 => "f16",
        }
    }
}

/// IEEE 754 binary16 from f32, round to nearest even; out-of-range magnitudes saturate to
/// the half infinities. Hand-rolled to keep the crate's dependency list where it is.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        // Inf or NaN.
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal or zero in half precision.
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half_mant = m >> shift;
        let remainder = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let mut h = half_mant as u16;
        if remainder > halfway || (remainder == halfway && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = ((e as u32) << 10 | (mant >> 13)) as u16;
    let remainder = mant & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (h & 1) == 1) {
        h += 1; // may carry into the exponent, which is the correct rounding
    }
    sign | h
}

/// f32 from IEEE 754 binary16 bits.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalise.
            let mut m = mant;
            let mut e: i32 = 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((e + 127 - 15) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub struct VectorFile {
    path: PathBuf,
    dim: usize,
    format: VectorFormat,
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
    /// Whether the mapping can pay for itself: false once the rows are too big to be held
    /// in the page cache, which makes adaptive mode plain parallel reads. Re-evaluated on
    /// every remap, because a table crosses the line by growing. Measured on the 100M table
    /// (153.6 GB of rows, 121 GB of RAM), cold p99 / warm p99 over 1,000 accurate queries:
    /// parallel 99 / 11 ms, mmap 697 / 158 ms. The mapping loses in every state there, warm
    /// included, so nothing about it is worth probing for.
    mappable: AtomicBool,
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
        Self::open_with(path, dim, VectorFormat::F32)
    }

    /// Open or create with an explicit row format. The caller keeps the format straight
    /// across reopens (by file extension, see `VectorFormat::extension`).
    pub fn open_with(
        path: &Path,
        dim: usize,
        format: VectorFormat,
    ) -> crate::Result<(Self, Vec<(String, u64)>)> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)?;
        let row_bytes = dim * format.bytes_per_dim();
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
            format,
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
            // Cold until a probe says otherwise: a fresh process has nothing mapped.
            cold: AtomicBool::new(true),
            since_probe: AtomicU32::new(0),
            mappable: AtomicBool::new(true),
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

    pub fn format(&self) -> VectorFormat {
        self.format
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
            match self.format {
                VectorFormat::F32 => {
                    for x in *v {
                        data.extend_from_slice(&x.to_le_bytes());
                    }
                }
                VectorFormat::F16 => {
                    for x in *v {
                        data.extend_from_slice(&f32_to_f16_bits(*x).to_le_bytes());
                    }
                }
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
        // A table crosses the line by growing, so this is decided here rather than at open,
        // where the file is usually empty.
        self.mappable.store(
            mapping_can_pay(rows.saturating_mul(self.row_bytes as u64)),
            Ordering::Relaxed,
        );
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
        Some(self.decode(&mmap[start..start + self.row_bytes]))
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
                out[i] = Some(self.decode(&mmap[start..start + self.row_bytes]));
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
        match io_mode() {
            IoMode::Mmap => self.with_rows_mapped(rows, f),
            IoMode::Parallel => self.with_rows_parallel(rows, f),
            IoMode::Adaptive => self.with_rows_adaptive(rows, f),
        }
    }

    /// Warm: the whole query through the mapping, and it stays warm until a mapped query
    /// takes a major fault. Cold: parallel reads, and every [`PROBE_EVERY`] queries the first
    /// [`PROBE_ROWS`] rows go through the mapping to see whether the disk still answers.
    ///
    /// Neither, on a file too big to be cached: see [`Self::mappable`]. That check is what
    /// makes the fault-driven part safe, because the state here is one bit for the whole
    /// file and a probe only ever samples eight rows of it. On a file that fits, those eight
    /// rows are representative -- the whole thing is either in cache or not. On a file twice
    /// the size of memory they are not, and every clean probe buys a query that maps 500
    /// rows and serialises a major fault for most of them.
    fn with_rows_adaptive(&self, rows: &[u64], mut f: impl FnMut(usize, Option<&[f32]>)) {
        if !self.mappable.load(Ordering::Relaxed) {
            self.with_rows_parallel(rows, f);
            return;
        }
        if !self.cold.load(Ordering::Relaxed) {
            let before = major_faults();
            self.with_rows_mapped(rows, &mut f);
            if let (Some(before), Some(after)) = (before, major_faults()) {
                if after > before {
                    self.cold.store(true, Ordering::Relaxed);
                    self.since_probe.store(0, Ordering::Relaxed);
                }
            }
            return;
        }
        let probe = self
            .since_probe
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(PROBE_EVERY);
        if !probe {
            self.with_rows_parallel(rows, f);
            return;
        }
        // The probe reads its eight rows through the mapping and the rest in parallel,
        // whichever way it comes out: eight rows are enough to tell whether the disk is
        // being touched, and nowhere near enough to bet the other 492 on.
        let (head, tail) = rows.split_at(PROBE_ROWS.min(rows.len()));
        let before = major_faults();
        self.with_rows_mapped(head, &mut f);
        let faulted = match (before, major_faults()) {
            (Some(b), Some(a)) => a > b,
            _ => false,
        };
        self.with_rows_parallel(tail, |i, r| f(i + head.len(), r));
        if !faulted {
            self.cold.store(false, Ordering::Relaxed);
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
                        out.push(Some(self.decode(&buf)));
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
        let mut scratch: Vec<f32> = Vec::new();
        for &i in &order {
            let row = rows[i];
            if row < m.rows {
                let start = row as usize * self.row_bytes;
                let bytes = &mmap[start..start + self.row_bytes];
                match self.format {
                    VectorFormat::F32 => {
                        // SAFETY: the mapping is page-aligned and every row starts at a
                        // multiple of `row_bytes`, itself a multiple of 4, so the slice is
                        // 4-byte aligned; it lies inside the mapping, and the file is never
                        // modified in place.
                        let floats = unsafe {
                            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.dim)
                        };
                        f(i, Some(floats));
                    }
                    VectorFormat::F16 => {
                        // One scratch buffer per query, not one per row.
                        self.decode_into(bytes, &mut scratch);
                        f(i, Some(&scratch));
                    }
                }
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

    fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dim);
        self.decode_into(bytes, &mut out);
        out
    }

    fn decode_into(&self, bytes: &[u8], out: &mut Vec<f32>) {
        out.clear();
        match self.format {
            VectorFormat::F32 => out.extend(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            ),
            VectorFormat::F16 => out.extend(
                bytes
                    .chunks_exact(2)
                    .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            ),
        }
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

    /// Adaptive mode returns exactly what the other two return, in every state, and never
    /// hands a query to the mapping without first bounding what a fault would cost. A page
    /// cache cannot be emptied from a unit test, so this asserts the contract that survives
    /// either answer: same rows, same order, same `None` past the end, warm or cold.
    #[test]
    fn adaptive_reads_agree_with_both_paths_whichever_way_it_guesses() {
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
        let rows: Vec<u64> = vec![99, 3, 250, 0, 42, 42, 7, 11, 12, 13, 14];
        let mut want = Vec::new();
        file.with_rows_mapped(&rows, |i, r| want.push((i, r.map(|r| r.to_vec()))));
        want.sort_by_key(|(i, _)| *i);

        // Both starting states, and enough queries from each to cross PROBE_EVERY.
        for start_cold in [false, true] {
            file.cold.store(start_cold, Ordering::Relaxed);
            file.since_probe.store(0, Ordering::Relaxed);
            for q in 0..(PROBE_EVERY as usize * 2 + 3) {
                let mut got = Vec::new();
                file.with_rows_adaptive(&rows, |i, r| got.push((i, r.map(|r| r.to_vec()))));
                got.sort_by_key(|(i, _)| *i);
                assert_eq!(got, want, "start_cold={start_cold} query {q}");
            }
        }

        // A short query is all probe and must still be complete.
        for start_cold in [false, true] {
            file.cold.store(start_cold, Ordering::Relaxed);
            let short: Vec<u64> = vec![5, 6];
            let mut want_short = Vec::new();
            file.with_rows_mapped(&short, |i, r| want_short.push((i, r.map(|r| r.to_vec()))));
            let mut got = Vec::new();
            file.with_rows_adaptive(&short, |i, r| got.push((i, r.map(|r| r.to_vec()))));
            want_short.sort_by_key(|(i, _)| *i);
            got.sort_by_key(|(i, _)| *i);
            assert_eq!(got, want_short, "short query, start_cold={start_cold}");
        }

        // A file this small is entirely in cache after the writes above, so the next probe
        // cannot fault and adaptive must settle warm rather than flapping. `since_probe` is
        // reset because only every PROBE_EVERY-th cold query probes at all.
        file.cold.store(true, Ordering::Relaxed);
        file.since_probe.store(0, Ordering::Relaxed);
        file.with_rows_adaptive(&rows, |_, _| {});
        #[cfg(target_os = "linux")]
        assert!(
            !file.is_cold(),
            "a fully cached file stayed on the cold path"
        );

        // Rows too big for the page cache are read in parallel whatever the fault counter
        // says, and the mapped/cold state is not consulted or written.
        assert!(
            file.mappable.load(Ordering::Relaxed),
            "100 rows of 8 floats fit in any machine's cache"
        );
        file.mappable.store(false, Ordering::Relaxed);
        file.cold.store(false, Ordering::Relaxed);
        let mut got = Vec::new();
        file.with_rows_adaptive(&rows, |i, r| got.push((i, r.map(|r| r.to_vec()))));
        got.sort_by_key(|(i, _)| *i);
        assert_eq!(got, want, "unmappable file");
        assert!(
            !file.is_cold(),
            "an unmappable file must not touch the mapped/cold state"
        );
        #[cfg(target_os = "linux")]
        assert!(
            !mapping_can_pay(u64::MAX / 4),
            "no machine has u64::MAX/2 bytes of memory"
        );
    }

    #[test]
    fn half_precision_round_trips_within_its_mantissa() {
        for &x in &[
            0.0f32, -0.0, 1.0, -1.0, 0.3333, 65504.0, 1e-5, 6.1e-5, -2.71, 1e-8,
        ] {
            let back = f16_bits_to_f32(f32_to_f16_bits(x));
            let tol = (x.abs() * 1e-3).max(1e-7);
            assert!((back - x).abs() <= tol, "{x} -> {back}");
        }
        assert!(f16_bits_to_f32(f32_to_f16_bits(1e6)).is_infinite());
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
    }

    #[test]
    fn an_f16_file_stores_half_the_bytes_and_reads_back_close() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.f16");
        let (file, _) = VectorFile::open_with(&path, 8, VectorFormat::F16).unwrap();
        let a: Vec<f32> = (0..8).map(|d| (d as f32 - 3.5) * 0.37).collect();
        file.append(&[("a", a.as_slice())]).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 16);
        let back = file.read(0).unwrap();
        for (x, y) in a.iter().zip(&back) {
            assert!((x - y).abs() <= x.abs() * 1e-3 + 1e-6, "{x} vs {y}");
        }
        let mut seen = Vec::new();
        file.with_rows_mapped(&[0], |_, r| seen.push(r.unwrap().to_vec()));
        assert_eq!(seen[0], back);
        let mut par = Vec::new();
        file.with_rows_parallel(&[0], |_, r| par.push(r.unwrap().to_vec()));
        assert_eq!(par[0], back);
        drop(file);
        let (file, ids) = VectorFile::open_with(&path, 8, VectorFormat::F16).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(file.read(0).unwrap(), back);
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
