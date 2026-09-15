//! Where a table's records live: the in-memory map, or a row file on disk.
//!
//! Phase 3 of the disk index (design/DISK_INDEX.md section 4). With the graph on codes and
//! the vectors in a row file, the record map was what remained resident: about 800 bytes a
//! record at 1M for a `Record` with an empty vector -- the struct itself rounded up by the
//! allocator, its `Arc`, the map entry, and the id string kept once more per structure that
//! needed it. A hundred million of those is 80 GB. [`RowStore`] keeps a record on disk and
//! holds in memory only what finding it needs: one copy of the id, the row's offset and
//! length, and a tombstone bit.
//!
//! Rows are appended in insertion order and never reused. An update appends a new row and
//! points the id at it; a delete drops the id. The row it left behind is dead space, and
//! [`RowStore::punch_dead`] gives the blocks back to the filesystem -- see `storage::punch`. Both are written to `<name>.rec.log` before
//! the caller is acknowledged, in the same order as the data, so a restart replays the log
//! into the id table without a snapshot having to carry the records at all.

use crate::types::record::Record;
use crate::types::RecordId;
use crate::utils::id_hash::{bytes_eq, id_hash, IdHashMap};
use memmap2::{Mmap, MmapOptions};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Ids kept once, in a byte arena, addressed by row; a hash table finds a row by id.
///
/// About 44 bytes a row for the short ids benchmarks use: the bytes themselves, a 4-byte
/// offset, and a `(u64, u32)` hash entry with hashbrown's slack. A `HashMap<String, _>`
/// costs a `String` header and a heap allocation per entry on top of the same hash entry.
#[derive(Default)]
struct IdTable {
    bytes: Vec<u8>,
    /// `rows() + 1` entries.
    offsets: Vec<u32>,
    by_hash: IdHashMap<u32>,
    /// Ids whose hash collided with a live different id. Empty in practice.
    overflow: HashMap<String, u32>,
}

impl IdTable {
    fn new() -> Self {
        Self {
            offsets: vec![0],
            ..Default::default()
        }
    }

    fn rows(&self) -> usize {
        self.offsets.len() - 1
    }

    fn id(&self, row: u32) -> &str {
        // Only ever filled from `&str`, so this is valid UTF-8.
        std::str::from_utf8(self.id_bytes(row)).unwrap_or("")
    }

    /// The stored id as bytes, for comparison. `id` runs a UTF-8 validation pass that
    /// equality does not need, and every row lookup confirms its hash hit this way -- 4.0%
    /// of the server's CPU at 1M was in `core::str::from_utf8` because of it.
    #[inline]
    fn id_bytes(&self, row: u32) -> &[u8] {
        let (a, b) = (
            self.offsets[row as usize] as usize,
            self.offsets[row as usize + 1] as usize,
        );
        &self.bytes[a..b]
    }

    /// Append a row for `id` and make it the id's current row.
    fn push(&mut self, id: &str) -> u32 {
        let row = self.rows() as u32;
        self.bytes.extend_from_slice(id.as_bytes());
        self.offsets.push(self.bytes.len() as u32);
        self.point(id, row);
        row
    }

    /// A slot that holds no id: the bytes of rows a compaction dropped. It takes a row number
    /// so the rows after it keep theirs, and it is deliberately not registered in the hash
    /// table -- nothing may ever resolve to it.
    fn push_gap(&mut self) -> u32 {
        let row = self.rows() as u32;
        self.offsets.push(self.bytes.len() as u32);
        row
    }

    fn point(&mut self, id: &str, row: u32) {
        let h = id_hash(id);
        match self.by_hash.get(&h).copied() {
            None => {
                self.by_hash.insert(h, row);
            }
            Some(existing) if bytes_eq(self.id_bytes(existing), id) => {
                self.by_hash.insert(h, row);
            }
            Some(_) => {
                self.overflow.insert(id.to_string(), row);
            }
        }
    }

    fn row_of(&self, id: &str) -> Option<u32> {
        if let Some(&row) = self.by_hash.get(&id_hash(id)) {
            if bytes_eq(self.id_bytes(row), id) {
                return Some(row);
            }
        }
        self.overflow.get(id).copied()
    }

    fn forget(&mut self, id: &str) -> Option<u32> {
        let h = id_hash(id);
        if self
            .by_hash
            .get(&h)
            .is_some_and(|&row| bytes_eq(self.id_bytes(row), id))
        {
            return self.by_hash.remove(&h);
        }
        self.overflow.remove(id)
    }

    fn live(&self) -> usize {
        self.by_hash.len() + self.overflow.len()
    }
}

/// The record file and its log, plus the row-to-bytes index.
struct RecordFile {
    path: PathBuf,
    data: Mutex<File>,
    log: Mutex<File>,
    map: RwLock<Option<Mmap>>,
    /// `rows + 1` entries: row `r` is `data[ends[r]..ends[r + 1]]`.
    ends: Vec<u64>,
}

impl RecordFile {
    fn log_path(path: &Path) -> PathBuf {
        let mut p = path.as_os_str().to_owned();
        p.push(".log");
        PathBuf::from(p)
    }

    fn remap(&self) -> crate::Result<()> {
        let len = *self.ends.last().unwrap_or(&0);
        let mut m = self.map.write();
        *m = if len == 0 {
            None
        } else {
            let file = File::open(&self.path)?;
            // SAFETY: append-only, never truncated or modified in place while mapped.
            Some(unsafe { MmapOptions::new().len(len as usize).map(&file)? })
        };
        Ok(())
    }

    fn read(&self, row: u32) -> Option<Vec<u8>> {
        let m = self.map.read();
        let mmap = m.as_ref()?;
        let (a, b) = (
            *self.ends.get(row as usize)? as usize,
            *self.ends.get(row as usize + 1)? as usize,
        );
        Some(mmap[a..b].to_vec())
    }
}

/// Records on disk: see the module docs.
pub struct RowStore {
    ids: IdTable,
    tombstone: Vec<bool>,
    /// The row of `vectors/<name>.f32` holding each record's vector, or `NO_VECTOR_ROW`.
    /// Eight bytes here replace a `HashMap<String, u64>` entry of about seventy.
    vector_rows: Vec<u64>,
    file: RecordFile,
}

const NO_VECTOR_ROW: u64 = u64::MAX;

impl RowStore {
    /// Open or create `<path>` and `<path>.log`, replaying the log into the id table.
    pub fn open(path: &Path) -> crate::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)?;
        let data_len = data.metadata()?.len();
        let log_path = RecordFile::log_path(path);
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&log_path)?;

        let mut ids = IdTable::new();
        let mut tombstone = Vec::new();
        let mut vector_rows = Vec::new();
        let mut ends = vec![0u64];
        for line in BufReader::new(File::open(&log_path)?).lines() {
            let line = line?;
            let mut parts = line.splitn(4, '\t');
            match (parts.next(), parts.next(), parts.next(), parts.next()) {
                // `add <len> <vector row or -> <id>`; a log written before the vector row
                // column reads as `add <len> <id>`.
                (Some("add"), Some(len), Some(third), fourth) => {
                    let (vrow, id) = match fourth {
                        Some(id) => (third, id),
                        None => ("-", third),
                    };
                    let len: u64 = len.parse().map_err(|_| {
                        crate::Error::Serialization(format!(
                            "{}: bad line {line:?}",
                            log_path.display()
                        ))
                    })?;
                    let vector_row = if vrow == "-" {
                        NO_VECTOR_ROW
                    } else {
                        vrow.parse().map_err(|_| {
                            crate::Error::Serialization(format!(
                                "{}: bad vector row in {line:?}",
                                log_path.display()
                            ))
                        })?
                    };
                    let end = ends.last().unwrap() + len;
                    if end > data_len {
                        // The log is written after the data, so this is a torn data file.
                        // Refuse, as a truncated vector file and an unreadable snapshot are.
                        return Err(crate::Error::Serialization(format!(
                            "{}: row for {id:?} ends at {end} but {} holds {data_len} bytes; \
                             the record file is behind its log",
                            log_path.display(),
                            path.display()
                        )));
                    }
                    let row = ids.push(id);
                    debug_assert_eq!(row as usize + 1, ends.len());
                    ends.push(end);
                    tombstone.push(false);
                    vector_rows.push(vector_row);
                }
                (Some("del"), Some(id), _, _) => {
                    if let Some(row) = ids.forget(id) {
                        tombstone[row as usize] = true;
                    }
                }
                // `gap <bytes>`: bytes of `<path>` that no row owns any more, skipped without
                // a row being created. Written only by `compact_log`, which drops the dead
                // rows' lines and leaves a gap in their place so the offsets of the rows that
                // remain still land on their own bytes. See that method for why the data file
                // is not rewritten with them.
                (Some("gap"), Some(bytes), _, _) => {
                    let bytes: u64 = bytes.parse().map_err(|_| {
                        crate::Error::Serialization(format!(
                            "{}: bad gap in {line:?}",
                            log_path.display()
                        ))
                    })?;
                    let end = ends.last().unwrap() + bytes;
                    if end > data_len {
                        return Err(crate::Error::Serialization(format!(
                            "{}: gap ends at {end} but {} holds {data_len} bytes",
                            log_path.display(),
                            path.display()
                        )));
                    }
                    ends.push(end);
                    // A gap owns its slot in `ends` but is not a row: nothing may resolve to
                    // it, so it is born tombstoned and with no vector.
                    ids.push_gap();
                    tombstone.push(true);
                    vector_rows.push(NO_VECTOR_ROW);
                }
                _ => {}
            }
        }
        // An id re-appended by an update leaves its older rows dead.
        for row in 0..ids.rows() as u32 {
            if ids.row_of(ids.id(row)) != Some(row) {
                tombstone[row as usize] = true;
            }
        }
        let file = RecordFile {
            path: path.to_path_buf(),
            data: Mutex::new(data),
            log: Mutex::new(log),
            map: RwLock::new(None),
            ends,
        };
        file.remap()?;
        Ok(Self {
            ids,
            tombstone,
            vector_rows,
            file,
        })
    }

    /// The vector file row recorded for `id`, if the record has one.
    pub fn vector_row(&self, id: &str) -> Option<u64> {
        let row = self.ids.row_of(id)?;
        let v = self.vector_rows[row as usize];
        (v != NO_VECTOR_ROW).then_some(v)
    }

    pub fn len(&self) -> usize {
        self.ids.live()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, id: &str) -> bool {
        self.ids.row_of(id).is_some()
    }

    /// Decode the record at `id`'s current row.
    pub fn get(&self, id: &str) -> Option<Arc<Record>> {
        let row = self.ids.row_of(id)?;
        let bytes = self.file.read(row)?;
        rmp_serde::from_slice::<Record>(&bytes).ok().map(Arc::new)
    }

    /// Append `records` as new rows, durably. An id already present is repointed to its new
    /// row and the old one tombstoned.
    pub fn append(&mut self, records: &[(Arc<Record>, Option<u64>)]) -> crate::Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut data = Vec::new();
        let mut log = String::new();
        let mut lens = Vec::with_capacity(records.len());
        for (r, vrow) in records {
            let bytes =
                rmp_serde::to_vec(&**r).map_err(|e| crate::Error::Serialization(e.to_string()))?;
            lens.push(bytes.len() as u64);
            data.extend_from_slice(&bytes);
            match vrow {
                Some(v) => log.push_str(&format!("add\t{}\t{}\t{}\n", bytes.len(), v, r.id)),
                None => log.push_str(&format!("add\t{}\t-\t{}\n", bytes.len(), r.id)),
            }
        }
        {
            let mut f = self.file.data.lock();
            f.write_all(&data)?;
            f.sync_data()?;
        }
        {
            let mut f = self.file.log.lock();
            f.write_all(log.as_bytes())?;
            f.sync_data()?;
        }
        for ((r, vrow), len) in records.iter().zip(lens) {
            if let Some(old) = self.ids.row_of(&r.id) {
                self.tombstone[old as usize] = true;
            }
            let row = self.ids.push(&r.id);
            debug_assert_eq!(row as usize + 1, self.file.ends.len());
            let end = self.file.ends.last().unwrap() + len;
            self.file.ends.push(end);
            self.tombstone.push(false);
            self.vector_rows.push(vrow.unwrap_or(NO_VECTOR_ROW));
        }
        self.file.remap()
    }

    /// Tombstone `id`'s row, durably, returning the record it held.
    pub fn remove(&mut self, id: &str) -> crate::Result<Option<Arc<Record>>> {
        let Some(row) = self.ids.row_of(id) else {
            return Ok(None);
        };
        let record = self.get(id);
        {
            let mut f = self.file.log.lock();
            f.write_all(format!("del\t{id}\n").as_bytes())?;
            f.sync_data()?;
        }
        self.ids.forget(id);
        self.tombstone[row as usize] = true;
        Ok(record)
    }

    pub fn ids(&self) -> Vec<RecordId> {
        (0..self.ids.rows() as u32)
            .filter(|&r| !self.tombstone[r as usize])
            .map(|r| self.ids.id(r).to_string())
            .collect()
    }

    /// Every live record, decoded in row order. A full scan; callers that only need ids or
    /// vectors have cheaper paths.
    pub fn for_each(&self, mut f: impl FnMut(Arc<Record>)) {
        for row in 0..self.ids.rows() as u32 {
            if self.tombstone[row as usize] {
                continue;
            }
            if let Some(bytes) = self.file.read(row) {
                if let Ok(record) = rmp_serde::from_slice::<Record>(&bytes) {
                    f(Arc::new(record));
                }
            }
        }
    }

    /// Give back the blocks behind rows an update or a delete left behind, and say how many
    /// bytes that released. See `storage::punch`.
    ///
    /// A tombstoned row is dead for good -- rows are never reused -- so this needs no
    /// agreement with a concurrent writer beyond the borrow it already holds.
    pub fn punch_dead(&self) -> crate::Result<u64> {
        let mut dead: Vec<(u64, u64)> = Vec::new();
        for (row, &gone) in self.tombstone.iter().enumerate() {
            if gone {
                if let (Some(&a), Some(&b)) = (self.file.ends.get(row), self.file.ends.get(row + 1))
                {
                    dead.push((a, b));
                }
            }
        }
        let data = self.file.data.lock();
        crate::storage::punch::punch_holes(&data, &crate::storage::punch::coalesce(dead))
    }

    /// Rewrite the log with only the live rows, and a `gap` in place of each run of dead ones.
    /// Returns the bytes it held before and after.
    ///
    /// The log takes a line per append, and an upsert appends: 303 MB of it for 400,000 live
    /// records after seventeen hours at 200 rows/s, growing without bound. Unlike the data file
    /// it has no dead regions to punch -- every line is well-formed, merely superseded.
    ///
    /// **Why the data file is not rewritten with it.** The log's `len` fields are accumulated
    /// at startup into the byte offset of each row, so dropping a line moves every row after it
    /// unless something takes its place. A `gap` line is that something: it advances the
    /// accumulator by the dead bytes without creating a row, which leaves the surviving rows
    /// pointing at their own bytes in a data file nothing has touched. So this commits with one
    /// `rename` of one file -- a crash leaves either the old log or the new one, both of which
    /// describe the same data file correctly -- where rewriting the pair would need a commit
    /// atomic across two files, and a crash between them would leave a log whose offsets name
    /// the wrong bytes.
    ///
    /// The dead bytes themselves are given back by [`Self::punch_dead`], which is the same
    /// division of labour the vector file uses: the blocks go back to the filesystem, the
    /// file's length stays.
    ///
    /// Compaction does renumber rows -- the survivors close up -- but only in the file. The
    /// in-memory state keeps the numbering it has and stays correct; the new numbering applies
    /// from the next restart, when everything that reads it is rebuilt from this log together.
    pub fn compact_log(&self) -> crate::Result<(u64, u64)> {
        use std::io::BufWriter;

        let path = RecordFile::log_path(&self.file.path);
        let before = crate::storage::punch::allocated_bytes(&path);

        let mut out = String::new();
        let mut gap = 0u64;
        for row in 0..self.tombstone.len() {
            let (Some(&start), Some(&end)) = (self.file.ends.get(row), self.file.ends.get(row + 1))
            else {
                break;
            };
            if self.tombstone[row] {
                gap += end - start;
                continue;
            }
            if gap > 0 {
                out.push_str(&format!(
                    "gap	{gap}
"
                ));
                gap = 0;
            }
            let id = self.ids.id(row as u32);
            match self.vector_rows[row] {
                NO_VECTOR_ROW => out.push_str(&format!(
                    "add	{}	-	{}
",
                    end - start,
                    id
                )),
                v => out.push_str(&format!(
                    "add	{}	{}	{}
",
                    end - start,
                    v,
                    id
                )),
            }
        }
        if gap > 0 {
            out.push_str(&format!(
                "gap	{gap}
"
            ));
        }

        // Written, synced and renamed while the log lock is held, so no append can land
        // between the last line read above and the swap and be lost.
        let mut log = self.file.log.lock();
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".compacting");
        let tmp = PathBuf::from(tmp);
        {
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)?;
            let mut w = BufWriter::new(f);
            w.write_all(out.as_bytes())?;
            w.flush()?;
            w.into_inner()
                .map_err(|e| crate::Error::Serialization(format!("{}: {e}", tmp.display())))?
                .sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        *log = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;

        Ok((before, crate::storage::punch::allocated_bytes(&path)))
    }

    /// Bytes the append log holds. One line per append, so an upsert adds another.
    pub fn log_bytes(&self) -> u64 {
        crate::storage::punch::allocated_bytes(&RecordFile::log_path(&self.file.path))
    }

    /// The vector-file rows the live records point at.
    pub fn live_vector_rows(&self) -> std::collections::HashSet<u64> {
        let mut live = std::collections::HashSet::with_capacity(self.ids.live());
        for row in 0..self.tombstone.len() {
            if !self.tombstone[row] {
                let v = self.vector_rows[row];
                if v != NO_VECTOR_ROW {
                    live.insert(v);
                }
            }
        }
        live
    }

    /// Rows an update or a delete left behind. O(1): every row is either live or dead, and
    /// the id table counts the live ones. The sweep asks this before deciding whether to walk
    /// the file, so it must not itself be a walk.
    pub fn dead_rows(&self) -> u64 {
        let rows = self.tombstone.len() as u64;
        rows.saturating_sub(self.ids.live() as u64)
    }

    /// Bytes this store holds in memory, from its containers.
    pub fn resident_bytes(&self) -> usize {
        self.ids.bytes.len()
            + self.ids.offsets.len() * 4
            + self.ids.by_hash.capacity() * 16
            + self.tombstone.len()
            + self.vector_rows.len() * 8
            + self.file.ends.len() * 8
    }
}

/// A table's records, wherever they live.
pub enum RecordStore {
    Memory(HashMap<RecordId, Arc<Record>>),
    Disk(Box<RowStore>),
}

impl RecordStore {
    pub fn memory() -> Self {
        RecordStore::Memory(HashMap::new())
    }

    pub fn is_on_disk(&self) -> bool {
        matches!(self, RecordStore::Disk(_))
    }

    /// Bytes this table's records occupy on disk -- the row file and its log -- or 0 for the
    /// in-memory map.
    pub fn disk_bytes(&self) -> u64 {
        match self {
            RecordStore::Memory(_) => 0,
            // Blocks allocated, not apparent length: `punch_dead` releases blocks without
            // changing the length.
            RecordStore::Disk(s) => [s.file.path.clone(), RecordFile::log_path(&s.file.path)]
                .iter()
                .map(|path| crate::storage::punch::allocated_bytes(path))
                .sum(),
        }
    }

    /// Release the disk behind dead rows, for a store that has any. Returns bytes released.
    pub fn punch_dead(&self) -> crate::Result<u64> {
        match self {
            RecordStore::Memory(_) => Ok(0),
            RecordStore::Disk(s) => s.punch_dead(),
        }
    }

    /// Rewrite the append log without the superseded lines. Returns bytes before and after.
    pub fn compact_log(&self) -> crate::Result<(u64, u64)> {
        match self {
            RecordStore::Memory(_) => Ok((0, 0)),
            RecordStore::Disk(s) => s.compact_log(),
        }
    }

    /// Bytes the append log holds; 0 for the in-memory map, which has none.
    pub fn log_bytes(&self) -> u64 {
        match self {
            RecordStore::Memory(_) => 0,
            RecordStore::Disk(s) => s.log_bytes(),
        }
    }

    /// The vector-file rows a disk store's live records point at; `None` for the in-memory
    /// map, whose owner keeps that mapping itself.
    pub fn live_vector_rows(&self) -> Option<std::collections::HashSet<u64>> {
        match self {
            RecordStore::Memory(_) => None,
            RecordStore::Disk(s) => Some(s.live_vector_rows()),
        }
    }

    /// Rows left behind by updates and deletes; 0 for the in-memory map, which has none.
    pub fn dead_rows(&self) -> u64 {
        match self {
            RecordStore::Memory(_) => 0,
            RecordStore::Disk(s) => s.dead_rows(),
        }
    }

    /// Bytes a disk store keeps resident (ids, offsets, tombstones), or `None` for the map,
    /// whose records a caller has to measure itself.
    pub fn disk_store_resident_bytes(&self) -> Option<usize> {
        match self {
            RecordStore::Memory(_) => None,
            RecordStore::Disk(s) => Some(s.resident_bytes()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            RecordStore::Memory(m) => m.len(),
            RecordStore::Disk(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, id: &str) -> bool {
        match self {
            RecordStore::Memory(m) => m.contains_key(id),
            RecordStore::Disk(s) => s.contains(id),
        }
    }

    /// The record, decoded from disk if that is where it lives.
    pub fn get(&self, id: &str) -> Option<Arc<Record>> {
        match self {
            RecordStore::Memory(m) => m.get(id).cloned(),
            RecordStore::Disk(s) => s.get(id),
        }
    }

    /// Insert one record, with the row of its vector in the vector file if it has one.
    pub fn insert(&mut self, record: Arc<Record>, vector_row: Option<u64>) -> crate::Result<()> {
        match self {
            RecordStore::Memory(m) => {
                m.insert(record.id.clone(), record);
                Ok(())
            }
            RecordStore::Disk(s) => s.append(&[(record, vector_row)]),
        }
    }

    /// The vector row this store remembers for `id`. Only the disk store keeps one; a
    /// table with records in memory keeps its own map.
    pub fn vector_row(&self, id: &str) -> Option<u64> {
        match self {
            RecordStore::Memory(_) => None,
            RecordStore::Disk(s) => s.vector_row(id),
        }
    }

    /// Insert many; on disk that is one append and one fsync.
    pub fn insert_batch(&mut self, records: Vec<(Arc<Record>, Option<u64>)>) -> crate::Result<()> {
        match self {
            RecordStore::Memory(m) => {
                for (r, _) in records {
                    m.insert(r.id.clone(), r);
                }
                Ok(())
            }
            RecordStore::Disk(s) => s.append(&records),
        }
    }

    pub fn remove(&mut self, id: &str) -> crate::Result<Option<Arc<Record>>> {
        match self {
            RecordStore::Memory(m) => Ok(m.remove(id)),
            RecordStore::Disk(s) => s.remove(id),
        }
    }

    pub fn ids(&self) -> Vec<RecordId> {
        match self {
            RecordStore::Memory(m) => m.keys().cloned().collect(),
            RecordStore::Disk(s) => s.ids(),
        }
    }

    /// Every live record. A full scan on disk.
    pub fn for_each(&self, mut f: impl FnMut(Arc<Record>)) {
        match self {
            RecordStore::Memory(m) => {
                for r in m.values() {
                    f(Arc::clone(r));
                }
            }
            RecordStore::Disk(s) => s.for_each(f),
        }
    }

    pub fn values_cloned(&self) -> Vec<Record> {
        let mut out = Vec::with_capacity(self.len());
        self.for_each(|r| out.push((*r).clone()));
        out
    }

    /// What a snapshot carries. Records on disk are their own durable copy, so the snapshot
    /// carries none of them -- which is also why snapshotting a 10M-row table stops taking
    /// two minutes.
    pub fn snapshot_records(&self) -> Vec<Arc<Record>> {
        match self {
            RecordStore::Memory(m) => m.values().cloned().collect(),
            RecordStore::Disk(_) => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::record::RecordStatus;

    fn record(id: &str, v: f32) -> Arc<Record> {
        Arc::new(Record {
            id: id.to_string(),
            vector: Vec::new(),
            sparse_vector: None,
            named_vectors: None,
            payload: serde_json::json!({ "v": v }),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
            status: RecordStatus::Active,
        })
    }

    #[test]
    fn compacting_the_log_keeps_every_live_row_where_it_is() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.rec");
        let mut store = RowStore::open(&path).unwrap();

        // Ten passes over four ids, plus one delete: a line per append, four live rows, and a
        // run of dead ones in between each -- the shape an upsert workload leaves behind.
        for pass in 0..10u32 {
            store
                .append(&[
                    (record("a", pass as f32), Some(pass as u64)),
                    (record("b", pass as f32 + 0.5), None),
                    (record("c", pass as f32 + 0.25), Some(100 + pass as u64)),
                    (record("gone", 0.0), Some(7)),
                ])
                .unwrap();
        }
        store.remove("gone").unwrap();
        assert_eq!(store.len(), 3);

        let before: Vec<(String, Option<u64>, f32)> = ["a", "b", "c"]
            .iter()
            .map(|id| {
                let r = store.get(id).expect("live");
                (
                    id.to_string(),
                    store.vector_row(id),
                    r.payload["v"].as_f64().unwrap() as f32,
                )
            })
            .collect();
        let log = RecordFile::log_path(&path);
        let lines_before = std::fs::read_to_string(&log).unwrap().lines().count();
        assert_eq!(lines_before, 41, "forty appends and a delete");

        let (was, now) = store.compact_log().unwrap();
        let text = std::fs::read_to_string(&log).unwrap();
        let adds = text.lines().filter(|l| l.starts_with("add\t")).count();
        assert_eq!(
            adds, 3,
            "one line per live row, whatever it took to get there"
        );
        assert!(
            text.lines().any(|l| l.starts_with("gap\t")),
            "dead runs became gaps"
        );
        assert!(now <= was, "{was} -> {now} bytes");

        // An append after the rewrite has to reach the file that was renamed into place.
        store.append(&[(record("d", 9.0), Some(42))]).unwrap();

        // The point of the gaps: a reopened store finds the same records, with the same
        // payloads and the same vector rows, in a data file nothing rewrote.
        drop(store);
        let reopened = RowStore::open(&path).unwrap();
        assert_eq!(reopened.len(), 4, "three survivors and the one added after");
        for (id, vrow, v) in &before {
            let r = reopened
                .get(id)
                .unwrap_or_else(|| panic!("{id} lost by the compaction"));
            assert_eq!(
                r.payload["v"].as_f64().unwrap() as f32,
                *v,
                "{id} decoded from the wrong bytes"
            );
            assert_eq!(reopened.vector_row(id), *vrow, "{id} lost its vector row");
        }
        assert_eq!(reopened.get("d").unwrap().payload["v"], 9.0);
        assert_eq!(reopened.vector_row("d"), Some(42));
        assert!(reopened.get("gone").is_none(), "a deleted id stays deleted");
    }

    #[test]
    fn rows_survive_update_delete_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.rec");
        let mut store = RowStore::open(&path).unwrap();
        store
            .append(&[
                (record("a", 1.0), Some(0)),
                (record("b", 2.0), Some(1)),
                (record("c", 3.0), None),
            ])
            .unwrap();
        assert_eq!(store.vector_row("a"), Some(0));
        assert_eq!(store.vector_row("c"), None);
        assert_eq!(store.len(), 3);
        assert_eq!(store.get("b").unwrap().payload["v"], 2.0);

        // Update: a new row, the old one dead.
        store.append(&[(record("b", 20.0), Some(3))]).unwrap();
        assert_eq!(
            store.vector_row("b"),
            Some(3),
            "an update repoints the vector row"
        );
        assert_eq!(store.len(), 3);
        assert_eq!(store.get("b").unwrap().payload["v"], 20.0);

        // Delete.
        let gone = store.remove("a").unwrap().unwrap();
        assert_eq!(gone.payload["v"], 1.0);
        assert!(store.get("a").is_none());
        assert_eq!(store.len(), 2);
        let mut ids = store.ids();
        ids.sort();
        assert_eq!(ids, vec!["b", "c"]);
        drop(store);

        // Reopen: the log replays into the same state.
        let store = RowStore::open(&path).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("b").unwrap().payload["v"], 20.0);
        assert_eq!(
            store.vector_row("b"),
            Some(3),
            "vector rows survive a reopen"
        );
        assert!(store.get("a").is_none());
        let mut seen = Vec::new();
        store.for_each(|r| seen.push(r.id.clone()));
        seen.sort();
        assert_eq!(seen, vec!["b", "c"]);
        assert!(store.resident_bytes() < 260, "{}", store.resident_bytes());
    }

    #[test]
    fn a_data_file_behind_its_log_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.rec");
        let mut store = RowStore::open(&path).unwrap();
        store.append(&[(record("a", 1.0), None)]).unwrap();
        drop(store);
        std::fs::write(&path, b"").unwrap();
        let err = RowStore::open(&path).err().expect("must refuse");
        assert!(err.to_string().contains("behind its log"), "{err}");
    }

    #[test]
    fn the_enum_behaves_the_same_in_memory_and_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        for mut store in [
            RecordStore::memory(),
            RecordStore::Disk(Box::new(RowStore::open(&dir.path().join("t.rec")).unwrap())),
        ] {
            store
                .insert_batch(vec![
                    (record("x", 1.0), Some(0)),
                    (record("y", 2.0), Some(1)),
                ])
                .unwrap();
            store.insert(record("x", 3.0), Some(2)).unwrap();
            assert_eq!(store.len(), 2);
            assert!(store.contains("y"));
            assert_eq!(store.get("x").unwrap().payload["v"], 3.0);
            assert!(store.remove("y").unwrap().is_some());
            assert_eq!(store.values_cloned().len(), 1);
            assert_eq!(store.ids(), vec!["x"]);
            assert_eq!(
                store.snapshot_records().len(),
                if store.is_on_disk() { 0 } else { 1 }
            );
        }
    }
}
