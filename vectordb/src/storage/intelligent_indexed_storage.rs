#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::search::brute_force::BruteForceIndex;
use crate::search::index_traits::VectorIndex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Mutex as StdMutex};
use tokio::sync::{mpsc, oneshot, RwLock};

use crate::wal::{WalEntry, WalManager};
use tracing::{debug, error, info, warn};

/// The object a primary writes when it takes a bucket; see `claim_primary`.
const PRIMARY_KEY: &str = "wal/PRIMARY";

tokio::task_local! {
    /// Set while a read replica applies entries it read from object storage -- at startup and
    /// in `follow_object_storage`. `ensure_writable` lets exactly those writes through: a
    /// request handler runs in a task without it, so a client's write is refused even while
    /// the follower is applying a batch.
    static APPLYING_REPLICATED: ();
}

/// Longest the table map's lock has been held since the last reset, in microseconds.
///
/// The warnings the guards emit are for operators; this is for tests. Four separate call paths
/// in this file have held that lock across an index search or a disk write, each one turning
/// "one slow operation" into "the whole server stops", and each was found only by measuring a
/// running server. A counter makes the property testable directly: a search may take as long as
/// it likes, but the map lock must be free again long before it finishes.
static MAX_TABLE_LOCK_HOLD_MICROS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

const LOCK_READ: usize = 0;
const LOCK_WRITE: usize = 1;
/// Cumulative nanoseconds spent waiting for, and holding, the table map's lock, and how often
/// it was taken -- `[read, write]`. The log names a single hold that lasted a second; these say
/// how much of the server's time the lock is costing overall, which is what a latency graph
/// that climbs without any one slow request needs.
static TABLE_LOCK_WAIT_NS: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
static TABLE_LOCK_HOLD_NS: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
static TABLE_LOCK_ACQUIRED: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
/// Holds of a second or more: the ones the log warns about.
static TABLE_LOCK_SLOW_HOLDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn record_lock_hold(held: std::time::Duration, mode: usize) {
    let micros = held.as_micros() as u64;
    MAX_TABLE_LOCK_HOLD_MICROS.fetch_max(micros, Ordering::Relaxed);
    TABLE_LOCK_HOLD_NS[mode].fetch_add(held.as_nanos() as u64, Ordering::Relaxed);
    if held >= std::time::Duration::from_secs(1) {
        TABLE_LOCK_SLOW_HOLDS.fetch_add(1, Ordering::Relaxed);
    }
}

fn record_lock_wait(waited: std::time::Duration, mode: usize) {
    TABLE_LOCK_WAIT_NS[mode].fetch_add(waited.as_nanos() as u64, Ordering::Relaxed);
    TABLE_LOCK_ACQUIRED[mode].fetch_add(1, Ordering::Relaxed);
}

/// Longest single hold of the table map's lock since [`reset_max_table_lock_hold`].
///
/// Process-wide rather than per-instance: the point is to catch a code path that holds the lock
/// too long, and which storage instance it ran on does not change that.
pub fn max_table_lock_hold() -> std::time::Duration {
    std::time::Duration::from_micros(MAX_TABLE_LOCK_HOLD_MICROS.load(Ordering::Relaxed))
}

/// Reset the high-water mark. Tests that read it must not run in parallel with each other.
pub fn reset_max_table_lock_hold() {
    MAX_TABLE_LOCK_HOLD_MICROS.store(0, Ordering::Relaxed);
}

/// A write guard over the table map that says how long it was held, and by whom.
///
/// The table map's lock is the one every request goes through, and tokio's `RwLock` is
/// write-preferring: while a writer waits, later readers queue behind it too. So one slow writer
/// stops the whole server, and from outside it looks exactly like a hang. A soak turned up a
/// 97-second stall where the only thing measurable was the *victim* — a snapshot waiting on the
/// lock — with nothing naming the holder. This names the holder.
///
/// Costs one `Instant::now()` per acquisition and logs nothing under a second.
struct TimedWrite<'a> {
    guard: tokio::sync::RwLockWriteGuard<'a, HashMap<String, IntelligentTableData>>,
    who: &'static str,
    since: std::time::Instant,
}

impl std::ops::Deref for TimedWrite<'_> {
    type Target = HashMap<String, IntelligentTableData>;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl std::ops::DerefMut for TimedWrite<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

/// As [`TimedWrite`], for the read side.
///
/// Readers do not block each other, so a long-held read lock looks harmless — until a writer
/// queues behind it. tokio's `RwLock` is write-preferring, so from that moment every later reader
/// queues too, and the server stops answering anything. The 97-second stall had no long *writer*
/// at all; it was one reader holding on while a writer waited behind it.
struct TimedRead<'a> {
    guard: tokio::sync::RwLockReadGuard<'a, HashMap<String, IntelligentTableData>>,
    who: &'static str,
    since: std::time::Instant,
}

impl std::ops::Deref for TimedRead<'_> {
    type Target = HashMap<String, IntelligentTableData>;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl Drop for TimedRead<'_> {
    fn drop(&mut self) {
        let held = self.since.elapsed();
        record_lock_hold(held, LOCK_READ);
        if held >= std::time::Duration::from_secs(1) {
            warn!(
                "{} held the table read lock for {:.1}s; anything that wanted to write queued \
                 behind it, and every later read queued behind that",
                self.who,
                held.as_secs_f64()
            );
        }
    }
}

impl Drop for TimedWrite<'_> {
    fn drop(&mut self) {
        let held = self.since.elapsed();
        record_lock_hold(held, LOCK_WRITE);
        if held >= std::time::Duration::from_secs(1) {
            warn!(
                "{} held the table write lock for {:.1}s; every read and write on every table \
                 queued behind it",
                self.who,
                held.as_secs_f64()
            );
        }
    }
}

pub use crate::storage::index_selector::{
    CostConstraints, HardwareCapabilities, IndexAlgorithm, IndexConfig, IndexDecision,
    IntelligentIndexSelector, PerformanceRequirements,
};

use crate::config::StorageConfig;
use crate::search::sparse_index::SparseIndex;
use crate::search::{IndexCollection, IndexWrapper};
use crate::storage::object::TieredStorage;
use crate::types::vector::Payload;
use crate::types::{
    record::{CreateRecordRequest, Record, SearchHit, SearchMode, SearchQuery, SearchResult},
    record::{
        FullTextSearchHit, FullTextSearchRequest, FullTextSearchResult, FusionMethod,
        HybridSearchRequest,
    },
    table::{Table, TableAlterRequest, TableConfig, TextSource},
    vector::{DenseVector, DistanceMetric, RecordId},
};

#[cfg(feature = "text-bm25")]
use crate::search::full_text::{
    FullTextConfig, FullTextDocument, FullTextSearchEngine,
    FullTextSearchRequest as EngineSearchRequest,
};

/// 快照清单：记住写快照那一刻 WAL 的下一个序号，恢复时只回放这之后的部分。
#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotManifest {
    wal_next_sequence: u64,
    tables: usize,
    /// 每个段对应的表名，下标即段号。
    ///
    /// 段文件按序号命名（`t0`、`t1`……）而不是表名，因为表名来自用户输入、不能直接当文件名。
    /// 但按需回填必须能从表名找到段，所以这里留一份映射。
    #[serde(default)]
    names: Vec<String>,
}

/// How many filtered searches each planner branch has answered since process start.
///
/// A filtered search takes one of five paths, chosen from the estimated selectivity, and they
/// differ by orders of magnitude in cost -- an unfiltered search is ~2 ms at a million rows
/// while a filter matching an eighth to a half of the table is 160-800 ms. Which one a
/// workload lands on is therefore the first thing to know when it is slow, and it was visible
/// only as an `info!` line in the log.
///
/// It is also what makes a test honest. Comparing a filtered search against `search_mode:
/// "exact"` proves nothing if the planner quietly sent both down the same exhaustive path;
/// the comparison passes and the branch under test never ran. Qdrant's filtered-HNSW test
/// asserts on a telemetry counter for exactly this reason. These are those counters.
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct FilterPathCounts {
    /// The match set was small enough to score exactly, without touching the graph.
    pub exact_scan: u64,
    /// `scan_ids`: score the match set by code, then rerank. Low selectivity, larger set.
    pub code_scan: u64,
    /// The graph walked with the filter applied inside it (Acorn-style).
    pub in_graph: u64,
    /// Selectivity 5-30%: an amplified unfiltered fetch, then the filter.
    pub pre_filter: u64,
    /// Selectivity at or above 30%: the same shape, a different amplification.
    pub post_filter: u64,
    /// Nanoseconds spent in each branch, from the selectivity estimate to its result.
    #[serde(default)]
    pub exact_scan_ns: u64,
    #[serde(default)]
    pub code_scan_ns: u64,
    #[serde(default)]
    pub in_graph_ns: u64,
    #[serde(default)]
    pub pre_filter_ns: u64,
    #[serde(default)]
    pub post_filter_ns: u64,
    /// Nanoseconds spent estimating filters and materialising their matching ids, before
    /// any branch ran; part of every branch's time above.
    #[serde(default)]
    pub eval_ns: u64,
    /// Filtered searches that could not use the node map -- its lock or the index's was
    /// held -- and fell back to the owned id set. A steady count here is a lock held too
    /// long somewhere.
    #[serde(default)]
    pub node_map_misses: u64,
}

/// Runtime persistence state exposed to health/readiness endpoints.
///
/// This is deliberately a small, read-only snapshot rather than a promise that every backing
/// store is healthy. A server is ready only when its local durability boundary is usable; object
/// storage mirroring is an optional best-effort copy and is reported separately by its own logs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StorageHealthSnapshot {
    pub ready: bool,
    pub wal_enabled: bool,
    pub wal_directory_accessible: bool,
    pub wal_next_sequence: Option<u64>,
    pub snapshot_present: bool,
    pub snapshot_manifest_valid: bool,
    pub snapshot_wal_next_sequence: Option<u64>,
    /// With object storage, the first WAL sequence it does not hold yet. Equal to
    /// `wal_next_sequence` when the bucket is up to date; the gap is what losing this machine
    /// would lose.
    pub wal_mirrored_next_sequence: Option<u64>,
    /// On a read replica, the first WAL sequence it has not applied yet.
    pub replica_applied_next_sequence: Option<u64>,
    /// On a read replica that can no longer catch up from the WAL chunks, why.
    pub replica_stale: Option<String>,
    pub durability_degraded: bool,
    pub durability_error: Option<String>,
    pub resident_table_count: usize,
    pub table_count: usize,
    pub total_records: usize,
    /// RFC3339 timestamp of the most recent successful local WAL fsync.
    pub last_wal_sync_at: Option<String>,
    /// RFC3339 timestamp of the most recent successfully installed snapshot.
    pub last_snapshot_at: Option<String>,
    /// When the last backup -- manual or scheduled -- finished writing, RFC 3339.
    pub last_backup_at: Option<String>,
    /// Backups that failed since process start.
    pub backup_failures: u64,
    /// Number of snapshot attempts that failed since process start.
    pub snapshot_failures: u64,
    /// Most recent snapshot failure, if any. A failed snapshot does not make the WAL unreadable;
    /// it is nevertheless useful to surface the reason to operators and metrics consumers.
    pub last_snapshot_error: Option<String>,
    /// Which branch answered each filtered search. See [`FilterPathCounts`].
    pub filter_paths: FilterPathCounts,
    /// Where time goes outside a request: the table lock, rebuilds, recovery, snapshots.
    pub timings: OperationTimings,
}

/// Cumulative timings for the work that does not show up as a slow request of its own but
/// makes every request slower: waiting on the table map's lock, index rebuilds, and the two
/// numbers an operator needs after a restart -- how long recovery took and how much log it
/// replayed. Lock figures are process-wide (see `TABLE_LOCK_WAIT_NS`); the rest are this
/// instance's. Nanoseconds; `/metrics` reports seconds.
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OperationTimings {
    pub table_lock_read_wait_ns: u64,
    pub table_lock_write_wait_ns: u64,
    pub table_lock_read_hold_ns: u64,
    pub table_lock_write_hold_ns: u64,
    pub table_lock_reads: u64,
    pub table_lock_writes: u64,
    /// Holds of a second or more.
    pub table_lock_slow_holds: u64,
    pub index_rebuilds: u64,
    pub index_rebuild_ns: u64,
    /// The part of the rebuilds spent waiting for the write lock to swap the new index in.
    pub index_rebuild_swap_wait_ns: u64,
    /// The last recovery at startup, when this instance opened a WAL directory.
    pub last_recovery_ns: Option<u64>,
    pub last_recovery_wal_entries: Option<u64>,
    /// The last snapshot that succeeded.
    pub last_snapshot_ns: Option<u64>,
}

/// 单表快照。索引在同名 `.idx` 文件里；`has_index` 为 false 表示该索引类型没有磁盘格式
/// （例如 BruteForce），恢复时从 `records` 重新插入即可。
///
/// In memory only: what the file holds is [`TableSnapshotFile`].
struct TableSnapshot {
    table: Table,
    algorithm: IndexAlgorithm,
    /// Shared with the live map, not copied out of it.
    ///
    /// serde's `rc` feature serialises an `Arc<T>` as a plain `T`, so the file on disk is
    /// unchanged; what changes is that writing a snapshot of 10M records no longer needs a
    /// second 90 GB of them in memory first (design doc section 34.3).
    records: Vec<Arc<Record>>,
    has_index: bool,
    /// See [`SnapshotHead::indexed_below`].
    indexed_below: Option<u64>,
}

/// What a `tN.meta` file holds, and why it is in two halves.
///
/// The records are bincode, because they are the bulk: a `f32` costs 4 bytes there and 5 in
/// MessagePack, and at 10M×768 that difference is 30 GB. Their type does not change.
///
/// The head does change -- it is where a table's configuration lives -- and bincode is
/// positional, so adding one field to `Table` moved every later byte and turned every
/// existing snapshot into `tag for enum is not valid, found 16`. The server refuses to start
/// on a snapshot it cannot read, by design, so that is a database that does not come up.
/// MessagePack carries field names, so from here a new field with `#[serde(default)]` costs
/// nothing and needs no frozen copy of the old struct.
///
/// Reading the head alone is also cheap: it is the first field, and bincode stops at the
/// length it was given, so a listing does not decode the records.
#[derive(serde::Serialize, serde::Deserialize)]
struct TableSnapshotFile {
    /// MessagePack of [`SnapshotHead`].
    head: Vec<u8>,
    records: Vec<Arc<Record>>,
}

/// The part of a snapshot that describes the table rather than its rows.
#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotHead {
    table: Table,
    algorithm: IndexAlgorithm,
    has_index: bool,
    /// For a table with a row file: every row below this had reached the graph this snapshot
    /// holds (`VectorFile::indexed_below`, read just before the graph was written). Recovery
    /// re-checks the rows at or above it. Absent in snapshots written before 2026-10-09, and
    /// for tables without a row file.
    #[serde(default)]
    indexed_below: Option<u64>,
}

/// Record entries of one table waiting to be replayed together: the table, each record with
/// the row its entry names, and the sequence number of the first.
type ReplayRun = (String, Vec<(Record, Option<u64>)>, u64);

/// The derived indexes of one table as a clean shutdown leaves them, with what the record
/// store looked like at that moment. Read once at the next start and deleted.
const DERIVED_FILE_VERSION: u32 = 1;

#[derive(serde::Serialize)]
struct PersistedDerivedOut<'a> {
    version: u32,
    rows: u32,
    live: usize,
    created: Option<Vec<i64>>,
    derived: &'a DerivedIndexes,
}

#[derive(serde::Deserialize)]
struct PersistedDerivedIn {
    version: u32,
    rows: u32,
    live: usize,
    created: Option<Vec<i64>>,
    derived: DerivedIndexes,
}

/// The name every scheduled backup starts with, so pruning touches only those.
pub const SCHEDULED_BACKUP_PREFIX: &str = "auto-";

/// What [`IntelligentIndexedStorage::restore_backup`] did.
#[derive(Debug)]
pub struct RestoreReport {
    /// Where the store's previous contents went, if it had any.
    pub moved_aside: Option<PathBuf>,
    /// Every table the restored store opened with, and its record count.
    pub tables: Vec<(String, u64)>,
}

/// The current meta layout: a MessagePack head, then bincode records.
const SNAPSHOT_KIND: &str = "table-snapshot-v2";
/// Everything written before 2026-09-24: one bincode `{table, algorithm, records, has_index}`.
/// Read through [`crate::storage::snapshot_v1`], never written again.
const SNAPSHOT_KIND_V1: &str = "table-snapshot";

/// The head of a `tN.meta`, without decoding its records.
fn read_snapshot_head(path: &std::path::Path) -> crate::Result<SnapshotHead> {
    match crate::search::index_persistence::read_kind(path)?.as_str() {
        SNAPSHOT_KIND => {
            #[derive(serde::Deserialize)]
            struct HeadOnly {
                head: Vec<u8>,
            }
            let file: HeadOnly = crate::search::index_persistence::read_index(path, SNAPSHOT_KIND)?;
            decode_head(&file.head)
        }
        SNAPSHOT_KIND_V1 => {
            // `table` and `algorithm` are the first two fields, so this stops before the
            // records too.
            #[derive(serde::Deserialize)]
            struct LegacyHead {
                table: crate::storage::snapshot_v1::TableV1,
                algorithm: crate::storage::snapshot_v1::IndexAlgorithmV1,
            }
            let old: LegacyHead =
                crate::search::index_persistence::read_index(path, SNAPSHOT_KIND_V1)?;
            Ok(SnapshotHead {
                table: old.table.into(),
                algorithm: old.algorithm.into(),
                // Not read, and the one caller that wants it re-reads the whole file.
                has_index: false,
                indexed_below: None,
            })
        }
        other => Err(crate::Error::Serialization(format!(
            "{}: holds a '{}' snapshot, which this build does not know",
            path.display(),
            other
        ))),
    }
}

/// A whole `tN.meta`, records included.
fn read_snapshot_meta(path: &std::path::Path) -> crate::Result<TableSnapshot> {
    match crate::search::index_persistence::read_kind(path)?.as_str() {
        SNAPSHOT_KIND => {
            let file: TableSnapshotFile =
                crate::search::index_persistence::read_index(path, SNAPSHOT_KIND)?;
            let head = decode_head(&file.head)?;
            Ok(TableSnapshot {
                table: head.table,
                algorithm: head.algorithm,
                records: file.records,
                has_index: head.has_index,
                indexed_below: head.indexed_below,
            })
        }
        SNAPSHOT_KIND_V1 => {
            let old: crate::storage::snapshot_v1::TableSnapshotV1 =
                crate::search::index_persistence::read_index(path, SNAPSHOT_KIND_V1)?;
            Ok(TableSnapshot {
                table: old.table.into(),
                algorithm: old.algorithm.into(),
                records: old.records,
                has_index: old.has_index,
                indexed_below: None,
            })
        }
        other => Err(crate::Error::Serialization(format!(
            "{}: holds a '{}' snapshot, which this build does not know",
            path.display(),
            other
        ))),
    }
}

fn decode_head(bytes: &[u8]) -> crate::Result<SnapshotHead> {
    rmp_serde::from_slice(bytes)
        .map_err(|e| crate::Error::Serialization(format!("snapshot head is unreadable: {}", e)))
}

/// What the verification keeps out of a snapshot it has just read end to end.
struct VerifiedMeta {
    table_name: String,
    has_index: bool,
}

/// [`TableSnapshot`] as the verification reads it: every record decoded, none kept.
///
/// Verification exists because compacting the WAL makes the snapshot the only copy, so
/// "written" is not enough -- it has to be readable. It did that by deserialising the whole
/// `TableSnapshot`, which materialises every record in order to look at one boolean at the
/// end. On a million records that was 6.2 GB, and it landed on top of the 8 GB already held:
/// measured as the entire remaining cost of a snapshot, once the record copying and the
/// serialisation buffers were gone (design doc section 34).
///
/// Same fields in the same order, so bincode reads the same bytes; `records` decodes each
/// element and drops it, which is what keeps the guarantee -- a record that will not decode
/// still fails the verification -- at constant memory.
fn verify_snapshot_meta(path: &std::path::Path) -> crate::Result<VerifiedMeta> {
    match crate::search::index_persistence::read_kind(path)?.as_str() {
        SNAPSHOT_KIND => {
            #[derive(serde::Deserialize)]
            struct VerifiedFile {
                head: Vec<u8>,
                /// Never read, and that is the point: the work happens in its `Deserialize`,
                /// which decodes each record and drops it.
                #[allow(dead_code)]
                records: DecodedAndDropped,
            }
            let file: VerifiedFile =
                crate::search::index_persistence::read_index(path, SNAPSHOT_KIND)?;
            let head = decode_head(&file.head)?;
            Ok(VerifiedMeta {
                table_name: head.table.name,
                has_index: head.has_index,
            })
        }
        SNAPSHOT_KIND_V1 => {
            #[derive(serde::Deserialize)]
            struct VerifiedV1 {
                table: crate::storage::snapshot_v1::TableV1,
                #[allow(dead_code)]
                algorithm: crate::storage::snapshot_v1::IndexAlgorithmV1,
                #[allow(dead_code)]
                records: DecodedAndDropped,
                has_index: bool,
            }
            let old: VerifiedV1 =
                crate::search::index_persistence::read_index(path, SNAPSHOT_KIND_V1)?;
            Ok(VerifiedMeta {
                table_name: old.table.name,
                has_index: old.has_index,
            })
        }
        other => Err(crate::Error::Serialization(format!(
            "{}: holds a '{}' snapshot, which this build does not know",
            path.display(),
            other
        ))),
    }
}

/// A `Vec<Record>` decoded for its errors and not for its contents.
struct DecodedAndDropped;

impl<'de> serde::Deserialize<'de> for DecodedAndDropped {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = DecodedAndDropped;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a sequence of records")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                // Each element is decoded, checked by that decoding, and dropped before the
                // next one is read.
                while seq.next_element::<Record>()?.is_some() {}
                Ok(DecodedAndDropped)
            }
        }
        d.deserialize_seq(Visitor)
    }
}

/// Everything a snapshot has to copy out of a table while the map lock is held.
///
/// It is `TableSnapshot` minus `has_index`, because that field is the *result* of writing the
/// index to disk, which now happens after the lock is released.
struct TableTemplate {
    table: Table,
    algorithm: IndexAlgorithm,
    /// Shared with the live map rather than copied out of it; see `records` there.
    records: Vec<Arc<Record>>,
    /// A disk table's rows when the snapshot is mirrored to object storage. The row files are
    /// the durable copy on this machine, but a machine recovering from the object store alone
    /// has no row files, and a snapshot that carries no records would give it an empty table.
    /// Only the list of live rows is taken under the lock; see `read_mirrored_rows`.
    mirrored_rows: Option<MirroredRows>,
    /// For `VectorFile::indexed_below`, read just before the graph is written.
    vector_file: Option<Arc<crate::storage::vector_file::VectorFile>>,
}

/// A disk table's rows as a mirrored snapshot lists them under the lock: each live record row
/// with the row its vector is at, and what is needed to read them afterwards.
struct MirroredRows {
    file: Option<Arc<crate::storage::vector_file::VectorFile>>,
    live: Vec<(u32, Option<u64>)>,
    /// The table's own id-to-vector-row map, which `vector_row` falls back to.
    fallback: HashMap<RecordId, u64>,
    /// Which table these rows belong to: a table dropped and created again under the name
    /// between two chunks has rows of its own under the same numbers.
    created_at: chrono::DateTime<chrono::Utc>,
}

/// 一条待写入索引的记录：id、稠密向量，以及可选的命名向量。
///
/// 索引写入在 `tables` 写锁之外进行，所以要先把这些从表数据里摘出来。
type PendingIndexInsert = (RecordId, DenseVector, Option<HashMap<String, DenseVector>>);

/// Data needed to finish an update after the table-map write lock is released.
/// A table's row file, if it has one, and the `id -> row` map read back from it.
type OpenedVectorFile = (
    Option<Arc<crate::storage::vector_file::VectorFile>>,
    HashMap<RecordId, u64>,
);

struct UpdateArtifacts {
    index: crate::search::IndexWrapper,
    named_indexes: Option<crate::search::IndexCollection>,
    #[cfg(feature = "text-bm25")]
    table: Table,
    record: Record,
    /// The vector for the index, which the stored record may no longer carry.
    vector: Vec<f32>,
    /// The row-file row the vector was appended to, when the table has a row file.
    vector_row: Option<u64>,
    /// Open from before the append until this is dropped, after the graph update.
    _unindexed: Option<crate::storage::vector_file::UnindexedRows>,
    #[cfg(feature = "text-bm25")]
    full_text_engine: Option<Arc<FullTextSearchEngine>>,
}

/// Data needed to finish a delete after the table-map write lock is released.
struct DeleteArtifacts {
    index: crate::search::IndexWrapper,
    /// The named vector spaces, which a delete used to leave alone.
    named_indexes: Option<crate::search::IndexCollection>,
    record_ids: Vec<RecordId>,
    #[cfg(feature = "text-bm25")]
    full_text_engine: Option<Arc<FullTextSearchEngine>>,
}

/// 一个 WAL 分块在对象存储里的位置与覆盖范围。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WalChunkRef {
    /// 完整路径而不是序号推导：分桶方案将来若要改，旧清单仍然指得对。
    path: String,
    first_seq: u64,
    last_seq: u64,
    entries: usize,
}

/// 对象存储上 WAL 的根。
///
/// 没有它就只能 LIST 前缀来发现分块：代价随日志增长，而且分块是不是齐的无从判断。
/// 清单让恢复变成"读一个对象"，并且能校验完整性。
///
/// **One writer, enforced where the store allows it.** Every change a primary makes to the
/// bucket's state is a write of this object conditional on the version it last wrote
/// (`put_manifest`), so a primary another has replaced finds out on its next write and stops,
/// instead of overwriting the new one's manifest. A store that ignores conditions falls back
/// to `wal/PRIMARY` polling (`ensure_still_primary`).
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
struct WalManifest {
    chunks: Vec<WalChunkRef>,
    /// The epoch of the primary that wrote this. A new primary rewrites the manifest with its
    /// own when it starts, which is what changes the version the old one's next write expects.
    #[serde(default)]
    epoch: u64,
    /// The snapshot recovery starts from: its directory under `snapshot/`. Kept here as well
    /// as in `snapshot/CURRENT` so that publishing one is a conditional write: a replaced
    /// primary's late snapshot cannot become the bucket's.
    #[serde(default)]
    snapshot: Option<String>,
    /// 已被快照覆盖、从清单里裁掉的条目数。
    pruned_entries: u64,
    /// 有史以来写入的条目总数。
    ///
    /// 不变式：`chunks 里的条目数之和 + pruned_entries == total_entries`。
    /// 这是完整性校验——少一个分块就对不上，而不是静默少几条记录。
    total_entries: u64,
}

impl WalManifest {
    fn live_entries(&self) -> u64 {
        self.chunks.iter().map(|c| c.entries as u64).sum()
    }

    fn is_consistent(&self) -> bool {
        self.live_entries() + self.pruned_entries == self.total_entries
    }
}

/// `Vec::retain` 的闭包不能同时可变借用外面的 vec，所以收集走这个自由函数。
fn guard_pruned(into: &mut Vec<WalChunkRef>, chunk: &WalChunkRef) {
    into.push(chunk.clone());
}

/// 插入请求（用于 channel-based 架构）
/// What the insertion processor answers one request with: the records as inserted and,
/// parallel to them, the row-file row each vector went to.
type Inserted = (Vec<Record>, Vec<Option<u64>>);
/// One request's share of a coalesced group: which responder it belongs to, then its answer.
type Owned = (usize, Vec<Record>, Vec<Option<u64>>);

struct InsertRequest {
    table_name: String,
    records: Vec<CreateRecordRequest>,
    /// Per record, the row-file row its vector already occupies, when it does: WAL replay
    /// of an entry that points at a row hands the vector in for indexing and this says not
    /// to append it again. Empty when nothing is known, which is every live insert.
    rows_in_file: Vec<Option<u64>>,
    /// The records as inserted and, per record, the row-file row its vector went to
    /// (`None` on a table whose vectors are in memory). The row is what the WAL entry
    /// points at instead of carrying the vector.
    response_tx: oneshot::Sender<crate::Result<Inserted>>,
    /// When the request entered the channel; the processor charges the gap to `queue`.
    enqueued_at: std::time::Instant,
}

/// What an edit to one record asks for, as a struct rather than eight positionals.
///
/// `apply_update` reached eight parameters, four of them `Option` or `bool` — which is the
/// shape with no compiler protection against a swap at the call site, and there are two call
/// sites in two protocols that must not disagree. `Default` is the "change nothing" update, so
/// a caller names only the fields it means.
#[derive(Debug, Default, Clone)]
pub struct RecordUpdate {
    /// `None` keeps the stored dense vector. The sparse and named vectors are kept either way.
    pub vector: Option<DenseVector>,
    /// `None` keeps the stored payload, in replace mode as much as in merge mode. To clear a
    /// payload, send an empty object.
    pub payload: Option<serde_json::Value>,
    /// New text for the table's model to embed, replacing the vector. Exclusive with `vector`.
    pub text: Option<String>,
    /// Merge the payload into the stored one rather than replacing it.
    pub merge: bool,
    /// Create the record when it does not exist, rather than failing.
    pub upsert: bool,
}

/// 智能索引存储引擎 - 自动选择最优索引算法
pub struct IntelligentIndexedStorage {
    /// 表数据
    tables: Arc<RwLock<HashMap<String, IntelligentTableData>>>,

    /// 分层存储：内存 → SSD → 对象存储。快照段以它为持久归宿。
    storage: Arc<TieredStorage>,

    /// 表级冷热判定。key 用表名——策略本身只按字符串 key 记访问，用向量 id 还是表名都行，
    /// 而这里的分层粒度是整张表进出内存，不是单条向量。
    tier_policy: Arc<crate::storage::tier_promotion::TierPromotionPolicy>,

    /// 是否真的配置了对象存储。
    ///
    /// `TieredStorage::new_from_config` 在没有对象存储配置时会退回到临时目录下的
    /// FilesystemStorage——把快照传到临时目录比不传更糟（看着像有异地副本，其实没有），
    /// 所以只有真配了才启用卸载。
    object_storage_enabled: bool,

    /// 智能索引选择器
    index_selector: Arc<IntelligentIndexSelector>,

    /// 硬件能力配置
    hardware_capabilities: HardwareCapabilities,

    /// 性能要求
    performance_requirements: PerformanceRequirements,

    /// 成本约束
    cost_constraints: CostConstraints,

    /// 全局默认内存优化配置
    default_memory_optimization: bool,

    /// Channel-based 插入：发送端（消除锁竞争）
    insert_tx: mpsc::UnboundedSender<InsertRequest>,

    /// 预写日志。`None` 表示不持久化——`new()` 保持这个行为，只有
    /// `new_with_wal()` 会打开它，见该函数的说明。
    wal: Option<Arc<StdMutex<WalManager>>>,

    /// WAL 目录，快照写在它下面的 `snapshot/`。
    wal_dir: Option<PathBuf>,

    /// 快照必须串行。四个调用点能在同一进程里并发：定期快照、淘汰前快照、重建后快照、
    /// `POST /snapshot`。它们共用一个 staging 目录、一套位置命名（`t0`/`t1`…）和同一次
    /// 日志截断，并发跑会互相拆台——后果从"索引被静默丢掉"一路到"日志被截断到落盘快照
    /// 覆盖不到的位置"，也就是静默丢数据。
    snapshotting: Arc<tokio::sync::Mutex<()>>,
    /// One text-index rebuild at a time; see `rebuild_text_index`.
    #[cfg_attr(not(feature = "text-bm25"), allow(dead_code))]
    text_rebuild: tokio::sync::Mutex<()>,

    /// See [`OperationTimings`].
    index_rebuilds: AtomicU64,
    index_rebuild_ns: AtomicU64,
    index_rebuild_swap_wait_ns: AtomicU64,
    /// `(nanoseconds, WAL entries replayed)` for the recovery this instance did at open.
    last_recovery: StdMutex<Option<(u64, u64)>>,
    last_snapshot_ns: StdMutex<Option<u64>>,

    /// `{wal_dir}/LOCK`, held exclusively for as long as this instance lives. Nothing stopped a
    /// second process -- a second `serve` on another port, `vectordb mcp` beside a running
    /// server, an embedded client starting its own -- from opening the same directory, and two
    /// processes appending to one WAL and one set of row files corrupt both.
    _dir_lock: Option<std::fs::File>,

    /// Recovery only, emptied when it ends. Each table the snapshot restored, with the row
    /// below which its graph was known to be current (`None`: a snapshot from before that was
    /// recorded); and for each table, the row each replayed record was indexed from.
    recovery_floors: StdMutex<HashMap<String, Option<u64>>>,
    replayed_rows: StdMutex<HashMap<String, HashMap<RecordId, u64>>>,

    /// Test-only: `backup_to` stops between its snapshot and its copy, says so on the first
    /// channel and waits on the second. See `pause_next_backup_for_test`.
    #[cfg(feature = "test-failpoints")]
    backup_pause: StdMutex<
        Option<(
            tokio::sync::oneshot::Sender<()>,
            tokio::sync::oneshot::Receiver<()>,
        )>,
    >,

    /// 对象存储上 WAL 的清单，常驻内存，每次变更后回写。
    wal_manifest: Arc<StdMutex<WalManifest>>,
    /// The first WAL sequence object storage does not hold yet: everything below it is in the
    /// manifest's chunks or covered by the snapshot. See `mirror_wal`.
    wal_mirrored_next: AtomicU64,
    /// One mirror upload at a time, so chunks are contiguous and in order.
    wal_mirror_lock: tokio::sync::Mutex<()>,
    /// Held from changing the in-memory WAL manifest until its upload has succeeded or been
    /// undone. Appending a chunk and pruning both write the whole manifest; unserialised, an
    /// older copy could land after a newer one and list chunks already deleted.
    manifest_write: tokio::sync::Mutex<()>,
    /// The version of `wal/MANIFEST` this process last read or wrote, for the next conditional
    /// write of it; `None` when there is none yet or the store has no versions.
    manifest_version: StdMutex<Option<String>>,
    /// Whether the object store honours conditional writes; probed at start
    /// (`probe_conditional_writes`). Without them, fencing is the `wal/PRIMARY` poll.
    conditional_writes: std::sync::atomic::AtomicBool,
    /// Set by `fence`: another primary has this bucket, and nothing more goes to it.
    fenced: std::sync::atomic::AtomicBool,
    /// The epoch this primary claimed (`claim_primary`); 0 without object storage. Part of the
    /// name of every chunk and snapshot it writes.
    primary_epoch: AtomicU64,
    /// A read replica: it follows the WAL another process mirrors to object storage, serves
    /// reads, refuses writes, and never writes the bucket. See `new_replica`.
    replica: bool,
    /// On a replica, the first WAL sequence it has not applied.
    replica_applied_next: AtomicU64,
    /// On a replica that can no longer catch up from the WAL chunks (the primary pruned some
    /// it had not applied), why; it reports not ready until restarted.
    replica_stale: StdMutex<Option<String>>,
    /// With object storage, the token this primary wrote to `wal/PRIMARY` when it started;
    /// see `claim_primary`. `None` on a replica and without object storage.
    primary_token: Option<String>,
    /// When `ensure_still_primary` last read the token back, and found it ours.
    primary_checked: StdMutex<Option<std::time::Instant>>,
    /// How often it reads it: `DXMT_PRIMARY_CHECK_MS`, default 1,000.
    primary_check_every: std::time::Duration,
    /// How long WAL chunks stay in the bucket after a snapshot covers them, in seconds:
    /// `DXMT_WAL_RETAIN_SECS`, default 600. See `retained_prune_point`.
    wal_retain_secs: AtomicU64,
    /// Snapshots uploaded since start, oldest first, with when: the candidates to prune to.
    offloaded_snapshots: StdMutex<std::collections::VecDeque<(std::time::Instant, u64)>>,
    /// The sequence of the last snapshot published to object storage; `u64::MAX` before one.
    last_offloaded: AtomicU64,

    /// 回放期间置位，用于抑制 WAL 写入。回放走的是 `create_table` /
    /// `insert_records` 这些正常入口（否则索引不会被填充），所以必须挡住它们
    /// 把刚读出来的条目又写回日志。
    replaying: Arc<AtomicBool>,

    /// Set after a local WAL write or sync failure. The failed mutation is deliberately not
    /// rolled back: indexes, payload postings and text state may already have been published by
    /// the time the filesystem reports the error. Returning an error while accepting more writes
    /// would make the durability boundary unknowable, so the process becomes read-only until a
    /// restart. The mutation that returned the error is not acknowledged and may or may not be
    /// present after recovery, depending on how far the filesystem write got.
    /// Refuses writes before the machine runs out of memory; see `memory_guard`.
    memory_guard: crate::storage::memory_guard::MemoryGuard,
    /// Refuses writes before the data directory's disk fills; see `disk_guard`. No directory
    /// (in-memory storage), no guard.
    disk_guard: crate::storage::disk_guard::DiskGuard,
    durability_error: Arc<StdMutex<Option<String>>>,

    /// Operational durability telemetry. These fields are intentionally process-local: a
    /// restart resets the counters, while the WAL/snapshot sequence remains the durable source
    /// of truth.
    last_wal_sync_at: Arc<StdMutex<Option<String>>>,
    last_snapshot_at: Arc<StdMutex<Option<String>>>,
    last_backup_at: StdMutex<Option<String>>,
    /// The model a table created with neither a dimension nor an `embedding` uses: set at
    /// startup from `DXMT_EMBEDDING_MODEL`, or found on a local ollama. See `api::quickstart`.
    default_embedding: StdMutex<Option<crate::types::table::EmbeddingConfig>>,
    /// The knowledge-base table's name, read once: see `storage::knowledge_base`.
    kb_table: Option<String>,
    backup_failures: AtomicU64,
    /// `wal_next_sequence` and table names of the snapshot this process last loaded or
    /// installed. Cached so a health probe does not have to read and deserialize the manifest —
    /// and one `.meta` file per evicted table — off disk on every call.
    snapshot_sequence: Arc<StdMutex<Option<u64>>>,
    snapshot_table_names: Arc<StdMutex<Vec<String>>>,
    /// Tables dropped since the last snapshot was installed.
    ///
    /// `delete_table` takes the table out of `tables` and logs `WalEntry::DeleteTable`, but
    /// the snapshot on disk still holds it, and the snapshot writer copies every
    /// *non-resident* table forward on the premise that an evicted table cannot have
    /// changed. A dropped table is non-resident and has changed. Without this set it is
    /// carried into every later snapshot: it keeps appearing in `list_tables`, the next
    /// access rehydrates it, and the drop stops being durable at all — each snapshot
    /// advances `wal_next_sequence` past the `DeleteTable` entry that recorded it, so a
    /// restart brings the table back with its rows.
    ///
    /// Emptied of the names a snapshot has just written without.
    dropped_since_snapshot: Arc<StdMutex<std::collections::HashSet<String>>>,
    snapshot_failures: AtomicU64,

    /// One counter per filtered-search branch; see [`FilterPathCounts`].
    filter_path_exact_scan: AtomicU64,
    filter_path_code_scan: AtomicU64,
    filter_path_in_graph: AtomicU64,
    filter_path_pre_filter: AtomicU64,
    filter_path_post_filter: AtomicU64,
    /// Nanoseconds spent answering, per branch, in the order of [`FilterPathCounts`]: from
    /// the selectivity estimate to the branch's result. A count says which branch a workload
    /// is on; this says what it costs there.
    filter_path_ns: [AtomicU64; 5],
    /// Nanoseconds spent before any branch runs: estimating the filter and materialising the
    /// matching ids. Measured apart because it is plumbing, not search.
    filter_eval_ns: AtomicU64,
    filter_node_map_misses: AtomicU64,
    last_snapshot_error: Arc<StdMutex<Option<String>>>,

    /// Alternative names for tables: `alias -> table`.
    ///
    /// What they are for is a swap. Rebuilding a table with different index parameters means
    /// building a second one and moving traffic to it, and without an alias that move is a
    /// client-side deploy. With one it is a single request, and every reader that arrives
    /// mid-swap sees one table or the other rather than a missing one.
    ///
    /// Kept in their own file (`{wal_dir}/aliases`) rather than in the WAL. They change rarely,
    /// they are tiny, and putting them in the log would mean the snapshot has to carry them too
    /// -- the manifest is bincode, and adding a field to it is the break that
    /// `src/storage/snapshot_v1.rs` exists to document. A missing file means no aliases, which
    /// is what every database written before this has.
    aliases: Arc<parking_lot::RwLock<std::collections::BTreeMap<String, String>>>,
}

/// 智能表数据
/// Per named vector space, the records to rebuild it from.
type NamedVectors = Vec<(String, Vec<(RecordId, Vec<f32>)>)>;

/// Indexes derived from the records: payload values for filtering, sparse postings for MIPS.
///
/// Both are built at insert time and neither is written anywhere, so a restart has to rebuild
/// them from the records. `restore_table` did not, and the result was silent in the way this
/// codebase keeps finding: after any restart **every filter matched nothing**. `records/list`
/// and `search` both answered 200 with an empty result, indistinguishable from a filter that
/// genuinely selects no rows, for `search` and `records/list` alike. Sparse search did the same.
/// Measured on the demo box: a record inserted after the restart was filterable, 2,500 records
/// inserted before it were not.
///
/// The whole existing test suite was green through this, because `tests/api/` inserts and
/// queries in one process. Nothing restarted the server before filtering.
///
/// # Why this is built on demand rather than at restore
///
/// Rebuilding costs one decode of every record -- about 150 ms per 120,000 rows, so roughly
/// fifteen seconds at ten million. Paying that during `restore_table` would turn a restart that
/// currently serves in 0.43 s into one that does not, for every server, including the ones that
/// never filter. Built on first use instead: the first filtered query on each table pays the
/// scan, everything after it is free, and a server that never filters never builds it.
///
/// Writes do not build it. An insert into a table whose indexes are not built leaves them not
/// built, because a later build reads the records the insert just landed. Only a read forces it.
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct DerivedIndexes {
    payload: crate::storage::payload_index::PayloadIndex,
    sparse: SparseIndex,
}

struct IntelligentTableData {
    /// 表元数据
    table: Table,

    /// 记录存储
    ///
    /// `Arc` 是为了快照:它要的是某个时刻的一致视图,于是在读锁里把每条记录拷出来,
    /// 而 10M 条 × 9.5 KB 就是 90 GB 的临时拷贝叠在 90 GB 常驻之上 —— 实测在 125 GB 的机器上
    /// 被内核杀掉。记录从不原地修改、只整体替换,所以引用计数就够,
    /// 一次快照的拷贝从每条 9.5 KB 降到 8 字节。
    /// Where the records live: the map, or a row file on disk (`DXMT_RECORD_STORE=disk`,
    /// phase 3 of docs/architecture/disk-index.md). Same interface either way; `get` decodes from
    /// disk when that is where the record is.
    records: crate::storage::record_store::RecordStore,

    /// Full vectors on disk (`DXMT_VECTOR_STORE=disk`, docs/architecture/disk-index.md §4, phase 2).
    /// When set, the `Record`s in `records` carry an empty `vector` and `vector_rows` says
    /// which row of the file holds each id's; the vector index still gets the vector on
    /// insert (it plans and encodes from it), and everything that reads a full vector
    /// afterwards -- the rerank, `with_vector`, `get_record`, a rebuild -- goes through
    /// [`IntelligentTableData::vector_for`].
    vector_file: Option<Arc<crate::storage::vector_file::VectorFile>>,
    vector_rows: HashMap<RecordId, u64>,

    /// 选择的索引算法
    selected_algorithm: IndexAlgorithm,

    /// 索引实例 (使用 IndexWrapper 提供 interior mutability)
    index: crate::search::IndexWrapper,

    /// Named vector indexes (multi-vector support)
    named_indexes: Option<crate::search::IndexCollection>,

    /// 索引决策信息
    index_decision: IndexDecision,

    /// 性能统计（Arc<Mutex> 支持在读锁持有期间并发更新）
    performance_stats: Arc<Mutex<PerformanceStats>>,

    /// What the last reachability repair found and what every pass has repaired so far.
    /// `stranded_last` is the number of nodes with no in-edge the last pass saw -- the
    /// figure a `GET /tables/{t}/stats` reader wants when "search with a stored vector
    /// does not return it" comes up. See `HnswIndex::repair_reachability`.
    stranded_last: Arc<std::sync::atomic::AtomicUsize>,
    repaired_total: Arc<AtomicU64>,

    /// The two indexes that are a pure function of the records, and are rebuilt rather than
    /// restored. `None` means "not built yet" -- see [`IntelligentTableData::derived`].
    derived: parking_lot::RwLock<Option<DerivedIndexes>>,
    /// While `derived` is being built outside the map lock: every id a write touched, with
    /// the payload it held before the first such write (`None`: it did not exist yet). The
    /// build catches these up before installing; see [`IntelligentIndexedStorage::warm_derived`].
    derived_changes: parking_lot::Mutex<Option<HashMap<RecordId, Option<serde_json::Value>>>>,
    /// One build of `derived` at a time; a second caller waits and finds it built.
    derived_build: Arc<tokio::sync::Mutex<()>>,
    /// Payload ordinals against the graph's node positions, filled lazily by filtered
    /// searches; see `storage::node_map`. Behind its own lock because it is written by the
    /// first search to meet new nodes while the table map is only read.
    node_map: Arc<tokio::sync::RwLock<crate::storage::node_map::NodeMap>>,
    /// Record writes applied to this table (inserts, updates, deletes), so a build that
    /// released the lock between chunks can tell whether what it scanned is still the
    /// table. See [`IntelligentIndexedStorage::warm_derived`].
    writes: Arc<AtomicU64>,

    /// 全文搜索引擎（仅当启用全文搜索时）
    #[cfg(feature = "text-bm25")]
    full_text_engine: Option<Arc<FullTextSearchEngine>>,
}

/// A k-nearest-neighbour graph over a sample of a table.
///
/// `neighbours[i]` and `scores[i]` describe node `i`, whose record is `ids[i]`; the entries are
/// indices into `ids`, not record ids, because a layout works in node numbers and the repetition
/// would be most of the response.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct KnnGraph {
    /// The sampled records, in node order.
    pub ids: Vec<RecordId>,
    /// Per node, its nearest neighbours as indices into `ids`, nearest first.
    pub neighbours: Vec<Vec<u32>>,
    /// Per node, the similarity to each of those neighbours. Higher is nearer, as in search.
    pub scores: Vec<Vec<f32>>,
    /// How many nodes the graph has.
    pub sampled: usize,
    /// How many records the sample was drawn from, after any filter.
    pub total: usize,
    /// Per node, its payload as JSON, when asked for.
    ///
    /// A layout needs labels for the nodes it draws — something to colour by, something to show
    /// on hover — and those cannot come from a second request: the sample is scattered across
    /// the table and there is no route that fetches a list of ids. They are cheap next to what
    /// this endpoint exists to avoid: a payload is tens of bytes where a 768-dimension vector
    /// is ten kilobytes of JSON floats.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payloads: Option<Vec<Option<String>>>,
}

/// `record` with every field but the vector: what a WAL entry carries when the vector is
/// in the row file. A clone-then-clear would copy the vector first.
fn without_vector(record: &Record) -> Record {
    Record {
        id: record.id.clone(),
        vector: Vec::new(),
        sparse_vector: record.sparse_vector.clone(),
        named_vectors: record.named_vectors.clone(),
        payload: record.payload.clone(),
        created_at: record.created_at,
        updated_at: record.updated_at,
        version: record.version,
        status: record.status.clone(),
    }
}

/// 性能统计
impl IntelligentTableData {
    /// Where `id`'s vector sits in the vector file: the record store knows when the records
    /// are on disk (eight bytes a row), the map knows when they are in memory.
    fn vector_row(&self, id: &str) -> Option<u64> {
        self.records
            .vector_row(id)
            .or_else(|| self.vector_rows.get(id).copied())
    }

    /// The full vector of `record`: from the record while it holds one, else from disk.
    fn vector_for(&self, record: &Record) -> Vec<f32> {
        if !record.vector.is_empty() {
            return record.vector.clone();
        }
        self.vector_file
            .as_ref()
            .zip(self.vector_row(&record.id))
            .and_then(|(file, row)| file.read(row))
            .unwrap_or_default()
    }

    /// Every live record's vector, for building an index. From disk this reads the rows in
    /// file order, a few thousand at a time, rather than one row per record.
    /// The derived indexes, built from the records if this is the first time anyone asked.
    ///
    /// The build is one pass over the records, which on a disk-backed table means decoding each
    /// one. That is the cost this defers until a filter or a sparse query actually needs it;
    /// see [`DerivedIndexes`] for why it is not paid at restore.
    fn derived(&self) -> parking_lot::MappedRwLockReadGuard<'_, DerivedIndexes> {
        if self.derived.read().is_none() {
            let mut built = DerivedIndexes::default();
            // Built under the table's policy, so a field it excludes is never indexed at all:
            // the memory and the pass over its values are what excluding it saves.
            if let Some(policy) = &self.table.payload_index {
                built.payload.set_policy(policy.clone());
            }
            built.payload.begin_bulk(self.records.len());
            let build_started = std::time::Instant::now();
            let records = Self::scan_into(&self.records, &mut built, None);
            let scanned = build_started.elapsed();
            built.payload.end_bulk();
            info!(
                "Table '{}': derived indexes built from {} records: scan+insert {:.1}s, end_bulk {:.1}s",
                self.table.name,
                records,
                scanned.as_secs_f64(),
                (build_started.elapsed() - scanned).as_secs_f64()
            );
            let mut slot = self.derived.write();
            // Another thread may have built it while this one was scanning. Theirs is as good
            // as this one -- both read the same records under the same lock -- so the first to
            // arrive wins and the loser's work is dropped rather than overwriting it.
            if slot.is_none() {
                debug!(
                    "Table '{}': built the payload and sparse indexes from {} records",
                    self.table.name, records
                );
                *slot = Some(built);
            }
        }
        parking_lot::RwLockReadGuard::map(self.derived.read(), |slot| {
            slot.as_ref().expect("built just above, and never unset")
        })
    }

    /// The derived indexes for a write to update, or `None` when they have not been built.
    ///
    /// A write to an unbuilt pair does nothing on purpose: whoever builds them later reads the
    /// records, which by then include this write. Forcing a build here would make the first
    /// insert into a restored table pay for a scan nobody asked for.
    /// Index every record of `store` into `built`. On a disk store the rows are decoded on
    /// every core, a chunk at a time, and inserted in row order; `stop_at` bounds the rows
    /// (what [`IntelligentIndexedStorage::warm_derived`] passes so a build that releases the
    /// lock between chunks indexes exactly the rows it saw). Returns the records indexed.
    fn scan_into(
        store: &crate::storage::record_store::RecordStore,
        built: &mut DerivedIndexes,
        stop_at: Option<u32>,
    ) -> usize {
        let mut records = 0usize;
        let mut index = |record: &Record| {
            records += 1;
            built.payload.insert_record(
                &record.id,
                &record.payload,
                crate::storage::payload_index::created_key(&record.created_at),
            );
            if let Some(sparse) = &record.sparse_vector {
                if let Err(e) = built.sparse.insert(record.id.clone(), sparse) {
                    warn!("Failed to index sparse vector for {}: {}", record.id, e);
                }
            }
        };
        match store.row_count() {
            Some(rows) => {
                let rows = stop_at.map_or(rows, |s| s.min(rows));
                let mut created: Vec<i64> = Vec::with_capacity(rows as usize);
                const CHUNK: u32 = 16_384;
                let mut from = 0u32;
                while from < rows {
                    let to = (from + CHUNK).min(rows);
                    for slot in store.decode_range(from, to).unwrap_or_default() {
                        match slot {
                            Some(record) => {
                                created.push(crate::storage::payload_index::created_key(
                                    &record.created_at,
                                ));
                                index(&record);
                            }
                            None => created.push(i64::MIN),
                        }
                    }
                    from = to;
                }
                store.publish_created(created);
            }
            None => store.for_each(|record| index(&record)),
        }
        records
    }

    /// A write that found `derived` unbuilt: if a build is running, note the id and the
    /// payload it held before (the first time only -- that is the payload the build may have
    /// indexed), so the build can catch it up.
    fn note_unindexed_change(&self, id: &str, before: Option<&serde_json::Value>) {
        if let Some(changes) = self.derived_changes.lock().as_mut() {
            changes
                .entry(id.to_string())
                .or_insert_with(|| before.cloned());
        }
    }

    fn derived_mut(&self) -> parking_lot::MappedRwLockWriteGuard<'_, Option<DerivedIndexes>> {
        parking_lot::RwLockWriteGuard::map(self.derived.write(), |slot| slot)
    }

    /// The stored vectors of `ids`, from the row file where there is one and from the record
    /// otherwise; ids that are gone are left out.
    fn vectors_for_ids(&self, ids: &[RecordId]) -> Vec<(RecordId, Vec<f32>)> {
        let mut out = Vec::with_capacity(ids.len());
        let mut by_row: Vec<(u64, &RecordId)> = Vec::new();
        for id in ids {
            match self.vector_row(id) {
                Some(row) => by_row.push((row, id)),
                None => {
                    if let Some(r) = self.records.get(id) {
                        if !r.vector.is_empty() {
                            out.push((id.clone(), r.vector.clone()));
                        }
                    }
                }
            }
        }
        if let (Some(file), false) = (&self.vector_file, by_row.is_empty()) {
            by_row.sort_unstable();
            for chunk in by_row.chunks(4096) {
                let rows: Vec<u64> = chunk.iter().map(|(row, _)| *row).collect();
                for ((_, id), vector) in chunk.iter().zip(file.read_rows(&rows)) {
                    if let Some(vector) = vector {
                        out.push(((*id).clone(), vector));
                    }
                }
            }
        }
        out
    }

    fn vectors_for_index(&self) -> Vec<(RecordId, Vec<f32>)> {
        let Some(file) = &self.vector_file else {
            let mut out = Vec::with_capacity(self.records.len());
            self.records
                .for_each(|r| out.push((r.id.clone(), r.vector.clone())));
            return out;
        };
        let ids = self.records.ids();
        let mut by_row: Vec<(u64, &RecordId)> = ids
            .iter()
            .filter_map(|id| self.vector_row(id).map(|row| (row, id)))
            .collect();
        by_row.sort_unstable();
        let mut out = Vec::with_capacity(ids.len());
        for chunk in by_row.chunks(4096) {
            let rows: Vec<u64> = chunk.iter().map(|(row, _)| *row).collect();
            for ((_, id), vector) in chunk.iter().zip(file.read_rows(&rows)) {
                if let Some(vector) = vector {
                    out.push(((*id).clone(), vector));
                }
            }
        }
        // Records that predate the vector file (no row) still carry their vector.
        for id in &ids {
            if self.vector_row(id).is_none() {
                if let Some(r) = self.records.get(id) {
                    if !r.vector.is_empty() {
                        out.push((id.clone(), r.vector.clone()));
                    }
                }
            }
        }
        out
    }
}

#[derive(Debug, Default)]
pub struct PerformanceStats {
    /// 总查询次数
    total_queries: u64,

    /// 平均查询延迟
    avg_query_latency_ms: f64,

    /// P95查询延迟
    p95_query_latency_ms: f64,

    /// 召回率
    avg_recall_rate: f64,

    /// 插入吞吐量
    insert_throughput: f64,
}

/// What a text index was built for: its fields and analyzers. Recorded in each generation's
/// directory, so an index built for another config -- a rebuild cut short, a change made
/// while the server was down -- is noticed at the next open and rebuilt.
fn text_fingerprint(table: &Table) -> String {
    serde_json::json!({
        "enabled": table.enable_full_text_search,
        "fields": table.text_fields,
    })
    .to_string()
}

/// The file in a text-index generation that records its `text_fingerprint`.
#[cfg(feature = "text-bm25")]
const TEXT_CONFIG_FILE: &str = "dxmt-text-config.json";

/// Text indexes live under `fulltext/{table_id}/` as generations `g1`, `g2`, ... with
/// `CURRENT` naming the one in use; a rebuild fills the next and then points `CURRENT` at it.
/// An index from before generations has its files directly in the root and no `CURRENT`.
#[cfg(feature = "text-bm25")]
fn current_text_generation(root: &std::path::Path) -> Option<PathBuf> {
    let name = std::fs::read_to_string(root.join("CURRENT")).ok()?;
    let path = root.join(name.trim());
    path.is_dir().then_some(path)
}

/// The generation to open for `table`: the current one if it was built for this config, an
/// index from before generations as it is, and otherwise a new empty generation made current
/// -- which `restore_table` fills from the records, as it does any empty text index.
#[cfg(feature = "text-bm25")]
fn text_generation_to_open(root: &std::path::Path, table: &Table) -> crate::Result<PathBuf> {
    let fingerprint = text_fingerprint(table);
    if let Some(current) = current_text_generation(root) {
        if std::fs::read_to_string(current.join(TEXT_CONFIG_FILE)).ok() == Some(fingerprint) {
            return Ok(current);
        }
    } else if root.join("meta.json").exists() {
        return Ok(root.to_path_buf());
    } else if !root.exists() {
        // A new table: its first generation.
        let generation = new_text_generation(root, table)?;
        make_text_generation_current(root, &generation)?;
        return Ok(generation);
    }
    warn!(
        "text index for table '{}' was built for another configuration; rebuilding it from the \
         records",
        table.name
    );
    let generation = new_text_generation(root, table)?;
    make_text_generation_current(root, &generation)?;
    Ok(generation)
}

/// A new, empty generation directory under `root`, recording `table`'s config.
#[cfg(feature = "text-bm25")]
fn new_text_generation(root: &std::path::Path, table: &Table) -> crate::Result<PathBuf> {
    let next = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            e.file_name()
                .to_str()?
                .strip_prefix('g')?
                .parse::<u64>()
                .ok()
        })
        .max()
        .map_or(1, |n| n + 1);
    let path = root.join(format!("g{next}"));
    std::fs::create_dir_all(&path)?;
    std::fs::write(path.join(TEXT_CONFIG_FILE), text_fingerprint(table))?;
    Ok(path)
}

/// Point `CURRENT` at `generation` (written whole, then renamed into place), and remove the
/// other generations and any pre-generation index files in the root. Removal is best effort:
/// on Windows a directory another handle still has open stays until the next time.
#[cfg(feature = "text-bm25")]
fn make_text_generation_current(
    root: &std::path::Path,
    generation: &std::path::Path,
) -> crate::Result<()> {
    let name = generation
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| crate::Error::Internal("text generation has no name".to_string()))?;
    let tmp = root.join("CURRENT.tmp");
    std::fs::write(&tmp, name)?;
    std::fs::rename(&tmp, root.join("CURRENT"))?;
    for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let path = entry.path();
        let file = entry.file_name();
        let file = file.to_string_lossy();
        if file == "CURRENT" || file == name {
            continue;
        }
        if path.is_dir() {
            if file.starts_with('g') {
                let _ = std::fs::remove_dir_all(&path);
            }
        } else {
            let _ = std::fs::remove_file(&path);
        }
    }
    Ok(())
}

/// The indexable document for a record, or `None` if this table indexes no text from it.
///
/// Shared by the insert path and by the restore backfill so the two cannot drift: a document
/// rebuilt from the rows has to be the same document the insert would have written, or a
/// recovered index answers differently from the one it replaced.
#[cfg(feature = "text-bm25")]
fn text_document_for(table: &Table, record: &Record) -> Option<FullTextDocument> {
    let text_content = extract_text_from_record(table, record)?;
    Some(FullTextDocument {
        id: record.id.clone(),
        title: text_content.get("title").cloned(),
        content: text_content.get("content").cloned().unwrap_or_default(),
        description: text_content.get("description").cloned(),
        tags: text_content
            .get("tags")
            .map(|s| vec![s.clone()])
            .unwrap_or_default(),
        payload: Some(record.payload.clone()),
        timestamp: chrono::Utc::now().timestamp(),
    })
}

/// 从记录中提取文本内容（辅助函数）
#[cfg(feature = "text-bm25")]
fn extract_text_from_record(table: &Table, record: &Record) -> Option<HashMap<String, String>> {
    if !table.enable_full_text_search {
        return None;
    }

    let mut text_content = HashMap::new();

    // Always include the record ID
    text_content.insert("id".to_string(), record.id.to_string());

    // Extract text from configured fields
    if let Some(text_fields) = &table.text_fields {
        for field_config in text_fields {
            let text = match &field_config.source {
                TextSource::MetadataField(field_name) => {
                    // Extract from metadata
                    if let serde_json::Value::Object(map) = &record.payload {
                        map.get(field_name).and_then(|v| {
                            match v {
                                serde_json::Value::String(s) => Some(s.clone()),
                                serde_json::Value::Number(n) => Some(n.to_string()),
                                serde_json::Value::Bool(b) => Some(b.to_string()),
                                serde_json::Value::Array(arr) => {
                                    // Join array elements with spaces
                                    Some(
                                        arr.iter()
                                            .filter_map(|v| v.as_str())
                                            .collect::<Vec<_>>()
                                            .join(" "),
                                    )
                                }
                                _ => None,
                            }
                        })
                    } else {
                        None
                    }
                }
                TextSource::Dedicated(field_name) => {
                    // For now, treat dedicated fields the same as metadata fields
                    if let serde_json::Value::Object(map) = &record.payload {
                        map.get(field_name)
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                }
            };

            if let Some(text_value) = text {
                text_content.insert(field_config.name.clone(), text_value);
            }
        }
    } else {
        // No text_fields configured - extract all string fields from payload as default
        if let serde_json::Value::Object(map) = &record.payload {
            for (key, value) in map {
                match value {
                    serde_json::Value::String(s) => {
                        text_content.insert(key.clone(), s.clone());
                    }
                    serde_json::Value::Number(n) => {
                        text_content.insert(key.clone(), n.to_string());
                    }
                    serde_json::Value::Bool(b) => {
                        text_content.insert(key.clone(), b.to_string());
                    }
                    serde_json::Value::Array(arr) => {
                        // Join array elements with spaces
                        let text = arr
                            .iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(" ");
                        if !text.is_empty() {
                            text_content.insert(key.clone(), text);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if text_content.len() > 1 {
        // More than just the ID
        Some(text_content)
    } else {
        None
    }
}

/// How many queued requests one pass may merge, and how many records that merge may cover.
///
/// Both caps bound the added latency of the request at the head of the queue, not throughput.
/// Coalescing only ever consumes requests that are *already* waiting: under a single writer
/// `try_recv` finds nothing, a group is one request, and the behaviour is what it was before.
/// Slots below which a graph is not worth rebuilding for its tombstones, whatever the ratio.
/// A rebuild is O(n) and a small table's garbage is measured in megabytes.
///
/// `DXMT_INDEX_COMPACTION_MIN_SLOTS` overrides it, for tests: reaching a hundred thousand slots
/// honestly means a hundred thousand insertions, which says nothing about the behaviour under
/// test that a thousand does not.
fn min_compaction_slots() -> usize {
    const DEFAULT: usize = 100_000;
    crate::config::env::var("DXMT_INDEX_COMPACTION_MIN_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT)
}

/// Whether a graph carries enough tombstones to be worth an O(n) rebuild.
///
/// Used twice, and it has to be the same test in both places: the sweep asks it to decide
/// whether to rebuild, and `rebuild_index` asks it to decide whether the graph it already has
/// counts as equivalent to the one a rebuild would produce. They disagreed for exactly one
/// commit, and the result was a sweep that called for a rebuild and a rebuild that declined it.
fn worth_compacting(live: usize, slots: usize) -> bool {
    slots >= min_compaction_slots() && live.saturating_mul(2) <= slots
}

/// Sidecar bytes worth rewriting the `(id, row)` list for. It is rewritten whole, so the
/// floor keeps a sweep from copying a small file every few minutes for nothing.
const MIN_SIDECAR_BYTES: u64 = 16 * 1024 * 1024;

/// Dead vector-file bytes worth a `fallocate` sweep. Below this the syscall costs more than
/// the space is worth, and a table that has just crossed a few thousand updates does not need
/// its file walked every ten minutes.
///
/// `DXMT_ROW_RECLAIM_MIN_BYTES` overrides it. That exists for the tests: reaching 64 MB of dead
/// rows honestly takes tens of thousands of insertions, which is a five-minute test in a debug
/// build to assert something that has nothing to do with the size of the number.
fn min_reclaim_bytes() -> u64 {
    const DEFAULT: u64 = 64 * 1024 * 1024;
    crate::config::env::var("DXMT_ROW_RECLAIM_MIN_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT)
}

const MAX_COALESCED_REQUESTS: usize = 32;
const MAX_COALESCED_RECORDS: usize = 50_000;

/// 后台插入处理器 - 单线程处理所有插入，消除锁竞争
/// Where an insert's wall time goes, cumulative, one line per 250 groups.
///
/// `HnswIndex` already reports its three build phases this way, and on a 1M x 960 load those
/// three summed to a third of the wall time: the other two thirds were somewhere in the path
/// around them -- row file, record store, WAL, the queue -- and nothing said where. These are
/// the stages of that path. Wall time, not CPU, for the same reason as the graph's: the
/// question is what a batch waits on. `prepare` and `wal` are measured on the caller's side
/// of the channel, per request; the rest in the processor, per group.
mod insert_stages {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub const QUEUE: usize = 0;
    pub const ROW_FILE: usize = 1;
    pub const MAP: usize = 2;
    pub const GRAPH: usize = 3;
    #[cfg_attr(not(feature = "text-bm25"), allow(dead_code))]
    pub const TEXT: usize = 4;
    pub const WAL: usize = 5;
    pub const PREPARE: usize = 6;
    const NAMES: [&str; 7] = [
        "queue", "row_file", "map", "graph", "text", "wal", "prepare",
    ];
    static NS: [AtomicU64; 7] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];
    static GROUPS: AtomicU64 = AtomicU64::new(0);
    static ROWS: AtomicU64 = AtomicU64::new(0);
    const LOG_EVERY: u64 = 250;

    pub fn add(stage: usize, started: std::time::Instant) {
        add_duration(stage, started.elapsed());
    }

    pub fn add_duration(stage: usize, d: std::time::Duration) {
        NS[stage].fetch_add(d.as_nanos() as u64, Relaxed);
    }

    /// Called once per group, after every stage the processor owns.
    pub fn group_done(rows: usize) {
        ROWS.fetch_add(rows as u64, Relaxed);
        let n = GROUPS.fetch_add(1, Relaxed) + 1;
        if n.is_multiple_of(LOG_EVERY) {
            let parts: Vec<String> = NAMES
                .iter()
                .enumerate()
                .map(|(i, name)| format!("{} {:.0} ms", name, NS[i].load(Relaxed) as f64 / 1e6))
                .collect();
            tracing::info!(
                "insert stages after {} groups ({} rows): {}",
                n,
                ROWS.load(Relaxed),
                parts.join(", ")
            );
        }
    }
}

async fn insertion_processor(
    mut rx: mpsc::UnboundedReceiver<InsertRequest>,
    tables: Arc<RwLock<HashMap<String, IntelligentTableData>>>,
) {
    info!("Insertion processor started");

    // Requests taken off the channel but not yet placed in a group. A group is a run of
    // requests naming one table and sharing no record id; the first request that breaks either
    // condition stays here and heads the next group. Per-table arrival order is therefore
    // preserved, and nothing starves — every pass consumes at least the request at the front.
    let mut pending: std::collections::VecDeque<InsertRequest> = std::collections::VecDeque::new();

    loop {
        if pending.is_empty() {
            match rx.recv().await {
                Some(req) => pending.push_back(req),
                None => break,
            }
        }
        // Whatever else has piled up meanwhile. `try_recv` does not wait, so an idle queue
        // costs one failed poll per pass.
        while pending.len() < MAX_COALESCED_REQUESTS {
            match rx.try_recv() {
                Ok(req) => pending.push_back(req),
                Err(_) => break,
            }
        }

        let group = take_group(&mut pending);
        process_insert_group(group, &tables).await;
    }

    info!("Insertion processor stopped");
}

/// Take off the front of `pending` the longest run one pass can handle as a single batch.
///
/// Two requests may share a group only if they name the same table and touch no id in common.
/// The table condition is what makes one merged index insert possible at all. The id condition
/// is what keeps upsert honest: were the same id written twice inside one group, the merged
/// batch would carry it twice, and the index would be asked to drop a vector it had not yet
/// been given. Splitting on the overlap costs one small group and needs no de-duplication.
fn take_group(pending: &mut std::collections::VecDeque<InsertRequest>) -> Vec<InsertRequest> {
    let first = pending
        .pop_front()
        .expect("insertion_processor only calls this with a non-empty queue");
    let table = first.table_name.clone();
    let mut ids: std::collections::HashSet<RecordId> =
        first.records.iter().map(|r| r.id.clone()).collect();
    let mut staged = first.records.len();
    let mut group = vec![first];

    while group.len() < MAX_COALESCED_REQUESTS && staged < MAX_COALESCED_RECORDS {
        let Some(next) = pending.front() else { break };
        if next.table_name != table || next.records.iter().any(|r| ids.contains(&r.id)) {
            break;
        }
        let next = pending
            .pop_front()
            .expect("front() just returned a request");
        ids.extend(next.records.iter().map(|r| r.id.clone()));
        staged += next.records.len();
        group.push(next);
    }
    group
}

/// Apply one group of requests and answer each of them.
///
/// Two phases, and the split is the point. Phase one holds the tables write lock just long
/// enough to update the record map and the payload/sparse indexes — cheap, bounded work. Phase
/// two does the expensive part, inserting into the vector index, with the lock released.
///
/// Holding the lock across the whole batch is what this replaces: a 5,000-record request
/// blocked every reader, `GET /tables` included, until the last vector was in the graph.
/// Records still become searchable only once the index insert lands, because search goes
/// through the index; what a group changes is that requests merged into it land together
/// rather than one after another.
async fn process_insert_group(
    group: Vec<InsertRequest>,
    tables: &Arc<RwLock<HashMap<String, IntelligentTableData>>>,
) {
    let start_time = std::time::Instant::now();
    let table_name = group[0].table_name.clone();

    // Each request keeps its own responder, so one bad request can be answered and dropped
    // without failing the requests it happened to be merged with.
    let mut senders: Vec<Option<oneshot::Sender<crate::Result<Inserted>>>> =
        Vec::with_capacity(group.len());
    let mut batches: Vec<Vec<CreateRecordRequest>> = Vec::with_capacity(group.len());
    let mut known_rows: Vec<Vec<Option<u64>>> = Vec::with_capacity(group.len());
    for req in group {
        // The oldest request's wait is the group's wait; the younger ones overlapped it.
        if batches.is_empty() {
            insert_stages::add_duration(insert_stages::QUEUE, req.enqueued_at.elapsed());
        }
        senders.push(Some(req.response_tx));
        known_rows.push(req.rows_in_file);
        batches.push(req.records);
    }
    let group_rows: usize = batches.iter().map(|b| b.len()).sum();

    let result: crate::Result<Vec<Owned>> = async {
        // Vectors go to the row file before the records become visible, outside the map
        // lock: the append fsyncs, and readers should not wait on a disk write. Rows for a
        // batch that is then rejected below are dead space, reclaimed on rebuild.
        let (vector_file, expected_dim) = {
            let tables = tables.read().await;
            let t = tables
                .get(&table_name)
                .ok_or_else(|| crate::Error::TableNotFound(table_name.clone()))?;
            (t.vector_file.clone(), t.table.dimension)
        };
        let mut appended_rows: Vec<Vec<Option<u64>>> = Vec::new();
        // Held until the graph has this group (the end of this function): a snapshot taken in
        // between must not claim these rows are in the graph it serializes.
        let _unindexed = vector_file.as_ref().map(|file| file.begin_unindexed());
        let stage = std::time::Instant::now();
        if let Some(file) = &vector_file {
            // A record whose vector is already in the file (replay of an entry that points
            // at its row) keeps that row; every other one is appended. Replay used to append
            // every replayed vector again, so the file grew by the whole replayed tail on
            // each restart until the sweep punched the duplicates.
            appended_rows = batches.iter().map(|b| vec![None; b.len()]).collect();
            let mut items: Vec<(&str, &[f32])> = Vec::new();
            let mut slots: Vec<(usize, usize)> = Vec::new();
            for (slot, batch) in batches.iter().enumerate() {
                for (i, r) in batch.iter().enumerate() {
                    if r.vector.len() != expected_dim {
                        continue;
                    }
                    match known_rows
                        .get(slot)
                        .and_then(|k| k.get(i).copied().flatten())
                    {
                        Some(row) if row < file.rows() => appended_rows[slot][i] = Some(row),
                        _ => {
                            items.push((r.id.as_str(), r.vector.as_slice()));
                            slots.push((slot, i));
                        }
                    }
                }
            }
            if !items.is_empty() {
                let rows = file.append(&items)?;
                for ((slot, i), row) in slots.into_iter().zip(rows) {
                    appended_rows[slot][i] = Some(row);
                }
            }
        }
        insert_stages::add(insert_stages::ROW_FILE, stage);

        let stage = std::time::Instant::now();
        let (index, named_indexes, table_meta, to_remove, to_insert, owners) = {
            let mut tables = TimedWrite {
                guard: tables.write().await,
                who: "insertion_processor",
                since: std::time::Instant::now(),
            };

            let table_data = tables
                .get_mut(&table_name)
                .ok_or_else(|| crate::Error::TableNotFound(table_name.clone()))?;

            let expected_dim = table_data.table.dimension;
            let mut to_remove: Vec<RecordId> = Vec::new();
            let mut to_insert: Vec<PendingIndexInsert> = Vec::new();
            // Which slice of the work belongs to which responder.
            let mut owners: Vec<Owned> = Vec::new();
            let mut new_records_count = 0;
            let mut staged = 0usize;
            let mut to_store: Vec<(Arc<Record>, Option<u64>)> = Vec::new();

            for (slot, batch) in batches.into_iter().enumerate() {
                // Dimensions are checked across the whole request before any of it is applied.
                // Checking as it went, as the single-request version did, left the records
                // ahead of the mismatch in the table map but out of the index; a merged group
                // would additionally have to decide what that meant for the requests alongside.
                // One pass first answers both: a bad request changes nothing and fails alone.
                if let Some(bad) = batch
                    .iter()
                    .find(|r| !r.vector.is_empty() && r.vector.len() != expected_dim)
                {
                    if let Some(tx) = senders[slot].take() {
                        let _ = tx.send(Err(crate::Error::DimensionMismatch {
                            expected: expected_dim,
                            actual: bad.vector.len(),
                        }));
                    }
                    continue;
                }

                // Non-finite values are refused here and not at the HTTP layer, because DXQL,
                // MCP reaches this processor without passing that validator.
                // `INSERT ... [1e999, 0]` parses to an infinity, and an infinite coordinate
                // makes a cosine distance NaN, which panicked the comparator HNSW sorts with:
                // the next search on the table closed the connection without answering, and the
                // record was in the WAL, so a restart brought the poisoned table back.
                if let Some((bad, at)) = batch
                    .iter()
                    .find_map(|r| r.vector.iter().position(|v| !v.is_finite()).map(|i| (r, i)))
                {
                    if let Some(tx) = senders[slot].take() {
                        let _ = tx.send(Err(crate::Error::InvalidVector(format!(
                            "record '{}': {} at index {}",
                            bad.id, bad.vector[at], at
                        ))));
                    }
                    continue;
                }

                let mut records = Vec::with_capacity(batch.len());
                let mut rows = Vec::with_capacity(batch.len());
                for (i, request) in batch.into_iter().enumerate() {
                    let record_id = request.id.clone();
                    let row = appended_rows.get(slot).and_then(|b| b[i]);
                    rows.push(row);

                    // UPSERT: an existing id keeps its created_at and bumps its version, and its
                    // old vector has to leave the index before the new one arrives.
                    let (is_update, existing_created_at, existing_version, old_payload) =
                        if let Some(existing) = table_data.records.get(&record_id) {
                            (
                                true,
                                existing.created_at,
                                existing.version + 1,
                                Some(existing.payload.clone()),
                            )
                        } else {
                            (false, chrono::Utc::now(), 1, None)
                        };

                    if is_update {
                        to_remove.push(record_id.clone());
                    } else {
                        new_records_count += 1;
                    }

                    let record = Record {
                        id: record_id.clone(),
                        vector: request.vector.clone(),
                        sparse_vector: request.sparse_vector.clone(),
                        named_vectors: request.named_vectors.clone(),
                        payload: request.payload.unwrap_or_else(|| serde_json::json!({})),
                        created_at: existing_created_at,
                        updated_at: chrono::Utc::now(),
                        version: existing_version,
                        status: crate::types::RecordStatus::Active,
                    };

                    // Upsert is replacement semantics. Leaving the previous sparse postings or
                    // payload values indexed makes filters and sparse search return a record for
                    // values it no longer has.
                    if let Some(derived) = table_data.derived_mut().as_mut() {
                        if is_update {
                            let _ = derived.sparse.delete(&record_id);
                        }
                        if let Some(sv) = &record.sparse_vector {
                            if let Err(e) = derived.sparse.insert(record_id.clone(), sv) {
                                warn!("Failed to index sparse vector for {}: {}", record_id, e);
                            }
                        }

                        if let Some(old_payload) = old_payload.as_ref() {
                            derived.payload.update_record(
                                &record_id,
                                old_payload,
                                &record.payload,
                                crate::storage::payload_index::created_key(&record.created_at),
                            );
                        } else {
                            derived.payload.insert_record(
                                &record_id,
                                &record.payload,
                                crate::storage::payload_index::created_key(&record.created_at),
                            );
                        }
                    } else {
                        table_data.note_unindexed_change(&record_id, old_payload.as_ref());
                    }
                    // The caller gets the record as sent; the map keeps it without the
                    // vector when that lives in the row file.
                    let stored = match row {
                        Some(row) => {
                            if !table_data.records.is_on_disk() {
                                table_data.vector_rows.insert(record_id.clone(), row);
                            }
                            let mut stored = record.clone();
                            stored.vector = Vec::new();
                            stored
                        }
                        None => record.clone(),
                    };
                    to_store.push((Arc::new(stored), row));

                    to_insert.push((record_id, request.vector, request.named_vectors));
                    records.push(record);
                }

                staged += records.len();
                owners.push((slot, records, rows));
            }

            // One append (and, on disk, one fsync) for the whole group.
            table_data.records.insert_batch(to_store)?;
            table_data.writes.fetch_add(1, Ordering::Relaxed);
            table_data.table.record_count += new_records_count;
            table_data.table.updated_at = chrono::Utc::now();

            let insert_time = start_time.elapsed();
            let throughput = staged as f64 / insert_time.as_secs_f64();
            if let Ok(mut stats) = table_data.performance_stats.lock() {
                stats.insert_throughput = (stats.insert_throughput * 0.9) + (throughput * 0.1);
            }

            (
                table_data.index.clone(),
                table_data.named_indexes.clone(),
                table_data.table.clone(),
                to_remove,
                to_insert,
                owners,
            )
        };

        // ── lock released ──

        for id in &to_remove {
            if let Err(e) = index.remove(id).await {
                warn!("Failed to remove old vector for {}: {}", id, e);
            }
            if let Some(collection) = &named_indexes {
                for name in collection.list_index_names().await {
                    if let Some(named_index) = collection.get_index(&name).await {
                        if let Err(e) = named_index.remove(id).await {
                            warn!(
                                "Failed to remove old named vector '{}' for {}: {}",
                                name, id, e
                            );
                        }
                    }
                }
            }
        }

        // One call for the whole group, not one per record and not one per request. A graph
        // index spends almost all of an insert searching the graph, which only reads it, so
        // handed a batch it can do that half on every core; handed one record at a time it
        // cannot, and an eight-core machine built the index on one of them. Merging the queued
        // requests is what makes the batch big enough for that to pay. The batch also takes the
        // index's write lock once rather than once per record.
        insert_stages::add(insert_stages::MAP, stage);
        let named_work: Vec<(RecordId, Option<HashMap<String, DenseVector>>)> = to_insert
            .iter()
            .map(|(id, _, named)| (id.clone(), named.clone()))
            .collect();
        // A record with no dense vector has nothing to put in the dense index. The check above
        // lets an empty vector through on purpose and the API refuses one at the door, so the
        // only way one arrives is from inside — replay of an entry that lost its vector was
        // one. Handed to the graph it used to take a node without data and corrupt every index
        // after it; the graph refuses it now, and refusing it there would fail the whole batch,
        // so it is left out here instead.
        let stage = std::time::Instant::now();
        index
            .insert_batch(
                to_insert
                    .into_iter()
                    .filter(|(_, vector, _)| !vector.is_empty())
                    .map(|(id, vector, _)| (id, vector))
                    .collect(),
            )
            .await?;
        insert_stages::add(insert_stages::GRAPH, stage);

        // Named vector spaces are a separate index per space; nothing measured points at
        // them, so they keep the straightforward loop.
        if let Some(ref collection) = named_indexes {
            for (id, named) in named_work {
                let Some(named_vecs) = named else { continue };
                for (vec_name, vec_data) in named_vecs {
                    if let Some(named_idx) = collection.get_index(&vec_name).await {
                        if let Err(e) = named_idx.insert(id.clone(), vec_data).await {
                            warn!(
                                "Failed to insert into named index '{}' for {}: {}",
                                vec_name, id, e
                            );
                        }
                    }
                }
            }
        }

        #[cfg(feature = "text-bm25")]
        {
            let stage = std::time::Instant::now();
            let engine = {
                let tables = tables.read().await;
                tables
                    .get(&table_name)
                    .and_then(|d| d.full_text_engine.clone())
            };
            if let Some(engine) = engine {
                for id in &to_remove {
                    if let Err(e) = engine.delete_document(id).await {
                        warn!("Failed to remove old full-text document for {}: {}", id, e);
                    }
                }
                for record in owners.iter().flat_map(|(_, records, _)| records) {
                    if let Some(doc) = text_document_for(&table_meta, record) {
                        // Text indexing must not fail an insert that already landed.
                        if let Err(e) = engine.add_document(doc).await {
                            warn!("Failed to index text for record {}: {}", record.id, e);
                        }
                    }
                }
                if let Err(e) = engine.commit().await {
                    warn!("Failed to commit full-text index: {}", e);
                }
            }
            insert_stages::add(insert_stages::TEXT, stage);
        }
        #[cfg(not(feature = "text-bm25"))]
        let _ = &table_meta;

        Ok(owners)
    }
    .await;
    insert_stages::group_done(group_rows);

    // 发送响应
    match result {
        Ok(owners) => {
            for (slot, records, rows) in owners {
                if let Some(tx) = senders[slot].take() {
                    if tx.send(Ok((records, rows))).is_err() {
                        error!("Failed to send insert response: insertion result was dropped");
                    }
                }
            }
        }
        Err(e) => {
            // Whatever went wrong here — the table disappeared, or the index refused the merged
            // batch — applies to every request still waiting. `crate::Error` is not `Clone`, so
            // the first of them carries the original and the rest carry its message; an insert
            // that failed for an internal reason is what `Internal` is for.
            let detail = e.to_string();
            let mut original = Some(e);
            for slot in senders.iter_mut() {
                let Some(tx) = slot.take() else { continue };
                let err = original
                    .take()
                    .unwrap_or_else(|| crate::Error::Internal(detail.clone()));
                let _ = tx.send(Err(err));
            }
        }
    }

    // Every responder should be spoken for by now; one left holding the line would block its
    // caller until the channel dropped, and is a bug worth seeing rather than a quiet hang.
    for slot in senders.iter_mut() {
        if slot.take().is_some() {
            error!("An insert request in a coalesced group went unanswered");
        }
    }
}

/// Map the algorithm the selector chose onto the type the table advertises.
fn index_type_of(algorithm: &IndexAlgorithm) -> crate::types::table::IndexType {
    use crate::types::table::IndexType;
    match algorithm {
        IndexAlgorithm::BruteForce { .. } => IndexType::BruteForce,
        IndexAlgorithm::HNSW { .. } => IndexType::Hnsw,
        IndexAlgorithm::HnswPQ { .. } => IndexType::HnswPq,
        IndexAlgorithm::EnhancedDiskANN { .. } => IndexType::EnhancedDiskAnn,
        IndexAlgorithm::PartitionedIvf { .. } => IndexType::PartitionedIvf,
        IndexAlgorithm::QuantizedIVF { .. } => IndexType::IvfPq,
        // Unreachable in practice: create_index_instance rejects these before a table exists.
        IndexAlgorithm::DiskANN { .. } => IndexType::EnhancedDiskAnn,
        IndexAlgorithm::ScaNN { .. } => IndexType::BruteForce,
    }
}

/// A reranked search, settled before the fetch: the provider, the query text, the payload
/// field the candidates' text is in, how many to fetch, and which page of the reranked order
/// the caller gets.
struct RerankPlan {
    config: crate::types::table::RerankConfig,
    query: String,
    field: String,
    pool: usize,
    page: usize,
    offset: usize,
}

impl IntelligentIndexedStorage {
    /// 创建新的智能索引存储实例（**不持久化**）。
    ///
    /// 保留这个签名是为了不动 40 多个已有调用点——单测、bench、example 都不需要 WAL，
    /// 让它们各自往磁盘写日志还会互相回放污染。要持久化用 [`Self::new_with_wal`]。
    pub async fn new(config: &StorageConfig) -> crate::Result<Self> {
        Self::new_with_wal(config, None).await
    }

    /// 带 WAL 的实例：`wal_dir` 为 `Some` 时开启预写日志，并在返回前把已有日志回放完。
    ///
    /// 回放走 `create_table` / `insert_records` 这些正常入口，而不是直接往 `tables` 里塞——
    /// 直接塞的话记录会在 map 里但不在索引里，搜索返回空，且这种错法测不出来。
    pub async fn new_with_wal(
        config: &StorageConfig,
        wal_dir: Option<PathBuf>,
    ) -> crate::Result<Self> {
        Self::open(config, wal_dir, false).await
    }

    /// A read replica of the store whose WAL and snapshots are mirrored to the object storage
    /// `config` names, with `local_dir` as its own working copy.
    ///
    /// It recovers the way a fresh machine does -- the bucket's snapshot, then its WAL chunks --
    /// and `follow_object_storage` then applies new chunks as the primary uploads them. It
    /// serves every read; every write is refused with `READ_ONLY_REPLICA`, and nothing it does
    /// writes the bucket (`TieredStorage::set_read_only`), because two writers would overwrite
    /// each other's WAL manifest. Aliases are not followed: they live in a local file, not the
    /// WAL.
    pub async fn new_replica(config: &StorageConfig, local_dir: PathBuf) -> crate::Result<Self> {
        if config.object_storage.is_none() {
            return Err(crate::Error::InvalidConfig(
                "a replica follows object storage, and none is configured \
                 (DXMT_OBJECT_STORAGE_PROVIDER)"
                    .to_string(),
            ));
        }
        Self::open(config, Some(local_dir), true).await
    }

    async fn open(
        config: &StorageConfig,
        wal_dir: Option<PathBuf>,
        replica: bool,
    ) -> crate::Result<Self> {
        let mut storage = Self::build(config).await?;
        if replica {
            storage.replica = true;
            storage.storage.set_uncached_reads(true);
            storage.storage.set_read_only(true);
        }

        if let Some(dir) = wal_dir {
            let recovery_started = std::time::Instant::now();
            // Recovery reads the bucket itself, never the memory or SSD tier. The SSD cache
            // outlives the process and is not inside the data directory, so after another
            // primary has written the bucket it holds the snapshot, MANIFEST and chunks of an
            // older epoch under the same keys: a machine failing back onto an empty directory
            // recovered the state it had before the failover, found by
            // `scripts/replica_soak.py`. A primary's own writes go through the tiers, so once it
            // is running they are its to cache again.
            storage.storage.set_uncached_reads(true);
            // Before the log is opened: opening it can already write (a torn tail is cut).
            storage._dir_lock = Some(Self::lock_dir(&dir)?);
            Self::prepare_local_dir(&dir, replica, storage.object_storage_enabled)?;
            let manager = WalManager::new(&dir)?;
            storage.wal = Some(Arc::new(StdMutex::new(manager)));
            storage.wal_dir = Some(dir.clone());
            storage.disk_guard = crate::storage::disk_guard::DiskGuard::from_env(Some(&dir));

            // Before anything reads chunks: the manifest is how they are found.
            if storage.object_storage_enabled {
                storage.load_wal_manifest().await?;
                if !replica {
                    storage.claim_primary().await?;
                }
            }

            // The aliases are names, not data, and are read before either: a request that
            // arrives while the log is still replaying should resolve to the right table, not
            // to nothing.
            storage.load_aliases();

            // Snapshot first, then only the log written after it. Replay has to run with the
            // snapshot already in place: the tail may delete or overwrite records the snapshot
            // restored, and applying it to an empty store would lose those edits.
            // Restoring the snapshot's tables goes through the write paths too; on a replica
            // that is replicated state being applied, not a client's write.
            let from_seq = APPLYING_REPLICATED
                .scope((), storage.load_snapshot(&dir))
                .await?;
            if storage.object_storage_enabled {
                // Object storage holds the snapshot (through `from_seq`) and the manifest's
                // chunks; mirroring resumes after whichever reaches further.
                let chunked_next = storage
                    .manifest()?
                    .chunks
                    .iter()
                    .map(|c| c.last_seq + 1)
                    .max()
                    .unwrap_or(0);
                storage
                    .wal_mirrored_next
                    .store(chunked_next.max(from_seq), Ordering::SeqCst);
            }

            let mut pending = {
                let mut guard = storage
                    .wal
                    .as_ref()
                    .unwrap()
                    .lock()
                    .map_err(|_| crate::Error::Internal("WAL lock poisoned".to_string()))?;
                // The snapshot compacted the log, possibly to nothing, and an empty log
                // opens with its counter at 0. Writes from here on must be numbered after
                // the snapshot's replay point, or the next restart replays past them: a
                // record inserted in the session after a snapshot, and gone at the restart
                // after that, measured before this line existed.
                guard.advance_sequence_to(from_seq);
                guard.replay_from(from_seq)?
            };

            // A fresh machine has an empty local log, so everything written since the last
            // snapshot lives only in the object store. Merge both sources and let the sequence
            // number decide: chunks a machine already has locally are duplicates, and dropping
            // them here is cheaper than replaying an insert twice.
            if storage.object_storage_enabled {
                let remote = storage.fetch_wal_chunks(from_seq).await?;
                if !remote.is_empty() {
                    let local: std::collections::HashSet<u64> =
                        pending.iter().map(|(seq, _)| *seq).collect();
                    let added = remote.len();
                    pending.extend(remote.into_iter().filter(|(seq, _)| !local.contains(seq)));
                    pending.sort_by_key(|(seq, _)| *seq);
                    info!(
                        "Merged {} WAL entries from object storage ({} already local)",
                        pending.len(),
                        added + local.len() - pending.len()
                    );
                }
            }

            // The log has to pick up exactly where the snapshot left off. Compaction keeps every
            // entry from the snapshot's replay point on, so a log that starts later belongs to a
            // later snapshot than the one loaded -- a backup whose `wal.log` was copied after
            // another snapshot compacted it, or a snapshot directory restored on its own. The
            // entries in between are in neither file. Inserts among them come back from the row
            // files, which is why this used to restore without complaint; a delete, an update or
            // a table created there does not, and the index and the rows then disagree for good.
            if let Some((first, _)) = pending.first() {
                if *first > from_seq {
                    return Err(crate::Error::Internal(format!(
                        "{} starts at entry {} but the snapshot in {} replays from entry {}; \
                         entries {}..{} are in neither, so starting would lose the writes they \
                         hold. Restore a snapshot and wal.log taken together.",
                        dir.join("wal.log").display(),
                        first,
                        Self::snapshot_dir(&dir).display(),
                        from_seq,
                        from_seq,
                        first
                    )));
                }
            }

            // Number this machine's next write after everything it is about to replay, and after
            // everything the bucket already holds. A machine recovering from object storage has
            // an empty local log, which opens counting from the snapshot's point -- so its first
            // write took the sequence of an entry already in a chunk, overwrote that chunk, and
            // was never mirrored (the mirror frontier was already past it). A promoted replica's
            // writes never reached the replicas following it; that is how this was found.
            if let Some(wal) = storage.wal.as_ref() {
                let after_pending = pending.last().map_or(0, |(seq, _)| seq + 1);
                let after_bucket = storage.wal_mirrored_next.load(Ordering::SeqCst);
                if let Ok(mut guard) = wal.lock() {
                    guard.advance_sequence_to(after_pending.max(after_bucket));
                }
            }
            let replayed = pending.len() as u64;
            if pending.is_empty() {
                info!(
                    "No WAL entries after sequence {}; recovery complete from snapshot alone",
                    from_seq
                );
            } else {
                storage.replica_applied_next.store(
                    pending.last().map_or(from_seq, |(seq, _)| seq + 1),
                    Ordering::SeqCst,
                );
                APPLYING_REPLICATED
                    .scope((), storage.replay(pending))
                    .await?;
            }
            if storage.replica_applied_next.load(Ordering::SeqCst) == 0 {
                storage
                    .replica_applied_next
                    .store(from_seq, Ordering::SeqCst);
            }
            APPLYING_REPLICATED
                .scope((), async {
                    storage.reindex_rows_the_graph_may_not_reflect().await?;
                    storage.index_rows_the_log_missed().await
                })
                .await?;
            let took = recovery_started.elapsed();
            info!(
                "Recovery took {:.2}s ({} WAL entries replayed)",
                took.as_secs_f64(),
                replayed
            );
            if let Ok(mut last) = storage.last_recovery.lock() {
                *last = Some((took.as_nanos() as u64, replayed));
            }
            storage.load_persisted_derived().await;
            // Entries this machine logged that never reached object storage -- an upload that
            // failed before the last shutdown -- go up now, before anything new is written.
            if storage.object_storage_enabled && !storage.replica {
                storage.mirror_wal(&[]).await?;
            }
            if !storage.replica {
                storage.storage.set_uncached_reads(false);
            }
        }

        Ok(storage)
    }

    async fn build(config: &StorageConfig) -> crate::Result<Self> {
        let storage = TieredStorage::new_from_config(config).await?;
        let index_selector = IntelligentIndexSelector::new();

        // 从配置或系统检测硬件能力
        let hardware_capabilities = Self::detect_hardware_capabilities();

        // 默认性能要求
        let performance_requirements = PerformanceRequirements {
            target_recall: 0.95,
            max_latency_ms: 100.0,
            min_throughput_qps: 1000.0,
            max_query_latency_ms: 50.0,
            min_recall_rate: 0.95,
            target_qps: 1000.0,
            min_insert_throughput: 100.0,
        };

        // 默认成本约束
        let cost_constraints = CostConstraints {
            max_memory_gb: 32.0,
            max_cpu_cores: 8,
            max_cost_per_month: 1000.0,
            max_storage_cost_per_tb: 50.0,
            max_compute_cost_per_hour: 10.0,
            total_budget_limit: 1000.0,
        };

        // 读取全局默认内存优化配置
        let default_memory_optimization = crate::config::env::var("DXMT_MEMORY_OPTIMIZATION")
            .ok()
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true); // 默认启用内存优化

        info!(
            "Global memory_optimization setting: {}",
            default_memory_optimization
        );

        // 创建 channel-based 插入架构（消除锁竞争）
        let (insert_tx, insert_rx) = mpsc::unbounded_channel();
        let tables_arc = Arc::new(RwLock::new(HashMap::new()));
        let replaying_flag = Arc::new(AtomicBool::new(false));

        // 启动后台插入处理器（单线程，无锁竞争）
        let tables_clone = tables_arc.clone();
        tokio::spawn(async move {
            insertion_processor(insert_rx, tables_clone).await;
        });

        info!("Channel-based insertion processor started (lock-free architecture)");

        Ok(Self {
            tables: tables_arc,
            storage: Arc::new(storage),
            tier_policy: Arc::new(crate::storage::tier_promotion::TierPromotionPolicy::new(
                crate::storage::tier_promotion::TierPromotionConfig::default(),
            )),
            object_storage_enabled: config.object_storage.is_some(),
            index_selector: Arc::new(index_selector),
            hardware_capabilities,
            performance_requirements,
            cost_constraints,
            default_memory_optimization,
            insert_tx,
            wal: None,
            wal_dir: None,
            snapshotting: Arc::new(tokio::sync::Mutex::new(())),
            text_rebuild: tokio::sync::Mutex::new(()),
            index_rebuilds: AtomicU64::new(0),
            index_rebuild_ns: AtomicU64::new(0),
            index_rebuild_swap_wait_ns: AtomicU64::new(0),
            last_recovery: StdMutex::new(None),
            last_snapshot_ns: StdMutex::new(None),
            _dir_lock: None,
            recovery_floors: StdMutex::new(HashMap::new()),
            replayed_rows: StdMutex::new(HashMap::new()),
            #[cfg(feature = "test-failpoints")]
            backup_pause: StdMutex::new(None),
            wal_manifest: Arc::new(StdMutex::new(WalManifest::default())),
            wal_mirrored_next: AtomicU64::new(0),
            wal_mirror_lock: tokio::sync::Mutex::new(()),
            manifest_write: tokio::sync::Mutex::new(()),
            manifest_version: StdMutex::new(None),
            conditional_writes: std::sync::atomic::AtomicBool::new(false),
            fenced: std::sync::atomic::AtomicBool::new(false),
            primary_epoch: AtomicU64::new(0),
            replica: false,
            replica_applied_next: AtomicU64::new(0),
            replica_stale: StdMutex::new(None),
            primary_token: None,
            primary_checked: StdMutex::new(None),
            primary_check_every: std::time::Duration::from_millis(
                crate::config::env::var("DXMT_PRIMARY_CHECK_MS")
                    .ok()
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(1_000),
            ),
            wal_retain_secs: AtomicU64::new(
                crate::config::env::var("DXMT_WAL_RETAIN_SECS")
                    .ok()
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(600),
            ),
            offloaded_snapshots: StdMutex::new(std::collections::VecDeque::new()),
            last_offloaded: AtomicU64::new(u64::MAX),
            replaying: replaying_flag,
            durability_error: Arc::new(StdMutex::new(None)),
            last_wal_sync_at: Arc::new(StdMutex::new(None)),
            last_snapshot_at: Arc::new(StdMutex::new(None)),
            last_backup_at: StdMutex::new(None),
            default_embedding: StdMutex::new(None),
            kb_table: crate::storage::knowledge_base::kb_table_name(),
            backup_failures: AtomicU64::new(0),
            snapshot_sequence: Arc::new(StdMutex::new(None)),
            snapshot_table_names: Arc::new(StdMutex::new(Vec::new())),
            dropped_since_snapshot: Arc::new(StdMutex::new(std::collections::HashSet::new())),
            snapshot_failures: AtomicU64::new(0),
            filter_path_exact_scan: AtomicU64::new(0),
            filter_path_code_scan: AtomicU64::new(0),
            filter_path_in_graph: AtomicU64::new(0),
            filter_path_pre_filter: AtomicU64::new(0),
            filter_path_post_filter: AtomicU64::new(0),
            filter_path_ns: Default::default(),
            filter_eval_ns: AtomicU64::new(0),
            filter_node_map_misses: AtomicU64::new(0),
            memory_guard: crate::storage::memory_guard::MemoryGuard::from_env(),
            disk_guard: crate::storage::disk_guard::DiskGuard::from_env(None),
            last_snapshot_error: Arc::new(StdMutex::new(None)),
            aliases: Arc::new(parking_lot::RwLock::new(std::collections::BTreeMap::new())),
        })
    }

    /// Whether this storage accepts writes at all: `Err` on a replica, and once durability has
    /// failed -- a WAL fsync failure, or another primary taking the bucket. For a caller that
    /// must say so before anything else.
    pub fn check_writable(&self) -> crate::Result<()> {
        self.ensure_writable()
    }

    fn ensure_writable(&self) -> crate::Result<()> {
        if self.replica && APPLYING_REPLICATED.try_with(|_| ()).is_err() {
            return Err(crate::Error::ReadOnlyReplica(
                "this server is a read replica following object storage; send writes to the \
                 primary"
                    .to_string(),
            ));
        }
        let failure = self
            .durability_error
            .lock()
            .map_err(|_| crate::Error::Internal("durability state lock poisoned".to_string()))?
            .clone();
        if let Some(message) = failure {
            return Err(crate::Error::ServiceUnavailable { message });
        }
        Ok(())
    }

    /// Refuse data that would grow the process when memory is nearly gone.
    ///
    /// Deliberately not part of `ensure_writable`, which eight paths share. Two of those are
    /// `delete_records` and `delete_table`: refusing a delete because memory is low blocks
    /// the one thing that would fix it. Two more are `wal_append` and `snapshot_inner`, and
    /// a server that cannot replay its log or write its snapshot after a memory event cannot
    /// come back from one. Only the paths that take new data from a caller consult this.
    fn ensure_memory_headroom(&self) -> crate::Result<()> {
        self.memory_guard.check()?;
        self.disk_guard.check()
    }

    fn mark_durability_failed(&self, error: &crate::Error) {
        if let Ok(mut state) = self.durability_error.lock() {
            if state.is_none() {
                *state = Some(format!(
                    "local WAL failed; storage is read-only until restart: {}",
                    error
                ));
            }
        }
    }

    /// Test-only hook for forcing the next WAL sync to fail.
    #[cfg(feature = "test-failpoints")]
    pub fn fail_next_wal_sync_for_test(&self) -> crate::Result<()> {
        self.with_wal_for_test(crate::wal::WalManager::fail_next_sync_for_test)
    }

    /// Test-only hook for tearing the next WAL write: half the frame reaches the file.
    #[cfg(feature = "test-failpoints")]
    pub fn tear_next_wal_write_for_test(&self) -> crate::Result<()> {
        self.with_wal_for_test(crate::wal::WalManager::tear_next_write_for_test)
    }

    /// Test-only: the next `backup_to` stops after its snapshot and before it copies anything.
    /// The first receiver fires when it gets there; dropping or sending on the sender lets it
    /// go on.
    #[cfg(feature = "test-failpoints")]
    pub fn pause_next_backup_for_test(
        &self,
    ) -> (
        tokio::sync::oneshot::Receiver<()>,
        tokio::sync::oneshot::Sender<()>,
    ) {
        let (reached_tx, reached_rx) = tokio::sync::oneshot::channel();
        let (resume_tx, resume_rx) = tokio::sync::oneshot::channel();
        if let Ok(mut pause) = self.backup_pause.lock() {
            *pause = Some((reached_tx, resume_rx));
        }
        (reached_rx, resume_tx)
    }

    #[cfg(feature = "test-failpoints")]
    fn with_wal_for_test(&self, f: impl FnOnce(&mut crate::wal::WalManager)) -> crate::Result<()> {
        let Some(ref wal) = self.wal else {
            return Err(crate::Error::InvalidInput(
                "WAL failpoint requires a WAL-backed storage".to_string(),
            ));
        };
        let mut guard = wal
            .lock()
            .map_err(|_| crate::Error::Internal("WAL lock poisoned".to_string()))?;
        f(&mut guard);
        Ok(())
    }

    /// 按序号顺序重放日志。
    ///
    /// `Insert` 直接交给 `insert_records`，UPSERT 语义让重复的 id 自然收敛到最后一次写入，
    /// 所以不需要单独的 Update 条目。`CreateTable`（旧变体）只有名字和维度，缺 distance_metric，
    /// 只能按 cosine 兜底并告警——真正的修复是让写入方改用 `CreateTableWithConfig`。
    async fn replay(&self, entries: Vec<(u64, WalEntry)>) -> crate::Result<()> {
        let total = entries.len();
        self.replaying.store(true, Ordering::SeqCst);

        let mut tables = 0usize;
        let mut records = 0usize;
        let mut skipped = 0usize;

        // Consecutive record entries for one table are applied as a run (see
        // `restore_records_from_wal`); anything else -- another table, a delete, a schema
        // change -- applies the run first, so the log's order is kept.
        const RUN: usize = 1_000;
        let mut run: Option<ReplayRun> = None;

        for (seq, entry) in entries {
            let entry = match entry {
                WalEntry::UpsertRecordV2 { table, record } => {
                    self.add_to_run(
                        &mut run,
                        table,
                        record,
                        None,
                        seq,
                        RUN,
                        &mut records,
                        &mut skipped,
                    )
                    .await;
                    continue;
                }
                WalEntry::UpsertRecordInRowFile { table, record, row } => {
                    self.add_to_run(
                        &mut run,
                        table,
                        record,
                        Some(row),
                        seq,
                        RUN,
                        &mut records,
                        &mut skipped,
                    )
                    .await;
                    continue;
                }
                other => other,
            };
            self.apply_run(&mut run, &mut records, &mut skipped).await;
            let outcome = match entry {
                WalEntry::CreateTableWithConfig { config } => {
                    self.create_table(config).await.map(|_| {
                        tables += 1;
                    })
                }
                WalEntry::CreateTable { table, dimension } => {
                    warn!(
                        "WAL seq {}: legacy CreateTable entry for '{}' carries no distance \
                         metric; assuming cosine. Search results may differ from before the \
                         restart.",
                        seq, table
                    );
                    let config = TableConfig {
                        name: table,
                        dimension,
                        distance_metric: DistanceMetric::Cosine,
                        description: None,
                        index_type: None,
                        text_fields: None,
                        enable_full_text_search: false,
                        memory_optimization: None,
                        payload_schema: None,
                        schema_version: 1,
                        named_vectors: None,
                        hnsw: None,
                        quantization: None,
                        payload_index: None,
                        embedding: None,
                        rerank: None,
                    };
                    self.create_table(config).await.map(|_| {
                        tables += 1;
                    })
                }
                WalEntry::Insert {
                    table,
                    record_id,
                    vector,
                    payload,
                } => {
                    let request = CreateRecordRequest {
                        id: record_id,
                        vector,
                        sparse_vector: None,
                        named_vectors: None,
                        payload,
                        text: None,
                    };
                    self.insert_records(&table, vec![request]).await.map(|_| {
                        records += 1;
                    })
                }
                WalEntry::UpsertRecord { table, request } => {
                    self.insert_records(&table, vec![request]).await.map(|_| {
                        records += 1;
                    })
                }
                WalEntry::UpsertRecordV2 { .. } | WalEntry::UpsertRecordInRowFile { .. } => {
                    unreachable!("record entries are batched above")
                }
                WalEntry::AlterTable { table, request } => {
                    self.alter_table(&table, request).await.map(|_| ())
                }
                WalEntry::Delete { table, record_id } => self
                    .delete_records(&table, vec![record_id])
                    .await
                    .map(|_| ()),
                WalEntry::DeleteTable { table } => self.delete_table(&table).await,
                WalEntry::Checkpoint { .. } => Ok(()),
            };

            // A single bad entry must not cost us the rest of the log. Most commonly this is a
            // Delete for a record the log never created, which is harmless.
            if let Err(e) = outcome {
                skipped += 1;
                warn!("WAL seq {} could not be replayed, skipping: {}", seq, e);
            }
        }

        self.apply_run(&mut run, &mut records, &mut skipped).await;

        self.replaying.store(false, Ordering::SeqCst);
        // Replay inserts one record at a time, so the normal insert path skips adaptive rebuilds
        // while the log is applied. Re-evaluate once against final table sizes so recovery
        // converges to the same index configuration as a fresh load.
        if let Err(e) = self.reevaluate_indexes().await {
            warn!("WAL replay completed but index re-evaluation failed: {}", e);
        }
        info!(
            "WAL replay finished: {} entries -> {} tables, {} records, {} skipped",
            total, tables, records, skipped
        );
        Ok(())
    }

    /// Take `{dir}/LOCK` exclusively, or say which directory is already in use. The OS drops
    /// the lock when the process exits, however it exits, so a crash leaves nothing stale.
    fn lock_dir(dir: &std::path::Path) -> crate::Result<std::fs::File> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("LOCK");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)?;
        match file.try_lock() {
            Ok(()) => Ok(file),
            Err(std::fs::TryLockError::WouldBlock) => Err(crate::Error::InvalidConfig(format!(
                "{} is in use by another process (it holds {}). Two processes writing one \
                 store corrupt it; stop the other one, or point this one at a different \
                 DXMT_ROOT_DIR / DXMT_WAL_DIR.",
                dir.display(),
                path.display()
            ))),
            Err(std::fs::TryLockError::Error(e)) => Err(crate::Error::Io(e)),
        }
    }

    /// After recovery, re-read into the graph every row it may hold an older version of.
    ///
    /// A write appends its row, updates the graph, and only then logs. A snapshot's graph
    /// reflects every row below the `indexed_below` it recorded; a row at or above it whose log
    /// entry never landed -- a crash, a full disk -- is a write the graph may not have, and for
    /// an update the graph then holds the *previous* vector under the same id: the counts
    /// agree, `GET` returns the new vector, and an ordinary search finds the record by the old
    /// one. So each record whose current row is at or above the floor is re-indexed from its
    /// row, unless replay indexed it from exactly that row. A table the snapshot did not hold
    /// was built by replay alone, so its floor is zero. A snapshot written before the floor was
    /// recorded is left alone, rather than turning the first start after an upgrade into a
    /// rebuild of every table.
    async fn reindex_rows_the_graph_may_not_reflect(&self) -> crate::Result<()> {
        let floors = self
            .recovery_floors
            .lock()
            .map(|mut f| std::mem::take(&mut *f))
            .unwrap_or_default();
        let mut replayed = self
            .replayed_rows
            .lock()
            .map(|mut r| std::mem::take(&mut *r))
            .unwrap_or_default();
        let names: Vec<String> = self
            .read_tables("reindex_rows_the_graph_may_not_reflect")
            .await
            .keys()
            .cloned()
            .collect();
        for name in names {
            let floor = match floors.get(&name) {
                Some(Some(floor)) => *floor,
                Some(None) => continue,
                None => 0,
            };
            let rows = replayed.remove(&name).unwrap_or_default();
            self.reindex_rows_from(&name, floor, &rows).await?;
        }
        Ok(())
    }

    /// Re-index, from the row file, every record of `name` whose row is at or above `floor`
    /// and was not indexed from that same row by replay (`replayed`).
    async fn reindex_rows_from(
        &self,
        name: &str,
        floor: u64,
        replayed: &HashMap<RecordId, u64>,
    ) -> crate::Result<usize> {
        let (index, vectors) = {
            let tables = self.read_tables("reindex_rows_from").await;
            let Some(data) = tables.get(name) else {
                return Ok(0);
            };
            let Some(file) = &data.vector_file else {
                return Ok(0);
            };
            if file.rows() <= floor {
                return Ok(0);
            }
            let stale: Vec<RecordId> = data
                .records
                .ids()
                .into_iter()
                .filter(|id| {
                    data.vector_row(id)
                        .is_some_and(|row| row >= floor && replayed.get(id) != Some(&row))
                })
                .collect();
            if stale.is_empty() {
                return Ok(0);
            }
            (data.index.clone(), data.vectors_for_ids(&stale))
        };
        warn!(
            "Table '{}': {} record(s) were written at or after the snapshot's graph without a \
             log entry for that version; re-indexing them from the row file",
            name,
            vectors.len()
        );
        for (id, _) in &vectors {
            // Not there at all is the other case this repairs; nothing to remove then.
            let _ = index.remove(id).await;
        }
        let count = vectors.len();
        index.insert_batch(vectors).await?;
        Ok(count)
    }

    /// After recovery, make each graph hold exactly the records the row files hold.
    ///
    /// A write reaches the row file, fsynced, before its WAL entry; when the entry never lands
    /// -- a crash in between, a disk that filled halfway through the frame, a backup that copied
    /// `records/` after the write and `wal.log` before its entry -- the rows have moved on and
    /// the log does not say so. Replay rebuilds the graph from the log, so:
    ///
    /// - an insert the log missed was never indexed;
    /// - an update it missed left the id out of the graph altogether, because replay skips each
    ///   older logged version of a record the row store already holds newer;
    /// - a delete it missed left the id in the graph with no record behind it.
    ///
    /// `GET` and the exact scan read the rows, the ordinary search reads the graph, and the two
    /// disagreed for the life of the table. `restore_table` already added the first kind against
    /// a snapshot's index; nothing covered replay or the other two. The counts gate the walk, as
    /// they do there. A missed insert and a missed delete that cancel out do not hide behind
    /// them: `reindex_rows_the_graph_may_not_reflect` runs first and puts the insert in the
    /// graph, which unbalances the counts (`a_lost_insert_and_a_lost_delete_that_cancel_out_are_
    /// both_repaired`). They can still hide in a table whose vectors have no row file, and
    /// behind a snapshot written before `indexed_below` was recorded.
    async fn index_rows_the_log_missed(&self) -> crate::Result<()> {
        let mut work = Vec::new();
        {
            let tables = self.read_tables("index_rows_the_log_missed").await;
            for (name, data) in tables.iter() {
                if data.index.len().await == data.records.len() {
                    continue;
                }
                let ids = data.records.ids();
                let missing = data.index.missing_ids(ids.iter()).await.unwrap_or_default();
                let extra = data
                    .index
                    .live_ids_not_in(|id| data.records.contains(id))
                    .await
                    .unwrap_or_default();
                if missing.is_empty() && extra.is_empty() {
                    continue;
                }
                warn!(
                    "Table '{}': after recovery {} record(s) in the row files were not in the \
                     index and {} in the index were not in the row files; reconciling",
                    name,
                    missing.len(),
                    extra.len()
                );
                work.push((data.index.clone(), data.vectors_for_ids(&missing), extra));
            }
        }
        for (index, vectors, extra) in work {
            for id in &extra {
                index.remove(id).await?;
            }
            if !vectors.is_empty() {
                index.insert_batch(vectors).await?;
            }
        }
        Ok(())
    }

    /// Add a record entry to the pending run, applying the run first if it is for another
    /// table or full.
    #[allow(clippy::too_many_arguments)]
    async fn add_to_run(
        &self,
        run: &mut Option<ReplayRun>,
        table: String,
        record: Record,
        row: Option<u64>,
        seq: u64,
        limit: usize,
        records: &mut usize,
        skipped: &mut usize,
    ) {
        if run
            .as_ref()
            .is_some_and(|(t, entries, _)| *t != table || entries.len() >= limit)
        {
            self.apply_run(run, records, skipped).await;
        }
        run.get_or_insert_with(|| (table, Vec::new(), seq))
            .1
            .push((record, row));
    }

    /// Apply the pending run, if any, counting what was applied and what was skipped.
    async fn apply_run(
        &self,
        run: &mut Option<ReplayRun>,
        records: &mut usize,
        skipped: &mut usize,
    ) {
        let Some((table, entries, first_seq)) = run.take() else {
            return;
        };
        let total = entries.len();
        match self.restore_records_from_wal(&table, entries).await {
            Ok(failed) => {
                for (id, e) in &failed {
                    warn!(
                        "WAL record '{}' (run from seq {}) could not be replayed, skipping: {}",
                        id, first_seq, e
                    );
                }
                *skipped += failed.len();
                *records += total - failed.len();
            }
            Err(e) => {
                warn!(
                    "WAL run of {} record(s) from seq {} could not be replayed, skipping: {}",
                    total, first_seq, e
                );
                *skipped += total;
            }
        }
    }

    /// Replay a run of current-format records for one table through the normal index path,
    /// then restore what a create request cannot carry. Replay is single-threaded and WAL
    /// writes are suppressed while it runs, so replacing the map values after indexing is
    /// deterministic. Returns the entries that could not be applied, each with its reason;
    /// the rest are applied.
    ///
    /// A run, not one record at a time: each record used to be its own trip through the
    /// insertion processor and two appends to the record store, each fsynced -- about 8.7 ms
    /// an entry, so the 3,000-entry tail of one test took 26 s to recover, and the tail a
    /// crash leaves after five minutes of ingest at 3,000 rows/s would have taken hours.
    ///
    /// `row` is where the entry says the vector is in the table's row file
    /// (`UpsertRecordInRowFile`); an entry that carries its vector passes `None`.
    async fn restore_records_from_wal(
        &self,
        table: &str,
        entries: Vec<(Record, Option<u64>)>,
    ) -> crate::Result<Vec<(RecordId, crate::Error)>> {
        self.restore_records_checked(table, entries, true).await
    }

    /// [`Self::restore_records_from_wal`], with the version check optional: the retry of a
    /// refused run has checked versions already, against the store as it was before the run.
    async fn restore_records_checked(
        &self,
        table: &str,
        entries: Vec<(Record, Option<u64>)>,
        check_versions: bool,
    ) -> crate::Result<Vec<(RecordId, crate::Error)>> {
        let mut failed = Vec::new();

        // One entry per id, the one sequential replay would have ended on: a later entry
        // replaces an earlier one unless the earlier carries a higher version (see below).
        let mut order: Vec<RecordId> = Vec::new();
        let mut latest: HashMap<RecordId, (Record, Option<u64>)> = HashMap::new();
        for (record, row) in entries {
            match latest.get(&record.id) {
                Some((kept, _)) if kept.version > record.version => {}
                Some(_) => {
                    latest.insert(record.id.clone(), (record, row));
                }
                None => {
                    order.push(record.id.clone());
                    latest.insert(record.id.clone(), (record, row));
                }
            }
        }

        // A mutation is logged after it has been applied and after the table lock is released,
        // so two concurrent updates to the same id can reach the log in the opposite order to
        // the one in which they took effect. Replaying that log blindly would resurrect the
        // older version. The record carries the version the writer assigned, so an entry that
        // is behind what is already restored is a reordered write, not a new one: skip it.
        //
        // A log written before `update_record` recorded its vector holds update entries
        // without one. The vector is still in the row file — the update wrote it there before it
        // logged anything — so read it back rather than re-index nothing. If it is not there
        // either, the record cannot be indexed and the entry is refused like any other bad one:
        // replay skips it and carries on, instead of corrupting the graph for every entry after.
        //
        // An entry that names its row reads exactly that row, whatever the record store says
        // the id's row is now: a later update or delete in the same log has moved or dropped
        // the id, and replay is about to re-apply those in order. The row file is append-only
        // and rows are never renumbered, so the row the entry was written with is still the
        // row. (A row the sweep has since punched reads as zeros; the entry after it in the
        // log is the delete that made it dead.)
        let mut records = Vec::with_capacity(order.len());
        let mut requests = Vec::with_capacity(order.len());
        let mut rows = Vec::with_capacity(order.len());
        {
            let tables = self.read_tables("restore_records_from_wal").await;
            let data = tables.get(table);
            for id in order {
                let (record, row) = latest.remove(&id).expect("every ordered id is kept");
                if check_versions
                    && data
                        .and_then(|data| data.records.get(&record.id))
                        .is_some_and(|existing| existing.version > record.version)
                {
                    continue;
                }
                let vector = if let Some(row) = row {
                    match data
                        .and_then(|data| data.vector_file.as_ref())
                        .and_then(|file| file.read(row))
                    {
                        Some(vector) => vector,
                        None => {
                            failed.push((
                                record.id.clone(),
                                crate::Error::InvalidVector(format!(
                                    "WAL entry for '{}' points at row {} of a row file that \
                                     does not hold it",
                                    record.id, row
                                )),
                            ));
                            continue;
                        }
                    }
                } else if record.vector.is_empty() {
                    data.map(|data| data.vector_for(&record))
                        .unwrap_or_default()
                } else {
                    record.vector.clone()
                };
                if vector.is_empty() {
                    failed.push((
                        record.id.clone(),
                        crate::Error::InvalidVector(format!(
                            "WAL entry for '{}' has no vector and none is stored for it",
                            record.id
                        )),
                    ));
                    continue;
                }
                requests.push(CreateRecordRequest {
                    id: record.id.clone(),
                    vector,
                    sparse_vector: record.sparse_vector.clone(),
                    named_vectors: record.named_vectors.clone(),
                    payload: Some(record.payload.clone()),
                    text: None,
                });
                rows.push(row);
                records.push(record);
            }
        }
        if records.is_empty() {
            return Ok(failed);
        }

        // An entry that named its row hands the row on, so the processor indexes the vector
        // without appending it a second time. The processor refuses a whole request for one
        // bad record; a run that fails is retried one entry at a time, so one bad entry costs
        // itself and not its neighbours.
        if let Err(e) = self
            .insert_records_with_rows(table, requests.clone(), rows.clone())
            .await
        {
            warn!(
                "WAL run of {} record(s) for '{}' refused as a whole ({}); replaying it one \
                 record at a time",
                records.len(),
                table,
                e
            );
            if records.len() == 1 {
                failed.push((records[0].id.clone(), e));
                return Ok(failed);
            }
            // Without the version check. The processor may have applied part of the refused
            // run before refusing it -- the record map first, then the graph, which is what
            // refused -- and that bumped each record's version past the one logged. Checked
            // again, every record of the run was then "behind what is restored" and skipped,
            // keeping the bumped version: the stateful walk's `r2 version 10 != 9`. The
            // versions were checked above against the store before the run; re-storing the
            // logged record is what restores them. The graph refused an id it already held
            // and the map did not (a node the snapshot's graph kept for a record the row
            // files have since deleted); the partial apply put the id in the map, so the
            // retry replaces the stale node rather than tripping on it.
            for (record, row) in records.into_iter().zip(rows) {
                let id = record.id.clone();
                match Box::pin(self.restore_records_checked(table, vec![(record, row)], false))
                    .await
                {
                    Ok(mut errors) => failed.append(&mut errors),
                    Err(e) => failed.push((id, e)),
                }
            }
            return Ok(failed);
        }

        let mut tables = self.write_tables("restore_records_from_wal").await;
        let data = tables
            .get_mut(table)
            .ok_or_else(|| crate::Error::TableNotFound(table.to_string()))?;
        let mut restored = Vec::with_capacity(records.len());
        let replaying = self.replaying.load(Ordering::SeqCst);
        let mut replayed = self.replayed_rows.lock().ok();
        for mut record in records {
            // The insert above put the vector in the row file and the row in the store; this
            // re-store of the record, which exists for the timestamps and version the create
            // request could not carry, has to keep that row or the record comes back with no
            // vector at all. It stored `None` here, which was hidden while every entry carried
            // its vector inline: the store then held a second copy of it per replayed record.
            let row = data
                .vector_file
                .as_ref()
                .and_then(|_| data.vector_row(&record.id));
            if row.is_some() {
                record.vector = Vec::new();
            }
            // Only while replaying: `import_table` comes through here too, and nothing would
            // ever empty the map behind it.
            if let (Some(row), true, Some(replayed)) = (row, replaying, replayed.as_mut()) {
                replayed
                    .entry(table.to_string())
                    .or_default()
                    .insert(record.id.clone(), row);
            }
            restored.push((Arc::new(record), row));
        }
        drop(replayed);
        // One append, and one fsync, for the run.
        data.records.insert_batch(restored.clone())?;
        for (record, _) in &restored {
            // The insert above indexed the record with the time it was *replayed*, because a
            // create request carries no timestamp; the map now holds the one it was written
            // with. A table created by this same replay has its payload index built from the
            // start, so that index kept the replay time, and `records/list` with a filter —
            // which pages by the index's timestamps — ordered every replayed record by when
            // recovery reached it. A record the log had upserted went through the path that
            // keeps its original time, which is older than every replay time, and sank to the
            // bottom. Found by the stateful walk: two bulk inserts sharing one id, then
            // `kill -9`. Re-file it under the time the map holds.
            if let Some(derived) = data.derived_mut().as_mut() {
                derived.payload.update_record(
                    &record.id,
                    &record.payload,
                    &record.payload,
                    crate::storage::payload_index::created_key(&record.created_at),
                );
            }
            if record.updated_at > data.table.updated_at {
                data.table.updated_at = record.updated_at;
            }
        }
        data.table.record_count = data.records.len() as u64;
        Ok(failed)
    }

    /// 一张表的快照元数据。索引本体在同名的 `.idx` 文件里。
    ///
    /// 记录也一并存下来：只存索引不够——索引里只有 id 和向量，payload、版本号、时间戳都在
    /// `records` map 里，光靠索引恢复出来的表 payload 全空。
    fn snapshot_dir(wal_dir: &std::path::Path) -> PathBuf {
        wal_dir.join("snapshot")
    }

    /// Whether writes go to a WAL, and so whether there is anything to snapshot.
    /// `DXMT_WAL_DIR=` (set and empty) builds a storage without one.
    pub fn is_persistent(&self) -> bool {
        self.wal.is_some()
    }

    /// 把当前所有表写成快照，并记下 WAL 的下一个序号。
    ///
    /// 恢复时先加载快照、再只回放该序号之后的日志，避免把每条记录重新插一遍索引——
    /// 那正是"启动时间随数据量线性增长"的来源。
    ///
    /// 快照与 WAL 之间的顺序很关键：先取序号、再落盘。反过来的话，取序号之前写入的条目
    /// 会既在快照里、又被当成尾部重放一次；插入是 upsert 所以不会错，只是白做功。
    pub async fn snapshot(&self) -> crate::Result<usize> {
        if self.replica {
            return Err(crate::Error::ReadOnlyReplica(
                "snapshots are the primary's: a replica's would be uploaded over the bucket it \
                 follows"
                    .to_string(),
            ));
        }
        // Serialize before reading the sequence number, not after: two snapshots that both
        // sampled the sequence and then queued would still truncate the log to the later one's
        // position while the earlier one's manifest is what ends up on disk.
        let _serialized = self.snapshotting.lock().await;
        self.snapshot_serialized().await
    }

    /// [`Self::snapshot`] for a caller already holding `snapshotting`, which `backup_to` does
    /// for as long as it copies: a snapshot in the middle of the copy compacts the log past the
    /// one the backup carries.
    async fn snapshot_serialized(&self) -> crate::Result<usize> {
        let started = std::time::Instant::now();
        let result = self.snapshot_inner().await;
        match &result {
            Ok(_) => {
                if let Ok(mut last) = self.last_snapshot_ns.lock() {
                    *last = Some(started.elapsed().as_nanos() as u64);
                }
                if let Ok(mut at) = self.last_snapshot_at.lock() {
                    *at = Some(chrono::Utc::now().to_rfc3339());
                }
                if let Ok(mut error) = self.last_snapshot_error.lock() {
                    *error = None;
                }
            }
            Err(error) => {
                self.snapshot_failures.fetch_add(1, Ordering::Relaxed);
                if let Ok(mut last_error) = self.last_snapshot_error.lock() {
                    *last_error = Some(error.to_string());
                }
            }
        }
        result
    }

    /// Snapshot implementation separated from the telemetry wrapper above. Keeping all exits
    /// in one place ensures `snapshot_failures` is updated even when an early filesystem check
    /// fails, without weakening the existing atomic-install/verified-before-compaction order.
    async fn snapshot_inner(&self) -> crate::Result<usize> {
        // A WAL fsync failure puts the process into a deliberately read-only state. Do not let a
        // later snapshot or backup turn an unacknowledged in-memory mutation into a new durable
        // recovery frontier; restart is the boundary at which the filesystem outcome is known.
        self.ensure_writable()?;
        let Some(ref wal) = self.wal else {
            return Err(crate::Error::InvalidInput(
                "snapshot requires a WAL-backed storage (use new_with_wal)".to_string(),
            ));
        };

        let seq = {
            let mut guard = wal
                .lock()
                .map_err(|_| crate::Error::Internal("WAL lock poisoned".to_string()))?;
            guard.force_flush()?;
            if let Ok(mut at) = self.last_wal_sync_at.lock() {
                *at = Some(chrono::Utc::now().to_rfc3339());
            }
            guard.next_sequence()
        };

        let dir = Self::snapshot_dir(self.wal_dir.as_ref().ok_or_else(|| {
            crate::Error::Internal("WAL enabled without a directory".to_string())
        })?);
        let staging = dir.with_extension("staging");
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging)?;

        // Two separate numbers, because they accuse different code. Waiting means somebody else
        // holds the table map's write lock; holding means this loop is slow. tokio's RwLock is
        // write-preferring, so either one stops every later reader as well — a soak turned up one
        // thread at 90% CPU and eight asleep for a minute and a half, and a single combined
        // figure would not have said which end to look at.
        // Under the lock: copy out what a consistent snapshot needs, and nothing else. The
        // serialising and the disk writes happen afterwards.
        //
        // This loop used to serialise every table and write both its index and its metadata to
        // disk while holding the table map's read lock — measured at 1.0-1.2s for a single
        // 60,000-row table, and tokio's `RwLock` is write-preferring, so for that whole time
        // writes queue and every later read queues behind them. Copying the rows is the one part
        // that genuinely has to happen under the lock: it is what makes the snapshot a point in
        // time rather than a smear across one.
        let before_lock = std::time::Instant::now();
        let waited;
        let pending: Vec<(String, crate::search::IndexWrapper, TableTemplate)> = {
            let tables = self.read_tables("snapshot").await;
            // Taken here, once the lock is ours. It was taken after the copy, so the copy was
            // reported as waiting and the copy itself as 0.0 s -- pointing at whoever held the
            // lock instead of at this loop.
            waited = before_lock.elapsed();
            tables
                .iter()
                .map(|(name, data)| {
                    (
                        name.clone(),
                        data.index.clone(),
                        TableTemplate {
                            table: data.table.clone(),
                            algorithm: data.selected_algorithm.clone(),
                            records: data.records.snapshot_records(),
                            // Which rows, not their contents: decoding every record of a disk
                            // table under this lock was seconds at 500K rows, with every write
                            // queued behind it (`scripts/replica_soak.py` against GCS).
                            mirrored_rows: self
                                .object_storage_enabled
                                .then(|| data.records.live_rows())
                                .flatten()
                                .map(|live| MirroredRows {
                                    file: data.vector_file.clone(),
                                    live,
                                    fallback: data.vector_rows.clone(),
                                    created_at: data.table.created_at,
                                }),
                            vector_file: data.vector_file.clone(),
                        },
                    )
                })
                .collect()
        };
        let lock_held = std::time::Instant::now();
        let copied_out = lock_held.duration_since(before_lock).saturating_sub(waited);

        let mut written = 0usize;
        let mut names = Vec::new();

        for (name, index, template) in pending {
            // 表名来自用户输入，不能直接当文件名。用序号命名，真实名字放进元数据。
            let stem = format!("t{}", written);
            let idx_path = staging.join(format!("{}.idx", stem));
            // `Ok(false)` means this index type has no on-disk form, which is normal for
            // BruteForce. An `Err` means one that does have a form failed to write it, and
            // recording that as `has_index: false` is a lie the restore path cannot detect —
            // it just rebuilds the graph from the rows, correctly and arbitrarily slowly.
            // Fail the snapshot instead: the WAL still covers everything, so the cost is a
            // warning and one skipped snapshot.
            // Before the graph is written, so every row under it is in what gets written.
            let indexed_below = template.vector_file.as_ref().map(|f| f.indexed_below());
            let has_index = index.snapshot_to(&idx_path).await.map_err(|e| {
                crate::Error::Internal(format!(
                    "table '{}': index would not write itself to the snapshot: {}",
                    name, e
                ))
            })?;

            let records = match template.mirrored_rows {
                None => template.records,
                Some(rows) => self.read_mirrored_rows(&name, rows).await?,
            };
            let head = rmp_serde::to_vec_named(&SnapshotHead {
                table: template.table,
                algorithm: template.algorithm,
                has_index,
                indexed_below,
            })
            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
            crate::search::index_persistence::write_index(
                &staging.join(format!("{}.meta", stem)),
                SNAPSHOT_KIND,
                &TableSnapshotFile { head, records },
            )?;

            names.push(name);
            written += 1;
        }
        let total = lock_held.elapsed() + waited + copied_out;
        if total >= std::time::Duration::from_secs(1) {
            warn!(
                "Snapshot took {:.1}s for {} table(s): {:.1}s waiting for the table lock, \
                 {:.1}s holding it to copy the rows out, {:.1}s writing to disk with it released",
                total.as_secs_f64(),
                written,
                waited.as_secs_f64(),
                copied_out.as_secs_f64(),
                lock_held.elapsed().as_secs_f64()
            );
        }

        // Carry over tables that are not in memory. Eviction removes a table from `tables`
        // while leaving it perfectly valid in the previous snapshot — writing only what is
        // resident drops it from the new manifest, and since the WAL is then compacted past
        // the point that created it, the table is gone at the next restart. Nobody deleted it.
        //
        // An evicted table cannot have changed: every write path calls ensure_resident first,
        // so it would be back in memory before anything touched it. Its segments from the
        // previous snapshot are therefore still current and can simply be copied across.
        // Taken before the carry-forward and cleared after the install, so a drop that
        // arrives while this snapshot is being written is held over for the next one
        // rather than forgotten by it.
        let dropped_at_start = self.dropped_table_names();
        if dir.exists() {
            if let Ok(previous) = crate::search::index_persistence::read_index::<SnapshotManifest>(
                &dir.join("manifest"),
                "snapshot-manifest",
            ) {
                let resident: std::collections::HashSet<String> = names.iter().cloned().collect();
                for (i, name) in previous.names.iter().enumerate() {
                    // A dropped table is non-resident too. Copying it forward is what made
                    // DROP non-durable: the deletion never reached the snapshot lineage.
                    if resident.contains(name) || dropped_at_start.contains(name) {
                        continue;
                    }
                    let from = format!("t{}", i);
                    let to = format!("t{}", written);
                    if std::fs::copy(
                        dir.join(format!("{}.meta", from)),
                        staging.join(format!("{}.meta", to)),
                    )
                    .is_err()
                    {
                        warn!(
                            "Evicted table '{}' has no snapshot metadata to carry over",
                            name
                        );
                        continue;
                    }
                    // The index file is optional — BruteForce has no on-disk form.
                    let _ = std::fs::copy(
                        dir.join(format!("{}.idx", from)),
                        staging.join(format!("{}.idx", to)),
                    );
                    names.push(name.clone());
                    written += 1;
                }
            }
        }

        crate::search::index_persistence::write_index(
            &staging.join("manifest"),
            "snapshot-manifest",
            &SnapshotManifest {
                wal_next_sequence: seq,
                tables: written,
                names: names.clone(),
            },
        )?;
        if let Ok(mut sequence) = self.snapshot_sequence.lock() {
            *sequence = Some(seq);
        }
        if let Ok(mut cached) = self.snapshot_table_names.lock() {
            cached.clone_from(&names);
        }

        // 落盘后先读一遍。截断日志会让快照成为唯一副本，所以"写完了"不够，必须"读得回来"
        // 才算数——否则一个写坏的快照配上被截断的日志，就是静默的数据丢失。
        Self::verify_snapshot(&staging, written)?;

        // 原子替换：中途崩了要么是旧快照、要么是新快照，不会是半个。
        let previous = dir.with_extension("previous");
        let _ = std::fs::remove_dir_all(&previous);
        if dir.exists() {
            std::fs::rename(&dir, &previous)?;
        }
        std::fs::rename(&staging, &dir)?;
        let _ = std::fs::remove_dir_all(&previous);

        // The manifest that is now live does not name them, so the tombstones have done
        // their job. Remove exactly the ones this snapshot filtered — a drop that landed
        // mid-write is not in that set and stays for the next snapshot.
        if let Ok(mut dropped) = self.dropped_since_snapshot.lock() {
            for name in &dropped_at_start {
                dropped.remove(name);
            }
        }

        // Push the segments to object storage before trimming the log. If the upload fails the
        // local snapshot is still there and still verified, so compaction stays safe — but the
        // remote copy is the one that survives losing this machine, so say so loudly.
        let offloaded = !self.object_storage_enabled
            // Nothing was logged since the last one: the bucket has this snapshot already. The
            // periodic snapshot used to upload the same tens of megabytes every interval on an
            // idle table, into the version a recovering machine may be reading.
            || self.last_offloaded.load(Ordering::SeqCst) == seq
            || match self.offload_snapshot(&dir, written, seq).await {
                Ok(bytes) => {
                    info!("Snapshot offloaded to object storage: {} bytes", bytes);
                    // The mirror frontier does not jump to `seq`. Recovery would not miss the
                    // entries between -- the snapshot holds them -- but a replica follows the
                    // chunks, not the snapshots, and found a hole there: every write logged
                    // before a snapshot and mirrored after it was skipped. On GCS, whose
                    // uploads are slow enough for that to be most snapshots, replicas went
                    // stale every minute.
                    true
                }
                Err(e) => {
                    warn!("Snapshot is local-only, upload failed: {}", e);
                    false
                }
            };
        // With object storage, the log may only be trimmed to what the bucket holds. When the
        // snapshot did not reach it, the entries the bucket lacks stay in the local log for
        // `mirror_wal` to upload, and the bucket's own chunks stay too. Trimming to `seq`
        // regardless -- as this did -- and pruning the chunks below it left a bucket holding
        // neither the snapshot nor the entries: a machine recovering from it lost them.
        //
        // That holds when the snapshot did reach it too: the entries it covers that are not yet
        // in a chunk are still to be mirrored, for the replicas, and are read back from here.
        let keep_from = if self.object_storage_enabled {
            seq.min(self.wal_mirrored_next.load(Ordering::SeqCst))
        } else {
            seq
        };

        // Only now is it safe to drop the log this snapshot covers. Compacting any earlier
        // leaves a window where neither the previous snapshot nor the log can reconstruct
        // those writes.
        match wal
            .lock()
            .map_err(|_| crate::Error::Internal("WAL lock poisoned".to_string()))
            .and_then(|mut g| g.truncate_before(keep_from))
        {
            Ok(reclaimed) if reclaimed > 0 => {
                info!(
                    "WAL compacted below sequence {}: {} bytes reclaimed",
                    keep_from, reclaimed
                );
            }
            Ok(_) => {}
            // The snapshot is durable already, so a failed compaction costs disk and startup
            // time, not correctness.
            Err(e) => warn!("Snapshot written but WAL compaction failed: {}", e),
        }
        if self.object_storage_enabled && offloaded {
            if let Some(keep_from) = self.retained_prune_point(seq) {
                self.prune_wal_chunks(keep_from).await;
                self.prune_snapshot_versions(keep_from).await;
            }
        }

        // Now, and only now, collect segments that maintenance superseded. The snapshot just
        // written names the partitions the index currently has, and the WAL below `seq` is
        // gone, so nothing that can still be recovered refers to anything else.
        //
        // Maintenance cannot do this itself: at the moment it splits a partition, the previous
        // snapshot still names the original, and recovery from it has to be able to read the
        // records inside.
        if self.object_storage_enabled {
            self.collect_superseded_segments().await;
        }

        info!(
            "Snapshot written: {} tables at WAL sequence {}",
            written, seq
        );
        Ok(written)
    }

    /// The records a mirrored snapshot listed under the lock, with their vectors, read in
    /// chunks under short read locks.
    ///
    /// A row is never reused or rewritten, so the rows listed are the table at the snapshot's
    /// point whatever has been written since; the one thing that could change them is the
    /// reclamation sweep punching a row that died in between, and it takes `snapshotting`.
    /// The lock is still needed per chunk, because the record store lives in the table map.
    async fn read_mirrored_rows(
        &self,
        name: &str,
        rows: MirroredRows,
    ) -> crate::Result<Vec<Arc<Record>>> {
        const CHUNK: usize = 16_384;
        let mut out = Vec::with_capacity(rows.live.len());
        let mut undecodable = 0usize;
        for chunk in rows.live.chunks(CHUNK) {
            let numbers: Vec<u32> = chunk.iter().map(|(row, _)| *row).collect();
            let decoded = {
                let tables = self.read_tables("snapshot").await;
                let data = tables
                    .get(name)
                    .filter(|d| d.table.created_at == rows.created_at)
                    .ok_or_else(|| {
                        crate::Error::Internal(format!(
                            "table '{name}' was dropped, replaced or evicted while it was being \
                             snapshotted; the next snapshot takes it"
                        ))
                    })?;
                data.records.decode_rows(&numbers)
            };
            for ((_, vector_row), record) in chunk.iter().zip(decoded) {
                let Some(record) = record else {
                    undecodable += 1;
                    continue;
                };
                if !record.vector.is_empty() {
                    out.push(record);
                    continue;
                }
                let row = vector_row.or_else(|| rows.fallback.get(&record.id).copied());
                match rows.file.as_ref().zip(row).and_then(|(f, r)| f.read(r)) {
                    Some(vector) => {
                        let mut with_vector = (*record).clone();
                        with_vector.vector = vector;
                        out.push(Arc::new(with_vector));
                    }
                    None => out.push(record),
                }
            }
        }
        if undecodable > 0 {
            warn!(
                "table '{name}': {undecodable} rows would not decode and are not in the snapshot"
            );
        }
        Ok(out)
    }

    /// 快照段在对象存储里的键。`version` is the snapshot's WAL sequence; `None` is the layout
    /// before versions, `snapshot/{name}`, still read when there is no `snapshot/CURRENT`.
    ///
    /// Each snapshot is written under its own prefix and published by rewriting
    /// `snapshot/CURRENT` once every segment is up. They used to be rewritten in place,
    /// manifest first: a primary killed during the upload -- 1-2 s for each snapshot against
    /// GCS -- left a manifest claiming the new sequence beside the previous snapshot's
    /// records, and a machine recovering from the bucket skipped the entries in between.
    /// `scripts/replica_soak.py` lost the last batch before a `kill -9` that way.
    fn snapshot_key(version: Option<&str>, name: &str) -> String {
        match version {
            Some(dir) => format!("snapshot/{dir}/{name}"),
            None => format!("snapshot/{name}"),
        }
    }

    /// A snapshot's directory: its sequence, then the epoch of the primary that wrote it. Two
    /// primaries can take a snapshot at one sequence -- the replaced one numbers its writes
    /// from its own log -- and without the epoch the late one's segments would land in the
    /// directory the new one had already published.
    fn snapshot_dir_name(seq: u64, epoch: u64) -> String {
        if epoch == 0 {
            format!("v{seq:020}")
        } else {
            format!("v{seq:020}-e{epoch}")
        }
    }

    const SNAPSHOT_CURRENT: &'static str = "snapshot/CURRENT";

    /// Delete the snapshots in object storage older than `keep_from`, and the unversioned
    /// layout, which a `CURRENT` supersedes. Called with the same point the WAL chunks are
    /// pruned to, so a machine that read `CURRENT` a moment ago still finds its segments.
    async fn prune_snapshot_versions(&self, keep_from: u64) {
        use crate::storage::object::traits::ObjectStorage;
        let options = crate::storage::object::types::ListObjectsOptions {
            prefix: Some("snapshot/".to_string()),
            max_keys: None,
            continuation_token: None,
            delimiter: None,
        };
        let Ok(listed) = self.storage.list_objects(&options).await else {
            return;
        };
        for object in listed.objects {
            let key = object.key;
            if key == Self::SNAPSHOT_CURRENT {
                continue;
            }
            let older = match key.strip_prefix("snapshot/v") {
                Some(rest) => rest
                    .split(['/', '-'])
                    .next()
                    .and_then(|v| v.parse::<u64>().ok())
                    .is_some_and(|v| v < keep_from),
                None => true, // unversioned
            };
            if older {
                let _ = self.storage.delete_object(&key).await;
            }
        }
    }

    /// 列出一个快照目录下需要随身携带的段文件。
    fn snapshot_segments(dir: &std::path::Path, tables: usize) -> Vec<String> {
        let mut names = vec!["manifest".to_string()];
        for i in 0..tables {
            names.push(format!("t{}.meta", i));
            let idx = format!("t{}.idx", i);
            if dir.join(&idx).exists() {
                names.push(idx);
            }
        }
        names
    }

    /// 把快照段推到对象存储。
    ///
    /// 这是分层存储在写路径上的落点：本地磁盘是缓存，对象存储才是持久归宿。段文件是不可变的
    /// ——一次快照写一组新的键——所以不需要读改写，也没有并发覆盖问题。
    async fn offload_snapshot(
        &self,
        dir: &std::path::Path,
        tables: usize,
        seq: u64,
    ) -> crate::Result<u64> {
        use crate::storage::object::traits::ObjectStorage;

        let version = Self::snapshot_dir_name(seq, self.primary_epoch.load(Ordering::SeqCst));
        let mut bytes = 0u64;
        for name in Self::snapshot_segments(dir, tables) {
            let data = std::fs::read(dir.join(&name))?;
            bytes += data.len() as u64;
            self.storage
                .put_object(
                    &Self::snapshot_key(Some(&version), &name),
                    bytes::Bytes::from(data),
                    Some("application/octet-stream"),
                )
                .await
                .map_err(|e| {
                    crate::Error::Internal(format!("uploading snapshot segment {}: {}", name, e))
                })?;
        }
        // Last, and only once every segment is up: this is what makes it the snapshot. The
        // manifest first, conditionally -- a primary another has replaced stops here -- and
        // then `CURRENT`, which only a bucket without one in the manifest is read by.
        self.update_manifest(|m| m.snapshot = Some(version.clone()))
            .await?;
        self.storage
            .put_object(
                Self::SNAPSHOT_CURRENT,
                bytes::Bytes::from(
                    serde_json::json!({ "version": seq, "dir": version }).to_string(),
                ),
                Some("application/json"),
            )
            .await
            .map_err(|e| crate::Error::Internal(format!("publishing snapshot: {}", e)))?;
        self.last_offloaded.store(seq, Ordering::SeqCst);
        Ok(bytes)
    }

    /// 从对象存储把快照取回本地，供恢复使用。返回是否取到。
    async fn fetch_snapshot(&self, dir: &std::path::Path) -> crate::Result<bool> {
        use crate::storage::object::traits::ObjectStorage;

        // The WAL manifest's, else `CURRENT`'s (a bucket written before the manifest named it),
        // else the unversioned layout.
        let current = || async {
            let v = self
                .storage
                .get_object(Self::SNAPSHOT_CURRENT)
                .await
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())?;
            v["dir"].as_str().map(str::to_string).or_else(|| {
                v["version"]
                    .as_u64()
                    .map(|seq| Self::snapshot_dir_name(seq, 0))
            })
        };
        let version = match self.manifest().ok().and_then(|m| m.snapshot) {
            Some(dir) => Some(dir),
            None => current().await,
        };
        let version = version.as_deref();
        let manifest_bytes = match self
            .storage
            .get_object(&Self::snapshot_key(version, "manifest"))
            .await
        {
            Ok(b) => b,
            // 对象存储里没有快照是正常情况（首次启动），不是错误。
            Err(_) => return Ok(false),
        };

        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("manifest"), &manifest_bytes)?;

        let manifest: SnapshotManifest = crate::search::index_persistence::read_index(
            &dir.join("manifest"),
            "snapshot-manifest",
        )?;

        for i in 0..manifest.tables {
            for name in [format!("t{}.meta", i), format!("t{}.idx", i)] {
                match self
                    .storage
                    .get_object(&Self::snapshot_key(version, &name))
                    .await
                {
                    Ok(data) => std::fs::write(dir.join(&name), data)?,
                    // `.idx` 是可选的：BruteForce 之类的索引没有磁盘格式，`has_index` 为 false。
                    Err(_) if name.ends_with(".idx") => {}
                    Err(e) => {
                        return Err(crate::Error::Internal(format!(
                            "downloading snapshot segment {}: {}",
                            name, e
                        )))
                    }
                }
            }
        }

        info!(
            "Snapshot fetched from object storage: {} tables",
            manifest.tables
        );
        Ok(true)
    }

    /// 确保 `table_name` 在内存里；被驱逐过的表从对象存储回填。
    ///
    /// 驱逐只发生在快照覆盖了该表之后，所以段文件一定存在——找不到就说明这张表真的不存在，
    /// 交给调用方的 TableNotFound 去报。
    /// Take the table map's read lock, reporting it if it is held for long enough to matter.
    async fn read_tables(&self, who: &'static str) -> TimedRead<'_> {
        let asked = std::time::Instant::now();
        let guard = self.tables.read().await;
        record_lock_wait(asked.elapsed(), LOCK_READ);
        TimedRead {
            guard,
            who,
            since: std::time::Instant::now(),
        }
    }

    /// Take the table map's write lock, reporting it if it is held for long enough to matter.
    async fn write_tables(&self, who: &'static str) -> TimedWrite<'_> {
        let asked = std::time::Instant::now();
        let guard = self.tables.write().await;
        record_lock_wait(asked.elapsed(), LOCK_WRITE);
        TimedWrite {
            guard,
            who,
            since: std::time::Instant::now(),
        }
    }

    async fn ensure_resident(&self, table_name: &str) -> crate::Result<()> {
        use crate::storage::tier_promotion::Tier;

        self.tier_policy.record_access(table_name);

        if self
            .read_tables("ensure_resident")
            .await
            .contains_key(table_name)
        {
            // Resident tables sit in Warm so the policy's demote_to_cold rule applies to them;
            // it only demotes out of Warm.
            self.tier_policy.set_current_tier(table_name, Tier::Warm);
            return Ok(());
        }
        // Not resident because it was dropped, not because it was evicted. The manifest on
        // disk still names it; loading from there would undo the delete.
        if self.dropped_table_names().contains(table_name) {
            return Ok(());
        }
        let Some(ref wal_dir) = self.wal_dir else {
            return Ok(());
        };

        let dir = Self::snapshot_dir(wal_dir);
        let manifest: SnapshotManifest = match crate::search::index_persistence::read_index(
            &dir.join("manifest"),
            "snapshot-manifest",
        ) {
            Ok(m) => m,
            Err(_) => return Ok(()),
        };

        let Some(i) = manifest.names.iter().position(|n| n == table_name) else {
            return Ok(());
        };

        let stem = format!("t{}", i);
        let meta = read_snapshot_meta(&dir.join(format!("{}.meta", stem)))?;

        // Two concurrent readers can both find the table missing and both start restoring. The
        // second one finds it present by the time it takes the write lock, and restore_table's
        // create_table would fail on the duplicate — so check again and let the loser bail.
        if self
            .read_tables("ensure_resident")
            .await
            .contains_key(table_name)
        {
            return Ok(());
        }

        info!("Rehydrating evicted table '{}' from snapshot", table_name);
        let floor = meta.indexed_below;
        self.restore_table(&dir.join(format!("{}.idx", stem)), meta)
            .await?;
        // Nothing is replayed on this path, so every row the snapshot's graph may not reflect
        // is re-read.
        if let Some(floor) = floor {
            self.reindex_rows_from(table_name, floor, &HashMap::new())
                .await?;
        }
        self.tier_policy.set_current_tier(table_name, Tier::Warm);
        Ok(())
    }

    /// 把一份自洽的备份写到 `dest`。
    ///
    /// WAL 会在快照校验通过后截断，所以快照是它所覆盖那段数据的**唯一副本**——`snapshot/` 目录
    /// 从缓存变成了必须备份的数据。这个方法产出的目录包含恢复所需的全部内容。
    ///
    /// 先强制快照再拷贝，顺序不能反：快照会把内存里尚未落盘的写入固定下来，并把日志压到
    /// 快照点。反过来拷的话，拿到的是一份旧快照配一段与之不匹配的日志。
    /// Where backups may be written: `DXMT_BACKUP_ROOT`, or the WAL directory's parent.
    ///
    /// The parent rather than the WAL directory itself, because a backup inside it is refused
    /// for a separate reason -- so the default is the nearest place that is both obvious and
    /// allowed.
    fn backup_root(wal_dir: &std::path::Path) -> PathBuf {
        crate::config::layout::Layout::from_env().backups_for(wal_dir)
    }

    /// The destination as it will actually be resolved, with symlinks followed.
    ///
    /// A directory that does not exist yet cannot be canonicalised, so its parent is resolved
    /// instead and the final component appended -- which is also what stops `../` and a symlink
    /// in the parent chain from stepping outside the root checked against it.
    fn resolve_backup_destination(dest: &std::path::Path) -> crate::Result<PathBuf> {
        if dest.exists() {
            return Ok(std::fs::canonicalize(dest).unwrap_or_else(|_| dest.to_path_buf()));
        }
        let parent = dest.parent().unwrap_or_else(|| std::path::Path::new("."));
        let parent = std::fs::canonicalize(parent).map_err(|e| {
            crate::Error::InvalidInput(format!(
                "backup destination {}: its parent directory does not exist ({e})",
                dest.display()
            ))
        })?;
        let name = dest.file_name().ok_or_else(|| {
            crate::Error::InvalidInput("backup destination must name a directory".to_string())
        })?;
        Ok(parent.join(name))
    }

    /// The path a table export may be written to or read from, checked against
    /// `DXMT_BACKUP_ROOT`.
    ///
    /// Both directions need this and for the same reason. The path comes off the wire and this
    /// server has no authentication: unbounded, an export is a write primitive that can drop a
    /// copy of a table anywhere the process can write, and an import is a read primitive that
    /// will happily try to parse `/etc/shadow`. The root is the same one `POST /backup` is held
    /// to, so an operator configures one directory and both obey it.
    /// `writing`: the path is an export's destination, which must also not be inside the
    /// store. The backup root defaults to the WAL directory's parent, so without that an export
    /// to `<store>/wal.log` -- or a snapshot, a row file -- replaced the database's own file
    /// with a table export. `backup_to_inner` already refused this for backups.
    fn resolve_export_path(&self, path: &std::path::Path, writing: bool) -> crate::Result<PathBuf> {
        let Some(ref wal_dir) = self.wal_dir else {
            return Err(crate::Error::InvalidInput(
                "export and import need WAL-backed storage; this instance has no durable state"
                    .to_string(),
            ));
        };
        let resolved = Self::resolve_backup_destination(path)?;
        let root = Self::backup_root(wal_dir);
        let root = std::fs::canonicalize(&root).unwrap_or(root);
        if !resolved.starts_with(&root) {
            return Err(crate::Error::InvalidInput(format!(
                "{} is outside {}; set DXMT_BACKUP_ROOT to the directory exports may be read \
                 from and written to",
                path.display(),
                root.display()
            )));
        }
        if writing {
            let store = std::fs::canonicalize(wal_dir).unwrap_or_else(|_| wal_dir.clone());
            if resolved.starts_with(&store) {
                return Err(crate::Error::InvalidInput(format!(
                    "{} is inside the data directory {}; an export there could overwrite the \
                     database's own files",
                    path.display(),
                    store.display()
                )));
            }
        }
        Ok(resolved)
    }

    /// The file aliases live in, beside the WAL.
    fn aliases_path(wal_dir: &std::path::Path) -> PathBuf {
        wal_dir.join("aliases")
    }

    /// Read the alias file at startup. A missing or unreadable file means no aliases.
    ///
    /// Unreadable is a warning rather than a startup error, unlike an unreadable snapshot: an
    /// alias is a name, not data, and refusing to start because one cannot be read would take a
    /// database offline over a pointer to it. The names are in the log of the requests that
    /// created them and can be set again.
    pub(crate) fn load_aliases(&self) {
        let Some(ref wal_dir) = self.wal_dir else {
            return;
        };
        let path = Self::aliases_path(wal_dir);
        if !path.exists() {
            return;
        }
        match std::fs::read(&path)
            .map_err(|e| e.to_string())
            .and_then(|bytes| {
                rmp_serde::from_slice::<std::collections::BTreeMap<String, String>>(&bytes)
                    .map_err(|e| e.to_string())
            }) {
            Ok(loaded) => {
                if !loaded.is_empty() {
                    info!("Loaded {} table alias(es)", loaded.len());
                }
                *self.aliases.write() = loaded;
            }
            Err(e) => warn!(
                "{} could not be read ({e}); starting with no aliases",
                path.display()
            ),
        }
    }

    /// Write the alias file, atomically.
    fn save_aliases(&self) -> crate::Result<()> {
        let Some(ref wal_dir) = self.wal_dir else {
            return Ok(());
        };
        let path = Self::aliases_path(wal_dir);
        let bytes = {
            let aliases = self.aliases.read();
            rmp_serde::to_vec_named(&*aliases)
                .map_err(|e| crate::Error::Serialization(e.to_string()))?
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Through a temporary file: a crash during the write must leave the previous set, not a
        // half-written one that loads as "no aliases" and points every caller at nothing.
        let tmp = path.with_extension("tmp");
        {
            // Written and synced through one writable handle, closed before the rename. It was
            // `File::open(&tmp)?.sync_all()`: a read-only handle, which Linux will flush and
            // Windows refuses with "Access is denied", so on Windows no alias could be set.
            use std::io::Write as _;
            let mut file = std::fs::File::create(&tmp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// The table a name refers to: itself, or what an alias points at.
    ///
    /// A real table wins over an alias of the same name. That cannot normally happen --
    /// [`Self::set_alias`] refuses a name a table already has, and `create_table` refuses a name
    /// an alias already has -- but a file edited by hand or a table created before the alias
    /// file was read should resolve to the thing that holds data.
    fn resolve<'a>(&self, name: &'a str) -> std::borrow::Cow<'a, str> {
        {
            let aliases = self.aliases.read();
            if aliases.is_empty() {
                return std::borrow::Cow::Borrowed(name);
            }
            if let Some(target) = aliases.get(name) {
                // Unless a table actually has this name. `try_read` because `resolve` is called
                // from paths that already hold the map, and a failed try must mean "resolve the
                // alias" rather than "do not" -- shadowing is refused at creation time, so the
                // alias is the answer in every case this cannot check.
                let shadowed = self
                    .tables
                    .try_read()
                    .map(|tables| tables.contains_key(name))
                    .unwrap_or(false);
                if !shadowed {
                    return std::borrow::Cow::Owned(target.clone());
                }
            }
        }
        std::borrow::Cow::Borrowed(name)
    }

    /// Point `alias` at `table`, creating it or moving it.
    ///
    /// Moving an existing alias is the swap this exists for, and it is one map write: a reader
    /// arriving during it sees the old table or the new one, never neither.
    pub async fn set_alias(&self, alias: &str, table: &str) -> crate::Result<()> {
        self.ensure_writable()?;
        crate::utils::validate_table_name(alias)
            .map_err(|e| crate::Error::InvalidInput(format!("alias {e}")))?;
        if !self.read_tables("set_alias").await.contains_key(table) {
            // Resolved through an alias first, so an alias may point at another alias's target,
            // which is a chain a reader never has to follow.
            let resolved = self.resolve(table).into_owned();
            if !self.read_tables("set_alias").await.contains_key(&resolved) {
                return Err(crate::Error::TableNotFound(table.to_string()));
            }
        }
        let target = self.resolve(table).into_owned();
        if self.read_tables("set_alias").await.contains_key(alias) {
            return Err(crate::Error::InvalidInput(format!(
                "'{alias}' is a table on this server; an alias cannot shadow one"
            )));
        }
        self.aliases.write().insert(alias.to_string(), target);
        self.save_aliases()?;
        Ok(())
    }

    /// Remove an alias. The table it pointed at is untouched.
    pub async fn delete_alias(&self, alias: &str) -> crate::Result<()> {
        self.ensure_writable()?;
        if self.aliases.write().remove(alias).is_none() {
            return Err(crate::Error::TableNotFound(format!("alias '{alias}'")));
        }
        self.save_aliases()?;
        Ok(())
    }

    /// Every alias and what it points at.
    pub fn list_aliases(&self) -> std::collections::BTreeMap<String, String> {
        self.aliases.read().clone()
    }

    /// Drop every alias pointing at a table, for when the table goes.
    ///
    /// An alias left behind would resolve to a name nothing holds, and the error a caller got
    /// would name the table rather than the alias they used.
    fn drop_aliases_for(&self, table: &str) {
        let removed = {
            let mut aliases = self.aliases.write();
            let before = aliases.len();
            aliases.retain(|_, target| target != table);
            before != aliases.len()
        };
        if removed {
            if let Err(e) = self.save_aliases() {
                warn!("aliases for the deleted table '{table}' were not written out: {e}");
            }
        }
    }

    /// Write one table -- its configuration and every record, vectors included -- to a file.
    ///
    /// This is the unit `POST /backup` is not. A backup is the whole instance and restoring it
    /// means stopping the server and replacing its directory; moving one table to another
    /// machine, or keeping a copy of it before a migration, had no answer short of paging it
    /// out through the API and writing the insert loop yourself.
    ///
    /// The file is a `tN.meta` in the snapshot format, which is the format `tests/compat/`
    /// already guards, so an export taken today opens on a later build for the same reason a
    /// snapshot does. **Vectors are read back out of the row files** and written into it: a
    /// local snapshot leaves them there on purpose, because the row files are the durable copy
    /// on this machine, and an export that did the same would arrive somewhere else with every
    /// vector empty.
    ///
    /// The graph is not exported. It is a pure function of the records, the import rebuilds it,
    /// and shipping it would tie the file to an index format as well as to a record format --
    /// so the cost is a rebuild on import, which is the same cost a restore from a WAL pays.
    pub async fn export_table(
        &self,
        name: &str,
        dest: &std::path::Path,
    ) -> crate::Result<serde_json::Value> {
        let name = &*self.resolve(name);
        let path = self.resolve_export_path(dest, true)?;
        self.ensure_resident(name).await?;

        // Copied out under the read lock so the export is a point in time, and the vectors are
        // read from the row file afterwards -- the same split `snapshot` makes, for the same
        // reason: holding the map across the file reads stops every other table.
        let (table, records, vector_file, rows) = {
            let tables = self.read_tables("export_table").await;
            let data = tables
                .get(name)
                .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))?;
            // `snapshot_records` is not what to use here: it returns nothing for a
            // disk-backed table, deliberately, because a local snapshot leaves the rows in the
            // row files. An export has to carry them, so the records are walked out of the
            // store the way the object-storage mirror does -- and this is the expensive half
            // of an export on a disk table, one decode per record.
            let mut records: Vec<Arc<Record>> = Vec::with_capacity(data.records.len());
            let mut rows: Vec<Option<u64>> = Vec::with_capacity(data.records.len());
            data.records.for_each(|record| {
                rows.push(data.vector_row(&record.id));
                records.push(record);
            });
            (data.table.clone(), records, data.vector_file.clone(), rows)
        };

        let mut missing_vectors = 0usize;
        let records: Vec<Arc<Record>> = records
            .into_iter()
            .zip(rows)
            .map(|(record, row)| {
                if !record.vector.is_empty() {
                    return record;
                }
                match vector_file.as_ref().zip(row).and_then(|(f, r)| f.read(r)) {
                    Some(vector) => {
                        let mut with_vector = (*record).clone();
                        with_vector.vector = vector;
                        Arc::new(with_vector)
                    }
                    None => {
                        missing_vectors += 1;
                        record
                    }
                }
            })
            .collect();

        let count = records.len();
        let head = rmp_serde::to_vec_named(&SnapshotHead {
            table: table.clone(),
            algorithm: IndexAlgorithm::BruteForce {
                enable_parallel: false,
            },
            has_index: false,
            indexed_below: None,
        })
        .map_err(|e| crate::Error::Serialization(e.to_string()))?;
        crate::search::index_persistence::write_index(
            &path,
            SNAPSHOT_KIND,
            &TableSnapshotFile { head, records },
        )?;

        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        info!(
            "Exported table '{}' to {}: {} records, {} bytes",
            name,
            path.display(),
            count,
            bytes
        );
        Ok(serde_json::json!({
            "table": name,
            "path": path.display().to_string(),
            "records": count,
            "bytes": bytes,
            // Zero on any healthy table. Anything else means a record whose vector was neither
            // in memory nor readable from the row file, and saying so beats an import that
            // quietly lands rows that cannot be searched.
            "records_without_vectors": missing_vectors,
        }))
    }

    /// Read a table back out of a file written by [`Self::export_table`].
    ///
    /// Records keep their ids, payloads, versions and `created_at`: they are applied through the
    /// same path WAL replay uses, not re-inserted through the API, so an import does not quietly
    /// restamp every row and reorder `records/list`. Each one is logged, so the imported table
    /// survives a restart without waiting for a snapshot.
    ///
    /// The index is rebuilt from the records as they land, because the file carries no graph.
    pub async fn import_table(
        &self,
        src: &std::path::Path,
        rename: Option<&str>,
        overwrite: bool,
    ) -> crate::Result<serde_json::Value> {
        self.ensure_writable()?;
        let path = self.resolve_export_path(src, false)?;
        if !path.is_file() {
            return Err(crate::Error::InvalidInput(format!(
                "{} is not a file written by POST /tables/{{t}}/export",
                src.display()
            )));
        }

        let meta = read_snapshot_meta(&path)?;
        let name = rename.unwrap_or(&meta.table.name).to_string();
        if name.is_empty() {
            return Err(crate::Error::InvalidInput(
                "the imported table needs a name".to_string(),
            ));
        }

        let exists = self.read_tables("import_table").await.contains_key(&name);
        if exists {
            if !overwrite {
                return Err(crate::Error::InvalidInput(format!(
                    "table '{name}' already exists; pass \"overwrite\": true to replace it, or \
                     \"name\" to import it under a different one"
                )));
            }
            self.delete_table(&name).await?;
        }

        let config = TableConfig {
            name: name.clone(),
            dimension: meta.table.dimension,
            distance_metric: meta.table.distance_metric,
            description: meta.table.description.clone(),
            index_type: None,
            text_fields: meta.table.text_fields.clone(),
            enable_full_text_search: meta.table.enable_full_text_search,
            memory_optimization: meta.table.memory_optimization,
            hnsw: meta.table.hnsw.clone(),
            quantization: meta.table.quantization.clone(),
            payload_index: meta.table.payload_index.clone(),
            payload_schema: meta.table.payload_schema.clone(),
            schema_version: meta.table.schema_version,
            named_vectors: meta.table.named_vectors.clone(),
            // Was `None`, so an exported text table imported as one that refuses text. The
            // config names an environment variable for the key, never the key, so carrying it
            // to another server carries nothing secret; if that server lacks the variable, the
            // first embed says so by name, which beats a table that silently stopped embedding.
            // `create_table` only probes the model when the dimension is zero, and an imported
            // table always has one, so this makes no provider call.
            embedding: meta.table.embedding.clone(),
            rerank: meta.table.rerank.clone(),
        };
        self.create_table(config).await?;

        let total = meta.records.len();
        let mut imported = 0usize;
        // In batches: one `wal_append` per record would fsync once per record, and holding all
        // of them for a single append would double the memory of a large import.
        const BATCH: usize = 1_000;
        for chunk in meta.records.chunks(BATCH) {
            let mut entries = Vec::with_capacity(chunk.len());
            let mut batch = Vec::with_capacity(chunk.len());
            for record in chunk {
                let record = (**record).clone();
                entries.push(WalEntry::UpsertRecordV2 {
                    table: name.clone(),
                    record: record.clone(),
                });
                batch.push((record, None));
            }
            // A chunk at a time, as replay does: one record at a time was one fsynced trip
            // through the processor and two record-store appends per record.
            if let Some((id, e)) = self
                .restore_records_from_wal(&name, batch)
                .await?
                .into_iter()
                .next()
            {
                return Err(crate::Error::InvalidInput(format!(
                    "record '{id}' in {} could not be imported: {e}",
                    path.display()
                )));
            }
            imported += chunk.len();
            self.wal_append(entries).await?;
        }

        info!(
            "Imported table '{}' from {}: {} records",
            name,
            path.display(),
            imported
        );
        Ok(serde_json::json!({
            "table": name,
            "path": path.display().to_string(),
            "records": imported,
            "records_in_file": total,
            "replaced": exists,
        }))
    }

    /// Copy the store to `dest`, a new directory inside the backup root. See
    /// `docs/operations/backup-restore.md`; [`Self::restore_backup`] is the way back.
    pub async fn backup_to(&self, dest: &std::path::Path) -> crate::Result<serde_json::Value> {
        let result = self.backup_to_inner(dest).await;
        match &result {
            Ok(_) => {
                if let Ok(mut at) = self.last_backup_at.lock() {
                    *at = Some(chrono::Utc::now().to_rfc3339());
                }
            }
            Err(_) => {
                self.backup_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
        result
    }

    /// Make the knowledge-base table if `tablename` is its name, it does not exist yet, and the
    /// server has a default model. Called by the paths a first note takes -- insert, search,
    /// text and hybrid search -- so the table appears when someone uses it rather than on
    /// every server at startup. Two first writes racing are fine: the loser finds it made.
    async fn ensure_kb_table(&self, tablename: &str) -> crate::Result<()> {
        // A replica has the primary's tables, the knowledge-base table among them once the
        // primary made it; it makes none of its own.
        if self.replica || self.kb_table.as_deref() != Some(tablename) {
            return Ok(());
        }
        let Some(embedding) = self.default_embedding() else {
            return Ok(());
        };
        // Not `get_table`, which calls this.
        self.ensure_resident(tablename).await?;
        if self
            .read_tables("ensure_kb_table")
            .await
            .contains_key(tablename)
        {
            return Ok(());
        }
        match self
            .create_table(crate::storage::knowledge_base::kb_table_config(
                tablename, embedding,
            ))
            .await
        {
            Ok(table) => {
                info!(
                    "Knowledge-base table '{}' created on first use ({} dimensions, model '{}')",
                    tablename,
                    table.dimension,
                    table
                        .embedding
                        .as_ref()
                        .map(|e| e.model.as_str())
                        .unwrap_or("")
                );
                Ok(())
            }
            Err(crate::Error::TableAlreadyExists(_)) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// The model a table gets when it is created with neither a dimension nor a model.
    pub fn default_embedding(&self) -> Option<crate::types::table::EmbeddingConfig> {
        self.default_embedding.lock().ok().and_then(|e| e.clone())
    }

    pub fn set_default_embedding(&self, embedding: Option<crate::types::table::EmbeddingConfig>) {
        if let Ok(mut slot) = self.default_embedding.lock() {
            *slot = embedding;
        }
    }

    /// The directory this store lives in, when it has one.
    pub fn data_dir(&self) -> Option<&std::path::Path> {
        self.wal_dir.as_deref()
    }

    /// Where backups may be written for this store: `DXMT_BACKUP_ROOT`, or beside the store.
    /// `None` without a WAL directory, which has nothing to back up.
    pub fn backup_root_dir(&self) -> Option<PathBuf> {
        self.wal_dir.as_deref().map(Self::backup_root)
    }

    /// Delete the oldest scheduled backups under `root` until `keep` remain, and return what was
    /// deleted.
    ///
    /// Only directories this server's schedule wrote are candidates: named `auto-<timestamp>`
    /// and holding the `BACKUP.json` a finished backup ends with. A backup someone took by hand,
    /// one still being written, and anything else an operator keeps there are never touched.
    /// The timestamp sorts as text, so the oldest come first.
    pub fn prune_scheduled_backups(
        root: &std::path::Path,
        keep: usize,
    ) -> crate::Result<Vec<PathBuf>> {
        let mut scheduled: Vec<PathBuf> = match std::fs::read_dir(root) {
            Ok(entries) => entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.is_dir()
                        && p.join("BACKUP.json").is_file()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with(SCHEDULED_BACKUP_PREFIX))
                })
                .collect(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        scheduled.sort();
        let excess = scheduled.len().saturating_sub(keep);
        let removed: Vec<PathBuf> = scheduled.into_iter().take(excess).collect();
        for path in &removed {
            std::fs::remove_dir_all(path)?;
        }
        Ok(removed)
    }

    /// Where a table's derived indexes are kept between a clean shutdown and the next start.
    fn derived_path(wal_dir: &std::path::Path, table: &str) -> PathBuf {
        wal_dir.join("derived").join(format!(
            "{}.bin",
            crate::types::table::Table::table_id(table)
        ))
    }

    /// Write every built payload and sparse index beside the store, for the next start to read
    /// back instead of rebuilding. Called at a clean shutdown, after the shutdown snapshot,
    /// when nothing can write any more; returns how many tables were written.
    ///
    /// The rebuild this saves is 23-27 s of every core at 10M rows. Each file records the
    /// record store's row count and live count, and the next start uses it only if both still
    /// match -- a crash, or anything that writes in between, leaves it unused -- and deletes it
    /// either way, so it is never read twice.
    pub async fn persist_derived(&self) -> crate::Result<usize> {
        let Some(wal_dir) = self.wal_dir.clone() else {
            return Ok(0);
        };
        let mut written = 0usize;
        let tables = self.read_tables("persist_derived").await;
        for (name, data) in tables.iter() {
            let Some(rows) = data.records.row_count() else {
                continue; // in memory: rebuilt from the records, which is a map walk
            };
            let derived = data.derived.read();
            let Some(derived) = derived.as_ref() else {
                continue;
            };
            let path = Self::derived_path(&wal_dir, name);
            let started = std::time::Instant::now();
            let result = tokio::task::block_in_place(|| -> crate::Result<()> {
                std::fs::create_dir_all(path.parent().expect("derived/ has a parent"))?;
                let tmp = path.with_extension("tmp");
                {
                    let file = std::fs::File::create(&tmp)?;
                    let mut writer = std::io::BufWriter::new(file);
                    bincode::serialize_into(
                        &mut writer,
                        &PersistedDerivedOut {
                            version: DERIVED_FILE_VERSION,
                            rows,
                            live: data.records.len(),
                            created: data.records.created_column(),
                            derived,
                        },
                    )?;
                    use std::io::Write as _;
                    writer.flush()?;
                    writer.get_ref().sync_all()?;
                }
                std::fs::rename(&tmp, &path)?;
                Ok(())
            });
            match result {
                Ok(()) => {
                    written += 1;
                    info!(
                        "Table '{}': derived indexes written for the next start in {:.1}s",
                        name,
                        started.elapsed().as_secs_f64()
                    );
                }
                Err(e) => warn!("Table '{}': derived indexes not written: {}", name, e),
            }
        }
        Ok(written)
    }

    /// Install the derived indexes a clean shutdown wrote, for every table whose record store is
    /// exactly as it was then. Runs at the end of recovery, before anything is served. Every
    /// file is deleted once read, used or not.
    async fn load_persisted_derived(&self) {
        let Some(wal_dir) = self.wal_dir.clone() else {
            return;
        };
        if !wal_dir.join("derived").is_dir() {
            return;
        }
        let tables = self.read_tables("load_persisted_derived").await;
        for (name, data) in tables.iter() {
            let path = Self::derived_path(&wal_dir, name);
            if !path.is_file() {
                continue;
            }
            let started = std::time::Instant::now();
            let read = tokio::task::block_in_place(|| -> crate::Result<PersistedDerivedIn> {
                let file = std::fs::File::open(&path)?;
                Ok(bincode::deserialize_from(std::io::BufReader::new(file))?)
            });
            let _ = std::fs::remove_file(&path);
            let persisted = match read {
                Ok(persisted) => persisted,
                Err(e) => {
                    warn!(
                        "Table '{}': derived indexes on disk unreadable, rebuilding: {}",
                        name, e
                    );
                    continue;
                }
            };
            let now = (data.records.row_count(), data.records.len());
            if persisted.version != DERIVED_FILE_VERSION
                || now != (Some(persisted.rows), persisted.live)
                || data.table.payload_index.clone().unwrap_or_default()
                    != *persisted.derived.payload.policy()
            {
                info!(
                    "Table '{}': derived indexes on disk do not match the records any more; rebuilding",
                    name
                );
                continue;
            }
            if let Some(created) = persisted.created {
                data.records.publish_created(created);
            }
            *data.derived.write() = Some(persisted.derived);
            info!(
                "Table '{}': derived indexes read from disk in {:.1}s instead of rebuilt",
                name,
                started.elapsed().as_secs_f64()
            );
        }
    }

    /// Replace the store at `store` with the backup at `backup`, offline, and open the result
    /// to prove it recovers.
    ///
    /// Refused while any process has the store open (its `LOCK`). Nothing is deleted: whatever
    /// the store held is moved into a sibling directory, named in the report, first. The
    /// backup is checked against its own manifest before anything moves, so an incomplete copy
    /// is refused rather than half-restored. Restoring used to be five `cp -a` lines in a
    /// document, one of which -- `vectors/` -- older versions of that document did not have.
    pub async fn restore_backup(
        config: &StorageConfig,
        backup: &std::path::Path,
        store: &std::path::Path,
    ) -> crate::Result<RestoreReport> {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(backup.join("BACKUP.json")).map_err(|e| {
                crate::Error::InvalidInput(format!(
                    "{} is not a backup: no readable BACKUP.json ({e})",
                    backup.display()
                ))
            })?)
            .map_err(|e| {
                crate::Error::InvalidInput(format!("{}/BACKUP.json: {e}", backup.display()))
            })?;
        let files: Vec<String> = manifest["files"]
            .as_array()
            .map(|files| {
                files
                    .iter()
                    .filter_map(|f| f.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let missing: Vec<&String> = files
            .iter()
            .filter(|f| !backup.join(f.as_str()).is_file())
            .collect();
        if !missing.is_empty() {
            return Err(crate::Error::InvalidInput(format!(
                "{} is incomplete; its manifest names files that are not there: {:?}",
                backup.display(),
                missing
            )));
        }

        let moved_aside = {
            let _lock = Self::lock_dir(store)?;
            let present: Vec<PathBuf> = std::fs::read_dir(store)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.file_name().is_some_and(|n| n != "LOCK"))
                .collect();
            let aside = if present.is_empty() {
                None
            } else {
                let name = format!(
                    "{}.before-restore-{}",
                    store
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("store"),
                    chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
                );
                let aside = store.with_file_name(name);
                std::fs::create_dir_all(&aside)?;
                for path in present {
                    let name = path.file_name().expect("a directory entry has a name");
                    std::fs::rename(&path, aside.join(name))?;
                }
                Some(aside)
            };
            for file in &files {
                let to = store.join(file);
                if let Some(parent) = to.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::copy(backup.join(file), &to)?;
            }
            aside
        };

        let opened = Self::new_with_wal(config, Some(store.to_path_buf()))
            .await
            .map_err(|e| {
                crate::Error::Internal(format!(
                    "the backup was copied into {} but does not open: {e}{}",
                    store.display(),
                    moved_aside
                        .as_ref()
                        .map(|a| format!("; the previous store is in {}", a.display()))
                        .unwrap_or_default()
                ))
            })?;
        let tables = opened
            .list_tables()
            .await?
            .into_iter()
            .map(|t| (t.name, t.record_count))
            .collect();
        Ok(RestoreReport {
            moved_aside,
            tables,
        })
    }

    async fn backup_to_inner(&self, dest: &std::path::Path) -> crate::Result<serde_json::Value> {
        let Some(ref wal_dir) = self.wal_dir else {
            return Err(crate::Error::InvalidInput(
                "backup requires WAL-backed storage; this instance has no durable state"
                    .to_string(),
            ));
        };

        // A backup must be a separate tree. Writing it into DXMT_WAL_DIR would make the source
        // and destination overlap (`snapshot/snapshot/...`), and writing it below DXMT_WAL_DIR
        // would make a later recursive copy increasingly self-referential. Reject both before
        // taking a snapshot so an operator gets a deterministic configuration error instead of
        // a partially written backup.
        let source = std::fs::canonicalize(wal_dir).unwrap_or_else(|_| wal_dir.clone());
        let destination = Self::resolve_backup_destination(dest)?;
        if destination == source || destination.starts_with(&source) {
            return Err(crate::Error::InvalidInput(format!(
                "backup destination {} must not be inside WAL directory {}",
                dest.display(),
                wal_dir.display()
            )));
        }

        // And it must be somewhere this server was told it may write.
        //
        // `path` comes off the wire, and nothing else bounded it: a request could name any
        // directory the process can write to, and the handler would create it and fill it with
        // a copy of the database. That is a write primitive as well as an exfiltration route,
        // and it needs no authentication because this server has none -- but it is worth
        // closing on its own terms, because an authenticated caller pointing a backup at the
        // wrong filesystem is a mistake rather than an attack and deserves an error either way.
        //
        // The root defaults to the WAL directory's parent, which is where a backup taken with
        // no configuration at all should land. `DXMT_BACKUP_ROOT` moves it -- to the mount the
        // backups actually belong on, usually.
        let root = Self::backup_root(wal_dir);
        let root = std::fs::canonicalize(&root).unwrap_or(root);
        if !destination.starts_with(&root) {
            return Err(crate::Error::InvalidInput(format!(
                "backup destination {} is outside {}; set DXMT_BACKUP_ROOT to the directory \
                 backups may be written to",
                dest.display(),
                root.display()
            )));
        }

        // Held until the last file is copied. The copy runs snapshot, row files, then
        // `wal.log`, and a periodic snapshot landing in between compacted the log past the one
        // this backup carries: the entries between the two replay points were in neither half,
        // and a delete or an update among them was gone from the restore.
        let _serialized = self.snapshotting.lock().await;
        let tables = self.snapshot_serialized().await?;

        #[cfg(feature = "test-failpoints")]
        {
            let pause = self.backup_pause.lock().ok().and_then(|mut p| p.take());
            if let Some((reached, resume)) = pause {
                let _ = reached.send(());
                let _ = resume.await;
            }
        }

        std::fs::create_dir_all(dest)?;
        let mut copied = Vec::new();
        let mut bytes = 0u64;

        // 快照目录
        let snapshot_src = Self::snapshot_dir(wal_dir);
        if snapshot_src.exists() {
            let snapshot_dst = dest.join("snapshot");
            std::fs::create_dir_all(&snapshot_dst)?;
            for entry in std::fs::read_dir(&snapshot_src)? {
                let entry = entry?;
                if entry.path().is_file() {
                    let name = entry.file_name();
                    let n = std::fs::copy(entry.path(), snapshot_dst.join(&name))?;
                    bytes += n;
                    copied.push(format!("snapshot/{}", name.to_string_lossy()));
                }
            }
        }

        // The row files: a table's records (`records/<t>.rec` + `.rec.log`) and full
        // vectors (`vectors/<t>.f32|f16` + `.rows`). Since these went to disk the snapshot
        // carries neither, so a backup without them restores every table empty. The log and
        // row list are copied before their data file: each store refuses a data file that
        // is behind its log, and a copy taken data-first while an insert lands would be.
        for sub in ["records", "vectors"] {
            let src = wal_dir.join(sub);
            if !src.exists() {
                continue;
            }
            let dst = dest.join(sub);
            std::fs::create_dir_all(&dst)?;
            let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&src)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file())
                .collect();
            let is_sidecar = |p: &std::path::Path| {
                let n = p.to_string_lossy();
                n.ends_with(".log") || n.ends_with(".rows")
            };
            files.sort_by_key(|p| (!is_sidecar(p), p.clone()));
            for path in files {
                let name = path.file_name().unwrap_or_default().to_owned();
                let n = std::fs::copy(&path, dst.join(&name))?;
                bytes += n;
                copied.push(format!("{sub}/{}", name.to_string_lossy()));
            }
        }

        // `fulltext/` is deliberately not copied, though it now sits beside these.
        //
        // It is derived state — every document in it was built from a record in `records/` by
        // `text_document_for` — and a backup is taken while the server is running, so copying a
        // live Tantivy index file by file could capture a half-written segment set with no way
        // to tell. Restoring rebuilds it instead: `restore_table` finds an index holding no
        // documents for a table that has records and re-indexes from the rows. Deterministic,
        // already tested, and the only cost is the first start after a restore.
        //
        // 快照点之后的日志尾巴。刚做完快照通常很短，但不能不带——恢复要靠它补上
        // 快照与备份完成之间的那几条写入。
        // The aliases, which are not in the log and not in the snapshot: a backup restored
        // without them comes up with the tables and none of the names anything calls them by.
        let aliases = Self::aliases_path(wal_dir);
        if aliases.exists() {
            let n = std::fs::copy(&aliases, dest.join("aliases"))?;
            bytes += n;
            copied.push("aliases".to_string());
        }

        let wal_log = wal_dir.join("wal.log");
        if wal_log.exists() {
            let n = std::fs::copy(&wal_log, dest.join("wal.log"))?;
            bytes += n;
            copied.push("wal.log".to_string());
        }

        let manifest = serde_json::json!({
            "created_at": chrono::Utc::now().to_rfc3339(),
            "tables": tables,
            "files": copied,
            "bytes": bytes,
            "restore": "Stop the server, replace the contents of DXMT_WAL_DIR with this \
                        directory, and start it again.",
        });
        std::fs::write(
            dest.join("BACKUP.json"),
            serde_json::to_vec_pretty(&manifest)
                .map_err(|e| crate::Error::Serialization(e.to_string()))?,
        )?;

        info!(
            "Backup written to {}: {} tables, {} bytes",
            dest.display(),
            tables,
            bytes
        );
        Ok(manifest)
    }

    /// 直接把一张表移出内存，不看热度。仅供测试构造"已驱逐"状态。
    ///
    /// 生产路径请用 [`Self::evict_cold_tables`]——它会先快照，这一步不会。
    pub async fn evict_table_for_test(&self, name: &str) {
        self.write_tables("evict_table_for_test").await.remove(name);
    }

    /// 给定规模，决定这张表该用什么索引。
    ///
    /// **只能有这一处。** 分区的判断原本只写在 `rebuild_index` 里，而 `reevaluate_indexes`
    /// 直接比较选择器的输出——于是一张已经分区的表每次扫描都被判为"不对"，重建一次又拿回
    /// 分区索引，下次扫描再来一遍。O(n) 的无限循环，有测试专门盯着。
    /// A table's own graph and quantization parameters, applied over whatever the selector
    /// chose for it.
    ///
    /// The selector decides the *family* — brute force, HNSW, HNSW-PQ — from the data's shape,
    /// and that stays automatic. What it should not decide alone is how hard to build: `m` was
    /// hardcoded at 16 and `ef_construction` came from an environment variable, so a server
    /// holding a 5,000-row lookup table beside a 10M-row corpus built both the same way. These
    /// overrides are per table, persisted with it, and applied on rebuilds too — otherwise the
    /// first re-evaluation would quietly undo them.
    fn apply_table_index_params(
        algorithm: &mut IndexAlgorithm,
        hnsw: Option<&crate::types::table::HnswParams>,
        quantization: Option<&crate::types::table::QuantizationParams>,
    ) {
        let (graph_m, graph_ef_construction, graph_ef_search) = hnsw
            .map(|h| (h.m, h.ef_construction, h.ef_search))
            .unwrap_or((None, None, None));

        match algorithm {
            IndexAlgorithm::HNSW {
                m,
                ef_construction,
                ef_search,
            } => {
                if let Some(v) = graph_m {
                    *m = v;
                }
                if let Some(v) = graph_ef_construction {
                    *ef_construction = v;
                }
                if let Some(v) = graph_ef_search {
                    *ef_search = v;
                }
            }
            IndexAlgorithm::HnswPQ {
                m,
                ef_construction,
                ef_search,
                num_subvectors,
                num_centroids,
                quantize_at_rows,
                fast_scan,
                ..
            } => {
                if let Some(v) = graph_m {
                    *m = v;
                }
                if let Some(v) = graph_ef_construction {
                    *ef_construction = v;
                }
                if let Some(v) = graph_ef_search {
                    *ef_search = v;
                }
                if let Some(q) = quantization {
                    if let Some(v) = q.num_subvectors {
                        *num_subvectors = v;
                    }
                    if let Some(v) = q.num_centroids {
                        *num_centroids = v;
                    }
                    if let Some(v) = q.quantize_at_rows {
                        *quantize_at_rows = Some(v);
                    }
                    if let Some(v) = q.fast_scan {
                        *fast_scan = v;
                    }
                }
            }
            // Brute force has no graph to tune, and the three unimplemented families are
            // refused before they are built. Silently ignoring parameters on those is right:
            // the alternative is refusing a create over a field that describes an index the
            // table did not get.
            _ => {}
        }
    }

    /// One decision function for `rebuild_index` and `reevaluate_indexes`.
    ///
    /// `memory_optimization` is the table's own setting (or the global default), the same
    /// switch `create_table` decides by. Until it was passed in, creation chose from the
    /// memory-optimized selector and every later re-evaluation from the size-based one, so
    /// the sweep saw a mismatch on a table that was built exactly as configured.
    async fn decide_algorithm(
        &self,
        record_count: usize,
        dimension: usize,
        distance_metric: DistanceMetric,
        memory_optimization: bool,
        hnsw: Option<&crate::types::table::HnswParams>,
        quantization: Option<&crate::types::table::QuantizationParams>,
    ) -> crate::Result<IndexDecision> {
        let config = IndexConfig {
            dataset_size: record_count,
            dimension,
            distance_metric,
            // Not hardcoded: a rebuild that assumed a different machine than table creation did
            // would pick a different algorithm on the same box.
            memory_budget_mb: (self.hardware_capabilities.available_memory_gb as usize) * 512,
            performance_requirements: self.performance_requirements.clone(),
            cost_constraints: self.cost_constraints.clone(),
            hardware_capabilities: self.hardware_capabilities.clone(),
        };

        let mut decision = if memory_optimization {
            self.select_memory_optimized_index(&config).await?
        } else {
            IntelligentIndexSelector::new()
                .select_optimal_index(&config)
                .await?
        };

        // The selector picks the family; the table says how hard to build. Applied here rather
        // than at creation so a rebuild and the periodic re-evaluation keep the table's
        // parameters instead of reverting it to the server's defaults.
        Self::apply_table_index_params(&mut decision.selected_algorithm, hnsw, quantization);

        // Threshold on bytes of vector data, not row count. Row count says nothing about
        // memory: a million 1536-dimension vectors is 6 GB, a million 64-dimension vectors is
        // 256 MB, and only one of those needs to leave memory.
        //
        // `DXMT_PARTITION_THRESHOLD` (a row count) is still read so existing configurations do
        // not silently change meaning, but it is deprecated.
        let vector_bytes = record_count * dimension * std::mem::size_of::<f32>();
        let threshold_bytes = crate::config::env::var("DXMT_PARTITION_THRESHOLD_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(512 * 1024 * 1024);

        let by_rows = crate::config::env::var("DXMT_PARTITION_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .is_some_and(|rows| record_count >= rows);

        if self.object_storage_enabled && (vector_bytes >= threshold_bytes || by_rows) {
            let nlist = ((record_count as f64).sqrt() as usize).clamp(16, 4096);
            decision.selected_algorithm = IndexAlgorithm::PartitionedIvf {
                nlist,
                nprobe: (nlist / 8).max(4),
                max_resident_partitions: (nlist / 4).max(8),
            };
        }

        if let Some(forced) = Self::forced_algorithm(dimension) {
            decision.selected_algorithm = forced;
        }

        Ok(decision)
    }

    #[inline(never)]
    fn attach_store(
        index: crate::search::partitioned_ivf::PartitionedIvfIndex,
        store: Arc<dyn crate::storage::object::traits::ObjectStorage + Send + Sync>,
    ) -> Box<dyn VectorIndex> {
        Box::new(index.with_store(store))
    }

    fn record_file_path(wal_dir: &std::path::Path, table_name: &str) -> PathBuf {
        wal_dir.join("records").join(format!("{table_name}.rec"))
    }

    /// Where a store lives, from `DXMT_RECORD_STORE` / `DXMT_VECTOR_STORE`: `disk` unless the
    /// variable says `memory`. Anything else is reported and treated as the default.
    fn store_on_disk_wanted(var: &str) -> bool {
        match crate::config::env::var(var).ok().as_deref() {
            None | Some("disk") => true,
            Some("memory") => false,
            Some(other) => {
                warn!("{var}={other} is neither disk nor memory; using disk");
                true
            }
        }
    }

    /// Records live in a row file under the WAL directory (`DXMT_RECORD_STORE=disk`, the
    /// default since docs/architecture/disk-index.md §4, phase 3 was accepted) or in memory
    /// (`DXMT_RECORD_STORE=memory`, and always when there is no WAL directory to put a file
    /// in). Like the vector file: keyed by table name, wiped by a fresh `CREATE TABLE`,
    /// reused by a restore or replay. A reopened table keeps the store it already has: a
    /// row file on disk is that table's data, whatever the environment says today.
    fn open_record_store(
        &self,
        table: &Table,
        fresh: bool,
    ) -> crate::Result<crate::storage::record_store::RecordStore> {
        use crate::storage::record_store::{RecordStore, RowStore};
        let wanted = Self::store_on_disk_wanted("DXMT_RECORD_STORE");
        let Some(wal_dir) = self.wal_dir.as_ref() else {
            return Ok(RecordStore::memory());
        };
        let path = Self::record_file_path(wal_dir, &table.name);
        let mut log = path.as_os_str().to_owned();
        log.push(".log");
        let log = PathBuf::from(log);
        if fresh {
            // Whatever a same-named table left behind is not this table's.
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(&log);
        }
        if !wanted {
            if !path.exists() && !log.exists() {
                return Ok(RecordStore::memory());
            }
            info!(
                "table '{}': DXMT_RECORD_STORE=memory, but its records are on disk at {}; keeping them there",
                table.name,
                path.display()
            );
        }
        let store = RowStore::open(&path)?;
        info!(
            "table '{}': records on disk at {} ({} live)",
            table.name,
            path.display(),
            store.len()
        );
        Ok(RecordStore::Disk(Box::new(store)))
    }

    fn vector_file_path_for(
        wal_dir: &std::path::Path,
        table_name: &str,
        format: crate::storage::vector_file::VectorFormat,
    ) -> PathBuf {
        wal_dir
            .join("vectors")
            .join(format!("{table_name}.{}", format.extension()))
    }

    /// Full vectors live in a row file under the WAL directory (`DXMT_VECTOR_STORE=disk`, the
    /// default since docs/architecture/disk-index.md §4, phase 2 was accepted) or inside the records
    /// (`DXMT_VECTOR_STORE=memory`, and always without a WAL directory). As with the record
    /// store, a reopened table keeps the file it already has.
    fn open_vector_file(&self, table: &Table, fresh: bool) -> crate::Result<OpenedVectorFile> {
        use crate::storage::vector_file::{VectorFile, VectorFormat};
        let wanted = Self::store_on_disk_wanted("DXMT_VECTOR_STORE");
        let Some(wal_dir) = self.wal_dir.as_ref() else {
            return Ok((None, HashMap::new()));
        };
        // Keyed by name, not id: a restore recreates the table with a new id and must find
        // the same file. A fresh `CREATE TABLE` (logged, not replayed or restored) starts
        // from an empty file -- whatever a same-named table left behind is not this table's.
        if fresh {
            for f in [VectorFormat::F32, VectorFormat::F16] {
                let p = Self::vector_file_path_for(wal_dir, &table.name, f);
                let _ = std::fs::remove_file(&p);
                let _ = std::fs::remove_file(VectorFile::rows_path(&p));
            }
        }
        let existing = [VectorFormat::F16, VectorFormat::F32]
            .into_iter()
            .find(|f| Self::vector_file_path_for(wal_dir, &table.name, *f).exists());
        if !wanted {
            let Some(found) = existing else {
                return Ok((None, HashMap::new()));
            };
            info!(
                "table '{}': DXMT_VECTOR_STORE=memory, but its vectors are on disk at {}; keeping them there",
                table.name,
                Self::vector_file_path_for(wal_dir, &table.name, found).display()
            );
        }
        // A fresh table takes the format the environment asks for (`DXMT_VECTOR_F16=1` for
        // half precision); a reopened one keeps whichever file it already has.
        let wanted_format =
            if crate::config::env::var("DXMT_VECTOR_F16").ok().as_deref() == Some("1") {
                VectorFormat::F16
            } else {
                VectorFormat::F32
            };
        let format = existing.unwrap_or(wanted_format);
        let path = Self::vector_file_path_for(wal_dir, &table.name, format);
        let (file, ids) = VectorFile::open_with(&path, table.dimension, format)?;
        // Later rows win: an update appends a new row for the same id.
        let rows: HashMap<RecordId, u64> = ids.into_iter().collect();
        info!(
            "table '{}': full vectors on disk at {} ({} rows)",
            table.name,
            path.display(),
            file.rows()
        );
        Ok((Some(Arc::new(file)), rows))
    }

    /// The table's own memory-optimization setting, else the global default
    /// (`DXMT_MEMORY_OPTIMIZATION`, on unless set to `false`).
    fn memory_optimization_of(&self, table: &Table) -> bool {
        table
            .memory_optimization
            .unwrap_or(self.default_memory_optimization)
    }

    /// `DXMT_INDEX_ALGORITHM=hnsw|hnsw-pq` pins the index family for every table, at creation
    /// and at re-evaluation alike, so the periodic re-check does not rebuild it back.
    ///
    /// For measuring one index against another on the same data without touching the
    /// selector's size thresholds -- the HNSW-PQ acceptance runs in DISK_INDEX.md section 4
    /// are made this way. Unset, the selector decides as before.
    fn forced_algorithm(dimension: usize) -> Option<IndexAlgorithm> {
        match crate::config::env::var("DXMT_INDEX_ALGORITHM")
            .ok()?
            .as_str()
        {
            "hnsw" => Some(IndexAlgorithm::HNSW {
                m: 16,
                ef_construction: Self::default_ef_construction(),
                ef_search: Self::DEFAULT_EF_SEARCH,
            }),
            "hnsw-pq" => Some(IndexAlgorithm::HnswPQ {
                m: 16,
                ef_construction: Self::default_ef_construction(),
                ef_search: Self::DEFAULT_EF_SEARCH,
                num_subvectors: crate::search::hnsw_pq::HnswPqConfig::default_subvectors(dimension),
                num_centroids: 256,
                enable_reranking: true,
                quantize_at_rows: None,
                fast_scan: true,
            }),
            other => {
                warn!("DXMT_INDEX_ALGORITHM={other} is not hnsw or hnsw-pq; ignoring it");
                None
            }
        }
    }

    /// 检查每张表当前的索引算法是否仍与其规模相称，不相称的重建。
    ///
    /// 这是"自适应索引"真正成立的地方。此前算法只在建表时选一次，用的还是写死的
    /// `estimated_size = 10_000`——一张长到百万行的表，索引仍然是按一万行挑的那个，
    /// 除非有人手工调 `/index/rebuild`。
    ///
    /// 只在**跨越边界**时才重建：拿当前记录数重跑一遍选择器，选出的算法与在用的不同才动手。
    /// 记录数每变一次就重建是不可行的——重建代价是 O(n)。
    pub async fn reevaluate_indexes(&self) -> crate::Result<Vec<String>> {
        let candidates: Vec<(String, usize, IndexAlgorithm)> = {
            let tables = self.read_tables("reevaluate_indexes").await;
            tables
                .values()
                .map(|d| {
                    (
                        d.table.name.clone(),
                        d.records.len(),
                        d.selected_algorithm.clone(),
                    )
                })
                .collect()
        };

        let mut rebuilt = Vec::new();
        let mut maintained: Vec<String> = Vec::new();
        let mut retuned: Vec<String> = Vec::new();

        for (name, count, current) in candidates {
            // An empty table has nothing to cluster and nothing to gain.
            if count == 0 {
                continue;
            }

            let (dimension, metric, memory_optimization, hnsw, quantization) = {
                let tables = self.read_tables("reevaluate_indexes").await;
                match tables.get(&name) {
                    Some(d) => (
                        d.table.dimension,
                        d.table.distance_metric,
                        self.memory_optimization_of(&d.table),
                        d.table.hnsw.clone(),
                        d.table.quantization.clone(),
                    ),
                    // Evicted or dropped while we were iterating; nothing to do.
                    None => continue,
                }
            };

            let want = self
                .decide_algorithm(
                    count,
                    dimension,
                    metric,
                    memory_optimization,
                    hnsw.as_ref(),
                    quantization.as_ref(),
                )
                .await?
                .selected_algorithm;

            // A rebuild is O(n) whichever way it goes, so the sweep only ever climbs. A
            // table that shrank keeps the index it has -- an oversized graph on a small table
            // costs memory nobody notices, while flapping across a threshold costs a full
            // rebuild each way, twice per crossing. `POST /tables/{t}/index/rebuild` still
            // re-selects from scratch for an operator who wants the smaller index back.
            let downgrade = current.is_downgrade_to(&want);
            if downgrade {
                debug!(
                    "Table '{}' at {} records would select {:?}, which is below its current \
                     {:?}; the periodic sweep does not demote",
                    name, count, want, current
                );
            }
            if downgrade || !current.requires_rebuild_to_reach(&want) {
                // The algorithm still fits, but its per-query parameters may have moved.
                // `ef_search` is read on every search and deliberately does not force a
                // rebuild — and until this ran, nothing wrote a new one back either, so a
                // changed default never reached a table that already existed. This is the
                // cheap half of what a rebuild would have done, and the next snapshot
                // persists it along with the rest of the algorithm.
                {
                    let mut tables = self.write_tables("reevaluate_indexes").await;
                    if let Some(data) = tables.get_mut(&name) {
                        if data.selected_algorithm.adopt_query_time_params(&want) {
                            retuned.push(name.clone());
                        }
                    }
                }

                // A partitioned index also has upkeep to do: writes
                // since the last pass are sitting in its overflow buffer, and partitions drift
                // out of their size range as data lands. Doing nothing here would leave the
                // buffer growing until something forced a full rebuild.
                let wrapper = {
                    let tables = self.read_tables("reevaluate_indexes").await;
                    tables.get(&name).map(|d| d.index.clone())
                };
                if let Some(wrapper) = wrapper {
                    match wrapper.maintain().await {
                        Ok(true) => {
                            maintained.push(name.clone());
                        }
                        Ok(false) => {}
                        Err(e) => warn!("Maintaining '{}' failed: {}", name, e),
                    }

                    // The algorithm fits; the tombstones may still not. An update cannot move
                    // a node -- where a node sits is decided by the vector it had -- so it
                    // deletes the old slot and appends a new one, and nothing gives the slot
                    // back. A table of constant size under steady upserts therefore grows one
                    // slot per write, for ever. Measured in a 17-hour soak at 200 rows/s over
                    // a 400,000-row key space: 11.1 million slots for 400,000 records, a
                    // 5.3 GB index where the live rows need about 200 MB, resident memory
                    // climbing 8.4 GB a day, snapshots from 1.2 s to 18.9 s, and 196 s to
                    // recover from `kill -9`.
                    //
                    // A rebuild is the reclamation: it builds a new graph from the live
                    // records and drops the tombstoned slots with the old one. It is also
                    // O(n), which is why it waits for the garbage to be worth it rather than
                    // running on a timer -- at the threshold below, a table rebuilds once per
                    // as many writes as it has rows, and holds at most twice the slots it
                    // needs.
                    let (live, slots) = wrapper.live_and_slots().await;
                    if worth_compacting(live, slots) {
                        info!(
                            "Table '{}': {} live entries in {} slots, rebuilding to reclaim the \
                             tombstones",
                            name, live, slots
                        );
                        match self.rebuild_index(&name).await {
                            Ok(()) => rebuilt.push(name.clone()),
                            Err(e) => warn!(
                                "Compacting '{}' failed, keeping the current index: {}",
                                name, e
                            ),
                        }
                    }
                }
                continue;
            }

            // Name both algorithms. This printed `std::mem::discriminant(&current)`, which
            // renders as `Discriminant(0)` — the reader has to know the variant order of
            // `IndexAlgorithm` to decode which index the table is actually leaving.
            info!(
                "Table '{}' at {} records: {:?} no longer suits it, rebuilding as {:?}",
                name, count, current, want
            );
            match self.rebuild_index(&name).await {
                Ok(()) => rebuilt.push(name),
                // A failed rebuild leaves the old index in place — that is the whole point of
                // building the replacement before swapping — so this costs efficiency, not
                // correctness.
                Err(e) => warn!(
                    "Re-evaluating '{}' failed, keeping the current index: {}",
                    name, e
                ),
            }
        }

        if !maintained.is_empty() {
            info!(
                "Maintained {} partitioned index(es): {:?}",
                maintained.len(),
                maintained
            );
        }
        if !retuned.is_empty() {
            info!(
                "Adopted new query-time parameters for {} table(s) without a rebuild: {:?}",
                retuned.len(),
                retuned
            );
        }

        // Reachability repair and recall calibration ride the same sweep: both want the
        // graph settled and both are bounded work. Repair first, so the calibration measures
        // the graph a query will see. A failure on one table is logged and does not stop the
        // others.
        let names: Vec<String> = {
            let tables = self.read_tables("reevaluate_indexes").await;
            tables.keys().cloned().collect()
        };
        for name in &names {
            match self.repair_reachability(name).await {
                Ok((0, _)) => {}
                Ok((stranded, repaired)) => info!(
                    "Reachability repair on '{}': {} node(s) had no in-edge, {} given one",
                    name, stranded, repaired
                ),
                Err(e) => warn!("Reachability repair on '{}' skipped: {}", name, e),
            }
        }
        for name in names {
            match self.calibrate_recall(&name, false).await {
                Ok(Some(cal)) => info!(
                    "Calibrated '{}': ef_search {} reaches recall@10 {:.4} (target {:.2}, {} queries, {} rows{})",
                    name,
                    cal.ef_search,
                    cal.measured,
                    cal.target,
                    cal.sample,
                    cal.at_rows,
                    if cal.met { "" } else { "; target NOT met at the largest budget tried" }
                ),
                Ok(None) => {}
                Err(e) => warn!("Recall calibration of '{}' skipped: {}", name, e),
            }
        }
        Ok(rebuilt)
    }

    /// One reachability repair pass over a table's graph; `(stranded, repaired)`. The sweep
    /// runs it every interval; `POST /tables/{t}/index/repair` runs it now.
    pub async fn repair_reachability(&self, tablename: &str) -> crate::Result<(usize, usize)> {
        let tablename = &*self.resolve(tablename);
        let (index, stranded_last, repaired_total) = {
            let tables = self.read_tables("repair_reachability").await;
            let data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
            (
                data.index.clone(),
                Arc::clone(&data.stranded_last),
                Arc::clone(&data.repaired_total),
            )
        };
        let (stranded, repaired) = index.repair_reachability().await;
        stranded_last.store(stranded, Ordering::Relaxed);
        repaired_total.fetch_add(repaired as u64, Ordering::Relaxed);
        Ok((stranded, repaired))
    }

    /// Server-wide recall target for tables that do not name one. `0` disables calibration.
    fn default_recall_target() -> f64 {
        crate::config::env::var("DXMT_RECALL_TARGET")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(Self::DEFAULT_RECALL_TARGET)
    }
    const DEFAULT_RECALL_TARGET: f64 = 0.95;
    /// Below this many rows the graph is exact or near it at any budget, and a calibration
    /// would measure nothing but noise. `force` ignores it, for the endpoint and the tests.
    const CALIBRATION_MIN_ROWS: usize = 20_000;
    /// Queries per calibration, each made from a pair of sampled rows. Fifty put the
    /// measurement within a few points of the truth, which is the resolution the doubling
    /// series below can act on anyway.
    const CALIBRATION_SAMPLE: usize = 50;
    const CALIBRATION_K: usize = 10;
    /// The largest multiple of the configured budget the calibration will try.
    const CALIBRATION_MAX_SCALE: usize = 8;

    /// Measure recall@10 on this table's own rows at a doubling series of budgets and keep
    /// the smallest that meets the target. See [`RecallCalibration`] for why.
    ///
    /// Returns `Ok(None)` when there is nothing to do: calibration is off for the table, the
    /// table is too small, the index has no budget to tune, or it was calibrated at a size
    /// the table has not yet doubled. `force` skips the size and staleness checks.
    pub async fn calibrate_recall(
        &self,
        tablename: &str,
        force: bool,
    ) -> crate::Result<Option<crate::types::table::RecallCalibration>> {
        use crate::types::table::{CalibrationPoint, RecallCalibration};
        let tablename = &*self.resolve(tablename);

        // Phase 1, under the map lock: decide, and draw the sample.
        let (target, configured_ef, rows, sample_ids) = {
            let tables = self.read_tables("calibrate_recall").await;
            let data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
            let target = data
                .table
                .recall_target
                .unwrap_or_else(Self::default_recall_target);
            // `<= 0`, and NaN, mean off; above 1 is a request nothing can meet.
            if target.is_nan() || target <= 0.0 {
                return Ok(None);
            }
            if target > 1.0 {
                return Err(crate::Error::InvalidInput(format!(
                    "recall_target {target} is not in (0, 1]"
                )));
            }
            let Some(configured_ef) = data.selected_algorithm.configured_ef_search() else {
                return Ok(None); // brute force: exact already, nothing to tune
            };
            let rows = data.records.len();
            if !force {
                if rows < Self::CALIBRATION_MIN_ROWS {
                    return Ok(None);
                }
                if let Some(prev) = &data.table.recall {
                    if prev.target == target && (rows as u64) < prev.at_rows.saturating_mul(2) {
                        return Ok(None);
                    }
                }
            }
            if rows < Self::CALIBRATION_K + 2 {
                return Ok(None);
            }
            let mut rng =
                <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(rows as u64 ^ 0x5eed);
            let ids = data
                .records
                .sample_ids(Self::CALIBRATION_SAMPLE * 2, &mut rng);
            (target, configured_ef, rows, ids)
        };
        let vectors = self.vectors_of(tablename, &sample_ids).await?;
        if vectors.len() < 2 {
            return Ok(None);
        }

        // The queries. A stored row as a query flatters the graph: it is a node, its true
        // neighbours are its own edges, and on a corpus where held-out queries get 0.21 at a
        // small budget the rows themselves got 0.90 (uniform 128-d noise, 20K rows). A point
        // a third of the way from one stored row to another is off the graph like a real
        // query and tracks held-out recall within one or two points on glove-100, erring on
        // the low side on gist and lastfm (hnswlib, 2026-10-03), so a budget chosen on it is
        // enough or slightly more than enough -- the right direction to be wrong in.
        let queries: Vec<Vec<f32>> = vectors
            .chunks(2)
            .filter(|pair| pair.len() == 2)
            .map(|pair| {
                let (a, b) = (&pair[0].1, &pair[1].1);
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| x + 0.35 * (y - x))
                    .collect()
            })
            .collect();

        // Ground truth: the exact scan, through the same entry point a caller would use.
        let k = Self::CALIBRATION_K;
        let mut truth: Vec<(Vec<f32>, Vec<RecordId>)> = Vec::with_capacity(queries.len());
        for query in queries {
            let exact = self
                .search_with_budget(
                    tablename,
                    SearchQuery {
                        vector: query.clone(),
                        top_k: k,
                        limit: k,
                        search_mode: SearchMode::Exact,
                        with_payload: false,
                        ..Default::default()
                    },
                    None,
                )
                .await?;
            let ids: Vec<RecordId> = exact.hits.into_iter().map(|h| h.id).collect();
            if ids.len() == k {
                truth.push((query, ids));
            }
        }
        if truth.is_empty() {
            return Ok(None);
        }

        // The graph, at the configured budget and doublings of it.
        let mut curve: Vec<CalibrationPoint> = Vec::new();
        let mut chosen: Option<(usize, f32)> = None;
        let mut scale = 1usize;
        while scale <= Self::CALIBRATION_MAX_SCALE {
            let ef = configured_ef * scale;
            let mut hits_sum = 0usize;
            let mut lat: Vec<f32> = Vec::with_capacity(truth.len());
            for (query, ids) in &truth {
                let started = std::time::Instant::now();
                let got = self
                    .search_with_budget(
                        tablename,
                        SearchQuery {
                            vector: query.clone(),
                            top_k: k,
                            limit: k,
                            search_mode: SearchMode::Normal,
                            with_payload: false,
                            ..Default::default()
                        },
                        Some(ef),
                    )
                    .await?;
                lat.push(started.elapsed().as_secs_f32() * 1000.0);
                let found: std::collections::HashSet<&str> =
                    got.hits.iter().map(|h| h.id.as_str()).collect();
                hits_sum += ids.iter().filter(|t| found.contains(t.as_str())).count();
            }
            let recall = hits_sum as f32 / (truth.len() * k) as f32;
            lat.sort_by(|a, b| a.total_cmp(b));
            let p50_ms = lat[lat.len() / 2];
            curve.push(CalibrationPoint {
                ef_search: ef,
                recall,
                p50_ms,
            });
            if f64::from(recall) >= target {
                chosen = Some((ef, recall));
                break;
            }
            scale *= 2;
        }
        let met = chosen.is_some();
        let (ef_search, measured) =
            chosen.unwrap_or_else(|| curve.last().map(|p| (p.ef_search, p.recall)).unwrap());
        let cal = RecallCalibration {
            target,
            ef_search,
            measured,
            met,
            sample: truth.len(),
            at_rows: rows as u64,
            at: chrono::Utc::now(),
            curve,
        };

        // Phase 2: record it. The table may have been dropped meanwhile; that is not an error.
        let mut tables = self.write_tables("calibrate_recall").await;
        if let Some(data) = tables.get_mut(tablename) {
            data.table.recall = Some(cal.clone());
            data.table.updated_at = chrono::Utc::now();
        }
        Ok(Some(cal))
    }

    /// The stored vectors of `ids`, in no particular order; ids that are gone are left out.
    async fn vectors_of(
        &self,
        tablename: &str,
        ids: &[RecordId],
    ) -> crate::Result<Vec<(RecordId, Vec<f32>)>> {
        let (in_memory, on_disk, file) = {
            let tables = self.read_tables("vectors_of").await;
            let data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
            let mut in_memory: Vec<(RecordId, Vec<f32>)> = Vec::new();
            let mut on_disk: Vec<(RecordId, u64)> = Vec::new();
            for id in ids {
                if let Some(row) = data.vector_row(id) {
                    on_disk.push((id.clone(), row));
                } else if let Some(r) = data.records.get(id) {
                    if !r.vector.is_empty() {
                        in_memory.push((id.clone(), r.vector.clone()));
                    }
                }
            }
            (in_memory, on_disk, data.vector_file.clone())
        };
        let mut out = in_memory;
        if let (Some(file), false) = (file, on_disk.is_empty()) {
            let rows: Vec<u64> = on_disk.iter().map(|(_, r)| *r).collect();
            for ((id, _), vector) in on_disk.into_iter().zip(file.read_rows(&rows)) {
                if let Some(vector) = vector {
                    out.push((id, vector));
                }
            }
        }
        Ok(out)
    }

    /// 把冷表从内存里放掉。返回被驱逐的表名。
    ///
    /// **先快照再驱逐**，而且只驱逐这次快照确实覆盖到的表。反过来会丢数据：一张表若有
    /// 快照之后的写入，驱逐就等于把那些写入扔了——WAL 里虽然还有，但驱逐后的表在内存里
    /// 不存在，回填只会读到快照那一份。
    /// Give back the disk behind row-file rows nothing points at any more.
    ///
    /// Both row files are append-only: an update appends a new row and repoints the id, and
    /// the row it left keeps its blocks. Nothing reclaimed them. The sweep that rebuilds an
    /// index -- the one thing `vector_file.rs` claimed would rewrite the file -- only fires
    /// when a table's size calls for a different algorithm, so a table of constant size under
    /// steady upserts never triggered it and grew forever. Measured at 200 rows/s over a
    /// 400,000-row key space: 5.2 GB of vector file holding 1.2 GB of live rows after 2.4
    /// hours, climbing at about 52 GB a day regardless of the table's size.
    ///
    /// This runs on its own timer (`DXMT_ROW_COMPACTION_INTERVAL_SECS`) rather than inside the
    /// eviction or re-index sweep: it is cheap, it wants to run more often than a rebuild, and
    /// folding it into a sweep that also snapshots would tie giving back disk to a decision
    /// about memory.
    ///
    /// Reported bytes are measured, not derived: allocated blocks before and after. Punching a
    /// range that is already a hole succeeds and releases nothing, and deriving the number
    /// from the dead rows would report the same bytes again on every sweep.
    pub async fn reclaim_dead_rows(&self) -> crate::Result<Vec<(String, u64)>> {
        // Not while a snapshot runs: a mirrored snapshot reads, after the lock, rows that were
        // live when it listed them and may have died since (`read_mirrored_rows`).
        let _serialized = self.snapshotting.lock().await;
        let names: Vec<String> = {
            let tables = self.read_tables("reclaim_dead_rows").await;
            tables.keys().cloned().collect()
        };
        let floor = min_reclaim_bytes();
        let mut reclaimed = Vec::new();
        for name in names {
            // One table at a time, under a read lock: punching needs no exclusive access --
            // a dead row stays dead, because rows are never reused -- and holding the map
            // across all tables would park every writer for the length of the sweep.
            let tables = self.read_tables("reclaim_dead_rows").await;
            let Some(data) = tables.get(&name) else {
                continue; // dropped or evicted between listing and here
            };

            let mut released = 0u64;

            if let Some(file) = data.vector_file.as_ref() {
                // Decide in constant time, then walk. Building the live set and scanning for
                // runs is O(rows), and it runs under the table map's read lock: on a
                // hundred-million-row table that is a stall on a timer, paid whether or not
                // there was anything to reclaim. Rows minus live records is an upper bound on
                // the live rows, so the estimate errs towards not sweeping.
                let live_upper = data.records.len() as u64 + data.vector_rows.len() as u64;
                let dead_estimate = file.rows().saturating_sub(live_upper);
                if dead_estimate.saturating_mul(file.row_bytes() as u64) >= floor {
                    // A row is live if any live record resolves to it. The two mappings are
                    // consulted in the same order `vector_row` consults them, and unioned
                    // rather than picked between: counting a live row as dead is the one
                    // mistake that loses data, and a row wrongly kept costs only space.
                    let mut live = data.records.live_vector_rows().unwrap_or_default();
                    live.extend(data.vector_rows.values().copied());
                    // The `(id, row)` sidecar beside the data file takes a line per append,
                    // so an upsert-heavy table writes the same ids over and over: 214 MB for
                    // 400,000 ids after seventeen hours at 200 rows/s. Every line in it is
                    // valid, merely superseded, so there is nothing to punch -- it has to be
                    // written again. Folded in here because the condition is the same one:
                    // many more rows than live ids.
                    if file.rows_sidecar_bytes() >= MIN_SIDECAR_BYTES {
                        match file.compact_rows() {
                            Ok((was, now)) => released += was.saturating_sub(now),
                            Err(e) => {
                                warn!("table '{}': row-list compaction failed: {}", name, e)
                            }
                        }
                    }

                    let before = crate::storage::punch::allocated_bytes(file.path());
                    // Logged rather than propagated: one table whose filesystem will not punch
                    // is no reason to stop sweeping the others, and nothing is at risk either
                    // way -- the rows were already dead.
                    match file.punch_dead(&live) {
                        Ok(_) => {
                            released += before
                                .saturating_sub(crate::storage::punch::allocated_bytes(file.path()))
                        }
                        Err(e) => warn!("table '{}': vector-row reclamation failed: {}", name, e),
                    }
                }
            }

            // Same shape for the records, with a floor on the smallest record worth counting:
            // `dead_rows` is O(1), the punch is not.
            const MIN_RECORD_BYTES: u64 = 64;
            if data.records.dead_rows().saturating_mul(MIN_RECORD_BYTES) >= floor {
                let before = data.records.disk_bytes();
                match data.records.punch_dead() {
                    Ok(_) => released += before.saturating_sub(data.records.disk_bytes()),
                    Err(e) => warn!("table '{}': record-row reclamation failed: {}", name, e),
                }
            }

            // And the record log, for the same reason as the vector file's row list: a line
            // per append, all of them valid, most of them superseded.
            if data.records.log_bytes() >= MIN_SIDECAR_BYTES {
                match data.records.compact_log() {
                    Ok((was, now)) => released += was.saturating_sub(now),
                    Err(e) => warn!("table '{}': record-log compaction failed: {}", name, e),
                }
            }

            drop(tables);
            if released > 0 {
                info!(
                    "table '{}': reclaimed {:.1} MB of dead row-file space",
                    name,
                    released as f64 / 1e6
                );
                reclaimed.push((name, released));
            }
        }
        Ok(reclaimed)
    }

    pub async fn evict_cold_tables(&self) -> crate::Result<Vec<String>> {
        use crate::storage::tier_promotion::Tier;

        if self.wal.is_none() {
            return Ok(Vec::new());
        }

        // Decide before snapshotting. This used to snapshot every table on every sweep and
        // only then ask which were cold: on a 1M-row table that is a 38 s serialisation under
        // the index read lock every 600 s, whether or not anything was going to be evicted,
        // and a load it landed in ran at half speed (design doc section 24.4). A table is only
        // dropped from memory once it is on disk, so the snapshot stays -- for the sweeps that
        // will drop something.
        //
        // `should_demote_to_cold` is the right predicate here, not `get_recommended_tier`: the
        // latter calls anything below the *warm* threshold Cold, and a table written to seconds
        // ago scores about 0.31 -- the frequency term is normalised against 100 accesses, so a
        // handful of touches barely registers and recency contributes at most 0.3. Demotion
        // instead compares against demote_to_cold_threshold (0.2), which a Warm table only
        // falls below after roughly a third of the window with nothing touching it.
        let cold: Vec<String> = {
            let tables = self.read_tables("evict_cold_tables").await;
            tables
                .keys()
                .filter(|name| self.tier_policy.should_demote_to_cold(name))
                .cloned()
                .collect()
        };
        if cold.is_empty() {
            return Ok(Vec::new());
        }
        self.snapshot().await?;

        let mut evicted = Vec::new();
        for name in cold {
            let mut tables = self.write_tables("evict_cold_tables").await;
            if tables.remove(&name).is_some() {
                evicted.push(name.clone());
            }
            drop(tables);
            self.tier_policy.set_current_tier(&name, Tier::Cold);
        }

        if !evicted.is_empty() {
            info!(
                "Evicted {} cold table(s) from memory: {:?}",
                evicted.len(),
                evicted
            );
        }
        Ok(evicted)
    }

    /// 把刚写好的快照完整读一遍，确认它真的能用来恢复。
    fn verify_snapshot(dir: &std::path::Path, tables: usize) -> crate::Result<()> {
        let manifest: SnapshotManifest = crate::search::index_persistence::read_index(
            &dir.join("manifest"),
            "snapshot-manifest",
        )?;
        if manifest.tables != tables {
            return Err(crate::Error::Internal(format!(
                "snapshot manifest claims {} tables, {} were written",
                manifest.tables, tables
            )));
        }

        for i in 0..tables {
            let stem = format!("t{}", i);
            let meta = verify_snapshot_meta(&dir.join(format!("{}.meta", stem)))?;
            if meta.has_index {
                let idx = dir.join(format!("{}.idx", stem));
                if !idx.exists() {
                    return Err(crate::Error::Internal(format!(
                        "snapshot for table '{}' claims an index but {} is missing",
                        meta.table_name,
                        idx.display()
                    )));
                }
            }
        }
        Ok(())
    }

    /// 加载快照，返回 WAL 应该从哪个序号开始回放。没有快照就返回 0（全量回放）。
    async fn load_snapshot(&self, wal_dir: &std::path::Path) -> crate::Result<u64> {
        let dir = Self::snapshot_dir(wal_dir);
        if !dir.exists() {
            // Local disk is a cache; object storage is where the snapshot lives. A fresh machine
            // — a new container, a replaced node — has an empty cache and must pull it back.
            if self.object_storage_enabled {
                match self.fetch_snapshot(&dir).await {
                    Ok(true) => {}
                    Ok(false) => return Ok(0),
                    Err(e) => {
                        warn!("Could not fetch snapshot from object storage: {}", e);
                        return Ok(0);
                    }
                }
            } else {
                return Ok(0);
            }
        }

        // From here on a failure is fatal rather than a fallback. The WAL is compacted once a
        // snapshot is verified, so the entries this snapshot covers are gone — coming up with
        // an empty database and a warning in the log would be silent data loss. Refuse to
        // start and let the operator restore or delete the snapshot deliberately.
        let manifest: SnapshotManifest = crate::search::index_persistence::read_index(
            &dir.join("manifest"),
            "snapshot-manifest",
        )
        .map_err(|e| {
            crate::Error::Internal(format!(
                "snapshot at {} is unreadable ({}). The WAL no longer covers the entries \
                         it holds, so starting would lose data. Restore it from backup, or delete \
                         the directory to accept the loss and start from whatever the WAL still \
                         has.",
                dir.display(),
                e
            ))
        })?;

        for i in 0..manifest.tables {
            let stem = format!("t{}", i);
            let meta = read_snapshot_meta(&dir.join(format!("{}.meta", stem))).map_err(|e| {
                crate::Error::Internal(format!("snapshot table {} is unreadable: {}", i, e))
            })?;

            let name = meta.table.name.clone();
            if let Ok(mut floors) = self.recovery_floors.lock() {
                floors.insert(name.clone(), meta.indexed_below);
            }
            self.restore_table(&dir.join(format!("{}.idx", stem)), meta)
                .await
                .map_err(|e| {
                    crate::Error::Internal(format!("restoring table '{}' failed: {}", name, e))
                })?;
        }

        info!(
            "Snapshot loaded: {} tables; replaying WAL from sequence {}",
            manifest.tables, manifest.wal_next_sequence
        );
        if let Ok(mut sequence) = self.snapshot_sequence.lock() {
            *sequence = Some(manifest.wal_next_sequence);
        }
        if let Ok(mut names) = self.snapshot_table_names.lock() {
            names.clone_from(&manifest.names);
        }
        Ok(manifest.wal_next_sequence)
    }

    async fn restore_table(
        &self,
        idx_path: &std::path::Path,
        meta: TableSnapshot,
    ) -> crate::Result<()> {
        use crate::search::index_traits::SerializableIndex;

        let name = meta.table.name.clone();
        let dimension = meta.table.dimension;
        let metric = meta.table.distance_metric;

        // Rebuild the container through the normal path so index selection, named vectors and
        // the full-text engine are set up exactly as a live CREATE TABLE would set them up.
        let config = TableConfig {
            name: name.clone(),
            dimension,
            distance_metric: metric,
            description: meta.table.description.clone(),
            index_type: None,
            text_fields: meta.table.text_fields.clone(),
            enable_full_text_search: meta.table.enable_full_text_search,
            memory_optimization: meta.table.memory_optimization,
            hnsw: meta.table.hnsw.clone(),
            quantization: meta.table.quantization.clone(),
            payload_index: meta.table.payload_index.clone(),
            payload_schema: meta.table.payload_schema.clone(),
            schema_version: meta.table.schema_version,
            named_vectors: meta.table.named_vectors.clone(),
            // Was `None`: every other field was copied from the snapshot head and this one was
            // dropped, so **every clean restart** -- `main` snapshots on SIGTERM -- turned a
            // text table into one that refuses text, rows intact. WAL replay was never affected
            // (`CreateTableWithConfig` carries the whole config), which is why the reopen test
            // that skipped the snapshot stayed green. No provider call: this path goes through
            // `create_table_inner`, which never probes the model.
            embedding: meta.table.embedding.clone(),
            rerank: meta.table.rerank.clone(),
        };
        // Not `create_table`: this container is being rebuilt from a snapshot that already
        // describes it, so re-logging the creation would grow the WAL on every restart.
        self.create_table_inner(config, false).await?;

        // A load failure is not the same as "this index has no on-disk form". The snapshot said
        // it wrote one; if it will not read back, the fallback below silently pays the full
        // re-index that the snapshot existed to avoid, and the only symptom is a startup that
        // takes minutes instead of seconds. Say so.
        let loaded = |r: crate::Result<Box<dyn VectorIndex>>| match r {
            Ok(i) => Some(i),
            Err(e) => {
                warn!(
                    "Table '{}': the snapshot's index would not load ({}); rebuilding it from \
                     the rows instead",
                    name, e
                );
                None
            }
        };
        let restored_index = if meta.has_index {
            match &meta.algorithm {
                IndexAlgorithm::HNSW { .. } => loaded(
                    crate::search::hnsw::HnswIndex::load(idx_path)
                        .map(|i| Box::new(i) as Box<dyn VectorIndex>),
                ),
                IndexAlgorithm::HnswPQ { .. } => loaded(
                    crate::search::hnsw_pq::HnswPqIndex::load(idx_path)
                        .map(|i| Box::new(i) as Box<dyn VectorIndex>),
                ),
                IndexAlgorithm::PartitionedIvf { .. } => {
                    // The saved state holds centroids and partition metadata, not the segments —
                    // those live in object storage — so the store has to be re-attached or every
                    // probe reads an empty partition. Falling through to the catch-all instead
                    // silently downgraded the table to whatever a zero-row table would get,
                    // orphaned its segments, and paid the full re-insert that snapshots exist to
                    // avoid.
                    // Spelled out rather than `.map(|i| ...)`: under the release profile (fat
                    // LTO, one codegen unit) the `vectordb` binary instantiated that closure
                    // itself and then failed to link against the library's private symbols
                    // it used (`with_store`, then `Arc::drop_slow`). A match has nothing for
                    // the binary to instantiate.
                    loaded(
                        match crate::search::partitioned_ivf::PartitionedIvfIndex::load(idx_path) {
                            Ok(i) => Ok(Self::attach_store(i, self.storage.clone())),
                            Err(e) => Err(e),
                        },
                    )
                }
                // BruteForce and the variants that fall back to it have no on-disk form worth
                // loading; re-inserting is as cheap as reading the file would be.
                _ => None,
            }
        } else {
            None
        };

        let mut tables = self.write_tables("restore_table").await;
        let data = tables
            .get_mut(&name)
            .ok_or_else(|| crate::Error::TableNotFound(name.clone()))?;

        let index_restored = restored_index.is_some();
        if let Some(index) = restored_index {
            data.index.replace(index).await;
        } else {
            // No on-disk form, so the rows get re-indexed below — but into *which* index? The
            // one `create_table` just chose from its fixed 10,000-row estimate, which is how a
            // 94,000-row BruteForce table came back as HNSW and spent over ten minutes building
            // a graph it never had, before the server could bind its port. The snapshot recorded
            // the algorithm; use it.
            let fresh = self.create_index_instance_for(
                &meta.algorithm,
                dimension,
                metric,
                &data.table.id.to_string(),
            )?;
            data.index.replace(fresh).await;
        }
        // Either way the snapshot, not the estimate, is the authority on this table's index.
        // Without carrying it over the metadata reports the wrong type and the next
        // re-evaluation sweep sees a mismatch and rebuilds a table that was already correct.
        data.table.index_type = index_type_of(&meta.algorithm);
        data.selected_algorithm = meta.algorithm.clone();

        // When the table was created, and when it was last written to. `create_table_inner`
        // just stamped both with `Utc::now()`, because it builds through `TableConfig` and
        // `TableConfig` carries neither -- the same gap that made a table's *id* change on
        // every restart. So `created_at` moved forward every time the server came back, and
        // `GET /tables/{t}` reported a table created seconds ago holding a year of records.
        data.table.created_at = meta.table.created_at;
        data.table.updated_at = meta.table.updated_at;

        // A table whose row file already holds rows is restored from that file: it was
        // written at insert time, so it is at least as new as any snapshot. The snapshot's
        // records matter when there is no such file -- a table that was in memory, or a
        // machine recovering from object storage alone, where the snapshot of a disk table
        // carries its rows for exactly this case.
        if !(data.records.is_on_disk() && !data.records.is_empty()) {
            data.records
                .insert_batch(meta.records.iter().map(|r| (Arc::clone(r), None)).collect())?;
        }
        data.table.record_count = data.records.len() as u64;

        // The payload and sparse indexes belong to the records that were just put back, and
        // `create_table_inner` handed this container an empty pair on the assumption that it
        // had none. Clearing them is what marks them for rebuilding; leaving the empty pair in
        // place is what made every filter match nothing after a restart, with no error and no
        // way to tell it from a filter that genuinely selects nothing. See `DerivedIndexes`.
        *data.derived.write() = None;

        // A restored table's text index normally survives on disk under the table's id, and this
        // finds it populated and does nothing. It comes back *empty* in two cases, and both used
        // to be silent for the life of the table:
        //
        //   * the table predates table ids being a function of the name, so its documents sit in
        //     an orphaned directory under an id no longer issued to anything;
        //   * the directory was removed while the records survived — the default location is
        //     under `/tmp`, which plenty of systems clear on reboot.
        //
        // Either way `/search/text` answers 200 with no hits, forever, for a table that reports
        // `enable_full_text_search: true`. The records are right here and carry everything the
        // index needs, so rebuild from them rather than leave that.
        #[cfg(feature = "text-bm25")]
        let text_backfill: Vec<FullTextDocument> = match &data.full_text_engine {
            Some(engine) if engine.document_count() == 0 && !data.records.is_empty() => {
                let table_meta = data.table.clone();
                let mut docs = Vec::with_capacity(data.records.len());
                data.records.for_each(|record| {
                    if let Some(doc) = text_document_for(&table_meta, &record) {
                        docs.push(doc);
                    }
                });
                docs
            }
            _ => Vec::new(),
        };
        #[cfg(feature = "text-bm25")]
        let text_engine = data.full_text_engine.clone();

        let to_index: Vec<(RecordId, Vec<f32>)> = if index_restored {
            Vec::new()
        } else {
            data.vectors_for_index()
        };
        // The rows can be newer than the index. A write reaches the row file, fsynced,
        // before its WAL entry is appended; a crash in that window -- or a backup whose
        // `records/` copy was taken after a write and whose `wal.log` copy before it -- leaves
        // the record in the store with nothing to put it in the graph: `GET` returns it, the
        // exact scan finds it, and no ordinary search ever will. The counts are a cheap gate;
        // when they disagree, every id the index does not hold is inserted below. A record
        // that is in the index and not in the rows (a delete that reached the rows and not the
        // index) can hide behind matching counts, and is left to the next rebuild.
        let catch_up: Vec<(RecordId, Vec<f32>)> =
            if index_restored && data.index.len().await != data.records.len() {
                let ids = data.records.ids();
                match data.index.missing_ids(ids.iter()).await {
                    Some(missing) if !missing.is_empty() => {
                        warn!(
                            "Table '{}': {} record(s) in the row files were not in the restored \
                         index; indexing them now",
                            name,
                            missing.len()
                        );
                        data.vectors_for_ids(&missing)
                    }
                    _ => Vec::new(),
                }
            } else {
                Vec::new()
            };
        let wrapper = data.index.clone();

        // The named vector spaces, which nothing restored. The snapshot carries the default
        // index and not these, and `create_table_inner` builds the collection empty — so a
        // table with named spaces came back answering `vector_name` searches with nothing at
        // all, while its ordinary search worked. Collected here, under the lock that already
        // holds the records, and fed to the indexes below.
        let named_to_index: NamedVectors = match &data.named_indexes {
            None => Vec::new(),
            Some(_) => {
                let mut by_space: HashMap<String, Vec<(RecordId, Vec<f32>)>> = HashMap::new();
                data.records.for_each(|record| {
                    if let Some(spaces) = &record.named_vectors {
                        for (space, vector) in spaces {
                            by_space
                                .entry(space.clone())
                                .or_default()
                                .push((record.id.clone(), vector.clone()));
                        }
                    }
                });
                by_space.into_iter().collect()
            }
        };
        let named_collection = data.named_indexes.clone();
        drop(tables);

        // Only reached when the index had no on-disk form, or would not load. Outside the write
        // lock, so a large table does not block every reader for the duration — and through
        // bulk_build rather than a loop of inserts, because a partitioned index has to see every
        // vector at once to cluster it; fed one at a time the lot ends up in its overflow buffer.
        if !to_index.is_empty() {
            wrapper.bulk_build(to_index).await?;
        }
        if !catch_up.is_empty() {
            wrapper.insert_batch(catch_up).await?;
        }

        if let Some(collection) = named_collection {
            for (space, vectors) in named_to_index {
                if vectors.is_empty() {
                    continue;
                }
                let Some(index) = collection.get_index(&space).await else {
                    warn!(
                        "Table '{}': the records name a vector space '{}' the table does not \
                         declare; those vectors are not searchable",
                        name, space
                    );
                    continue;
                };
                let count = vectors.len();
                if let Err(e) = index.bulk_build(vectors).await {
                    warn!(
                        "Table '{}': named vector space '{}' would not rebuild ({}); searches \
                         naming it will answer nothing",
                        name, space, e
                    );
                } else {
                    debug!(
                        "Table '{}': rebuilt named vector space '{}' from {} records",
                        name, space, count
                    );
                }
            }
        }

        // Outside the lock for the same reason, and after the vector index: a table with a lost
        // text index is still fully usable for vector search while this runs.
        #[cfg(feature = "text-bm25")]
        if !text_backfill.is_empty() {
            if let Some(engine) = text_engine {
                let count = text_backfill.len();
                // `warn`, not `info`: an index that lost its contents is not a normal start,
                // and a server running at the default `RUST_LOG=warn` would otherwise rebuild
                // 20,000 documents without leaving a line to say it had. Silence about this
                // class of repair is what the whole bug was made of.
                warn!(
                    "Table '{}': the full-text index holds no documents for {} records; \
                     rebuilding it from the rows",
                    name, count
                );
                for doc in text_backfill {
                    let id = doc.id.clone();
                    if let Err(e) = engine.add_document(doc).await {
                        warn!("Failed to re-index text for record {}: {}", id, e);
                    }
                }
                if let Err(e) = engine.commit().await {
                    warn!(
                        "Table '{}': full-text rebuild would not commit: {}",
                        name, e
                    );
                } else {
                    info!("Table '{}': re-indexed {} documents", name, count);
                }
            }
        }

        Ok(())
    }

    /// 写一条 WAL 条目并落盘。回放期间是空操作。
    ///
    /// 每次调用都 `force_flush`：`write_entry` 默认攒够 256KB 才写文件，攒着的部分
    /// `kill -9` 就没了，而"已经返回成功的写入不能丢"正是 WAL 存在的理由。
    /// Whether a logged write may point at its row in the vector file instead of carrying
    /// the vector (`WalEntry::UpsertRecordInRowFile`). Not when the log is mirrored to
    /// object storage: a node recovering from the bucket alone has no row files, and the
    /// entry is then the only copy of the vector off the machine.
    fn log_points_at_rows(&self) -> bool {
        !self.object_storage_enabled
    }

    async fn wal_append(&self, entries: Vec<WalEntry>) -> crate::Result<()> {
        if self.replaying.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.ensure_writable()?;
        self.ensure_still_primary().await?;
        let Some(ref wal) = self.wal else {
            return Ok(());
        };

        let stamped: Vec<(u64, WalEntry)> = match (|| -> crate::Result<Vec<(u64, WalEntry)>> {
            let mut guard = wal
                .lock()
                .map_err(|_| crate::Error::Internal("WAL lock poisoned".to_string()))?;
            let mut stamped = Vec::with_capacity(entries.len());
            for entry in entries {
                let seq = guard.write_entry(entry.clone())?;
                stamped.push((seq, entry));
            }
            guard.force_flush()?;
            if let Ok(mut at) = self.last_wal_sync_at.lock() {
                *at = Some(chrono::Utc::now().to_rfc3339());
            }
            Ok(stamped)
        })() {
            Ok(stamped) => stamped,
            Err(error) => {
                self.mark_durability_failed(&error);
                return Err(crate::Error::ServiceUnavailable {
                    message: format!(
                        "local WAL write failed; storage is read-only until restart: {}",
                        error
                    ),
                });
            }
        };

        // The local log alone leaves a hole in "object storage is the source of truth": every
        // write acknowledged between two snapshots lives only on this machine, so losing the
        // machine loses it. Mirror the entries to object storage as an immutable chunk — the
        // same model the snapshot segments use, so a chunk is written once and never rewritten.
        if self.object_storage_enabled {
            self.mirror_wal(&stamped).await?;
        }
        Ok(())
    }

    /// WAL 清单的键。
    fn wal_manifest_key() -> &'static str {
        "wal/MANIFEST"
    }

    /// 一段 WAL 分块的键。
    ///
    /// 按 4096 一组分前缀：S3 是**按前缀限流**的，所有分块挤在一个前缀下，写入速率到一定
    /// 程度就会被限。定宽序号让字典序等于序号序。
    ///
    /// The writer's epoch is in the name too: a replaced primary that has not yet noticed can
    /// still upload a chunk under a sequence the new one also uses, and a late upload would
    /// otherwise replace the new primary's chunk under the manifest that lists it.
    fn wal_chunk_key(first_seq: u64, epoch: u64) -> String {
        if epoch == 0 {
            format!("wal/{:08}/{:020}.chunk", first_seq / 4096, first_seq)
        } else {
            format!(
                "wal/{:08}/{:020}.e{}.chunk",
                first_seq / 4096,
                first_seq,
                epoch
            )
        }
    }

    /// 读回内存里那份清单。
    fn manifest(&self) -> crate::Result<WalManifest> {
        self.wal_manifest
            .lock()
            .map(|m| m.clone())
            .map_err(|_| crate::Error::Internal("WAL manifest lock poisoned".to_string()))
    }

    /// 把清单写到对象存储。
    /// Write `manifest` as the bucket's, conditionally when the store allows: `Ok(false)` if
    /// another process has written it since this one last did -- another primary took over.
    async fn write_manifest_once(&self, manifest: &WalManifest) -> crate::Result<bool> {
        use crate::storage::object::errors::ObjectStorageError;
        use crate::storage::object::traits::{ObjectStorage, PutCondition};

        let bytes = bytes::Bytes::from(
            serde_json::to_vec(manifest).map_err(|e| crate::Error::Serialization(e.to_string()))?,
        );
        let conditional = self.conditional_writes.load(Ordering::SeqCst);
        // The manifest is one object rewritten on every mirrored write, and object stores limit
        // how often one object may be rewritten -- GCS to about once a second, answering 429
        // beyond it. A failure here is retried by the next write anyway; retrying a few times
        // with backoff first keeps a burst of writes from turning into a run of warnings and a
        // replica that sees nothing new until the burst ends. The write is a whole overwrite,
        // so repeating it is safe.
        let mut delay = std::time::Duration::from_millis(200);
        let mut attempt = 0;
        loop {
            let written = if conditional {
                let condition = match self.manifest_version.lock().ok().and_then(|v| v.clone()) {
                    Some(version) => PutCondition::IfMatch(version),
                    None => PutCondition::IfAbsent,
                };
                self.storage
                    .put_object_if(
                        Self::wal_manifest_key(),
                        bytes.clone(),
                        Some("application/json"),
                        condition,
                    )
                    .await
                    .map(Some)
            } else {
                self.storage
                    .put_object(
                        Self::wal_manifest_key(),
                        bytes.clone(),
                        Some("application/json"),
                    )
                    .await
                    .map(|()| None)
            };
            match written {
                Ok(version) => {
                    if let (Some(version), Ok(mut slot)) = (version, self.manifest_version.lock()) {
                        *slot = Some(version);
                    }
                    return Ok(true);
                }
                Err(ObjectStorageError::PreconditionFailed { .. }) => {
                    // An earlier attempt of this same write may have landed with only its
                    // answer lost; then the object is ours, byte for byte.
                    if attempt > 0 {
                        if let Ok((current, version)) = self
                            .storage
                            .get_object_versioned(Self::wal_manifest_key())
                            .await
                        {
                            if current == bytes.as_ref() {
                                if let Ok(mut slot) = self.manifest_version.lock() {
                                    *slot = version;
                                }
                                return Ok(true);
                            }
                        }
                    }
                    return Ok(false);
                }
                Err(e) if attempt < 4 => {
                    debug!("writing WAL manifest failed, retrying in {delay:?}: {e}");
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                    attempt += 1;
                }
                Err(e) => {
                    return Err(crate::Error::Internal(format!(
                        "writing WAL manifest: {}",
                        e
                    )))
                }
            }
        }
    }

    /// Change the bucket's manifest: apply `change` to a copy, write it, and only then make it
    /// this process's. All changes go through here, one at a time, so a failed write leaves
    /// nothing behind in memory and no older copy can land after a newer one. A write another
    /// primary has superseded fences this one (`fence`) and is an error.
    async fn update_manifest(&self, change: impl FnOnce(&mut WalManifest)) -> crate::Result<()> {
        let _write = self.manifest_write.lock().await;
        let mut next = self.manifest()?;
        change(&mut next);
        if !self.write_manifest_once(&next).await? {
            return Err(self.fence());
        }
        *self
            .wal_manifest
            .lock()
            .map_err(|_| crate::Error::Internal("WAL manifest lock poisoned".to_string()))? = next;
        Ok(())
    }

    /// Stop writing: another primary has written the bucket's manifest since this one did.
    /// Local durability is marked failed, so every later write is a 503 and `/ready` says why.
    fn fence(&self) -> crate::Error {
        let error = crate::Error::Internal(
            "another primary took over this object storage bucket (its manifest changed under \
             this one); this server stopped writing to it"
                .to_string(),
        );
        self.fenced.store(true, Ordering::SeqCst);
        self.mark_durability_failed(&error);
        crate::Error::ServiceUnavailable {
            message: error.to_string(),
        }
    }

    /// 启动时把清单读进内存；没有清单就从前缀里重建一份。
    async fn load_wal_manifest(&self) -> crate::Result<()> {
        use crate::storage::object::traits::ObjectStorage;

        let read = self
            .storage
            .get_object_versioned(Self::wal_manifest_key())
            .await;
        if let Ok(mut version) = self.manifest_version.lock() {
            *version = read.as_ref().ok().and_then(|(_, v)| v.clone());
        }
        let manifest = match read.map(|(bytes, _)| bytes) {
            Ok(bytes) => match serde_json::from_slice::<WalManifest>(&bytes) {
                Ok(m) => {
                    if !m.is_consistent() {
                        warn!(
                            "WAL manifest is inconsistent: {} live + {} pruned != {} total; \
                             rebuilding from the prefix",
                            m.live_entries(),
                            m.pruned_entries,
                            m.total_entries
                        );
                        self.rebuild_wal_manifest().await?
                    } else {
                        m
                    }
                }
                Err(e) => {
                    warn!(
                        "WAL manifest unreadable ({}); rebuilding from the prefix",
                        e
                    );
                    self.rebuild_wal_manifest().await?
                }
            },
            // No manifest is the normal first-boot case — and also the upgrade case, where
            // chunks already exist under the old flat prefix. Both are handled by scanning.
            Err(_) => self.rebuild_wal_manifest().await?,
        };

        *self
            .wal_manifest
            .lock()
            .map_err(|_| crate::Error::Internal("WAL manifest lock poisoned".to_string()))? =
            manifest;
        Ok(())
    }

    /// 扫描 `wal/` 前缀重建清单。仅用于首次启动、升级，或清单损坏。
    async fn rebuild_wal_manifest(&self) -> crate::Result<WalManifest> {
        use crate::storage::object::traits::ObjectStorage;

        let mut chunks = Vec::new();
        for key in self.list_wal_chunk_keys().await {
            let Ok(bytes) = self.storage.get_object(&key).await else {
                continue;
            };
            let Ok(entries) = rmp_serde::from_slice::<Vec<(u64, WalEntry)>>(&bytes) else {
                continue;
            };
            let (Some((first, _)), Some((last, _))) = (entries.first(), entries.last()) else {
                continue;
            };
            chunks.push(WalChunkRef {
                path: key,
                first_seq: *first,
                last_seq: *last,
                entries: entries.len(),
            });
        }
        chunks.sort_by_key(|c| c.first_seq);

        let total_entries = chunks.iter().map(|c| c.entries as u64).sum();
        if !chunks.is_empty() {
            info!(
                "Rebuilt WAL manifest from prefix scan: {} chunks",
                chunks.len()
            );
        }
        Ok(WalManifest {
            chunks,
            // Anything already pruned is unknowable from a scan; treating it as zero keeps the
            // invariant satisfied for everything still present, which is what it is for.
            pruned_entries: 0,
            total_entries,
            ..Default::default()
        })
    }

    /// Make `dir` ready for this process. A replica's directory is a cache of the bucket: it is
    /// emptied at every start (and marked `REPLICA`), so a restart recovers from the bucket's
    /// newest snapshot rather than mixing with whatever an earlier run applied. A directory
    /// holding anything else is refused rather than emptied -- `--replica` pointed at a
    /// primary's data by mistake must not erase it. A primary started on a replica's directory
    /// is a promotion: the cache is emptied and the primary recovers from the bucket.
    fn prepare_local_dir(
        dir: &std::path::Path,
        replica: bool,
        object_storage: bool,
    ) -> crate::Result<()> {
        const MARKER: &str = "REPLICA";
        let marked = dir.join(MARKER).exists();
        let contents: Vec<PathBuf> = std::fs::read_dir(dir)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name != "LOCK" && name != MARKER
            })
            .collect();
        let empty = |paths: &[PathBuf]| -> crate::Result<()> {
            for path in paths {
                if path.is_dir() {
                    std::fs::remove_dir_all(path)?;
                } else {
                    std::fs::remove_file(path)?;
                }
            }
            Ok(())
        };
        if replica {
            if !contents.is_empty() && !marked {
                return Err(crate::Error::InvalidConfig(format!(
                    "{} holds data that is not a replica's. A replica empties its data \
                     directory at every start and rebuilds it from object storage, so it needs \
                     one of its own",
                    dir.display()
                )));
            }
            empty(&contents)?;
            std::fs::write(
                dir.join(MARKER),
                b"a read replica's cache of object storage\n",
            )?;
        } else if marked {
            if !object_storage {
                return Err(crate::Error::InvalidConfig(format!(
                    "{} is a read replica's directory and holds nothing of its own; start the \
                     primary with the object storage it followed configured, and it recovers \
                     from there",
                    dir.display()
                )));
            }
            warn!(
                "{} was a read replica's directory; recovering this primary from object storage",
                dir.display()
            );
            empty(&contents)?;
            std::fs::remove_file(dir.join(MARKER))?;
        }
        Ok(())
    }

    /// Take the bucket for this process: write `wal/PRIMARY` with an epoch one past the last
    /// and a token of our own. A primary that finds someone else's token there afterwards --
    /// another primary started on the same bucket, a promoted replica -- stops writing
    /// (`ensure_still_primary`). The WAL manifest has no conditional write, so two primaries
    /// would otherwise overwrite each other's manifest with no sign of it.
    async fn claim_primary(&mut self) -> crate::Result<()> {
        use crate::storage::object::traits::ObjectStorage;
        let previous = self
            .storage
            .get_object_fresh(PRIMARY_KEY)
            .await
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        let epoch = previous
            .as_ref()
            .and_then(|p| p["epoch"].as_u64())
            .unwrap_or(0)
            + 1;
        let token = uuid::Uuid::new_v4().to_string();

        // The directory remembers which claims were its own. A directory that was a primary of
        // this bucket, and finds the bucket claimed by someone else since, holds a history the
        // bucket no longer continues: the other primary wrote after it, possibly under the same
        // sequence numbers, and may have pruned the chunks between. Merging the two is not
        // possible, and claiming the bucket back would depose the primary that took over -- which
        // is exactly what a supervisor restarting a crashed old primary would do. So it refuses.
        // An empty directory, a replica's (emptied on promotion) and one from before this file
        // existed have no record and claim as before.
        let local_path = self.wal_dir.as_ref().map(|d| d.join("PRIMARY"));
        let ours: Option<Vec<String>> = local_path
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .map(|v| {
                v["tokens"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|t| t.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default()
            });
        let current = previous
            .as_ref()
            .and_then(|p| p["token"].as_str())
            .map(str::to_string);
        if let (Some(ours), Some(current)) = (&ours, &current) {
            if !ours.contains(current) {
                return Err(crate::Error::InvalidConfig(format!(
                    "another primary (epoch {}) has written to this object storage bucket since \
                     {} last did, so this directory's log is not the bucket's any more. Start \
                     on an empty directory to recover from the bucket -- after stopping the \
                     current primary -- or move this one aside first to keep it",
                    epoch - 1,
                    self.wal_dir
                        .as_ref()
                        .map(|d| d.display().to_string())
                        .unwrap_or_default()
                )));
            }
        }
        // Written before the bucket, holding the claim it replaces too: a crash between the two
        // writes leaves a directory that still recognises whichever one the bucket holds.
        if let Some(path) = &local_path {
            let record = serde_json::json!({
                "tokens": current.iter().chain(std::iter::once(&token)).collect::<Vec<_>>(),
                "epoch": epoch,
            });
            let tmp = path.with_extension("tmp");
            std::fs::write(&tmp, record.to_string())?;
            std::fs::File::open(&tmp)?.sync_all()?;
            std::fs::rename(&tmp, path)?;
        }
        let body = serde_json::json!({
            "epoch": epoch,
            "token": token,
            "started_at": chrono::Utc::now().to_rfc3339(),
        });
        self.storage
            .put_object(
                PRIMARY_KEY,
                bytes::Bytes::from(body.to_string()),
                Some("application/json"),
            )
            .await
            .map_err(|e| crate::Error::Internal(format!("claiming the bucket: {e}")))?;
        if previous.is_some() {
            info!("primary of this bucket, epoch {epoch}");
        }
        self.primary_token = Some(token);
        self.primary_epoch.store(epoch, Ordering::SeqCst);

        // Then the manifest, which is the lock: rewritten with this epoch, conditionally on the
        // version just read. That changes its version under a primary still running on it, so
        // that primary's next write of it fails and it stops (`fence`) -- no window, where the
        // `wal/PRIMARY` poll left up to a second of writes acknowledged by the old one.
        let conditional = self.probe_conditional_writes().await;
        self.conditional_writes.store(conditional, Ordering::SeqCst);
        if !conditional {
            warn!(
                "this object store does not honour conditional writes; a replaced primary is \
                 stopped by polling wal/PRIMARY every DXMT_PRIMARY_CHECK_MS instead, and may \
                 acknowledge writes for that long"
            );
        }
        for attempt in 0..5 {
            let mut manifest = self.manifest()?;
            manifest.epoch = epoch;
            if self.write_manifest_once(&manifest).await? {
                *self.wal_manifest.lock().map_err(|_| {
                    crate::Error::Internal("WAL manifest lock poisoned".to_string())
                })? = manifest;
                return Ok(());
            }
            // Written since it was read: the primary being replaced is still writing. Read the
            // newer one and take that.
            if attempt < 4 {
                self.load_wal_manifest().await?;
            }
        }
        Err(crate::Error::Internal(
            "could not take the WAL manifest: it kept changing under this server -- another \
             primary is writing the bucket"
                .to_string(),
        ))
    }

    /// Whether the object store makes a write conditional on an object's version: a write
    /// against a version the object does not have must be refused. Some stores accept the
    /// header and ignore it, and fencing that trusts one of those is no fencing at all.
    async fn probe_conditional_writes(&self) -> bool {
        use crate::storage::object::errors::ObjectStorageError;
        use crate::storage::object::traits::{ObjectStorage, PutCondition};
        const PROBE: &str = "wal/CONDITIONAL-WRITE-PROBE";
        let body = || bytes::Bytes::from_static(b"dxmt conditional write probe");
        match self
            .storage
            .put_object_if(PROBE, body(), None, PutCondition::IfAbsent)
            .await
        {
            Ok(_) | Err(ObjectStorageError::PreconditionFailed { .. }) => {}
            Err(ObjectStorageError::ConditionalWritesUnsupported) => return false,
            Err(e) => {
                warn!("probing for conditional writes: {e}");
                return false;
            }
        }
        match self
            .storage
            .put_object_if(
                PROBE,
                body(),
                None,
                // No ETag or content hash is ever "1".
                PutCondition::IfMatch("1".to_string()),
            )
            .await
        {
            Err(ObjectStorageError::PreconditionFailed { .. }) => true,
            Ok(_) => false,
            Err(e) => {
                warn!("probing for conditional writes: {e}");
                false
            }
        }
    }

    /// Refuse to write once another primary has taken the bucket. Reads `wal/PRIMARY` at most
    /// every `primary_check_every`; when the token there is not ours, local durability is marked
    /// failed -- every later write is a 503 and `/ready` says why -- since this process no
    /// longer owns what it would write to.
    async fn ensure_still_primary(&self) -> crate::Result<()> {
        let Some(ours) = self.primary_token.as_deref() else {
            return Ok(());
        };
        // With conditional writes the manifest is the lock, and every write that reaches the
        // bucket checks it; this poll is the fallback for stores without them.
        if self.conditional_writes.load(Ordering::SeqCst) {
            return Ok(());
        }
        let due = self
            .primary_checked
            .lock()
            .map(|at| at.is_none_or(|t| t.elapsed() >= self.primary_check_every))
            .unwrap_or(true);
        if !due {
            return Ok(());
        }
        let current = self
            .storage
            .get_object_fresh(PRIMARY_KEY)
            .await
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        match current {
            Some(p) if p["token"].as_str() != Some(ours) => {
                let error = crate::Error::Internal(format!(
                    "another primary (epoch {}) took over this object storage bucket; this \
                     server stopped writing to it",
                    p["epoch"]
                ));
                self.mark_durability_failed(&error);
                Err(crate::Error::ServiceUnavailable {
                    message: error.to_string(),
                })
            }
            // Ours, or unreadable for now: a bucket that cannot be read is the mirror's
            // problem to report, not a reason to stop writing locally.
            _ => {
                if let Ok(mut at) = self.primary_checked.lock() {
                    *at = Some(std::time::Instant::now());
                }
                Ok(())
            }
        }
    }

    /// Whether this storage is a read replica (`new_replica`).
    pub fn is_replica(&self) -> bool {
        self.replica
    }

    /// On a replica, apply WAL chunks the primary has uploaded since the last call, in order.
    /// Returns how many entries were applied.
    ///
    /// The manifest is read from the bucket each time (a replica's reads skip the caches, see
    /// `new_replica`), and only a contiguous run starting at `replica_applied_next` is applied:
    /// a chunk listed but not readable yet, or out of order, means nothing is applied this time
    /// and the next call tries again, since skipping an entry would diverge for good. If the
    /// primary has pruned chunks this replica had not applied -- it snapshotted while the
    /// replica was far behind -- the entries are gone from the bucket's WAL and only the
    /// snapshot holds them: the replica marks itself stale and reports not ready, and a restart
    /// recovers it from that snapshot.
    pub async fn follow_object_storage(&self) -> crate::Result<usize> {
        use crate::storage::object::traits::ObjectStorage;
        if !self.replica {
            return Ok(0);
        }
        let bytes = match self.storage.get_object(Self::wal_manifest_key()).await {
            Ok(bytes) => bytes,
            Err(_) => return Ok(0), // nothing mirrored yet
        };
        let manifest: WalManifest = match serde_json::from_slice(&bytes) {
            Ok(m) => m,
            // A manifest being replaced as it is read; the next call sees the whole one.
            Err(_) => return Ok(0),
        };
        if !manifest.is_consistent() {
            return Ok(0);
        }
        let next = self.replica_applied_next.load(Ordering::SeqCst);
        if let Some(first) = manifest.chunks.iter().map(|c| c.first_seq).min() {
            let lowest_needed_gone = first > next
                && manifest
                    .chunks
                    .iter()
                    .all(|c| c.first_seq > next || c.last_seq < next);
            if lowest_needed_gone {
                let message = format!(
                    "the primary pruned WAL entries {next}..{first} before this replica applied \
                     them; restart the replica to recover from the primary's latest snapshot"
                );
                if let Ok(mut stale) = self.replica_stale.lock() {
                    if stale.is_none() {
                        warn!("{message}");
                    }
                    *stale = Some(message);
                }
                return Ok(0);
            }
        }
        if !manifest.chunks.iter().any(|c| c.last_seq >= next) {
            return Ok(0);
        }
        *self
            .wal_manifest
            .lock()
            .map_err(|_| crate::Error::Internal("WAL manifest lock poisoned".to_string()))? =
            manifest;
        let mut entries = self.fetch_wal_chunks(next).await?;
        entries.sort_by_key(|(seq, _)| *seq);
        entries.dedup_by_key(|(seq, _)| *seq);
        // Contiguous from `next`, or nothing this time.
        let contiguous = entries
            .iter()
            .enumerate()
            .take_while(|(i, (seq, _))| *seq == next + *i as u64)
            .count();
        if contiguous == 0 {
            return Ok(0);
        }
        entries.truncate(contiguous);
        let last = entries.last().map(|(seq, _)| *seq).unwrap_or(next);
        APPLYING_REPLICATED.scope((), self.replay(entries)).await?;
        self.replica_applied_next.store(last + 1, Ordering::SeqCst);
        Ok(contiguous)
    }

    /// Run `follow_object_storage` every `every`, for as long as the process lives.
    pub fn spawn_replica_follower(storage: Arc<Self>, every: std::time::Duration) {
        if !storage.replica {
            return;
        }
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(every);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // A stale replica can only recover by starting again from the newest snapshot, and a
            // process supervisor (systemd, Kubernetes) restarts one that exits -- which is
            // faster than waiting for someone to read /ready. DXMT_REPLICA_EXIT_ON_STALE=0
            // keeps it running, not ready, instead.
            let exit_on_stale = crate::config::env::var("DXMT_REPLICA_EXIT_ON_STALE")
                .map(|v| !matches!(v.trim(), "0" | "false" | "no" | "off"))
                .unwrap_or(true);
            loop {
                ticker.tick().await;
                match storage.follow_object_storage().await {
                    Ok(0) => {}
                    Ok(n) => debug!("replica applied {n} WAL entries from object storage"),
                    Err(e) => warn!("replica could not apply WAL from object storage: {e}"),
                }
                let stale = storage.replica_stale.lock().ok().and_then(|s| s.clone());
                if let (true, Some(why)) = (exit_on_stale, stale) {
                    error!("{why}; exiting so the process supervisor restarts this replica");
                    std::process::exit(3);
                }
            }
        });
    }

    /// Bring object storage up to the local WAL: upload, as one chunk, every entry from
    /// `wal_mirrored_next` on.
    ///
    /// Each write used to upload its own entries and, if that failed, log a warning and move
    /// on. Nothing uploaded them later, so the bucket kept a hole for good -- and the manifest's
    /// `live + pruned == total` check cannot see it, because the missing chunk was never
    /// registered. A machine recovering from object storage alone then came back without those
    /// writes and said nothing; a replica following the bucket would never have them. Now the
    /// next write (or the next start) uploads everything from the first entry not yet mirrored,
    /// in order, so the chunks stay contiguous. `fresh` is the batch just logged, which is all
    /// there is to send in the usual case; anything else is read back from the local log. The
    /// write itself is durable locally either way, so a failure here is logged, not returned.
    async fn mirror_wal(&self, fresh: &[(u64, WalEntry)]) -> crate::Result<()> {
        let _turn = self.wal_mirror_lock.lock().await;
        let next = self.wal_mirrored_next.load(Ordering::SeqCst);
        let Some(wal) = self.wal.as_ref() else {
            return Ok(());
        };
        let read_back = {
            let Ok(guard) = wal.lock() else {
                return Ok(());
            };
            let local_next = guard.next_sequence();
            if next >= local_next {
                return Ok(()); // another write's upload already carried these
            }
            let fresh_is_all = fresh.first().map(|(s, _)| *s) == Some(next)
                && fresh.last().map(|(s, _)| *s + 1) == Some(local_next);
            if fresh_is_all {
                None
            } else {
                match guard.replay_from(next) {
                    Ok(entries) => Some(entries),
                    Err(e) => {
                        warn!(
                            "could not read WAL entries from {} to mirror them: {}",
                            next, e
                        );
                        return Ok(());
                    }
                }
            }
        };
        let entries: &[(u64, WalEntry)] = read_back.as_deref().unwrap_or(fresh);
        let (Some((first, _)), Some((last, _))) = (entries.first(), entries.last()) else {
            return Ok(());
        };
        match self.offload_wal_chunk(*first, entries).await {
            Ok(()) => self.wal_mirrored_next.store(last + 1, Ordering::SeqCst),
            // Superseded: the caller's write must not be acknowledged. It is in this machine's
            // log, but the bucket -- the history the cluster goes on with -- is another
            // primary's now, and will never have it.
            Err(e) if self.fenced.load(Ordering::SeqCst) => return Err(e),
            Err(e) => warn!(
                "{} WAL entries from {} are not in object storage yet ({}); the next write \
                 retries them",
                last + 1 - next.max(*first),
                next.max(*first),
                e
            ),
        }
        Ok(())
    }

    /// 把一批 WAL 条目作为不可变分块推到对象存储，并登记进清单。
    async fn offload_wal_chunk(
        &self,
        first_seq: u64,
        entries: &[(u64, WalEntry)],
    ) -> crate::Result<()> {
        use crate::storage::object::traits::ObjectStorage;

        // MessagePack, not bincode: WalEntry carries a serde_json::Value payload, which bincode
        // cannot round-trip. This is the same reason wal.rs uses rmp.
        let bytes =
            rmp_serde::to_vec(entries).map_err(|e| crate::Error::Serialization(e.to_string()))?;

        let path = Self::wal_chunk_key(first_seq, self.primary_epoch.load(Ordering::SeqCst));
        self.storage
            .put_object(
                &path,
                bytes::Bytes::from(bytes),
                Some("application/octet-stream"),
            )
            .await
            .map_err(|e| crate::Error::Internal(format!("uploading WAL chunk: {}", e)))?;

        // Chunk first, then the manifest that references it. The other order publishes a
        // manifest pointing at an object that may not exist, and recovery fails outright; this
        // way a failed manifest write leaves an unreferenced object — wasted space, nothing
        // more.
        //
        // Through `update_manifest`, so a failed write leaves no entry behind: kept, it stayed in
        // memory while the bucket never had it, and the retry -- the same entries and more,
        // from the same first sequence, so the same key -- added a second, overlapping one.
        // Found against GCS, whose per-object write rate limit fails manifest writes under load.
        let chunk = WalChunkRef {
            path,
            first_seq,
            last_seq: entries.last().map(|(s, _)| *s).unwrap_or(first_seq),
            entries: entries.len(),
        };
        self.update_manifest(|m| {
            m.total_entries += chunk.entries as u64;
            m.chunks.push(chunk);
        })
        .await
    }

    /// 列出 `wal/` 下所有分块的键，按序号（即字典序）排好。
    async fn list_wal_chunk_keys(&self) -> Vec<String> {
        use crate::storage::object::traits::ObjectStorage;

        let options = crate::storage::object::types::ListObjectsOptions {
            prefix: Some("wal/".to_string()),
            max_keys: None,
            continuation_token: None,
            delimiter: None,
        };
        match self.storage.list_objects(&options).await {
            Ok(result) => {
                let mut keys: Vec<String> = result.objects.into_iter().map(|o| o.key).collect();
                keys.sort();
                keys
            }
            // No chunks yet is the normal first-boot case, not an error.
            Err(_) => Vec::new(),
        }
    }

    /// 从对象存储取回序号 >= `from_seq` 的 WAL 条目。
    ///
    /// 走清单，不 LIST 前缀：LIST 的代价随日志增长，而且**列不出"少了一块"**——
    /// 清单记了条目总数，对不上就说得出来。
    async fn fetch_wal_chunks(&self, from_seq: u64) -> crate::Result<Vec<(u64, WalEntry)>> {
        use crate::storage::object::traits::ObjectStorage;

        let manifest = self.manifest()?;
        let mut out = Vec::new();
        let mut expected = 0usize;
        let mut missing = 0usize;

        for chunk in &manifest.chunks {
            // last_seq, not first_seq: a chunk straddling the cutoff still holds entries we
            // need.
            if chunk.last_seq < from_seq {
                continue;
            }
            expected += chunk.entries;

            let Ok(bytes) = self.storage.get_object(&chunk.path).await else {
                warn!("WAL chunk {} is listed but unreadable", chunk.path);
                missing += 1;
                continue;
            };
            let entries: Vec<(u64, WalEntry)> = match rmp_serde::from_slice(&bytes) {
                Ok(e) => e,
                Err(e) => {
                    warn!("WAL chunk {} could not be decoded: {}", chunk.path, e);
                    missing += 1;
                    continue;
                }
            };
            out.extend(entries.into_iter().filter(|(seq, _)| *seq >= from_seq));
        }

        if missing > 0 {
            // Loud, and specific about the consequence. Without the manifest this would have
            // been invisible: a LIST simply would not have mentioned the chunk.
            warn!(
                "{} of {} WAL chunks after sequence {} could not be read; recovery will be \
                 missing up to {} entries",
                missing,
                manifest.chunks.len(),
                from_seq,
                expected.saturating_sub(out.len())
            );
        }

        Ok(out)
    }

    /// How far the bucket's WAL may be pruned now that a snapshot through `seq` is uploaded:
    /// to the newest snapshot at least `DXMT_WAL_RETAIN_SECS` old, or `None` if there is none.
    ///
    /// Pruning to the snapshot just written left a replica exactly one snapshot interval to
    /// apply each chunk. One that fell further behind -- or simply restarted, since recovery
    /// takes seconds and the primary keeps snapshotting meanwhile -- found its next chunks gone,
    /// went stale, restarted, and was behind again by the time it was up: under steady writes
    /// with 5 s snapshots it never caught up (`scripts/replica_soak.py`). Kept chunks cost
    /// bucket space for the window's writes and nothing else; the local log is compacted as
    /// before. `0` restores pruning at every snapshot.
    fn retained_prune_point(&self, seq: u64) -> Option<u64> {
        let retain = self.wal_retain_secs.load(Ordering::Relaxed);
        if retain == 0 {
            return Some(seq);
        }
        let retain = std::time::Duration::from_secs(retain);
        let mut points = self.offloaded_snapshots.lock().ok()?;
        points.push_back((std::time::Instant::now(), seq));
        let ripe = points.iter().rposition(|(at, _)| at.elapsed() >= retain)?;
        // Older candidates than the one used can never be used again.
        points.drain(..ripe);
        points.front().map(|(_, s)| *s)
    }

    /// Set how long WAL chunks stay in the bucket after a snapshot covers them
    /// (`DXMT_WAL_RETAIN_SECS`); `Duration::ZERO` prunes at every snapshot.
    pub fn set_wal_retention(&self, retain: std::time::Duration) {
        self.wal_retain_secs
            .store(retain.as_secs(), Ordering::Relaxed);
    }

    /// 删掉全部条目都低于 `keep_from` 的 WAL 分块。
    ///
    /// **先更新清单再删对象。** 反过来的话，中途失败会留下一份指向已删对象的清单，
    /// 恢复就会报"分块读不出来"——而实际上那些数据已经在快照里了，警告是假的。
    async fn prune_wal_chunks(&self, keep_from: u64) {
        use crate::storage::object::traits::ObjectStorage;

        let gone: Vec<WalChunkRef> = match self.manifest() {
            Ok(m) => m
                .chunks
                .iter()
                .filter(|c| c.last_seq < keep_from)
                .cloned()
                .collect(),
            Err(_) => return,
        };
        if gone.is_empty() {
            return;
        }
        let mut dropped = Vec::new();
        let updated = self
            .update_manifest(|m| {
                m.chunks.retain(|c| {
                    if c.last_seq < keep_from {
                        guard_pruned(&mut dropped, c);
                        false
                    } else {
                        true
                    }
                });
                m.pruned_entries += dropped.iter().map(|c| c.entries as u64).sum::<u64>();
            })
            .await;
        if let Err(e) = updated {
            // The objects stay, and so does the manifest that lists them -- consistent, just
            // larger than it needs to be.
            warn!(
                "WAL manifest update failed; leaving {} chunks in place: {e}",
                gone.len()
            );
            return;
        }
        for chunk in dropped {
            let _ = self.storage.delete_object(&chunk.path).await;
        }
    }

    /// 把刚写好的快照完整读一遍，确认它真的能用来恢复。
    /// 创建表时自动选择最优索引
    pub async fn create_table(&self, config: TableConfig) -> crate::Result<Table> {
        // An alias and a table of the same name would make one of the two unreachable, and
        // which one it is depends on the order of two maps. Refuse it here rather than pick.
        if self.aliases.read().contains_key(&config.name) {
            return Err(crate::Error::InvalidInput(format!(
                "'{}' is an alias on this server; drop it or choose another name",
                config.name
            )));
        }
        if let Some(q) = &config.quantization {
            // A PQ code is one byte per subspace: centroid indices are stored `as u8`, so more
            // than 256 wrapped silently and every code above 255 named the wrong centroid --
            // search quality fell with no error anywhere. And zero subspaces divides by zero.
            if let Some(k) = q.num_centroids {
                if !(2..=256).contains(&k) {
                    return Err(crate::Error::InvalidInput(format!(
                        "quantization.num_centroids must be between 2 and 256 (got {k}): a code \
                         is one byte per subspace"
                    )));
                }
            }
            if q.num_subvectors == Some(0) {
                return Err(crate::Error::InvalidInput(
                    "quantization.num_subvectors must be at least 1".to_string(),
                ));
            }
        }
        let config = self.dimension_from_the_model(config).await?;
        self.create_table_inner(config, true).await
    }

    /// Fill in a dimension of zero by asking the table's model how wide its vectors are.
    ///
    /// Zero is the sentinel for "not given", set by the API layer, which has already refused
    /// it without an `embedding`. It is only read here and not in `create_table_inner`,
    /// because that is also the recovery and rehydration path: a replayed table already knows
    /// its width, and a restart must not depend on a provider being reachable.
    ///
    /// The cost is one call to the provider during a create, and a create that fails when the
    /// provider is down. That is the right failure: the alternative is a table whose declared
    /// width and whose model's width disagree, which is discovered at the first insert, by
    /// someone else, as a dimension mismatch naming neither the model nor the table's origin.
    async fn dimension_from_the_model(&self, config: TableConfig) -> crate::Result<TableConfig> {
        if config.dimension != 0 {
            return Ok(config);
        }
        let Some(embedding) = config.embedding.as_ref() else {
            return Err(crate::Error::InvalidInput(format!(
                "table '{}' has no dimension and no embedding model to take one from",
                config.name
            )));
        };
        let probe = Self::embed(embedding, &["dimension probe".to_string()], None).await?;
        let dimension = probe.first().map(|v| v.len()).unwrap_or(0);
        if dimension == 0 {
            return Err(crate::Error::InvalidConfig(format!(
                "model '{}' returned an empty vector, so this table has no width to take",
                embedding.model
            )));
        }
        info!(
            "Table '{}' takes dimension {} from model '{}'",
            config.name, dimension, embedding.model
        );
        Ok(TableConfig {
            dimension,
            ..config
        })
    }

    /// Build the in-memory table container, optionally logging the creation.
    ///
    /// Recovery and rehydration rebuild the container from a snapshot that already records the
    /// table, so they pass `log = false`. Logging it again appends a `CreateTableWithConfig`
    /// frame on every boot and every rehydration, and the log then grows without bound across
    /// restarts even though nothing was written. The `replaying` flag cannot be used for this:
    /// rehydration runs concurrently with live writes, and flipping a process-wide flag would
    /// swallow their WAL entries.
    async fn create_table_inner(&self, config: TableConfig, log: bool) -> crate::Result<Table> {
        self.ensure_writable()?;
        // Check if table exists (quickly release the lock)
        {
            let tables = self.read_tables("create_table").await;
            if tables.contains_key(&config.name) {
                return Err(crate::Error::TableAlreadyExists(config.name.clone()));
            }
        }

        // 创建表元数据
        let mut table = Table::new(config.clone())?;

        // 配置层级决策：表级配置覆盖全局配置
        let memory_optimization = self.memory_optimization_of(&table);

        info!(
            "Table '{}' memory_optimization: {} (table_level: {:?}, global: {})",
            table.name,
            memory_optimization,
            config.memory_optimization,
            self.default_memory_optimization
        );

        // 准备索引配置上下文
        // Estimate initial dataset size: assume at least 10K vectors for better index selection
        // This triggers HNSW for production use cases (HNSW selected for datasets ≥ 5K)
        let estimated_size = 10_000;

        let index_config = IndexConfig {
            dataset_size: estimated_size,
            dimension: config.dimension,
            distance_metric: config.distance_metric,
            memory_budget_mb: self.hardware_capabilities.available_memory_gb as usize * 512, // 使用一半内存
            performance_requirements: self.performance_requirements.clone(),
            cost_constraints: self.cost_constraints.clone(),
            hardware_capabilities: self.hardware_capabilities.clone(),
        };

        // 根据 memory_optimization 配置选择索引算法
        let mut index_decision = if memory_optimization {
            // 内存优化模式：优先选择压缩和聚类索引
            self.select_memory_optimized_index(&index_config).await?
        } else {
            // 高精度模式：使用标准索引选择逻辑
            self.index_selector
                .select_optimal_index(&index_config)
                .await?
        };
        if let Some(forced) = Self::forced_algorithm(config.dimension) {
            index_decision.selected_algorithm = forced;
        }
        // The table's own parameters win over both the selector and the environment: they are
        // the most specific statement anyone made about this table.
        Self::apply_table_index_params(
            &mut index_decision.selected_algorithm,
            config.hnsw.as_ref(),
            config.quantization.as_ref(),
        );

        info!(
            "Selected {} algorithm for table {} with confidence {:.2}",
            format!("{:?}", index_decision.selected_algorithm)
                .split('{')
                .next()
                .unwrap_or("Unknown"),
            table.name,
            index_decision.confidence
        );

        // The table's advertised index_type came from the caller's config and defaulted to
        // BruteForce, so a table running HNSW reported "brute_force" over the API forever.
        // Make the metadata say what was actually chosen.
        table.index_type = index_type_of(&index_decision.selected_algorithm);

        // 创建索引实例
        let index_impl = self.create_index_instance(
            &index_decision.selected_algorithm,
            config.dimension,
            config.distance_metric,
        )?;

        // 用 IndexWrapper 包装索引以支持 interior mutability
        let index = crate::search::IndexWrapper::new(index_impl);

        // Create named vector indexes if configured
        let named_indexes = if let Some(ref named_vec_configs) = config.named_vectors {
            let collection = IndexCollection::new();
            for vec_cfg in named_vec_configs {
                let named_index_impl = self.create_index_instance(
                    &index_decision.selected_algorithm,
                    vec_cfg.dimension,
                    vec_cfg.distance_metric,
                )?;
                collection
                    .add_index(vec_cfg.name.clone(), IndexWrapper::new(named_index_impl))
                    .await;
            }
            Some(collection)
        } else {
            None
        };

        // 创建全文搜索引擎（如果启用）
        #[cfg(feature = "text-bm25")]
        let full_text_engine = if table.enable_full_text_search {
            match self.create_full_text_engine(&table).await {
                Ok(engine) => {
                    info!(
                        "Successfully created full-text search engine for table {}",
                        table.name
                    );
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    warn!("Failed to create full-text search engine for table {}: {}. Full-text search will be disabled.", table.name, e);
                    None
                }
            }
        } else {
            None
        };

        // A create is fresh -- whatever a same-named table left on disk is not this table's
        // -- when it is a live `CREATE TABLE`, and not when it is one being replayed: the
        // row files on disk are then this table's, written before the crash, and the log
        // entries about to be replayed point into the vector file rather than carrying the
        // vectors. (Replaying with `fresh` wiped the files, and was harmless only while the
        // log held a copy of every vector.)
        let fresh = log && !self.replaying.load(Ordering::SeqCst);
        let (vector_file, vector_rows) = self.open_vector_file(&table, fresh)?;
        let records = self.open_record_store(&table, fresh)?;
        let table_data = IntelligentTableData {
            table: table.clone(),
            records,
            vector_file,
            vector_rows,
            selected_algorithm: index_decision.selected_algorithm.clone(),
            index,
            named_indexes,
            index_decision,
            performance_stats: Arc::new(Mutex::new(PerformanceStats::default())),
            stranded_last: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            repaired_total: Arc::new(AtomicU64::new(0)),
            // A table being created has no records, so an empty pair is already correct and
            // there is nothing to rebuild. `restore_table` clears this back to `None`.
            node_map: Default::default(),
            writes: Default::default(),
            derived_changes: parking_lot::Mutex::new(None),
            derived_build: Arc::new(tokio::sync::Mutex::new(())),
            derived: parking_lot::RwLock::new(Some(DerivedIndexes {
                payload: {
                    let mut payload = crate::storage::payload_index::PayloadIndex::new();
                    if let Some(policy) = &table.payload_index {
                        payload.set_policy(policy.clone());
                    }
                    payload
                },
                ..DerivedIndexes::default()
            })),
            #[cfg(feature = "text-bm25")]
            full_text_engine,
        };

        // Re-acquire the lock to insert the table
        {
            let mut tables = self.write_tables("create_table").await;
            tables.insert(table.name.clone(), table_data);
        }

        if log {
            self.wal_append(vec![WalEntry::CreateTableWithConfig { config }])
                .await?;
        }

        Ok(table)
    }

    /// 插入记录 - Channel-based 无锁架构
    /// Give every record that arrived as text a vector, and file its text in the payload.
    ///
    /// Called before the insert goes to the channel, so the provider round trip is *not* taken
    /// with the table lock held and does not sit in front of the single insert processor — the
    /// one thing that would make one slow model everyone's problem.
    ///
    /// Not a method on the insert processor for the same reason, and not on the API handler
    /// because then MCP, gRPC and DXQL would each have to remember it. This is the same
    /// argument alias resolution settled: one place in the storage layer, and a protocol added
    /// later cannot forget.
    async fn fill_vectors_from_text(
        &self,
        table: &Table,
        requests: &mut [CreateRecordRequest],
    ) -> crate::Result<()> {
        let has_text = requests.iter().any(|r| r.text.is_some());
        let needs_vector = requests.iter().any(|r| r.vector.is_empty());
        if !has_text && !needs_vector {
            return Ok(());
        }

        let Some(config) = table.embedding.as_ref() else {
            if has_text {
                return Err(crate::Error::InvalidInput(format!(
                    "table '{}' has no embedding model, so it cannot accept `text`. Create it \
                     with an \"embedding\" config naming a model, or send a \"vector\".",
                    table.name
                )));
            }
            let id = requests
                .iter()
                .find(|r| r.vector.is_empty())
                .map(|r| r.id.clone())
                .unwrap_or_default();
            return Err(crate::Error::InvalidInput(format!(
                "record '{id}' has an empty vector and table '{}' has no embedding model to \
                 fill it from",
                table.name
            )));
        };

        // The text goes into the payload whether or not it is what produced the vector: a
        // caller who embedded client-side still wants the chunk back with the hit, and having
        // the field mean two things depending on who computed the vector would be worse.
        let field = config.field().to_string();
        for record in requests.iter_mut() {
            if let Some(text) = record.text.take() {
                let payload = record
                    .payload
                    .get_or_insert_with(|| serde_json::Value::Object(Default::default()));
                let object = payload.as_object_mut().ok_or_else(|| {
                    crate::Error::InvalidInput(format!(
                        "record '{}' sends `text`, which is filed under payload field \
                         '{field}', but its payload is not an object",
                        record.id
                    ))
                })?;
                object.insert(field.clone(), serde_json::Value::String(text));
            }
        }

        let mut pending: Vec<usize> = Vec::new();
        let mut texts: Vec<String> = Vec::new();
        for (i, record) in requests.iter().enumerate() {
            if !record.vector.is_empty() {
                continue;
            }
            let text = record
                .payload
                .as_ref()
                .and_then(|p| p.get(&field))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if text.trim().is_empty() {
                return Err(crate::Error::InvalidInput(format!(
                    "record '{}' has no vector and nothing to embed: send \"text\", or a \
                     payload field '{field}', or a \"vector\"",
                    record.id
                )));
            }
            pending.push(i);
            texts.push(text.to_string());
        }
        if pending.is_empty() {
            return Ok(());
        }

        let vectors = Self::embed(config, &texts, Some(table.dimension)).await?;
        for (i, vector) in pending.into_iter().zip(vectors) {
            requests[i].vector = vector;
        }
        Ok(())
    }

    /// Call the table's model, or say which build cannot.
    ///
    /// The `cfg` is here and nowhere else so that the two paths cannot drift: every caller gets
    /// the same message from a binary built without the feature, rather than the feature being
    /// invisible at one entry point and an error at another.
    async fn embed(
        config: &crate::types::table::EmbeddingConfig,
        texts: &[String],
        dimension: Option<usize>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        #[cfg(feature = "server-embedding")]
        {
            crate::embeddings::service::embed_texts(config, texts, dimension).await
        }
        #[cfg(not(feature = "server-embedding"))]
        {
            let _ = (config, texts, dimension);
            Err(crate::Error::NotImplemented(
                "this build has no embedding support: rebuild with --features server-embedding, \
                 or send vectors instead of text"
                    .to_string(),
            ))
        }
    }

    /// Turn a query's `text` into its `vector`, using the table's model.
    ///
    /// The embedding has to be the table's, not the caller's choice, and not a server default:
    /// a query vector from a different model than the corpus is not a worse answer, it is a
    /// meaningless one, and it comes back looking like a normal ranked page.
    /// The dense half of a hybrid search from its `text`, when the body has text and no
    /// vector and the table has a model -- as `/search` does with `text`. Without this a
    /// text-only hybrid search on a table with a model ran the keyword half alone: a question
    /// sharing no word with the note that answers it came back empty, which on the knowledge
    /// base table is most questions. A table without a model, or a body with a vector, is left
    /// as it was.
    async fn embed_hybrid_text(
        &self,
        tablename: &str,
        mut request: HybridSearchRequest,
    ) -> crate::Result<HybridSearchRequest> {
        if request.vector.is_some() {
            return Ok(request);
        }
        let Some(text) = request.text.clone().filter(|t| !t.trim().is_empty()) else {
            return Ok(request);
        };
        let model = self
            .read_tables("embed_hybrid_text")
            .await
            .get(tablename)
            .and_then(|data| data.table.embedding.clone());
        let Some(config) = model else {
            return Ok(request);
        };
        request.vector = Self::embed(&config, &[text], None)
            .await?
            .into_iter()
            .next();
        Ok(request)
    }

    async fn embed_query(
        &self,
        tablename: &str,
        mut query: SearchQuery,
    ) -> crate::Result<SearchQuery> {
        let Some(text) = query.text.take() else {
            return Ok(query);
        };
        if !query.vector.is_empty() {
            return Err(crate::Error::InvalidInput(
                "a search takes `vector` or `text`, not both".to_string(),
            ));
        }
        if text.trim().is_empty() {
            return Err(crate::Error::InvalidInput(
                "`text` is empty; there is nothing to embed".to_string(),
            ));
        }
        if query.vector_name.is_some() {
            return Err(crate::Error::InvalidInput(
                "`text` embeds into the table's default vector space; a named space has its own \
                 model and this server does not hold one. Send `vector` with `vector_name`."
                    .to_string(),
            ));
        }
        let table = self
            .read_tables("embed_query")
            .await
            .get(tablename)
            .map(|data| data.table.clone())
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        let config = table.embedding.as_ref().ok_or_else(|| {
            crate::Error::InvalidInput(format!(
                "table '{tablename}' has no embedding model, so it cannot search by `text`. \
                 Send a `vector`, or recreate the table with an \"embedding\" config."
            ))
        })?;
        let vectors =
            Self::embed(config, std::slice::from_ref(&text), Some(table.dimension)).await?;
        query.vector = vectors.into_iter().next().ok_or_else(|| {
            crate::Error::Internal("the embedding provider returned no vector".to_string())
        })?;
        Ok(query)
    }

    pub async fn insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>> {
        self.ensure_kb_table(&self.resolve(tablename)).await?;
        self.insert_records_with_rows(tablename, requests, Vec::new())
            .await
    }

    /// `insert_records`, with `rows_in_file` saying, per request, which vectors are already
    /// in the table's row file and at which row. Only WAL replay knows that; a live insert
    /// passes nothing and every vector is appended.
    async fn insert_records_with_rows(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
        rows_in_file: Vec<Option<u64>>,
    ) -> crate::Result<Vec<Record>> {
        let tablename = &*self.resolve(tablename);
        self.ensure_writable()?;
        self.ensure_memory_headroom()?;
        // Writing to an evicted table has to bring it back first, or the insert lands in a
        // fresh empty table and the snapshot's records vanish on the next snapshot.
        self.ensure_resident(tablename).await?;

        let mut requests = requests;
        let stage = std::time::Instant::now();
        // Checked before the lock is taken, because this is the insert path: every batch on
        // every table would otherwise pay a `tables` read lock and a `Table` clone to discover
        // that it has nothing to embed.
        if requests
            .iter()
            .any(|r| r.text.is_some() || r.vector.is_empty())
        {
            let table = self
                .read_tables("insert_records")
                .await
                .get(tablename)
                .map(|data| data.table.clone())
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
            self.fill_vectors_from_text(&table, &mut requests).await?;
        }
        insert_stages::add(insert_stages::PREPARE, stage);
        // 创建响应 channel
        let (response_tx, response_rx) = oneshot::channel();

        // 发送插入请求到后台处理器
        self.insert_tx
            .send(InsertRequest {
                table_name: tablename.to_string(),
                records: requests,
                rows_in_file,
                response_tx,
                enqueued_at: std::time::Instant::now(),
            })
            .map_err(|e| crate::Error::Internal(format!("Failed to send insert request: {}", e)))?;

        // 等待后台处理器响应
        let (inserted, rows) = response_rx.await.map_err(|e| {
            crate::Error::Internal(format!("Insert request was cancelled: {}", e))
        })??;

        // 先落内存再记日志，最后才返回给调用方。中间崩掉的话这次写入还没被确认，丢了也符合语义；
        // 反过来先记日志的话，处理器若因维度不符等原因失败，日志里就留下了一条永远重放不成功的
        // 幽灵记录。
        // A record whose vector went to the row file is logged as a pointer to that row, not
        // as the vector again: the row was fsynced before the processor answered, so it is
        // as durable as the entry about to be written, and at 960 dimensions it is the
        // difference between 3.9 KB and 100 bytes per record in the log. A table whose
        // vectors are in memory has no row to point at and logs the vector.
        let stage = std::time::Instant::now();
        let by_row = self.log_points_at_rows();
        self.wal_append(
            inserted
                .iter()
                .zip(rows.iter().copied().chain(std::iter::repeat(None)))
                .map(|(r, row)| match row.filter(|_| by_row) {
                    Some(row) => WalEntry::UpsertRecordInRowFile {
                        table: tablename.to_string(),
                        record: without_vector(r),
                        row,
                    },
                    None => WalEntry::UpsertRecordV2 {
                        table: tablename.to_string(),
                        record: r.clone(),
                    },
                })
                .collect(),
        )
        .await?;
        insert_stages::add(insert_stages::WAL, stage);

        // Deliberately no index re-evaluation here. A rebuild is O(n) over the whole table and
        // runs to completion before the caller's write returns, so triggering it from the write
        // path hands the entire cost to whichever request happened to cross a size threshold —
        // and `reevaluate_indexes` sweeps every table, not just this one. The sweep spawned by
        // `spawn_index_reevaluation` (DXMT_REINDEX_INTERVAL_SECS, default 900) already owns this
        // job, off the serving path.

        Ok(inserted)
    }

    /// The index algorithm a table is currently using, with the parameters it was built from.
    ///
    /// `Table::index_type` names the family; this carries the numbers — `m`, `ef_construction`,
    /// `ef_search`, partition counts — which is what anyone diagnosing a recall or latency
    /// question actually needs.
    pub async fn describe_index(&self, tablename: &str) -> Option<IndexAlgorithm> {
        let tablename = &*self.resolve(tablename);
        let tables = self.read_tables("describe_index").await;
        tables
            .get(tablename)
            .map(|data| data.selected_algorithm.clone())
    }

    /// 搜索向量
    /// Every entry point that takes a filter checks it here first. See [`Filter::validate`].
    fn check_filter(filter: Option<&crate::types::Filter>) -> crate::Result<()> {
        match filter {
            Some(filter) => filter.validate().map_err(crate::Error::InvalidInput),
            None => Ok(()),
        }
    }

    /// Refuse a filter this table's payload index does not cover.
    ///
    /// The index is the only thing that answers a filter here: a condition on a field with no
    /// postings yields an empty set, not a scan. That is right when no record has a value
    /// there and wrong when the field was deliberately left out of the index, and the two are
    /// indistinguishable from inside the index. So the policy is consulted before the index
    /// is, and an excluded field is an error naming the way back rather than an empty page
    /// that looks like an answer.
    ///
    /// Checked against the table's policy, not the built index, so it costs nothing on the
    /// default table and does not force a lazy index to be built in order to say no.
    fn check_filter_policy(
        table: &crate::types::table::Table,
        filter: Option<&crate::types::Filter>,
    ) -> crate::Result<()> {
        let (Some(policy), Some(filter)) = (&table.payload_index, filter) else {
            return Ok(());
        };
        match policy.unanswerable_field(filter) {
            None => Ok(()),
            Some(field) => Err(crate::Error::InvalidInput(format!(
                "table '{}' does not index payload field '{}', so a filter on it cannot be \
                 answered. Index it with PUT /tables/{}/payload_index/{} (the index is rebuilt \
                 from the records), or filter on an indexed field -- \
                 GET /tables/{}/payload_index lists them.",
                table.name, field, table.name, field, table.name
            ))),
        }
    }

    /// How many records hold each value of one payload field, optionally under a filter.
    ///
    /// The payload index already is this answer — it maps value to records — so a facet is a
    /// walk of postings rather than of rows. What it is not, and what the result says so, is a
    /// description of the whole table: an index-backed facet loses whatever the index does not
    /// hold as a *count of zero* rather than as an absence, which is the same failure the
    /// excluded-field refusal exists to prevent. `covered` against `matched` is the number that
    /// makes that visible, and the field being excluded is refused outright.
    pub async fn facet(
        &self,
        tablename: &str,
        key: &str,
        filter: Option<&crate::types::Filter>,
        limit: usize,
    ) -> crate::Result<crate::storage::payload_index::FacetResult> {
        let tablename = &*self.resolve(tablename);
        Self::check_filter(filter)?;
        self.ensure_resident(tablename).await?;
        self.ensure_derived(tablename).await?;
        let tables = self.read_tables("facet").await;
        let data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        Self::check_filter_policy(&data.table, filter)?;
        // The faceted field itself, not only the filter's fields. Counting from an index that
        // was told not to hold this field would answer every value with zero, which reads as
        // "the corpus has none of these" rather than as "this server cannot tell you".
        if let Some(policy) = &data.table.payload_index {
            if !policy.indexes(key) {
                return Err(crate::Error::InvalidInput(format!(
                    "table '{}' does not index payload field '{key}', so it cannot be faceted. \
                     Index it with PUT /tables/{}/payload_index/{key} -- the index is rebuilt \
                     from the records -- or facet a field GET /tables/{}/payload_index lists.",
                    data.table.name, data.table.name, data.table.name
                )));
            }
        }
        // Computed while the guard is held and returned by value: `derived()` borrows from
        // the table map, and a facet is a snapshot rather than a view of it.
        let result = data.derived().payload.facet(key, filter, limit);
        drop(tables);
        Ok(result)
    }

    pub async fn search(&self, tablename: &str, query: SearchQuery) -> crate::Result<SearchResult> {
        self.ensure_kb_table(&self.resolve(tablename)).await?;
        self.search_with_budget(tablename, query, None).await
    }

    /// `search`, with the traversal budget named by the caller instead of derived from the
    /// table. `None` is the ordinary path. Recall calibration is the caller that names one:
    /// it walks the same graph through the same code at a series of budgets, so what it
    /// measures is what a query at that budget would get.
    pub async fn search_with_budget(
        &self,
        tablename: &str,
        query: SearchQuery,
        ef_override: Option<usize>,
    ) -> crate::Result<SearchResult> {
        let tablename = &*self.resolve(tablename);
        let plan = self
            .rerank_plan(
                tablename,
                query.rerank.as_ref(),
                query.text.as_deref(),
                Self::page_size(&query),
                query.offset,
                "/search",
            )
            .await?;
        let Some(plan) = plan else {
            return self.search_unranked(tablename, query, ef_override).await;
        };
        let start = std::time::Instant::now();
        let want_payload = query.with_payload;
        // The pool is fetched as one page from offset zero and the caller's page is cut out
        // of the reranked order, so that page 2 is "the next ten by the provider's score" and
        // not "the provider's order of a different ten".
        let mut inner = query;
        inner.rerank = None;
        inner.offset = 0;
        inner.limit = plan.pool;
        inner.top_k = plan.pool;
        inner.with_payload = true;
        let fetched = self.search_unranked(tablename, inner, ef_override).await?;
        let mut hits = Self::rerank_hits(
            &plan,
            fetched.hits,
            |h| h.payload.as_ref(),
            |h, s| h.rerank_score = Some(s),
        )
        .await?;
        if !want_payload {
            for hit in &mut hits {
                hit.payload = None;
            }
        }
        Ok(SearchResult {
            total: hits.len(),
            hits,
            tookms: start.elapsed().as_millis() as u64,
        })
    }

    /// What a reranked search settles before anything is fetched.
    ///
    /// `None` when the request did not ask. Otherwise every reason the rerank could not
    /// happen is an error here, before the fetch: a search that quietly skipped the rerank
    /// would hand back a page that looks reranked and is not.
    async fn rerank_plan(
        &self,
        tablename: &str,
        request: Option<&crate::types::record::RerankRequest>,
        text: Option<&str>,
        page: usize,
        offset: usize,
        route: &str,
    ) -> crate::Result<Option<RerankPlan>> {
        let Some(request) = request else {
            return Ok(None);
        };
        let text = text
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .ok_or_else(|| {
                crate::Error::InvalidInput(format!(
                "`rerank` scores candidates against the query's text, and this {route} request \
                 has none. Send `text` (the table embeds it), or drop `rerank`."
            ))
            })?;
        self.ensure_resident(tablename).await?;
        let table = self
            .read_tables("rerank_plan")
            .await
            .get(tablename)
            .map(|data| data.table.clone())
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        let config = table.rerank.clone().ok_or_else(|| {
            crate::Error::InvalidInput(format!(
                "table '{tablename}' has no rerank provider. Create it with a \"rerank\" config, \
                 or add one with PATCH /tables/{tablename}."
            ))
        })?;
        let field = request
            .field
            .clone()
            .or_else(|| config.field.clone())
            .or_else(|| table.embedding.as_ref().map(|e| e.field().to_string()))
            .unwrap_or_else(|| "text".to_string());
        let pool = request.pool(config.top_n, page, offset);
        Ok(Some(RerankPlan {
            config,
            query: text.to_string(),
            field,
            pool,
            page,
            offset,
        }))
    }

    /// Reorder `hits` by the provider's score and cut the caller's page out of the result.
    ///
    /// A hit with no text in the rerank field is not sent -- a provider given an empty string
    /// either refuses the call or scores it arbitrarily -- and trails the scored hits in its
    /// original order with no `rerank_score`. The provider is asked for `page + offset`
    /// results, the most that can be shown, rather than the whole pool.
    async fn rerank_hits<H>(
        plan: &RerankPlan,
        hits: Vec<H>,
        payload_of: impl Fn(&H) -> Option<&Payload>,
        set_score: impl Fn(&mut H, f32),
    ) -> crate::Result<Vec<H>> {
        let mut documents: Vec<String> = Vec::with_capacity(hits.len());
        let mut hit_of_document: Vec<usize> = Vec::with_capacity(hits.len());
        for (i, hit) in hits.iter().enumerate() {
            let text = payload_of(hit)
                .and_then(|p| p.get(&plan.field))
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|t| !t.is_empty());
            if let Some(text) = text {
                documents.push(text.to_string());
                hit_of_document.push(i);
            }
        }
        let want = plan.page.saturating_add(plan.offset);
        let scored = Self::rerank_call(
            &plan.config,
            &plan.query,
            &documents,
            want.min(documents.len()),
        )
        .await?;
        let mut order: Vec<(usize, Option<f32>)> = scored
            .into_iter()
            .map(|(d, score)| (hit_of_document[d], Some(score)))
            .collect();
        let mut placed = vec![false; hits.len()];
        for (i, _) in &order {
            placed[*i] = true;
        }
        order.extend((0..hits.len()).filter(|i| !placed[*i]).map(|i| (i, None)));
        let mut slots: Vec<Option<H>> = hits.into_iter().map(Some).collect();
        let mut page = Vec::with_capacity(plan.page);
        for (i, score) in order.into_iter().skip(plan.offset).take(plan.page) {
            let Some(mut hit) = slots[i].take() else {
                continue;
            };
            if let Some(score) = score {
                set_score(&mut hit, score);
            }
            page.push(hit);
        }
        Ok(page)
    }

    /// Call the table's rerank provider, or say which build cannot. The `cfg` is here and
    /// nowhere else, as for [`Self::embed`].
    async fn rerank_call(
        config: &crate::types::table::RerankConfig,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> crate::Result<Vec<(usize, f32)>> {
        #[cfg(feature = "server-embedding")]
        {
            crate::embeddings::rerank::rerank(config, query, documents, top_n).await
        }
        #[cfg(not(feature = "server-embedding"))]
        {
            let _ = (config, query, documents, top_n);
            Err(crate::Error::NotImplemented(
                "this build has no rerank support: rebuild with --features server-embedding, \
                 or drop `rerank` from the request"
                    .to_string(),
            ))
        }
    }

    /// `search_with_budget` without the rerank step.
    async fn search_unranked(
        &self,
        tablename: &str,
        query: SearchQuery,
        ef_override: Option<usize>,
    ) -> crate::Result<SearchResult> {
        Self::check_filter(query.filter.as_ref())?;
        self.ensure_resident(tablename).await?;
        if query.filter.is_some() {
            self.ensure_derived(tablename).await?;
        }
        let query = self.embed_query(tablename, query).await?;
        if query.filter.is_some() {
            let tables = self.read_tables("search").await;
            if let Some(data) = tables.get(tablename) {
                Self::check_filter_policy(&data.table, query.filter.as_ref())?;
            }
        }
        let start_time = std::time::Instant::now();
        // 优先使用 limit，如果 limit 未设置或为默认值则使用 top_k
        // Python SDK 发送 "limit" 字段，Rust API 使用 "top_k"
        let page = Self::page_size(&query);
        // Everything below fetches, scores and truncates to `result_limit`; the page is cut out
        // of that prefix in `assemble_hits`. An approximate index cannot start in the middle of
        // a ranking it has not computed, so an offset is paid for by computing the prefix --
        // which is what the field's documentation says it costs.
        let result_limit = match &query.mmr {
            // Diversity can only come from candidates that were fetched, so the index is asked
            // for `fetch_k` and the page is chosen out of it in `assemble_hits`.
            Some(mmr) => mmr.fetch_k.max(page.saturating_add(query.offset)),
            None => page.saturating_add(query.offset),
        };

        // `Exact` is not a setting of the machinery below -- it is the absence of it. Every
        // branch from here on picks an `ef`, a candidate count, or a filter strategy, and all
        // three are ways of deciding what *not* to look at. Diverting before any of that is
        // what keeps this mode's one claim true: it looked at everything.
        if query.search_mode.is_exhaustive() {
            let (results, metric) = self.exact_scan(tablename, &query, result_limit).await?;
            let hits = self
                .assemble_hits(tablename, &query, results, false, metric, result_limit)
                .await?;
            let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            if let Some(table_data) = self.read_tables("search").await.get(tablename) {
                if let Ok(mut stats) = table_data.performance_stats.lock() {
                    stats.total_queries += 1;
                    stats.avg_query_latency_ms =
                        (stats.avg_query_latency_ms * 0.9) + (latency_ms * 0.1);
                }
            }
            let total = hits.len();
            return Ok(SearchResult {
                hits,
                total,
                tookms: latency_ms as u64,
            });
        }

        // Phase 1: take everything out of the table map that the search needs, then let go of it.
        //
        // This used to hold the map's read lock for the whole function, index search included,
        // and that is enough to stop the entire server. A brute-force search over 60,000 vectors
        // takes about a second; three concurrent readers overlap so the read lock is never free;
        // the insertion processor queues for the write lock and starves; and because tokio's
        // `RwLock` is write-preferring, every later read queues behind that pending writer. Under
        // a forced rebuild the measured effect was read p99 of 25.6s and write max of 27.1s,
        // exactly the length of the rebuild — with no single lock holder over one second, because
        // the holders were one second each, back to back. The fix is not to hold a global lock
        // for a search at all — take what the search needs out of the map, then release it.
        let filter_clock = std::time::Instant::now();
        let mut filter_branch: Option<usize> = None;
        let mut filter_eval_ns = 0u64;
        let (
            index,
            metric,
            configured_ef,
            total_records,
            cardinality_and_sel,
            precomputed_filter_ids,
            mut exact_scored,
            node_filter,
            walk_filter,
            matched_rows,
        ) = {
            let tables = self.read_tables("search").await;

            let table_data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

            // The query's width against the space that will answer it: the named space's when
            // one is named and exists, else the table's (a name the table does not have falls
            // through to the default index below). This skipped every query naming a space,
            // and the index does not check either, so a short vector reached the SIMD kernels,
            // which read `a.len()` floats from both sides -- past the end of the query.
            let expected_dim = query
                .vector_name
                .as_ref()
                .and_then(|name| {
                    table_data
                        .table
                        .named_vectors
                        .as_ref()?
                        .iter()
                        .find(|space| &space.name == name)
                })
                .map_or(table_data.table.dimension, |space| space.dimension);
            if !query.vector.is_empty() && query.vector.len() != expected_dim {
                return Err(crate::Error::DimensionMismatch {
                    expected: expected_dim,
                    actual: query.vector.len(),
                });
            }

            // Resolve named index (if vector_name is specified in the query).
            // get_index() returns an owned clone of the IndexWrapper (Arc-backed, cheap clone).
            let named_index_opt: Option<crate::search::IndexWrapper> =
                if let Some(ref vec_name) = query.vector_name {
                    if let Some(ref named_indexes) = table_data.named_indexes {
                        named_indexes.get_index(vec_name).await
                    } else {
                        None
                    }
                } else {
                    None
                };
            // An `IndexWrapper` is Arc-backed, so this clone costs a refcount and buys the
            // whole point of the phase split: the search below runs without the map lock.
            let answered_by_named_space = named_index_opt.is_some();
            let index = named_index_opt.unwrap_or_else(|| table_data.index.clone());
            // The metric of whichever index is about to answer, which is not the table's when
            // a named space answered. `similarity_from_distance` turns a distance into a
            // score with a formula *per metric*, so using the table's here silently scored a
            // named space with the wrong one: a `dot` space under a `cosine` table came back
            // `1 - (-dot)`, above 1 for every hit and clamped flat to 1.0, and a `euclidean`
            // space came back `1 - d` instead of `1 / (1 + d)`. The order survived, because
            // the index sorts by distance before any of this, which is exactly why it went
            // unnoticed -- the ranking looked right and every score was wrong.
            let metric = query
                .vector_name
                .as_ref()
                .filter(|_| answered_by_named_space)
                .and_then(|name| {
                    table_data
                        .table
                        .named_vectors
                        .as_ref()?
                        .iter()
                        .find(|space| &space.name == name)
                })
                .map_or(table_data.table.distance_metric, |space| {
                    space.distance_metric
                });
            let total_records = table_data.records.len();

            // Intelligent PRE/POST-FILTERING / IN-GRAPH decision
            let eval_clock = std::time::Instant::now();
            let cardinality_and_sel = if let Some(ref filter) = query.filter {
                let cardinality = table_data.derived().payload.estimate_cardinality(filter);
                let selectivity = if total_records > 0 {
                    cardinality.exp as f64 / total_records as f64
                } else {
                    1.0
                };
                Some((cardinality, selectivity))
            } else {
                None
            };

            // The in-graph filter branch needs the matching ids, and its small-set special case
            // needs those records' vectors. Both are cheap and both need the map, so they happen
            // here rather than forcing the search to hold the lock for them.
            let mut precomputed_filter_ids: Option<std::collections::HashSet<RecordId>> = None;
            let mut exact_scored = None;
            let mut node_filter: Option<crate::search::node_filter::NodeFilter> = None;
            let mut walk_filter: Option<(
                Arc<tokio::sync::RwLock<crate::storage::node_map::NodeMap>>,
                roaring::RoaringBitmap,
            )> = None;
            let mut matched_rows: Option<usize> = None;
            if let (Some(filter), Some((cardinality, selectivity))) =
                (&query.filter, &cardinality_and_sel)
            {
                // The matching ids are computed for every selective filter, not only the ones
                // small enough for the in-graph walk: the pre-filtering branch below uses
                // them to drop non-matching candidates *before* the exact rerank. Without
                // that, a 1%-selectivity query fetched 30,000 candidates and reranked all of
                // them -- 30,000 vector reads per query, 480 ms p99 with the vectors on disk.
                // Before anything is searched: the index can often prove the filter matches
                // no record at all, by walking its fields' distinct values rather than the
                // table's rows. Answering that here rather than in the post-filter fallback
                // is the difference between skipping the whole query and skipping only its
                // last step -- the fallback is reached after an initial fetch and at least
                // one escalating refetch, and each of those reranks thousands of candidates
                // against their stored vectors to discover what this already knows.
                let derived = table_data.derived();
                if derived.payload.cannot_match_anything(filter) {
                    exact_scored = Some(Vec::new());
                } else if *selectivity < Self::FILTER_MATERIALISE_BELOW {
                    // The matching ids, borrowed from the payload index, and their count --
                    // which is the real selectivity, where `selectivity` above is an estimate
                    // that treats AND-ed conditions as independent and is off by a lot when
                    // they are not (co-occurring tags). Every decision below uses the count.
                    let matched_set = derived.payload.matched_bitmap(filter);
                    let m = matched_set.len() as usize;
                    matched_rows = Some(m);
                    // Exact scoring over the candidate set, when the set is small enough that
                    // doing so beats walking the graph. `result_limit * 4` alone made that
                    // threshold depend on the caller's top_k rather than on the work involved:
                    // a top-10 query over 400 candidates fell through to a filtered traversal
                    // that measured 24 ms p50, where scoring 400 vectors takes microseconds and
                    // is exact rather than approximate. The absolute floor is what matters —
                    // a filtered walk visits on the order of `ef * m` nodes anyway.
                    // With the vectors on disk a row read is a page fault when the file is
                    // not in cache -- 653 rows took 91 ms a query on a 7.7 GB file at 10M --
                    // so there the direct scan stops at about the rerank pool, which the code
                    // scan reads anyway; in memory it is cheap up to the old ceiling.
                    let exact_up_to = if table_data.vector_file.is_some() {
                        (result_limit * 4).max(Self::EXACT_FILTER_SCAN_ON_DISK)
                    } else {
                        (result_limit * 4).max(Self::EXACT_FILTER_SCAN_MAX)
                    };
                    if m <= exact_up_to {
                        let mut scored: Vec<(RecordId, f32)> = matched_set
                            .iter()
                            .filter_map(|ordinal| derived.payload.id_at(ordinal))
                            .filter_map(|id| {
                                table_data.records.get(id).map(|rec| {
                                    // `vector_for`, not `rec.vector`: with the vectors on
                                    // disk the record's own field is empty.
                                    let dist = metric
                                        .calculate(&query.vector, &table_data.vector_for(&rec))
                                        .unwrap_or(f32::MAX);
                                    (id.to_string(), dist)
                                })
                            })
                            .collect();
                        // A total order before the truncate below, for the same reason the
                        // filtered path above sorts by id on ties: equal distances would
                        // otherwise be cut in whatever order the ids arrived in.
                        scored.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.0.cmp(&b.0))
                        });
                        scored.truncate(result_limit);
                        self.filter_path_exact_scan.fetch_add(1, Ordering::Relaxed);
                        filter_branch = Some(0);
                        debug!(
                            "FILTER-EXACT-SCAN: selectivity={:.2}%, candidates={m}",
                            selectivity * 100.0
                        );
                        exact_scored = Some(scored);
                    } else {
                        // Cross from payload ordinals to node positions through the table's
                        // node map, filled here for any node it has not seen (one hash each,
                        // once per node ever). A small match set becomes a bitmap of nodes to
                        // score; a large one stays a bitmap of ordinals the walk tests nodes
                        // against. An index without positions, or one a rebuild is swapping
                        // this instant, gets the owned id set it always had.
                        let mut resolved = false;
                        if m > Self::in_graph_filter_max_ids() {
                            // Capped off (a test switch): the amplified fetch below, which
                            // keeps the match set only to drop non-matching candidates.
                            resolved = true;
                            precomputed_filter_ids = Some(
                                matched_set
                                    .iter()
                                    .filter_map(|ordinal| derived.payload.id_at(ordinal))
                                    .map(str::to_string)
                                    .collect(),
                            );
                        } else if let Some(graph) = index.try_read() {
                            // Readers share the map; the write lock is taken only when a fill
                            // is due (new nodes, or a rebuilt graph). Taking it every time
                            // made each query wait on every walk in flight, and the 16-thread
                            // pass on 10M rows ran at 6 QPS against 84 for one thread.
                            let generation = index.generation();
                            let slots = graph.slot_count();
                            let due = table_data
                                .node_map
                                .try_read()
                                .map(|m| m.len() < slots || !m.is_for(generation))
                                .unwrap_or(true);
                            if due {
                                if let Ok(mut map) = table_data.node_map.try_write() {
                                    let payload = &derived.payload;
                                    map.fill(generation, &**graph, |id| payload.ordinal_of(id));
                                }
                            }
                            if let Ok(map) = table_data.node_map.try_read() {
                                if map.is_for(generation) && map.len() >= slots && slots > 0 {
                                    // Measured on YFCC-10M with the block-scan codes: above
                                    // this line the walk is still the cheaper answer (14.7 ms
                                    // at a million matches against a 20 ms scan of every
                                    // block), so the scan keeps its ceiling and the codes
                                    // only change what the scan costs under it.
                                    if m <= Self::code_scan_max(total_records) {
                                        let mut nodes =
                                            crate::search::node_filter::NodeFilter::with_capacity(
                                                map.len(),
                                            );
                                        for ordinal in matched_set.iter() {
                                            if let Some(node) = map.node_of(ordinal) {
                                                nodes.insert(node);
                                            }
                                        }
                                        node_filter = Some(nodes);
                                    } else {
                                        walk_filter = Some((
                                            Arc::clone(&table_data.node_map),
                                            matched_set.clone(),
                                        ));
                                    }
                                    resolved = true;
                                }
                            }
                        }
                        if !resolved {
                            self.filter_node_map_misses.fetch_add(1, Ordering::Relaxed);
                            precomputed_filter_ids = Some(
                                matched_set
                                    .iter()
                                    .filter_map(|ordinal| derived.payload.id_at(ordinal))
                                    .map(str::to_string)
                                    .collect(),
                            );
                        }
                    }
                }
            }
            if query.filter.is_some() {
                filter_eval_ns = eval_clock.elapsed().as_nanos() as u64;
            }

            (
                index,
                metric,
                // A named vector space has its own index; the table's algorithm describes the
                // default one, so only claim its `ef_search` when that is what is being searched.
                if query.vector_name.is_none() {
                    // The calibrated budget, when there is one, is the table's own answer
                    // to "what does this corpus need"; it only ever raises the configured
                    // value, so a table whose configured budget is already enough keeps it.
                    let configured = table_data.selected_algorithm.configured_ef_search();
                    match (&table_data.table.recall, configured) {
                        (Some(cal), Some(c)) => Some(cal.ef_search.max(c)),
                        (Some(cal), None) => Some(cal.ef_search),
                        (None, c) => c,
                    }
                } else {
                    None
                },
                total_records,
                cardinality_and_sel,
                precomputed_filter_ids,
                exact_scored,
                node_filter,
                walk_filter,
                matched_rows,
            )
        };
        let active_index = &index;

        // The index's own `ef_search`, not a constant. This was hardcoded to 100, which made the
        // selector's `ef_search` inert — it reached `HnswConfig` at build time and was then
        // ignored on every query, so tuning it moved nothing. 100 remains the fallback for
        // indexes that have no such parameter (brute force, DiskANN).
        // Scaled by the table's size, because `ef` is a budget against a graph whose diameter
        // grows with it; see `ef_for_rows`. Read here rather than at create time because the
        // size a table is created with (10,000 rows, an estimate) is not the size it is
        // searched at.
        let default_ef = match ef_override {
            Some(ef) => ef,
            None => Self::ef_for_rows(configured_ef.unwrap_or(100), total_records),
        };
        // Use adaptive EF when SearchMode::Auto and a filter is present
        let ef_hint = if let Some((_, selectivity)) = &cardinality_and_sel {
            query
                .search_mode
                .ef_search_adaptive(default_ef, result_limit, *selectivity)
        } else {
            query.search_mode.ef_search(default_ef, result_limit)
        };

        // Determine strategy and execute search
        let (results, post_filter, fetched) =
            if let (Some(_), Some((ref cardinality, selectivity))) =
                (&query.filter, &cardinality_and_sel)
            {
                if let (None, Some(filter)) = (&exact_scored, node_filter.as_ref()) {
                    // A small match set, resolved to node positions: score every one of
                    // them from the resident codes -- no walk -- and rerank the best exactly.
                    let want = active_index
                        .rerank_candidates(result_limit)
                        .await
                        .unwrap_or(result_limit)
                        .max(result_limit);
                    match active_index.scan_nodes(&query.vector, filter, want).await {
                        Some(scanned) => {
                            self.filter_path_code_scan.fetch_add(1, Ordering::Relaxed);
                            filter_branch = Some(1);
                            debug!(
                                "FILTER-CODE-SCAN: matched={}, kept={}",
                                filter.len(),
                                scanned.len()
                            );
                            let r = self
                                .rerank_exact(
                                    tablename,
                                    metric,
                                    &query.vector,
                                    scanned,
                                    result_limit,
                                )
                                .await?;
                            exact_scored = Some(r);
                        }
                        None => {
                            // An exact graph has no codes to scan; walk it with the filter.
                            self.filter_path_in_graph.fetch_add(1, Ordering::Relaxed);
                            filter_branch = Some(2);
                            let gate: &dyn crate::search::node_filter::NodeGate = filter;
                            let r = self
                                .search_reranked(
                                    tablename,
                                    active_index,
                                    metric,
                                    &query.vector,
                                    result_limit,
                                    ef_hint,
                                    query.search_mode,
                                    None,
                                    None,
                                    Some(gate),
                                )
                                .await?;
                            exact_scored = Some(r);
                        }
                    }
                }
                if let (None, Some((map, matched))) = (&exact_scored, walk_filter.as_ref()) {
                    // A large match set: the walk, with every node it visits tested against
                    // the matched ordinals through the node map.
                    self.filter_path_in_graph.fetch_add(1, Ordering::Relaxed);
                    filter_branch = Some(2);
                    debug!("FILTER-IN-GRAPH: matched={}, ef={ef_hint}", matched.len());
                    let map = map.read().await;
                    let gate = crate::storage::node_map::OrdinalGate { map: &map, matched };
                    // The walk only runs where the match set is large (above `code_scan_max`, 8% of a
                    // 2M table, 3.5% of a 10M one), and there the matching nodes are dense enough in
                    // the graph that a quarter of the budget finds the same neighbours: measured on
                    // the big-ann filtered track at 2M rows, 234,000 matched, recall@10 1.0000 against
                    // the exact scan at the full budget, at a half and at a quarter -- 15.6, 9.0 and
                    // 5.3 ms. The floor keeps the rerank pool full on a small `ef`.
                    let walk_ef = (ef_hint / 4).max(result_limit * 20);
                    let r = self
                        .search_reranked(
                            tablename,
                            active_index,
                            metric,
                            &query.vector,
                            result_limit,
                            walk_ef,
                            query.search_mode,
                            None,
                            None,
                            Some(&gate),
                        )
                        .await?;
                    exact_scored = Some(r);
                }
                if exact_scored.is_some() {
                    // answered above
                } else if exact_scored.is_none()
                    && *selectivity < 0.05
                    && precomputed_filter_ids
                        .as_ref()
                        .is_some_and(|ids| ids.len() <= Self::CODE_SCAN_MAX)
                {
                    // A selective filter over a quantized graph: score the matching ids from
                    // the resident codes -- one ADC lookup each -- and rerank the best of
                    // them exactly. No walk. The 30,000-wide walk this replaces measured
                    // 43 QPS at 480 ms p99 on the 1%-selectivity 1M case with the vectors
                    // on disk; 10,000 code scores take a few milliseconds.
                    // Borrowed. `precomputed_filter_ids` already owns them, and cloning a
                    // second copy to hand to the scan was 100,000 heap allocations a query on
                    // the 10M case -- for strings the scan only ever reads.
                    let ids: Vec<&str> = precomputed_filter_ids
                        .as_ref()
                        .map(|s| s.iter().map(String::as_str).collect())
                        .unwrap_or_default();
                    let want = active_index
                        .rerank_candidates(result_limit)
                        .await
                        .unwrap_or(result_limit)
                        .max(result_limit);
                    if let Some(scanned) = active_index.scan_ids(&query.vector, &ids, want).await {
                        self.filter_path_code_scan.fetch_add(1, Ordering::Relaxed);
                        filter_branch = Some(1);
                        info!(
                            "FILTER-CODE-SCAN: selectivity={:.2}%, candidates={}, kept={}",
                            selectivity * 100.0,
                            ids.len(),
                            scanned.len()
                        );
                        let r = self
                            .rerank_exact(tablename, metric, &query.vector, scanned, result_limit)
                            .await?;
                        exact_scored = Some(r);
                    }
                }
                if let Some(scored) = exact_scored.take() {
                    (scored, false, result_limit)
                } else if *selectivity < 0.05 && cardinality.exp < Self::in_graph_filter_max_ids() {
                    // Acorn-style in-graph filter: selectivity < 5%. Both the matching ids and,
                    // for a small enough set, their exact scores were computed in phase 1 while
                    // the map lock was held.
                    let empty = std::collections::HashSet::new();
                    let filter_ids = precomputed_filter_ids.as_ref().unwrap_or(&empty);
                    self.filter_path_in_graph.fetch_add(1, Ordering::Relaxed);
                    filter_branch = Some(2);
                    info!(
                        "FILTER-IN-GRAPH: selectivity={:.2}%, candidates={}",
                        selectivity * 100.0,
                        filter_ids.len()
                    );

                    // When filter_ids is very small, kNN approximation is unreliable:
                    // fetching top-K from the full index may miss all filter candidates.
                    // Instead, exact distances over just the matching IDs (O(|filter_ids|)).
                    // `ef_hint` as-is, deliberately.
                    //
                    // Scaling it by 1/selectivity looks right and is not: in `search_layer`
                    // the filter gates the *result* set, so an `ef` of N means the walk keeps
                    // going until it holds N matching nodes, which at selectivity s is about
                    // N/s visits. Scaling N by 1/s multiplies the work by 1/s a second time.
                    // Measured on Performance768D1M99P (1% of rows pass): with the scaling,
                    // ef 8,192 meant walking most of a million-node graph -- 6.4 QPS and a
                    // 7,986 ms p99, against the 76.9 QPS and 245 ms of the unfiltered-fetch
                    // path it replaced. Unscaled, the walk visits roughly ef/s nodes, which is
                    // the budget the amplified fetch was spending anyway.
                    let filtered_ef = ef_hint;
                    let r = match exact_scored {
                        Some(scored) => scored,
                        None => {
                            self.search_reranked(
                                tablename,
                                active_index,
                                metric,
                                &query.vector,
                                result_limit,
                                filtered_ef,
                                query.search_mode,
                                Some(filter_ids),
                                None,
                                None,
                            )
                            .await?
                        }
                    };
                    (r, false, result_limit)
                } else if *selectivity < 0.3 {
                    // PRE-FILTERING: still fetch amplified and post-filter.
                    //
                    // The safety factor is what decides whether a request comes back short.
                    // Fetching `limit/selectivity` yields `limit` survivors *on average*, so half
                    // the time fewer. At 2.0 and 5% selectivity a top-10 fetches 400, expects 20
                    // survivors, and falls under 10 roughly once in a hundred requests; at 3.0 that
                    // is about four in a million.
                    let safety = 3.0;
                    let fetch = ((result_limit as f64 / selectivity * safety).ceil() as usize)
                        .min(total_records)
                        .max(result_limit * 2);
                    self.filter_path_pre_filter.fetch_add(1, Ordering::Relaxed);
                    filter_branch = Some(3);
                    info!(
                        "PRE-FILTERING: selectivity={:.2}%, fetch={}",
                        selectivity * 100.0,
                        fetch
                    );
                    let r = self
                        .search_reranked(
                            tablename,
                            active_index,
                            metric,
                            &query.vector,
                            fetch,
                            ef_hint,
                            query.search_mode,
                            None,
                            precomputed_filter_ids.as_ref(),
                            None,
                        )
                        .await?;
                    (r, true, fetch)
                } else {
                    // POST-FILTERING: selectivity >= 30%, fetch extra and filter.
                    //
                    // Scaled by selectivity, as the pre-filtering branch above already does. A flat
                    // `result_limit * 3` ignores how many of the fetched rows the filter will throw
                    // away: at 50% selectivity it fetches 30 for a top-10, and the number that
                    // survive is a coin toss repeated 30 times — under ten about 2% of the time, so
                    // roughly one request in fifty silently returned short. That is what made
                    // `test_filter_with_in_operation` fail intermittently; the test was reporting a
                    // real defect, not being flaky.
                    //
                    // Still not a guarantee — see the retry below, which is.
                    let safety = 3.0;
                    let fetch = ((result_limit as f64 / selectivity * safety).ceil() as usize)
                        .min(total_records)
                        .max(result_limit * 3);
                    self.filter_path_post_filter.fetch_add(1, Ordering::Relaxed);
                    filter_branch = Some(4);
                    info!(
                        "POST-FILTERING: selectivity={:.2}%, fetch={}",
                        selectivity * 100.0,
                        fetch
                    );
                    let r = self
                        .search_reranked(
                            tablename,
                            active_index,
                            metric,
                            &query.vector,
                            fetch,
                            ef_hint,
                            query.search_mode,
                            None,
                            None,
                            None,
                        )
                        .await?;
                    (r, true, fetch)
                }
            } else {
                // No filter
                let r = self
                    .search_reranked(
                        tablename,
                        active_index,
                        metric,
                        &query.vector,
                        result_limit,
                        ef_hint,
                        query.search_mode,
                        None,
                        None,
                        None,
                    )
                    .await?;
                (r, false, result_limit)
            };

        if let Some(branch) = filter_branch {
            self.filter_path_ns[branch]
                .fetch_add(filter_clock.elapsed().as_nanos() as u64, Ordering::Relaxed);
            self.filter_eval_ns
                .fetch_add(filter_eval_ns, Ordering::Relaxed);
        }

        // Phase 3: assemble, and if post-filtering left the caller short, fetch more.
        //
        // The amplified fetch above is sized from an *estimated* selectivity, and some estimates
        // are not estimates at all: `Contains` is hardcoded to 0.25 and `EndsWith` to 0.15
        // regardless of the data. A `Contains` whose true selectivity is 1% therefore fetched
        // 240 rows for a top-20 and returned 3. Amplification cannot fix that — no constant can,
        // because the error is in the input. Refetching when the answer comes back short can,
        // and it is bounded: each attempt multiplies the fetch, and the last one asks for every
        // record in the table, after which nothing more exists to find.
        let mut fetched = fetched;
        let mut results = results;
        let mut hits = self
            .assemble_hits(
                tablename,
                &query,
                results,
                post_filter,
                metric,
                result_limit,
            )
            .await?;
        let mut attempts = 0;
        while post_filter
            && hits.len() < result_limit
            && fetched < total_records
            && attempts < Self::POST_FILTER_REFETCH_ATTEMPTS
        {
            attempts += 1;
            // The last attempt, or the first one that would already span the table.
            //
            // What decides that is the fetch the index will actually make, not the number
            // asked for: `search_reranked` widens `k` to the rerank pool, so a 4,800-candidate
            // refetch reaches for 24,000 and covers a 20,000-row table completely. A graph
            // walk that visits everything cannot find a matching row the scan below would
            // miss, so there is no reason to pay for it first.
            let next = fetched.saturating_mul(8);
            let widened = active_index
                .rerank_candidates(next)
                .await
                .unwrap_or(next)
                .max(next);
            if attempts == Self::POST_FILTER_REFETCH_ATTEMPTS || widened >= total_records {
                // The exhaustive answer, enumerated by the predicate instead of by the graph.
                //
                // This step used to set `fetched = total_records` and go back through
                // `search_reranked`, which multiplies whatever `k` it is handed by the rerank
                // factor: a 5M-row table was asked for 25M results at ef 25M, and every
                // candidate that came back was reranked against its stored vector. That is
                // where the 16.6 s empty answer in `docs/architecture/disk-index.md` §5 came from,
                // 4.6 s of it holding the table read lock, against 3.6 ms for the same query
                // unfiltered. At `ef = n` the graph has stopped being an index -- the walk
                // reaches every node and carries an n-sized heap -- so it is strictly more
                // work than the scan it is imitating.
                //
                // The condition that brings us here bounds what the scan then costs: three
                // escalating fetches failed to find `result_limit` matching rows, so the
                // filter is highly selective and few rows survive it. A filter that matches
                // nothing reads no vectors at all.
                if let Some(filter) = &query.filter {
                    debug!(
                        "POST-FILTER exhausted the graph ({} < {}), scanning {} records by                          the predicate instead",
                        hits.len(),
                        result_limit,
                        total_records
                    );
                    hits = self
                        .exhaustive_post_filter(tablename, &query, filter, metric, result_limit)
                        .await?;
                }
                break;
            }
            fetched = next.min(total_records);
            debug!(
                "POST-FILTER short ({} < {}), refetching {} of {} records (attempt {})",
                hits.len(),
                result_limit,
                fetched,
                total_records,
                attempts
            );
            results = self
                .search_reranked(
                    tablename,
                    active_index,
                    metric,
                    &query.vector,
                    fetched,
                    ef_hint.max(fetched),
                    query.search_mode,
                    None,
                    precomputed_filter_ids.as_ref(),
                    None,
                )
                .await?;
            hits = self
                .assemble_hits(
                    tablename,
                    &query,
                    results,
                    post_filter,
                    metric,
                    result_limit,
                )
                .await?;
        }

        let tables = self.read_tables("search").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let query_time = start_time.elapsed();
        let latency_ms = query_time.as_secs_f64() * 1000.0;

        // 在读锁持有期间通过 Mutex 并发更新查询统计
        if let Ok(mut stats) = table_data.performance_stats.lock() {
            stats.total_queries += 1;
            // 指数移动平均：90% 历史 + 10% 新值
            stats.avg_query_latency_ms = (stats.avg_query_latency_ms * 0.9) + (latency_ms * 0.1);
        }

        let total = hits.len();
        Ok(SearchResult {
            hits,
            total,
            tookms: latency_ms as u64,
        })
    }

    /// Score every live row against the query. The implementation of `SearchMode::Exact`.
    ///
    /// Returns `(id, distance)` pairs -- unsorted and already truncated per chunk; the caller
    /// hands them to `assemble_hits`, which owns the total order and the score conversion.
    ///
    /// Three things this has to get right, and each one is a lesson already paid for elsewhere
    /// in this file:
    ///
    /// * **The map lock is not held across the scan.** `search`'s phase split exists because a
    ///   brute-force pass over 60,000 vectors takes about a second, and holding the table map's
    ///   read lock for it starves the insertion processor -- measured at 25.6 s read p99 during
    ///   a rebuild. A full pass over a million rows is worse. So phase 1 here takes out the row
    ///   list and the file handle, and phase 2 runs with nothing held.
    /// * **It does not run on a runtime worker.** This is CPU-bound and synchronous, the same
    ///   shape as PQ training, which blocked every insert until it moved to `spawn_blocking`
    ///   with a pool of its own (`IndexWrapper::quantize_now`). Same treatment: half the cores,
    ///   so a diagnostic query cannot monopolise the machine it is diagnosing.
    /// * **It reads in file order.** `vectors_for_index` already established that reading rows
    ///   sorted and chunked beats one seek per record; the chunks here are the unit of
    ///   parallelism as well, which is why they carry their own partial top-k.
    async fn exact_scan(
        &self,
        tablename: &str,
        query: &SearchQuery,
        result_limit: usize,
    ) -> crate::Result<(Vec<(RecordId, f32)>, DistanceMetric)> {
        /// Rows per chunk: the unit of both the sequential read and the parallel split.
        /// 4,096 is what `vectors_for_index` reads at, and at 768 dimensions it is 12 MB of
        /// vectors per chunk -- big enough that the prefetch is worth its syscall, small enough
        /// that a machine with 32 of them in flight is not holding a gigabyte of mappings.
        const CHUNK: usize = 4_096;

        // Phase 1, under the map lock: take what the scan needs, then let go.
        let (metric, file, by_row, resident) = {
            let tables = self.read_tables("exact_scan").await;
            let table_data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

            // A named vector space has its own file and its own dimension, and resolving which
            // is which here would duplicate the named-index plumbing for a diagnostic. Say so
            // rather than silently scanning the default space and calling the answer exact.
            if query.vector_name.is_some() {
                return Err(crate::Error::InvalidInput(
                    "search_mode \"exact\" does not support named vector spaces; omit \
                     vector_name, or use one of fast/normal/accurate/auto"
                        .to_string(),
                ));
            }

            let expected_dim = table_data.table.dimension;
            if !query.vector.is_empty() && query.vector.len() != expected_dim {
                return Err(crate::Error::DimensionMismatch {
                    expected: expected_dim,
                    actual: query.vector.len(),
                });
            }

            // Pre-filter, so the cap is measured against what will actually be read rather than
            // against the table. A selective filter is exactly when an exact answer is cheap.
            let allowed = query
                .filter
                .as_ref()
                .map(|f| table_data.derived().payload.get_matching_ids(f));

            let candidates = match &allowed {
                Some(ids) => ids.len(),
                None => table_data.records.len(),
            };
            let cap = Self::exact_scan_max();
            if candidates > cap {
                return Err(crate::Error::InvalidInput(format!(
                    "search_mode \"exact\" would have to score {candidates} rows, over the \
                     {cap}-row limit. Narrow it with a filter, raise DXMT_EXACT_SCAN_MAX, or use \
                     search_mode \"accurate\""
                )));
            }

            // Two shapes, because a table's vectors are either in a row file or inside the
            // records, and a table that predates its row file has some of both.
            let mut by_row: Vec<(u64, RecordId)> = Vec::new();
            let mut resident: Vec<Arc<Record>> = Vec::new();
            let keep = |id: &RecordId| allowed.as_ref().is_none_or(|set| set.contains(id));

            if table_data.vector_file.is_some() {
                for id in table_data.records.ids() {
                    if !keep(&id) {
                        continue;
                    }
                    match table_data.vector_row(&id) {
                        Some(row) => by_row.push((row, id)),
                        // No row: an older record still carrying its own vector.
                        None => {
                            if let Some(r) = table_data.records.get(&id) {
                                if !r.vector.is_empty() {
                                    resident.push(r);
                                }
                            }
                        }
                    }
                }
                // File order. The point of the chunking is lost if the chunks are scattered.
                by_row.sort_unstable();
            } else {
                // `Arc<Record>`, not `Record`: cloning the records themselves is the 90 GB
                // temporary that got a 10M-row snapshot killed by the kernel. Eight bytes each.
                table_data.records.for_each(|r| {
                    if keep(&r.id) && !r.vector.is_empty() {
                        resident.push(r);
                    }
                });
            }

            (
                table_data.table.distance_metric,
                table_data.vector_file.clone(),
                by_row,
                resident,
            )
        };

        // Phase 2: nothing held, off the runtime's workers, on a pool of its own.
        let vector = query.vector.clone();
        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(1);
        tokio::task::spawn_blocking(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("exact-scan-{i}"))
                .build()
                .map_err(|e| crate::Error::Internal(format!("exact scan pool: {e}")))?;

            let mut scored: Vec<(RecordId, f32)> = pool.install(|| {
                use rayon::prelude::*;

                let from_file = file
                    .as_ref()
                    .map(|file| {
                        by_row
                            .par_chunks(CHUNK)
                            .map(|chunk| {
                                let rows: Vec<u64> = chunk.iter().map(|(row, _)| *row).collect();
                                let mut best: Vec<(RecordId, f32)> =
                                    Vec::with_capacity(chunk.len());
                                file.with_rows(&rows, |i, v| {
                                    if let Some(v) = v {
                                        let d =
                                            metric.calculate(&vector, v).unwrap_or(f32::INFINITY);
                                        best.push((chunk[i].1.clone(), d));
                                    }
                                });
                                Self::keep_best(best, result_limit)
                            })
                            .reduce(Vec::new, |mut a, b| {
                                a.extend(b);
                                Self::keep_best(a, result_limit)
                            })
                    })
                    .unwrap_or_default();

                let from_memory = resident
                    .par_chunks(CHUNK)
                    .map(|chunk| {
                        let best: Vec<(RecordId, f32)> = chunk
                            .iter()
                            .map(|r| {
                                let d = metric
                                    .calculate(&vector, &r.vector)
                                    .unwrap_or(f32::INFINITY);
                                (r.id.clone(), d)
                            })
                            .collect();
                        Self::keep_best(best, result_limit)
                    })
                    .reduce(Vec::new, |mut a, b| {
                        a.extend(b);
                        Self::keep_best(a, result_limit)
                    });

                let mut all = from_file;
                all.extend(from_memory);
                all
            });

            scored = Self::keep_best(scored, result_limit);
            Ok::<_, crate::Error>((scored, metric))
        })
        .await
        .map_err(|e| crate::Error::Internal(format!("exact scan task: {e}")))?
    }

    /// A k-nearest-neighbour graph over a sample of the table.
    ///
    /// This exists to move a specific piece of work off the browser. A projection needs to know
    /// which points are near which, and the only way to answer that from a client was to
    /// download the vectors and compute it there — which caps the sample at what fits down a
    /// wire: about 1,300 rows at 768 dimensions before the JSON is thirteen megabytes. The
    /// graph is the answer without the vectors. At 2,000 nodes and 15 neighbours it is 30,000
    /// edges, roughly a fortieth of what the vectors weigh, and the ceiling stops being the
    /// network.
    ///
    /// Neighbours are computed **within the sample**, not against the table. A layout is drawn
    /// from the points on screen, so an edge to a point that is not being drawn describes
    /// nothing; and a sample's own nearest neighbours are what a reader is comparing the layout
    /// against.
    ///
    /// Exact, not indexed. The graph is small and the answer is the reference the projection
    /// gets judged against — an approximate one would make "how much structure survived the
    /// layout" a measurement of two approximations at once. It is O(n² · d), which is why
    /// `sample` is capped: 4,000 nodes at 384 dimensions is about six billion multiply-adds,
    /// a couple of seconds on the pool below, and the cost is quadratic from there.
    pub async fn knn_graph(
        &self,
        tablename: &str,
        sample: usize,
        neighbours: usize,
        filter: Option<&crate::types::Filter>,
        seed: u64,
        with_payload: bool,
    ) -> crate::Result<KnnGraph> {
        let tablename = &*self.resolve(tablename);
        Self::check_filter(filter)?;
        self.ensure_resident(tablename).await?;
        if filter.is_some() {
            self.ensure_derived(tablename).await?;
        }

        let cap = Self::knn_graph_max_nodes();
        if sample > cap {
            return Err(crate::Error::InvalidInput(format!(
                "a {sample}-node graph is over the {cap}-node limit; the computation is \
                 quadratic in the node count. Lower `sample`, or raise DXMT_KNN_GRAPH_MAX_NODES"
            )));
        }

        // Phase 1, under the map lock: choose the sample and read its vectors, then let go.
        let (metric, ids, flat, dimension, total, payloads) = {
            let tables = self.read_tables("knn_graph").await;
            let table_data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

            let dimension = table_data.table.dimension;
            let allowed = filter.map(|f| table_data.derived().payload.get_matching_ids(f));
            let mut candidates: Vec<RecordId> = table_data
                .records
                .ids()
                .into_iter()
                .filter(|id| allowed.as_ref().is_none_or(|set| set.contains(id)))
                .collect();
            // Sorted, so which rows a given seed picks does not depend on the map's iteration
            // order -- otherwise the same request answers differently between restarts.
            candidates.sort_unstable();
            let total = candidates.len();

            // Evenly spaced rather than random, and stated as such by the caller. A table is
            // usually loaded in some order, and taking a contiguous block samples whenever that
            // block was written rather than the table.
            let wanted = sample.min(total);
            let picked: Vec<RecordId> = if wanted == 0 {
                Vec::new()
            } else if wanted >= total {
                candidates
            } else {
                let offset = (seed as usize) % (total / wanted).max(1);
                (0..wanted)
                    .map(|i| candidates[(i * total / wanted + offset).min(total - 1)].clone())
                    .collect()
            };

            let mut flat = vec![0.0f32; picked.len() * dimension];
            let mut kept = Vec::with_capacity(picked.len());
            let mut payloads = with_payload.then(|| Vec::with_capacity(picked.len()));
            for id in picked {
                let Some(record) = table_data.records.get(&id) else {
                    continue;
                };
                let vector = table_data.vector_for(&record);
                if vector.len() != dimension {
                    continue;
                }
                let at = kept.len() * dimension;
                flat[at..at + dimension].copy_from_slice(&vector);
                kept.push(id);
                if let Some(out) = payloads.as_mut() {
                    // The same JSON string `records/list` sends, so a client has one shape to
                    // parse rather than two.
                    out.push(
                        (!record.payload.is_null())
                            .then(|| serde_json::to_string(&record.payload).ok())
                            .flatten(),
                    );
                }
            }
            flat.truncate(kept.len() * dimension);

            (
                table_data.table.distance_metric,
                kept,
                flat,
                dimension,
                total,
                payloads,
            )
        };

        let n = ids.len();
        if n < 2 {
            return Ok(KnnGraph {
                ids,
                neighbours: Vec::new(),
                scores: Vec::new(),
                sampled: n,
                total,
                payloads,
            });
        }
        let k = neighbours.min(n - 1).max(1);

        // Phase 2: quadratic and synchronous, so off the runtime and on a pool of its own --
        // the same treatment PQ training and the exact scan get, for the same reason.
        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(1);
        let (neighbours, scores) = tokio::task::spawn_blocking(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("knn-graph-{i}"))
                .build()
                .map_err(|e| crate::Error::Internal(format!("knn graph pool: {e}")))?;

            let rows: Vec<(Vec<u32>, Vec<f32>)> = pool.install(|| {
                use rayon::prelude::*;
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let me = &flat[i * dimension..(i + 1) * dimension];
                        let mut scored: Vec<(f32, u32)> = (0..n)
                            .filter(|&j| j != i)
                            .map(|j| {
                                let other = &flat[j * dimension..(j + 1) * dimension];
                                let d = metric.calculate(me, other).unwrap_or(f32::INFINITY);
                                (d, j as u32)
                            })
                            .collect();
                        // Partial sort: k is ten or so against a few thousand candidates, and
                        // sorting all of them is most of the arithmetic spent on edges nobody
                        // will draw. The index tiebreak keeps the answer reproducible.
                        if scored.len() > k {
                            scored.select_nth_unstable_by(k - 1, |a, b| {
                                a.0.partial_cmp(&b.0)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                                    .then_with(|| a.1.cmp(&b.1))
                            });
                            scored.truncate(k);
                        }
                        scored.sort_by(|a, b| {
                            a.0.partial_cmp(&b.0)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.1.cmp(&b.1))
                        });
                        (
                            scored.iter().map(|(_, j)| *j).collect(),
                            // Similarities, matching what search returns, so a client does not
                            // have to know which way round this particular metric runs.
                            scored
                                .iter()
                                .map(|(d, _)| Self::similarity_from_distance(metric, *d))
                                .collect(),
                        )
                    })
                    .collect()
            });
            Ok::<_, crate::Error>(rows.into_iter().unzip::<_, _, Vec<_>, Vec<_>>())
        })
        .await
        .map_err(|e| crate::Error::Internal(format!("knn graph task: {e}")))??;

        Ok(KnnGraph {
            ids,
            neighbours,
            scores,
            sampled: n,
            total,
            payloads,
        })
    }

    /// Largest graph `knn_graph` will build. The work is O(n² · d).
    ///
    /// 4,000 nodes at 384 dimensions is about six billion multiply-adds — a couple of seconds
    /// on half the cores — and doubling the nodes quadruples that. A ceiling that refuses is
    /// better than one that takes a minute, because a caller cannot tell a slow answer from a
    /// hung one.
    fn knn_graph_max_nodes() -> usize {
        crate::config::env::var("DXMT_KNN_GRAPH_MAX_NODES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4_000)
    }

    /// The `limit` smallest distances, with an id tiebreak so a run is reproducible.
    ///
    /// `select_nth_unstable_by` rather than a full sort: the chunks that feed this are 4,096
    /// wide and the limit is usually ten, so sorting every chunk in full is most of the scan's
    /// arithmetic spent on records nobody will see. The tiebreak matters for the same reason it
    /// matters in `assemble_hits` -- equal distances are common, and which of them survives the
    /// truncation must not depend on which thread got there first.
    fn keep_best(mut scored: Vec<(RecordId, f32)>, limit: usize) -> Vec<(RecordId, f32)> {
        let order = |a: &(RecordId, f32), b: &(RecordId, f32)| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        };
        if scored.len() > limit && limit > 0 {
            scored.select_nth_unstable_by(limit - 1, order);
            scored.truncate(limit);
        }
        scored.sort_by(order);
        scored
    }

    /// Every record the filter accepts, scored exactly against the query.
    ///
    /// The fallback a post-filtered search takes when the graph keeps handing back candidates
    /// that all fail the predicate. Two passes, both bounded by the table: one over the
    /// payloads to find what matches, one over those rows' vectors to order them. The payload
    /// pass is what makes an empty filter cheap -- it reads no vectors at all -- and it runs
    /// under the read lock only for as long as it takes to collect ids.
    ///
    /// The predicate stays `Record::matches_filter`, the same one `assemble_hits` applies, and
    /// deliberately not `payload_index.get_matching_ids`: the index keys on scalar values that
    /// `to_value_key` can encode, so an array- or object-valued field never enters it. Reading
    /// the answer out of the index would be faster and would quietly change which rows match.
    async fn exhaustive_post_filter(
        &self,
        tablename: &str,
        query: &SearchQuery,
        filter: &crate::types::record::Filter,
        metric: DistanceMetric,
        result_limit: usize,
    ) -> crate::Result<Vec<SearchHit>> {
        // The distances here are placeholders: `rerank_exact` replaces every one of them with
        // the exact distance from the row's own vector.
        let matching: Vec<(RecordId, f32)> = {
            let tables = self.read_tables("search-exhaustive").await;
            let table_data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
            // A provably empty filter never reaches here -- `cannot_match_anything` answers
            // that before the first search. What is left is a filter that matches something,
            // too little of it for the walk to find, so the predicate is the enumerator.
            let mut ids = Vec::new();
            table_data.records.for_each(|record| {
                if record.matches_filter(filter) {
                    ids.push((record.id.clone(), 0.0));
                }
            });
            ids
        };

        let scored = if matching.is_empty() {
            Vec::new()
        } else {
            self.rerank_exact(tablename, metric, &query.vector, matching, result_limit)
                .await?
        };

        // `post_filter` is false: these ids are the ones that passed the filter, so applying
        // it again would only re-read the payloads.
        self.assemble_hits(tablename, query, scored, false, metric, result_limit)
            .await
    }

    /// Number of escalating refetches a short post-filtered search is allowed.
    ///
    /// The last attempt asks for every record, so three is not a heuristic budget: it is one
    /// geometric step, one more, then the exhaustive answer.
    const POST_FILTER_REFETCH_ATTEMPTS: usize = 3;

    /// Turn raw `(id, distance)` index output into hits, applying the payload filter when the
    /// search did not already do it in-graph.
    ///
    /// An index search, reranked against the stored vectors when the index asks for it.
    ///
    /// A quantized graph (`HnswPqIndex`) walks on codes and returns candidates ordered by an
    /// approximate distance; it says how many to fetch through `rerank_candidates`, and the
    /// exact distance from the record's own vector decides the final order and the score the
    /// caller sees. An exact index passes straight through. With a filter, the fetch goes
    /// through the in-graph filtered walk with the same enlarged `k`.
    #[allow(clippy::too_many_arguments)]
    async fn search_reranked(
        &self,
        tablename: &str,
        index: &crate::search::index_wrapper::IndexWrapper,
        metric: DistanceMetric,
        query: &DenseVector,
        k: usize,
        ef: usize,
        mode: SearchMode,
        filter_ids: Option<&std::collections::HashSet<String>>,
        keep: Option<&std::collections::HashSet<String>>,
        gate: Option<&dyn crate::search::node_filter::NodeGate>,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        // A `k` larger than the index cannot be answered by asking for more: past that point
        // every extra candidate is a slot the walk keeps open and a node the heap carries, and
        // there is nothing left to put in them. The post-filter escalation used to hand this
        // `k = total_records`, which the rerank factor below then multiplied by five.
        let rows = index.len().await;
        let k = if rows > 0 { k.min(rows) } else { k };
        let Some(candidates) = index.rerank_candidates(k).await else {
            return match (gate, filter_ids) {
                (Some(gate), _) => index.search_with_gate(query, k, ef, gate).await,
                (None, Some(ids)) => index.search_with_filter(query, k, ef, ids).await,
                (None, None) => index.search_with_ef_hint(query, k, ef).await,
            };
        };
        // The mode widens the pool as well as the walk. On a quantized graph the pool is what
        // decides the answer -- see `SearchMode::rerank_widening` for the measurement that
        // says so -- and widening only `ef` left `accurate` paying for candidates it then
        // threw away.
        let candidates = candidates.saturating_mul(mode.rerank_widening());
        let fetch = if rows > 0 {
            candidates.max(k).min(rows)
        } else {
            candidates.max(k)
        };
        let mut approximate = match (gate, filter_ids) {
            (Some(gate), _) => {
                index
                    .search_with_gate(query, fetch, ef.max(fetch), gate)
                    .await?
            }
            (None, Some(ids)) => {
                index
                    .search_with_filter(query, fetch, ef.max(fetch), ids)
                    .await?
            }
            (None, None) => {
                index
                    .search_with_ef_hint(query, fetch, ef.max(fetch))
                    .await?
            }
        };
        // A pre-filtered fetch is amplified by 1/selectivity; the candidates that fail the
        // filter are dropped here, before any vector is read for them.
        if let Some(keep) = keep {
            approximate.retain(|(id, _)| keep.contains(id));
        }
        self.rerank_exact(tablename, metric, query, approximate, k)
            .await
    }

    /// Score `approximate` candidates against the stored vectors and keep the best `k`.
    async fn rerank_exact(
        &self,
        tablename: &str,
        metric: DistanceMetric,
        query: &DenseVector,
        approximate: Vec<(RecordId, f32)>,
        k: usize,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        // Resolve under the map lock, read and score outside it: with the vectors on disk
        // the reads can take milliseconds cold, and writers must not wait on them.
        let (vectors, file_rows, file) = {
            let tables = self.read_tables("rerank").await;
            let table_data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
            let mut in_memory: Vec<(RecordId, Vec<f32>)> = Vec::new();
            let mut on_disk: Vec<(RecordId, u64)> = Vec::new();
            for (id, _) in approximate {
                // The row map answers first: with the records on disk, `records.get`
                // would decode a msgpack row per candidate -- 500 decodes a query, which
                // cost 10-13% of QPS on the 1M acceptance run -- only to learn what the
                // map already says.
                if let Some(row) = table_data.vector_row(&id) {
                    on_disk.push((id, row));
                    continue;
                }
                let Some(record) = table_data.records.get(&id) else {
                    continue;
                };
                if !record.vector.is_empty() {
                    in_memory.push((id, record.vector.clone()));
                }
            }
            (in_memory, on_disk, table_data.vector_file.clone())
        };
        // One scorer for the whole rerank. Cosine's `|query|` does not depend on the
        // candidate, and `calculate` recomputed it for each of the 500 -- 11.5% of the
        // server's CPU at 1M went into this function.
        let scorer = crate::types::vector::ExactScorer::new(metric, query);
        let mut exact: Vec<(RecordId, f32)> = vectors
            .into_iter()
            .filter_map(|(id, vector)| Some((id, scorer.try_score(&vector)?)))
            .collect();
        if let (Some(file), false) = (file, file_rows.is_empty()) {
            // Scored straight off the mapping: no copy, no allocation per candidate.
            let rows: Vec<u64> = file_rows.iter().map(|(_, row)| *row).collect();
            let mut ids: Vec<Option<RecordId>> =
                file_rows.into_iter().map(|(id, _)| Some(id)).collect();
            file.with_rows(&rows, |i, vector| {
                if let (Some(vector), Some(id)) = (vector, ids[i].take()) {
                    if let Some(dist) = scorer.try_score(vector) {
                        exact.push((id, dist));
                    }
                }
            });
        }
        // Distance, then id, before the cut to `k`. A tie at the cutoff settled by the order the
        // candidates happened to be scored in is the defect `IndexWrapper::top_k` closes for the
        // fetch; it would reopen here.
        exact.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        exact.truncate(k);
        Ok(exact)
    }

    /// Separate from `search` because a short post-filtered result has to be reassembled after
    /// a larger fetch, and the map lock must not be held across that fetch.
    #[allow(clippy::too_many_arguments)]
    /// How many hits the caller asked for, from the two fields that can say it.
    ///
    /// `limit` wins when it is set to anything but its default, because the Python SDK sends it
    /// and the Rust API is written around `top_k`. One definition, because `search` sizes the
    /// fetch from it and `assemble_hits` cuts the page with it, and the two disagreeing would
    /// show up as a page of the wrong length.
    fn page_size(query: &SearchQuery) -> usize {
        if query.limit > 0 && query.limit != 10 {
            query.limit
        } else {
            query.top_k
        }
    }

    /// Reorder hits for diversity: maximal marginal relevance over the candidates.
    ///
    /// Greedy and O(k · candidates) in similarity computations, which is why `fetch_k` is the
    /// knob that matters. Similarity between two candidates is the table's own metric turned
    /// into a similarity the same way a score is, so "near" means here what it means in the
    /// ranking -- computing cosine on a table built for euclidean would diversify against a
    /// notion of nearness the index does not use.
    ///
    /// A candidate whose vector cannot be read is kept in its original position rather than
    /// dropped: a row whose vector is missing is a bug elsewhere, and losing it from a result
    /// it legitimately earned would hide that.
    fn mmr_rerank(
        query_vector: &[f32],
        candidates: Vec<(SearchHit, Option<Vec<f32>>)>,
        metric: DistanceMetric,
        want: usize,
        lambda: f32,
    ) -> Vec<SearchHit> {
        let similarity = |a: &[f32], b: &[f32]| -> f32 {
            metric
                .calculate(a, b)
                .map(|d| Self::similarity_from_distance(metric, d))
                .unwrap_or(0.0)
        };
        let lambda = lambda.clamp(0.0, 1.0);

        let mut pool: Vec<(SearchHit, Option<Vec<f32>>)> = candidates;
        let mut chosen: Vec<SearchHit> = Vec::with_capacity(want.min(pool.len()));
        let mut chosen_vectors: Vec<Vec<f32>> = Vec::with_capacity(want.min(pool.len()));

        while chosen.len() < want && !pool.is_empty() {
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (i, (hit, vector)) in pool.iter().enumerate() {
                let to_query = hit.score;
                let redundancy = match vector {
                    // Without a vector there is nothing to compare, so it competes on its
                    // distance to the query alone.
                    None => 0.0,
                    Some(v) => chosen_vectors
                        .iter()
                        .map(|c| similarity(v, c))
                        .fold(f32::NEG_INFINITY, f32::max)
                        .max(0.0),
                };
                let score = lambda * to_query - (1.0 - lambda) * redundancy;
                // `>` not `>=`, so ties keep the order the ranking put them in.
                if score > best_score {
                    best_score = score;
                    best = i;
                }
            }
            let (hit, vector) = pool.remove(best);
            if let Some(v) = vector {
                chosen_vectors.push(v);
            }
            chosen.push(hit);
        }
        chosen
    }

    async fn assemble_hits(
        &self,
        tablename: &str,
        query: &SearchQuery,
        results: Vec<(RecordId, f32)>,
        post_filter: bool,
        metric: DistanceMetric,
        result_limit: usize,
    ) -> crate::Result<Vec<SearchHit>> {
        // Back under the map lock, but only for hash lookups — the expensive part is done. The
        // table can have been dropped while the search ran, and a returned id can have been
        // deleted; both are handled below rather than pretended away, because that window is the
        // price of not holding the lock across the search.
        let tables = self.read_tables("search").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        // 应用filter并构建搜索结果
        // A total order before anything is truncated. Distances tie -- three records the same
        // step either side of the probe tie exactly -- and until now the tie was broken by
        // whatever order the ids happened to arrive in, which for a filtered search is the
        // iteration order of a `HashSet`. That is not stable across index types, and not even
        // across runs of the same one: `a_selective_filter_works_on_a_partitioned_table`
        // compares an HNSW answer against a partitioned one and saw the same five records in
        // two different orders. Because `take` happens below, it also decided *which* records
        // survived, not only how they were arranged.
        let mut results = results;
        results.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let hits: Vec<SearchHit> = results
            .into_iter()
            .filter_map(|(id, score)| {
                // POST-FILTERING: apply payload filter on results that weren't in-graph filtered
                if post_filter {
                    if let Some(ref filter) = query.filter {
                        {
                            let record = table_data.records.get(&id)?;
                            if !record.matches_filter(filter) {
                                return None;
                            }
                        }
                    }
                }

                // Load payload and vector based on query flags
                let (payload, vector) = if query.with_payload || query.with_vector {
                    table_data
                        .records
                        .get(&id)
                        .map(|r| {
                            (
                                if query.with_payload {
                                    Some(crate::types::record::project_payload(
                                        &r.payload,
                                        query.payload_fields.as_deref(),
                                    ))
                                } else {
                                    None
                                },
                                if query.with_vector {
                                    Some(table_data.vector_for(&r))
                                } else {
                                    None
                                },
                            )
                        })
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };

                let similarity_score = Self::similarity_from_distance(metric, score);

                // The floor is on the similarity the caller sees, not on the internal
                // distance: those run in opposite directions, and a threshold that silently
                // meant the other one would return exactly the hits it was asked to remove.
                if let Some(floor) = query.score_threshold {
                    if similarity_score < floor {
                        return None;
                    }
                }

                Some(SearchHit {
                    id,
                    score: similarity_score,
                    payload,
                    vector,
                    rerank_score: None,
                })
            })
            // The offset is dropped here rather than before scoring: `score_threshold` and the
            // post-filter both remove hits, and skipping first would skip rows that were never
            // going to be in the answer.
            .skip(if query.mmr.is_some() { 0 } else { query.offset })
            .take(if query.mmr.is_some() {
                result_limit
            } else {
                result_limit.saturating_sub(query.offset)
            })
            .collect();

        let Some(mmr) = &query.mmr else {
            return Ok(hits);
        };

        // Diversity reranking, still under the map lock, because it needs the candidates'
        // vectors and this is where they can be read without a second pass over the table. The
        // page is cut afterwards: the offset applies to the reranked order, so pages of a
        // diversified search still tile.
        let want = Self::page_size(query).saturating_add(query.offset);
        let with_vectors: Vec<(SearchHit, Option<Vec<f32>>)> = hits
            .into_iter()
            .map(|hit| {
                let vector = table_data
                    .records
                    .get(&hit.id)
                    .map(|record| table_data.vector_for(&record))
                    .filter(|v| !v.is_empty());
                (hit, vector)
            })
            .collect();
        let mut reranked =
            Self::mmr_rerank(&query.vector, with_vectors, metric, want, mmr.lambda_mult);
        if query.offset > 0 {
            reranked.drain(..query.offset.min(reranked.len()));
        }
        Ok(reranked)
    }

    /// 全文搜索
    #[cfg(feature = "text-bm25")]
    pub async fn full_text_search(
        &self,
        tablename: &str,
        request: FullTextSearchRequest,
    ) -> crate::Result<FullTextSearchResult> {
        let tablename = &*self.resolve(tablename);
        self.ensure_kb_table(tablename).await?;
        let plan = self
            .rerank_plan(
                tablename,
                request.rerank.as_ref(),
                Some(&request.query),
                request.limit,
                request.offset,
                "/search/text",
            )
            .await?;
        let Some(plan) = plan else {
            return self.full_text_unranked(tablename, request).await;
        };
        let start = std::time::Instant::now();
        let mut inner = request;
        inner.rerank = None;
        inner.offset = 0;
        inner.limit = plan.pool;
        let fetched = self.full_text_unranked(tablename, inner).await?;
        let hits = Self::rerank_hits(
            &plan,
            fetched.hits,
            |h| h.payload.as_ref(),
            |h, s| h.rerank_score = Some(s),
        )
        .await?;
        Ok(FullTextSearchResult {
            total: hits.len(),
            hits,
            took_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// `full_text_search` without the rerank step.
    #[cfg(feature = "text-bm25")]
    async fn full_text_unranked(
        &self,
        tablename: &str,
        request: FullTextSearchRequest,
    ) -> crate::Result<FullTextSearchResult> {
        Self::check_filter(request.filter.as_ref())?;
        let start_time = std::time::Instant::now();
        // The engine is behind an Arc, so take it out of the map and drop the guard rather than
        // running a BM25 query with every other table's reads and writes queued behind it.
        let engine = {
            let tables = self.read_tables("full_text_search").await;
            let table_data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

            // Check if full-text search is enabled
            if !table_data.table.enable_full_text_search {
                return Err(crate::Error::InvalidInput(
                    "Full-text search is not enabled for this table".to_string(),
                ));
            }

            table_data.full_text_engine.clone().ok_or_else(|| {
                crate::Error::Internal("Full-text search engine not initialized".to_string())
            })?
        };

        // 如果有过滤器，获取更多结果以补偿过滤后的损失
        let fetch_limit = if request.filter.is_some() {
            request.limit * 3 // 获取3倍结果
        } else {
            request.limit
        };

        // Build the search request for the engine
        let engine_request = EngineSearchRequest {
            query: request.query.clone(),
            search_fields: request.fields.clone(),
            limit: fetch_limit,
            offset: request.offset,
            highlight: request.highlight,
            highlight_pre_tag: None,
            highlight_post_tag: None,
            fuzzy: request
                .fuzzy
                .as_ref()
                .map(|f| crate::search::full_text::FuzzyConfig {
                    max_distance: f.max_distance,
                    prefix_length: f.prefix_length as u8,
                    transpositions: true,
                }),
            filters: None,
            sort_by: None,
        };

        // Perform the search, holding no map lock.
        let search_results = engine.search(&engine_request).await?;

        // Back under the map for the record lookups behind the filter and the payloads.
        let tables = self.read_tables("full_text_search").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        // Convert engine results to API results
        let hits: Vec<FullTextSearchHit> = search_results
            .into_iter()
            .filter_map(|result| {
                // Apply filters if specified
                if let Some(ref filter) = request.filter {
                    {
                        let record = table_data.records.get(&result.id)?;
                        if !record.matches_filter(filter) {
                            return None;
                        }
                    }
                }

                // Get the full record if needed
                let payload = table_data
                    .records
                    .get(&result.id)
                    .map(|r| r.payload.clone());

                Some(FullTextSearchHit {
                    id: result.id,
                    score: result.score,
                    payload,
                    highlights: if request.highlight {
                        // Flatten HashMap<String, Vec<String>> to Vec<String>
                        result
                            .highlights
                            .map(|h| h.into_values().flatten().collect())
                    } else {
                        None
                    },
                    rerank_score: None,
                })
            })
            .take(request.limit) // 过滤后只保留limit个
            .collect();

        let total = hits.len();
        let took_ms = start_time.elapsed().as_millis() as u64;

        Ok(FullTextSearchResult {
            hits,
            total,
            took_ms,
        })
    }

    /// 加权求和融合（使用 Min-Max 归一化）
    /// 适合大数据集，但在小数据集上可能不稳定
    /// Distribution-Based Score Fusion (DBSF) for up to three result lists.
    ///
    /// Each list is normalised independently via (s-mean)/std, clipped to [-3σ,+3σ],
    /// then rescaled to [0,1]. Normalised scores are summed. This removes unit
    /// mismatch between cosine/distance scores, sparse MIPS scores, and BM25 scores
    /// without requiring alpha tuning.
    /// Turn the distance an index returns into the similarity the API returns.
    ///
    /// Every `VectorIndex::search` yields a distance — lower is better. Everything above this
    /// layer, including the fusion in `hybrid_search`, ranks by score with higher better. The
    /// conversion is therefore not cosmetic: feeding a raw distance into a "higher is better"
    /// ranker returns the results in exactly the wrong order.
    fn similarity_from_distance(metric: DistanceMetric, distance: f32) -> f32 {
        match metric {
            // Cosine distance = 1 - similarity, so a similarity in [-1, 1] is a distance in
            // [0, 2]. This clamped the *distance* to [0, 1], which reported every negative
            // similarity as 0.0: vectors pointing away from the query all scored the same, and
            // anything that re-ranked by score lost their order. Qdrant returns the negative
            // cosine as it is; fusion normalises each list by its own mean and spread, so
            // nothing downstream needs the score to be non-negative. The clamp left is for
            // floating-point overshoot only.
            DistanceMetric::Cosine => (1.0 - distance).clamp(-1.0, 1.0),
            // Dot-product distance = -dot.
            DistanceMetric::Dot => -distance,
            // Unbounded metrics: any decreasing function will do, and this one keeps the
            // result in (0, 1] so it can be fused with the bounded ones.
            DistanceMetric::Euclidean | DistanceMetric::Manhattan => 1.0 / (1.0 + distance),
        }
    }

    fn fuse_dbsf(
        vector: Vec<(RecordId, f32)>,
        sparse: Vec<(RecordId, f32)>,
        text: Vec<(RecordId, f32)>,
    ) -> HashMap<RecordId, f32> {
        fn dbsf_norm(list: &[(RecordId, f32)]) -> HashMap<RecordId, f32> {
            if list.is_empty() {
                return HashMap::new();
            }
            let n = list.len() as f32;
            let mean = list.iter().map(|(_, s)| s).sum::<f32>() / n;
            let var = list.iter().map(|(_, s)| (s - mean).powi(2)).sum::<f32>() / n;
            let std = var.sqrt().max(1e-8);
            let low = mean - 3.0 * std;
            let high = mean + 3.0 * std;
            let range = (high - low).max(1e-8);
            list.iter()
                .map(|(id, s)| {
                    let clipped = s.max(low).min(high);
                    (id.clone(), (clipped - low) / range)
                })
                .collect()
        }

        let mut scores: HashMap<RecordId, f32> = HashMap::new();
        for (id, s) in dbsf_norm(&vector) {
            *scores.entry(id).or_insert(0.0) += s;
        }
        for (id, s) in dbsf_norm(&sparse) {
            *scores.entry(id).or_insert(0.0) += s;
        }
        for (id, s) in dbsf_norm(&text) {
            *scores.entry(id).or_insert(0.0) += s;
        }
        scores
    }

    fn fuse_weighted_sum(
        &self,
        vector_results: Vec<(RecordId, f32)>,
        text_results: Vec<(RecordId, f32)>,
        vector_weight: f32,
    ) -> HashMap<RecordId, f32> {
        let mut combined_scores: HashMap<RecordId, f32> = HashMap::new();

        // Normalize and weight vector scores
        if !vector_results.is_empty() {
            let scores: Vec<f32> = vector_results.iter().map(|(_, s)| *s).collect();
            let max_score = scores
                .iter()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);
            let min_score = scores
                .iter()
                .cloned()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let score_range = max_score - min_score;

            for (id, score) in vector_results {
                // When all scores are similar (range < 0.01), treat them equally
                let normalized_score = if score_range < 0.01 {
                    1.0
                } else {
                    (score - min_score) / score_range
                };
                combined_scores.insert(id, normalized_score * vector_weight);
            }
        }

        // Normalize and weight text scores
        if !text_results.is_empty() {
            let scores: Vec<f32> = text_results.iter().map(|(_, s)| *s).collect();
            let max_score = scores
                .iter()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);
            let min_score = scores
                .iter()
                .cloned()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let score_range = max_score - min_score;

            let text_weight = 1.0 - vector_weight;
            for (id, score) in text_results {
                let normalized_score = if score_range < 0.01 {
                    1.0
                } else {
                    (score - min_score) / score_range
                };
                let current_score = combined_scores.get(&id).unwrap_or(&0.0);
                combined_scores.insert(id, current_score + (normalized_score * text_weight));
            }
        }

        combined_scores
    }

    /// Reciprocal Rank Fusion (RRF)
    /// 基于排名而非分数融合，对分数尺度不敏感
    /// 特别适合小数据集和分数范围差异大的场景
    ///
    /// RRF_score(d) = sum over all rankings r of: 1 / (k + rank_r(d))
    ///
    /// 参数:
    /// - k: 常数，通常为60，用于平滑排名差异
    /// - vector_weight: 向量搜索权重，用于加权RRF分数
    fn fuse_reciprocal_rank(
        &self,
        vector_results: Vec<(RecordId, f32)>,
        text_results: Vec<(RecordId, f32)>,
        vector_weight: f32,
        k: f32,
    ) -> HashMap<RecordId, f32> {
        let mut combined_scores: HashMap<RecordId, f32> = HashMap::new();

        // Build rank maps (1-indexed)
        let mut vector_ranks: HashMap<RecordId, usize> = HashMap::new();
        for (rank, (id, _score)) in vector_results.iter().enumerate() {
            vector_ranks.insert(id.clone(), rank + 1);
        }

        let mut text_ranks: HashMap<RecordId, usize> = HashMap::new();
        for (rank, (id, _score)) in text_results.iter().enumerate() {
            text_ranks.insert(id.clone(), rank + 1);
        }

        // Collect all document IDs
        let mut all_ids: std::collections::HashSet<RecordId> = std::collections::HashSet::new();
        all_ids.extend(vector_ranks.keys().cloned());
        all_ids.extend(text_ranks.keys().cloned());

        // Calculate RRF score for each document
        let text_weight = 1.0 - vector_weight;
        for id in all_ids {
            let mut rrf_score = 0.0;

            // Add vector ranking contribution
            if let Some(&rank) = vector_ranks.get(&id) {
                rrf_score += vector_weight / (k + rank as f32);
            }

            // Add text ranking contribution
            if let Some(&rank) = text_ranks.get(&id) {
                rrf_score += text_weight / (k + rank as f32);
            }

            combined_scores.insert(id, rrf_score);
        }

        combined_scores
    }

    /// 混合搜索（稠密向量 + 稀疏向量 + 文本）；融合方式见 `HybridSearchRequest::fusion`
    pub async fn hybrid_search(
        &self,
        tablename: &str,
        request: HybridSearchRequest,
    ) -> crate::Result<SearchResult> {
        let tablename = &*self.resolve(tablename);
        self.ensure_kb_table(tablename).await?;
        let request = self.embed_hybrid_text(tablename, request).await?;
        let plan = self
            .rerank_plan(
                tablename,
                request.rerank.as_ref(),
                request.text.as_deref(),
                request.limit,
                0,
                "/search/hybrid",
            )
            .await?;
        let Some(plan) = plan else {
            return self.hybrid_unranked(tablename, request).await;
        };
        let start = std::time::Instant::now();
        let want_payload = request.with_payload;
        let mut inner = request;
        inner.rerank = None;
        inner.limit = plan.pool;
        inner.with_payload = true;
        let fetched = self.hybrid_unranked(tablename, inner).await?;
        let mut hits = Self::rerank_hits(
            &plan,
            fetched.hits,
            |h| h.payload.as_ref(),
            |h, s| h.rerank_score = Some(s),
        )
        .await?;
        if !want_payload {
            for hit in &mut hits {
                hit.payload = None;
            }
        }
        Ok(SearchResult {
            total: hits.len(),
            hits,
            tookms: start.elapsed().as_millis() as u64,
        })
    }

    /// `hybrid_search` without the rerank step.
    async fn hybrid_unranked(
        &self,
        tablename: &str,
        request: HybridSearchRequest,
    ) -> crate::Result<SearchResult> {
        Self::check_filter(request.filter.as_ref())?;
        if request.filter.is_some() || request.sparse_vector.is_some() {
            self.ensure_derived(tablename).await?;
        }
        let start_time = std::time::Instant::now();

        let fetch_limit = if request.filter.is_some() {
            request.limit * 10
        } else {
            request.limit * 3
        };

        // Phase 1: validate, take the index and the text engine out of the map, run the sparse
        // search, then release. Both indexes are Arc-backed so taking them costs a refcount.
        //
        // The sparse search stays here because `SparseIndex` is a plain struct of hash maps —
        // taking a copy would cost more than the search does. It is a WAND-pruned walk over
        // posting lists, far cheaper than the dense side, which is what actually needed moving.
        #[allow(unused_mut)]
        let (index, metric, mut sparse_results, text_engine, text_enabled) = {
            let tables = self.read_tables("hybrid_search").await;
            let table_data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

            // Validate dense vector dimension if provided
            let expected_dim = table_data.table.dimension;
            if let Some(ref v) = request.vector {
                if !v.is_empty() && v.len() != expected_dim {
                    return Err(crate::Error::DimensionMismatch {
                        expected: expected_dim,
                        actual: v.len(),
                    });
                }
            }

            let sparse_results: Vec<(RecordId, f32)> = match request.sparse_vector {
                Some(ref sv) => table_data.derived().sparse.search(sv, fetch_limit),
                None => Vec::new(),
            };

            #[cfg(feature = "text-bm25")]
            let engine = table_data.full_text_engine.clone();
            #[cfg(not(feature = "text-bm25"))]
            let engine: Option<()> = None;

            (
                table_data.index.clone(),
                table_data.table.distance_metric,
                sparse_results,
                engine,
                table_data.table.enable_full_text_search,
            )
        };

        // A text half that cannot be answered is refused here rather than dropped. The HTTP
        // handler checks the same thing first (it has its own error codes for it), but MCP and
        // anything else that reaches this method did not: a hybrid query with text on a table
        // with no keyword index ran the dense half and returned it shaped like a fusion.
        if request.text.is_some() {
            #[cfg(not(feature = "text-bm25"))]
            return Err(crate::Error::NotImplemented(
                "this build has no full-text index, so the text half of a hybrid query \
                 cannot be answered; send the vector alone"
                    .to_string(),
            ));
            #[cfg(feature = "text-bm25")]
            if !text_enabled || text_engine.is_none() {
                return Err(crate::Error::InvalidInput(format!(
                    "table '{tablename}' has no full-text index, so the text half of a hybrid \
                     query cannot be answered; declare `text_fields` at creation, or search \
                     by vector alone"
                )));
            }
        }

        // 1. Dense vector search — no map lock held.
        //
        // The index answers in distances, and the sparse and text lists that this is about to be
        // fused with are similarities. Convert here, the way `search` does: `fuse_dbsf`
        // normalises "higher is better" per list, so handing it distances ranked the dense
        // component backwards — the nearest vector came out last.
        let mut vector_results: Vec<(RecordId, f32)> = Vec::new();
        if let Some(ref vector) = request.vector {
            vector_results = index
                .search(vector, fetch_limit)
                .await?
                .into_iter()
                .map(|(id, distance)| (id, Self::similarity_from_distance(metric, distance)))
                .collect();
        }

        // 3. Full-text search
        #[cfg(feature = "text-bm25")]
        let mut text_results: Vec<(RecordId, f32)> = Vec::new();
        #[cfg(not(feature = "text-bm25"))]
        let text_results: Vec<(RecordId, f32)> = Vec::new();
        #[cfg(feature = "text-bm25")]
        if let Some(ref text_query) = request.text {
            if text_enabled {
                if let Some(engine) = &text_engine {
                    let text_request = EngineSearchRequest {
                        query: text_query.clone(),
                        search_fields: None,
                        limit: fetch_limit,
                        offset: 0,
                        highlight: false,
                        highlight_pre_tag: None,
                        highlight_post_tag: None,
                        fuzzy: None,
                        filters: None,
                        sort_by: None,
                    };
                    // An engine error fails the query: swallowed, it left the dense half alone
                    // behind a fusion-shaped answer.
                    text_results = engine
                        .search(&text_request)
                        .await?
                        .into_iter()
                        .map(|r| (r.id, r.score))
                        .collect();
                }
            }
        }

        // Phase 3: fusion and assembly need the records, so take the map again — for hash
        // lookups only, with every search already done.
        let tables = self.read_tables("hybrid_search").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        // 4. Fuse results using the selected method
        let combined_scores = match request.fusion() {
            FusionMethod::WeightedSum => {
                // Legacy two-path weighted sum; sparse is included as part of vector weight
                let mut merged_vec = vector_results.clone();
                // Merge sparse into dense side using simple sum before normalising
                for (id, s) in &sparse_results {
                    if let Some(entry) = merged_vec.iter_mut().find(|(eid, _)| eid == id) {
                        entry.1 = (entry.1 + s) / 2.0;
                    } else {
                        merged_vec.push((id.clone(), *s));
                    }
                }
                self.fuse_weighted_sum(merged_vec, text_results, request.vector_weight)
            }
            FusionMethod::ReciprocalRankFusion => {
                // Merge sparse into vector rank list, then RRF with text
                let mut merged_vec = vector_results.clone();
                for (id, s) in &sparse_results {
                    if !merged_vec.iter().any(|(eid, _)| eid == id) {
                        merged_vec.push((id.clone(), *s));
                    }
                }
                // This order becomes the rank RRF scores from, so a tie broken arbitrarily here
                // changes the fused score itself, not just the arrangement.
                merged_vec.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                self.fuse_reciprocal_rank(
                    merged_vec,
                    text_results,
                    request.vector_weight,
                    request.rrf_k,
                )
            }
            FusionMethod::Dbsf => {
                // Distribution-Based Score Fusion: normalise each of the three lists independently
                Self::fuse_dbsf(vector_results, sparse_results, text_results)
            }
        };

        // 3. Apply filters FIRST (before truncating)
        let mut scored_results: Vec<(RecordId, f32)> = combined_scores
            .into_iter()
            .filter(|(id, _score)| {
                // Apply filters if specified
                if let Some(ref filter) = request.filter {
                    if let Some(record) = table_data.records.get(id) {
                        return record.matches_filter(filter);
                    }
                    return false;
                }
                true
            })
            .collect();

        // 4. THEN sort by combined score, ties by id. RRF builds its scores in a `HashMap`, so
        // without the tiebreak the order of equally-scored documents -- and, because of the
        // truncate below, *which* of them survive -- follows hash iteration order.
        scored_results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        // 5. FINALLY take top-k
        scored_results.truncate(request.limit);

        // Save total before consuming the vector
        let total = scored_results.len();

        // 6. Build search hits with payload if requested
        let hits: Vec<SearchHit> = scored_results
            .into_iter()
            .map(|(id, score)| {
                let (payload, vector) = if request.with_payload || request.with_vector {
                    table_data
                        .records
                        .get(&id)
                        .map(|r| {
                            (
                                if request.with_payload {
                                    Some(r.payload.clone())
                                } else {
                                    None
                                },
                                if request.with_vector {
                                    Some(table_data.vector_for(&r))
                                } else {
                                    None
                                },
                            )
                        })
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };

                SearchHit {
                    id,
                    score,
                    payload,
                    vector,
                    rerank_score: None,
                }
            })
            .collect();

        let took_ms = start_time.elapsed().as_millis() as u64;

        Ok(SearchResult {
            hits,
            total,
            tookms: took_ms,
        })
    }

    /// 获取表信息
    pub async fn get_table(&self, name: &str) -> crate::Result<Table> {
        let name = &*self.resolve(name);
        // Every route checks its table here first, so this is where the knowledge-base table
        // is made on first use, for REST, MCP and gRPC alike.
        self.ensure_kb_table(name).await?;
        self.ensure_resident(name).await?;
        let tables = self.read_tables("get_table").await;

        tables
            .get(name)
            .map(|data| data.table.clone())
            .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))
    }

    /// 列出所有表，含被驱逐到对象存储的。
    ///
    /// 只看内存的话，一张表会因为没人访问而从列表里消失——对调用方来说和被删掉没有区别。
    /// 驱逐是内存管理，不是删除，列表必须照旧。
    pub async fn list_tables(&self) -> crate::Result<Vec<Table>> {
        let mut out: Vec<Table> = {
            let tables = self.read_tables("list_tables").await;
            tables.values().map(|data| data.table.clone()).collect()
        };

        let Some(ref wal_dir) = self.wal_dir else {
            return Ok(out);
        };
        let dir = Self::snapshot_dir(wal_dir);
        let Ok(manifest) = crate::search::index_persistence::read_index::<SnapshotManifest>(
            &dir.join("manifest"),
            "snapshot-manifest",
        ) else {
            return Ok(out);
        };

        let resident: std::collections::HashSet<String> =
            out.iter().map(|t| t.name.clone()).collect();
        let dropped = self.dropped_table_names();
        for (i, name) in manifest.names.iter().enumerate() {
            if resident.contains(name) || dropped.contains(name) {
                continue;
            }
            // Read only the metadata; the point of eviction is not to pull the records back in
            // just to answer a listing.
            if let Ok(head) = read_snapshot_head(&dir.join(format!("t{}.meta", i))) {
                out.push(head.table);
            }
        }

        Ok(out)
    }

    /// 修改表元数据，并把变更写入 WAL。
    pub async fn alter_table(&self, name: &str, req: TableAlterRequest) -> crate::Result<Table> {
        let name = &*self.resolve(name);
        self.ensure_writable()?;
        self.ensure_resident(name).await?;

        let updated = {
            let mut tables = self.write_tables("alter_table").await;
            let data = tables
                .get_mut(name)
                .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))?;
            if let Some(next) = req.embedding.as_ref() {
                // Repointing the endpoint is fine; changing the model is not. The rows already
                // in this table came from the old one, and mixing two embedding spaces in one
                // index produces a ranked page that looks entirely normal and means nothing.
                if let Some(current) = data.table.embedding.as_ref() {
                    if current.model != next.model {
                        return Err(crate::Error::InvalidInput(format!(
                            "table '{name}' is embedded with '{}' and its vectors cannot be \
                             re-embedded in place. Load a new table with model '{}' instead.",
                            current.model, next.model
                        )));
                    }
                } else if data.table.record_count > 0 {
                    return Err(crate::Error::InvalidInput(format!(
                        "table '{name}' already holds {} records with vectors the caller \
                         computed; giving it a model now would embed later rows in a different \
                         space from the ones already here.",
                        data.table.record_count
                    )));
                }
            }
            let updated = data.table.apply_alter(&req);
            if req.payload_index.is_some() {
                // The payload index cannot be narrowed in place -- a field's values are merged
                // into shared sorted arrays and postings, and there is no undo. Dropping it
                // makes the next read that needs a filter rebuild it under the new policy,
                // which is also when a narrowed policy gives its memory back.
                *data.derived.write() = None;
            }
            let text_changed = text_fingerprint(&data.table) != text_fingerprint(&updated);
            data.table = updated.clone();
            (updated, text_changed)
        };
        let (updated, text_changed) = updated;

        self.wal_append(vec![WalEntry::AlterTable {
            table: name.to_string(),
            request: req,
        }])
        .await?;

        // Logged first, so a crash during the rebuild leaves the table's new text config with
        // an index built for the old one -- which the next start notices (`text_fingerprint`)
        // and rebuilds.
        #[cfg(feature = "text-bm25")]
        if text_changed {
            self.rebuild_text_index(name).await?;
        }
        #[cfg(not(feature = "text-bm25"))]
        let _ = text_changed;

        Ok(updated)
    }

    /// 删除表
    /// Names dropped since the last snapshot. The manifest on disk still lists them, so
    /// every reader of it has to filter them out until a snapshot is written without them.
    fn dropped_table_names(&self) -> std::collections::HashSet<String> {
        self.dropped_since_snapshot
            .lock()
            .map(|set| set.clone())
            .unwrap_or_default()
    }

    pub async fn delete_table(&self, name: &str) -> crate::Result<()> {
        self.ensure_writable()?;
        let name = &*self.resolve(name);
        // Before the table goes, so nothing can resolve an alias to a name that no longer
        // holds anything -- the error a caller got would name the table, not the alias they
        // asked for.
        self.drop_aliases_for(name);
        let table_id = {
            let mut tables = self.write_tables("delete_table").await;
            let data = tables
                .remove(name)
                .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))?;
            // The row file goes with the table. Best effort: a file that will not delete is
            // orphaned, not a failed drop.
            //
            // Not while replaying. A log that drops a table and creates it again under the
            // same name holds, after the drop, entries that point at rows of the vector
            // file on disk -- which is the recreated table's file, the one the crash left.
            // Deleting it here would lose every row the log cannot supply. The records
            // file does go: the entries carry their records, and the replayed create reads
            // an empty store rather than one holding the dropped table's rows. The rows
            // the dropped table's entries appended meanwhile are dead space, reclaimed by
            // the sweep like any other.
            if let Some(wal_dir) = &self.wal_dir {
                if !self.replaying.load(Ordering::SeqCst) {
                    for f in [
                        crate::storage::vector_file::VectorFormat::F32,
                        crate::storage::vector_file::VectorFormat::F16,
                    ] {
                        let path = Self::vector_file_path_for(wal_dir, name, f);
                        let _ = std::fs::remove_file(&path);
                        let _ = std::fs::remove_file(
                            crate::storage::vector_file::VectorFile::rows_path(&path),
                        );
                    }
                }
                let rec = Self::record_file_path(wal_dir, name);
                let _ = std::fs::remove_file(&rec);
                let mut log = rec.as_os_str().to_owned();
                log.push(".log");
                let _ = std::fs::remove_file(PathBuf::from(log));
            }

            // The full-text index, a sibling of the row files above.
            //
            // This became necessary when table ids stopped being random: an id is now a
            // function of the name, so a table dropped and recreated under the same name gets
            // the same id and would otherwise open the previous table's index and answer text
            // searches with its documents. Leaving it behind was survivable while every restart
            // minted a new id and simply leaked a directory; it is not survivable now.
            #[cfg(feature = "text-bm25")]
            {
                let _ = std::fs::remove_dir_all(self.full_text_index_path(&data.table.id));
            }

            data.table.id.to_string()
        };

        self.wal_append(vec![WalEntry::DeleteTable {
            table: name.to_string(),
        }])
        .await?;

        // The removal is durable in the log now, but the snapshot on disk still carries
        // this table and the snapshot writer would copy it forward. Record it so the next
        // snapshot leaves it out, and so nothing lists or rehydrates it in the meantime.
        if let Ok(mut dropped) = self.dropped_since_snapshot.lock() {
            dropped.insert(name.to_string());
        }

        // A partitioned table leaves its segments in object storage. Delete them here rather
        // than at the next snapshot: the table is out of `tables` and its removal is in the
        // log, so nothing that can still be recovered refers to them.
        //
        // The caveat is real and the same one every database has with DROP: a backup taken
        // before this point can no longer restore that table's index, because its segments are
        // gone. A segment here belongs to exactly one table, so no cross-version reference
        // graph is needed to decide what is safe to remove — but the "do not delete what
        // something still references" rule is why this waits until the removal is durable in
        // the WAL.
        if self.object_storage_enabled {
            self.delete_partition_segments(&table_id).await;
        }

        Ok(())
    }

    /// 删掉已被维护取代、且当前快照不再引用的分区段。
    async fn collect_superseded_segments(&self) {
        use crate::storage::object::traits::ObjectStorage;

        let live: Vec<(String, Vec<String>)> = {
            let tables = self.read_tables("collect_superseded_segments").await;
            let mut out = Vec::new();
            for data in tables.values() {
                if let Some(keys) = data.index.live_segment_keys().await {
                    out.push((data.table.id.to_string(), keys));
                }
            }
            out
        };

        for (table_id, keys) in live {
            let live: std::collections::HashSet<String> = keys.into_iter().collect();
            let prefix = format!("partitions/{}/", table_id);
            let options = crate::storage::object::types::ListObjectsOptions {
                prefix: Some(prefix.clone()),
                max_keys: None,
                continuation_token: None,
                delimiter: None,
            };
            let Ok(listed) = self.storage.list_objects(&options).await else {
                continue;
            };

            let mut collected = 0usize;
            for object in listed.objects {
                if !live.contains(&object.key)
                    && self.storage.delete_object(&object.key).await.is_ok()
                {
                    collected += 1;
                }
            }
            if collected > 0 {
                debug!(
                    "Collected {} superseded segment(s) under {}",
                    collected, prefix
                );
            }
        }
    }

    /// 删掉一张表在对象存储上的全部分区段。
    async fn delete_partition_segments(&self, table_id: &str) {
        use crate::storage::object::traits::ObjectStorage;

        let prefix = format!("partitions/{}/", table_id);
        let options = crate::storage::object::types::ListObjectsOptions {
            prefix: Some(prefix.clone()),
            max_keys: None,
            continuation_token: None,
            delimiter: None,
        };
        let Ok(listed) = self.storage.list_objects(&options).await else {
            warn!(
                "Could not list {} to clean up; segments may be left behind",
                prefix
            );
            return;
        };

        let mut removed = 0usize;
        for object in listed.objects {
            if self.storage.delete_object(&object.key).await.is_ok() {
                removed += 1;
            }
        }
        if removed > 0 {
            info!("Deleted {} partition segment(s) under {}", removed, prefix);
        }
    }

    /// Drop the vector index for a table, replacing it with a BruteForce index.
    ///
    /// This frees the memory consumed by the ANN index (HNSW graph, PQ codebooks, etc.)
    /// while preserving all records. Subsequent searches fall back to exact brute-force
    /// scan until the index is rebuilt (e.g. via re-insertion or a future rebuild API).
    ///
    /// Exposed as an explicit operation rather than an automatic eviction so that
    /// callers control when the memory is released relative to their query load.
    pub async fn drop_index(&self, table_name: &str) -> crate::Result<()> {
        let table_name = &*self.resolve(table_name);
        // Phase 1: snapshot records and create new brute-force wrapper under the write lock
        let (new_wrapper, records_snapshot) = {
            let mut tables = self.write_tables("drop_index").await;

            let table_data = tables
                .get_mut(table_name)
                .ok_or_else(|| crate::Error::TableNotFound(table_name.to_string()))?;

            let distance_metric = table_data.table.distance_metric;
            let new_wrapper =
                crate::search::IndexWrapper::new(Box::new(BruteForceIndex::new(distance_metric)));

            let records_snapshot: Vec<(RecordId, Vec<f32>)> = table_data.vectors_for_index();

            // Swap in the new (empty) brute-force index
            table_data.index = new_wrapper.clone();

            // Say what the table now searches with. Leaving the old algorithm in place made the
            // table report an index it no longer had, and made `rebuild_index` see a brute-force
            // index holding every row as up to date and skip the rebuild that `DELETE /index`
            // tells the caller to run -- measured: `test_rebuild_index` failed with "the rebuild
            // after a drop was skipped" before this. A snapshot taken now records brute force, so
            // the drop also survives a restart instead of being silently undone by it.
            table_data.selected_algorithm = IndexAlgorithm::BruteForce {
                enable_parallel: true,
            };
            table_data.table.index_type = crate::types::table::IndexType::BruteForce;

            (new_wrapper, records_snapshot)
        };

        // Phase 2: re-populate the brute-force index outside the write lock
        for (id, vector) in records_snapshot {
            new_wrapper.insert(id, vector).await?;
        }

        Ok(())
    }

    /// Build `tablename`'s payload and sparse indexes outside the table map's lock if they are
    /// not built yet, so a caller that goes on to read them under the lock finds them there
    /// instead of building them inside it. Every read path that needs them calls this first.
    async fn ensure_derived(&self, tablename: &str) -> crate::Result<()> {
        let unbuilt = self
            .read_tables("ensure_derived")
            .await
            .get(tablename)
            .is_some_and(|data| data.derived.read().is_none());
        if unbuilt {
            self.warm_derived(tablename).await?;
        }
        Ok(())
    }

    /// Build a table's payload and sparse indexes, and fill its node map, without holding the
    /// table map's lock for the build: the rows are decoded a chunk at a time under short read
    /// locks. `Ok(true)` when it installed something.
    ///
    /// Run ahead of the first filtered query after a restart (the server's warm-up), and by that
    /// query itself when the warm-up has not finished ([`Self::ensure_derived`]). Before this
    /// the query built inside `IntelligentTableData::derived`, under the map's read lock -- 27 s
    /// at 10M rows during which a waiting writer queued every later request to every table.
    ///
    /// Writes during the build are caught up rather than voiding it. A write that finds the
    /// indexes unbuilt notes the id and the payload it held before its first change
    /// (`note_unindexed_change`). The build scans the rows that existed when it started, so
    /// for a noted id it holds either nothing or exactly that earlier payload; under the read
    /// lock that keeps writers out, each noted id is taken out with its earlier payload and
    /// put back as the store now has it, and the rows appended meanwhile get their creation
    /// times. A write used to void the build, which on a server taking writes was every build.
    pub async fn warm_derived(&self, tablename: &str) -> crate::Result<bool> {
        let tablename = &*self.resolve(tablename);
        let (built_already, build_lock) = {
            let tables = self.read_tables("warm_derived").await;
            let data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
            if data.records.row_count().is_none() {
                // In memory: small by construction, where the lazy build is a map walk.
                return Ok(false);
            }
            let built = data.derived.read().is_some();
            (built, Arc::clone(&data.derived_build))
        };
        if built_already {
            // Read from disk at startup (`load_persisted_derived`): only the node map is left.
            let map_in = self.fill_node_map(tablename).await;
            debug!(
                "Table '{tablename}': node map filled in {:.1}s",
                map_in.as_secs_f64()
            );
            return Ok(false);
        }
        let _building = build_lock.lock().await;

        let (rows, policy) = {
            let tables = self.read_tables("warm_derived").await;
            let data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
            if data.derived.read().is_some() {
                return Ok(false); // the build we waited behind installed it
            }
            // From here every write that skips the unbuilt indexes is noted.
            *data.derived_changes.lock() = Some(HashMap::new());
            (
                data.records.row_count().unwrap_or(0),
                data.table.payload_index.clone(),
            )
        };
        let started = std::time::Instant::now();
        let mut built = DerivedIndexes::default();
        if let Some(policy) = policy.clone() {
            built.payload.set_policy(policy);
        }
        built.payload.begin_bulk(rows as usize);
        const CHUNK: u32 = 16_384;
        let mut from = 0u32;
        let mut created: Vec<i64> = Vec::with_capacity(rows as usize);
        let mut records = 0usize;
        while from < rows {
            let to = (from + CHUNK).min(rows);
            let chunk = {
                let tables = self.read_tables("warm_derived").await;
                let Some(data) = tables.get(tablename) else {
                    return Ok(false);
                };
                tokio::task::block_in_place(|| data.records.decode_range(from, to))
                    .unwrap_or_default()
            };
            for slot in chunk {
                match slot {
                    Some(record) => {
                        records += 1;
                        created.push(crate::storage::payload_index::created_key(
                            &record.created_at,
                        ));
                        built.payload.insert_record(
                            &record.id,
                            &record.payload,
                            crate::storage::payload_index::created_key(&record.created_at),
                        );
                        if let Some(sparse) = &record.sparse_vector {
                            if let Err(e) = built.sparse.insert(record.id.clone(), sparse) {
                                warn!("Failed to index sparse vector for {}: {}", record.id, e);
                            }
                        }
                    }
                    None => created.push(i64::MIN),
                }
            }
            from = to;
            tokio::task::yield_now().await;
        }
        tokio::task::block_in_place(|| built.payload.end_bulk());

        let (installed, caught_up) = {
            // A read lock: every write that touches these indexes holds the write lock, so
            // none can land between the catch-up and the install.
            let tables = self.read_tables("warm_derived").await;
            let Some(data) = tables.get(tablename) else {
                return Ok(false);
            };
            let Some(changes) = data.derived_changes.lock().take() else {
                // Not the table this build started on: it was dropped and created again.
                return Ok(false);
            };
            if data.derived.read().is_some() || data.table.payload_index != policy {
                // Built by the lazy path meanwhile, or the policy changed under the build.
                return Ok(false);
            }
            let caught_up = changes.len();
            tokio::task::block_in_place(|| {
                for (id, before) in changes {
                    if built.payload.ordinal_of(&id).is_some() {
                        if let Some(before) = &before {
                            built.payload.remove_record(&id, before);
                        }
                        let _ = built.sparse.delete(&id);
                    }
                    if let Some(record) = data.records.get(&id) {
                        built.payload.insert_record(
                            &id,
                            &record.payload,
                            crate::storage::payload_index::created_key(&record.created_at),
                        );
                        if let Some(sparse) = &record.sparse_vector {
                            if let Err(e) = built.sparse.insert(id.clone(), sparse) {
                                warn!("Failed to index sparse vector for {}: {}", id, e);
                            }
                        }
                    }
                }
                // Rows appended during the build, which the scan did not reach.
                let now = data.records.row_count().unwrap_or(rows);
                for slot in data.records.decode_range(rows, now).unwrap_or_default() {
                    created.push(slot.map_or(i64::MIN, |record| {
                        crate::storage::payload_index::created_key(&record.created_at)
                    }));
                }
            });
            data.records.publish_created(created);
            let mut slot = data.derived.write();
            if slot.is_none() {
                *slot = Some(built);
                (true, caught_up)
            } else {
                (false, caught_up)
            }
        };
        if !installed {
            return Ok(false);
        }
        if caught_up > 0 {
            debug!("warm_derived('{tablename}'): caught up {caught_up} record(s) written during the build");
        }
        let built_in = started.elapsed();
        let map_in = self.fill_node_map(tablename).await;
        info!(
            "Table '{tablename}': payload and sparse indexes warmed from {records} records in {:.1}s, node map in {:.1}s",
            built_in.as_secs_f64(),
            map_in.as_secs_f64()
        );
        Ok(true)
    }

    /// Fill a table's node map from its graph, so the first filtered query does not pay for
    /// it, and return how long that took.
    async fn fill_node_map(&self, tablename: &str) -> std::time::Duration {
        // The node map too, so the first filtered query does not pay that either -- in
        // slices, releasing the table lock between them: one pass over 10M nodes held it
        // for 3.2 s and queued every writer behind it.
        let map_started = std::time::Instant::now();
        loop {
            let done = {
                let tables = self.read_tables("warm_derived").await;
                let Some(data) = tables.get(tablename) else {
                    break;
                };
                let index = data.index.clone_arc();
                let generation = data.index.generation();
                let Ok(graph) = index.try_read() else {
                    break;
                };
                let mut map = data.node_map.write().await;
                let derived = data.derived();
                let payload = &derived.payload;
                const SLICE: usize = 200_000;
                let before = map.len();
                let supported = tokio::task::block_in_place(|| {
                    map.fill_up_to(generation, &**graph, |id| payload.ordinal_of(id), SLICE)
                });
                !supported || map.len() == before || map.len() >= graph.slot_count()
            };
            if done {
                break;
            }
            tokio::task::yield_now().await;
        }
        map_started.elapsed()
    }

    /// Warm every table's derived indexes in the background, one table at a time, once the
    /// storage is open. Called by the server after recovery; a table that is written to
    /// before its turn is simply left to the lazy build.
    pub fn spawn_derived_warmup(storage: Arc<Self>) {
        tokio::spawn(async move {
            let tables = match storage.list_tables().await {
                Ok(tables) => tables,
                Err(e) => {
                    warn!("derived-index warm-up skipped: {e}");
                    return;
                }
            };
            for table in tables {
                if let Err(e) = storage.warm_derived(&table.name).await {
                    warn!("derived-index warm-up for '{}' failed: {e}", table.name);
                }
            }
        });
    }

    /// Give a quantized table the block-scan codes (`quantization::fast_scan`) when it has
    /// none -- a table quantized before they existed, or whose sidecar was lost. The
    /// vectors come from the row file, node by node, and the codebooks are fitted to a
    /// stride sample of them; the index is only written to install the finished codes.
    /// `Ok(true)` when built, `Ok(false)` when there was nothing to do or the graph grew
    /// under the build (it is retried by the next call).
    pub async fn ensure_fast_scan(&self, tablename: &str) -> crate::Result<bool> {
        use crate::search::quantization::fast_scan::FastScan;
        let tablename = &*self.resolve(tablename);
        self.ensure_resident(tablename).await?;
        let (index, metric, dim) = {
            let tables = self.read_tables("ensure_fast_scan").await;
            let data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
            if let IndexAlgorithm::HnswPQ {
                fast_scan: false, ..
            } = data.selected_algorithm
            {
                return Ok(false);
            }
            (
                data.index.clone_arc(),
                data.table.distance_metric,
                data.table.dimension,
            )
        };
        let slots = {
            let guard = index.read().await;
            let Some(graph) = guard.as_hnsw() else {
                return Ok(false);
            };
            if !graph.is_quantized() || graph.has_fast_scan() {
                return Ok(false);
            }
            guard.slot_count()
        };
        if slots == 0 || FastScan::subvector_dim_for(dim).is_none() {
            return Ok(false);
        }
        // The sample: up to 65,536 rows at a stride.
        const SAMPLE: usize = 65_536;
        let stride = (slots / SAMPLE).max(1);
        let mut sample: Vec<Vec<f32>> = Vec::with_capacity(SAMPLE);
        let mut at = 0;
        while at < slots && sample.len() < SAMPLE {
            let to = (at + 4_096).min(slots);
            let rows = self
                .fast_scan_rows(tablename, &index, metric, at, to)
                .await?;
            for v in rows.into_iter().step_by(stride).flatten() {
                sample.push(v);
                if sample.len() >= SAMPLE {
                    break;
                }
            }
            at = to;
        }
        if sample.len() < 1_024 {
            return Ok(false);
        }
        let started = std::time::Instant::now();
        let mut fast = tokio::task::block_in_place(|| {
            FastScan::train(dim, DistanceMetric::Euclidean, &sample, 25)
        })?;
        let zero = vec![0u8; fast.subspaces()];
        let mut from = 0;
        while from < slots {
            let to = (from + 65_536).min(slots);
            let rows = self
                .fast_scan_rows(tablename, &index, metric, from, to)
                .await?;
            let codes: Vec<Vec<u8>> = tokio::task::block_in_place(|| {
                use rayon::prelude::*;
                rows.par_iter()
                    .map(|v| match v {
                        Some(v) => fast.encode(v).unwrap_or_else(|_| zero.clone()),
                        None => zero.clone(),
                    })
                    .collect()
            });
            for code in &codes {
                fast.push_code(code);
            }
            from = to;
        }
        let mut guard = index.write().await;
        if guard.slot_count() != fast.len() {
            info!(
                "fast-scan for '{tablename}': the graph grew from {} to {} nodes during the build; not installed",
                fast.len(),
                guard.slot_count()
            );
            return Ok(false);
        }
        let Some(graph) = guard.as_hnsw_mut() else {
            return Ok(false);
        };
        if graph.has_fast_scan() {
            return Ok(false);
        }
        graph.set_fast_scan(Some(fast));
        info!(
            "fast-scan for '{tablename}': {} codes built from the row file in {:.1}s",
            slots,
            started.elapsed().as_secs_f64()
        );
        Ok(true)
    }

    /// The metric-prepared vectors of node positions `from..to`, in position order, from
    /// the row file; `None` for a tombstoned node or a row that is gone.
    async fn fast_scan_rows(
        &self,
        tablename: &str,
        index: &Arc<tokio::sync::RwLock<Box<dyn crate::search::index_traits::VectorIndex>>>,
        metric: DistanceMetric,
        from: usize,
        to: usize,
    ) -> crate::Result<Vec<Option<Vec<f32>>>> {
        let (ids, deleted): (Vec<RecordId>, Vec<bool>) = {
            let guard = index.read().await;
            let graph = guard
                .as_hnsw()
                .ok_or_else(|| crate::Error::Internal("index changed under the build".into()))?;
            (
                (from..to)
                    .map(|idx| graph.node_id(idx).to_string())
                    .collect(),
                (from..to).map(|idx| graph.is_deleted(idx)).collect(),
            )
        };
        let tables = self.read_tables("ensure_fast_scan").await;
        let data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        let found: HashMap<RecordId, Vec<f32>> = data.vectors_for_ids(&ids).into_iter().collect();
        Ok(ids
            .iter()
            .zip(deleted)
            .map(|(id, gone)| {
                if gone {
                    return None;
                }
                let mut v = found.get(id).cloned()?;
                // Metric-prepared, as the graph holds them: unit length under cosine.
                if matches!(metric, DistanceMetric::Cosine) {
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        v.iter_mut().for_each(|x| *x /= norm);
                    }
                }
                Some(v)
            })
            .collect())
    }

    /// Rebuild the vector index from scratch using all persisted records.
    ///
    /// This is the recovery path after `drop_index`, a server restart, or any
    /// situation where the in-memory index is empty but records still exist.
    /// The algorithm is re-selected adaptively based on the current record count
    /// and configured memory budget.
    ///
    /// Implementation is two-phase to minimise lock hold time:
    /// 1. Snapshot records + choose algorithm + swap in a fresh empty index (write lock).
    /// 2. Re-insert all record vectors into the new index (outside lock).
    pub async fn rebuild_index(&self, table_name: &str) -> crate::Result<()> {
        let table_name = &*self.resolve(table_name);
        let started = std::time::Instant::now();

        // Phase 1: snapshot + select + swap (write lock)
        let t_phase1 = std::time::Instant::now();
        let (new_wrapper, records_snapshot, algorithm, decision) = {
            let tables = self.read_tables("rebuild_index").await;

            let table_data = tables
                .get(table_name)
                .ok_or_else(|| crate::Error::TableNotFound(table_name.to_string()))?;

            let dimension = table_data.table.dimension;
            let distance_metric = table_data.table.distance_metric;
            let record_count = table_data.records.len();

            // One decision function, shared with reevaluate_indexes. Keeping a copy of the
            // partitioning rule here made a partitioned table look wrong to every re-evaluation
            // sweep, which rebuilt it, got a partitioned index back, and repeated — forever.
            let decision = self
                .decide_algorithm(
                    record_count,
                    dimension,
                    distance_metric,
                    self.memory_optimization_of(&table_data.table),
                    table_data.table.hnsw.as_ref(),
                    table_data.table.quantization.as_ref(),
                )
                .await?;

            let algorithm = decision.selected_algorithm.clone();

            // Nothing to do when the rebuild would reproduce the graph already serving.
            //
            // Measured on Cohere 1M x 768, recall@100: once `select_memory_optimized_index`
            // and `index_selector` agree on `ef_construction`, the graph insert builds
            // incrementally scores 0.9582 and one rebuilt from scratch here scores 0.9591 --
            // 0.09 points, inside this machine's noise band, for about 1,259 s of work. The
            // rebuild looked essential only while the create-table path built at
            // ef_construction 64 and this one at 200, worth 8.5 points; rebuilding every time
            // was paying for that divergence forever instead of fixing it.
            //
            // The test is `requires_rebuild_to_reach`, not equality. Equality never holds:
            // the two selection paths used to hand out different `ef_search` (128 against 256,
            // both 256 now), and `ef_search` is a per-query traversal budget rather than
            // anything baked into the graph -- which is precisely why that method leaves it out and
            // `adopt_query_time_params` exists to write it in place. An equality check here
            // compiles, passes a small test, and then never fires on a real table.
            //
            // The row-count check catches an index that has fallen behind its record map,
            // from a restore that rebuilt from rows or a build that died partway. That is the
            // case a rebuild repairs, and it must still run.
            // An HNSW-PQ index that streamed in below its quantization threshold holds every
            // record as an exact graph; the rebuild is what trains and quantizes it, so it is
            // not skipped as "equivalent". The first 1M acceptance run was skipped this way and
            // measured the exact graph -- 8,295 MB resident, the same as plain HNSW.
            // A graph holding twice the slots it needs is not equivalent to a freshly built
            // one, whatever the algorithm says: it is the same answers out of several times
            // the memory. Without this clause the sweep asked for a compaction and this
            // function declined it, because every other test it makes looks at live entries
            // and the tombstones are not live.
            let (live, slots) = table_data.index.live_and_slots().await;
            let up_to_date = !table_data
                .selected_algorithm
                .requires_rebuild_to_reach(&algorithm)
                && table_data.index.len().await == record_count
                && !table_data.index.wants_rebuild().await
                && !worth_compacting(live, slots);
            if up_to_date {
                drop(tables);
                let mut tables = self.write_tables("rebuild_index").await;
                let table_data = tables
                    .get_mut(table_name)
                    .ok_or_else(|| crate::Error::TableNotFound(table_name.to_string()))?;
                let adopted = table_data
                    .selected_algorithm
                    .adopt_query_time_params(&algorithm);
                info!(
                    "rebuild_index: table='{}' already holds all {} records under an \
                     equivalent graph; skipped the rebuild{}",
                    table_name,
                    record_count,
                    if adopted {
                        " (adopted the new ef_search in place)"
                    } else {
                        ""
                    }
                );
                return Ok(());
            }

            let table_id = table_data.table.id.to_string();
            let new_index =
                self.create_index_instance_for(&algorithm, dimension, distance_metric, &table_id)?;
            let new_wrapper = crate::search::IndexWrapper::new(new_index);

            let records_snapshot: Vec<(RecordId, Vec<f32>)> = table_data.vectors_for_index();

            // Deliberately *not* swapped in yet. Publishing the empty index here and filling
            // it afterwards does not expose a half-built index — IndexWrapper's own RwLock
            // makes readers wait instead — but that is its own problem: queries then block on
            // the rebuild. Measured at 3,000 vectors, a single search waited 102ms against a
            // 243ms rebuild. The old index stays live until the replacement is complete.
            info!(
                "rebuild_index: table='{}' records={} algorithm={:?}",
                table_name, record_count, algorithm
            );

            (new_wrapper, records_snapshot, algorithm, decision)
        };
        let t_phase1 = t_phase1.elapsed();

        // Phase 2: build the replacement outside any lock. bulk_build rather than a loop of
        // inserts, because a partitioned index has to see every vector at once to cluster it —
        // inserting one at a time would leave the lot in its overflow buffer.
        let t_build = std::time::Instant::now();
        let indexed: std::collections::HashSet<RecordId> =
            records_snapshot.iter().map(|(id, _)| id.clone()).collect();
        new_wrapper.bulk_build(records_snapshot).await?;
        let t_build = t_build.elapsed();

        // Phase 3: converge, then swap. The catch-up runs *outside* the lock and repeats,
        // because each pass can only be overtaken by writes that landed while it ran — and
        // those are fewer each time. Only the final delta is applied under the write lock.
        //
        // Doing the whole catch-up under that lock, as this did, stalls every table: it is the
        // global table map's lock, not this table's. Converging outside the lock avoids that
        // stall without needing a separate segment to absorb writes during the rebuild.
        let t_catchup = std::time::Instant::now();
        let mut indexed = indexed;
        let mut caught_up = 0usize;

        // Bounded so a table under continuous write load cannot spin here forever. After this
        // many passes the remaining delta is applied under the lock however large it is —
        // finishing slowly beats not finishing.
        const MAX_CONVERGENCE_PASSES: usize = 8;
        for _ in 0..MAX_CONVERGENCE_PASSES {
            let missing: Vec<(RecordId, Vec<f32>)> = {
                let tables = self.read_tables("rebuild_index").await;
                let table_data = tables
                    .get(table_name)
                    .ok_or_else(|| crate::Error::TableNotFound(table_name.to_string()))?;
                table_data
                    .records
                    .ids()
                    .into_iter()
                    .filter(|id| !indexed.contains(id))
                    .filter_map(|id| {
                        let r = table_data.records.get(&id)?;
                        Some((id, table_data.vector_for(&r)))
                    })
                    .collect()
            };
            if missing.is_empty() {
                break;
            }
            for (id, vector) in missing {
                new_wrapper.insert(id.clone(), vector).await?;
                indexed.insert(id);
                caught_up += 1;
            }
        }

        let t_catchup = t_catchup.elapsed();

        let t_swap_wait = std::time::Instant::now();
        {
            let mut tables = self.write_tables("rebuild_index").await;
            let t_swap_wait = t_swap_wait.elapsed();
            let table_data = tables
                .get_mut(table_name)
                .ok_or_else(|| crate::Error::TableNotFound(table_name.to_string()))?;

            // Whatever arrived since the last pass. No write can slip between this and the swap
            // because both happen under the same lock acquisition.
            let mut final_delta = 0usize;
            for id in table_data.records.ids() {
                if !indexed.contains(&id) {
                    let Some(record) = table_data.records.get(&id) else {
                        continue;
                    };
                    new_wrapper
                        .insert(id.clone(), table_data.vector_for(&record))
                        .await?;
                    final_delta += 1;
                }
            }

            // Unconditional, because this is the only line that says the swap happened. With it
            // conditional on there being a delta, a rebuild that was still running looked
            // identical to one that had finished: the start line and nothing after it. A 37k-row
            // HNSW rebuild ran for 47s under write load and I could not tell which it was.
            // Broken out by phase. A rebuild that ran for 101s and said only "101s" cost me a
            // reproduction cycle: the phases accuse different code, and only the swap wait
            // implicates lock contention rather than the build itself.
            info!(
                "rebuild_index: table='{}' swapped in after {:.2}s (select {:.2}s, build {:.2}s, \
                 catch-up {:.2}s for {} records, waited {:.2}s for the write lock, {} under it)",
                table_name,
                started.elapsed().as_secs_f64(),
                t_phase1.as_secs_f64(),
                t_build.as_secs_f64(),
                t_catchup.as_secs_f64(),
                caught_up,
                t_swap_wait.as_secs_f64(),
                final_delta
            );

            self.index_rebuilds.fetch_add(1, Ordering::Relaxed);
            self.index_rebuild_ns
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            self.index_rebuild_swap_wait_ns
                .fetch_add(t_swap_wait.as_nanos() as u64, Ordering::Relaxed);

            table_data.index = new_wrapper.clone();
            table_data.table.index_type = index_type_of(&algorithm);
            table_data.selected_algorithm = algorithm;
            table_data.index_decision = decision;
        }

        // A rebuild is the one moment we know the index is both current and expensive to
        // reproduce, so snapshot it. Failure is logged, not propagated: the rebuild itself
        // succeeded, and a missing snapshot only costs startup time.
        if self.wal.is_some() {
            if let Err(e) = self.snapshot().await {
                warn!("Index rebuilt but snapshot failed: {}", e);
            }
        }

        Ok(())
    }

    /// Create a named branch (full copy) of `source_table` as `branch_name`.
    ///
    /// The branch is a fully independent table with the same schema and records as
    /// the source at the moment of branching. Subsequent writes to either table do
    /// not affect the other — this is a snapshot, not copy-on-write.
    ///
    /// Snapshot semantics are chosen over copy-on-write because branches here are
    /// short-lived experiment targets, where write isolation matters more than space.
    pub async fn branch_table(
        &self,
        source_table: &str,
        branch_name: &str,
    ) -> crate::Result<Table> {
        // Snapshot source records under the read lock
        let (source_config, records_snapshot) = {
            let tables = self.read_tables("branch_table").await;
            let src = tables
                .get(source_table)
                .ok_or_else(|| crate::Error::TableNotFound(source_table.to_string()))?;

            if tables.contains_key(branch_name) {
                return Err(crate::Error::TableAlreadyExists(branch_name.to_string()));
            }

            let config = TableConfig {
                name: branch_name.to_string(),
                dimension: src.table.dimension,
                distance_metric: src.table.distance_metric,
                description: Some(format!(
                    "Branch of '{}' at {}",
                    source_table,
                    chrono::Utc::now()
                )),
                index_type: None,
                text_fields: src.table.text_fields.clone(),
                enable_full_text_search: src.table.enable_full_text_search,
                memory_optimization: src.table.memory_optimization,
                // A branch is a copy: the same graph parameters, or it would be a copy of the
                // data under a different index and useless for comparing anything.
                hnsw: src.table.hnsw.clone(),
                quantization: src.table.quantization.clone(),
                payload_index: src.table.payload_index.clone(),
                payload_schema: src.table.payload_schema.clone(),
                schema_version: src.table.schema_version,
                named_vectors: None,
                // A copy embeds the way its source does. The records below carry their vectors,
                // so nothing is re-embedded while branching; the model matters for what is
                // written to the branch afterwards.
                embedding: src.table.embedding.clone(),
                rerank: src.table.rerank.clone(),
            };

            let records: Vec<CreateRecordRequest> = src
                .records
                .values_cloned()
                .iter()
                .map(|r| CreateRecordRequest {
                    id: r.id.clone(),
                    vector: src.vector_for(r),
                    sparse_vector: r.sparse_vector.clone(),
                    payload: Some(r.payload.clone()),
                    named_vectors: None,
                    text: None,
                })
                .collect();

            (config, records)
        };

        // Create the branch table (reuses existing create_table + insert_records)
        self.create_table(source_config).await?;
        if !records_snapshot.is_empty() {
            self.insert_records(branch_name, records_snapshot).await?;
        }
        // Described after the copy. Returning what `create_table` handed back reported a branch
        // of eight records as `record_count: 0`.
        self.get_table(branch_name).await
    }

    /// 列出表中所有记录 ID（用于一致性检查和数据迁移）
    pub async fn list_all_record_ids(&self, tablename: &str) -> crate::Result<Vec<RecordId>> {
        let tablename = &*self.resolve(tablename);
        let tables = self.read_tables("list_all_record_ids").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        Ok(table_data.records.ids())
    }

    /// 获取向量索引中的记录数（用于一致性检查）
    pub async fn get_index_record_count(&self, tablename: &str) -> crate::Result<usize> {
        let tablename = &*self.resolve(tablename);
        let tables = self.read_tables("get_index_record_count").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        Ok(table_data.index.len().await)
    }

    /// 表的规模与内存占用统计。
    ///
    /// 字段名对齐 dxmt-js-sdk 的 `TableStats`——它的 `getTableStats()` 一直在调用一个不存在的
    /// 路由，所以这个方法对每个 JS 客户端用户都是抛异常的。
    pub async fn get_table_stats(&self, tablename: &str) -> crate::Result<serde_json::Value> {
        let tablename = &*self.resolve(tablename);
        // Residency *before* touching it, or the answer is always true — asking is what makes
        // it resident.
        let was_resident = self
            .read_tables("get_table_stats")
            .await
            .contains_key(tablename);
        self.ensure_resident(tablename).await?;

        let tables = self.read_tables("get_table_stats").await;
        let data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let index_stats = data.index.get_stats().await;
        // Which regime this table's index is in. An HNSW-PQ index is an exact graph until it
        // has enough rows to train codebooks on, and everything about recall and memory
        // changes when it crosses: a table that answered exactly yesterday answers
        // approximately today, and nothing said so.
        let quantized = data.index.is_quantized().await;
        let fast_scan = data.index.has_fast_scan().await;
        let index_record_count = data.index.len().await;
        let dimension = data.table.dimension;
        let record_count = data.records.len();

        // Resident memory, counted where it is. Records in the map carry their payloads (the
        // serialized length is the nearest size to hand) and, unless the table keeps vectors in
        // a row file, their vectors; a disk store holds only its id table, offsets and
        // tombstones. This used to add `record_count * dimension * 4` whatever the table was,
        // counting as memory the vectors that live in a row file.
        let record_bytes = match data.records.disk_store_resident_bytes() {
            Some(resident) => resident,
            None => {
                let mut bytes = 0usize;
                data.records.for_each(|r| {
                    bytes += r.id.len()
                        + r.vector.len() * std::mem::size_of::<f32>()
                        + serde_json::to_string(&r.payload)
                            .map(|s| s.len())
                            .unwrap_or(0)
                });
                bytes
            }
        };
        let vector_row_bytes: usize = data
            .vector_rows
            .keys()
            .map(|id| std::mem::size_of::<(RecordId, u64)>() + id.len())
            .sum();

        // The row files are this table's own. The WAL and snapshots are shared by every table
        // and are not attributed to one. This was a literal 0, including in disk mode, where
        // the records and vectors are nowhere but in those files.
        let disk_bytes = data.records.disk_bytes()
            + data
                .vector_file
                .as_ref()
                .map_or(0, |file| file.disk_bytes());

        Ok(serde_json::json!({
            "name": data.table.name,
            // Whether this table was in memory before the call. A query against an evicted
            // table rehydrates it transparently, which is convenient and hides a latency cliff:
            // the caller has no way to know the next query will pay for a fetch. Exposing the
            // state is the least we can do; refusing the query until reactivation would be the
            // stricter alternative.
            "resident": was_resident,
            "record_count": record_count,
            // How many of those the *index* holds. It should equal `record_count`, and the
            // day it does not is the day searches start missing records that `GET` returns
            // and `search_mode: "exact"` finds -- a shortfall with no other symptom, because
            // a record missing from the graph is simply never a result. The count existed on
            // the storage type and was not served, so the one number that identifies this
            // class of fault was the one an operator could not ask for.
            "index_record_count": index_record_count,
            "dimension": dimension,
            "index_type": data.table.index_type.to_string(),
            "quantized": quantized,
            "fast_scan": fast_scan,
            "index_size_bytes": index_stats.index_size_bytes,
            "memory_usage_bytes": record_bytes + vector_row_bytes + index_stats.index_size_bytes,
            "disk_usage_bytes": disk_bytes,
            // The budget `normal` actually searches with, and where it came from. The
            // configured value alone said nothing about what a query on this corpus gets.
            "ef_search": {
                "configured": data.selected_algorithm.configured_ef_search(),
                "calibrated": data.table.recall.as_ref().map(|c| c.ef_search),
                "recall_target": data.table.recall_target.unwrap_or_else(Self::default_recall_target),
            },
            "recall": data.table.recall,
            // Nodes the last repair pass found with no in-edge -- in the graph and unreachable
            // by any query until the pass gave them one -- and how many every pass has
            // repaired. A non-zero `repaired_total` on a table is the graph doing what the
            // heuristic's pruning undid; see `HnswIndex::repair_reachability`.
            "reachability": {
                "stranded_at_last_repair": data.stranded_last.load(Ordering::Relaxed),
                "repaired_total": data.repaired_total.load(Ordering::Relaxed),
            },
        }))
    }

    /// 获取表的性能统计
    pub async fn get_performance_stats(&self, tablename: &str) -> crate::Result<PerformanceStats> {
        let tablename = &*self.resolve(tablename);
        let tables = self.read_tables("get_performance_stats").await;

        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let stats = table_data
            .performance_stats
            .lock()
            .map_err(|_| crate::Error::Internal("PerformanceStats lock poisoned".to_string()))?;
        Ok(stats.clone())
    }

    /// Default traversal budget for a table created here. It matches `index_selector`, which
    /// hands out 256 on a rebuild; until it did, a table that had never been through
    /// `optimize()` searched at 128 and one that had searched at 256, and recall depended on
    /// whether the caller happened to have called that endpoint. Measured on Cohere 1M x 768,
    /// ef_construction 200, same graph (design doc section 24.6.1):
    ///
    /// | ef_search | 128            | 256            | 512            | 1024           |
    /// |-----------|----------------|----------------|----------------|----------------|
    /// | recall@10 | 0.911 @ 3,637  | 0.958 @ 2,200  | 0.991 @ 1,301  | 0.991 @ 800    |
    ///
    /// 256 is where the benchmark's recall clears 0.95; past 512 the extra work buys nothing.
    /// `SearchMode::Accurate` is 4x this per query and `SearchMode::Fast` 1x; neither needs a
    /// rebuild -- `ef_search` is a per-query budget, not part of the graph.
    ///
    /// 512 rather than 256 because 256 was never what ran. `search_reranked` floors the walk
    /// at the rerank pool, `ef.max(fetch)` with `fetch = rerank_candidates(k) = max(5k, 100)`,
    /// so every recall@100 measurement this repo has ever taken walked at 500, not 256. The
    /// constant said one thing and the code did another; this makes them agree, and 512 is
    /// close enough to 500 that no measured number moves.
    const DEFAULT_EF_SEARCH: usize = 512;

    /// Rows per decade of `ef_search` growth, and the ceiling on that growth.
    ///
    /// A fixed `ef` does not hold recall as the graph grows: measured on 768-dimension
    /// embeddings (`docs/architecture/disk-index.md` §5), one graph reaches recall@100 0.9800
    /// at 1M with ef 500, but only 0.9638 at 10M with the same budget -- it needs 1024 there
    /// for 0.9800 and 2048 for 0.9887.
    ///
    /// A segmented engine gets this growth without naming it: splitting 10M into 8 segments
    /// and walking each at ef 256 is a 2,048-candidate budget in total, which is why that
    /// shape holds its recall at 10M where a single graph at 500 does not. At a *matched*
    /// total budget the single graph is the better of the two, so the fix is to spend the
    /// budget, not to shard.
    ///
    /// One doubling per decade of rows: 1M keeps 512, 10M gets 1024, 100M gets 2048.
    const EF_SCALE_BASE_ROWS: usize = 1_000_000;
    const EF_SCALE_MAX: usize = 8;

    /// The traversal budget for a table of `rows`, from the budget it was configured with.
    ///
    /// Below `EF_SCALE_BASE_ROWS` this is the configured value unchanged -- a small table's
    /// graph is short and a bigger `ef` only costs.
    fn ef_for_rows(configured: usize, rows: usize) -> usize {
        let mut scale = 1usize;
        let mut rows_at_scale = Self::EF_SCALE_BASE_ROWS;
        while scale < Self::EF_SCALE_MAX {
            let next = match rows_at_scale.checked_mul(10) {
                Some(n) => n,
                None => break,
            };
            if rows < next {
                break;
            }
            scale *= 2;
            rows_at_scale = next;
        }
        configured.saturating_mul(scale)
    }

    /// Graph build effort for the index a table gets when it is created.
    ///
    /// This is the path a table actually takes: `select_memory_optimized_index` runs at
    /// create-table time and `memory_optimization` defaults to true globally, so it is what
    /// every record inserted before the first rebuild is linked into. It used to be 64 while
    /// `index_selector.rs` used 200 for the same table -- and a comment there asserted the two
    /// paths matched, so nothing looked wrong.
    ///
    /// Measured on Cohere 1M x 768, recall@100 with no rebuild forced: a graph built at 64
    /// scored 0.874, against 0.9591 once rebuilt at 200. Unlike `ef_search`, this cannot be
    /// recovered per query -- `SearchMode::Accurate` searches harder, it does not add edges
    /// the build never made. A user who streams data in and queries it was reading a graph
    /// built at less than a third of the effort the rest of the codebase had settled on.
    const DEFAULT_EF_CONSTRUCTION: usize = 200;

    /// `DXMT_HNSW_EF_CONSTRUCTION` overrides the build effort for tables created from here on.
    /// It exists to measure the trade: on Cohere 1M the neighbour search at 200 is 88% of the
    /// insert time (design doc section 24.4.3), and the only way to know what 100 costs in
    /// recall is to build at 100. Rebuilds do not read it; skip the rebuild when comparing.
    fn default_ef_construction() -> usize {
        crate::config::env::var("DXMT_HNSW_EF_CONSTRUCTION")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(Self::DEFAULT_EF_CONSTRUCTION)
    }

    /// Candidate-set size below which a filtered search scores the candidates directly instead
    /// of walking the graph.
    ///
    /// Measured on 20K x 32 with `bucket` filters (200 queries each, release build): at 5%
    /// selectivity the graph walk was 1.6 ms p50, at 2% it was 24.3 ms and at 1% 27.6 ms —
    /// traversal cost rises as the filter gets more selective, because the walk spends its
    /// budget on nodes it must discard. Scoring the candidates is O(|candidates| x dim) and
    /// exact. 2048 keeps the direct path bounded at roughly the number of nodes a walk with
    /// `ef = 128` and `m = 16` would touch regardless.
    const EXACT_FILTER_SCAN_MAX: usize = 2048;
    /// The same ceiling when the vectors are in a row file: about one rerank pool of rows.
    const EXACT_FILTER_SCAN_ON_DISK: usize = 128;

    /// Largest match set scored node by node rather than walked. Scoring costs one code per
    /// match; a filtered walk visits about `ef / selectivity` nodes to collect `ef` matching
    /// ones, and a visit costs several times a code score (the heap, the visited set, the
    /// links). The two cross where `m * c_scan = ef * rows / m * c_visit`, so at
    /// `m = sqrt(ef * rows * c_visit / c_scan)`: with `ef` 512 and `WALK_VISIT_IN_SCORES`
    /// scores to a visit, 157,000 on 2M rows and 350,000 on 10M. Clamped so a tiny or a
    /// huge table still gets a sane answer.
    fn code_scan_max(rows: usize) -> usize {
        ((512.0 * rows as f64 * Self::WALK_VISIT_IN_SCORES).sqrt() as usize)
            .clamp(Self::CODE_SCAN_MAX / 10, Self::CODE_SCAN_MAX * 10)
    }

    /// What one visit of the filtered walk costs in code scores: every neighbour of a
    /// visited node is scored to steer the walk, matching or not, so a visit is about the
    /// node's degree. Measured on the big-ann filtered track at 2M rows: 205,000 matched
    /// rows walked in 17 ms against 116 ns per scored code, which puts the crossover near
    /// 170,000 and this constant at 24.
    const WALK_VISIT_IN_SCORES: f64 = 24.0;

    /// Below this estimated selectivity the filter's matching ids are materialised before
    /// the branch is chosen, so the choice rests on the real match count and not on the
    /// estimate; from here up an amplified unfiltered fetch is cheap enough to try first.
    const FILTER_MATERIALISE_BELOW: f64 = 0.3;

    /// Largest matching set scored from the index's resident codes instead of walked. At
    /// a few hundred nanoseconds a code, 100,000 is tens of milliseconds -- past that a
    /// walk with the filter applied in-graph is the cheaper shape.
    const CODE_SCAN_MAX: usize = 100_000;

    /// Largest number of rows `SearchMode::Exact` will scan before refusing the query.
    ///
    /// Unlike the two caps above, this one does not pick between strategies -- there is no
    /// cheaper shape to fall back to, because the whole point of the mode is that it does not
    /// approximate. So it refuses instead, and says how many rows it would have had to read.
    ///
    /// The number was a million, on the reasoning that a million 768-dimension rows is 3 GB
    /// off disk and "a few seconds of arithmetic". Measured on 2026-10-02 (release build,
    /// 16 cores, the row file in page cache): **1M x 960 scores in 300 ms** (ten queries,
    /// 292-371 ms), on the scan's own rayon pool, so nothing else waited. The arithmetic was
    /// never the cost; a cold read of the row file is, at roughly a second per gigabyte on
    /// ordinary NVMe. Ten million is therefore ~3 s warm and under a minute cold at 768-960
    /// dimensions, and a million rows was exactly the size at which a trustworthy ground
    /// truth is most wanted and was being refused. The cap stays, because at 100M rows a cold
    /// scan is minutes and the caller cannot tell that from a hang; `DXMT_EXACT_SCAN_MAX`
    /// moves it either way.
    const EXACT_SCAN_MAX_DEFAULT: usize = 10_000_000;

    fn exact_scan_max() -> usize {
        crate::config::env::var("DXMT_EXACT_SCAN_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(Self::EXACT_SCAN_MAX_DEFAULT)
    }

    /// Largest matching set the in-graph filter will materialise ids for.
    ///
    /// 10,000 is where it was before this was a named constant, and three measurements of
    /// Performance768D1M99P (1% of a million rows pass, so exactly 10,000 of them -- the case
    /// sits one row outside the cap) say it should stay there:
    ///
    /// | path                                    | recall | QPS  | p99      |
    /// |-----------------------------------------|--------|------|----------|
    /// | unfiltered fetch of 30,000, post-filter | 0.9963 | 76.9 |   245 ms |
    /// | in-graph, ef scaled by 1/selectivity    | 0.9998 |  6.4 | 7,987 ms |
    /// | in-graph, ef left alone                 | 0.9940 | 26.7 | 2,613 ms |
    ///
    /// Raising the cap to admit this case made it 2.9x slower even after the second mistake
    /// was taken back out. Walking the graph with the filter applied is not automatically
    /// cheaper than over-fetching without it: the walk still has to visit about `ef /
    /// selectivity` nodes to fill its result set, each visit costs a set lookup, and every
    /// query first materialises the whole matching id set -- 10,000 strings hashed, then
    /// looked up in `id_to_idx`, per query. Below the cap the sets are small enough that this
    /// is cheap; above it, over-fetching wins.
    ///
    /// `DXMT_IN_GRAPH_FILTER_MAX_IDS` overrides it, which is how both paths get exercised from
    /// one test.
    ///
    /// Since 2026-10-05 the paragraphs above describe the cost that is gone: the match set
    /// is a bitmap of ordinals, resolved to node positions through a map kept across
    /// queries (`storage::node_map`), and the walk tests a node with one lookup. The cap is
    /// therefore off by default -- every materialised match set takes the scan or the
    /// walk, priced by `code_scan_max` -- and the variable remains as the switch that sends
    /// a match set above it to the amplified fetch instead, for tests of that path.
    const IN_GRAPH_FILTER_MAX_IDS: usize = usize::MAX;

    fn in_graph_filter_max_ids() -> usize {
        crate::config::env::var("DXMT_IN_GRAPH_FILTER_MAX_IDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(Self::IN_GRAPH_FILTER_MAX_IDS)
    }

    /// The memory-optimized choice, which is the default: HNSW-PQ at every size.
    ///
    /// This used to pick plain HNSW below 100,000 rows and HNSW-PQ above, which combined
    /// badly with the fixed 10,000-row estimate `create_table` selects against: every table
    /// started as HNSW, and one that grew past the boundary was rebuilt from scratch by the
    /// next re-evaluation sweep -- O(n), on a table that by then held a million rows. The
    /// index does not need the selector to know the size. An `HnswPqIndex` is an exact HNSW
    /// graph until it holds `quantize_at_rows` (65,536) vectors and quantizes itself in
    /// place from there (docs/architecture/disk-index.md §3), so choosing it up front gives a
    /// small table exactly what HNSW gave it and a large one the graph-on-codes layout
    /// without a rebuild in between.
    async fn select_memory_optimized_index(
        &self,
        config: &IndexConfig,
    ) -> crate::Result<IndexDecision> {
        info!(
            "Selecting memory-optimized index for dataset size: {} -> HNSW-PQ (exact below {} rows)",
            config.dataset_size,
            crate::search::hnsw_pq::HnswPqConfig::DEFAULT_QUANTIZE_AT_ROWS
        );
        let selected_algorithm = IndexAlgorithm::HnswPQ {
            m: 16,
            ef_construction: Self::default_ef_construction(),
            ef_search: Self::DEFAULT_EF_SEARCH,
            // Four dimensions per subspace at 768 (192 bytes a code, the size that kept
            // recall@100 at 1M and 10M, docs/architecture/disk-index.md §2), finer below:
            // `HnswPqConfig::default_subvectors` has the measurements.
            num_subvectors: crate::search::hnsw_pq::HnswPqConfig::default_subvectors(
                config.dimension,
            ),
            num_centroids: 256,
            enable_reranking: true,
            quantize_at_rows: None,
            fast_scan: true,
        };
        // Measured, not estimated: 1M x 768 Cohere, records and vectors on disk, HNSW-PQ192
        // with a 500-candidate rerank (docs/architecture/disk-index.md §3–4).
        let quantized =
            config.dataset_size >= crate::search::hnsw_pq::HnswPqConfig::DEFAULT_QUANTIZE_AT_ROWS;
        let bytes_per_row = if quantized {
            // graph + codes + id map, the full vectors being on disk
            config.dimension / 4 + 256
        } else {
            config.dimension * 4 + 256
        };
        Ok(IndexDecision {
            selected_algorithm,
            confidence: 0.95,
        })
    }

    /// 创建索引实例
    fn create_index_instance(
        &self,
        algorithm: &IndexAlgorithm,
        dimension: usize,
        distance_metric: DistanceMetric,
    ) -> crate::Result<Box<dyn VectorIndex + Send + Sync>> {
        self.create_index_instance_for(algorithm, dimension, distance_metric, "default")
    }

    fn create_index_instance_for(
        &self,
        algorithm: &IndexAlgorithm,
        dimension: usize,
        distance_metric: DistanceMetric,
        table_id: &str,
    ) -> crate::Result<Box<dyn VectorIndex + Send + Sync>> {
        match algorithm {
            IndexAlgorithm::BruteForce { .. } => {
                // BruteForce 索引
                Ok(Box::new(BruteForceIndex::new(distance_metric)))
            }

            IndexAlgorithm::HNSW {
                m,
                ef_construction,
                ef_search,
            } => {
                let config = crate::search::hnsw::HnswConfig {
                    max_connections: *m,
                    ef_construction: *ef_construction,
                    ef_search: *ef_search,
                    ..Default::default()
                };
                Ok(Box::new(crate::search::hnsw::HnswIndex::new(
                    distance_metric,
                    Some(config),
                )))
            }

            IndexAlgorithm::HnswPQ {
                m,
                ef_construction,
                ef_search,
                num_subvectors,
                num_centroids,
                enable_reranking,
                quantize_at_rows,
                fast_scan,
            } => {
                use crate::search::hnsw::HnswConfig;
                use crate::search::hnsw_pq::{HnswPqConfig, HnswPqIndex};

                // The same graph parameters plain HNSW gets. This used to set `ml` to
                // 1/ln(2) -- four times the HNSW paper's 1/ln(m) -- so three quarters of
                // the nodes had an upper layer instead of one in sixteen; the graph the
                // 10M acceptance run built that way reached recall@100 0.9267 where the
                // probe's graph with 1/ln(16) reached 0.9787 on the same codes.
                let config = HnswPqConfig {
                    hnsw_config: HnswConfig::from_params(*m, *ef_construction, *ef_search),
                    num_subvectors: *num_subvectors,
                    num_centroids: *num_centroids,
                    quantize_at_rows: quantize_at_rows
                        .unwrap_or(HnswPqConfig::DEFAULT_QUANTIZE_AT_ROWS),
                    // Reranking is not optional any more: the graph holds codes, not
                    // vectors, so the storage layer always reranks (`search_reranked`).
                    ..HnswPqConfig::default()
                };

                Ok(Box::new(
                    HnswPqIndex::new(dimension, distance_metric, Some(config))?
                        .with_fast_scan(*fast_scan),
                ))
            }

            // These three have no implementation behind them. They used to fall through to
            // BruteForceIndex, which is the worst of both: the caller believes it asked for an
            // ANN index and gets an exact scan, so recall looks perfect and latency looks
            // inexplicable at scale. Fail instead, and name the alternative.
            IndexAlgorithm::DiskANN { .. } => Err(crate::Error::InvalidInput(
                "DiskANN is not implemented; use EnhancedDiskANN, which is".to_string(),
            )),

            IndexAlgorithm::EnhancedDiskANN {
                max_degree,
                search_list_size,
                use_pq,
                num_pq_subvectors,
                cache_size,
            } => {
                use crate::search::diskann_enhanced::{
                    EnhancedDiskANNConfig, EnhancedDiskANNIndex,
                };
                // The index needs a directory it can keep. This used to be
                // `TempDir::new()...keep()`, which deliberately leaks the temp directory so it
                // survives the handle — every restart orphaned another copy of the index under
                // /tmp, and nothing ever cleaned them up. Put it beside the WAL instead, where
                // it belongs and where the operator can find it.
                let index_dir = match self.wal_dir {
                    Some(ref dir) => {
                        let path = dir.join("diskann").join(uuid::Uuid::new_v4().to_string());
                        std::fs::create_dir_all(&path).map_err(crate::Error::Io)?;
                        path
                    }
                    None => {
                        // No WAL means no durable state anywhere, so a temp directory is
                        // consistent with the rest of this configuration.
                        let temp_dir = tempfile::TempDir::new().map_err(crate::Error::Io)?;
                        temp_dir.keep()
                    }
                };

                let config = EnhancedDiskANNConfig {
                    max_degree: *max_degree,
                    search_list_size: *search_list_size,
                    alpha: 1.2,
                    use_pq_compression: *use_pq,
                    num_pq_subvectors: *num_pq_subvectors,
                    num_pq_centroids: 256,
                    cache_size: *cache_size,
                    beam_width: 8,
                    sector_size: 4096,
                    write_batch_size: 1000,
                };

                Ok(Box::new(EnhancedDiskANNIndex::new(
                    dimension,
                    distance_metric,
                    index_dir,
                    Some(config),
                )?))
            }

            IndexAlgorithm::PartitionedIvf {
                nlist,
                nprobe,
                max_resident_partitions,
            } => {
                use crate::search::partitioned_ivf::{PartitionedIvfConfig, PartitionedIvfIndex};

                // The prefix must be stable for a table and safe as an object key, so it uses
                // the table's UUID rather than its name — names come from user input.
                let index = PartitionedIvfIndex::new(
                    distance_metric,
                    format!("partitions/{}", table_id),
                    Some(PartitionedIvfConfig {
                        nlist: *nlist,
                        nprobe: *nprobe,
                        max_resident_partitions: *max_resident_partitions,
                        max_iterations: 20,
                        ..Default::default()
                    }),
                )
                .with_store(self.storage.clone());
                Ok(Box::new(index))
            }

            IndexAlgorithm::ScaNN { .. } => Err(crate::Error::InvalidInput(
                "ScaNN is not implemented; src/search/scann.rs does not match the index trait \
                 and its tests are commented out. Use HNSW or HnswPQ."
                    .to_string(),
            )),

            IndexAlgorithm::QuantizedIVF { .. } => Err(crate::Error::InvalidInput(
                "QuantizedIVF is not implemented; use HnswPQ for a quantized index.".to_string(),
            )),
        }
    }
    //
    //     /// 重新评估索引选择
    //     async fn reevaluate_index_selection(
    //         &self,
    //         tablename: &str,
    //         table_data: &mut IntelligentTableData) -> crate::Result<()> {
    //         debug!("Re-evaluating index selection for table {}", tablename);
    //
    //         // 准备新的索引配置
    //         let index_config = IndexConfig {
    //             dataset_size: table_data.table.record_count as usize,
    //             dimension: table_data.table.dimension,
    //             distance_metric: table_data.table.distance_metric,
    //             memory_budget_mb: self.hardware_capabilities.available_memory_gb as usize * 512,
    //             performance_requirements: self.performance_requirements.clone(),
    //             cost_constraints: self.cost_constraints.clone(),
    //             hardware_capabilities: self.hardware_capabilities.clone(),
    //         };
    //
    //         // 重新选择索引
    //         let new_decision = self.index_selector.select_optimal_index(&index_config).await?;
    //
    //         // 检查是否需要切换算法
    //         if !Self::algorithms_equal(&new_decision.selected_algorithm, &table_data.selected_algorithm) &&
    //            new_decision.confidence > table_data.index_decision.confidence * 1.2 {
    //
    //             info!(
    //                 "Switching index algorithm for table {} from {:?} to {:?}",
    //                 tablename,
    //                 table_data.selected_algorithm,
    //                 new_decision.selected_algorithm
    //             );
    //
    //             // index migration: create new index, copy data, atomic swap, drop old
    //         }
    //
    //         Ok(())
    //     }

    /// 检查两个算法是否相同
    #[allow(dead_code)]
    fn algorithms_equal(a: &IndexAlgorithm, b: &IndexAlgorithm) -> bool {
        !a.requires_rebuild_to_reach(b)
    }

    /// 检测硬件能力
    fn detect_hardware_capabilities() -> HardwareCapabilities {
        let has_gpu_available = {
            {
                false
            }
        };

        let available_memory_gb = {
            #[cfg(target_os = "linux")]
            {
                std::fs::read_to_string("/proc/meminfo")
                    .ok()
                    .and_then(|info| {
                        info.lines()
                            .find(|l| l.starts_with("MemTotal:"))
                            .and_then(|l| l.split_whitespace().nth(1))
                            .and_then(|kb| kb.parse::<u64>().ok())
                            .map(|kb| kb as f32 / 1_048_576.0)
                    })
                    .unwrap_or(16.0)
            }
            #[cfg(not(target_os = "linux"))]
            {
                16.0_f32
            }
        };

        let has_avx2 = {
            #[cfg(target_arch = "x86_64")]
            {
                is_x86_feature_detected!("avx2")
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };

        if has_gpu_available {
            info!(
                "Hardware detection: {} CPU cores, {:.1} GB RAM, AVX2={}, GPU=available",
                num_cpus::get(),
                available_memory_gb,
                has_avx2
            );
        } else {
            info!(
                "Hardware detection: {} CPU cores, {:.1} GB RAM, AVX2={}, GPU=not available",
                num_cpus::get(),
                available_memory_gb,
                has_avx2
            );
        }

        HardwareCapabilities {
            cpu_cores: num_cpus::get(),
            available_memory_gb,
            has_avx2,
            has_gpu: has_gpu_available,
            ssd_storage_gb: 1000.0,
            network_bandwidth_gbps: 10.0,
            supports_simd: has_avx2,
            supports_gpu: has_gpu_available,
        }
    }

    /// Bulk optimized insert
    /// Deferred index insert
    /// Get a single record
    pub async fn get_record(&self, tablename: &str, record_id: &RecordId) -> crate::Result<Record> {
        let tablename = &*self.resolve(tablename);
        self.ensure_resident(tablename).await?;
        let tables = self.read_tables("get_record").await;
        let table = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        table
            .records
            .get(record_id)
            .map(|r| {
                let mut record = (*r).clone();
                if record.vector.is_empty() {
                    record.vector = table.vector_for(&r);
                }
                record
            })
            .ok_or_else(|| crate::Error::RecordNotFound(record_id.clone()))
    }

    /// Update a record
    /// Update a record the way a caller means it: change what was given, keep what was not.
    ///
    /// [`Self::update_record`] takes a whole `CreateRecordRequest` and replaces the record with
    /// it. Everything between that and what an API caller sends -- deciding whether the record
    /// has to exist, keeping the stored vector when none is given, keeping the sparse and named
    /// vectors that hybrid search depends on, and merging a payload rather than replacing it --
    /// lived inline in the HTTP handler. gRPC needs the same semantics, and two copies of sixty
    /// lines of "which of these do we keep" is how two interfaces come to disagree about what
    /// an update means. It lives here so there is one answer.
    ///
    /// `vector: None` keeps the stored dense vector, `payload: None` keeps the stored payload,
    /// and the sparse and named vectors are kept
    /// either way -- the update shape exposes only the dense one, and silently dropping the
    /// postings that hybrid and named-vector search rely on is not something a payload edit
    /// should do.
    pub async fn apply_update(
        &self,
        tablename: &str,
        record_id: &RecordId,
        update: RecordUpdate,
    ) -> crate::Result<Record> {
        let RecordUpdate {
            vector,
            payload,
            text,
            merge,
            upsert,
        } = update;
        let existing = self.get_record(tablename, record_id).await.ok();
        if !upsert && existing.is_none() {
            return Err(crate::Error::RecordNotFound(record_id.to_string()));
        }

        // `text` replaces the vector, so it is not "no vector given, keep the old one": the
        // vector is left empty here and `update_record` embeds the text into it. Sending both
        // is refused rather than resolved, for the same reason a search carrying both is.
        if text.is_some() && vector.is_some() {
            return Err(crate::Error::InvalidInput(
                "an update takes `vector` or `text`, not both".to_string(),
            ));
        }
        let final_vector = match (vector, text.is_some()) {
            (Some(v), _) => v,
            (None, true) => Vec::new(),
            (None, false) => existing
                .as_ref()
                .map(|r| r.vector.clone())
                .unwrap_or_default(),
        };
        let final_payload = match (merge, &existing) {
            (true, Some(record)) => {
                let mut merged = record.payload.clone();
                if let Some(new_payload) = payload {
                    match (merged.as_object_mut(), new_payload.as_object()) {
                        (Some(into), Some(from)) => {
                            for (key, value) in from {
                                into.insert(key.clone(), value.clone());
                            }
                        }
                        // Merging into something that is not an object has no meaning; the
                        // new value wins, which is what replace would have done anyway.
                        _ => merged = new_payload,
                    }
                }
                Some(merged)
            }
            // No payload given keeps the one stored, in replace mode as much as in merge mode:
            // `{"vector": [...]}` is a request to change the vector. It used to replace the
            // payload with nothing, so re-embedding a table erased every record's metadata.
            // To clear a payload, send `"payload": {}`.
            _ => payload.or_else(|| existing.as_ref().map(|r| r.payload.clone())),
        };

        self.update_record(
            tablename,
            record_id,
            CreateRecordRequest {
                id: record_id.clone(),
                vector: final_vector,
                sparse_vector: existing.as_ref().and_then(|r| r.sparse_vector.clone()),
                payload: final_payload,
                named_vectors: existing.as_ref().and_then(|r| r.named_vectors.clone()),
                text,
            },
        )
        .await
    }

    pub async fn update_record(
        &self,
        tablename: &str,
        record_id: &RecordId,
        request: CreateRecordRequest,
    ) -> crate::Result<Record> {
        let tablename = &*self.resolve(tablename);
        self.ensure_writable()?;
        self.ensure_memory_headroom()?;
        self.ensure_resident(tablename).await?;

        // An update is its own write path, so it needs the embedding step of its own. Without
        // it a `PUT` carrying `text` would be accepted, the text dropped, and the record left
        // with whatever vector it had -- the edit silently not happening.
        //
        // Note this is the one place where an empty vector is not an error: it means "keep the
        // vector this record already has". So `fill_vectors_from_text` only runs when there is
        // something to embed, rather than on every empty vector as it does on insert.
        let mut request = request;
        if request.text.is_some() {
            let table = self
                .read_tables("update_record")
                .await
                .get(tablename)
                .map(|data| data.table.clone())
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
            self.fill_vectors_from_text(&table, std::slice::from_mut(&mut request))
                .await?;
        }

        let artifacts = {
            let mut tables = self.write_tables("update_record").await;
            let table_data = tables
                .get_mut(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

            let expected_dim = table_data.table.dimension;
            if !request.vector.is_empty() && request.vector.len() != expected_dim {
                return Err(crate::Error::DimensionMismatch {
                    expected: expected_dim,
                    actual: request.vector.len(),
                });
            }

            let (created_at, version, old_payload) = table_data
                .records
                .get(record_id)
                .map(|r| (r.created_at, r.version + 1, Some(r.payload.clone())))
                .unwrap_or_else(|| (chrono::Utc::now(), 1, None));

            let index_vector = request.vector.clone();
            let mut updated_vector_row: Option<u64> = None;
            let mut appended_row: Option<u64> = None;
            let mut unindexed = None;
            let stored_vector = match &table_data.vector_file {
                Some(file) if request.vector.len() == expected_dim => {
                    unindexed = Some(file.begin_unindexed());
                    // Appended here under the map lock -- updates are rare and single.
                    let row = file.append(&[(record_id.as_str(), request.vector.as_slice())])?[0];
                    appended_row = Some(row);
                    if table_data.records.is_on_disk() {
                        updated_vector_row = Some(row);
                    } else {
                        table_data.vector_rows.insert(record_id.clone(), row);
                    }
                    Vec::new()
                }
                _ => request.vector.clone(),
            };
            let record = Record {
                id: record_id.clone(),
                vector: stored_vector,
                sparse_vector: request.sparse_vector,
                named_vectors: request.named_vectors.clone(),
                payload: request
                    .payload
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({})),
                created_at,
                updated_at: chrono::Utc::now(),
                version,
                status: crate::types::RecordStatus::Active,
            };

            table_data
                .records
                .insert(Arc::new(record.clone()), updated_vector_row)?;
            table_data.writes.fetch_add(1, Ordering::Relaxed);

            if let Some(derived) = table_data.derived_mut().as_mut() {
                if let Some(old_payload_val) = old_payload.as_ref() {
                    derived.payload.update_record(
                        record_id,
                        old_payload_val,
                        &record.payload,
                        crate::storage::payload_index::created_key(&record.created_at),
                    );
                } else {
                    derived.payload.insert_record(
                        record_id,
                        &record.payload,
                        crate::storage::payload_index::created_key(&record.created_at),
                    );
                }

                // Always remove the old sparse posting first. If the new record has no sparse
                // vector, leaving the old postings behind makes hybrid search return stale ids.
                let _ = derived.sparse.delete(record_id);
                if let Some(sv) = &record.sparse_vector {
                    if let Err(e) = derived.sparse.insert(record_id.clone(), sv) {
                        warn!(
                            "Failed to update sparse index for record {}: {}",
                            record_id, e
                        );
                    }
                }
            } else {
                table_data.note_unindexed_change(record_id, old_payload.as_ref());
            }

            table_data.table.record_count = table_data.records.len() as u64;
            table_data.table.updated_at = chrono::Utc::now();

            UpdateArtifacts {
                index: table_data.index.clone(),
                named_indexes: table_data.named_indexes.clone(),
                #[cfg(feature = "text-bm25")]
                table: table_data.table.clone(),
                record,
                vector: index_vector,
                vector_row: appended_row,
                _unindexed: unindexed,
                #[cfg(feature = "text-bm25")]
                full_text_engine: table_data.full_text_engine.clone(),
            }
        };

        // Dense and named-vector index work can block, so it runs after the global table-map
        // lock is released. Search may briefly miss this record while the index catches up, the
        // same visibility model inserts already have.
        // Without a vector there is nothing to re-index, and removing first would drop the
        // record from search: the graph refuses an empty vector, so the insert that should
        // follow the remove would fail and leave nothing behind.
        if !artifacts.vector.is_empty() {
            if let Err(e) = artifacts.index.remove(record_id).await {
                warn!("Failed to remove old vector for {}: {}", record_id, e);
            }
            if let Err(e) = artifacts
                .index
                .insert(record_id.clone(), artifacts.vector.clone())
                .await
            {
                warn!(
                    "Failed to update vector index for record {}: {}",
                    record_id, e
                );
            }
        }

        if let Some(collection) = &artifacts.named_indexes {
            for name in collection.list_index_names().await {
                if let Some(index) = collection.get_index(&name).await {
                    if let Err(e) = index.remove(record_id).await {
                        warn!(
                            "Failed to remove old named vector '{}' for {}: {}",
                            name, record_id, e
                        );
                    }
                }
            }
            if let Some(named_vectors) = &artifacts.record.named_vectors {
                for (name, vector) in named_vectors {
                    if let Some(index) = collection.get_index(name).await {
                        if let Err(e) = index.insert(record_id.clone(), vector.clone()).await {
                            warn!(
                                "Failed to update named vector '{}' for {}: {}",
                                name, record_id, e
                            );
                        }
                    }
                }
            }
        }

        #[cfg(feature = "text-bm25")]
        if let Some(engine) = &artifacts.full_text_engine {
            if let Some(text_content) =
                extract_text_from_record(&artifacts.table, &artifacts.record)
            {
                let doc = FullTextDocument {
                    id: artifacts.record.id.clone(),
                    title: text_content.get("title").cloned(),
                    content: text_content.get("content").cloned().unwrap_or_default(),
                    description: text_content.get("description").cloned(),
                    tags: text_content
                        .get("tags")
                        .map(|s| vec![s.clone()])
                        .unwrap_or_default(),
                    payload: Some(artifacts.record.payload.clone()),
                    timestamp: chrono::Utc::now().timestamp(),
                };

                if let Err(e) = engine.update_document(doc).await {
                    warn!(
                        "Failed to update text index for record {}: {}",
                        artifacts.record.id, e
                    );
                }
            }
            if let Err(e) = engine.commit().await {
                warn!("Failed to commit full-text index: {}", e);
            }
        }

        // The entry has to be able to rebuild the record on its own. `artifacts.record` is the
        // record as the map keeps it, and on a disk-backed table the map keeps it *without* its
        // vector, which went to the row file. Logging that meant a replay re-indexed an empty
        // vector, the graph's flat vector array fell one entry behind its nodes, and after any
        // update followed by a crash the server panicked on its way up — every time, since the
        // log does not change. The insert path logs the vector already.
        // Since then the entry points at the row instead (`UpsertRecordInRowFile`), the same
        // as an insert; a table without a row file still logs the vector itself.
        let entry = match artifacts.vector_row.filter(|_| self.log_points_at_rows()) {
            Some(row) => WalEntry::UpsertRecordInRowFile {
                table: tablename.to_string(),
                record: without_vector(&artifacts.record),
                row,
            },
            None => {
                let mut logged = artifacts.record.clone();
                if logged.vector.is_empty() {
                    logged.vector = artifacts.vector.clone();
                }
                WalEntry::UpsertRecordV2 {
                    table: tablename.to_string(),
                    record: logged,
                }
            }
        };
        self.wal_append(vec![entry]).await?;

        Ok(artifacts.record)
    }

    /// Delete multiple records
    /// Delete records, returning how many of the requested ids were actually there.
    ///
    /// The count is the point. `DELETE /records` reported `deleted_count` as the number of ids
    /// the caller *asked* about, because this returned `()` and the handler had nothing else to
    /// put there -- so deleting three ids that do not exist answered `deleted_count: 3`. A
    /// caller checking that number to find out whether anything was removed was told yes every
    /// time.
    pub async fn delete_records(
        &self,
        tablename: &str,
        record_ids: Vec<RecordId>,
    ) -> crate::Result<usize> {
        let tablename = &*self.resolve(tablename);
        self.ensure_writable()?;
        self.ensure_resident(tablename).await?;

        let mut deleted = 0usize;
        let artifacts = {
            let mut tables = self.write_tables("delete_records").await;
            let table_data = tables
                .get_mut(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

            for id in &record_ids {
                let existing = table_data.records.get(id);
                let old_payload = existing.as_ref().map(|r| r.payload.clone());
                if existing.is_some() {
                    deleted += 1;
                }

                table_data.records.remove(id)?;
                table_data.writes.fetch_add(1, Ordering::Relaxed);
                table_data.vector_rows.remove(id);

                if let Some(derived) = table_data.derived_mut().as_mut() {
                    if let Some(payload) = old_payload {
                        derived.payload.remove_record(id, &payload);
                    }
                    if let Err(e) = derived.sparse.delete(id) {
                        warn!("Failed to remove record {} from sparse index: {}", id, e);
                    }
                } else {
                    table_data.note_unindexed_change(id, old_payload.as_ref());
                }
            }

            table_data.table.record_count = table_data.records.len() as u64;
            table_data.table.updated_at = chrono::Utc::now();

            DeleteArtifacts {
                index: table_data.index.clone(),
                named_indexes: table_data.named_indexes.clone(),
                record_ids: record_ids.clone(),
                #[cfg(feature = "text-bm25")]
                full_text_engine: table_data.full_text_engine.clone(),
            }
        };

        for id in &artifacts.record_ids {
            if let Err(e) = artifacts.index.remove(id).await {
                warn!("Failed to remove record {} from vector index: {}", id, e);
            } else {
                debug!("Successfully removed record {} from vector index", id);
            }
        }

        // And from every named vector space. `update_record` has always done this; a delete did
        // not, so a record removed from the table kept answering searches that named a space --
        // at rank one, with the score of an exact match, because it was still the nearest thing
        // in a graph nobody had told. An absent record is a wrong answer that looks like a right
        // one, which is worse than the empty result the rest of this family produces.
        if let Some(collection) = &artifacts.named_indexes {
            for name in collection.list_index_names().await {
                if let Some(index) = collection.get_index(&name).await {
                    for id in &artifacts.record_ids {
                        if let Err(e) = index.remove(id).await {
                            warn!(
                                "Failed to remove record {} from named vector '{}': {}",
                                id, name, e
                            );
                        }
                    }
                }
            }
        }

        #[cfg(feature = "text-bm25")]
        if let Some(engine) = &artifacts.full_text_engine {
            for id in &artifacts.record_ids {
                if let Err(e) = engine.delete_document(id).await {
                    warn!("Failed to remove record {} from text index: {}", id, e);
                }
            }
            if let Err(e) = engine.commit().await {
                warn!("Failed to commit full-text index: {}", e);
            }
        }

        self.wal_append(
            artifacts
                .record_ids
                .into_iter()
                .map(|id| WalEntry::Delete {
                    table: tablename.to_string(),
                    record_id: id,
                })
                .collect(),
        )
        .await?;

        Ok(deleted)
    }

    /// Get statistics
    pub async fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        let tables = self.read_tables("get_stats").await;

        stats.insert("table_count".to_string(), serde_json::json!(tables.len()));
        let total_records: usize = tables.values().map(|t| t.records.len()).sum();
        stats.insert(
            "total_records".to_string(),
            serde_json::json!(total_records),
        );

        stats
    }

    /// Return a point-in-time view used by liveness/readiness and operator diagnostics.
    ///
    /// An empty database is healthy, so table count is intentionally not used as a proxy for
    /// storage health. The only hard readiness failures are an unavailable local WAL directory
    /// or a durability error that put this process into read-only mode.
    pub async fn health_snapshot(&self) -> StorageHealthSnapshot {
        let (resident_names, total_records): (std::collections::HashSet<String>, usize) = {
            let tables = self.read_tables("health_snapshot").await;
            (
                tables.keys().cloned().collect(),
                tables.values().map(|table| table.records.len()).sum(),
            )
        };
        let resident_table_count = resident_names.len();
        // Deliberately not `list_tables()`: that reads the snapshot manifest plus one `.meta`
        // per evicted table, and this runs on every health and readiness probe. The cached
        // snapshot names give the same total without touching the disk.
        let table_count = self
            .snapshot_table_names
            .lock()
            .map(|names| {
                resident_table_count
                    + names
                        .iter()
                        .filter(|name| !resident_names.contains(*name))
                        .count()
            })
            .unwrap_or(resident_table_count);

        let wal_enabled = self.wal.is_some();
        // `try_lock`, not `lock`: the WAL mutex is held across `write_all` + `sync_data`, so a
        // blocking probe queues behind an fsync and reports a timeout as a dead server. A
        // sequence number that is momentarily unavailable is reported as unavailable.
        let wal_next_sequence = self
            .wal
            .as_ref()
            .and_then(|wal| wal.try_lock().ok().map(|guard| guard.next_sequence()));

        // The cached sequence comes from the manifest this process loaded at startup or wrote
        // itself. Re-reading and deserializing the manifest here would put disk I/O on every
        // health check, and orchestrators poll these endpoints continuously.
        let snapshot_wal_next_sequence = self
            .snapshot_sequence
            .lock()
            .ok()
            .and_then(|sequence| *sequence);

        let (wal_directory_accessible, snapshot_present, snapshot_manifest_valid) =
            if let Some(dir) = &self.wal_dir {
                let directory_accessible = std::fs::metadata(dir)
                    .map(|metadata| metadata.is_dir())
                    .unwrap_or(false)
                    && std::fs::metadata(dir.join("wal.log"))
                        .map(|metadata| metadata.is_file())
                        .unwrap_or(false);
                let present = Self::snapshot_dir(dir).join("manifest").exists();
                (
                    directory_accessible,
                    present,
                    present && snapshot_wal_next_sequence.is_some(),
                )
            } else {
                (true, false, false)
            };

        let durability_error = self
            .durability_error
            .lock()
            .ok()
            .and_then(|state| state.clone());
        let durability_degraded = durability_error.is_some();
        let replica_stale = if self.replica {
            self.replica_stale.lock().ok().and_then(|s| s.clone())
        } else {
            None
        };
        let ready = (!wal_enabled || wal_directory_accessible)
            && !durability_degraded
            && replica_stale.is_none();

        let last_wal_sync_at = self.last_wal_sync_at.lock().ok().and_then(|at| at.clone());
        let last_snapshot_at = self.last_snapshot_at.lock().ok().and_then(|at| at.clone());
        let last_snapshot_error = self
            .last_snapshot_error
            .lock()
            .ok()
            .and_then(|error| error.clone());

        StorageHealthSnapshot {
            ready,
            wal_enabled,
            wal_directory_accessible,
            wal_next_sequence,
            snapshot_present,
            snapshot_manifest_valid,
            snapshot_wal_next_sequence,
            wal_mirrored_next_sequence: (self.object_storage_enabled && !self.replica)
                .then(|| self.wal_mirrored_next.load(Ordering::SeqCst)),
            replica_applied_next_sequence: self
                .replica
                .then(|| self.replica_applied_next.load(Ordering::SeqCst)),
            replica_stale,
            durability_degraded,
            durability_error,
            resident_table_count,
            table_count,
            total_records,
            last_wal_sync_at,
            last_snapshot_at,
            last_backup_at: self.last_backup_at.lock().ok().and_then(|at| at.clone()),
            backup_failures: self.backup_failures.load(Ordering::Relaxed),
            snapshot_failures: self.snapshot_failures.load(Ordering::Relaxed),
            last_snapshot_error,
            filter_paths: FilterPathCounts {
                exact_scan: self.filter_path_exact_scan.load(Ordering::Relaxed),
                code_scan: self.filter_path_code_scan.load(Ordering::Relaxed),
                in_graph: self.filter_path_in_graph.load(Ordering::Relaxed),
                pre_filter: self.filter_path_pre_filter.load(Ordering::Relaxed),
                post_filter: self.filter_path_post_filter.load(Ordering::Relaxed),
                exact_scan_ns: self.filter_path_ns[0].load(Ordering::Relaxed),
                code_scan_ns: self.filter_path_ns[1].load(Ordering::Relaxed),
                in_graph_ns: self.filter_path_ns[2].load(Ordering::Relaxed),
                pre_filter_ns: self.filter_path_ns[3].load(Ordering::Relaxed),
                post_filter_ns: self.filter_path_ns[4].load(Ordering::Relaxed),
                eval_ns: self.filter_eval_ns.load(Ordering::Relaxed),
                node_map_misses: self.filter_node_map_misses.load(Ordering::Relaxed),
            },
            timings: {
                let (last_recovery_ns, last_recovery_wal_entries) = self
                    .last_recovery
                    .lock()
                    .ok()
                    .and_then(|r| *r)
                    .map_or((None, None), |(ns, entries)| (Some(ns), Some(entries)));
                OperationTimings {
                    table_lock_read_wait_ns: TABLE_LOCK_WAIT_NS[LOCK_READ].load(Ordering::Relaxed),
                    table_lock_write_wait_ns: TABLE_LOCK_WAIT_NS[LOCK_WRITE]
                        .load(Ordering::Relaxed),
                    table_lock_read_hold_ns: TABLE_LOCK_HOLD_NS[LOCK_READ].load(Ordering::Relaxed),
                    table_lock_write_hold_ns: TABLE_LOCK_HOLD_NS[LOCK_WRITE]
                        .load(Ordering::Relaxed),
                    table_lock_reads: TABLE_LOCK_ACQUIRED[LOCK_READ].load(Ordering::Relaxed),
                    table_lock_writes: TABLE_LOCK_ACQUIRED[LOCK_WRITE].load(Ordering::Relaxed),
                    table_lock_slow_holds: TABLE_LOCK_SLOW_HOLDS.load(Ordering::Relaxed),
                    index_rebuilds: self.index_rebuilds.load(Ordering::Relaxed),
                    index_rebuild_ns: self.index_rebuild_ns.load(Ordering::Relaxed),
                    index_rebuild_swap_wait_ns: self
                        .index_rebuild_swap_wait_ns
                        .load(Ordering::Relaxed),
                    last_recovery_ns,
                    last_recovery_wal_entries,
                    last_snapshot_ns: self.last_snapshot_ns.lock().ok().and_then(|n| *n),
                }
            },
        }
    }

    // A comment here used to say eight redundant insert stubs had been removed. They had not:
    // `insert_records_lockfree`, `_streaming`, `_with_memory_pool`, `_deferred_index`,
    // `_ultra_fast`, `_vm_optimized`, `_nvme_optimized` and `_ultimate` were all still present,
    // all seven lines long, and all reduced to `insert_records`. They are gone now, with the
    // `mode` field on `/records/bulk` that was the only thing selecting between them.

    /// Add a single record to a table.
    pub async fn add_record(
        &self,
        tablename: &str,
        request: CreateRecordRequest,
    ) -> crate::Result<Record> {
        let records = self.insert_records(tablename, vec![request]).await?;
        records
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::Internal("Failed to add record".to_string()))
    }

    /// Search for records in a table
    pub async fn search_records(
        &self,
        tablename: &str,
        query: &DenseVector,
        limit: usize,
        filter: Option<serde_json::Value>,
    ) -> crate::Result<Vec<(Record, f32)>> {
        // Take the index out of the map and let go, as `search` does — an index search is the
        // slowest thing this function does, and holding the map across it stops every table.
        let (index, metric) = {
            let tables = self.read_tables("search_records").await;
            let table_data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

            // Validate query vector dimension
            let expected_dim = table_data.table.dimension;
            if !query.is_empty() && query.len() != expected_dim {
                return Err(crate::Error::DimensionMismatch {
                    expected: expected_dim,
                    actual: query.len(),
                });
            }
            (table_data.index.clone(), table_data.table.distance_metric)
        };

        // 如果有过滤器，获取更多结果以补偿过滤后的损失
        let fetch_limit = if filter.is_some() {
            limit * 3 // 获取3倍结果
        } else {
            limit
        };

        let results = index.search(query, fetch_limit).await?;

        // Back under the map for the record lookups, which are hash hits.
        let tables = self.read_tables("search_records").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        // Retrieve records and apply filter if provided
        let mut records_with_scores = Vec::new();
        for (record_id, score) in results {
            if let Some(record) = table_data.records.get(&record_id) {
                // Apply filter if provided
                if let Some(ref filter_json) = filter {
                    // Simple filter matching (can be enhanced)
                    if !Self::matches_filter(&record.payload, filter_json) {
                        continue;
                    }
                }

                let similarity_score = Self::similarity_from_distance(metric, score);

                records_with_scores.push(((*record).clone(), similarity_score));

                // 只保留limit个结果
                if records_with_scores.len() >= limit {
                    break;
                }
            }
        }

        Ok(records_with_scores)
    }

    /// List records by metadata filter (no vector search required)
    ///
    /// This method supports pure metadata filtering without requiring a query vector,
    /// enabling efficient retrieval of records based solely on payload attributes.
    ///
    /// # Arguments
    /// * `tablename` - Name of the table to search
    /// * `filter` - Optional filter to apply to record payloads
    /// * `limit` - Maximum number of records to return
    /// * `offset` - Number of records to skip (for pagination)
    ///
    /// # Returns
    /// List of records matching the filter criteria, sorted by creation time (newest first)
    /// Returns `(page, total_matching)` so callers can surface pagination info.
    /// A page of records, newest first.
    ///
    /// `with_vector` decides whether each record carries its vector, and it decides it the same
    /// way whatever the storage mode. That is the whole reason the parameter exists: this used
    /// to hand back the record's own `vector` field, which holds the vector for an in-memory
    /// table and nothing at all once the vectors live in a row file — so the same call answered
    /// `[1.0, 0.0, ...]` or `[]` depending on `DXMT_VECTOR_STORE`, with no way for the caller to
    /// tell that from a record that genuinely has no vector.
    /// One page of records after `cursor` (from the start when `None`), newest first with ties
    /// by id, and the cursor for the next page — `None` on the last one.
    ///
    /// Paging by cursor rather than offset is what makes a scroll through a table that is being
    /// written to return every record present throughout exactly once; see
    /// `payload_index::ListCursor`. One more record than asked for is fetched, so the last page
    /// says it is the last instead of sending the client for an empty one.
    pub async fn list_records_after(
        &self,
        tablename: &str,
        filter: Option<&crate::types::Filter>,
        cursor: Option<&crate::storage::payload_index::ListCursor>,
        limit: usize,
        with_vector: bool,
    ) -> crate::Result<(
        Vec<Record>,
        usize,
        Option<crate::storage::payload_index::ListCursor>,
    )> {
        let tablename = &*self.resolve(tablename);
        use crate::storage::payload_index::{created_key, ListCursor};
        Self::check_filter(filter)?;
        self.ensure_resident(tablename).await?;
        if filter.is_some() {
            self.ensure_derived(tablename).await?;
        }
        let tables = self.read_tables("list_records_after").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        Self::check_filter_policy(&table_data.table, filter)?;
        let fetch = limit.saturating_add(1);

        let (mut page, total): (Vec<Record>, usize) = match filter {
            Some(filter) => {
                let derived = table_data.derived();
                let (ids, total) = derived
                    .payload
                    .page_of_matching_after(filter, cursor, fetch);
                let page = ids
                    .iter()
                    .filter_map(|(id, _)| table_data.records.get(id))
                    .map(|r| (*r).clone())
                    .collect();
                (page, total)
            }
            None => {
                let total = table_data.records.len();
                let from_disk = match cursor {
                    Some(c) => table_data.records.page_newest_first_after(c, fetch),
                    None => table_data.records.page_newest_first(0, fetch),
                };
                let page = match from_disk {
                    Some(page) => page.iter().map(|r| (**r).clone()).collect(),
                    // In memory: the records are `Arc`s already; sort the ones after the cursor.
                    None => {
                        let mut all = Vec::with_capacity(total);
                        table_data.records.for_each(|r| {
                            if cursor.is_none_or(|c| c.precedes(created_key(&r.created_at), &r.id))
                            {
                                all.push(r);
                            }
                        });
                        let newest_first = |a: &Arc<Record>, b: &Arc<Record>| {
                            created_key(&b.created_at)
                                .cmp(&created_key(&a.created_at))
                                .then_with(|| a.id.cmp(&b.id))
                        };
                        let end = fetch.min(all.len());
                        if end > 0 && end < all.len() {
                            all.select_nth_unstable_by(end - 1, newest_first);
                        }
                        all.truncate(end);
                        all.sort_by(newest_first);
                        all.iter().map(|r| (**r).clone()).collect()
                    }
                };
                (page, total)
            }
        };

        let next = if page.len() > limit {
            page.truncate(limit);
            page.last()
                .map(|last| ListCursor::after(created_key(&last.created_at), &last.id))
        } else {
            None
        };
        Self::fill_vectors(table_data, &mut page, with_vector);
        Ok((page, total, next))
    }

    /// How many records a filter matches, without building any of them.
    ///
    /// "How many of these are there" had no answer short of paging the whole result: table
    /// stats carry a record count for the table and nothing for a condition, so a caller
    /// counting matches downloaded them. The page functions already return the total as a
    /// by-product of choosing a page, so a count is that number with a page of zero.
    /// Nearest neighbours, grouped by a payload field, at most `group_size` per group.
    ///
    /// The case this exists for is a corpus chunked for retrieval: twenty chunks of one
    /// document, and a top-10 that is the same document ten times. Grouping by `document_id`
    /// with `group_size: 1` turns that into ten documents.
    ///
    /// **It groups a window, not the table.** There is no index over (field, distance), so the
    /// only way to fill `groups` groups is to score neighbours until that many distinct values
    /// have appeared. The window starts at `groups * group_size * 4` and doubles while it is
    /// short, up to `MAX_GROUP_CANDIDATES`; the result says how wide it got, because a caller
    /// cannot otherwise tell "there are only three documents near this query" from "the fourth
    /// one's chunks are all further away than we looked".
    pub async fn search_groups(
        &self,
        tablename: &str,
        query: SearchQuery,
        group_by: &str,
        groups: usize,
        group_size: usize,
    ) -> crate::Result<crate::types::record::SearchGroupsResult> {
        let tablename = &*self.resolve(tablename);
        /// A cap, not a target. Each round is a real search, and an unbounded loop on a field
        /// almost every record shares a value of would scan the table one doubling at a time.
        const MAX_GROUP_CANDIDATES: usize = 10_000;

        let start = std::time::Instant::now();
        if query.rerank.is_some() {
            // `SearchGroupsRequest` flattens `SearchQuery`, so the field arrives here. Nothing
            // below reranks, and a field that is accepted and ignored is the failure
            // `tests/api/test_request_fields.py` exists for.
            return Err(crate::Error::InvalidInput(
                "`rerank` is not supported on /search/groups; rerank a plain search".to_string(),
            ));
        }
        if group_by.is_empty() {
            return Err(crate::Error::InvalidInput(
                "group_by must name a payload field".to_string(),
            ));
        }
        if groups == 0 || group_size == 0 {
            return Err(crate::Error::InvalidInput(
                "groups and group_size must be greater than 0".to_string(),
            ));
        }

        // The grouping key has to be read off every hit, whatever the caller asked to see. So
        // the inner search always returns the payload, projected to the caller's fields plus
        // this one, and the projection is applied again at the end for what leaves.
        let wanted_fields = query.payload_fields.clone();
        let wanted_payload = query.with_payload;
        let mut inner_fields = wanted_fields.clone();
        if let Some(fields) = &mut inner_fields {
            if !fields.iter().any(|f| f == group_by) {
                fields.push(group_by.to_string());
            }
        }

        let mut fetch = groups
            .saturating_mul(group_size)
            .saturating_mul(4)
            .clamp(1, MAX_GROUP_CANDIDATES);
        loop {
            let round = SearchQuery {
                top_k: fetch,
                limit: fetch,
                with_payload: true,
                payload_fields: inner_fields.clone(),
                ..query.clone()
            };
            let found = self.search(tablename, round).await?;
            let fetched = found.hits.len();

            let mut order: Vec<serde_json::Value> = Vec::new();
            let mut by_key: std::collections::HashMap<String, (usize, Vec<SearchHit>)> =
                std::collections::HashMap::new();
            let mut ungrouped = 0usize;
            for hit in found.hits {
                let value = hit
                    .payload
                    .as_ref()
                    .and_then(|p| p.get(group_by))
                    .filter(|v| !v.is_null())
                    .cloned();
                let Some(value) = value else {
                    ungrouped += 1;
                    continue;
                };
                // Keyed by the value's JSON text: a group id is whatever the payload holds,
                // and `serde_json::Value` is not `Hash` because floats are not.
                let key = value.to_string();
                let entry = by_key.entry(key).or_insert_with(|| {
                    order.push(value.clone());
                    (0, Vec::with_capacity(group_size))
                });
                entry.0 += 1;
                if entry.1.len() < group_size {
                    entry.1.push(hit);
                }
            }

            // Short, and the window may be why. Widen it and try again rather than answering
            // with groups the data does not support.
            if order.len() < groups && fetched >= fetch && fetch < MAX_GROUP_CANDIDATES {
                fetch = fetch.saturating_mul(2).min(MAX_GROUP_CANDIDATES);
                continue;
            }

            let out: Vec<crate::types::record::SearchGroup> = order
                .into_iter()
                .take(groups)
                .map(|value| {
                    let (matched, hits) = by_key
                        .remove(&value.to_string())
                        .expect("every ordered key was inserted");
                    crate::types::record::SearchGroup {
                        group_id: value,
                        hits: hits
                            .into_iter()
                            .map(|mut hit| {
                                hit.payload = if wanted_payload {
                                    hit.payload.as_ref().map(|p| {
                                        crate::types::record::project_payload(
                                            p,
                                            wanted_fields.as_deref(),
                                        )
                                    })
                                } else {
                                    None
                                };
                                hit
                            })
                            .collect(),
                        matched,
                    }
                })
                .collect();

            return Ok(crate::types::record::SearchGroupsResult {
                groups: out,
                candidates_fetched: fetched,
                ungrouped,
                tookms: start.elapsed().as_millis() as u64,
            });
        }
    }

    /// Nearest neighbours by sparse vector alone.
    ///
    /// Sparse vectors could be written, were indexed, and could not be retrieved by: the only
    /// path into `SparseIndex` was `POST /search/hybrid`, which refuses a body with no dense
    /// vector and no text ("At least one of 'vector' or 'text' must be provided") and needs the
    /// `text-bm25` build besides. So a table full of SPLADE vectors answered every request with
    /// the dense side or with an error -- stored, indexed, unreachable.
    ///
    /// The scoring is the dot product the sparse index already computes, WAND-pruned. Filters
    /// are applied to the candidates it returns, by `matches_filter` rather than through the
    /// payload index, which is why the fetch is widened when there is one: the alternative is
    /// a page that comes back short because the filter ate it.
    pub async fn sparse_search(
        &self,
        tablename: &str,
        request: &crate::types::record::SparseSearchQuery,
    ) -> crate::Result<SearchResult> {
        let tablename = &*self.resolve(tablename);
        let start = std::time::Instant::now();
        let query = &request.sparse_vector;
        let limit = request.limit;
        let filter = request.filter.as_ref();
        let with_payload = request.with_payload;
        let with_vector = request.with_vector;
        let payload_fields = request.payload_fields.as_deref();
        let score_threshold = request.score_threshold;
        query
            .validate()
            .map_err(|e| crate::Error::InvalidInput(e.to_string()))?;
        if limit == 0 {
            return Err(crate::Error::InvalidInput(
                "limit must be greater than 0".to_string(),
            ));
        }
        Self::check_filter(filter)?;
        self.ensure_resident(tablename).await?;
        self.ensure_derived(tablename).await?;

        let tables = self.read_tables("sparse_search").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        // Held to the same contract as every other read: a table that excluded a field from
        // its payload index declared it not filterable, and answering here because this path
        // happens to filter by scanning the candidates would make that mean different things
        // at different endpoints.
        Self::check_filter_policy(&table_data.table, filter)?;

        // Three times the page when filtering, as the hybrid path does. It is a guess either
        // way: what the filter removes is unknown until it is applied.
        let fetch = if filter.is_some() {
            limit.saturating_mul(3)
        } else {
            limit
        };
        let candidates = table_data.derived().sparse.search(query, fetch);

        let mut scored: Vec<(RecordId, f32)> = candidates
            .into_iter()
            .filter(|(id, score)| {
                if score_threshold.is_some_and(|floor| *score < floor) {
                    return false;
                }
                match filter {
                    None => true,
                    Some(filter) => table_data
                        .records
                        .get(id)
                        .is_some_and(|record| record.matches_filter(filter)),
                }
            })
            .collect();
        // The index returns its own order; the tiebreak by id is what makes a page well defined
        // when two records score the same, as it does on the dense side.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(limit);
        let total = scored.len();

        let hits: Vec<SearchHit> = scored
            .into_iter()
            .map(|(id, score)| {
                let (payload, vector) = if with_payload || with_vector {
                    table_data
                        .records
                        .get(&id)
                        .map(|record| {
                            (
                                with_payload.then(|| {
                                    crate::types::record::project_payload(
                                        &record.payload,
                                        payload_fields,
                                    )
                                }),
                                with_vector.then(|| table_data.vector_for(&record)),
                            )
                        })
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };
                SearchHit {
                    id,
                    score,
                    payload,
                    vector,
                    rerank_score: None,
                }
            })
            .collect();

        Ok(SearchResult {
            hits,
            total,
            tookms: start.elapsed().as_millis() as u64,
        })
    }

    /// What this table's payload index holds and what it costs, per field.
    ///
    /// Deliberately does not build the index in order to describe it: the build is a pass over
    /// every record, and a question about cost that costs that much is not one an operator can
    /// ask. An unbuilt index reports `built: false` and no fields, which is the truth — a
    /// restored table has none until the first filter.
    pub async fn payload_index_report(&self, tablename: &str) -> crate::Result<serde_json::Value> {
        let tablename = &*self.resolve(tablename);
        self.ensure_resident(tablename).await?;
        let tables = self.read_tables("payload_index_report").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let policy = table_data.table.payload_index.clone().unwrap_or_default();
        let built = table_data.derived.read();
        let Some(derived) = built.as_ref() else {
            return Ok(serde_json::json!({
                "table": tablename,
                "built": false,
                "indexes_every_field": policy.indexes_everything(),
                "only": policy.only,
                "except": policy.except,
                "fields": [],
                "note": "the payload index is built on the first query that filters; nothing to report yet",
            }));
        };
        let fields = derived.payload.field_report();
        let total_bytes: usize = fields.iter().map(|f| f.estimated_bytes).sum();
        Ok(serde_json::json!({
            "table": tablename,
            "built": true,
            "indexes_every_field": policy.indexes_everything(),
            "only": policy.only,
            "except": policy.except,
            "record_count": table_data.records.len(),
            "estimated_bytes": total_bytes + derived.payload.shared_bytes(),
            "shared_bytes": derived.payload.shared_bytes(),
            "fields": fields,
        }))
    }

    pub async fn count_records(
        &self,
        tablename: &str,
        filter: Option<&crate::types::Filter>,
    ) -> crate::Result<usize> {
        let tablename = &*self.resolve(tablename);
        Self::check_filter(filter)?;
        self.ensure_resident(tablename).await?;
        if filter.is_some() {
            self.ensure_derived(tablename).await?;
        }
        let tables = self.read_tables("count_records").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        Self::check_filter_policy(&table_data.table, filter)?;

        Ok(match filter {
            None => table_data.records.len(),
            Some(filter) => {
                let derived = table_data.derived();
                let (_, total) = derived.payload.page_of_matching(filter, 0, 0);
                total
            }
        })
    }

    /// Delete every record a filter matches, and report how many there were.
    ///
    /// Deleting by id is the only shape this had, so "remove everything belonging to tenant X"
    /// could not be expressed at all: a caller had to page the table, collect ids, and send
    /// them back — a round trip per page, at the mercy of anything inserted meanwhile.
    ///
    /// It pages rather than materialising every matching id, because a filter can match the
    /// whole table and the write lock is held per batch, not for the whole delete. Each pass
    /// asks for the *first* page again: the previous pass removed what it returned, so the
    /// matching set shrinks under the loop by construction. A record inserted mid-delete and
    /// matching the filter is deleted too, which is the same answer a single-pass
    /// implementation would give on a table that is being written to.
    pub async fn delete_records_matching(
        &self,
        tablename: &str,
        filter: &crate::types::Filter,
    ) -> crate::Result<usize> {
        let tablename = &*self.resolve(tablename);
        const BATCH: usize = 1_000;

        Self::check_filter(Some(filter))?;
        self.ensure_writable()?;
        self.ensure_resident(tablename).await?;

        let mut deleted = 0usize;
        loop {
            let (records, _) = self
                .list_records(tablename, Some(filter), BATCH, 0, false)
                .await?;
            if records.is_empty() {
                break;
            }
            let ids: Vec<RecordId> = records.into_iter().map(|record| record.id).collect();
            let asked = ids.len();
            deleted += self.delete_records(tablename, ids).await?;
            if asked < BATCH {
                break;
            }
        }
        Ok(deleted)
    }

    pub async fn list_records(
        &self,
        tablename: &str,
        filter: Option<&crate::types::Filter>,
        limit: usize,
        offset: usize,
        with_vector: bool,
    ) -> crate::Result<(Vec<Record>, usize)> {
        let tablename = &*self.resolve(tablename);
        Self::check_filter(filter)?;
        self.ensure_resident(tablename).await?;
        if filter.is_some() {
            self.ensure_derived(tablename).await?;
        }
        let tables = self.read_tables("list_records").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        Self::check_filter_policy(&table_data.table, filter)?;

        // On disk, the store can answer a page without decoding the rest of the table -- which
        // is the whole cost here. Unfiltered it was 194 ms for 25 rows out of 120,000, against
        // 3 ms for a vector search over the same table. Filtered it was worse: the ids came
        // from the payload index and every one of them was decoded to be sorted by
        // `created_at`, so a filter matching 2.5M records spent **7.8 s** to return five,
        // measured at 10M rows.
        match filter {
            None => {
                let total = table_data.records.len();
                if let Some(page) = table_data.records.page_newest_first(offset, limit) {
                    let mut page: Vec<Record> = page.iter().map(|r| (**r).clone()).collect();
                    Self::fill_vectors(table_data, &mut page, with_vector);
                    return Ok((page, total));
                }
            }
            // The index chooses the page and says how many match; only the page is fetched.
            // Every match used to be turned into an owned id and hashed back to a row to be
            // sorted — 2.9 s for a page of five out of 2.5M matches at 10M rows. See
            // `PayloadIndex::page_of_matching`. For an in-memory table as well: its records
            // are cheap to reach, but sorting 2.5M of them by timestamp is not.
            //
            // An id the index names that the store no longer has is skipped rather than
            // failing the page. The index is mutated by deletes as well as rebuilt from the
            // records, so a stale entry is possible, and a short page is the better answer.
            Some(filter) => {
                let derived = table_data.derived();
                let (ids, total) = derived.payload.page_of_matching(filter, offset, limit);
                let mut page: Vec<Record> = ids
                    .iter()
                    .filter_map(|id| table_data.records.get(id))
                    .map(|r| (*r).clone())
                    .collect();
                drop(derived);
                Self::fill_vectors(table_data, &mut page, with_vector);
                return Ok((page, total));
            }
        }

        // `Arc<Record>`, not `Record`. This used to deep-clone every matching record -- payload
        // `serde_json::Value` and all -- to return a page of twenty-five, so a table of 120,000
        // paid 120,000 JSON clones for 25 rows. A refcount bump costs nothing and the page is
        // cloned at the end, where it is twenty-five of them.
        // Unfiltered, on an in-memory table: every filtered list returned above.
        let mut matching: Vec<std::sync::Arc<Record>> = {
            let mut out = Vec::with_capacity(table_data.records.len());
            table_data.records.for_each(|r| out.push(r));
            out
        };

        let total_matching = matching.len();
        let start = offset.min(total_matching);
        let end = (offset + limit).min(total_matching);

        // Newest first, ties by id. The tiebreak is not decoration: without it the order
        // between records sharing a `created_at` is whatever the partition below leaves, and
        // *which* of them lands on a page would then be unspecified -- a page that repeats or
        // skips a record between two calls. Timestamps here are nanosecond-precision and ties
        // are rare, which is exactly the kind of thing that works until someone inserts a batch
        // on a coarser clock.
        let newest_first = |a: &std::sync::Arc<Record>, b: &std::sync::Arc<Record>| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| a.id.cmp(&b.id))
        };

        // Only the page has to be in order. `select_nth_unstable_by` partitions in O(n) and
        // leaves the `end` newest records before the split -- a full sort of 120,000 to answer
        // with 25 of them was the other half of this.
        if end > 0 && end < total_matching {
            matching.select_nth_unstable_by(end - 1, newest_first);
        }
        matching[..end].sort_by(newest_first);
        let mut page: Vec<Record> = matching[start..end].iter().map(|r| (**r).clone()).collect();

        Self::fill_vectors(table_data, &mut page, with_vector);
        Ok((page, total_matching))
    }

    /// Attach or clear each record's vector. Called on the page, never before it is cut:
    /// reading vectors for the whole table to return ten of them would be the expensive half of
    /// a rerank, on every page.
    fn fill_vectors(table_data: &IntelligentTableData, page: &mut [Record], with_vector: bool) {
        for record in page {
            if with_vector {
                if record.vector.is_empty() {
                    record.vector = table_data.vector_for(record);
                }
            } else {
                // Cleared rather than left alone, so an in-memory table answers the same shape
                // as a row-file one. A caller that asked for no vectors and got some anyway
                // would be paying for them without knowing why.
                record.vector = Vec::new();
            }
        }
    }

    /// Simple filter matching helper
    fn matches_filter(payload: &serde_json::Value, filter: &serde_json::Value) -> bool {
        // Simple implementation - checks if all filter fields match
        if let (Some(payload_obj), Some(filter_obj)) = (payload.as_object(), filter.as_object()) {
            for (key, value) in filter_obj {
                if payload_obj.get(key) != Some(value) {
                    return false;
                }
            }
        }
        true
    }

    /// Where a table's Tantivy index lives: `{wal_dir}/fulltext/{table_id}`.
    ///
    /// Named rather than inlined because two places need to agree on it: the engine that opens
    /// it and the drop that removes it. They did not, and the index outlived its table.
    ///
    /// # It used to default to `/tmp`
    ///
    /// The old default was `{DXMT_STORAGE_PATH:-/tmp}/dxmt_full_text_{id}`, which put a real
    /// index — data, not a cache; the only copy of the term postings — in the one directory an
    /// operating system feels free to empty. `systemd-tmpfiles` clears `/tmp` on a schedule on
    /// most distributions and many images wipe it on boot, so a server could come back with its
    /// WAL, its row files and its snapshots all intact and its text search silently answering
    /// nothing. It also sat outside `DXMT_WAL_DIR`, so a backup of that directory — which is what
    /// `docs/operations/backup-restore.md` tells you to take — never contained it.
    ///
    /// Beside the WAL is where the rest of a table's durable state already lives: `records/`,
    /// `vectors/` and `snapshot/` are all siblings of this. One directory to back up, one to
    /// restore, one to delete.
    ///
    /// The shared default had also become a collision. A table id is now a function of its
    /// name, so two servers on one machine — two instances, a test suite, a staging copy — with
    /// a table of the same name derived the *same* id and, under `/tmp`, opened the *same*
    /// index: one holding the other's documents, and contending for Tantivy's writer lock.
    /// Under `DXMT_WAL_DIR` they are as separate as the rest of their state. Reverting this
    /// function to the old path turns five of the six tests in
    /// `tests/full_text_recovery_test.rs` red, and that is how.
    ///
    /// Moving it is `DXMT_WAL_DIR`, or `DXMT_ROOT_DIR` to move everything at once. There is no
    /// separate override: `DXMT_STORAGE_PATH` used to be one, defaulting to `/tmp`, and a second
    /// way to say where an index lives is a second place for it to be lost.
    #[cfg(feature = "text-bm25")]
    fn full_text_index_path(&self, table_id: &uuid::Uuid) -> PathBuf {
        match &self.wal_dir {
            Some(wal_dir) => wal_dir.join("fulltext").join(table_id.to_string()),
            // No durable storage at all -- embedded use, most tests. An empty path tells the
            // engine to keep the index in memory: nothing in this configuration survives the
            // process, so neither should it. A path under the temporary directory looked
            // harmless until table ids became a function of the table's name, at which point
            // two processes using the same table name shared one index.
            None => PathBuf::new(),
        }
    }

    /// 创建全文搜索引擎
    #[cfg(feature = "text-bm25")]
    async fn create_full_text_engine(&self, table: &Table) -> crate::Result<FullTextSearchEngine> {
        let root = self.full_text_index_path(&table.id);
        let path = if root.as_os_str().is_empty() {
            root
        } else {
            text_generation_to_open(&root, table)?
        };
        Self::text_engine_at(table, &path).await
    }

    /// Rebuild a table's text index for its current text config, from its records, without
    /// stopping reads or writes, and make it current. Called by `alter_table` after the change
    /// is logged.
    ///
    /// The new index is built in a new generation directory. While it fills, the current index
    /// keeps answering searches and forwards every write to the new one (`set_shadow`); a table
    /// with no index yet gets the new one at once and fills it in place. Each record is copied
    /// as delete-then-add, under the table map's read lock for its chunk, so however a copy and
    /// a concurrent write interleave the new index ends with one document per record holding
    /// its latest text. Then the new index replaces the old and `CURRENT` names it.
    #[cfg(feature = "text-bm25")]
    async fn rebuild_text_index(&self, name: &str) -> crate::Result<()> {
        let _one_at_a_time = self.text_rebuild.lock().await;
        let (table, old) = {
            let tables = self.read_tables("rebuild_text_index").await;
            let data = tables
                .get(name)
                .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))?;
            (data.table.clone(), data.full_text_engine.clone())
        };
        let root = self.full_text_index_path(&table.id);
        if !table.enable_full_text_search {
            if let Some(data) = self.write_tables("rebuild_text_index").await.get_mut(name) {
                data.full_text_engine = None;
            }
            if !root.as_os_str().is_empty() {
                let _ = std::fs::remove_dir_all(&root);
            }
            return Ok(());
        }

        let generation = if root.as_os_str().is_empty() {
            None
        } else {
            Some(new_text_generation(&root, &table)?)
        };
        let path = generation.clone().unwrap_or_default();
        let engine = Arc::new(Self::text_engine_at(&table, &path).await?);
        match &old {
            Some(old) => old.set_shadow(Some(engine.clone())),
            None => {
                if let Some(data) = self.write_tables("rebuild_text_index").await.get_mut(name) {
                    data.full_text_engine = Some(engine.clone());
                }
            }
        }
        // Counted only now that writes reach the new index: a row written before this is in
        // the range copied below, one written after reached it directly. Counted first, a write
        // between the count and the line above was in neither -- the test that writes during a
        // rebuild found exactly that.
        // `None`: the records are in memory, with no rows to page through.
        let rows = {
            let tables = self.read_tables("rebuild_text_index").await;
            tables.get(name).and_then(|data| data.records.row_count())
        };

        const CHUNK: u32 = 4_096;
        let started = std::time::Instant::now();
        let mut copied = 0usize;
        if rows.is_none() {
            // An in-memory store is copied in one pass, under the read lock throughout, which
            // is what keeps a concurrent write from slipping between a copy and its add.
            let tables = self.read_tables("rebuild_text_index").await;
            if let Some(data) = tables.get(name) {
                let mut docs = Vec::new();
                data.records.for_each(|record| {
                    if let Some(doc) = text_document_for(&table, &record) {
                        docs.push(doc);
                    }
                });
                for doc in docs {
                    engine.delete_document(&doc.id).await?;
                    engine.add_document(doc).await?;
                    copied += 1;
                }
            }
        }
        let rows = rows.unwrap_or(0);
        let mut from = 0u32;
        while from < rows {
            let to = (from + CHUNK).min(rows);
            {
                let tables = self.read_tables("rebuild_text_index").await;
                let Some(data) = tables.get(name) else {
                    return Ok(()); // dropped meanwhile
                };
                let chunk = tokio::task::block_in_place(|| data.records.decode_range(from, to))
                    .unwrap_or_default();
                for record in chunk.into_iter().flatten() {
                    if let Some(doc) = text_document_for(&table, &record) {
                        engine.delete_document(&doc.id).await?;
                        engine.add_document(doc).await?;
                        copied += 1;
                    }
                }
            }
            from = to;
            tokio::task::yield_now().await;
        }
        engine.commit().await?;

        if let Some(data) = self.write_tables("rebuild_text_index").await.get_mut(name) {
            data.full_text_engine = Some(engine.clone());
        }
        // The old index keeps forwarding to the new one: a write that took a handle to it just
        // before the swap still reaches the index that now answers.
        if let Some(generation) = generation {
            make_text_generation_current(&root, &generation)?;
        }
        info!(
            "Table '{name}': text index rebuilt from {copied} records in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// A text engine for `table`'s config over the index at `path` (empty: in memory).
    #[cfg(feature = "text-bm25")]
    async fn text_engine_at(
        table: &Table,
        path: &std::path::Path,
    ) -> crate::Result<FullTextSearchEngine> {
        let index_path = path.to_string_lossy().into_owned();

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&index_path).map_err(|e| {
            crate::Error::Internal(format!("Failed to create index directory: {}", e))
        })?;

        // Build fields configuration for Tantivy
        let mut fields = Vec::new();

        // Always add an ID field
        fields.push((
            "id".to_string(),
            tantivy::schema::TEXT | tantivy::schema::STORED,
        ));

        // Add configured text fields
        if let Some(text_fields) = &table.text_fields {
            for field_config in text_fields {
                // For now, all fields are TEXT | STORED
                // In the future, we can customize based on analyzer type
                fields.push((
                    field_config.name.clone(),
                    tantivy::schema::TEXT | tantivy::schema::STORED,
                ));
            }
        } else {
            // Default: add a generic "content" field
            fields.push((
                "content".to_string(),
                tantivy::schema::TEXT | tantivy::schema::STORED,
            ));
        }

        let default_fields: Vec<String> = fields.iter().map(|(name, _)| name.clone()).collect();

        // The analyzer is per text field in the table's configuration and per index in
        // tantivy's schema, so the table's fields have to agree; one `chinese` field makes the
        // whole index jieba-tokenized, which also segments the ASCII words in mixed text.
        let tokenizer = match &table.text_fields {
            Some(fields)
                if fields
                    .iter()
                    .any(|f| f.analyzer == crate::types::table::TextAnalyzer::Chinese) =>
            {
                "jieba"
            }
            _ => "default",
        };

        let config = FullTextConfig {
            index_path,
            index_memory_mb: 256,
            num_threads: 4,
            enable_phrase_queries: true,
            enable_fuzzy_search: true,
            default_search_fields: default_fields,
            field_boosts: std::collections::HashMap::new(),
            tokenizer: tokenizer.to_string(),
        };

        FullTextSearchEngine::new(config).await
    }
}

// Implement Storage trait for IntelligentIndexedStorage
#[async_trait::async_trait]
impl crate::storage::traits::_Storage for IntelligentIndexedStorage {
    async fn create_table(&self, config: TableConfig) -> crate::Result<Table> {
        IntelligentIndexedStorage::create_table(self, config).await
    }

    async fn hybrid_search(
        &self,
        tablename: &str,
        request: HybridSearchRequest,
    ) -> crate::Result<SearchResult> {
        IntelligentIndexedStorage::hybrid_search(self, tablename, request).await
    }

    async fn put_document(
        &self,
        tablename: &str,
        doc_id: &str,
        request: crate::storage::documents::PutDocumentRequest,
    ) -> crate::Result<crate::storage::documents::DocumentWritten> {
        IntelligentIndexedStorage::put_document(self, tablename, doc_id, request).await
    }

    async fn delete_document(&self, tablename: &str, doc_id: &str) -> crate::Result<usize> {
        IntelligentIndexedStorage::delete_document(self, tablename, doc_id).await
    }

    async fn get_table(&self, name: &str) -> crate::Result<Table> {
        IntelligentIndexedStorage::get_table(self, name).await
    }

    async fn list_tables(&self) -> crate::Result<Vec<Table>> {
        IntelligentIndexedStorage::list_tables(self).await
    }

    async fn delete_table(&self, name: &str) -> crate::Result<()> {
        IntelligentIndexedStorage::delete_table(self, name).await
    }

    async fn alter_table(&self, name: &str, req: TableAlterRequest) -> crate::Result<Table> {
        IntelligentIndexedStorage::alter_table(self, name, req).await
    }

    async fn insert_record(
        &self,
        tablename: &str,
        request: CreateRecordRequest,
    ) -> crate::Result<Record> {
        let records = self.insert_records(tablename, vec![request]).await?;
        records
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::Internal("Failed to insert record".to_string()))
    }

    async fn batch_insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>> {
        self.insert_records(tablename, requests).await
    }

    async fn search(&self, tablename: &str, query: SearchQuery) -> crate::Result<SearchResult> {
        self.search(tablename, query).await
    }

    async fn get_record(&self, tablename: &str, record_id: &RecordId) -> crate::Result<Record> {
        IntelligentIndexedStorage::get_record(self, tablename, record_id).await
    }

    async fn update_record(
        &self,
        tablename: &str,
        record_id: &RecordId,
        request: CreateRecordRequest,
    ) -> crate::Result<Record> {
        self.update_record(tablename, record_id, request).await
    }

    async fn delete_record(&self, tablename: &str, record_id: &RecordId) -> crate::Result<()> {
        self.delete_records(tablename, vec![record_id.clone()])
            .await
            .map(|_| ())
    }

    async fn insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>> {
        self.insert_records(tablename, requests).await
    }

    async fn delete_records(
        &self,
        tablename: &str,
        record_ids: Vec<RecordId>,
    ) -> crate::Result<()> {
        // The trait says nothing about how many were there; the inherent method now does, and
        // `delete_records` in `api/handlers.rs` is the caller that wants it.
        self.delete_records(tablename, record_ids).await.map(|_| ())
    }

    async fn list_all_record_ids(&self, tablename: &str) -> crate::Result<Vec<RecordId>> {
        IntelligentIndexedStorage::list_all_record_ids(self, tablename).await
    }

    async fn backup(&self, table_name: &str, backup_path: &str) -> crate::Result<()> {
        #[derive(serde::Serialize)]
        struct Snapshot {
            table: Table,
            records: Vec<Record>,
        }
        // Copy under the lock, serialise and write without it — the same split the periodic
        // snapshot needed. Measured there: copying 60,000 rows is under 50ms while serialising
        // and writing them is a full second, and a second of held lock stops every table.
        let snap = {
            let tables = self.read_tables("backup").await;
            let data = tables
                .get(table_name)
                .ok_or_else(|| crate::Error::TableNotFound(table_name.to_string()))?;
            Snapshot {
                table: data.table.clone(),
                records: data.records.values_cloned(),
            }
        };
        let json =
            serde_json::to_vec(&snap).map_err(|e| crate::Error::Serialization(e.to_string()))?;
        std::fs::write(backup_path, json)
            .map_err(|e| crate::Error::Internal(format!("backup write failed: {}", e)))?;
        Ok(())
    }

    async fn restore(&self, table_name: &str, backup_path: &str) -> crate::Result<()> {
        #[derive(serde::Deserialize)]
        struct Snapshot {
            table: Table,
            records: Vec<Record>,
        }
        let json = std::fs::read(backup_path)
            .map_err(|e| crate::Error::Internal(format!("backup read failed: {}", e)))?;
        let snap: Snapshot = serde_json::from_slice(&json)
            .map_err(|e| crate::Error::Serialization(e.to_string()))?;

        // Re-create the table with the restored schema, then bulk-insert records,
        // which rebuilds the vector index automatically.
        let table_cfg = TableConfig {
            name: table_name.to_string(),
            hnsw: snap.table.hnsw.clone(),
            quantization: snap.table.quantization.clone(),
            payload_index: snap.table.payload_index.clone(),
            dimension: snap.table.dimension,
            distance_metric: snap.table.distance_metric,
            description: snap.table.description.clone(),
            index_type: None,
            text_fields: snap.table.text_fields.clone(),
            enable_full_text_search: snap.table.enable_full_text_search,
            memory_optimization: snap.table.memory_optimization,
            payload_schema: snap.table.payload_schema.clone(),
            schema_version: snap.table.schema_version,
            named_vectors: None,
            embedding: None,
            rerank: None,
        };
        // Drop existing table if present (best-effort)
        let _ = self.delete_table(table_name).await;
        self.create_table(table_cfg).await?;

        let requests: Vec<CreateRecordRequest> = snap
            .records
            .into_iter()
            .map(|r| CreateRecordRequest {
                id: r.id,
                vector: r.vector,
                sparse_vector: r.sparse_vector,
                payload: Some(r.payload),
                named_vectors: None,
                text: None,
            })
            .collect();
        if !requests.is_empty() {
            self.insert_records(table_name, requests).await?;
        }
        Ok(())
    }
}

impl Clone for PerformanceStats {
    fn clone(&self) -> Self {
        Self {
            total_queries: self.total_queries,
            avg_query_latency_ms: self.avg_query_latency_ms,
            p95_query_latency_ms: self.p95_query_latency_ms,
            avg_recall_rate: self.avg_recall_rate,
            insert_throughput: self.insert_throughput,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CreateRecordRequest;
    use crate::RecordId;
    use crate::SearchQuery;

    /// The traversal budget doubles per decade of rows, and only above a million.
    ///
    /// The numbers on the right are the measured operating points, not a formula's output:
    /// on Cohere 768d one graph reaches recall@100 0.9800 at 1M with a 500-candidate walk
    /// and 0.9638 at 10M with the same walk, needing 1024 there to get back to 0.9800
    /// (`docs/architecture/disk-index.md` §4). A table below the base row count keeps what it was
    /// configured with -- growing `ef` on a short graph buys nothing and costs latency.
    #[test]
    fn the_traversal_budget_doubles_per_decade_of_rows() {
        type S = IntelligentIndexedStorage;
        let ef = S::DEFAULT_EF_SEARCH;

        // Below and at the base: unchanged.
        assert_eq!(S::ef_for_rows(ef, 0), ef);
        assert_eq!(S::ef_for_rows(ef, 10_000), ef);
        assert_eq!(S::ef_for_rows(ef, 1_000_000), ef);
        assert_eq!(S::ef_for_rows(ef, 9_999_999), ef);

        // One doubling per decade, at the decade boundary exactly.
        assert_eq!(S::ef_for_rows(ef, 10_000_000), ef * 2);
        assert_eq!(S::ef_for_rows(ef, 99_999_999), ef * 2);
        assert_eq!(S::ef_for_rows(ef, 100_000_000), ef * 4);
        assert_eq!(S::ef_for_rows(ef, 1_000_000_000), ef * 8);

        // Capped, so a table that outgrows the measurements does not walk unboundedly.
        assert_eq!(S::ef_for_rows(ef, 10_000_000_000), ef * S::EF_SCALE_MAX);
        assert_eq!(S::ef_for_rows(ef, usize::MAX), ef * S::EF_SCALE_MAX);

        // The measured operating points, spelled out.
        assert_eq!(S::ef_for_rows(512, 1_000_000), 512);
        assert_eq!(S::ef_for_rows(512, 10_000_000), 1024);
        assert_eq!(S::ef_for_rows(512, 100_000_000), 2048);
    }

    /// The periodic sweep climbs the size ladder and does not come back down.
    ///
    /// A rebuild is O(n) in both directions, so a table sampled at a low point while it grows
    /// pays it twice. A soak watched `hot` go HNSW -> brute force at 1,000 rows and back at
    /// 38,400 ten seconds later: two full rebuilds for a table that only ever got bigger.
    ///
    /// Both halves are asserted. A test that only checked the refusal would pass if the sweep
    /// stopped rebuilding altogether.
    #[tokio::test]
    async fn the_sweep_upgrades_an_index_but_never_demotes_one(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 1024,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;
        // The size-based selector, not the memory-optimized one, which answers HNSW-PQ at
        // every size and has no ladder to climb.
        let mut config = TableConfig {
            name: "ladder".to_string(),
            dimension: 8,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: Some(false),
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };
        config.name = "ladder".to_string();
        storage.create_table(config).await?;

        // Past the 5,000-row line where the selector leaves brute force for the graph.
        const ROWS: usize = 6_000;
        let records: Vec<CreateRecordRequest> = (0..ROWS)
            .map(|i| CreateRecordRequest {
                id: i.to_string(),
                vector: (0..8).map(|j| ((i * 8 + j) % 11) as f32).collect(),
                sparse_vector: None,
                named_vectors: None,
                payload: None,
                text: None,
            })
            .collect();
        storage.insert_records("ladder", records).await?;

        // Up: a table created at 10,000 estimated rows and filled past the threshold has to
        // reach the graph, or the sweep is doing nothing at all.
        storage.reevaluate_indexes().await?;
        let climbed = storage
            .describe_index("ladder")
            .await
            .expect("the table has an algorithm");
        assert!(
            matches!(climbed, IndexAlgorithm::HNSW { .. }),
            "the sweep did not upgrade a 6,000-row table to the graph: {climbed:?}"
        );

        // Down: delete back under the threshold, where the selector answers brute force again.
        let ids: Vec<String> = (0..ROWS - 100).map(|i| i.to_string()).collect();
        storage.delete_records("ladder", ids).await?;
        let wanted = storage
            .decide_algorithm(100, 8, DistanceMetric::Cosine, false, None, None)
            .await?
            .selected_algorithm;
        assert!(
            matches!(wanted, IndexAlgorithm::BruteForce { .. }),
            "this test needs a size where the selector demotes; it answered {wanted:?}"
        );

        storage.reevaluate_indexes().await?;
        let after = storage
            .describe_index("ladder")
            .await
            .expect("the table has an algorithm");
        assert!(
            matches!(after, IndexAlgorithm::HNSW { .. }),
            "the sweep demoted a shrunken table to {after:?}; a rebuild each way is what the \
             one-way rule exists to prevent"
        );
        Ok(())
    }

    /// A changed `ef_search` reaches an existing table without rebuilding its graph.
    ///
    /// `requires_rebuild_to_reach` leaves `ef_search` out of the rebuild triggers because the
    /// search path reads it per query, but nothing used to write a new value back: a table
    /// kept whatever it was built or restored with, so changing the selector's default moved
    /// nothing until someone paid for a full rebuild. Measured on a 1M-row table: setting the
    /// default to 256 and restarting left recall at 0.9135, exactly the ef=128 figure.
    ///
    /// The expected value is read from a table created now rather than written as a literal,
    /// so this keeps testing the behaviour and not the current constant.
    #[tokio::test]
    async fn a_reevaluation_adopts_a_new_ef_search() -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 1024,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;

        let table = |name: &str| TableConfig {
            name: name.to_string(),
            dimension: 16,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };
        // Below 5,000 rows the selector answers BruteForce, which has no ef_search to adopt
        // and would make this test vacuous. The size-based selector, explicitly: the
        // memory-optimized default answers HNSW-PQ at every size, and the stale algorithm
        // planted below is HNSW, which would make this a rebuild rather than a retune.
        const ROWS: usize = 6_000;
        let mut stale = table("stale");
        stale.memory_optimization = Some(false);
        storage.create_table(stale).await?;
        let records: Vec<CreateRecordRequest> = (0..ROWS)
            .map(|i| CreateRecordRequest {
                id: format!("{}", i),
                vector: (0..16).map(|j| ((i * 16 + j) % 7) as f32).collect(),
                sparse_vector: None,
                named_vectors: None,
                payload: None,
                text: None,
            })
            .collect();
        storage.insert_records("stale", records).await?;

        // Settle the table onto the graph index the selector wants for its size.
        storage.reevaluate_indexes().await?;

        // What the selector answers today. Asked directly rather than read off a second table
        // built for the purpose: the assertion is about the selector's answer, and a second
        // 6,000-row graph build is a minute of the unit suite for nothing.
        let wanted = storage
            .decide_algorithm(ROWS, 16, DistanceMetric::Cosine, false, None, None)
            .await?
            .selected_algorithm
            .configured_ef_search()
            .expect("a graph index with an ef_search");

        // Stand in for a table built or restored under an older default.
        let stale_ef = wanted / 2 + 1;
        {
            let mut tables = storage.write_tables("test").await;
            let data = tables.get_mut("stale").expect("table present");
            data.selected_algorithm = IndexAlgorithm::HNSW {
                m: 16,
                ef_construction: 64,
                ef_search: stale_ef,
            };
        }
        assert_ne!(stale_ef, wanted, "the test set up no divergence to close");

        storage.reevaluate_indexes().await?;

        let adopted = storage
            .describe_index("stale")
            .await
            .and_then(|a| a.configured_ef_search())
            .expect("a graph index with an ef_search");
        assert_eq!(
            adopted, wanted,
            "a re-evaluation left the table on its old ef_search ({} rather than {})",
            adopted, wanted
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_intelligent_storage_creation() -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 1024,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;

        let table_config = TableConfig {
            name: "test_table".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Test table".to_string()),
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        let table = storage.create_table(table_config).await?;
        assert_eq!(table.name, "test_table");
        assert_eq!(table.dimension, 128);
        Ok(())
    }

    #[tokio::test]
    async fn test_intelligent_index_selection() -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 1024,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;

        // Test with different table sizes to trigger different algorithm selections
        let configs = vec![
            ("small_table", 64, 100),
            ("medium_table", 256, 10000),
            ("large_table", 768, 100000),
        ];

        for (name, dim, expected_size) in configs {
            let table_config = TableConfig {
                name: name.to_string(),
                dimension: dim,
                distance_metric: DistanceMetric::Euclidean,
                description: None,
                index_type: None,
                text_fields: None,
                enable_full_text_search: false,
                memory_optimization: None,
                payload_schema: None,
                schema_version: 1,
                named_vectors: None,
                hnsw: None,
                quantization: None,
                payload_index: None,
                embedding: None,
                rerank: None,
            };

            let table = storage.create_table(table_config).await?;
            println!(
                "Created table {} with dimension {}",
                table.name, table.dimension
            );

            // Check that appropriate algorithm was selected
            let tables = storage.tables.read().await;
            let table_data = tables.get(name).ok_or("Table not found")?;
            println!("Selected algorithm: {:?}", table_data.selected_algorithm);
            println!(
                "Decision confidence: {:.2}",
                table_data.index_decision.confidence
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_branch_table() -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 256,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;

        // Create source table and insert a record
        let table_config = TableConfig {
            name: "source".to_string(),
            dimension: 4,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };
        storage.create_table(table_config).await?;

        let req = CreateRecordRequest {
            id: "r1".into(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
            sparse_vector: None,
            payload: Some(serde_json::json!({"tag": "original"})),
            named_vectors: None,
            text: None,
        };
        storage.insert_records("source", vec![req]).await?;

        // Branch the table
        let branch = storage.branch_table("source", "branch_v1").await?;
        assert_eq!(branch.name, "branch_v1");
        assert_eq!(branch.dimension, 4);

        // Branch should have the same record
        let rec = storage.get_record("branch_v1", &"r1".into()).await?;
        assert_eq!(rec.id.to_string(), "r1");

        // Modifications to source do not affect branch
        let req2 = CreateRecordRequest {
            id: "r2".into(),
            vector: vec![0.0, 1.0, 0.0, 0.0],
            sparse_vector: None,
            payload: None,
            named_vectors: None,
            text: None,
        };
        storage.insert_records("source", vec![req2]).await?;
        let branch_tables = storage.tables.read().await;
        let branch_data = branch_tables.get("branch_v1").unwrap();
        assert_eq!(
            branch_data.records.len(),
            1,
            "branch should still have 1 record after source insert"
        );
        Ok(())
    }

    /// The periodic sweep must not move a default table onto EnhancedDiskANN.
    ///
    /// `decide_algorithm` ran the size-based selector for every table, whatever the table was
    /// created with. That selector answers `EnhancedDiskANN` once a table passes a million rows
    /// and its notional all-in-memory HNSW would exceed half the machine's RAM -- about 18M rows
    /// at 768 dimensions on a 121 GB box, about 2M on a 16 GB one. So a default table growing
    /// past that line was rebuilt, O(n), onto an index with no delete support and no test of its
    /// own, and the only reason the 100M acceptance run did not hit it is that
    /// `DXMT_INDEX_ALGORITHM=hnsw-pq` was pinned; the server log still carries the sweep's
    /// "Selecting Enhanced DiskANN: dataset_size=21115700". The sweep now asks with the table's
    /// own setting, so a memory-optimized table stays HNSW-PQ at every size.
    #[tokio::test]
    async fn the_sweep_leaves_a_default_table_on_hnsw_pq_however_large_it_grows(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 256,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;

        // Row counts, not rows: this asks the decision function directly, the way the sweep
        // does. Building a 25M-row table to assert a branch would cost hours.
        for rows in [1_000_000usize, 25_000_000, 100_000_000] {
            let want = storage
                .decide_algorithm(rows, 768, DistanceMetric::Cosine, true, None, None)
                .await?
                .selected_algorithm;
            assert!(
                matches!(want, IndexAlgorithm::HnswPQ { .. }),
                "a memory-optimized table at {rows} rows would be rebuilt as {want:?}"
            );
        }

        // And the size-based selector still answers what it always did, so the assertion above
        // is about the routing rather than about a selector that stopped choosing DiskANN.
        // A billion rows, not 25M: the selector's line is half the machine's memory, and 25M
        // x 768 is about 80 GB -- past half of the 62 GB box this was written on and well
        // short of half of a 503 GB one, where this failed. A billion rows is 3 TB.
        let sized = storage
            .decide_algorithm(
                1_000_000_000,
                768,
                DistanceMetric::Cosine,
                false,
                None,
                None,
            )
            .await?
            .selected_algorithm;
        assert!(
            matches!(sized, IndexAlgorithm::EnhancedDiskANN { .. }),
            "expected the size-based selector to still choose DiskANN, got {sized:?}"
        );
        Ok(())
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn a_rebuild_that_would_change_nothing_is_skipped(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Rebuilding is not free -- 1,259 s for a 1M-row table on the benchmark machine --
        // and once the two index-selection paths agree on ef_construction it reproduces the
        // graph that insert already built (recall@100 0.9591 rebuilt against 0.9582 not).
        // So it has to be skipped when nothing about the decision has changed, and still run
        // when something has. Both halves are asserted here; only asserting the skip would
        // pass just as well if `rebuild_index` had become a no-op outright.
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 256,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;

        // The size-based selector, explicitly. Under the memory-optimized default the table
        // is HNSW-PQ, and for a 6,000-row exact graph a rebuild is not a no-op: it is what
        // trains the codebooks and quantizes it (`wants_rebuild`), so it must not be skipped.
        let table_config = TableConfig {
            name: "skip_rebuild".to_string(),
            dimension: 4,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: Some(false),
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };
        storage.create_table(table_config).await?;

        // Six thousand rows, not eight: below five thousand the selector answers BruteForce
        // while create-table has already installed HNSW, so a small table is a genuine
        // family change and must rebuild. Testing the skip needs both paths to agree.
        const ROWS: usize = 6_000;
        let requests: Vec<_> = (0..ROWS)
            .map(|i| CreateRecordRequest {
                id: format!("rec-{i}"),
                vector: vec![i as f32, 1.0, (i % 7) as f32, (i % 3) as f32],
                sparse_vector: None,
                payload: None,
                named_vectors: None,
                text: None,
            })
            .collect();
        storage.insert_records("skip_rebuild", requests).await?;

        // The index insert path is asynchronous; wait for it to catch up rather than
        // racing it, or this asserts against an index that is legitimately behind.
        for _ in 0..200 {
            let indexed = {
                let tables = storage.read_tables("test").await;
                tables["skip_rebuild"].index.len().await
            };
            if indexed == ROWS {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // Nothing has changed: the rebuild must leave the very same index in place. Identity
        // is checked by pointer, not by counting rows -- a fresh index built from the same
        // eight records would hold eight records too.
        // `clone_arc` hands back the Arc the wrapper holds; its address identifies the
        // allocation, and the table keeps its own reference throughout, so the address
        // cannot be recycled underneath this comparison.
        let before = std::sync::Arc::as_ptr(&{
            let tables = storage.read_tables("test").await;
            tables["skip_rebuild"].index.clone_arc()
        });
        storage.rebuild_index("skip_rebuild").await?;
        let after = std::sync::Arc::as_ptr(&{
            let tables = storage.read_tables("test").await;
            tables["skip_rebuild"].index.clone_arc()
        });
        assert_eq!(
            before, after,
            "a rebuild with nothing to do replaced the index"
        );

        // Now make the live index disagree with the record map, which is what a restore that
        // rebuilt from rows, or a build that died partway, leaves behind. The rebuild has to
        // run for that.
        storage.drop_index("skip_rebuild").await?;
        storage.rebuild_index("skip_rebuild").await?;
        let repaired = {
            let tables = storage.read_tables("test").await;
            tables["skip_rebuild"].index.len().await
        };
        assert_eq!(
            repaired, ROWS,
            "a rebuild skipped the repair it exists to perform"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_rebuild_index() -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 256,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;

        let table_config = TableConfig {
            name: "rebuild_test".to_string(),
            dimension: 4,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };
        storage.create_table(table_config).await?;

        // Insert a few records
        let requests: Vec<_> = (0..5)
            .map(|i| CreateRecordRequest {
                id: format!("rec-{}", i),
                vector: vec![i as f32, 0.0, 0.0, 0.0],
                sparse_vector: None,
                payload: None,
                named_vectors: None,
                text: None,
            })
            .collect();
        storage.insert_records("rebuild_test", requests).await?;

        // Drop the index (leaves BruteForce in place) then rebuild
        storage.drop_index("rebuild_test").await?;
        let (dropped, reported_brute_force_after_drop) = {
            let tables = storage.read_tables("test").await;
            let data = &tables["rebuild_test"];
            (
                data.index.clone_arc(),
                matches!(
                    data.table.index_type,
                    crate::types::table::IndexType::BruteForce
                ),
            )
        };
        storage.rebuild_index("rebuild_test").await?;
        // The rebuild has to run. While the dropped table still claimed its old algorithm, a
        // brute-force index holding every row looked up to date, the rebuild was skipped, and
        // brute force stayed in place. `dropped` is held across the rebuild so its address
        // cannot be reused by the index that replaces it.
        {
            let tables = storage.read_tables("test").await;
            let data = &tables["rebuild_test"];
            assert!(
                !std::sync::Arc::ptr_eq(&data.index.clone_arc(), &dropped),
                "the rebuild after a drop was skipped"
            );
            assert!(!matches!(
                data.table.index_type,
                crate::types::table::IndexType::BruteForce
            ));
        }
        assert!(
            reported_brute_force_after_drop,
            "after a drop the table must report what it now searches with"
        );

        // After rebuild, search should still return all 5 records
        let query = crate::types::record::SearchQuery {
            vector: vec![0.0, 0.0, 0.0, 0.0],
            top_k: 10,
            limit: 10,
            ..Default::default()
        };
        let result = storage.search("rebuild_test", query).await?;
        assert_eq!(
            result.hits.len(),
            5,
            "all records should be reachable after rebuild"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_tie_at_the_cutoff_is_broken_by_id_before_and_after_a_rebuild(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Twenty records at exactly the same distance from the query, inserted in descending id
        // order, and one at the query itself. An index that cuts to `k` inside keeps whichever
        // tied records it met first, so the answer followed build order: the HTTP suite saw
        // r1's fifth neighbour change after a rebuild in memory mode.
        const TIED: usize = 20;
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 256,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;
        storage
            .create_table(TableConfig {
                name: "ties".to_string(),
                dimension: 2,
                distance_metric: DistanceMetric::Euclidean,
                description: None,
                index_type: None,
                text_fields: None,
                enable_full_text_search: false,
                memory_optimization: None,
                payload_schema: None,
                schema_version: 1,
                named_vectors: None,
                hnsw: None,
                quantization: None,
                payload_index: None,
                embedding: None,
                rerank: None,
            })
            .await?;
        let record = |id: String, vector: Vec<f32>| CreateRecordRequest {
            id,
            vector,
            sparse_vector: None,
            payload: None,
            named_vectors: None,
            text: None,
        };
        let mut requests = vec![record("exact".to_string(), vec![1.0, 0.0])];
        requests.extend(
            (0..TIED)
                .rev()
                .map(|i| record(format!("t{i:02}"), vec![0.0, 1.0])),
        );
        storage.insert_records("ties", requests).await?;
        for _ in 0..200 {
            let indexed = {
                let tables = storage.read_tables("test").await;
                tables["ties"].index.len().await
            };
            if indexed == TIED + 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        async fn top3(storage: &IntelligentIndexedStorage) -> crate::Result<Vec<String>> {
            let query = crate::types::record::SearchQuery {
                vector: vec![1.0, 0.0],
                top_k: 3,
                limit: 3,
                ..Default::default()
            };
            Ok(storage
                .search("ties", query)
                .await?
                .hits
                .into_iter()
                .map(|hit| hit.id)
                .collect())
        }

        assert_eq!(top3(&storage).await?, ["exact", "t00", "t01"]);
        storage.drop_index("ties").await?;
        assert_eq!(
            top3(&storage).await?,
            ["exact", "t00", "t01"],
            "after a drop"
        );
        storage.rebuild_index("ties").await?;
        assert_eq!(
            top3(&storage).await?,
            ["exact", "t00", "t01"],
            "after a rebuild"
        );
        Ok(())
    }

    /// One request, the way `insert_records` builds it, with the responder handed back.
    fn queued(
        table: &str,
        ids: &[&str],
        dim: usize,
    ) -> (InsertRequest, oneshot::Receiver<crate::Result<Inserted>>) {
        let (response_tx, response_rx) = oneshot::channel();
        (
            InsertRequest {
                table_name: table.to_string(),
                records: ids
                    .iter()
                    .map(|id| CreateRecordRequest {
                        id: id.to_string(),
                        vector: vec![0.5; dim],
                        sparse_vector: None,
                        named_vectors: None,
                        payload: None,
                        text: None,
                    })
                    .collect(),
                rows_in_file: Vec::new(),
                response_tx,
                enqueued_at: std::time::Instant::now(),
            },
            response_rx,
        )
    }

    /// A group stops at the first request for a different table.
    ///
    /// Merging exists so one index insert can cover several requests, and an index belongs to
    /// one table; a group spanning two would have nothing to insert into.
    #[test]
    fn a_group_covers_one_table() {
        let mut pending = std::collections::VecDeque::new();
        let mut keep = Vec::new();
        for (table, ids) in [("a", ["1"]), ("a", ["2"]), ("b", ["3"]), ("a", ["4"])] {
            let (req, rx) = queued(table, &ids, 2);
            pending.push_back(req);
            keep.push(rx);
        }

        let group = take_group(&mut pending);
        assert_eq!(group.len(), 2, "the run for table a, and no further");
        assert!(group.iter().all(|r| r.table_name == "a"));
        assert_eq!(pending.len(), 2, "table b heads the next group");
        assert_eq!(pending[0].table_name, "b");
    }

    /// A group stops rather than carry the same id twice.
    ///
    /// The second write of an id is an upsert: it wants the first one's vector out of the graph
    /// before its own goes in. Inside one merged batch there is no "before" — the batch would
    /// hold the id twice and the removal would target a vector the index had never been given.
    /// Splitting the group is what keeps that ordering real.
    #[test]
    fn a_group_stops_at_a_repeated_id() {
        let mut pending = std::collections::VecDeque::new();
        let mut keep = Vec::new();
        for ids in [["x", "y"], ["p", "q"], ["y", "z"]] {
            let (req, rx) = queued("t", &ids, 2);
            pending.push_back(req);
            keep.push(rx);
        }

        let group = take_group(&mut pending);
        assert_eq!(group.len(), 2, "stopped before the request that repeats y");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].records[0].id, "y");

        // And the one left behind heads a group of its own, so it is not stranded.
        let next = take_group(&mut pending);
        assert_eq!(next.len(), 1);
        assert!(pending.is_empty());
    }

    /// A single request larger than the record cap still goes through.
    ///
    /// The cap bounds how much merging adds, not how much a caller may send. Reading it as a
    /// limit on the group would leave an oversized request in the queue forever.
    #[test]
    fn the_record_cap_never_strands_a_request() {
        let ids: Vec<String> = (0..MAX_COALESCED_RECORDS + 10)
            .map(|i| i.to_string())
            .collect();
        let refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let (big, _rx) = queued("t", &refs, 2);
        let (after, _rx2) = queued("t", &["late"], 2);
        let mut pending = std::collections::VecDeque::from(vec![big, after]);

        let group = take_group(&mut pending);
        assert_eq!(group.len(), 1, "the oversized request travels alone");
        assert_eq!(group[0].records.len(), MAX_COALESCED_RECORDS + 10);
        assert_eq!(pending.len(), 1, "and does not swallow the one behind it");
    }

    /// A request that fails validation does not take down the requests merged with it.
    ///
    /// Before merging, a bad request could only fail itself. Now several requests share one
    /// pass and one index insert, so the failure has to be contained by hand — and the records
    /// belonging to the bad request must leave no trace, because the check that rejects it now
    /// runs before anything is written rather than partway through.
    #[tokio::test]
    async fn a_bad_request_fails_alone_inside_a_group() -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 256,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;
        storage
            .create_table(TableConfig {
                name: "mixed".to_string(),
                dimension: 4,
                distance_metric: DistanceMetric::Cosine,
                description: None,
                index_type: None,
                text_fields: None,
                enable_full_text_search: false,
                memory_optimization: None,
                payload_schema: None,
                schema_version: 1,
                named_vectors: None,
                hnsw: None,
                quantization: None,
                payload_index: None,
                embedding: None,
                rerank: None,
            })
            .await?;

        let (first, first_rx) = queued("mixed", &["a", "b"], 4);
        // Wrong width, and not in first position: the records staged ahead of it inside the
        // group are the ones a partial failure would corrupt.
        let (bad, bad_rx) = queued("mixed", &["c"], 3);
        let (last, last_rx) = queued("mixed", &["d"], 4);

        process_insert_group(vec![first, bad, last], &storage.tables).await;

        assert_eq!(first_rx.await?.expect("first request succeeded").0.len(), 2);
        assert_eq!(last_rx.await?.expect("last request succeeded").0.len(), 1);
        assert!(
            matches!(
                bad_rx.await?,
                Err(crate::Error::DimensionMismatch {
                    expected: 4,
                    actual: 3
                })
            ),
            "the mismatched request reports its own error"
        );

        for id in ["a", "b", "d"] {
            storage
                .get_record("mixed", &id.to_string())
                .await
                .unwrap_or_else(|e| panic!("{id} should have landed: {e}"));
        }
        assert!(
            storage.get_record("mixed", &"c".to_string()).await.is_err(),
            "the rejected request left nothing behind"
        );
        Ok(())
    }

    /// A non-finite coordinate never reaches the index, whichever entry point sent it.
    ///
    /// The record routes validate vectors in the handler, but DXQL and MCP reach
    /// this processor without passing that validator, and `[1e999, 0, 0, 0]` is valid to the
    /// DXQL grammar and parses to an infinity. Measured before this check existed: the insert
    /// was accepted, and the next search on the table answered `curl: (52) Empty reply from
    /// server` with `panicked at src/search/hnsw.rs:1440` in the log -- an infinite coordinate
    /// makes the cosine distance NaN, and the comparator unwrapped the `None` NaN compares to.
    /// The record was in the WAL, so a restart brought the unsearchable table back.
    #[tokio::test]
    async fn a_non_finite_vector_is_refused_and_leaves_the_table_searchable(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 256,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;
        storage
            .create_table(TableConfig {
                name: "finite".to_string(),
                dimension: 4,
                distance_metric: DistanceMetric::Cosine,
                description: None,
                index_type: None,
                text_fields: None,
                enable_full_text_search: false,
                memory_optimization: None,
                payload_schema: None,
                schema_version: 1,
                named_vectors: None,
                hnsw: None,
                quantization: None,
                payload_index: None,
                embedding: None,
                rerank: None,
            })
            .await?;
        let request = |id: &str, vector: Vec<f32>| CreateRecordRequest {
            id: id.to_string(),
            vector,
            sparse_vector: None,
            named_vectors: None,
            payload: None,
            text: None,
        };

        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let outcome = storage
                .insert_records("finite", vec![request("bad", vec![bad, 0.0, 0.0, 0.0])])
                .await;
            assert!(
                matches!(outcome, Err(crate::Error::InvalidVector(_))),
                "{bad} was accepted: {outcome:?}"
            );
        }

        storage
            .insert_records("finite", vec![request("ok", vec![1.0, 0.0, 0.0, 0.0])])
            .await?;
        let answer = storage
            .search(
                "finite",
                SearchQuery {
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    top_k: 3,
                    limit: 3,
                    ..Default::default()
                },
            )
            .await?;
        assert_eq!(
            answer
                .hits
                .iter()
                .map(|h| h.id.as_str())
                .collect::<Vec<_>>(),
            ["ok"],
            "the rejected records left nothing behind and search still answers"
        );
        Ok(())
    }

    /// Merged requests each get back their own records, in their own order.
    ///
    /// One index insert now covers several callers; each still has to be answered with exactly
    /// the rows it sent, or a client would see someone else's ids come back from its own write.
    #[tokio::test]
    async fn a_group_answers_each_request_with_its_own_rows(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 256,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = IntelligentIndexedStorage::new(&config).await?;
        storage
            .create_table(TableConfig {
                name: "split".to_string(),
                dimension: 4,
                distance_metric: DistanceMetric::Cosine,
                description: None,
                index_type: None,
                text_fields: None,
                enable_full_text_search: false,
                memory_optimization: None,
                payload_schema: None,
                schema_version: 1,
                named_vectors: None,
                hnsw: None,
                quantization: None,
                payload_index: None,
                embedding: None,
                rerank: None,
            })
            .await?;

        let (one, one_rx) = queued("split", &["a", "b"], 4);
        let (two, two_rx) = queued("split", &["c"], 4);
        let (three, three_rx) = queued("split", &["d", "e", "f"], 4);

        process_insert_group(vec![one, two, three], &storage.tables).await;

        let ids = |rows: Vec<Record>| rows.into_iter().map(|r| r.id).collect::<Vec<_>>();
        assert_eq!(ids(one_rx.await??.0), vec!["a", "b"]);
        assert_eq!(ids(two_rx.await??.0), vec!["c"]);
        assert_eq!(ids(three_rx.await??.0), vec!["d", "e", "f"]);
        Ok(())
    }
}
