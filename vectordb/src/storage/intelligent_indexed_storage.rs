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

/// Longest the table map's lock has been held since the last reset, in microseconds.
///
/// The warnings the guards emit are for operators; this is for tests. Four separate call paths
/// in this file have held that lock across an index search or a disk write, each one turning
/// "one slow operation" into "the whole server stops", and each was found only by measuring a
/// running server. A counter makes the property testable directly: a search may take as long as
/// it likes, but the map lock must be free again long before it finishes.
static MAX_TABLE_LOCK_HOLD_MICROS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn record_lock_hold(held: std::time::Duration) {
    let micros = held.as_micros() as u64;
    MAX_TABLE_LOCK_HOLD_MICROS.fetch_max(micros, Ordering::Relaxed);
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
        record_lock_hold(held);
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
        record_lock_hold(held);
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
    CostConstraints, ExpectedPerformance, HardwareCapabilities, IndexAlgorithm, IndexConfig,
    IndexDecision, IntelligentIndexSelector, PerformanceRequirements,
};

use crate::config::StorageConfig;
use crate::search::sparse_index::SparseIndex;
use crate::search::{IndexCollection, IndexWrapper};
use crate::storage::object::TieredStorage;
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
    pub durability_degraded: bool,
    pub durability_error: Option<String>,
    pub resident_table_count: usize,
    pub table_count: usize,
    pub total_records: usize,
    /// RFC3339 timestamp of the most recent successful local WAL fsync.
    pub last_wal_sync_at: Option<String>,
    /// RFC3339 timestamp of the most recent successfully installed snapshot.
    pub last_snapshot_at: Option<String>,
    /// Number of snapshot attempts that failed since process start.
    pub snapshot_failures: u64,
    /// Most recent snapshot failure, if any. A failed snapshot does not make the WAL unreadable;
    /// it is nevertheless useful to surface the reason to operators and metrics consumers.
    pub last_snapshot_error: Option<String>,
}

/// 单表快照。索引在同名 `.idx` 文件里；`has_index` 为 false 表示该索引类型没有磁盘格式
/// （例如 BruteForce），恢复时从 `records` 重新插入即可。
#[derive(serde::Serialize, serde::Deserialize)]
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
#[derive(serde::Deserialize)]
struct VerifiedSnapshot {
    #[allow(dead_code)]
    table: Table,
    #[allow(dead_code)]
    algorithm: IndexAlgorithm,
    /// Never read, and that is the point: the work happens in its `Deserialize`, which
    /// decodes each record and drops it. Reading it would defeat the purpose.
    #[allow(dead_code)]
    records: DecodedAndDropped,
    has_index: bool,
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
    /// A disk table's rows when the snapshot is mirrored to object storage: the records
    /// (without vectors) and where each vector is, so the vectors can be read in after the
    /// lock is released. The row files are the durable copy on this machine, but a machine
    /// recovering from the object store alone has no row files, and a snapshot that carries
    /// no records would give it an empty table. Read from the row file rather than copied
    /// out of memory, so this is the slow snapshot again -- only in that configuration.
    mirrored_rows: Option<MirroredRows>,
}

/// A disk table's rows as a mirrored snapshot copies them out: the vector file, and each
/// record with the row its vector is at.
type MirroredRows = (
    Option<Arc<crate::storage::vector_file::VectorFile>>,
    Vec<(Arc<Record>, Option<u64>)>,
);

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
/// **单写者。** 更新清单是读-改-写，没有 CAS——`ObjectStorage` trait 没有条件写
/// （`If-Match`），而 wal3 正是靠它做到多写者安全的。两个进程共用同一个桶会互相覆盖清单。
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
struct WalManifest {
    chunks: Vec<WalChunkRef>,
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
struct InsertRequest {
    table_name: String,
    records: Vec<CreateRecordRequest>,
    response_tx: oneshot::Sender<crate::Result<Vec<Record>>>,
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

    /// 对象存储上 WAL 的清单，常驻内存，每次变更后回写。
    wal_manifest: Arc<StdMutex<WalManifest>>,

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
    durability_error: Arc<StdMutex<Option<String>>>,

    /// Operational durability telemetry. These fields are intentionally process-local: a
    /// restart resets the counters, while the WAL/snapshot sequence remains the durable source
    /// of truth.
    last_wal_sync_at: Arc<StdMutex<Option<String>>>,
    last_snapshot_at: Arc<StdMutex<Option<String>>>,
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
    last_snapshot_error: Arc<StdMutex<Option<String>>>,

    /// GPU distance calculator (None if no GPU available or feature not enabled)
    #[cfg(feature = "gpu")]
    gpu_calculator: Option<Arc<crate::acceleration::GpuDistanceCalculator>>,
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
#[derive(Default)]
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
    /// 被内核杀掉(design doc §34.3)。记录从不原地修改、只整体替换,所以引用计数就够,
    /// 一次快照的拷贝从每条 9.5 KB 降到 8 字节。
    /// Where the records live: the map, or a row file on disk (`DB_RECORD_STORE=disk`,
    /// phase 3 of design/DISK_INDEX.md). Same interface either way; `get` decodes from
    /// disk when that is where the record is.
    records: crate::storage::record_store::RecordStore,

    /// Full vectors on disk (`DB_VECTOR_STORE=disk`, design/DISK_INDEX.md section 4, phase 2).
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

    /// The two indexes that are a pure function of the records, and are rebuilt rather than
    /// restored. `None` means "not built yet" -- see [`IntelligentTableData::derived`].
    derived: parking_lot::RwLock<Option<DerivedIndexes>>,

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
            built.payload.begin_bulk();
            let mut records = 0usize;
            self.records.for_each(|record| {
                records += 1;
                built.payload.insert_record(&record.id, &record.payload);
                if let Some(sparse) = &record.sparse_vector {
                    if let Err(e) = built.sparse.insert(record.id.clone(), sparse) {
                        warn!("Failed to index sparse vector for {}: {}", record.id, e);
                    }
                }
            });
            built.payload.end_bulk();
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
    fn derived_mut(&self) -> parking_lot::MappedRwLockWriteGuard<'_, Option<DerivedIndexes>> {
        parking_lot::RwLockWriteGuard::map(self.derived.write(), |slot| slot)
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
/// `DB_INDEX_COMPACTION_MIN_SLOTS` overrides it, for tests: reaching a hundred thousand slots
/// honestly means a hundred thousand insertions, which says nothing about the behaviour under
/// test that a thousand does not.
fn min_compaction_slots() -> usize {
    const DEFAULT: usize = 100_000;
    std::env::var("DB_INDEX_COMPACTION_MIN_SLOTS")
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
/// `DB_ROW_RECLAIM_MIN_BYTES` overrides it. That exists for the tests: reaching 64 MB of dead
/// rows honestly takes tens of thousands of insertions, which is a five-minute test in a debug
/// build to assert something that has nothing to do with the size of the number.
fn min_reclaim_bytes() -> u64 {
    const DEFAULT: u64 = 64 * 1024 * 1024;
    std::env::var("DB_ROW_RECLAIM_MIN_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT)
}

const MAX_COALESCED_REQUESTS: usize = 32;
const MAX_COALESCED_RECORDS: usize = 50_000;

/// 后台插入处理器 - 单线程处理所有插入，消除锁竞争
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
    let mut senders: Vec<Option<oneshot::Sender<crate::Result<Vec<Record>>>>> =
        Vec::with_capacity(group.len());
    let mut batches: Vec<Vec<CreateRecordRequest>> = Vec::with_capacity(group.len());
    for req in group {
        senders.push(Some(req.response_tx));
        batches.push(req.records);
    }

    let result: crate::Result<Vec<(usize, Vec<Record>)>> = async {
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
        if let Some(file) = &vector_file {
            let mut items: Vec<(&str, &[f32])> = Vec::new();
            let mut slots: Vec<(usize, usize)> = Vec::new();
            for (slot, batch) in batches.iter().enumerate() {
                for (i, r) in batch.iter().enumerate() {
                    if r.vector.len() == expected_dim {
                        items.push((r.id.as_str(), r.vector.as_slice()));
                        slots.push((slot, i));
                    }
                }
            }
            let rows = file.append(&items)?;
            appended_rows = batches.iter().map(|b| vec![None; b.len()]).collect();
            for ((slot, i), row) in slots.into_iter().zip(rows) {
                appended_rows[slot][i] = Some(row);
            }
        }

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
            let mut owners: Vec<(usize, Vec<Record>)> = Vec::new();
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
                // MCP and simple mode all reach this processor without passing that validator.
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
                for (i, request) in batch.into_iter().enumerate() {
                    let record_id = request.id.clone();
                    let row = appended_rows.get(slot).and_then(|b| b[i]);

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
                            derived
                                .payload
                                .update_record(&record_id, old_payload, &record.payload);
                        } else {
                            derived.payload.insert_record(&record_id, &record.payload);
                        }
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
                owners.push((slot, records));
            }

            // One append (and, on disk, one fsync) for the whole group.
            table_data.records.insert_batch(to_store)?;
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
        let named_work: Vec<(RecordId, Option<HashMap<String, DenseVector>>)> = to_insert
            .iter()
            .map(|(id, _, named)| (id.clone(), named.clone()))
            .collect();
        index
            .insert_batch(
                to_insert
                    .into_iter()
                    .map(|(id, vector, _)| (id, vector))
                    .collect(),
            )
            .await?;

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
                for record in owners.iter().flat_map(|(_, records)| records) {
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
        }
        #[cfg(not(feature = "text-bm25"))]
        let _ = &table_meta;

        Ok(owners)
    }
    .await;

    // 发送响应
    match result {
        Ok(owners) => {
            for (slot, records) in owners {
                if let Some(tx) = senders[slot].take() {
                    if tx.send(Ok(records)).is_err() {
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
        let mut storage = Self::build(config).await?;

        if let Some(dir) = wal_dir {
            let manager = WalManager::new(&dir)?;
            storage.wal = Some(Arc::new(StdMutex::new(manager)));
            storage.wal_dir = Some(dir.clone());

            // Before anything reads chunks: the manifest is how they are found.
            if storage.object_storage_enabled {
                storage.load_wal_manifest().await?;
            }

            // Snapshot first, then only the log written after it. Replay has to run with the
            // snapshot already in place: the tail may delete or overwrite records the snapshot
            // restored, and applying it to an empty store would lose those edits.
            let from_seq = storage.load_snapshot(&dir).await?;

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

            if pending.is_empty() {
                info!(
                    "No WAL entries after sequence {}; recovery complete from snapshot alone",
                    from_seq
                );
            } else {
                storage.replay(pending).await?;
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
        let default_memory_optimization = std::env::var("DB_MEMORY_OPTIMIZATION")
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

        // Initialize GPU distance calculator if available
        #[cfg(feature = "gpu")]
        let gpu_calculator = {
            let gpu_config = crate::acceleration::GpuDistanceConfig::from_env();
            match crate::acceleration::GpuDistanceCalculator::try_new(gpu_config) {
                Some(calc) => {
                    info!("GPU acceleration initialized: {}", calc.device_name());
                    Some(Arc::new(calc))
                }
                None => {
                    info!("No GPU detected, using CPU+SIMD for vector operations");
                    None
                }
            }
        };

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
            wal_manifest: Arc::new(StdMutex::new(WalManifest::default())),
            replaying: replaying_flag,
            durability_error: Arc::new(StdMutex::new(None)),
            last_wal_sync_at: Arc::new(StdMutex::new(None)),
            last_snapshot_at: Arc::new(StdMutex::new(None)),
            snapshot_sequence: Arc::new(StdMutex::new(None)),
            snapshot_table_names: Arc::new(StdMutex::new(Vec::new())),
            dropped_since_snapshot: Arc::new(StdMutex::new(std::collections::HashSet::new())),
            snapshot_failures: AtomicU64::new(0),
            memory_guard: crate::storage::memory_guard::MemoryGuard::from_env(),
            last_snapshot_error: Arc::new(StdMutex::new(None)),
            #[cfg(feature = "gpu")]
            gpu_calculator,
        })
    }

    fn ensure_writable(&self) -> crate::Result<()> {
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
        self.memory_guard.check()
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
        let Some(ref wal) = self.wal else {
            return Err(crate::Error::InvalidInput(
                "WAL failpoint requires a WAL-backed storage".to_string(),
            ));
        };
        let mut guard = wal
            .lock()
            .map_err(|_| crate::Error::Internal("WAL lock poisoned".to_string()))?;
        guard.fail_next_sync_for_test();
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

        for (seq, entry) in entries {
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
                WalEntry::UpsertRecordV2 { table, record } => {
                    self.restore_record_from_wal(&table, record).await.map(|_| {
                        records += 1;
                    })
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

    /// Replay one current-format record through the normal index path, then restore metadata
    /// that is not part of a create request. Replay is single-threaded and WAL writes are
    /// suppressed while this runs, so replacing the map value after indexing is deterministic.
    async fn restore_record_from_wal(&self, table: &str, record: Record) -> crate::Result<()> {
        // A mutation is logged after it has been applied and after the table lock is released,
        // so two concurrent updates to the same id can reach the log in the opposite order to
        // the one in which they took effect. Replaying that log blindly would resurrect the
        // older version. The record carries the version the writer assigned, so an entry that
        // is behind what is already restored is a reordered write, not a new one: skip it.
        {
            let tables = self.read_tables("restore_record_from_wal").await;
            if let Some(existing) = tables
                .get(table)
                .and_then(|data| data.records.get(&record.id))
            {
                if existing.version > record.version {
                    return Ok(());
                }
            }
        }

        let request = CreateRecordRequest {
            id: record.id.clone(),
            vector: record.vector.clone(),
            sparse_vector: record.sparse_vector.clone(),
            named_vectors: record.named_vectors.clone(),
            payload: Some(record.payload.clone()),
        };
        self.insert_records(table, vec![request]).await?;

        let mut tables = self.write_tables("restore_record_from_wal").await;
        let data = tables
            .get_mut(table)
            .ok_or_else(|| crate::Error::TableNotFound(table.to_string()))?;
        data.records.insert(Arc::new(record.clone()), None)?;
        data.table.record_count = data.records.len() as u64;
        if record.updated_at > data.table.updated_at {
            data.table.updated_at = record.updated_at;
        }
        Ok(())
    }

    /// 一张表的快照元数据。索引本体在同名的 `.idx` 文件里。
    ///
    /// 记录也一并存下来：只存索引不够——索引里只有 id 和向量，payload、版本号、时间戳都在
    /// `records` map 里，光靠索引恢复出来的表 payload 全空。
    fn snapshot_dir(wal_dir: &std::path::Path) -> PathBuf {
        wal_dir.join("snapshot")
    }

    /// 把当前所有表写成快照，并记下 WAL 的下一个序号。
    ///
    /// 恢复时先加载快照、再只回放该序号之后的日志，避免把每条记录重新插一遍索引——
    /// 那正是"启动时间随数据量线性增长"的来源。
    ///
    /// 快照与 WAL 之间的顺序很关键：先取序号、再落盘。反过来的话，取序号之前写入的条目
    /// 会既在快照里、又被当成尾部重放一次；插入是 upsert 所以不会错，只是白做功。
    pub async fn snapshot(&self) -> crate::Result<usize> {
        let result = self.snapshot_inner().await;
        match &result {
            Ok(_) => {
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

        // Serialize before reading the sequence number, not after: two snapshots that both
        // sampled the sequence and then queued would still truncate the log to the later one's
        // position while the earlier one's manifest is what ends up on disk.
        let _serialized = self.snapshotting.lock().await;

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
        let pending: Vec<(String, crate::search::IndexWrapper, TableTemplate)> = {
            let tables = self.read_tables("snapshot").await;
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
                            mirrored_rows: (self.object_storage_enabled
                                && data.records.is_on_disk())
                            .then(|| {
                                let mut rows = Vec::with_capacity(data.records.len());
                                data.records.for_each(|r| {
                                    let row = data.vector_row(&r.id);
                                    rows.push((r, row));
                                });
                                (data.vector_file.clone(), rows)
                            }),
                        },
                    )
                })
                .collect()
        };
        let waited = before_lock.elapsed();
        let lock_held = std::time::Instant::now();
        let copied_out = lock_held.duration_since(before_lock) - waited;

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
            let has_index = index.snapshot_to(&idx_path).await.map_err(|e| {
                crate::Error::Internal(format!(
                    "table '{}': index would not write itself to the snapshot: {}",
                    name, e
                ))
            })?;

            let records = match template.mirrored_rows {
                None => template.records,
                Some((file, rows)) => rows
                    .into_iter()
                    .map(|(record, row)| {
                        if !record.vector.is_empty() {
                            return record;
                        }
                        let Some(vector) = file.as_ref().zip(row).and_then(|(f, r)| f.read(r))
                        else {
                            return record;
                        };
                        let mut with_vector = (*record).clone();
                        with_vector.vector = vector;
                        Arc::new(with_vector)
                    })
                    .collect(),
            };
            let meta = TableSnapshot {
                table: template.table,
                algorithm: template.algorithm,
                records,
                has_index,
            };
            crate::search::index_persistence::write_index(
                &staging.join(format!("{}.meta", stem)),
                "table-snapshot",
                &meta,
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
        if self.object_storage_enabled {
            match self.offload_snapshot(&dir, written).await {
                Ok(bytes) => info!("Snapshot offloaded to object storage: {} bytes", bytes),
                Err(e) => warn!("Snapshot is local-only, upload failed: {}", e),
            }
        }

        // Only now is it safe to drop the log this snapshot covers. Compacting any earlier
        // leaves a window where neither the previous snapshot nor the log can reconstruct
        // those writes.
        match wal
            .lock()
            .map_err(|_| crate::Error::Internal("WAL lock poisoned".to_string()))
            .and_then(|mut g| g.truncate_before(seq))
        {
            Ok(reclaimed) if reclaimed > 0 => {
                info!(
                    "WAL compacted below sequence {}: {} bytes reclaimed",
                    seq, reclaimed
                );
                if self.object_storage_enabled {
                    self.prune_wal_chunks(seq).await;
                }
            }
            Ok(_) => {}
            // The snapshot is durable already, so a failed compaction costs disk and startup
            // time, not correctness.
            Err(e) => warn!("Snapshot written but WAL compaction failed: {}", e),
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

    /// 快照段在对象存储里的键。
    fn snapshot_key(name: &str) -> String {
        format!("snapshot/{}", name)
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
    async fn offload_snapshot(&self, dir: &std::path::Path, tables: usize) -> crate::Result<u64> {
        use crate::storage::object::traits::ObjectStorage;

        let mut bytes = 0u64;
        for name in Self::snapshot_segments(dir, tables) {
            let data = std::fs::read(dir.join(&name))?;
            bytes += data.len() as u64;
            self.storage
                .put_object(
                    &Self::snapshot_key(&name),
                    bytes::Bytes::from(data),
                    Some("application/octet-stream"),
                )
                .await
                .map_err(|e| {
                    crate::Error::Internal(format!("uploading snapshot segment {}: {}", name, e))
                })?;
        }
        Ok(bytes)
    }

    /// 从对象存储把快照取回本地，供恢复使用。返回是否取到。
    async fn fetch_snapshot(&self, dir: &std::path::Path) -> crate::Result<bool> {
        use crate::storage::object::traits::ObjectStorage;

        let manifest_bytes = match self
            .storage
            .get_object(&Self::snapshot_key("manifest"))
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
                match self.storage.get_object(&Self::snapshot_key(&name)).await {
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
        let guard = self.tables.read().await;
        TimedRead {
            guard,
            who,
            since: std::time::Instant::now(),
        }
    }

    /// Take the table map's write lock, reporting it if it is held for long enough to matter.
    async fn write_tables(&self, who: &'static str) -> TimedWrite<'_> {
        let guard = self.tables.write().await;
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
        let meta: TableSnapshot = crate::search::index_persistence::read_index(
            &dir.join(format!("{}.meta", stem)),
            "table-snapshot",
        )?;

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
        self.restore_table(&dir.join(format!("{}.idx", stem)), meta)
            .await?;
        self.tier_policy.set_current_tier(table_name, Tier::Warm);
        Ok(())
    }

    /// 把一份自洽的备份写到 `dest`。
    ///
    /// W7 给 WAL 加了截断之后，快照就成了它所覆盖那段数据的**唯一副本**——`snapshot/` 目录
    /// 从缓存变成了必须备份的数据。这个方法产出的目录包含恢复所需的全部内容。
    ///
    /// 先强制快照再拷贝，顺序不能反：快照会把内存里尚未落盘的写入固定下来，并把日志压到
    /// 快照点。反过来拷的话，拿到的是一份旧快照配一段与之不匹配的日志。
    /// Where backups may be written: `DB_BACKUP_ROOT`, or the WAL directory's parent.
    ///
    /// The parent rather than the WAL directory itself, because a backup inside it is refused
    /// for a separate reason -- so the default is the nearest place that is both obvious and
    /// allowed.
    fn backup_root(wal_dir: &std::path::Path) -> PathBuf {
        match std::env::var("DB_BACKUP_ROOT") {
            Ok(root) if !root.trim().is_empty() => PathBuf::from(root),
            _ => wal_dir
                .parent()
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from(".")),
        }
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

    pub async fn backup_to(&self, dest: &std::path::Path) -> crate::Result<serde_json::Value> {
        let Some(ref wal_dir) = self.wal_dir else {
            return Err(crate::Error::InvalidInput(
                "backup requires WAL-backed storage; this instance has no durable state"
                    .to_string(),
            ));
        };

        // A backup must be a separate tree. Writing it into DB_WAL_DIR would make the source
        // and destination overlap (`snapshot/snapshot/...`), and writing it below DB_WAL_DIR
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
        // no configuration at all should land. `DB_BACKUP_ROOT` moves it -- to the mount the
        // backups actually belong on, usually.
        let root = Self::backup_root(wal_dir);
        let root = std::fs::canonicalize(&root).unwrap_or(root);
        if !destination.starts_with(&root) {
            return Err(crate::Error::InvalidInput(format!(
                "backup destination {} is outside {}; set DB_BACKUP_ROOT to the directory \
                 backups may be written to",
                dest.display(),
                root.display()
            )));
        }

        let tables = self.snapshot().await?;

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
            "restore": "Stop the server, replace the contents of DB_WAL_DIR with this \
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

        // Threshold on bytes of vector data, not row count. Row count says nothing about
        // memory: a million 1536-dimension vectors is 6 GB, a million 64-dimension vectors is
        // 256 MB, and only one of those needs to leave memory.
        //
        // `DB_PARTITION_THRESHOLD` (a row count) is still read so existing configurations do
        // not silently change meaning, but it is deprecated.
        let vector_bytes = record_count * dimension * std::mem::size_of::<f32>();
        let threshold_bytes = std::env::var("DB_PARTITION_THRESHOLD_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(512 * 1024 * 1024);

        let by_rows = std::env::var("DB_PARTITION_THRESHOLD")
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

    /// Where a store lives, from `DB_RECORD_STORE` / `DB_VECTOR_STORE`: `disk` unless the
    /// variable says `memory`. Anything else is reported and treated as the default.
    fn store_on_disk_wanted(var: &str) -> bool {
        match std::env::var(var).ok().as_deref() {
            None | Some("disk") => true,
            Some("memory") => false,
            Some(other) => {
                warn!("{var}={other} is neither disk nor memory; using disk");
                true
            }
        }
    }

    /// Records live in a row file under the WAL directory (`DB_RECORD_STORE=disk`, the
    /// default since design/DISK_INDEX.md phase 3 was accepted) or in memory
    /// (`DB_RECORD_STORE=memory`, and always when there is no WAL directory to put a file
    /// in). Like the vector file: keyed by table name, wiped by a fresh `CREATE TABLE`,
    /// reused by a restore or replay. A reopened table keeps the store it already has: a
    /// row file on disk is that table's data, whatever the environment says today.
    fn open_record_store(
        &self,
        table: &Table,
        fresh: bool,
    ) -> crate::Result<crate::storage::record_store::RecordStore> {
        use crate::storage::record_store::{RecordStore, RowStore};
        let wanted = Self::store_on_disk_wanted("DB_RECORD_STORE");
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
                "table '{}': DB_RECORD_STORE=memory, but its records are on disk at {}; keeping them there",
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

    /// Full vectors live in a row file under the WAL directory (`DB_VECTOR_STORE=disk`, the
    /// default since design/DISK_INDEX.md phase 2 was accepted) or inside the records
    /// (`DB_VECTOR_STORE=memory`, and always without a WAL directory). As with the record
    /// store, a reopened table keeps the file it already has.
    fn open_vector_file(&self, table: &Table, fresh: bool) -> crate::Result<OpenedVectorFile> {
        use crate::storage::vector_file::{VectorFile, VectorFormat};
        let wanted = Self::store_on_disk_wanted("DB_VECTOR_STORE");
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
                "table '{}': DB_VECTOR_STORE=memory, but its vectors are on disk at {}; keeping them there",
                table.name,
                Self::vector_file_path_for(wal_dir, &table.name, found).display()
            );
        }
        // A fresh table takes the format the environment asks for (`DB_VECTOR_F16=1` for
        // half precision); a reopened one keeps whichever file it already has.
        let wanted_format = if std::env::var("DB_VECTOR_F16").ok().as_deref() == Some("1") {
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
    /// (`DB_MEMORY_OPTIMIZATION`, on unless set to `false`).
    fn memory_optimization_of(&self, table: &Table) -> bool {
        table
            .memory_optimization
            .unwrap_or(self.default_memory_optimization)
    }

    /// `DB_INDEX_ALGORITHM=hnsw|hnsw-pq` pins the index family for every table, at creation
    /// and at re-evaluation alike, so the periodic re-check does not rebuild it back.
    ///
    /// For measuring one index against another on the same data without touching the
    /// selector's size thresholds -- the HNSW-PQ acceptance runs in DISK_INDEX.md section 4
    /// are made this way. Unset, the selector decides as before.
    fn forced_algorithm(dimension: usize) -> Option<IndexAlgorithm> {
        match std::env::var("DB_INDEX_ALGORITHM").ok()?.as_str() {
            "hnsw" => Some(IndexAlgorithm::HNSW {
                m: 16,
                ef_construction: Self::default_ef_construction(),
                ef_search: Self::DEFAULT_EF_SEARCH,
            }),
            "hnsw-pq" => Some(IndexAlgorithm::HnswPQ {
                m: 16,
                ef_construction: Self::default_ef_construction(),
                ef_search: Self::DEFAULT_EF_SEARCH,
                num_subvectors: dimension / 4,
                num_centroids: 256,
                enable_reranking: true,
            }),
            other => {
                warn!("DB_INDEX_ALGORITHM={other} is not hnsw or hnsw-pq; ignoring it");
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

            let (dimension, metric, memory_optimization) = {
                let tables = self.read_tables("reevaluate_indexes").await;
                match tables.get(&name) {
                    Some(d) => (
                        d.table.dimension,
                        d.table.distance_metric,
                        self.memory_optimization_of(&d.table),
                    ),
                    // Evicted or dropped while we were iterating; nothing to do.
                    None => continue,
                }
            };

            let want = self
                .decide_algorithm(count, dimension, metric, memory_optimization)
                .await?
                .selected_algorithm;

            if !current.requires_rebuild_to_reach(&want) {
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
        Ok(rebuilt)
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
    /// This runs on its own timer (`DB_ROW_COMPACTION_INTERVAL_SECS`) rather than inside the
    /// eviction or re-index sweep: it is cheap, it wants to run more often than a rebuild, and
    /// folding it into a sweep that also snapshots would tie giving back disk to a decision
    /// about memory.
    ///
    /// Reported bytes are measured, not derived: allocated blocks before and after. Punching a
    /// range that is already a hole succeeds and releases nothing, and deriving the number
    /// from the dead rows would report the same bytes again on every sweep.
    pub async fn reclaim_dead_rows(&self) -> crate::Result<Vec<(String, u64)>> {
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
            let meta: VerifiedSnapshot = crate::search::index_persistence::read_index(
                &dir.join(format!("{}.meta", stem)),
                "table-snapshot",
            )?;
            if meta.has_index {
                let idx = dir.join(format!("{}.idx", stem));
                if !idx.exists() {
                    return Err(crate::Error::Internal(format!(
                        "snapshot for table '{}' claims an index but {} is missing",
                        meta.table.name,
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
            let meta: TableSnapshot = crate::search::index_persistence::read_index(
                &dir.join(format!("{}.meta", stem)),
                "table-snapshot",
            )
            .map_err(|e| {
                crate::Error::Internal(format!("snapshot table {} is unreadable: {}", i, e))
            })?;

            let name = meta.table.name.clone();
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
            payload_schema: meta.table.payload_schema.clone(),
            schema_version: meta.table.schema_version,
            named_vectors: meta.table.named_vectors.clone(),
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
    async fn wal_append(&self, entries: Vec<WalEntry>) -> crate::Result<()> {
        if self.replaying.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.ensure_writable()?;
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
            if let Some((first_seq, _)) = stamped.first() {
                if let Err(e) = self.offload_wal_chunk(*first_seq, &stamped).await {
                    // The write is already durable locally, so this is a reduction in blast
                    // radius, not a lost write. Say so rather than failing the caller.
                    warn!("WAL chunk {} stayed local, upload failed: {}", first_seq, e);
                }
            }
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
    fn wal_chunk_key(first_seq: u64) -> String {
        format!("wal/{:08}/{:020}.chunk", first_seq / 4096, first_seq)
    }

    /// 读回内存里那份清单。
    fn manifest(&self) -> crate::Result<WalManifest> {
        self.wal_manifest
            .lock()
            .map(|m| m.clone())
            .map_err(|_| crate::Error::Internal("WAL manifest lock poisoned".to_string()))
    }

    /// 把清单写到对象存储。
    async fn put_manifest(&self, manifest: &WalManifest) -> crate::Result<()> {
        use crate::storage::object::traits::ObjectStorage;

        let bytes =
            serde_json::to_vec(manifest).map_err(|e| crate::Error::Serialization(e.to_string()))?;
        self.storage
            .put_object(
                Self::wal_manifest_key(),
                bytes::Bytes::from(bytes),
                Some("application/json"),
            )
            .await
            .map_err(|e| crate::Error::Internal(format!("writing WAL manifest: {}", e)))
    }

    /// 启动时把清单读进内存；没有清单就从前缀里重建一份。
    async fn load_wal_manifest(&self) -> crate::Result<()> {
        use crate::storage::object::traits::ObjectStorage;

        let manifest = match self.storage.get_object(Self::wal_manifest_key()).await {
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
        })
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

        let path = Self::wal_chunk_key(first_seq);
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
        let manifest = {
            let mut guard = self
                .wal_manifest
                .lock()
                .map_err(|_| crate::Error::Internal("WAL manifest lock poisoned".to_string()))?;
            guard.chunks.push(WalChunkRef {
                path,
                first_seq,
                last_seq: entries.last().map(|(s, _)| *s).unwrap_or(first_seq),
                entries: entries.len(),
            });
            guard.total_entries += entries.len() as u64;
            guard.clone()
        };
        self.put_manifest(&manifest).await
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

    /// 删掉全部条目都低于 `keep_from` 的 WAL 分块。
    ///
    /// **先更新清单再删对象。** 反过来的话，中途失败会留下一份指向已删对象的清单，
    /// 恢复就会报"分块读不出来"——而实际上那些数据已经在快照里了，警告是假的。
    async fn prune_wal_chunks(&self, keep_from: u64) {
        use crate::storage::object::traits::ObjectStorage;

        let (manifest, dropped) = {
            let Ok(mut guard) = self.wal_manifest.lock() else {
                return;
            };
            let mut dropped = Vec::new();
            guard.chunks.retain(|c| {
                if c.last_seq < keep_from {
                    guard_pruned(&mut dropped, c);
                    false
                } else {
                    true
                }
            });
            let pruned: u64 = dropped.iter().map(|c: &WalChunkRef| c.entries as u64).sum();
            guard.pruned_entries += pruned;
            (guard.clone(), dropped)
        };

        if dropped.is_empty() {
            return;
        }
        if self.put_manifest(&manifest).await.is_err() {
            // Leave the objects alone: a manifest that still lists them is consistent, just
            // larger than it needs to be.
            warn!(
                "WAL manifest update failed; leaving {} chunks in place",
                dropped.len()
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
        self.create_table_inner(config, true).await
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

        let (vector_file, vector_rows) = self.open_vector_file(&table, log)?;
        let records = self.open_record_store(&table, log)?;
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
            // A table being created has no records, so an empty pair is already correct and
            // there is nothing to rebuild. `restore_table` clears this back to `None`.
            derived: parking_lot::RwLock::new(Some(DerivedIndexes::default())),
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
    pub async fn insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>> {
        self.ensure_writable()?;
        self.ensure_memory_headroom()?;
        // Writing to an evicted table has to bring it back first, or the insert lands in a
        // fresh empty table and the snapshot's records vanish on the next snapshot.
        self.ensure_resident(tablename).await?;
        // 创建响应 channel
        let (response_tx, response_rx) = oneshot::channel();

        // 发送插入请求到后台处理器
        self.insert_tx
            .send(InsertRequest {
                table_name: tablename.to_string(),
                records: requests,
                response_tx,
            })
            .map_err(|e| crate::Error::Internal(format!("Failed to send insert request: {}", e)))?;

        // 等待后台处理器响应
        let inserted = response_rx.await.map_err(|e| {
            crate::Error::Internal(format!("Insert request was cancelled: {}", e))
        })??;

        // 先落内存再记日志，最后才返回给调用方。中间崩掉的话这次写入还没被确认，丢了也符合语义；
        // 反过来先记日志的话，处理器若因维度不符等原因失败，日志里就留下了一条永远重放不成功的
        // 幽灵记录。
        // Each entry clones the record, vector included -- a third 3 KB allocation per record
        // on this path, freed as soon as the entry is serialised. It cannot simply be moved:
        // `inserted` is this function's return value and has 157 call sites. Removing it
        // means serialising the WAL entry from a borrow (a `WalEntryRef` with the same serde
        // shape, so replay keeps decoding) -- see design/INSERT_PERFORMANCE.md §23.17 for
        // what that clone costs and what it would take.
        self.wal_append(
            inserted
                .iter()
                .map(|r| WalEntry::UpsertRecordV2 {
                    table: tablename.to_string(),
                    record: r.clone(),
                })
                .collect(),
        )
        .await?;

        // Deliberately no index re-evaluation here. A rebuild is O(n) over the whole table and
        // runs to completion before the caller's write returns, so triggering it from the write
        // path hands the entire cost to whichever request happened to cross a size threshold —
        // and `reevaluate_indexes` sweeps every table, not just this one. The sweep spawned by
        // `spawn_index_reevaluation` (DB_REINDEX_INTERVAL_SECS, default 900) already owns this
        // job, off the serving path.

        Ok(inserted)
    }

    /// The index algorithm a table is currently using, with the parameters it was built from.
    ///
    /// `Table::index_type` names the family; this carries the numbers — `m`, `ef_construction`,
    /// `ef_search`, partition counts — which is what anyone diagnosing a recall or latency
    /// question actually needs.
    pub async fn describe_index(&self, tablename: &str) -> Option<IndexAlgorithm> {
        let tables = self.read_tables("describe_index").await;
        tables
            .get(tablename)
            .map(|data| data.selected_algorithm.clone())
    }

    /// 搜索向量
    pub async fn search(&self, tablename: &str, query: SearchQuery) -> crate::Result<SearchResult> {
        self.ensure_resident(tablename).await?;
        let start_time = std::time::Instant::now();
        // 优先使用 limit，如果 limit 未设置或为默认值则使用 top_k
        // Python SDK 发送 "limit" 字段，Rust API 使用 "top_k"
        let result_limit = if query.limit > 0 && query.limit != 10 {
            query.limit
        } else {
            query.top_k
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
        let (
            index,
            metric,
            configured_ef,
            total_records,
            cardinality_and_sel,
            precomputed_filter_ids,
            mut exact_scored,
        ) = {
            let tables = self.read_tables("search").await;

            let table_data = tables
                .get(tablename)
                .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

            // Validate query vector dimension against the default space only when no named vector
            // space is requested (named spaces may have different dimensions).
            if query.vector_name.is_none() {
                let expected_dim = table_data.table.dimension;
                if !query.vector.is_empty() && query.vector.len() != expected_dim {
                    return Err(crate::Error::DimensionMismatch {
                        expected: expected_dim,
                        actual: query.vector.len(),
                    });
                }
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
            let index = named_index_opt.unwrap_or_else(|| table_data.index.clone());
            let metric = table_data.table.distance_metric;
            let total_records = table_data.records.len();

            // Intelligent PRE/POST-FILTERING / IN-GRAPH decision
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
            let mut precomputed_filter_ids = None;
            let mut exact_scored = None;
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
                if table_data.derived().payload.cannot_match_anything(filter) {
                    exact_scored = Some(Vec::new());
                } else if *selectivity < 0.05 {
                    let filter_ids = table_data.derived().payload.get_matching_ids(filter);
                    // Exact scoring over the candidate set, when the set is small enough that
                    // doing so beats walking the graph. `result_limit * 4` alone made that
                    // threshold depend on the caller's top_k rather than on the work involved:
                    // a top-10 query over 400 candidates fell through to a filtered traversal
                    // that measured 24 ms p50, where scoring 400 vectors takes microseconds and
                    // is exact rather than approximate. The absolute floor is what matters —
                    // a filtered walk visits on the order of `ef * m` nodes anyway.
                    if cardinality.exp < Self::in_graph_filter_max_ids()
                        && filter_ids.len() <= (result_limit * 4).max(Self::EXACT_FILTER_SCAN_MAX)
                    {
                        let mut scored: Vec<(RecordId, f32)> = filter_ids
                            .iter()
                            .filter_map(|id| {
                                table_data.records.get(id).map(|rec| {
                                    // `vector_for`, not `rec.vector`: with the vectors on
                                    // disk the record's own field is empty.
                                    let dist = metric
                                        .calculate(&query.vector, &table_data.vector_for(&rec))
                                        .unwrap_or(f32::MAX);
                                    (id.clone(), dist)
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
                        exact_scored = Some(scored);
                    }
                    precomputed_filter_ids = Some(filter_ids);
                }
            }

            (
                index,
                metric,
                // A named vector space has its own index; the table's algorithm describes the
                // default one, so only claim its `ef_search` when that is what is being searched.
                if query.vector_name.is_none() {
                    table_data.selected_algorithm.configured_ef_search()
                } else {
                    None
                },
                total_records,
                cardinality_and_sel,
                precomputed_filter_ids,
                exact_scored,
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
        let default_ef = Self::ef_for_rows(configured_ef.unwrap_or(100), total_records);
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
                if exact_scored.is_none()
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
                                Some(filter_ids),
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
                            None,
                            precomputed_filter_ids.as_ref(),
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
                        None,
                        None,
                    )
                    .await?;
                (r, false, result_limit)
            };

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
                // where the 16.6 s empty answer in `design/DISK_INDEX.md` 5.5.1 came from,
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
                    None,
                    precomputed_filter_ids.as_ref(),
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
                     {cap}-row limit. Narrow it with a filter, raise DB_EXACT_SCAN_MAX, or use \
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
        self.ensure_resident(tablename).await?;

        let cap = Self::knn_graph_max_nodes();
        if sample > cap {
            return Err(crate::Error::InvalidInput(format!(
                "a {sample}-node graph is over the {cap}-node limit; the computation is \
                 quadratic in the node count. Lower `sample`, or raise DB_KNN_GRAPH_MAX_NODES"
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
        std::env::var("DB_KNN_GRAPH_MAX_NODES")
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
        filter_ids: Option<&std::collections::HashSet<String>>,
        keep: Option<&std::collections::HashSet<String>>,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        // A `k` larger than the index cannot be answered by asking for more: past that point
        // every extra candidate is a slot the walk keeps open and a node the heap carries, and
        // there is nothing left to put in them. The post-filter escalation used to hand this
        // `k = total_records`, which the rerank factor below then multiplied by five.
        let rows = index.len().await;
        let k = if rows > 0 { k.min(rows) } else { k };
        let Some(candidates) = index.rerank_candidates(k).await else {
            return match filter_ids {
                Some(ids) => index.search_with_filter(query, k, ef, ids).await,
                None => index.search_with_ef_hint(query, k, ef).await,
            };
        };
        let fetch = if rows > 0 {
            candidates.max(k).min(rows)
        } else {
            candidates.max(k)
        };
        let mut approximate = match filter_ids {
            Some(ids) => {
                index
                    .search_with_filter(query, fetch, ef.max(fetch), ids)
                    .await?
            }
            None => {
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
                                    Some(r.payload.clone())
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

                Some(SearchHit {
                    id,
                    score: similarity_score,
                    payload,
                    vector,
                })
            })
            .take(result_limit) // 过滤后只保留指定数量的结果
            .collect();

        Ok(hits)
    }

    /// 全文搜索
    #[cfg(feature = "text-bm25")]
    pub async fn full_text_search(
        &self,
        tablename: &str,
        request: FullTextSearchRequest,
    ) -> crate::Result<FullTextSearchResult> {
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
            // Cosine distance = 1 - similarity.
            DistanceMetric::Cosine => 1.0 - distance.clamp(0.0, 1.0),
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

    /// 混合搜索（稠密向量 + 稀疏向量 + 文本，三路 DBSF 融合）
    pub async fn hybrid_search(
        &self,
        tablename: &str,
        request: HybridSearchRequest,
    ) -> crate::Result<SearchResult> {
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
                    if let Ok(results) = engine.search(&text_request).await {
                        text_results = results.into_iter().map(|r| (r.id, r.score)).collect();
                    }
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
        let combined_scores = match request.fusion_method {
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
            if let Ok(meta) = crate::search::index_persistence::read_index::<TableSnapshot>(
                &dir.join(format!("t{}.meta", i)),
                "table-snapshot",
            ) {
                out.push(meta.table);
            }
        }

        Ok(out)
    }

    /// 修改表元数据，并把变更写入 WAL。
    pub async fn alter_table(&self, name: &str, req: TableAlterRequest) -> crate::Result<Table> {
        self.ensure_writable()?;
        self.ensure_resident(name).await?;

        let updated = {
            let mut tables = self.write_tables("alter_table").await;
            let data = tables
                .get_mut(name)
                .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))?;
            let updated = data.table.apply_alter(&req);
            data.table = updated.clone();
            updated
        };

        self.wal_append(vec![WalEntry::AlterTable {
            table: name.to_string(),
            request: req,
        }])
        .await?;

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
        let table_id = {
            let mut tables = self.write_tables("delete_table").await;
            let data = tables
                .remove(name)
                .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))?;
            // The row file goes with the table. Best effort: a file that will not delete is
            // orphaned, not a failed drop.
            if let Some(wal_dir) = &self.wal_dir {
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
                payload_schema: src.table.payload_schema.clone(),
                schema_version: src.table.schema_version,
                named_vectors: None,
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
        let tables = self.read_tables("list_all_record_ids").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        Ok(table_data.records.ids())
    }

    /// 获取向量索引中的记录数（用于一致性检查）
    pub async fn get_index_record_count(&self, tablename: &str) -> crate::Result<usize> {
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
            "dimension": dimension,
            "index_type": data.table.index_type.to_string(),
            "index_size_bytes": index_stats.index_size_bytes,
            "memory_usage_bytes": record_bytes + vector_row_bytes + index_stats.index_size_bytes,
            "disk_usage_bytes": disk_bytes,
        }))
    }

    /// 获取表的性能统计
    pub async fn get_performance_stats(&self, tablename: &str) -> crate::Result<PerformanceStats> {
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
    /// A fixed `ef` does not hold recall as the graph grows: measured on Cohere 768d
    /// (`design/DISK_INDEX.md` 5.5, `probe_10m_streambuild.log`), one graph reaches
    /// recall@100 0.9800 at 1M with ef 500, but only 0.9638 at 10M with the same budget --
    /// it needs 1024 there for 0.9800 and 2048 for 0.9887.
    ///
    /// Segmented engines get this growth without naming it. Qdrant splits 10M into 8 segments
    /// and walks each at ef 256, a 2,048-candidate budget in total, which is why its recall
    /// held at 10M (0.9828) where a single graph at 500 fell to 0.9626 -- and why it paid 594
    /// QPS for it against our 4,698. At a matched total budget the single graph is the better
    /// of the two (0.9887 at ef 2048 against its 0.9828), so the fix is to spend the budget,
    /// not to shard.
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

    /// `DB_HNSW_EF_CONSTRUCTION` overrides the build effort for tables created from here on.
    /// It exists to measure the trade: on Cohere 1M the neighbour search at 200 is 88% of the
    /// insert time (design doc section 24.4.3), and the only way to know what 100 costs in
    /// recall is to build at 100. Rebuilds do not read it; skip the rebuild when comparing.
    fn default_ef_construction() -> usize {
        std::env::var("DB_HNSW_EF_CONSTRUCTION")
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
    /// A million 768-dimension rows is 3 GB off disk and a few seconds of arithmetic. That is
    /// a reasonable thing to ask for deliberately and an unreasonable thing to trigger from a
    /// button, which is why the number is here rather than unbounded: at 100M rows the same
    /// request is minutes of full-file reads, and the caller has no way to tell the difference
    /// from a hang. `DB_EXACT_SCAN_MAX` raises it for anyone who means it.
    const EXACT_SCAN_MAX_DEFAULT: usize = 1_000_000;

    fn exact_scan_max() -> usize {
        std::env::var("DB_EXACT_SCAN_MAX")
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
    /// `DB_IN_GRAPH_FILTER_MAX_IDS` overrides it, which is how both paths get exercised from
    /// one test.
    const IN_GRAPH_FILTER_MAX_IDS: usize = 10_000;

    fn in_graph_filter_max_ids() -> usize {
        std::env::var("DB_IN_GRAPH_FILTER_MAX_IDS")
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
    /// place from there (design/DISK_INDEX.md section 4), so choosing it up front gives a
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
            // 4 dimensions per subspace (192 bytes at 768): the code size that kept
            // recall@100 at 1M and 10M (design/DISK_INDEX.md section 2.7). `HnswPqIndex::new`
            // rounds it to a divisor of the dimension.
            num_subvectors: config.dimension / 4,
            num_centroids: 256,
            enable_reranking: true,
        };
        // Measured, not estimated: 1M x 768 Cohere, records and vectors on disk, HNSW-PQ192
        // with a 500-candidate rerank (design/DISK_INDEX.md section 4, phase 3 / fp16).
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
            expected_performance: ExpectedPerformance {
                recall: 0.99,
                latency_p95_ms: 1.0,
                throughput_qps: 3000.0,
                memory_usage_mb: config.dataset_size * bytes_per_row / 1024 / 1024,
            },
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
                // BruteForce索引, with optional GPU acceleration
                #[allow(unused_mut)]
                let mut index = BruteForceIndex::new(distance_metric);
                #[cfg(feature = "gpu")]
                {
                    if let Some(ref gpu) = self.gpu_calculator {
                        index.set_gpu_calculator(gpu.clone());
                        debug!("GPU acceleration injected into BruteForce index");
                    }
                }
                Ok(Box::new(index))
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
                    // Reranking is not optional any more: the graph holds codes, not
                    // vectors, so the storage layer always reranks (`search_reranked`).
                    ..HnswPqConfig::default()
                };

                Ok(Box::new(HnswPqIndex::new(
                    dimension,
                    distance_metric,
                    Some(config),
                )?))
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
            #[cfg(feature = "gpu")]
            {
                crate::acceleration::gpu::is_cuda_available()
            }
            #[cfg(not(feature = "gpu"))]
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
    /// `vector: None` keeps the stored dense vector, and the sparse and named vectors are kept
    /// either way -- the update shape exposes only the dense one, and silently dropping the
    /// postings that hybrid and named-vector search rely on is not something a payload edit
    /// should do.
    pub async fn apply_update(
        &self,
        tablename: &str,
        record_id: &RecordId,
        vector: Option<DenseVector>,
        payload: Option<serde_json::Value>,
        merge: bool,
        upsert: bool,
    ) -> crate::Result<Record> {
        let existing = self.get_record(tablename, record_id).await.ok();
        if !upsert && existing.is_none() {
            return Err(crate::Error::RecordNotFound(record_id.to_string()));
        }

        let final_vector = match vector {
            Some(v) => v,
            None => existing
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
            _ => payload,
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
        self.ensure_writable()?;
        self.ensure_memory_headroom()?;
        self.ensure_resident(tablename).await?;

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
            let stored_vector = match &table_data.vector_file {
                Some(file) if request.vector.len() == expected_dim => {
                    // Appended here under the map lock -- updates are rare and single.
                    let row = file.append(&[(record_id.as_str(), request.vector.as_slice())])?[0];
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

            if let Some(derived) = table_data.derived_mut().as_mut() {
                if let Some(old_payload_val) = old_payload.as_ref() {
                    derived
                        .payload
                        .update_record(record_id, old_payload_val, &record.payload);
                } else {
                    derived.payload.insert_record(record_id, &record.payload);
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
                #[cfg(feature = "text-bm25")]
                full_text_engine: table_data.full_text_engine.clone(),
            }
        };

        // Dense and named-vector index work can block, so it runs after the global table-map
        // lock is released. Search may briefly miss this record while the index catches up, the
        // same visibility model inserts already have.
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

        self.wal_append(vec![WalEntry::UpsertRecordV2 {
            table: tablename.to_string(),
            record: artifacts.record.clone(),
        }])
        .await?;

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
                table_data.vector_rows.remove(id);

                if let Some(derived) = table_data.derived_mut().as_mut() {
                    if let Some(payload) = old_payload {
                        derived.payload.remove_record(id, &payload);
                    }
                    if let Err(e) = derived.sparse.delete(id) {
                        warn!("Failed to remove record {} from sparse index: {}", id, e);
                    }
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
        let ready = (!wal_enabled || wal_directory_accessible) && !durability_degraded;

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
            durability_degraded,
            durability_error,
            resident_table_count,
            table_count,
            total_records,
            last_wal_sync_at,
            last_snapshot_at,
            snapshot_failures: self.snapshot_failures.load(Ordering::Relaxed),
            last_snapshot_error,
        }
    }

    // A comment here used to say eight redundant insert stubs had been removed. They had not:
    // `insert_records_lockfree`, `_streaming`, `_with_memory_pool`, `_deferred_index`,
    // `_ultra_fast`, `_vm_optimized`, `_nvme_optimized` and `_ultimate` were all still present,
    // all seven lines long, and all reduced to `insert_records`. They are gone now, with the
    // `mode` field on `/records/bulk` that was the only thing selecting between them.

    /// Add a single record to a table (for simple mode API)
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
    /// `[1.0, 0.0, ...]` or `[]` depending on `DB_VECTOR_STORE`, with no way for the caller to
    /// tell that from a record that genuinely has no vector.
    pub async fn list_records(
        &self,
        tablename: &str,
        filter: Option<&crate::types::Filter>,
        limit: usize,
        offset: usize,
        with_vector: bool,
    ) -> crate::Result<(Vec<Record>, usize)> {
        self.ensure_resident(tablename).await?;
        let tables = self.read_tables("list_records").await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

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
            Some(filter) => {
                let matching_ids = table_data.derived().payload.get_matching_ids(filter);
                let total = matching_ids.len();
                if let Some(page) = table_data.records.page_of_ids(&matching_ids, offset, limit) {
                    let mut page: Vec<Record> = page.iter().map(|r| (**r).clone()).collect();
                    Self::fill_vectors(table_data, &mut page, with_vector);
                    return Ok((page, total));
                }
            }
        }

        // `Arc<Record>`, not `Record`. This used to deep-clone every matching record -- payload
        // `serde_json::Value` and all -- to return a page of twenty-five, so a table of 120,000
        // paid 120,000 JSON clones for 25 rows. A refcount bump costs nothing and the page is
        // cloned at the end, where it is twenty-five of them.
        let mut matching: Vec<std::sync::Arc<Record>> = if let Some(filter) = filter {
            let matching_ids = table_data.derived().payload.get_matching_ids(filter);
            matching_ids
                .iter()
                .filter_map(|id| table_data.records.get(id))
                .collect()
        } else {
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
    /// The old default was `{DB_STORAGE_PATH:-/tmp}/dxmt_full_text_{id}`, which put a real
    /// index — data, not a cache; the only copy of the term postings — in the one directory an
    /// operating system feels free to empty. `systemd-tmpfiles` clears `/tmp` on a schedule on
    /// most distributions and many images wipe it on boot, so a server could come back with its
    /// WAL, its row files and its snapshots all intact and its text search silently answering
    /// nothing. It also sat outside `DB_WAL_DIR`, so a backup of that directory — which is what
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
    /// Under `DB_WAL_DIR` they are as separate as the rest of their state. Reverting this
    /// function to the old path turns five of the six tests in
    /// `tests/full_text_recovery_test.rs` red, and that is how.
    ///
    /// `DB_STORAGE_PATH` still wins when it is set, in its original shape, so a deployment that
    /// pointed it somewhere deliberate keeps that placement and finds its index where it left
    /// it. Without a WAL directory at all — embedded use, most tests — `/tmp` remains, because
    /// nothing in that configuration survives the process anyway.
    ///
    /// Moving is safe without a migration step: a table whose index is not where this now looks
    /// comes back with an empty one, and `restore_table` rebuilds it from the rows. That costs
    /// one re-index on the first start after upgrading and nothing afterwards.
    #[cfg(feature = "text-bm25")]
    fn full_text_index_path(&self, table_id: &uuid::Uuid) -> PathBuf {
        if let Ok(explicit) = std::env::var("DB_STORAGE_PATH") {
            if !explicit.is_empty() {
                return PathBuf::from(explicit).join(format!("dxmt_full_text_{table_id}"));
            }
        }
        match &self.wal_dir {
            Some(wal_dir) => wal_dir.join("fulltext").join(table_id.to_string()),
            None => PathBuf::from("/tmp").join(format!("dxmt_full_text_{table_id}")),
        }
    }

    /// 创建全文搜索引擎
    #[cfg(feature = "text-bm25")]
    async fn create_full_text_engine(&self, table: &Table) -> crate::Result<FullTextSearchEngine> {
        let index_path = self.full_text_index_path(&table.id);
        let index_path = index_path.to_string_lossy().into_owned();

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

        let config = FullTextConfig {
            index_path,
            index_memory_mb: 256,
            num_threads: 4,
            enable_phrase_queries: true,
            enable_fuzzy_search: true,
            default_search_fields: default_fields,
            field_boosts: std::collections::HashMap::new(),
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

        // Re-create the table with the restored schema, then bulk-insert records
        // which rebuilds the vector index automatically.
        let table_cfg = TableConfig {
            name: table_name.to_string(),
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
    /// (`design/DISK_INDEX.md` 5.5). A table below the base row count keeps what it was
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
            r#type: "memory".to_string(),
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
            })
            .collect();
        storage.insert_records("stale", records).await?;

        // Settle the table onto the graph index the selector wants for its size.
        storage.reevaluate_indexes().await?;

        // What the selector answers today. Asked directly rather than read off a second table
        // built for the purpose: the assertion is about the selector's answer, and a second
        // 6,000-row graph build is a minute of the unit suite for nothing.
        let wanted = storage
            .decide_algorithm(ROWS, 16, DistanceMetric::Cosine, false)
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
            r#type: "memory".to_string(),
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
        };

        let table = storage.create_table(table_config).await?;
        assert_eq!(table.name, "test_table");
        assert_eq!(table.dimension, 128);
        Ok(())
    }

    #[tokio::test]
    async fn test_intelligent_index_selection() -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            r#type: "memory".to_string(),
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
            r#type: "memory".to_string(),
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
        };
        storage.create_table(table_config).await?;

        let req = CreateRecordRequest {
            id: "r1".into(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
            sparse_vector: None,
            payload: Some(serde_json::json!({"tag": "original"})),
            named_vectors: None,
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
    /// `DB_INDEX_ALGORITHM=hnsw-pq` was pinned; the server log still carries the sweep's
    /// "Selecting Enhanced DiskANN: dataset_size=21115700". The sweep now asks with the table's
    /// own setting, so a memory-optimized table stays HNSW-PQ at every size.
    #[tokio::test]
    async fn the_sweep_leaves_a_default_table_on_hnsw_pq_however_large_it_grows(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = StorageConfig {
            r#type: "memory".to_string(),
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
                .decide_algorithm(rows, 768, DistanceMetric::Cosine, true)
                .await?
                .selected_algorithm;
            assert!(
                matches!(want, IndexAlgorithm::HnswPQ { .. }),
                "a memory-optimized table at {rows} rows would be rebuilt as {want:?}"
            );
        }

        // And the size-based selector still answers what it always did, so the assertion above
        // is about the routing rather than about a selector that stopped choosing DiskANN.
        let sized = storage
            .decide_algorithm(25_000_000, 768, DistanceMetric::Cosine, false)
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
            r#type: "memory".to_string(),
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
            r#type: "memory".to_string(),
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
            r#type: "memory".to_string(),
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
            })
            .await?;
        let record = |id: String, vector: Vec<f32>| CreateRecordRequest {
            id,
            vector,
            sparse_vector: None,
            payload: None,
            named_vectors: None,
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
    ) -> (InsertRequest, oneshot::Receiver<crate::Result<Vec<Record>>>) {
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
                    })
                    .collect(),
                response_tx,
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
            r#type: "memory".to_string(),
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
            })
            .await?;

        let (first, first_rx) = queued("mixed", &["a", "b"], 4);
        // Wrong width, and not in first position: the records staged ahead of it inside the
        // group are the ones a partial failure would corrupt.
        let (bad, bad_rx) = queued("mixed", &["c"], 3);
        let (last, last_rx) = queued("mixed", &["d"], 4);

        process_insert_group(vec![first, bad, last], &storage.tables).await;

        assert_eq!(first_rx.await?.expect("first request succeeded").len(), 2);
        assert_eq!(last_rx.await?.expect("last request succeeded").len(), 1);
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
    /// The record routes validate vectors in the handler, but DXQL, MCP and simple mode reach
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
            r#type: "memory".to_string(),
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
            })
            .await?;
        let request = |id: &str, vector: Vec<f32>| CreateRecordRequest {
            id: id.to_string(),
            vector,
            sparse_vector: None,
            named_vectors: None,
            payload: None,
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
            r#type: "memory".to_string(),
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
            })
            .await?;

        let (one, one_rx) = queued("split", &["a", "b"], 4);
        let (two, two_rx) = queued("split", &["c"], 4);
        let (three, three_rx) = queued("split", &["d", "e", "f"], 4);

        process_insert_group(vec![one, two, three], &storage.tables).await;

        let ids = |rows: Vec<Record>| rows.into_iter().map(|r| r.id).collect::<Vec<_>>();
        assert_eq!(ids(one_rx.await??), vec!["a", "b"]);
        assert_eq!(ids(two_rx.await??), vec!["c"]);
        assert_eq!(ids(three_rx.await??), vec!["d", "e", "f"]);
        Ok(())
    }
}
