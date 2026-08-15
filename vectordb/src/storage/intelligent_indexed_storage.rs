#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::search::brute_force::BruteForceIndex;
use crate::search::index_traits::VectorIndex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Mutex as StdMutex};
use tokio::sync::{mpsc, oneshot, RwLock};

use crate::wal::{WalEntry, WalManager};
use tracing::{debug, error, info, warn};

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

/// 单表快照。索引在同名 `.idx` 文件里；`has_index` 为 false 表示该索引类型没有磁盘格式
/// （例如 BruteForce），恢复时从 `records` 重新插入即可。
#[derive(serde::Serialize, serde::Deserialize)]
struct TableSnapshot {
    table: Table,
    algorithm: IndexAlgorithm,
    records: Vec<Record>,
    has_index: bool,
}

/// 一条待写入索引的记录：id、稠密向量，以及可选的命名向量。
///
/// 索引写入在 `tables` 写锁之外进行，所以要先把这些从表数据里摘出来。
type PendingIndexInsert = (RecordId, DenseVector, Option<HashMap<String, DenseVector>>);

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
    /// 而 Turbopuffer 式的分层是整个命名空间进出内存，不是单条向量。
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

    /// 回放期间置位，用于抑制 WAL 写入。回放走的是 `create_table` /
    /// `insert_records` 这些正常入口（否则索引不会被填充），所以必须挡住它们
    /// 把刚读出来的条目又写回日志。
    replaying: Arc<AtomicBool>,

    /// GPU distance calculator (None if no GPU available or feature not enabled)
    #[cfg(feature = "gpu")]
    gpu_calculator: Option<Arc<crate::acceleration::GpuDistanceCalculator>>,
}

/// 智能表数据
struct IntelligentTableData {
    /// 表元数据
    table: Table,

    /// 记录存储
    records: HashMap<RecordId, Record>,

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

    /// Payload索引（用于元数据过滤优化）
    payload_index: crate::storage::payload_index::PayloadIndex,

    /// 稀疏向量倒排索引（支持 WAND 剪枝的 MIPS）
    sparse_index: SparseIndex,

    /// 全文搜索引擎（仅当启用全文搜索时）
    #[cfg(feature = "text-bm25")]
    full_text_engine: Option<Arc<FullTextSearchEngine>>,
}

/// 性能统计
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

/// 后台插入处理器 - 单线程处理所有插入，消除锁竞争
async fn insertion_processor(
    mut rx: mpsc::UnboundedReceiver<InsertRequest>,
    tables: Arc<RwLock<HashMap<String, IntelligentTableData>>>,
) {
    info!("Insertion processor started");

    while let Some(req) = rx.recv().await {
        let start_time = std::time::Instant::now();

        // Two phases, and the split is the point. Phase one holds the tables write lock just
        // long enough to update the record map and the payload/sparse indexes — cheap, bounded
        // work. Phase two does the expensive part, inserting into the vector index, with the
        // lock released.
        //
        // Holding the lock across the whole batch is what this replaces: a 5,000-record request
        // blocked every reader, `GET /tables` included, until the last vector was in the graph.
        // Requests are still serialised against each other — this task processes one at a time —
        // so nothing about ordering or visibility changes. Records only become searchable once
        // the index insert lands, exactly as before, because search goes through the index.
        let result = async {
            let (index, named_indexes, table_meta, to_remove, to_insert, records) = {
                let mut tables = tables.write().await;

                let table_data = tables
                    .get_mut(&req.table_name)
                    .ok_or_else(|| crate::Error::TableNotFound(req.table_name.clone()))?;

                let expected_dim = table_data.table.dimension;
                let mut records = Vec::new();
                let mut to_remove: Vec<RecordId> = Vec::new();
                let mut to_insert: Vec<PendingIndexInsert> = Vec::new();
                let mut new_records_count = 0;

                for request in req.records {
                    let record_id = request.id.clone();

                    if !request.vector.is_empty() && request.vector.len() != expected_dim {
                        return Err(crate::Error::DimensionMismatch {
                            expected: expected_dim,
                            actual: request.vector.len(),
                        });
                    }

                    // UPSERT: an existing id keeps its created_at and bumps its version, and its
                    // old vector has to leave the index before the new one arrives.
                    let (is_update, existing_created_at, existing_version) =
                        if let Some(existing) = table_data.records.get(&record_id) {
                            (true, existing.created_at, existing.version + 1)
                        } else {
                            (false, chrono::Utc::now(), 1)
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

                    if let Some(sv) = &record.sparse_vector {
                        if let Err(e) = table_data.sparse_index.insert(record_id.clone(), sv) {
                            warn!("Failed to index sparse vector for {}: {}", record_id, e);
                        }
                    }

                    table_data
                        .payload_index
                        .insert_record(&record_id, &record.payload);
                    table_data.records.insert(record_id.clone(), record.clone());

                    to_insert.push((record_id, request.vector, request.named_vectors));
                    records.push(record);
                }

                table_data.table.record_count += new_records_count;
                table_data.table.updated_at = chrono::Utc::now();

                let insert_time = start_time.elapsed();
                let throughput = records.len() as f64 / insert_time.as_secs_f64();
                if let Ok(mut stats) = table_data.performance_stats.lock() {
                    stats.insert_throughput = (stats.insert_throughput * 0.9) + (throughput * 0.1);
                }

                (
                    table_data.index.clone(),
                    table_data.named_indexes.clone(),
                    table_data.table.clone(),
                    to_remove,
                    to_insert,
                    records,
                )
            };

            // ── lock released ──

            for id in &to_remove {
                if let Err(e) = index.remove(id).await {
                    warn!("Failed to remove old vector for {}: {}", id, e);
                }
            }

            for (id, vector, named) in to_insert {
                index.insert(id.clone(), vector).await?;

                if let (Some(ref collection), Some(named_vecs)) = (&named_indexes, named) {
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
                        .get(&req.table_name)
                        .and_then(|d| d.full_text_engine.clone())
                };
                if let Some(engine) = engine {
                    for id in &to_remove {
                        if let Err(e) = engine.delete_document(id).await {
                            warn!("Failed to remove old full-text document for {}: {}", id, e);
                        }
                    }
                    for record in &records {
                        if let Some(text_content) = extract_text_from_record(&table_meta, record) {
                            let doc = FullTextDocument {
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
                            };
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

            Ok(records)
        }
        .await;

        // 发送响应
        if req.response_tx.send(result).is_err() {
            error!("Failed to send insert response: insertion result was dropped");
        }
    }

    info!("Insertion processor stopped");
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

            // Snapshot first, then only the log written after it. Replay has to run with the
            // snapshot already in place: the tail may delete or overwrite records the snapshot
            // restored, and applying it to an empty store would lose those edits.
            let from_seq = storage.load_snapshot(&dir).await?;

            let mut pending = {
                let guard = storage
                    .wal
                    .as_ref()
                    .unwrap()
                    .lock()
                    .map_err(|_| crate::Error::Internal("WAL lock poisoned".to_string()))?;
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
            replaying: replaying_flag,
            #[cfg(feature = "gpu")]
            gpu_calculator,
        })
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
                WalEntry::Delete { table, record_id } => {
                    self.delete_records(&table, vec![record_id]).await
                }
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
        info!(
            "WAL replay finished: {} entries -> {} tables, {} records, {} skipped",
            total, tables, records, skipped
        );
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
            guard.next_sequence()
        };

        let dir = Self::snapshot_dir(self.wal_dir.as_ref().ok_or_else(|| {
            crate::Error::Internal("WAL enabled without a directory".to_string())
        })?);
        let staging = dir.with_extension("staging");
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging)?;

        let tables = self.tables.read().await;
        let mut written = 0usize;
        let mut names = Vec::new();

        for (name, data) in tables.iter() {
            // 表名来自用户输入，不能直接当文件名。用序号命名，真实名字放进元数据。
            let stem = format!("t{}", written);
            let idx_path = staging.join(format!("{}.idx", stem));
            let has_index = data.index.snapshot_to(&idx_path).await.unwrap_or(false);

            let meta = TableSnapshot {
                table: data.table.clone(),
                algorithm: data.selected_algorithm.clone(),
                records: data.records.values().cloned().collect(),
                has_index,
            };
            crate::search::index_persistence::write_index(
                &staging.join(format!("{}.meta", stem)),
                "table-snapshot",
                &meta,
            )?;

            names.push(name.clone());
            written += 1;
        }
        drop(tables);

        crate::search::index_persistence::write_index(
            &staging.join("manifest"),
            "snapshot-manifest",
            &SnapshotManifest {
                wal_next_sequence: seq,
                tables: written,
                names: names.clone(),
            },
        )?;

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
    async fn ensure_resident(&self, table_name: &str) -> crate::Result<()> {
        use crate::storage::tier_promotion::Tier;

        self.tier_policy.record_access(table_name);

        if self.tables.read().await.contains_key(table_name) {
            // Resident tables sit in Warm so the policy's demote_to_cold rule applies to them;
            // it only demotes out of Warm.
            self.tier_policy.set_current_tier(table_name, Tier::Warm);
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
        if self.tables.read().await.contains_key(table_name) {
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
    pub async fn backup_to(&self, dest: &std::path::Path) -> crate::Result<serde_json::Value> {
        let Some(ref wal_dir) = self.wal_dir else {
            return Err(crate::Error::InvalidInput(
                "backup requires WAL-backed storage; this instance has no durable state"
                    .to_string(),
            ));
        };

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
        self.tables.write().await.remove(name);
    }

    /// 把冷表从内存里放掉。返回被驱逐的表名。
    ///
    /// **先快照再驱逐**，而且只驱逐这次快照确实覆盖到的表。反过来会丢数据：一张表若有
    /// 快照之后的写入，驱逐就等于把那些写入扔了——WAL 里虽然还有，但驱逐后的表在内存里
    /// 不存在，回填只会读到快照那一份。
    pub async fn evict_cold_tables(&self) -> crate::Result<Vec<String>> {
        use crate::storage::tier_promotion::Tier;

        if self.wal.is_none() {
            return Ok(Vec::new());
        }

        let covered: Vec<String> = {
            let tables = self.tables.read().await;
            tables.keys().cloned().collect()
        };
        if covered.is_empty() {
            return Ok(Vec::new());
        }
        self.snapshot().await?;

        let mut evicted = Vec::new();
        for name in covered {
            // `should_demote_to_cold` is the right predicate here, not
            // `get_recommended_tier`: the latter calls anything below the *warm* threshold
            // Cold, and a table written to seconds ago scores about 0.31 — the frequency term
            // is normalised against 100 accesses, so a handful of touches barely registers and
            // recency contributes at most 0.3. Demotion instead compares against
            // demote_to_cold_threshold (0.2), which a Warm table only falls below after roughly
            // a third of the window with nothing touching it.
            if !self.tier_policy.should_demote_to_cold(&name) {
                continue;
            }
            let mut tables = self.tables.write().await;
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
            let meta: TableSnapshot = crate::search::index_persistence::read_index(
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
        self.create_table(config).await?;

        let restored_index = if meta.has_index {
            match &meta.algorithm {
                IndexAlgorithm::HNSW { .. } => crate::search::hnsw::HnswIndex::load(idx_path)
                    .map(|i| Box::new(i) as Box<dyn VectorIndex>)
                    .ok(),
                IndexAlgorithm::HnswPQ { .. } => {
                    crate::search::hnsw_pq::HnswPqIndex::load(idx_path)
                        .map(|i| Box::new(i) as Box<dyn VectorIndex>)
                        .ok()
                }
                // BruteForce and the variants that fall back to it have no on-disk form worth
                // loading; re-inserting is as cheap as reading the file would be.
                _ => None,
            }
        } else {
            None
        };

        let mut tables = self.tables.write().await;
        let data = tables
            .get_mut(&name)
            .ok_or_else(|| crate::Error::TableNotFound(name.clone()))?;

        let index_restored = restored_index.is_some();
        if let Some(index) = restored_index {
            data.index.replace(index).await;
        }

        for record in &meta.records {
            data.records.insert(record.id.clone(), record.clone());
        }
        data.table.record_count = data.records.len() as u64;

        let to_index: Vec<(RecordId, Vec<f32>)> = if index_restored {
            Vec::new()
        } else {
            meta.records
                .iter()
                .map(|r| (r.id.clone(), r.vector.clone()))
                .collect()
        };
        let wrapper = data.index.clone();
        drop(tables);

        // Only reached when the index had no on-disk form. Doing it outside the write lock
        // keeps a large table from blocking every reader for the duration.
        for (id, vector) in to_index {
            wrapper.insert(id, vector).await?;
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
        let Some(ref wal) = self.wal else {
            return Ok(());
        };

        let stamped: Vec<(u64, WalEntry)> = {
            let mut guard = wal
                .lock()
                .map_err(|_| crate::Error::Internal("WAL lock poisoned".to_string()))?;
            let mut stamped = Vec::with_capacity(entries.len());
            for entry in entries {
                let seq = guard.write_entry(entry.clone())?;
                stamped.push((seq, entry));
            }
            guard.force_flush()?;
            stamped
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

    /// 一段 WAL 分块在对象存储里的键。用定宽序号，好让字典序等于序号序。
    fn wal_chunk_key(first_seq: u64) -> String {
        format!("wal/{:020}.chunk", first_seq)
    }

    /// 把一批 WAL 条目作为不可变分块推到对象存储。
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

        self.storage
            .put_object(
                &Self::wal_chunk_key(first_seq),
                bytes::Bytes::from(bytes),
                Some("application/octet-stream"),
            )
            .await
            .map_err(|e| crate::Error::Internal(format!("uploading WAL chunk: {}", e)))
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
    async fn fetch_wal_chunks(&self, from_seq: u64) -> crate::Result<Vec<(u64, WalEntry)>> {
        use crate::storage::object::traits::ObjectStorage;

        let mut out = Vec::new();
        for key in self.list_wal_chunk_keys().await {
            let Ok(bytes) = self.storage.get_object(&key).await else {
                warn!("WAL chunk {} could not be read; skipping", key);
                continue;
            };
            let entries: Vec<(u64, WalEntry)> = match rmp_serde::from_slice(&bytes) {
                Ok(e) => e,
                Err(e) => {
                    warn!("WAL chunk {} is unreadable ({}); skipping", key, e);
                    continue;
                }
            };
            out.extend(entries.into_iter().filter(|(seq, _)| *seq >= from_seq));
        }
        Ok(out)
    }

    /// 删掉序号全部低于 `keep_from` 的 WAL 分块。
    async fn prune_wal_chunks(&self, keep_from: u64) {
        use crate::storage::object::traits::ObjectStorage;

        for key in self.list_wal_chunk_keys().await {
            // The key embeds the chunk's first sequence number. A chunk whose *last* entry is
            // still below the cutoff is fully covered by the snapshot; parse the first and only
            // prune when the next chunk also starts below the cutoff, which the sorted order
            // gives us for free — so compare conservatively against the first sequence.
            let Some(seq) = key
                .rsplit('/')
                .next()
                .and_then(|f| f.strip_suffix(".chunk"))
                .and_then(|s| s.parse::<u64>().ok())
            else {
                continue;
            };
            if seq < keep_from {
                if let Ok(bytes) = self.storage.get_object(&key).await {
                    if let Ok(entries) = rmp_serde::from_slice::<Vec<(u64, WalEntry)>>(&bytes) {
                        // Only drop the chunk when every entry in it is covered.
                        if entries.iter().all(|(s, _)| *s < keep_from) {
                            let _ = self.storage.delete_object(&key).await;
                        }
                    }
                }
            }
        }
    }

    /// 创建表时自动选择最优索引
    pub async fn create_table(&self, config: TableConfig) -> crate::Result<Table> {
        // Check if table exists (quickly release the lock)
        {
            let tables = self.tables.read().await;
            if tables.contains_key(&config.name) {
                return Err(crate::Error::TableAlreadyExists(config.name.clone()));
            }
        }

        // 创建表元数据
        let mut table = Table::new(config.clone())?;

        // ✨ 配置层级决策：表级配置覆盖全局配置
        let memory_optimization = config
            .memory_optimization
            .unwrap_or(self.default_memory_optimization);

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
        let index_decision = if memory_optimization {
            // 内存优化模式：优先选择压缩和聚类索引
            self.select_memory_optimized_index(&index_config).await?
        } else {
            // 高精度模式：使用标准索引选择逻辑
            self.index_selector
                .select_optimal_index(&index_config)
                .await?
        };

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
            match Self::create_full_text_engine(&table).await {
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

        let table_data = IntelligentTableData {
            table: table.clone(),
            records: HashMap::new(),
            selected_algorithm: index_decision.selected_algorithm.clone(),
            index,
            named_indexes,
            index_decision,
            performance_stats: Arc::new(Mutex::new(PerformanceStats::default())),
            payload_index: crate::storage::payload_index::PayloadIndex::new(),
            sparse_index: SparseIndex::new(),
            #[cfg(feature = "text-bm25")]
            full_text_engine,
        };

        // Re-acquire the lock to insert the table
        {
            let mut tables = self.tables.write().await;
            tables.insert(table.name.clone(), table_data);
        }

        self.wal_append(vec![WalEntry::CreateTableWithConfig { config }])
            .await?;

        Ok(table)
    }

    /// 插入记录 - Channel-based 无锁架构
    pub async fn insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>> {
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
        self.wal_append(
            inserted
                .iter()
                .map(|r| WalEntry::Insert {
                    table: tablename.to_string(),
                    record_id: r.id.clone(),
                    vector: r.vector.clone(),
                    payload: serde_json::to_value(&r.payload).ok(),
                })
                .collect(),
        )
        .await?;

        Ok(inserted)
    }

    /// 搜索向量
    pub async fn search(&self, tablename: &str, query: SearchQuery) -> crate::Result<SearchResult> {
        self.ensure_resident(tablename).await?;
        let start_time = std::time::Instant::now();
        let tables = self.tables.read().await;

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

        // 优先使用 limit，如果 limit 未设置或为默认值则使用 top_k
        // Python SDK 发送 "limit" 字段，Rust API 使用 "top_k"
        let result_limit = if query.limit > 0 && query.limit != 10 {
            query.limit
        } else {
            query.top_k
        };

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
        let active_index: &crate::search::IndexWrapper =
            named_index_opt.as_ref().unwrap_or(&table_data.index);

        // Intelligent PRE/POST-FILTERING / IN-GRAPH decision
        let cardinality_and_sel = if let Some(ref filter) = query.filter {
            let cardinality = table_data.payload_index.estimate_cardinality(filter);
            let total_records = table_data.records.len();
            let selectivity = if total_records > 0 {
                cardinality.exp as f64 / total_records as f64
            } else {
                1.0
            };
            Some((cardinality, selectivity))
        } else {
            None
        };

        let default_ef = 100usize;
        // Use adaptive EF when SearchMode::Auto and a filter is present
        let ef_hint = if let Some((_, selectivity)) = &cardinality_and_sel {
            query
                .search_mode
                .ef_search_adaptive(default_ef, result_limit, *selectivity)
        } else {
            query.search_mode.ef_search(default_ef, result_limit)
        };

        // Determine strategy and execute search
        let (results, post_filter) =
            if let (Some(ref filter), Some((ref cardinality, selectivity))) =
                (&query.filter, &cardinality_and_sel)
            {
                let total_records = table_data.records.len();

                if *selectivity < 0.05 && cardinality.exp < 10000 {
                    // Acorn-style in-graph filter: selectivity < 5%
                    let filter_ids = table_data.payload_index.get_matching_ids(filter);
                    info!(
                        "FILTER-IN-GRAPH: selectivity={:.2}%, candidates={}",
                        selectivity * 100.0,
                        filter_ids.len()
                    );

                    // When filter_ids is very small, kNN approximation is unreliable:
                    // fetching top-K from the full index may miss all filter candidates.
                    // Instead, compute exact distances only for the matching IDs (O(|filter_ids|)).
                    let r = if filter_ids.len() <= result_limit * 4 {
                        let metric = table_data.table.distance_metric;
                        let mut scored: Vec<(RecordId, f32)> = filter_ids
                            .iter()
                            .filter_map(|id| {
                                table_data.records.get(id).map(|rec| {
                                    let dist = metric
                                        .calculate(&query.vector, &rec.vector)
                                        .unwrap_or(f32::MAX);
                                    (id.clone(), dist)
                                })
                            })
                            .collect();
                        scored.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        scored.truncate(result_limit);
                        scored
                    } else {
                        active_index
                            .search_with_filter(&query.vector, result_limit, ef_hint, &filter_ids)
                            .await?
                    };
                    (r, false)
                } else if *selectivity < 0.3 {
                    // PRE-FILTERING: still fetch amplified and post-filter
                    let safety = 2.0;
                    let fetch = ((result_limit as f64 / selectivity * safety).ceil() as usize)
                        .min(total_records)
                        .max(result_limit * 2);
                    info!(
                        "PRE-FILTERING: selectivity={:.2}%, fetch={}",
                        selectivity * 100.0,
                        fetch
                    );
                    let r = active_index
                        .search_with_ef_hint(&query.vector, fetch, ef_hint)
                        .await?;
                    (r, true)
                } else {
                    // POST-FILTERING: selectivity >= 30%, fetch extra and filter
                    let fetch = result_limit * 3;
                    info!(
                        "POST-FILTERING: selectivity={:.2}%, fetch={}",
                        selectivity * 100.0,
                        fetch
                    );
                    let r = active_index
                        .search_with_ef_hint(&query.vector, fetch, ef_hint)
                        .await?;
                    (r, true)
                }
            } else {
                // No filter
                let r = active_index
                    .search_with_ef_hint(&query.vector, result_limit, ef_hint)
                    .await?;
                (r, false)
            };

        // 应用filter并构建搜索结果
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
                                    Some(r.vector.clone())
                                } else {
                                    None
                                },
                            )
                        })
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };

                // Convert distance to similarity score based on distance metric
                // Index returns distance (lower is better), but API should return similarity (higher is better)
                let similarity_score = match table_data.table.distance_metric {
                    DistanceMetric::Cosine => {
                        // Cosine distance = 1.0 - cosine_similarity
                        // Convert back: similarity = 1.0 - distance
                        1.0 - score.clamp(0.0, 1.0)
                    }
                    DistanceMetric::Dot => {
                        // Dot product distance = -dot_product
                        // Convert back: similarity = -distance
                        -score
                    }
                    DistanceMetric::Euclidean | DistanceMetric::Manhattan => {
                        // For distance metrics, smaller is better
                        // Convert to similarity: 1 / (1 + distance)
                        1.0 / (1.0 + score)
                    }
                };

                Some(SearchHit {
                    id,
                    score: similarity_score,
                    payload,
                    vector,
                })
            })
            .take(result_limit) // 过滤后只保留指定数量的结果
            .collect();

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

    /// 全文搜索
    #[cfg(feature = "text-bm25")]
    pub async fn full_text_search(
        &self,
        tablename: &str,
        request: FullTextSearchRequest,
    ) -> crate::Result<FullTextSearchResult> {
        let start_time = std::time::Instant::now();
        let tables = self.tables.read().await;

        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        // Check if full-text search is enabled
        if !table_data.table.enable_full_text_search {
            return Err(crate::Error::InvalidInput(
                "Full-text search is not enabled for this table".to_string(),
            ));
        }

        // Get the full-text engine
        let engine = table_data.full_text_engine.as_ref().ok_or_else(|| {
            crate::Error::Internal("Full-text search engine not initialized".to_string())
        })?;

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

        // Perform the search
        let search_results = engine.search(&engine_request).await?;

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
        let tables = self.tables.read().await;

        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let fetch_limit = if request.filter.is_some() {
            request.limit * 10
        } else {
            request.limit * 3
        };

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

        // 1. Dense vector search
        let mut vector_results: Vec<(RecordId, f32)> = Vec::new();
        if let Some(ref vector) = request.vector {
            vector_results = table_data.index.search(vector, fetch_limit).await?;
        }

        // 2. Sparse vector MIPS (WAND pruning)
        let mut sparse_results: Vec<(RecordId, f32)> = Vec::new();
        if let Some(ref sv) = request.sparse_vector {
            sparse_results = table_data.sparse_index.search(sv, fetch_limit);
        }

        // 3. Full-text search
        let mut text_results: Vec<(RecordId, f32)> = Vec::new();
        #[cfg(feature = "text-bm25")]
        if let Some(ref text_query) = request.text {
            if table_data.table.enable_full_text_search {
                if let Some(engine) = &table_data.full_text_engine {
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
                merged_vec
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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

        // 4. THEN sort by combined score
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
                                    Some(r.vector.clone())
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
        let tables = self.tables.read().await;

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
            let tables = self.tables.read().await;
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
        for (i, name) in manifest.names.iter().enumerate() {
            if resident.contains(name) {
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

    /// 删除表
    pub async fn delete_table(&self, name: &str) -> crate::Result<()> {
        {
            let mut tables = self.tables.write().await;
            tables
                .remove(name)
                .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))?;
        }

        self.wal_append(vec![WalEntry::DeleteTable {
            table: name.to_string(),
        }])
        .await?;

        Ok(())
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
            let mut tables = self.tables.write().await;

            let table_data = tables
                .get_mut(table_name)
                .ok_or_else(|| crate::Error::TableNotFound(table_name.to_string()))?;

            let distance_metric = table_data.table.distance_metric;
            let new_wrapper =
                crate::search::IndexWrapper::new(Box::new(BruteForceIndex::new(distance_metric)));

            let records_snapshot: Vec<(RecordId, Vec<f32>)> = table_data
                .records
                .iter()
                .map(|(id, r)| (id.clone(), r.vector.clone()))
                .collect();

            // Swap in the new (empty) brute-force index
            table_data.index = new_wrapper.clone();

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
        // Phase 1: snapshot + select + swap (write lock)
        let (new_wrapper, records_snapshot) = {
            let mut tables = self.tables.write().await;

            let table_data = tables
                .get_mut(table_name)
                .ok_or_else(|| crate::Error::TableNotFound(table_name.to_string()))?;

            let dimension = table_data.table.dimension;
            let distance_metric = table_data.table.distance_metric;
            let record_count = table_data.records.len();

            // Re-run adaptive selection for the current dataset size
            let config = IndexConfig {
                dataset_size: record_count,
                dimension,
                distance_metric,
                // These used to be hardcoded — 512 MB of budget, no AVX2, no SIMD — which
                // meant a rebuild could pick a different algorithm than table creation did on
                // the very same machine. Use what was actually detected, and the same
                // requirements the storage was constructed with.
                memory_budget_mb: (self.hardware_capabilities.available_memory_gb as usize) * 512,
                performance_requirements: self.performance_requirements.clone(),
                cost_constraints: self.cost_constraints.clone(),
                hardware_capabilities: self.hardware_capabilities.clone(),
            };

            let selector = IntelligentIndexSelector::new();
            let mut decision = selector.select_optimal_index(&config).await?;

            // Above this size, and with somewhere to put the segments, partition instead. This
            // is the only point where partitioning can be decided: it needs the vectors in hand
            // to cluster them, and at CREATE TABLE there are none. Below the threshold a
            // resident index is simply faster — fetching a segment costs a round trip that a
            // small table never needs to pay.
            let partition_threshold = std::env::var("DB_PARTITION_THRESHOLD")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(100_000);

            if self.object_storage_enabled && record_count >= partition_threshold {
                let nlist = ((record_count as f64).sqrt() as usize).clamp(16, 4096);
                decision.selected_algorithm = IndexAlgorithm::PartitionedIvf {
                    nlist,
                    nprobe: (nlist / 8).max(4),
                    max_resident_partitions: (nlist / 4).max(8),
                };
                info!(
                    "rebuild_index: {} records with object storage configured — partitioning \
                     into {} segments",
                    record_count, nlist
                );
            }

            let algorithm = decision.selected_algorithm.clone();
            let table_id = table_data.table.id.to_string();
            let new_index =
                self.create_index_instance_for(&algorithm, dimension, distance_metric, &table_id)?;
            let new_wrapper = crate::search::IndexWrapper::new(new_index);

            let records_snapshot: Vec<(RecordId, Vec<f32>)> = table_data
                .records
                .iter()
                .map(|(id, r)| (id.clone(), r.vector.clone()))
                .collect();

            // Swap in the empty new index; update table_data bookkeeping
            table_data.index = new_wrapper.clone();
            table_data.table.index_type = index_type_of(&algorithm);
            table_data.selected_algorithm = algorithm;
            table_data.index_decision = decision;

            info!(
                "rebuild_index: table='{}' records={} algorithm={:?}",
                table_name, record_count, table_data.selected_algorithm
            );

            (new_wrapper, records_snapshot)
        };

        // Phase 2: populate the new index outside the write lock. bulk_build rather than a
        // loop of inserts, because a partitioned index has to see every vector at once to
        // cluster it — inserting one at a time would leave the lot in its overflow buffer.
        new_wrapper.bulk_build(records_snapshot).await?;

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
            let tables = self.tables.read().await;
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
                .values()
                .map(|r| CreateRecordRequest {
                    id: r.id.clone(),
                    vector: r.vector.clone(),
                    sparse_vector: r.sparse_vector.clone(),
                    payload: Some(r.payload.clone()),
                    named_vectors: None,
                })
                .collect();

            (config, records)
        };

        // Create the branch table (reuses existing create_table + insert_records)
        let branch = self.create_table(source_config).await?;
        if !records_snapshot.is_empty() {
            self.insert_records(branch_name, records_snapshot).await?;
        }
        Ok(branch)
    }

    /// 列出表中所有记录 ID（用于一致性检查和数据迁移）
    pub async fn list_all_record_ids(&self, tablename: &str) -> crate::Result<Vec<RecordId>> {
        let tables = self.tables.read().await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        Ok(table_data.records.keys().cloned().collect())
    }

    /// 获取向量索引中的记录数（用于一致性检查）
    pub async fn get_index_record_count(&self, tablename: &str) -> crate::Result<usize> {
        let tables = self.tables.read().await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        Ok(table_data.index.len().await)
    }

    /// 获取表的性能统计
    pub async fn get_performance_stats(&self, tablename: &str) -> crate::Result<PerformanceStats> {
        let tables = self.tables.read().await;

        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let stats = table_data
            .performance_stats
            .lock()
            .map_err(|_| crate::Error::Internal("PerformanceStats lock poisoned".to_string()))?;
        Ok(stats.clone())
    }

    /// 选择内存优化索引（memory_optimization=true 时使用）
    async fn select_memory_optimized_index(
        &self,
        config: &IndexConfig,
    ) -> crate::Result<IndexDecision> {
        info!(
            "Selecting memory-optimized index for dataset size: {}",
            config.dataset_size
        );

        // 根据数据集大小选择内存优化索引
        let (selected_algorithm, recall, latency_p95_ms, throughput_qps, memory_usage_mb) =
            if config.dataset_size < 10_000 {
                // 小数据集 (<10K): 使用 HNSW（高精度，速度快）
                info!("Small dataset (<10K): HNSW");
                (
                    IndexAlgorithm::HNSW {
                        m: 16,
                        ef_construction: 200,
                        ef_search: 50,
                    },
                    0.98,   // 98% recall
                    5.0,    // 5ms latency
                    2000.0, // 2K QPS
                    (config.dimension * config.dataset_size * 4 * 120 / 100) / 1024 / 1024, // HNSW memory
                )
            } else if config.dataset_size < 100_000 {
                // 中数据集 (10K-100K): 使用 Hierarchical Clustering (Phase 3)
                info!(
                    "Medium dataset (10K-100K): HNSW (HierarchicalClustering is not implemented)"
                );
                (
                    IndexAlgorithm::HNSW {
                        // HierarchicalClustering not yet available; using HNSW
                        m: 16,
                        ef_construction: 200,
                        ef_search: 50,
                    },
                    0.95,   // 95% recall
                    2.0,    // 2ms latency
                    3000.0, // 3K QPS
                    (config.dimension * config.dataset_size * 4 * 105 / 100) / 1024 / 1024, // Clustering memory
                )
            } else {
                // 大数据集 (>100K): 使用 Hybrid RaBitQ + Clustering (Phase 4)
                info!("Large dataset (>100K): HNSW-PQ (HybridRaBitQClustering is not implemented)");
                (
                    IndexAlgorithm::HnswPQ {
                        // HybridRaBitQClustering not yet available; using HnswPQ
                        m: 16,
                        ef_construction: 200,
                        ef_search: 50,
                        num_subvectors: config.dimension / 8,
                        num_centroids: 256,
                        enable_reranking: true,
                    },
                    0.90,   // 90% recall (Phase 4 tradeoff)
                    1.0,    // 1ms latency
                    5000.0, // 5K QPS
                    (config.dimension * config.dataset_size * 4 * 10 / 100) / 1024 / 1024, // 90% memory savings
                )
            };

        Ok(IndexDecision {
            selected_algorithm,
            confidence: 0.95,
            expected_performance: ExpectedPerformance {
                recall,
                latency_p95_ms,
                throughput_qps,
                memory_usage_mb,
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

                let config = HnswPqConfig {
                    hnsw_config: HnswConfig {
                        m: *m,
                        max_m: *m,
                        max_m0: m * 2,
                        ef_construction: *ef_construction,
                        ef_search: *ef_search,
                        max_connections: *m,
                        level_multiplier: 1.0 / (2.0_f64.ln()),
                        max_level: 16,
                        ml: 1.0 / (2.0_f64.ln()),
                        seed: 42,
                    },
                    num_subvectors: *num_subvectors,
                    num_centroids: *num_centroids,
                    enable_reranking: *enable_reranking,
                    rerank_factor: 3,
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
        std::mem::discriminant(a) == std::mem::discriminant(b)
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
    pub async fn insert_records_batch(
        &self,
        tablename: &str,
        records: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<RecordId>> {
        // For now, just use regular insert
        let inserted = self.insert_records(tablename, records).await?;
        Ok(inserted.into_iter().map(|r| r.id).collect())
    }

    /// Deferred index insert
    pub async fn insert_records_deferred_index(
        &self,
        tablename: &str,
        records: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<RecordId>> {
        // For now, just use regular insert
        // In a full implementation, this would defer index building
        let inserted = self.insert_records(tablename, records).await?;
        Ok(inserted.into_iter().map(|r| r.id).collect())
    }

    /// Get a single record
    pub async fn get_record(&self, tablename: &str, record_id: &RecordId) -> crate::Result<Record> {
        self.ensure_resident(tablename).await?;
        let tables = self.tables.read().await;
        let table = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        table
            .records
            .get(record_id)
            .cloned()
            .ok_or_else(|| crate::Error::RecordNotFound(record_id.clone()))
    }

    /// Update a record
    pub async fn update_record(
        &self,
        tablename: &str,
        record_id: &RecordId,
        request: CreateRecordRequest,
    ) -> crate::Result<Record> {
        let mut tables = self.tables.write().await;
        let table_data = tables
            .get_mut(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        // Validate vector dimension
        let expected_dim = table_data.table.dimension;
        if !request.vector.is_empty() && request.vector.len() != expected_dim {
            return Err(crate::Error::DimensionMismatch {
                expected: expected_dim,
                actual: request.vector.len(),
            });
        }

        // Get existing record's created_at, version, and payload if it exists (preserves created_at on update)
        let (created_at, version, old_payload) = table_data
            .records
            .get(record_id)
            .map(|r| (r.created_at, r.version + 1, Some(r.payload.clone())))
            .unwrap_or_else(|| (chrono::Utc::now(), 1, None));

        let record = Record {
            id: record_id.clone(),
            vector: request.vector.clone(),
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

        // Update vector index: remove old entry then re-insert with new vector
        let _ = table_data.index.remove(record_id).await;
        if let Err(e) = table_data
            .index
            .insert(record_id.clone(), record.vector.clone())
            .await
        {
            warn!(
                "Failed to update vector index for record {}: {}",
                record_id, e
            );
        }

        // Update sparse index if present
        if let Some(sv) = &record.sparse_vector {
            let _ = table_data.sparse_index.delete(record_id);
            if let Err(e) = table_data.sparse_index.insert(record_id.clone(), sv) {
                warn!(
                    "Failed to update sparse index for record {}: {}",
                    record_id, e
                );
            }
        }

        // Update full-text index (if enabled)
        #[cfg(feature = "text-bm25")]
        if let Some(engine) = &table_data.full_text_engine {
            if let Some(text_content) = extract_text_from_record(&table_data.table, &record) {
                let doc = FullTextDocument {
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
                };

                // Update document in full-text engine
                if let Err(e) = engine.update_document(doc).await {
                    warn!(
                        "Failed to update text index for record {}: {}",
                        record.id, e
                    );
                }
            }
        }

        // Commit full-text index changes
        #[cfg(feature = "text-bm25")]
        if let Some(engine) = &table_data.full_text_engine {
            if let Err(e) = engine.commit().await {
                warn!("Failed to commit full-text index: {}", e);
            }
        }

        table_data.records.insert(record_id.clone(), record.clone());

        // Update PayloadIndex
        if let Some(old_payload_val) = old_payload.as_ref() {
            // Update existing record
            table_data
                .payload_index
                .update_record(record_id, old_payload_val, &record.payload);
        } else {
            // Insert new record (upsert case)
            table_data
                .payload_index
                .insert_record(record_id, &record.payload);
        }

        Ok(record)
    }

    /// Delete multiple records
    pub async fn delete_records(
        &self,
        tablename: &str,
        record_ids: Vec<RecordId>,
    ) -> crate::Result<()> {
        let mut tables = self.tables.write().await;
        let table_data = tables
            .get_mut(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        for id in &record_ids {
            // Remove from full-text index (if enabled)
            #[cfg(feature = "text-bm25")]
            if let Some(engine) = &table_data.full_text_engine {
                if let Err(e) = engine.delete_document(id).await {
                    warn!("Failed to remove record {} from text index: {}", id, e);
                }
            }

            // Get payload before removing (for payload_index update)
            let old_payload = table_data.records.get(id).map(|r| r.payload.clone());

            // Remove from records map
            table_data.records.remove(id);

            // Remove from PayloadIndex
            if let Some(payload) = old_payload {
                table_data.payload_index.remove_record(id, &payload);
            }

            // Remove from dense vector index
            if let Err(e) = table_data.index.remove(id).await {
                warn!("Failed to remove record {} from vector index: {}", id, e);
            } else {
                debug!("Successfully removed record {} from vector index", id);
            }

            // Remove from sparse index
            if let Err(e) = table_data.sparse_index.delete(id) {
                warn!("Failed to remove record {} from sparse index: {}", id, e);
            }
        }

        // Commit full-text index changes
        #[cfg(feature = "text-bm25")]
        if let Some(engine) = &table_data.full_text_engine {
            if let Err(e) = engine.commit().await {
                warn!("Failed to commit full-text index: {}", e);
            }
        }

        // Sync record_count from actual HashMap size (handles concurrent deletes correctly)
        table_data.table.record_count = table_data.records.len() as u64;
        table_data.table.updated_at = chrono::Utc::now();

        drop(tables);

        self.wal_append(
            record_ids
                .into_iter()
                .map(|id| WalEntry::Delete {
                    table: tablename.to_string(),
                    record_id: id,
                })
                .collect(),
        )
        .await?;

        Ok(())
    }

    /// Get statistics
    pub async fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        let tables = self.tables.read().await;

        stats.insert("table_count".to_string(), serde_json::json!(tables.len()));
        let total_records: usize = tables.values().map(|t| t.records.len()).sum();
        stats.insert(
            "total_records".to_string(),
            serde_json::json!(total_records),
        );

        stats
    }

    // Lock-free insert (stub)
    // Simplified insertion methods - removed 8 redundant stubs that all called the same function
    // All "optimized" modes (lockfree, streaming, memory_pool, ultra_fast, vm_optimized, nvme, ultimate)
    // were just calling insert_records_batch, so we keep only the essential ones:

    /// Optimized batch insert - for backwards compatibility
    pub async fn insert_records_lockfree(
        &self,
        tablename: &str,
        records: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<RecordId>> {
        // All "optimized" modes delegate to batch insert since they had no real implementation
        self.insert_records_batch(tablename, records).await
    }

    // Aliases for backwards compatibility - will be deprecated
    pub async fn insert_records_streaming(
        &self,
        t: &str,
        r: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<RecordId>> {
        self.insert_records_batch(t, r).await
    }
    pub async fn insert_records_with_memory_pool(
        &self,
        t: &str,
        r: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<RecordId>> {
        self.insert_records_batch(t, r).await
    }
    pub async fn insert_records_ultra_fast(
        &self,
        t: &str,
        r: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<RecordId>> {
        self.insert_records_batch(t, r).await
    }
    pub async fn insert_records_vm_optimized(
        &self,
        t: &str,
        r: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<RecordId>> {
        self.insert_records_batch(t, r).await
    }
    pub async fn insert_records_nvme_optimized(
        &self,
        t: &str,
        r: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<RecordId>> {
        self.insert_records_batch(t, r).await
    }
    pub async fn insert_records_ultimate(
        &self,
        t: &str,
        r: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<RecordId>> {
        self.insert_records_batch(t, r).await
    }

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
        let tables = self.tables.read().await;

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

        // 如果有过滤器，获取更多结果以补偿过滤后的损失
        let fetch_limit = if filter.is_some() {
            limit * 3 // 获取3倍结果
        } else {
            limit
        };

        // Use the index to search
        let results = table_data.index.search(query, fetch_limit).await?;

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

                // Convert distance to similarity score
                let similarity_score = match table_data.table.distance_metric {
                    DistanceMetric::Cosine => 1.0 - score.clamp(0.0, 1.0),
                    DistanceMetric::Dot => -score,
                    DistanceMetric::Euclidean | DistanceMetric::Manhattan => 1.0 / (1.0 + score),
                };

                records_with_scores.push((record.clone(), similarity_score));

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
    pub async fn list_records(
        &self,
        tablename: &str,
        filter: Option<&crate::types::Filter>,
        limit: usize,
        offset: usize,
    ) -> crate::Result<(Vec<Record>, usize)> {
        self.ensure_resident(tablename).await?;
        let tables = self.tables.read().await;
        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let mut matching_records: Vec<Record> = if let Some(filter) = filter {
            let matching_ids = table_data.payload_index.get_matching_ids(filter);
            matching_ids
                .iter()
                .filter_map(|id| table_data.records.get(id).cloned())
                .collect()
        } else {
            table_data.records.values().cloned().collect()
        };

        // Sort by created_at (newest first)
        matching_records.sort_by_key(|r| std::cmp::Reverse(r.created_at));

        let total_matching = matching_records.len();
        let start = offset.min(total_matching);
        let end = (offset + limit).min(total_matching);

        Ok((matching_records[start..end].to_vec(), total_matching))
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

    /// 创建全文搜索引擎
    #[cfg(feature = "text-bm25")]
    async fn create_full_text_engine(table: &Table) -> crate::Result<FullTextSearchEngine> {
        let base_dir = std::env::var("DB_STORAGE_PATH").unwrap_or_else(|_| "/tmp".to_string());
        let index_path = format!("{}/dxmt_full_text_{}", base_dir, table.id);

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
        self.create_table(config).await
    }

    async fn get_table(&self, name: &str) -> crate::Result<Table> {
        let tables = self.tables.read().await;
        tables
            .get(name)
            .map(|data| data.table.clone())
            .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))
    }

    async fn list_tables(&self) -> crate::Result<Vec<Table>> {
        let tables = self.tables.read().await;
        Ok(tables.values().map(|data| data.table.clone()).collect())
    }

    async fn delete_table(&self, name: &str) -> crate::Result<()> {
        let mut tables = self.tables.write().await;
        tables
            .remove(name)
            .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))?;
        Ok(())
    }

    async fn alter_table(&self, name: &str, req: TableAlterRequest) -> crate::Result<Table> {
        let mut tables = self.tables.write().await;
        let data = tables
            .get_mut(name)
            .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))?;
        let updated = data.table.apply_alter(&req);
        data.table = updated.clone();
        Ok(updated)
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
        let tables = self.tables.read().await;
        let table = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        table
            .records
            .get(record_id)
            .cloned()
            .ok_or_else(|| crate::Error::RecordNotFound(record_id.clone()))
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
        self.delete_records(tablename, record_ids).await
    }

    async fn list_all_record_ids(&self, tablename: &str) -> crate::Result<Vec<RecordId>> {
        self.list_all_record_ids(tablename).await
    }

    async fn backup(&self, table_name: &str, backup_path: &str) -> crate::Result<()> {
        #[derive(serde::Serialize)]
        struct Snapshot<'a> {
            table: &'a Table,
            records: Vec<&'a Record>,
        }
        let tables = self.tables.read().await;
        let data = tables
            .get(table_name)
            .ok_or_else(|| crate::Error::TableNotFound(table_name.to_string()))?;
        let snap = Snapshot {
            table: &data.table,
            records: data.records.values().collect(),
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
        storage.rebuild_index("rebuild_test").await?;

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
}
