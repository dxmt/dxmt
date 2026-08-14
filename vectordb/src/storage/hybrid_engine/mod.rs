use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use crate::types::{RecordId, DenseVector};
use crate::storage::lmdb::{LmdbStorage, VectorRecord};

/// A raw key/value pair as returned by a prefix scan.
type KeyValue = (Vec<u8>, Vec<u8>);

/// 混合存储引擎：结合 LMDB 的读性能和 RocksDB 的写性能
///
/// 架构设计：
/// 1. 写入路径：RocksDB (WAL) -> MemTable -> LMDB
/// 2. 读取路径：LMDB (主) -> RocksDB (备)
/// 3. 后台合并：异步将 RocksDB 数据迁移到 LMDB
pub struct HybridStorageEngine {
    /// LMDB - 读优化存储（主存储）
    lmdb: Arc<LmdbStorage>,

    /// RocksDB - 写优化存储（写缓冲）
    rocksdb: Arc<RocksDbStorage>,

    /// 写缓冲区
    write_buffer: Arc<RwLock<WriteBuffer>>,

    /// 配置
    config: HybridConfig,

    /// 统计信息
    stats: Arc<RwLock<HybridStats>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridConfig {
    /// LMDB 配置
    pub lmdb_path: String,
    pub lmdb_map_size: usize,

    /// RocksDB 配置
    pub rocksdb_path: String,
    pub rocksdb_write_buffer_size: usize,
    pub rocksdb_max_write_buffers: u32,

    /// 混合策略配置
    pub write_buffer_threshold: usize,  // 触发刷新的缓冲区大小
    pub compaction_interval: Duration,  // 压缩间隔
    pub migration_batch_size: usize,    // 批量迁移大小
    pub read_preference: ReadPreference, // 读取偏好
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReadPreference {
    LmdbFirst,      // 优先从 LMDB 读取
    RocksDbFirst,   // 优先从 RocksDB 读取
    Parallel,       // 并行读取，返回最快的
}

/// RocksDB 存储实现
#[cfg(feature = "storage-rocksdb")]
pub struct RocksDbStorage {
    db: rocksdb::DB,
    write_options: rocksdb::WriteOptions,
    read_options: rocksdb::ReadOptions,
}

#[cfg(not(feature = "storage-rocksdb"))]
pub struct RocksDbStorage {
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(feature = "storage-rocksdb")]
impl RocksDbStorage {
    pub fn new(path: &str, config: &RocksDbConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.set_write_buffer_size(config.write_buffer_size);
        opts.set_max_write_buffer_number(config.max_write_buffers as i32);
        opts.set_target_file_size_base(config.target_file_size_base);
        opts.set_level_compaction_dynamic_level_bytes(true);

        // 优化写入性能
        opts.set_max_background_jobs(8);
        opts.set_max_background_compactions(4);
        opts.set_max_background_flushes(2);

        // 使用 Direct I/O 避免页缓存
        opts.set_use_direct_io_for_flush_and_compaction(true);

        let db = rocksdb::DB::open(&opts, path)?;
        let mut write_options = rocksdb::WriteOptions::default();
        write_options.set_sync(false); // 异步写入提高性能
        let read_options = rocksdb::ReadOptions::default();

        Ok(Self {
            db,
            write_options,
            read_options,
        })
    }

    pub fn compact_range(&self) {
        self.db.compact_range::<&[u8], &[u8]>(None, None);
    }
}

#[cfg(not(feature = "storage-rocksdb"))]
impl RocksDbStorage {
    pub fn new(_path: &str, _config: &RocksDbConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn write_batch(&self, _batch: WriteBatch) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    pub fn get(&self, _key: &[u8]) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(None)
    }

    pub fn scan_prefix(&self, _prefix: &[u8], _limit: usize) -> Result<Vec<KeyValue>, Box<dyn std::error::Error>> {
        Ok(Vec::new())
    }

    pub fn compact_range(&self) {
        // No-op for non-rocksdb builds
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RocksDbConfig {
    pub write_buffer_size: usize,
    pub max_write_buffers: u32,
    pub target_file_size_base: u64,
}

impl Default for RocksDbConfig {
    fn default() -> Self {
        Self {
            write_buffer_size: 128 * 1024 * 1024,  // 128MB
            max_write_buffers: 4,
            target_file_size_base: 64 * 1024 * 1024, // 64MB
        }
    }
}

/// 写缓冲区
pub struct WriteBuffer {
    operations: Vec<WriteOp>,
    size_bytes: usize,
    last_flush: Instant,
}

impl Default for WriteBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl WriteBuffer {
    pub fn new() -> Self {
        Self {
            operations: Vec::with_capacity(1000),
            size_bytes: 0,
            last_flush: Instant::now(),
        }
    }

    pub fn add_operation(&mut self, op: WriteOp) {
        self.size_bytes += op.size_estimate();
        self.operations.push(op);
    }

    pub fn should_flush(&self, threshold: usize) -> bool {
        self.size_bytes >= threshold ||
        self.operations.len() >= 10000 ||
        self.last_flush.elapsed() > Duration::from_secs(5)
    }

    pub fn take_batch(&mut self) -> WriteBatch {
        let operations = std::mem::take(&mut self.operations);
        self.size_bytes = 0;
        self.last_flush = Instant::now();

        WriteBatch { operations }
    }
}

#[derive(Clone, Debug)]
pub enum WriteOp {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
}

impl WriteOp {
    fn size_estimate(&self) -> usize {
        match self {
            WriteOp::Put { key, value } => key.len() + value.len(),
            WriteOp::Delete { key } => key.len(),
        }
    }
}

#[derive(Clone)]
pub struct WriteBatch {
    operations: Vec<WriteOp>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct HybridStats {
    pub total_reads: u64,
    pub lmdb_hits: u64,
    pub rocksdb_hits: u64,
    pub total_writes: u64,
    pub buffer_flushes: u64,
    pub compactions: u64,
    pub migrations: u64,
    pub avg_read_latency_us: f64,
    pub avg_write_latency_us: f64,
}

impl HybridStorageEngine {
    pub fn new(config: HybridConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let lmdb = Arc::new(LmdbStorage::new(&config.lmdb_path, config.lmdb_map_size)?);

        let rocksdb_config = RocksDbConfig {
            write_buffer_size: config.rocksdb_write_buffer_size,
            max_write_buffers: config.rocksdb_max_write_buffers,
            ..Default::default()
        };

        let rocksdb = Arc::new(RocksDbStorage::new(&config.rocksdb_path, &rocksdb_config)?);

        let engine = Self {
            lmdb,
            rocksdb,
            write_buffer: Arc::new(RwLock::new(WriteBuffer::new())),
            config,
            stats: Arc::new(RwLock::new(HybridStats::default())),
        };

        // 启动后台任务
        engine.start_background_tasks();

        Ok(engine)
    }

    /// 高性能写入接口
    pub async fn write(&self, id: RecordId, vector: DenseVector, payload: Option<serde_json::Value>)
        -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
        let start = Instant::now();

        let record = VectorRecord {
            id: id.clone(),
            vector: vector.to_vec(),
            payload,
        };

        let key = id.as_bytes();
        let value = bincode::serialize(&record)?;

        // 添加到写缓冲区
        {
            let mut buffer = self.write_buffer.write().await;
            buffer.add_operation(WriteOp::Put {
                key: key.to_vec(),
                value: value.clone()
            });

            // 检查是否需要刷新
            if buffer.should_flush(self.config.write_buffer_threshold) {
                let batch = buffer.take_batch();
                drop(buffer); // 释放锁

                // 异步写入 RocksDB
                self.flush_to_rocksdb(batch).await?;
            }
        }

        // 更新统计
        let mut stats = self.stats.write().await;
        stats.total_writes += 1;
        stats.avg_write_latency_us =
            (stats.avg_write_latency_us * (stats.total_writes - 1) as f64 +
             start.elapsed().as_micros() as f64) / stats.total_writes as f64;

        Ok(())
    }

    /// 高性能批量写入
    pub async fn write_batch(&self, records: Vec<VectorRecord>)
        -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
        let mut batch = WriteBatch { operations: Vec::with_capacity(records.len()) };

        for record in records {
            let key = record.id.as_bytes();
            let value = bincode::serialize(&record)?;
            batch.operations.push(WriteOp::Put {
                key: key.to_vec(),
                value
            });
        }

        // 直接写入 RocksDB，绕过缓冲区
        let batch_size = batch.operations.len();
        self.rocksdb.write_batch(batch)?;

        let mut stats = self.stats.write().await;
        stats.total_writes += batch_size as u64;

        Ok(())
    }

    /// 高性能读取接口
    pub async fn read(&self, id: &RecordId) -> Result<Option<VectorRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        let key = id.as_bytes();

        // First check write buffer for unflushed writes
        {
            let buffer = self.write_buffer.read().await;
            for op in buffer.operations.iter().rev() {  // Check in reverse for most recent
                if let WriteOp::Put { key: op_key, value } = op {
                    if op_key == key {
                        let record: VectorRecord = bincode::deserialize(value)?;
                        return Ok(Some(record));
                    }
                }
            }
        }

        let result = match self.config.read_preference {
            ReadPreference::LmdbFirst => {
                // 先尝试 LMDB
                if let Some(record) = self.lmdb.get_vector(id)? {
                    let mut stats = self.stats.write().await;
                    stats.lmdb_hits += 1;
                    Some(record)
                } else {
                    // 回退到 RocksDB
                    if let Some(value) = self.rocksdb.get(key)? {
                        let record: VectorRecord = bincode::deserialize(&value)?;
                        let mut stats = self.stats.write().await;
                        stats.rocksdb_hits += 1;
                        Some(record)
                    } else {
                        None
                    }
                }
            }
            ReadPreference::RocksDbFirst => {
                // Check RocksDB first, then LMDB (write buffer already checked above)
                // 先尝试 RocksDB
                if let Some(value) = self.rocksdb.get(key)? {
                    let record: VectorRecord = bincode::deserialize(&value)?;
                    let mut stats = self.stats.write().await;
                    stats.rocksdb_hits += 1;
                    Some(record)
                } else {
                    // 回退到 LMDB
                    if let Some(record) = self.lmdb.get_vector(id)? {
                        let mut stats = self.stats.write().await;
                        stats.lmdb_hits += 1;
                        Some(record)
                    } else {
                        None
                    }
                }
            }
            ReadPreference::Parallel => {
                // Write buffer already checked above, now parallel check LMDB and RocksDB
                // 并行读取
                let lmdb = self.lmdb.clone();
                let rocksdb = self.rocksdb.clone();
                let id_clone = id.clone();
                let key_clone = key.to_vec();

                let (lmdb_result, rocksdb_result) = tokio::join!(
                    tokio::task::spawn_blocking(move || lmdb.get_vector(&id_clone)),
                    tokio::task::spawn_blocking(move || rocksdb.get(&key_clone))
                );

                if let Ok(Ok(Some(record))) = lmdb_result {
                    let mut stats = self.stats.write().await;
                    stats.lmdb_hits += 1;
                    Some(record)
                } else if let Ok(Ok(Some(value))) = rocksdb_result {
                    let record: VectorRecord = bincode::deserialize(&value)?;
                    let mut stats = self.stats.write().await;
                    stats.rocksdb_hits += 1;
                    Some(record)
                } else {
                    None
                }
            }
        };

        // 更新统计
        let mut stats = self.stats.write().await;
        stats.total_reads += 1;
        stats.avg_read_latency_us =
            (stats.avg_read_latency_us * (stats.total_reads - 1) as f64 +
             start.elapsed().as_micros() as f64) / stats.total_reads as f64;

        Ok(result)
    }

    /// 刷新到 RocksDB
    async fn flush_to_rocksdb(&self, batch: WriteBatch) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let rocksdb = self.rocksdb.clone();

        tokio::task::spawn_blocking(move || {
            rocksdb.write_batch(batch)
        }).await??;

        let mut stats = self.stats.write().await;
        stats.buffer_flushes += 1;

        Ok(())
    }

    /// 启动后台任务
    fn start_background_tasks(&self) {
        // 启动压缩任务
        let engine = self.clone();
        tokio::spawn(async move {
            engine.compaction_task().await;
        });

        // 启动迁移任务
        let engine = self.clone();
        tokio::spawn(async move {
            engine.migration_task().await;
        });
    }

    /// 压缩任务：定期触发 RocksDB 压缩
    async fn compaction_task(&self) {
        let mut interval = tokio::time::interval(self.config.compaction_interval);

        loop {
            interval.tick().await;

            // 触发 RocksDB 压缩
            self.rocksdb.compact_range();

            let mut stats = self.stats.write().await;
            stats.compactions += 1;
        }
    }

    /// 迁移任务：将 RocksDB 中的冷数据迁移到 LMDB
    async fn migration_task(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));

        loop {
            interval.tick().await;

            // 扫描 RocksDB 中的旧数据
            let candidates = match self.rocksdb.scan_prefix(b"", self.config.migration_batch_size) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Migration scan error: {}", e);
                    continue;
                }
            };

            // 批量迁移到 LMDB
            for (_key, value) in candidates {
                if let Ok(record) = bincode::deserialize::<VectorRecord>(&value) {
                    if let Err(e) = self.lmdb.insert_vector(&record) {
                        eprintln!("Migration insert error: {}", e);
                    }
                }
            }

            let mut stats = self.stats.write().await;
            stats.migrations += 1;
        }
    }

    /// 获取统计信息
    pub async fn stats(&self) -> HybridStats {
        self.stats.read().await.clone()
    }
}

// 实现 Clone
impl Clone for HybridStorageEngine {
    fn clone(&self) -> Self {
        Self {
            lmdb: self.lmdb.clone(),
            rocksdb: self.rocksdb.clone(),
            write_buffer: self.write_buffer.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_hybrid_storage() {
        let dir = tempdir().unwrap();
        let lmdb_path = dir.path().join("lmdb");
        let rocksdb_path = dir.path().join("rocksdb");

        let config = HybridConfig {
            lmdb_path: lmdb_path.to_str().unwrap().to_string(),
            lmdb_map_size: 100 * 1024 * 1024,
            rocksdb_path: rocksdb_path.to_str().unwrap().to_string(),
            rocksdb_write_buffer_size: 16 * 1024 * 1024,
            rocksdb_max_write_buffers: 4,
            write_buffer_threshold: 1024 * 1024,
            compaction_interval: Duration::from_secs(300),
            migration_batch_size: 100,
            read_preference: ReadPreference::LmdbFirst,
        };

        let engine = HybridStorageEngine::new(config).unwrap();

        // 测试写入
        let id = uuid::Uuid::new_v4().to_string();
        let vector = vec![1.0, 2.0, 3.0];
        engine.write(id.clone(), vector.clone(), None).await.unwrap();

        // 测试读取
        let record = engine.read(&id).await.unwrap();
        assert!(record.is_some());
        assert_eq!(record.unwrap().vector, vector);

        // 测试批量写入
        let mut batch = Vec::new();
        for i in 0..100 {
            batch.push(VectorRecord {
                id: uuid::Uuid::new_v4().to_string(),
                vector: vec![i as f32; 128],
                payload: None,
            });
        }
        engine.write_batch(batch).await.unwrap();

        // 检查统计
        let stats = engine.stats().await;
        assert!(stats.total_writes > 0);
    }
}