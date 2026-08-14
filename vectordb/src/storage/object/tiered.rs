use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::stream::BoxStream;
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::errors::ObjectStorageError;
use super::filesystem::FilesystemStorage;
use super::s3::S3Storage;
use super::traits::{ObjectStorage, ObjectStorageResult, StorageStats};
use super::types::{ListObjectsOptions, ListObjectsResult, ObjectMetadata, PartInfo};

/// 数据访问层级
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StorageTier {
    /// 内存层 - 最快访问
    Memory,
    /// SSD层 - 中等速度
    Ssd,
    /// 对象存储层 - 最大容量但最慢
    ObjectStorage,
}

/// 三层存储配置
#[derive(Debug, Clone)]
pub struct TieredStorageConfig {
    /// 内存缓存大小（字节）
    pub memory_cache_size: usize,
    /// 内存缓存条目数量限制
    pub memory_cache_capacity: usize,
    /// SSD 缓存目录
    pub ssd_cache_dir: PathBuf,
    /// SSD 缓存大小限制（字节）
    pub ssd_cache_size: u64,
    /// 对象存储桶名称
    pub bucket_name: String,
    /// 数据访问统计采样间隔（秒）
    pub stats_interval_secs: u64,
    /// 自动分层开关
    pub auto_tiering_enabled: bool,
}

impl Default for TieredStorageConfig {
    fn default() -> Self {
        Self {
            memory_cache_size: 1024 * 1024 * 512, // 512MB
            memory_cache_capacity: 10000,
            ssd_cache_dir: PathBuf::from("./cache"),
            ssd_cache_size: 1024 * 1024 * 1024 * 10, // 10GB
            bucket_name: "vectordb".to_string(),
            stats_interval_secs: 300, // 5分钟
            auto_tiering_enabled: true,
        }
    }
}

/// 对象访问统计
#[derive(Debug, Clone)]
struct ObjectStats {
    /// 访问次数
    access_count: u64,
    /// 最后访问时间
    last_access: DateTime<Utc>,
    /// 对象大小
    size: u64,
    /// 当前所在层级
    current_tier: StorageTier,
}

/// 三层存储实现
pub struct TieredStorage {
    /// 配置
    config: TieredStorageConfig,
    /// 内存缓存
    memory_cache: Arc<RwLock<LruCache<String, Vec<u8>>>>,
    /// SSD 存储
    ssd_storage: FilesystemStorage,
    /// 对象存储
    object_storage: Box<dyn ObjectStorage + Send + Sync>,
    /// 对象访问统计
    stats: Arc<RwLock<HashMap<String, ObjectStats>>>,
    /// 当前内存使用量
    memory_usage: Arc<RwLock<usize>>,
    /// 内存层命中计数
    memory_hit_count: Arc<AtomicU64>,
    /// 内存层未命中计数
    memory_miss_count: Arc<AtomicU64>,
    /// SSD 层命中计数
    ssd_hit_count: Arc<AtomicU64>,
    /// SSD 层未命中计数
    ssd_miss_count: Arc<AtomicU64>,
    /// 对象存储访问计数
    object_storage_access_count: Arc<AtomicU64>,
}

impl std::fmt::Debug for TieredStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TieredStorage")
            .field("config", &self.config)
            .field("memory_cache", &"<Arc<RwLock<LruCache>>>")
            .field("ssd_storage", &self.ssd_storage)
            .field("object_storage", &"<Box<dyn ObjectStorage>>")
            .field("stats", &"<Arc<RwLock<HashMap>>>")
            .field("memory_usage", &self.memory_usage)
            .field("memory_hits", &self.memory_hit_count.load(Ordering::Relaxed))
            .field("ssd_hits", &self.ssd_hit_count.load(Ordering::Relaxed))
            .finish()
    }
}

impl TieredStorage {
    /// 创建新的三层存储实例
    pub fn new(
        config: TieredStorageConfig,
        object_storage: Box<dyn ObjectStorage + Send + Sync>) -> ObjectStorageResult<Self> {
        let memory_cache = Arc::new(RwLock::new(LruCache::new(
            NonZeroUsize::new(config.memory_cache_capacity).ok_or_else(|| ObjectStorageError::ConfigurationError { message: "Invalid memory cache capacity".to_string() })?)));

        let ssd_storage = FilesystemStorage::new(&config.ssd_cache_dir, &config.bucket_name)?;

        Ok(Self {
            config,
            memory_cache,
            ssd_storage,
            object_storage,
            stats: Arc::new(RwLock::new(HashMap::new())),
            memory_usage: Arc::new(RwLock::new(0)),
            memory_hit_count: Arc::new(AtomicU64::new(0)),
            memory_miss_count: Arc::new(AtomicU64::new(0)),
            ssd_hit_count: Arc::new(AtomicU64::new(0)),
            ssd_miss_count: Arc::new(AtomicU64::new(0)),
            object_storage_access_count: Arc::new(AtomicU64::new(0)),
        })
    }

    pub async fn new_from_config(
        config: &crate::config::StorageConfig) -> ObjectStorageResult<Self> {
        // 对于纯内存模式，只需要内存配置
        let memory_config =
            config
                .memory
                .as_ref()
                .ok_or_else(|| ObjectStorageError::ConfigurationError {
                    message: "Memory configuration is missing".to_string(),
                })?;

        // SSD 配置是可选的
        let (ssd_cache_dir, ssd_cache_size) = if let Some(ssd_config) = config.ssd.as_ref() {
            (
                PathBuf::from(&ssd_config.cache_dir),
                ssd_config.max_size_gb as u64 * 1024 * 1024 * 1024,
            )
        } else {
            // 如果没有 SSD 配置，使用临时目录（但不会实际使用）
            (
                std::env::temp_dir().join("dxmt_temp_cache"),
                0, // 0 表示不使用 SSD 缓存
            )
        };

        let bucket_name = config
            .object_storage
            .as_ref()
            .map(|c| c.bucket.clone())
            .unwrap_or_else(|| "default-bucket".to_string());

        let tiered_config = TieredStorageConfig {
            memory_cache_size: memory_config.max_size_mb * 1024 * 1024,
            memory_cache_capacity: 10000, // Default capacity, can be made configurable later
            ssd_cache_dir: ssd_cache_dir.clone(),
            ssd_cache_size,
            bucket_name,
            stats_interval_secs: 300,   // Default, can be made configurable
            auto_tiering_enabled: config.ssd.is_some(), // 只有配置了 SSD 才启用自动分层
        };

        let object_storage: Box<dyn ObjectStorage + Send + Sync> =
            if let Some(object_storage_config) = config.object_storage.as_ref() {
                match object_storage_config.provider.as_str() {
                    "filesystem" => Box::new(FilesystemStorage::new(
                        ssd_cache_dir, // Use ssd_cache_dir as base path for filesystem
                        &object_storage_config.bucket,
                    )?),
                    "s3" => Box::new(S3Storage::new(object_storage_config.clone().into()).await?),
                    _ => {
                        return Err(ObjectStorageError::ConfigurationError {
                            message: format!(
                                "Unsupported object storage provider: {}",
                                object_storage_config.provider
                            ),
                        })
                    }
                }
            } else {
                // If no object storage config is provided, use a dummy FilesystemStorage for tests
                Box::new(FilesystemStorage::new(
                    std::env::temp_dir().join("dxmt_test_storage"),
                    "test-bucket",
                )?)
            };

        Self::new(tiered_config, object_storage)
    }

    /// 更新对象访问统计
    async fn update_stats(&self, key: &str, size: u64, tier: StorageTier) -> crate::Result<()> {
        let mut stats = self.stats.write().await;
        let entry = stats.entry(key.to_string()).or_insert(ObjectStats {
            access_count: 0,
            last_access: Utc::now(),
            size,
            current_tier: tier,
        });

        entry.access_count += 1;
        entry.last_access = Utc::now();
        entry.current_tier = tier;
        Ok(())
    }

    /// 从内存获取对象
    async fn get_from_memory(&self, key: &str) -> crate::Result<Option<Vec<u8>>> {
        let mut cache = self.memory_cache.write().await;
        if let Some(data) = cache.get(key) {
            self.update_stats(key, data.len() as u64, StorageTier::Memory)
                .await?;
            return Ok(Some(data.clone()));
        }
        Ok(None)
    }

    /// 存储到内存
    async fn put_to_memory(&self, key: &str, data: &Bytes) -> ObjectStorageResult<()> {
        let data_size = data.len();

        // 检查是否超过内存限制
        if data_size > self.config.memory_cache_size {
            return Ok(()); // 太大的对象直接跳过内存缓存
        }

        let mut cache = self.memory_cache.write().await;
        let mut memory_usage = self.memory_usage.write().await;

        // 清理内存以腾出空间
        while *memory_usage + data_size > self.config.memory_cache_size {
            if let Some((evicted_key, evicted_data)) = cache.pop_lru() {
                *memory_usage -= evicted_data.len();
                // 将被驱逐的数据移动到 SSD
                drop(cache);
                drop(memory_usage);
                self
                    .ssd_storage
                    .put_object(&evicted_key, Bytes::from(evicted_data), None)
                    .await?;
                cache = self.memory_cache.write().await;
                memory_usage = self.memory_usage.write().await;
            } else {
                break;
            }
        }

        cache.put(key.to_string(), data.to_vec());
        *memory_usage += data_size;

        self.update_stats(key, data_size as u64, StorageTier::Memory)
            .await.map_err(|e| ObjectStorageError::InternalError { message: format!("Stats update failed: {:?}", e) })?;
        Ok(())
    }

    /// 从 SSD 获取对象
    async fn get_from_ssd(&self, key: &str) -> ObjectStorageResult<Option<Vec<u8>>> {
        match self.ssd_storage.get_object(key).await {
            Ok(data) => {
                self.update_stats(key, data.len() as u64, StorageTier::Ssd)
                    .await.map_err(|e| ObjectStorageError::InternalError { message: format!("Stats update failed: {:?}", e) })?;

                // 将热数据提升到内存
                if data.len() <= self.config.memory_cache_size / 10 {

                    self.put_to_memory(key, &Bytes::from(data.clone())).await?;

                }
                Ok(Some(data))
            }
            Err(ObjectStorageError::ObjectNotFound { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// 存储到 SSD
    async fn put_to_ssd(&self, key: &str, data: &Bytes) -> ObjectStorageResult<()> {
        self.ssd_storage
            .put_object(key, Bytes::from(data.to_vec()), None)
            .await?;
        self.update_stats(key, data.len() as u64, StorageTier::Ssd)
            .await.map_err(|e| ObjectStorageError::InternalError { message: format!("Stats update failed: {:?}", e) })?;
        Ok(())
    }

    /// 从对象存储获取
    async fn get_from_object_storage(&self, key: &str) -> ObjectStorageResult<Vec<u8>> {
        let data = self.object_storage.get_object(key).await?;
        self.update_stats(key, data.len() as u64, StorageTier::ObjectStorage)
            .await.map_err(|e| ObjectStorageError::InternalError { message: format!("Stats update failed: {:?}", e) })?;

        // 将数据缓存到上层
        if data.len() <= self.config.memory_cache_size / 5 {
            self.put_to_memory(key, &Bytes::from(data.clone())).await?;
        } else {
            self.put_to_ssd(key, &Bytes::from(data.clone())).await?;
        }

        Ok(data)
    }

    /// 获取分层统计信息
    pub async fn get_tier_stats(&self) -> ObjectStorageResult<TierStats> {
        let stats = self.stats.read().await;
        let memory_usage = *self.memory_usage.read().await;

        let mut memory_objects = 0u64;
        let mut ssd_objects = 0u64;
        let mut object_storage_objects = 0u64;

        let memory_size = memory_usage as u64;
        let mut ssd_size = 0u64;
        let mut object_storage_size = 0u64;

        for object_stats in stats.values() {
            match object_stats.current_tier {
                StorageTier::Memory => memory_objects += 1,
                StorageTier::Ssd => {
                    ssd_objects += 1;
                    ssd_size += object_stats.size
                }
                StorageTier::ObjectStorage => {
                    object_storage_objects += 1;
                    object_storage_size += object_stats.size
                }
            }
        }

        Ok(TierStats {
            memory: TierStat {
                object_count: memory_objects,
                total_size: memory_size,
                hit_count: self.memory_hit_count.load(Ordering::Relaxed),
                miss_count: self.memory_miss_count.load(Ordering::Relaxed),
            },
            ssd: TierStat {
                object_count: ssd_objects,
                total_size: ssd_size,
                hit_count: self.ssd_hit_count.load(Ordering::Relaxed),
                miss_count: self.ssd_miss_count.load(Ordering::Relaxed),
            },
            object_storage: TierStat {
                object_count: object_storage_objects,
                total_size: object_storage_size,
                hit_count: self.object_storage_access_count.load(Ordering::Relaxed),
                miss_count: 0, // 对象存储是最底层，没有未命中概念
            },
        })
    }
}

#[async_trait]
impl ObjectStorage for TieredStorage {
    async fn put_object(
        &self,
        key: &str,
        data: Bytes,
        content_type: Option<&str>) -> ObjectStorageResult<()> {
        self.object_storage
            .put_object(key, data.clone(), content_type)
            .await?;
        if data.len() <= self.config.memory_cache_size / 10 {
            self.put_to_memory(key, &data).await?;
        } else if data.len() <= self.config.ssd_cache_size as usize / 100 {
    self.put_to_ssd(key, &data).await?;
}
        Ok(())
    }

    async fn get_object(&self, key: &str) -> ObjectStorageResult<Vec<u8>> {
        // 尝试从内存获取
        if let Some(data) = self.get_from_memory(key).await.map_err(|e| ObjectStorageError::InternalError { message: e.to_string() })? {
            self.memory_hit_count.fetch_add(1, Ordering::Relaxed);
            return Ok(data);
        }
        self.memory_miss_count.fetch_add(1, Ordering::Relaxed);

        // 尝试从 SSD 获取
        if let Some(data) = self.get_from_ssd(key).await? {
            self.ssd_hit_count.fetch_add(1, Ordering::Relaxed);
            return Ok(data);
        }
        self.ssd_miss_count.fetch_add(1, Ordering::Relaxed);

        // 从对象存储获取
        self.object_storage_access_count.fetch_add(1, Ordering::Relaxed);
        self.get_from_object_storage(key).await
    }

    async fn get_object_range(
        &self,
        key: &str,
        start: u64,
        end: u64) -> ObjectStorageResult<Vec<u8>> {
        self.object_storage.get_object_range(key, start, end).await
    }

    async fn delete_object(&self, key: &str) -> ObjectStorageResult<()> {
        // 从所有层删除
        let mut cache = self.memory_cache.write().await;
        if let Some(data) = cache.pop(key) {
            let mut memory_usage = self.memory_usage.write().await;
            *memory_usage -= data.len()
        }
        drop(cache);

        let _ = self.ssd_storage.delete_object(key).await;
        self.object_storage.delete_object(key).await?;

        // 清理统计信息
        let mut stats = self.stats.write().await;
        stats.remove(key);

        Ok(())
    }

    async fn delete_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<String>> {
        let mut deleted = Vec::new();

        for key in keys {
            if self.delete_object(key).await.is_ok() {
                deleted.push(key.clone())
            }
        }

        Ok(deleted)
    }

    async fn object_exists(&self, key: &str) -> ObjectStorageResult<bool> {
        // 检查内存
        {
            let cache = self.memory_cache.read().await;
            if cache.contains(key) {
                return Ok(true);
            }
        }

        // 检查 SSD
        if self.ssd_storage.object_exists(key).await.unwrap_or(false) {
            return Ok(true);
        }

        // 检查对象存储
        self.object_storage.object_exists(key).await
    }

    async fn head_object(&self, key: &str) -> ObjectStorageResult<ObjectMetadata> {
        // 总是从对象存储获取最权威的元数据
        self.object_storage.head_object(key).await
    }

    async fn list_objects(
        &self,
        options: &ListObjectsOptions) -> ObjectStorageResult<ListObjectsResult> {
        self.object_storage.list_objects(options).await
    }

    async fn copy_object(&self, source_key: &str, dest_key: &str) -> ObjectStorageResult<()> {
        self.object_storage.copy_object(source_key, dest_key).await
    }

    async fn put_objects(
        &self,
        objects: Vec<(String, Vec<u8>)>,
    ) -> ObjectStorageResult<Vec<String>> {
        let mut uploaded = Vec::new();

        for (key, data) in objects {
            if self.put_object(&key, Bytes::from(data), None).await.is_ok() {

                uploaded.push(key);

            }
        }

        Ok(uploaded)
    }

    async fn get_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<(String, Vec<u8>)>> {
        let mut objects = Vec::new();

        for key in keys {
            if let Ok(data) = self.get_object(key).await {
                objects.push((key.clone(), data))
            }
        }

        Ok(objects)
    }

    async fn put_stream(
        &self,
        key: &str,
        stream: BoxStream<'static, Result<Bytes, std::io::Error>>,
        content_length: Option<u64>,
        content_type: Option<&str>) -> ObjectStorageResult<()> {
        self.object_storage
            .put_stream(key, stream, content_length, content_type)
            .await
    }

    async fn get_stream(
        &self,
        key: &str) -> ObjectStorageResult<BoxStream<'static, Result<Bytes, ObjectStorageError>>> {
        self.object_storage.get_stream(key).await
    }

    async fn create_multipart_upload(
        &self,
        key: &str,
        content_type: Option<&str>) -> ObjectStorageResult<String> {
        self.object_storage
            .create_multipart_upload(key, content_type).await
    }

    async fn upload_part(
        &self,
        key: &str,
        upload_id: &str,
        part_number: u32,
        data: &[u8]) -> ObjectStorageResult<PartInfo> {
        self.object_storage
            .upload_part(key, upload_id, part_number, data)
            .await
    }

    async fn complete_multipart_upload(
        &self,
        key: &str,
        upload_id: &str,
        parts: &[PartInfo]) -> ObjectStorageResult<()> {
        self.object_storage
            .complete_multipart_upload(key, upload_id, parts)
            .await
    }

    async fn abort_multipart_upload(&self, key: &str, upload_id: &str) -> ObjectStorageResult<()> {
        self.object_storage
            .abort_multipart_upload(key, upload_id)
            .await
    }

    async fn get_storage_stats(&self) -> ObjectStorageResult<StorageStats> {
        self.object_storage.get_storage_stats().await
    }

    async fn health_check(&self) -> ObjectStorageResult<()> {
        self.object_storage.health_check().await
    }
}

/// 单层统计信息
#[derive(Debug, Clone)]
pub struct TierStat {
    pub object_count: u64,
    pub total_size: u64,
    pub hit_count: u64,
    pub miss_count: u64,
}

/// 分层统计信息
#[derive(Debug, Clone)]
pub struct TierStats {
    pub memory: TierStat,
    pub ssd: TierStat,
    pub object_storage: TierStat,
}
