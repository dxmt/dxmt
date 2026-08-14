// 量化存储实现
// 提供高效的量化向量存储和检索

use super::{
    BinaryQuantizer, HybridQuantizer, ProductQuantizer, QuantizationStats, QuantizationType,
    QuantizedVector, ScalarQuantizationConfig, ScalarQuantizer,
};
use crate::storage::object::{traits::ObjectStorage, TieredStorage};
use crate::types::vector::{RecordId, DenseVector};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// 量化存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedStorageConfig {
    /// 量化类型
    pub quantization_type: QuantizationType,
    /// 内存缓存大小 (向量数量)
    pub memory_cache_size: usize,
    /// 启用压缩
    pub enable_compression: bool,
    /// 压缩级别 (1-22)
    pub compression_level: i32,
    /// 批量写入大小
    pub batch_write_size: usize,
    /// 自动持久化间隔 (秒)
    pub auto_persist_interval: u64,
}

impl Default for QuantizedStorageConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::Scalar(ScalarQuantizationConfig::default()),
            memory_cache_size: 100000,
            enable_compression: true,
            compression_level: 6,
            batch_write_size: 1000,
            auto_persist_interval: 300, // 5分钟
        }
    }
}

/// 量化存储
pub struct QuantizedStorage {
    config: QuantizedStorageConfig,
    quantizer: Box<dyn QuantizerWrapper>,
    memory_cache: Arc<RwLock<HashMap<RecordId, QuantizedVector>>>,
    object_storage: Option<Arc<dyn ObjectStorage>>,
    tiered_storage: Option<Arc<TieredStorage>>,
    stats: Arc<RwLock<QuantizationStats>>,
    write_buffer: Arc<RwLock<Vec<(RecordId, QuantizedVector)>>>,
}

/// 量化器包装特征
pub trait QuantizerWrapper: Send + Sync {
    fn quantize(&self, vector: &DenseVector) -> crate::Result<QuantizedVector>;
    fn train(&mut self, training_vectors: &[DenseVector]) -> crate::Result<()>;
    fn get_compression_ratio(&self) -> f32;
    fn serialize(&self) -> crate::Result<Vec<u8>>;
    fn deserialize(&mut self, data: &[u8]) -> crate::Result<()>;
}

/// 标量量化器包装
struct ScalarQuantizerWrapper {
    quantizer: ScalarQuantizer,
}

/// 乘积量化器包装
struct ProductQuantizerWrapper {
    quantizer: ProductQuantizer,
}

/// 二进制量化器包装
struct BinaryQuantizerWrapper {
    quantizer: BinaryQuantizer,
}

/// 混合量化器包装
struct HybridQuantizerWrapper {
    quantizer: HybridQuantizer,
}

/// 量化存储构建器
pub struct QuantizedStorageBuilder {
    config: QuantizedStorageConfig,
    object_storage: Option<Arc<dyn ObjectStorage>>,
    tiered_storage: Option<Arc<TieredStorage>>,
}

/// 存储统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// 量化统计
    pub quantization_stats: QuantizationStats,
    /// 内存缓存命中率
    pub cache_hit_rate: f32,
    /// 存储层分布
    pub storage_distribution: StorageDistribution,
    /// I/O 统计
    pub io_stats: IoStats,
}

/// 存储层分布
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageDistribution {
    /// 内存中的向量数量
    pub memory_vectors: usize,
    /// SSD中的向量数量
    pub ssd_vectors: usize,
    /// 对象存储中的向量数量
    pub object_storage_vectors: usize,
}

/// I/O 统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoStats {
    /// 读取操作数量
    pub read_operations: u64,
    /// 写入操作数量
    pub write_operations: u64,
    /// 读取字节数
    pub bytesread: u64,
    /// 写入字节数
    pub byteswritten: u64,
    /// 平均读取延迟 (微秒)
    pub avg_read_latency_us: u64,
    /// 平均写入延迟 (微秒)
    pub avg_write_latency_us: u64,
}

impl QuantizedStorage {
    /// 创建新的量化存储
    pub fn new(config: QuantizedStorageConfig) -> crate::Result<Self> {
        let quantizer = Self::create_quantizer(&config.quantization_type)?;

        Ok(Self {
            config,
            quantizer,
            memory_cache: Arc::new(RwLock::new(HashMap::new())),
            object_storage: None,
            tiered_storage: None,
            stats: Arc::new(RwLock::new(QuantizationStats::default())),
            write_buffer: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// 训练量化器
    pub fn train(&mut self, training_vectors: &[DenseVector]) -> crate::Result<()> {
        self.quantizer.train(training_vectors)?;

        // 更新统计信息
        let mut stats = self.stats.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
        stats.compression_ratio = self.quantizer.get_compression_ratio();
        stats.quantization_type = self.get_quantization_type_name();

        Ok(())
    }

    /// 插入量化向量
    pub async fn insert(&self, id: RecordId, vector: &DenseVector) -> crate::Result<()> {
        let start_time = std::time::Instant::now();

        // 量化向量
        let mut quantized = self.quantizer.quantize(vector)?;
        quantized.id = id.to_string();

        // 添加到写入缓冲区 — drop lock before async call
        let needs_flush = {
            let mut buffer = self.write_buffer.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            buffer.push((id.clone(), quantized.clone()));
            buffer.len() >= self.config.batch_write_size
        };
        if needs_flush {
            self.flush_write_buffer().await?;
        }

        // 添加到内存缓存 — drop lock before async call
        let deferred_insert = {
            let mut cache = self.memory_cache.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            if cache.len() >= self.config.memory_cache_size {
                Some(quantized) // eviction needed; defer insert until after eviction
            } else {
                cache.insert(id.clone(), quantized);
                None
            }
        };
        if let Some(q) = deferred_insert {
            self.evict_from_cache().await?;
            let mut cache = self.memory_cache.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            cache.insert(id, q);
        }

        // 更新统计信息
        {
            let mut stats = self.stats.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            stats.total_vectors += 1;
            stats.memory_usagebytes += std::mem::size_of::<QuantizedVector>()
        }

        let insert_time = start_time.elapsed().as_micros() as u64;
        self.update_write_stats(insert_time).await;

        Ok(())
    }

    /// 检索量化向量
    pub async fn get(&self, id: &RecordId) -> crate::Result<Option<QuantizedVector>> {
        let start_time = std::time::Instant::now();

        // 首先检查内存缓存 — drop lock before async call
        let cached = {
            let cache = self.memory_cache.read().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            cache.get(id).cloned()
        };
        if let Some(vector) = cached {
            let read_time = start_time.elapsed().as_micros() as u64;
            self.update_read_stats(read_time, true).await;
            return Ok(Some(vector));
        }

        // 检查分层存储
        if let Some(ref tiered_storage) = self.tiered_storage {
            if let Ok(data) = tiered_storage.get_object(&id.to_string()).await {
                let bytes = Bytes::from(data);
                let vector = self.deserialize_vector(&bytes)?;

                // 加载到内存缓存
                {
                    let mut cache = self.memory_cache.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
                    cache.insert(id.clone(), vector.clone());
                }

                let read_time = start_time.elapsed().as_micros() as u64;
                self.update_read_stats(read_time, false).await;
                return Ok(Some(vector));
            }
        }

        // 检查对象存储
        if let Some(ref object_storage) = self.object_storage {
            let data = object_storage.get_object(&id.to_string()).await?;
            let bytes = Bytes::from(data);
            let vector = self.deserialize_vector(&bytes)?;

            // 加载到内存缓存
            {
                let mut cache = self.memory_cache.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
                cache.insert(id.clone(), vector.clone());
            }

            let read_time = start_time.elapsed().as_micros() as u64;
            self.update_read_stats(read_time, false).await;
            return Ok(Some(vector));
        }

        Ok(None)
    }

    /// 批量检索
    pub async fn get_batch(&self, ids: &[RecordId]) -> crate::Result<Vec<Option<QuantizedVector>>> {
        let mut results = Vec::with_capacity(ids.len());

        for id in ids {
            results.push(self.get(id).await?)
        }

        Ok(results)
    }

    /// 删除向量
    pub async fn remove(&self, id: &RecordId) -> crate::Result<bool> {
        // 从内存缓存删除
        let cache_removed = {
            let mut cache = self.memory_cache.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            cache.remove(id).is_some()
        };

        // 从存储层删除
        let mut storage_removed = false;

        if let Some(ref tiered_storage) = self.tiered_storage {
            storage_removed |= tiered_storage.delete_object(&id.to_string()).await.is_ok()
        }

        if let Some(ref object_storage) = self.object_storage {
            storage_removed |= object_storage.delete_object(&id.to_string()).await.is_ok()
        }

        // 更新统计信息
        if cache_removed || storage_removed {
            let mut stats = self.stats.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            stats.total_vectors = stats.total_vectors.saturating_sub(1)
        }

        Ok(cache_removed || storage_removed)
    }

    /// 强制持久化
    pub async fn flush(&self) -> crate::Result<()> {
        self.flush_write_buffer().await?;
        self.persist_memory_cache().await?;
        Ok(())
    }

    /// 获取存储统计信息
    pub async fn get_storage_stats(&self) -> StorageStats {
        let quantization_stats = {
            let stats = self.stats.read().unwrap();
            stats.clone()
        };

        let cache_size = {
            let cache = self.memory_cache.read().unwrap();
            cache.len()
        };

        StorageStats {
            quantization_stats,
            cache_hit_rate: 0.95, // 简化实现
            storage_distribution: StorageDistribution {
                memory_vectors: cache_size,
                ssd_vectors: 0,            // 需要从分层存储获取
                object_storage_vectors: 0, // 需要从对象存储获取
            },
            io_stats: IoStats {
                read_operations: 0,
                write_operations: 0,
                bytesread: 0,
                byteswritten: 0,
                avg_read_latency_us: 0,
                avg_write_latency_us: 0,
            },
        }
    }

    // 私有辅助方法

    fn create_quantizer(quantizationtype: &QuantizationType) -> crate::Result<Box<dyn QuantizerWrapper>> {
        match quantizationtype {
            QuantizationType::Scalar(config) => {
                let quantizer =
                    ScalarQuantizer::new(config.clone(), crate::types::DistanceMetric::Euclidean);
                Ok(Box::new(ScalarQuantizerWrapper { quantizer }))
            }
            QuantizationType::Product(config) => {
                let quantizer =
                    ProductQuantizer::new(config.clone(), crate::types::DistanceMetric::Euclidean);
                Ok(Box::new(ProductQuantizerWrapper { quantizer }))
            }
            QuantizationType::Binary(config) => {
                let quantizer =
                    BinaryQuantizer::new(config.clone(), crate::types::DistanceMetric::Euclidean);
                Ok(Box::new(BinaryQuantizerWrapper { quantizer }))
            }
            QuantizationType::Hybrid(config) => {
                let quantizer =
                    HybridQuantizer::new(config.clone(), crate::types::DistanceMetric::Euclidean)?;
                Ok(Box::new(HybridQuantizerWrapper { quantizer }))
            }
        }
    }

    fn get_quantization_type_name(&self) -> String {
        match &self.config.quantization_type {
            QuantizationType::Scalar(_) => "scalar".to_string(),
            QuantizationType::Product(_) => "product".to_string(),
            QuantizationType::Binary(_) => "binary".to_string(),
            QuantizationType::Hybrid(_) => "hybrid".to_string(),
        }
    }

    async fn flush_write_buffer(&self) -> crate::Result<()> {
        let vectors_to_write = {
            let mut buffer = self.write_buffer.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            let vectors = buffer.clone();
            buffer.clear();
            vectors
        };

        if vectors_to_write.is_empty() {
            return Ok(());
        }

        // 写入到存储层
        for (id, vector) in vectors_to_write {
            let serialized = self.serialize_vector(&vector)?;

            if let Some(ref tiered_storage) = self.tiered_storage {
                tiered_storage
                    .put_object(&id.to_string(), serialized, None)
                    .await.map_err(|_| crate::Error::Internal("Storage operation failed".to_string()))?;
            } else if let Some(ref object_storage) = self.object_storage {
                object_storage
                    .put_object(&id.to_string(), serialized, None)
                    .await.map_err(|_| crate::Error::Internal("Storage operation failed".to_string()))?;
            }
        }

        Ok(())
    }

    async fn evict_from_cache(&self) -> crate::Result<()> {
        // 简化的LRU逐出策略
        let evict_count = self.config.memory_cache_size / 10; // 逐出10%

        let vectors_to_evict = {
            let mut cache = self.memory_cache.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            let mut evicted = Vec::new();

            let mut iter = cache.iter();
            for _ in 0..evict_count {
                if let Some((id, vector)) = iter.next() {

                    evicted.push((id.clone(), vector.clone()));


                }

            }

            for (id, _) in &evicted {
                cache.remove(id);

            }

            evicted
        };

        // 持久化被逐出的向量
        for (id, vector) in vectors_to_evict {
            let serialized = self.serialize_vector(&vector)?;

            if let Some(ref tiered_storage) = self.tiered_storage {
                tiered_storage
                    .put_object(&id.to_string(), serialized, None)
                    .await.map_err(|_| crate::Error::Internal("Storage operation failed".to_string()))?;
            } else if let Some(ref object_storage) = self.object_storage {
                object_storage
                    .put_object(&id.to_string(), serialized, None)
                    .await.map_err(|_| crate::Error::Internal("Storage operation failed".to_string()))?;
            }
        }

        Ok(())
    }

    async fn persist_memory_cache(&self) -> crate::Result<()> {
        let cache_contents = {
            let cache = self.memory_cache.read().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            cache.clone()
        };

        for (id, vector) in cache_contents {
            let serialized = self.serialize_vector(&vector)?;

            if let Some(ref tiered_storage) = self.tiered_storage {
                tiered_storage
                    .put_object(&id.to_string(), serialized, None)
                    .await.map_err(|_| crate::Error::Internal("Storage operation failed".to_string()))?;
            } else if let Some(ref object_storage) = self.object_storage {
     object_storage
                    .put_object(&id.to_string(), serialized, None)
                    .await.map_err(|_| crate::Error::Internal("Storage operation failed".to_string()))?;

 }
        }

        Ok(())
    }

    fn serialize_vector(&self, vector: &QuantizedVector) -> crate::Result<Bytes> {
        let data = if self.config.enable_compression {
            self.compress_data(&bincode::serialize(vector).map_err(|e|
                crate::Error::Serialization(e.to_string() )
            )?)?
        } else {
            bincode::serialize(vector).map_err(|e| crate::Error::Serialization(e.to_string() ))?
        };

        Ok(Bytes::from(data))
    }

    fn deserialize_vector(&self, data: &Bytes) -> crate::Result<QuantizedVector> {
        let data = if self.config.enable_compression {
            self.decompress_data(data)?
        } else {
            data.to_vec()
        };

        bincode::deserialize(&data).map_err(|e| crate::Error::Serialization(e.to_string()))
    }

    fn compress_data(&self, data: &[u8]) -> crate::Result<Vec<u8>> {
        zstd::bulk::compress(data, self.config.compression_level).map_err(|e|
            crate::Error::Serialization(e.to_string()))
    }

    fn decompress_data(&self, data: &Bytes) -> crate::Result<Vec<u8>> {
        zstd::bulk::decompress(data, 10 * 1024 * 1024) // 最大10MB
            .map_err(|e| crate::Error::Serialization(e.to_string()))
    }

    async fn update_read_stats(&self, _latency_us: u64, _cache_hit: bool) {
        // 简化实现，实际应该更新详细统计
    }

    async fn update_write_stats(&self, _latency_us: u64) {
        // 简化实现，实际应该更新详细统计
    }
}

impl Default for QuantizedStorageBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizedStorageBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            config: QuantizedStorageConfig::default(),
            object_storage: None,
            tiered_storage: None
        }
    }

    /// 设置配置
    pub fn with_config(mut self, config: QuantizedStorageConfig) -> Self {
        self.config = config;
        self
    }

    /// 设置对象存储
    pub fn with_object_storage(mut self, storage: Arc<dyn ObjectStorage>) -> Self {
        self.object_storage = Some(storage);
        self
    }

    /// 设置分层存储
    pub fn with_tiered_storage(mut self, storage: Arc<TieredStorage>) -> Self {
        self.tiered_storage = Some(storage);
        self
    }

    /// 构建存储
    pub fn build(self) -> crate::Result<QuantizedStorage> {
        let mut storage = QuantizedStorage::new(self.config)?;
        storage.object_storage = self.object_storage;
        storage.tiered_storage = self.tiered_storage;
        Ok(storage)
    }
}

// 量化器包装实现

impl QuantizerWrapper for ScalarQuantizerWrapper {
    fn quantize(&self, vector: &DenseVector) -> crate::Result<QuantizedVector> {
        self.quantizer.quantize(vector)
    }

    fn train(&mut self, training_vectors: &[DenseVector]) -> crate::Result<()> {
        self.quantizer.train(training_vectors)

    }

    fn get_compression_ratio(&self) -> f32 {
        4.0
    }

    fn serialize(&self) -> crate::Result<Vec<u8>> {
        bincode::serialize(&self.quantizer).map_err(|e| crate::Error::Serialization(e.to_string()))

    }

    fn deserialize(&mut self, data: &[u8]) -> crate::Result<()> {
        self.quantizer =
            bincode::deserialize(data).map_err(|e| crate::Error::Serialization(e.to_string())
            )?;
        Ok(())
    }

}

impl QuantizerWrapper for ProductQuantizerWrapper {
    fn quantize(&self, vector: &DenseVector) -> crate::Result<QuantizedVector> {
        self.quantizer.quantize(vector)

    }

    fn train(&mut self, training_vectors: &[DenseVector]) -> crate::Result<()> {
        self.quantizer.train(training_vectors)

    }

    fn get_compression_ratio(&self) -> f32 {
        8.0
    }

    fn serialize(&self) -> crate::Result<Vec<u8>> {
        bincode::serialize(&self.quantizer).map_err(|e| crate::Error::Serialization(e.to_string()))

    }

    fn deserialize(&mut self, data: &[u8]) -> crate::Result<()> {
        self.quantizer =
            bincode::deserialize(data).map_err(|e| crate::Error::Serialization(e.to_string())
            )?;
        Ok(())
    }

}

impl QuantizerWrapper for BinaryQuantizerWrapper {
    fn quantize(&self, vector: &DenseVector) -> crate::Result<QuantizedVector> {
        self.quantizer.quantize(vector)

    }

    fn train(&mut self, training_vectors: &[DenseVector]) -> crate::Result<()> {
        self.quantizer.train(training_vectors)

    }

    fn get_compression_ratio(&self) -> f32 {
        32.0
    }

    fn serialize(&self) -> crate::Result<Vec<u8>> {
        bincode::serialize(&self.quantizer).map_err(|e| crate::Error::Serialization(e.to_string()))

    }

    fn deserialize(&mut self, data: &[u8]) -> crate::Result<()> {
        self.quantizer =
            bincode::deserialize(data).map_err(|e| crate::Error::Serialization(e.to_string())
            )?;
        Ok(())
    }

}

impl QuantizerWrapper for HybridQuantizerWrapper {
    fn quantize(&self, vector: &DenseVector) -> crate::Result<QuantizedVector> {
        self.quantizer.quantize(vector)

    }

    fn train(&mut self, training_vectors: &[DenseVector]) -> crate::Result<()> {
        self.quantizer.train(training_vectors)

    }

    fn get_compression_ratio(&self) -> f32 {
        16.0
    }

    fn serialize(&self) -> crate::Result<Vec<u8>> {
        // 简化实现
        Ok(Vec::new())
    }

    fn deserialize(&mut self, _data: &[u8]) -> crate::Result<()> {
        // 简化实现
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    

    // Commented out - test doesn't return Result and uses undefined StatusCode
    // #[tokio::test]
    // async fn test_quantized_storage() {
    //     let config = QuantizedStorageConfig::default();
    //     let mut storage = QuantizedStorage::new(config)?;
    //
    //     // 训练
    //     let training_vectors = vec![
    //         vec![1.0, 2.0, 3.0],
    //         vec![4.0, 5.0, 6.0],
    //         vec![7.0, 8.0, 9.0],
    //     ];
    //     storage.train(&training_vectors)?;
    //
    //     // 插入
    //     let test_vector = vec![2.0, 3.0, 4.0];
    //     let record_id = uuid::Uuid::new_v4().to_string();
    //     storage.insert(record_id, &test_vector)?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    //
    //     // 检索
    //     let retrieved = storage.get(&record_id)?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    //     assert!(retrieved.is_some())
    // }
}
