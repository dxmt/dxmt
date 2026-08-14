// 批量操作优化 - 高效的批量插入和搜索实现
// 包含并行处理、内存池管理和流水线优化

use crate::types::{vector::DenseVector, DistanceMetric};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, Semaphore};
use futures;
use crate::types::record::Filter;
use crate::Result;
use rayon::prelude::*;

use crate::search::VectorIndex;
// use crate::search::adaptive_simd // Removed::AdaptiveSimdEngine;
use crate::types::{vector::RecordId};

/// 批量操作配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// 最大批量大小
    pub max_batch_size: usize,
    /// 批量超时时间 (毫秒)
    pub batch_timeout_ms: u64,
    /// 并行度
    pub parallelism: usize,
    /// 启用流水线处理
    pub enable_pipelining: bool,
    /// 内存池大小
    pub memory_pool_size: usize,
    /// 启用自适应批量大小
    pub adaptive_batch_size: bool,
    /// 压缩阈值
    pub compression_threshold: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            batch_timeout_ms: 100,
            parallelism: num_cpus::get(),
            enable_pipelining: true,
            memory_pool_size: 50000,
            adaptive_batch_size: true,
            compression_threshold: 10000,
        }
    }
}

/// 批量插入请求
#[derive(Debug)]
pub struct BatchInsertRequest {
    /// 记录ID
    pub id: RecordId,
    /// 向量数据
    pub vector: DenseVector,
    /// 元数据
    pub payload: Option<serde_json::Value>,
    /// 响应通道
    pub responsetx: Option<oneshot::Sender<Result<()>>>,
}

/// 批量搜索请求
#[derive(Debug)]
pub struct BatchSearchRequest {
    /// 查询向量
    pub query: DenseVector,
    /// 返回结果数量
    pub k: usize,
    /// 距离度量
    pub metric: DistanceMetric,
    /// 过滤条件
    pub filter: Option<Filter>,
    /// 响应通道
    pub responsetx: oneshot::Sender<Result<Vec<(RecordId, f32)>>>,
}

/// 批量操作统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    /// 处理的批次数量
    pub total_batches: u64,
    /// 平均批量大小
    pub avg_batch_size: f64,
    /// 平均处理时间 (微秒)
    pub avg_processing_time_us: f64,
    /// 吞吐量 (操作/秒)
    pub throughput_ops_per_sec: f64,
    /// 队列长度
    pub queue_length: usize,
    /// 内存使用情况
    pub memory_usage: MemoryUsage,
}

/// 内存使用情况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// 已分配内存 (字节)
    pub allocatedbytes: usize,
    /// 使用中内存 (字节)
    pub usedbytes: usize,
    /// 缓存命中率
    pub cache_hit_rate: f32,
}

/// 批量操作引擎
pub struct BatchOperationEngine {
    config: BatchConfig,
    insert_queue: Arc<Mutex<VecDeque<BatchInsertRequest>>>,
    search_queue: Arc<Mutex<VecDeque<BatchSearchRequest>>>,
    statistics: Arc<RwLock<BatchStatistics>>,
    // simd_engine removed after cleanup
    memory_pool: Arc<MemoryPool>,
    processing_semaphore: Arc<Semaphore>,
    is_running: Arc<std::sync::atomic::AtomicBool>,
}

/// 内存池管理
struct MemoryPool {
    #[allow(dead_code)]
    config: BatchConfig,
    vector_pool: Mutex<Vec<Vec<f32>>>,
    result_pool: Mutex<Vec<Vec<(RecordId, f32)>>>,
    allocated_count: std::sync::atomic::AtomicUsize,
    hit_count: std::sync::atomic::AtomicUsize,
    miss_count: std::sync::atomic::AtomicUsize,
}

/// 批量处理结果
#[derive(Debug)]
pub struct BatchProcessingResult {
    /// 成功处理的数量
    pub successful_count: usize,
    /// 失败处理的数量
    pub failedcount: usize,
    /// 处理时间
    pub processing_time: Duration,
    /// 平均每个操作的时间
    pub avg_operation_time: Duration,
}

impl BatchOperationEngine {
    /// 创建新的批量操作引擎
    pub fn new(config: BatchConfig) -> Self {
        let memory_pool = Arc::new(MemoryPool::new(config.clone()));
        let processing_semaphore = Arc::new(Semaphore::new(config.parallelism));

        Self {
            config,
            insert_queue: Arc::new(Mutex::new(VecDeque::new())),
            search_queue: Arc::new(Mutex::new(VecDeque::new())),
            statistics: Arc::new(RwLock::new(BatchStatistics::default())),
            memory_pool,
            processing_semaphore,
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// 启动批量处理
    pub async fn start(&self) -> crate::Result<()> {
        self.is_running
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // 启动插入处理任务
        let insert_task = self.spawn_insert_processor();

        // 启动搜索处理任务
        let search_task = self.spawn_search_processor();

        // 启动统计更新任务
        let stats_task = self.spawn_statistics_updater();

        tokio::try_join!(insert_task, search_task, stats_task)?;

        Ok(())
    }

    /// 停止批量处理
    pub fn stop(&self) {
        self.is_running
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// 提交批量插入请求
    pub async fn submit_insert(&self, requests: Vec<BatchInsertRequest>) -> crate::Result<()> {
        let mut queue = self.insert_queue.lock().unwrap();
        for request in requests {
            queue.push_back(request);
        }
        Ok(())
    }

    /// 提交单个插入请求
    pub async fn submit_single_insert(
        &self,
        id: RecordId,
        vector: DenseVector,
        payload: Option<serde_json::Value>) -> crate::Result<()> {
        let (tx, rx) = oneshot::channel();
        let request = BatchInsertRequest {
            id,
            vector,
            payload,
            responsetx: Some(tx),
        };

        {
            let mut queue = self.insert_queue.lock().unwrap();
            queue.push_back(request);
        }

        match rx.await {
            Ok(result) => result.map_err(|_| crate::Error::Internal("Operation failed".to_string()))?,
            Err(_) => return Err(crate::Error::InvalidInput("Insert request cancelled".to_string())),
        };
        Ok(())
    }

    /// 提交批量搜索请求
    pub async fn submit_search(
        &self,
        query: DenseVector,
        k: usize,
        metric: DistanceMetric,
        filter: Option<Filter>) -> crate::Result<Vec<(RecordId, f32)>> {
        let (tx, rx) = oneshot::channel();
        let request = BatchSearchRequest {
            query,
            k,
            metric,
            filter,
            responsetx: tx,
        };

        {
            let mut queue = self.search_queue.lock().unwrap();
            queue.push_back(request);
        }

        let result = match rx.await {
            Ok(result) => result.map_err(|_| crate::Error::InvalidInput("Search request cancelled".to_string()))?,
            Err(_) => return Err(crate::Error::InvalidInput("Search request cancelled".to_string())),
        };
        Ok(result)
    }

    /// 批量搜索优化
    pub async fn batch_search(
        &self,
        queries: &[DenseVector],
        k: usize,
        metric: &DistanceMetric,
        targets: &[DenseVector]) -> crate::Result<Vec<Vec<(usize, f32)>>> {
        let start_time = Instant::now();

        // 计算距离矩阵
        let distance_matrix: Vec<Vec<f32>> = queries
            .iter()
            .map(|query| {
                targets
                    .iter()
                    .map(|target| {
                        metric.calculate(query, target).unwrap_or(f32::MAX)
                    })
                    .collect()
            })
            .collect();

        // 并行处理每个查询的TopK选择
        let results: Vec<Vec<(usize, f32)>> = distance_matrix
            .into_par_iter()
            .map(|distances| {
                // 选择TopK结果
                let mut indexeddistances: Vec<(usize, f32)> =
                    distances.into_iter().enumerate().collect();

                // 部分排序优化
                let k_actual = k.min(indexeddistances.len());
                indexeddistances.select_nth_unstable_by(k_actual - 1, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                indexeddistances.truncate(k_actual);
                indexeddistances
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                indexeddistances
            })
            .collect();

        // 更新统计信息
        self.update_search_statistics(queries.len(), start_time.elapsed(), 0).await;

        Ok(results)
    }

    /// 流水线批量插入
    // VectorIndex::add is async in signature but all implementations are synchronous (no yield points).
    #[allow(clippy::await_holding_lock)]
    pub async fn pipeline_insert<I: VectorIndex + Send + Sync>(
        &self,
        index: Arc<RwLock<I>>,
        requests: Vec<BatchInsertRequest>) -> crate::Result<BatchProcessingResult> {
        let start_time = Instant::now();
        let total_requests = requests.len();

        if !self.config.enable_pipelining {
            return self.sequential_insert(index, requests).await
        }

        // 分批处理
        let batch_size = if self.config.adaptive_batch_size {
            self.calculate_adaptive_batch_size(total_requests)
        } else {
            self.config.max_batch_size
};

        let batches: Vec<_> = requests.chunks(batch_size).collect();
        let mut successful_count = 0;
        let mut failed_count = 0;

        // 并行处理批次
        let mut batch_futures = Vec::new();

        for batch in batches {
            let index_clone = index.clone();
            let semaphore = self.processing_semaphore.clone();

            let future = async move {
                let _permit = semaphore.acquire().await.map_err(|_| crate::Error::InvalidInput("Processing capacity exceeded".to_string()))?;

                let mut batch_successful = 0;
                for request in batch {
                    let result = async {
                        let mut index_write = index_clone.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
                        index_write.add(request.id.clone(), request.vector.clone()).await
                    }.await;
                    match result {
                        Ok(_) => { batch_successful += 1; }
                        Err(_) => {}
                    }
                }
                Ok(batch_successful)
            };

            batch_futures.push(future);
        }

        let results: Vec<Result<usize>> = futures::future::join_all(batch_futures).await;

        // 汇总结果
        for result in results {
            match result {
                Ok(count) => successful_count += count,
                Err(_) => failed_count += batch_size,
            }
        }

        let processing_time = start_time.elapsed();
        Ok(BatchProcessingResult {
            successful_count,
            failedcount: failed_count,
            processing_time,
            avg_operation_time: processing_time / total_requests as u32,
        })
    }

    /// 自适应批量大小计算
    fn calculate_adaptive_batch_size(&self, total_requests: usize) -> usize {
        let stats = self.statistics.read().unwrap();
        let current_throughput = stats.throughput_ops_per_sec;

        // 基于当前吞吐量调整批量大小
        let base_size = if current_throughput > 1000.0 {
            self.config.max_batch_size
        } else if current_throughput > 100.0 {
            self.config.max_batch_size / 2
        } else {
            self.config.max_batch_size / 4
        };

        base_size.min(total_requests)
    }

    /// 顺序插入处理
    #[allow(clippy::await_holding_lock)]
    async fn sequential_insert<I: VectorIndex + Send + Sync>(
        &self,
        index: Arc<RwLock<I>>,
        requests: Vec<BatchInsertRequest>) -> crate::Result<BatchProcessingResult> {
        let start_time = Instant::now();
        let mut successful_count = 0;
        let mut failed_count = 0;

        for request in requests {
            let result = async {
                let mut index_write = index.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
                index_write.add(request.id.clone(), request.vector.clone()).await
            }.await;
            match result {
                Ok(_) => {
                    successful_count += 1;
                    if let Some(tx) = request.responsetx {
                        let _ = tx.send(Ok(()));
                    }
                }
                Err(e) => {
                    failed_count += 1;
                    if let Some(tx) = request.responsetx {
                        let _ = tx.send(Err(e));
                    }
                }
            }
        }

        let processing_time = start_time.elapsed();
        Ok(BatchProcessingResult {
            successful_count,
            failedcount: failed_count,
            processing_time,
            avg_operation_time: processing_time / (successful_count + failed_count) as u32,
        })
    }

    /// 启动插入处理器
    async fn spawn_insert_processor(&self) -> crate::Result<()> {
        let queue = Arc::clone(&self.insert_queue);
        let is_running = Arc::clone(&self.is_running);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut batch = Vec::new();
            let mut last_process_time = Instant::now();

            while is_running.load(std::sync::atomic::Ordering::Relaxed) {
                // 收集批量请求
                {
                    let mut queue_lock = queue.lock().unwrap();
                    while let Some(request) = queue_lock.pop_front() {
                        batch.push(request);
                        if batch.len() >= config.max_batch_size {

                            break;


                        }
                    }
                }

                // 检查是否需要处理批次
                let should_process = !batch.is_empty()
                    && batch.len() >= config.max_batch_size
                        || last_process_time.elapsed()
                            >= Duration::from_millis(config.batch_timeout_ms);

                if should_process {
                    // 这里实际处理插入操作
                    // 具体实现需要传入索引实例
                    batch.clear();
                    last_process_time = Instant::now()
                }

                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });

        Ok(())
    }

    /// 启动搜索处理器
    async fn spawn_search_processor(&self) -> crate::Result<()> {
        let queue = Arc::clone(&self.search_queue);
        let is_running = Arc::clone(&self.is_running);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut batch = Vec::new();
            let mut last_process_time = Instant::now();

            while is_running.load(std::sync::atomic::Ordering::Relaxed) {
                // 收集批量请求
                {
                    let mut queue_lock = queue.lock().unwrap();
                    while let Some(request) = queue_lock.pop_front() {
                        batch.push(request);
                        if batch.len() >= config.max_batch_size {

                            break;


                        }
                    }
                }

                // 检查是否需要处理批次
                let should_process = !batch.is_empty()
                    && batch.len() >= config.max_batch_size
                        || last_process_time.elapsed()
                            >= Duration::from_millis(config.batch_timeout_ms);

                if should_process {
                    // 这里实际处理搜索操作
                    batch.clear();
                    last_process_time = Instant::now()
                }

                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });

        Ok(())
    }

    /// 启动统计更新器
    async fn spawn_statistics_updater(&self) -> crate::Result<()> {
        let statistics = Arc::clone(&self.statistics);
        let memory_pool = Arc::clone(&self.memory_pool);
        let is_running = Arc::clone(&self.is_running);

        tokio::spawn(async move {
            while is_running.load(std::sync::atomic::Ordering::Relaxed) {
                // 更新统计信息
                {
                    let mut stats = statistics.write().unwrap();
                    stats.memory_usage = memory_pool.get_usage_stats();
                }

                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });

        Ok(())
    }

    /// 更新搜索统计信息
    async fn update_search_statistics(&self, query_count: usize, processing_time: Duration, _failed_operations: usize) {
        let mut stats = self.statistics.write().unwrap();
        stats.total_batches += 1;

        let total_operations =
            (stats.total_batches - 1) as f64 * stats.avg_batch_size + query_count as f64;
        stats.avg_batch_size = total_operations / stats.total_batches as f64;

        let total_time_us = (stats.total_batches - 1) as f64 * stats.avg_processing_time_us
            + processing_time.as_micros() as f64;
        stats.avg_processing_time_us = total_time_us / stats.total_batches as f64;

        stats.throughput_ops_per_sec = query_count as f64 / processing_time.as_secs_f64();
    }

    /// 获取统计信息
    pub fn get_statistics(&self) -> BatchStatistics {
        let stats = self.statistics.read().unwrap();
        let mut result = stats.clone();

        // 更新队列长度
        result.queue_length = {
            let insert_queue = self.insert_queue.lock().unwrap();
            let search_queue = self.search_queue.lock().unwrap();
            insert_queue.len() + search_queue.len()
        };

        result
    }
}

impl MemoryPool {
    fn new(config: BatchConfig) -> Self {
        Self {
            config,
            vector_pool: Mutex::new(Vec::new()),
            result_pool: Mutex::new(Vec::new()),
            allocated_count: std::sync::atomic::AtomicUsize::new(0),
            hit_count: std::sync::atomic::AtomicUsize::new(0),
            miss_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    #[allow(dead_code)]
    fn get_vector(&self, capacity: usize) -> Vec<f32> {
        let mut pool = self.vector_pool.lock().unwrap();

        if let Some(mut vector) = pool.pop() {
            self.hit_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            vector.clear();
            vector.reserve(capacity);
            vector
        } else {
            self.miss_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.allocated_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Vec::with_capacity(capacity)
        }
    }

    #[allow(dead_code)]
    fn return_vector(&self, vector: Vec<f32>) {
        let mut pool = self.vector_pool.lock().unwrap();
        if pool.len() < self.config.memory_pool_size {

            pool.push(vector);


        }
    }

    fn get_usage_stats(&self) -> MemoryUsage {
        let allocated = self
            .allocated_count
            .load(std::sync::atomic::Ordering::Relaxed);
        let hits = self.hit_count.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.miss_count.load(std::sync::atomic::Ordering::Relaxed);

        let total_requests = hits + misses;
        let cache_hit_rate = if total_requests > 0 {
            hits as f32 / total_requests as f32
        } else {
            0.0
        };

        MemoryUsage {
            allocatedbytes: allocated * std::mem::size_of::<Vec<f32>>(),
            usedbytes: {
                let vector_pool = self.vector_pool.lock().unwrap();
                let result_pool = self.result_pool.lock().unwrap();
                (vector_pool.len() + result_pool.len()) * std::mem::size_of::<Vec<f32>>()
            },
            cache_hit_rate,
        }
    }
}

impl Default for BatchStatistics {
    fn default() -> Self {
        Self {
            total_batches: 0,
            avg_batch_size: 0.0,
            avg_processing_time_us: 0.0,
            throughput_ops_per_sec: 0.0,
            queue_length: 0,
            memory_usage: MemoryUsage {
                allocatedbytes: 0,
                usedbytes: 0,
                cache_hit_rate: 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[tokio::test]
    async fn test_batch_engine_creation() {
        let config = BatchConfig::default();
        let engine = BatchOperationEngine::new(config);

        let stats = engine.get_statistics();
        assert_eq!(stats.total_batches, 0);
        assert_eq!(stats.queue_length, 0)
    }

    #[tokio::test]
    async fn test_batch_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = BatchConfig::default();
        let engine = BatchOperationEngine::new(config);

        let queries = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let targets = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let results = engine
            .batch_search(&queries, 2, &DistanceMetric::Cosine, &targets)
            .await?;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_memory_pool() {
        let config = BatchConfig::default();
        let pool = MemoryPool::new(config);

        // 获取向量
        let vector1 = pool.get_vector(100);
        assert_eq!(vector1.capacity(), 100);

        // 归还向量
        pool.return_vector(vector1);

        // 再次获取应该复用
        let vector2 = pool.get_vector(50);
        assert!(vector2.capacity() >= 50);

        let stats = pool.get_usage_stats();
        assert!(stats.cache_hit_rate > 0.0)
    }

    #[test]
    fn test_adaptive_batch_size() {
        let config = BatchConfig::default();
        let engine = BatchOperationEngine::new(config);

        let size1 = engine.calculate_adaptive_batch_size(100);
        let size2 = engine.calculate_adaptive_batch_size(2000);

        assert!(size1 <= 100);
        assert!(size2 <= 2000);
        assert!(size1 > 0);
        assert!(size2 > 0)
    }
}
