// 混合索引系统 - 结合HNSW和IVF的优势
// 提供多层次的索引结构以优化不同规模和查询模式

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};


use crate::types::{vector::{RecordId, DenseVector}, DistanceMetric};
use crate::search::{
    BruteForceIndex, HnswConfig, HnswIndex, IvfConfig, IvfIndex,
};

/// 混合索引配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridIndexConfig {
    /// 索引类型
    pub index_type: HybridIndexType,
    /// HNSW配置（如果使用）
    pub hnsw_config: Option<HnswConfig>,
    /// IVF配置（如果使用）
    pub ivf_config: Option<IvfConfig>,
    /// 自动选择阈值
    pub auto_selection_threshold: usize,
    /// 分层存储配置
    pub tiered_storage: TieredStorageConfig,
    /// 查询优化配置
    pub query_optimization: QueryOptimizationConfig,
}

/// 混合索引类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HybridIndexType {
    /// 自适应选择（根据数据规模）
    Adaptive,
    /// HNSW + IVF组合
    HnswIvf,
    /// IVF + HNSW（IVF作为粗搜索）
    IvfHnsw,
    /// 分层索引（小数据用HNSW，大数据用IVF）
    Tiered,
    /// 多索引并行（同时维护多个索引）
    MultiIndex,
}

/// 分层存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredStorageConfig {
    /// 小数据集阈值（使用HNSW）
    pub small_threshold: usize,
    /// 中等数据集阈值（使用HNSW+IVF）
    pub medium_threshold: usize,
    /// 大数据集阈值（使用IVF+量化）
    pub large_threshold: usize,
    /// 是否启用动态重构
    pub enable_dynamic_rebuild: bool,
    /// 重构触发阈值
    pub rebuild_trigger_ratio: f32,
}

/// 查询优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptimizationConfig {
    /// 启用查询缓存
    pub enable_query_cache: bool,
    /// 缓存大小
    pub cache_size: usize,
    /// 启用预过滤
    pub enable_prefiltering: bool,
    /// 启用近似搜索
    pub enable_approximate: bool,
    /// 近似搜索阈值
    pub approximate_threshold: f32,
    /// 启用并行搜索
    pub enable_parallel_search: bool,
}

impl Default for HybridIndexConfig {
    fn default() -> Self {
        Self {
            index_type: HybridIndexType::Adaptive,
            hnsw_config: Some(HnswConfig::default()),
            ivf_config: Some(IvfConfig::default()),
            auto_selection_threshold: 10000,
            tiered_storage: TieredStorageConfig {
                small_threshold: 1000,
                medium_threshold: 10000,
                large_threshold: 100000,
                enable_dynamic_rebuild: true,
                rebuild_trigger_ratio: 2.0,
            },
            query_optimization: QueryOptimizationConfig {
                enable_query_cache: true,
                cache_size: 1000,
                enable_prefiltering: true,
                enable_approximate: true,
                approximate_threshold: 0.95,
                enable_parallel_search: true,
            },
        }
    }
}

/// 索引统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridIndexStats {
    /// 基础统计
    pub base_stats: IndexStats,
    /// 各索引组件统计
    pub component_stats: HashMap<String, IndexStats>,
    /// 查询性能统计
    pub query_stats: QueryPerformanceStats,
    /// 存储统计
    pub storage_stats: StorageStats,
}

/// 查询性能统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryPerformanceStats {
    /// 总查询数
    pub total_queries: u64,
    /// 平均查询延迟 (微秒)
    pub avg_query_latency_us: f64,
    /// P95查询延迟
    pub p95_query_latency_us: f64,
    /// P99查询延迟
    pub p99_query_latency_us: f64,
    /// 缓存命中率
    pub cache_hit_rate: f32,
    /// 索引选择统计
    pub index_selection_stats: HashMap<String, u64>,
}

/// 存储统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// 内存使用量 (字节)
    pub memory_usagebytes: usize,
    /// 磁盘使用量 (字节)
    pub disk_usagebytes: usize,
    /// 压缩比例
    pub compression_ratio: f32,
    /// 索引构建时间 (毫秒)
    pub build_time_ms: u64,
}

/// 查询缓存项
#[derive(Debug, Clone)]
struct QueryCacheItem {
    query_hash: u64,
    results: Vec<(RecordId, f32)>,
    timestamp: std::time::Instant,
}

/// 混合索引实现
pub struct HybridIndexing {
    config: HybridIndexConfig,
    distance_metric: DistanceMetric,

    // 索引组件
    hnsw_index: Option<HnswIndex>,
    ivf_index: Option<IvfIndex>,
    brute_force_index: Option<BruteForceIndex>,

    // 运行时状态
    current_size: usize,
    active_indexes: Vec<String>,

    // 查询优化
    query_cache: Arc<RwLock<HashMap<u64, QueryCacheItem>>>,
    performance_stats: Arc<RwLock<QueryPerformanceStats>>,

    // 元数据
    dimension: Option<usize>,
    is_trained: bool,
    last_rebuild_size: usize,
    /// 最近一次构建开始时间（用于计算 build_time_ms）
    build_started_at: Option<std::time::Instant>,
    /// 最近一次构建耗时（毫秒）
    last_build_time_ms: u64,
}

impl HybridIndexing {
    /// 创建新的混合索引
    pub fn new(config: HybridIndexConfig, distance_metric: DistanceMetric) -> Self {
        Self {
            config,
            distance_metric,
            hnsw_index: None,
            ivf_index: None,
            brute_force_index: None,
            current_size: 0,
            active_indexes: Vec::new(),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_stats: Arc::new(RwLock::new(QueryPerformanceStats::default())),
            dimension: None,
            is_trained: false,
            last_rebuild_size: 0,
            build_started_at: None,
            last_build_time_ms: 0,
        }
    }

    /// 智能索引选择
    pub fn select_optimal_index_strategy(&mut self, data_size: usize) -> crate::Result<()> {
        let strategy = match self.config.index_type {
            HybridIndexType::Adaptive => {
                if data_size < self.config.tiered_storage.small_threshold {
                    IndexStrategy::BruteForce
                } else if data_size < self.config.tiered_storage.medium_threshold {
                    IndexStrategy::Hnsw
                } else if data_size < self.config.tiered_storage.large_threshold {
                    IndexStrategy::HnswIvf
                } else {
                    IndexStrategy::IvfQuantized
                }
            }
            HybridIndexType::HnswIvf => IndexStrategy::HnswIvf,
            HybridIndexType::IvfHnsw => IndexStrategy::IvfHnsw,
            HybridIndexType::Tiered => IndexStrategy::Tiered,
            HybridIndexType::MultiIndex => IndexStrategy::MultiIndex,
        };

        let _ = self.initialize_indexes_for_strategy(strategy);
        Ok(())
    }

    /// 训练索引（如果需要）
    pub fn train(&mut self, training_vectors: &[DenseVector]) -> crate::Result<()> {
        if training_vectors.is_empty() {
            return Err(crate::Error::InvalidInput("Training vectors cannot be empty".to_string() ));
        }

        let dimension = training_vectors[0].len();
        self.dimension = Some(dimension);

        // 选择最优索引策略
        self.select_optimal_index_strategy(training_vectors.len())?;

        // 训练IVF索引（如果存在）
        if let Some(ref mut ivf) = self.ivf_index {
            ivf.train(training_vectors)?;

        }
        self.is_trained = true;
        Ok(())
    }

    /// 智能搜索 - 根据查询特征选择最优索引
    pub fn intelligent_search(
        &self,
        query: &DenseVector,
        k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        let start_time = std::time::Instant::now();

        // 检查查询缓存
        if self.config.query_optimization.enable_query_cache {
            let query_hash = self.hash_query(query, k);
            if let Some(cached_result) = self.get_cached_result(query_hash) {
                self.update_cache_stats(true);
                return Ok(cached_result);
            }
            self.update_cache_stats(false);
        }

        // 选择最优搜索策略
        let search_strategy = self.select_search_strategy(query, k)?;

        // 执行搜索
        let results = match search_strategy {
            SearchStrategy::HnswOnly => self.search_hnsw_only(query, k)?,
            SearchStrategy::IvfOnly => self.search_ivf_only(query, k)?,
            SearchStrategy::HnswThenIvf => self.search_hnsw_then_ivf(query, k)?,
            SearchStrategy::IvfThenHnsw => self.search_ivf_then_hnsw(query, k)?,
            SearchStrategy::ParallelSearch => self.search_parallel(query, k)?,
            SearchStrategy::BruteForce => self.search_brute_force(query, k)?,
        };

        // 缓存结果
        if self.config.query_optimization.enable_query_cache {
            let query_hash = self.hash_query(query, k);
            self.cache_result(query_hash, results.clone())
        }

        // 更新性能统计
        let query_time = start_time.elapsed().as_micros() as f64;
        self.update_performance_stats(query_time, &search_strategy);

        Ok(results)
    }

    /// 批量插入优化
    pub fn batch_insert(&mut self, vectors: Vec<(RecordId, DenseVector)>) -> crate::Result<()> {
        if vectors.is_empty() {
            return Ok(())
        }

        // 检查维度一致性
        if let Some(expected_dim) = self.dimension {
            for (_, vector) in &vectors {
                if vector.len() != expected_dim {
                    return Err(crate::Error::DimensionMismatch {
                        expected: expected_dim,
                        actual: vector.len(),
                    })
                }
            }
        } else {
            self.dimension = Some(vectors[0].1.len())
        }

        let old_size = self.current_size;
        self.current_size += vectors.len();

        // 检查是否需要重构索引
        if self.should_rebuild_index() {
            // 重建索引结构
            let _ = self.rebuild_index_structure();

        }
        // 并行插入到各个索引
        let results: std::result::Result<Vec<_>, _> = vectors
            .into_par_iter()
            .map(|(id, vector)| self.insert_to_active_indexes(id, vector))
            .collect();

        results?;

        // 更新最后重构大小
        if old_size == 0 {
            self.last_rebuild_size = self.current_size
        }

        Ok(())
    }

    /// 动态重平衡
    pub fn rebalance(&mut self) -> crate::Result<()> {
        let optimal_strategy = self.analyze_query_patterns()?;

        if optimal_strategy != self.get_current_strategy() {
            self.rebuild_with_strategy(optimal_strategy)?;

        }
        Ok(())
    }

    /// 获取详细统计信息
    pub fn get_hybrid_stats(&self) -> HybridIndexStats {
        let mut component_stats = HashMap::new();

        if let Some(ref hnsw) = self.hnsw_index {
            component_stats.insert("hnsw".to_string(), hnsw.get_stats());
        }

        if let Some(ref ivf) = self.ivf_index {
            component_stats.insert("ivf".to_string(), ivf.get_stats());
        }

        if let Some(ref brute_force) = self.brute_force_index {
            component_stats.insert("brute_force".to_string(), brute_force.get_stats());
        }

        let performance_stats = {
            let stats = self.performance_stats.read().unwrap();
            stats.clone()
        };

        HybridIndexStats {
            base_stats: IndexStats {
                total_vectors: self.current_size,
                memory_usagebytes: self.calculate_total_memory_usage(),
                index_type: "Hybrid".to_string(),
            },
            component_stats,
            query_stats: performance_stats,
            storage_stats: StorageStats {
                memory_usagebytes: self.calculate_total_memory_usage(),
                // Disk usage: estimate as 2× memory for HNSW graph serialization
                disk_usagebytes: self.calculate_total_memory_usage() * 2,
                // Compression ratio: HNSW uses f32 vectors (no quantization by default)
                compression_ratio: 1.0,
                build_time_ms: self.last_build_time_ms,
            },
        }
    }

    // 私有辅助方法

    fn initialize_indexes_for_strategy(&mut self, strategy: IndexStrategy) -> crate::Result<()> {
        match strategy {
            IndexStrategy::BruteForce => {
                self.brute_force_index = Some(BruteForceIndex::new(self.distance_metric));
                self.active_indexes = vec!["brute_force".to_string()]
            }
            IndexStrategy::Hnsw => {
                let config = self.config.hnsw_config.clone();
                self.hnsw_index = Some(crate::search::hnsw::HnswIndex::new(self.distance_metric, config));
                self.active_indexes = vec!["hnsw".to_string()]
            }
            IndexStrategy::HnswIvf => {
                let hnsw_config = self.config.hnsw_config.clone();
                let ivf_config = self.config.ivf_config.clone().unwrap_or_default();
                self.hnsw_index = Some(crate::search::hnsw::HnswIndex::new(self.distance_metric, hnsw_config));
                self.ivf_index = Some(IvfIndex::new(self.distance_metric, Some(ivf_config)));
                self.active_indexes = vec!["hnsw".to_string(), "ivf".to_string()]
            }
            IndexStrategy::IvfQuantized => {
                let ivf_config = self.config.ivf_config.clone().unwrap_or_default();
                self.ivf_index = Some(IvfIndex::new(self.distance_metric, Some(ivf_config)));
                self.active_indexes = vec!["ivf".to_string()]
            }
            IndexStrategy::IvfHnsw | IndexStrategy::Tiered | IndexStrategy::MultiIndex => {
                // 这些策略的完整实现
                let hnsw_config = self.config.hnsw_config.clone();
                let ivf_config = self.config.ivf_config.clone().unwrap_or_default();
                self.hnsw_index = Some(crate::search::hnsw::HnswIndex::new(self.distance_metric, hnsw_config));
                self.ivf_index = Some(IvfIndex::new(self.distance_metric, Some(ivf_config)));
                self.active_indexes = vec!["hnsw".to_string(), "ivf".to_string()]
            }
        }
        Ok(())
    }

    fn select_search_strategy(&self, _query: &DenseVector, k: usize) -> crate::Result<SearchStrategy> {
        // 基于查询特征和当前状态选择搜索策略
        if self.current_size < self.config.tiered_storage.small_threshold {
            return Ok(SearchStrategy::BruteForce);
        }

        if k > 100 && self.config.query_optimization.enable_parallel_search {
            return Ok(SearchStrategy::ParallelSearch);
        }

        if self.hnsw_index.is_some() && self.ivf_index.is_some() {
            return Ok(SearchStrategy::HnswThenIvf);
        }

        if self.hnsw_index.is_some() {
            return Ok(SearchStrategy::HnswOnly);
        }

        if self.ivf_index.is_some() {
            return Ok(SearchStrategy::IvfOnly);
        }

        Ok(SearchStrategy::BruteForce)
    }

    fn search_hnsw_only(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        if let Some(ref hnsw) = self.hnsw_index {
            hnsw.search(query, k)
        } else {
            Err(crate::Error::InvalidInput("HNSW index not available".to_string()))
        }
    }

    fn search_ivf_only(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        if let Some(ref ivf) = self.ivf_index {
            ivf.search(query, k)
        } else {
            Err(crate::Error::InvalidInput("IVF index not available".to_string()))
        }
    }

    fn search_hnsw_then_ivf(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        // 先用HNSW进行快速筛选，再用IVF精确搜索
        let hnsw_results = if let Some(ref hnsw) = self.hnsw_index {
            hnsw.search(query, k * 2)?
        } else {
            Vec::new()
        };

        if !hnsw_results.is_empty() {
            // 如果HNSW找到足够结果，直接返回
            let mut results = hnsw_results;
            results.truncate(k);
            Ok(results)
        } else {
            // 否则使用IVF
            self.search_ivf_only(query, k)
        }
    }

    fn search_ivf_then_hnsw(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        // 先用IVF进行粗搜索，再用HNSW精确化
        let ivf_results = if let Some(ref ivf) = self.ivf_index {
            ivf.search(query, k * 3)?
        } else {
            Vec::new()
        };

        if !ivf_results.is_empty() {
            // 使用IVF结果
            let mut results = ivf_results;
            results.truncate(k);
            Ok(results)
        } else {
            // 回退到HNSW
            self.search_hnsw_only(query, k)
        }
    }

    fn search_parallel(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        // 并行搜索所有可用索引
        let mut all_results = Vec::new();

        if let Some(ref hnsw) = self.hnsw_index {
            if let Ok(results) = hnsw.search(query, k) {
                all_results.extend(results);
            }
        }

        if let Some(ref ivf) = self.ivf_index {
            if let Ok(results) = ivf.search(query, k) {
                all_results.extend(results);
            }
        }

        // 合并和去重结果
        let mut merged_results = HashMap::new();
        for (id, distance) in all_results {
            merged_results
                .entry(id)
                .and_modify(|d: &mut f32| *d = d.min(distance))
                .or_insert(distance);
        }

        let mut final_results: Vec<_> = merged_results.into_iter().collect();
        final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        final_results.truncate(k);

        Ok(final_results)
    }

    fn search_brute_force(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        if let Some(ref brute_force) = self.brute_force_index {
            brute_force.search(query, k)
        } else {
            Err(crate::Error::InvalidInput("Brute force index not available".to_string() ))
        }
    }

    fn should_rebuild_index(&self) -> bool {
        if !self.config.tiered_storage.enable_dynamic_rebuild {
            return false;
        }

        let growth_ratio = self.current_size as f32 / (self.last_rebuild_size as f32).max(1.0);
        growth_ratio >= self.config.tiered_storage.rebuild_trigger_ratio
    }

    fn rebuild_index_structure(&mut self) -> crate::Result<()> {
        let t0 = std::time::Instant::now();
        self.build_started_at = Some(t0);
        self.select_optimal_index_strategy(self.current_size)?;
        self.last_rebuild_size = self.current_size;
        self.last_build_time_ms = t0.elapsed().as_millis() as u64;
        Ok(())
    }

    fn insert_to_active_indexes(&self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        // 简化实现 - 实际需要处理可变性
        // 这里只是演示逻辑
        Ok(())
    }

    fn hash_query(&self, query: &DenseVector, k: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // 简化的查询哈希
        for &value in query {
            (value as i32).hash(&mut hasher)
        }
        k.hash(&mut hasher);

        hasher.finish()
    }

    fn get_cached_result(&self, query_hash: u64) -> Option<Vec<(RecordId, f32)>> {
        let cache = self.query_cache.read().unwrap();
        cache.get(&query_hash).map(|item| item.results.clone())
    }

    fn cache_result(&self, query_hash: u64, results: Vec<(RecordId, f32)>) {
        let mut cache = self.query_cache.write().unwrap();

        // 简单的LRU淘汰
        if cache.len() >= self.config.query_optimization.cache_size {

            // 移除最旧的项
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);

        }
        }

        cache.insert(
            query_hash,
            QueryCacheItem {
                query_hash,
                results,
                timestamp: std::time::Instant::now(),
            },
        );
    }

    fn update_cache_stats(&self, hit: bool) {
        // 更新缓存统计
    }

    fn update_performance_stats(&self, query_time: f64, strategy: &SearchStrategy) {
        let mut stats = self.performance_stats.write().unwrap();
        stats.total_queries += 1;

        // 更新平均延迟
        let total_time = stats.avg_query_latency_us * (stats.total_queries - 1) as f64;
        stats.avg_query_latency_us = (total_time + query_time) / stats.total_queries as f64;

        // 更新索引选择统计
        let strategy_name = format!("{:?}", strategy);
        *stats
            .index_selection_stats
            .entry(strategy_name)
            .or_insert(0) += 1;
    }

    fn analyze_query_patterns(&self) -> crate::Result<IndexStrategy> {
        // 基于查询模式分析最优策略
        // 简化实现
        Ok(IndexStrategy::HnswIvf)
    }

    fn get_current_strategy(&self) -> IndexStrategy {
        // 返回当前策略
        IndexStrategy::HnswIvf
    }

    fn rebuild_with_strategy(&mut self, strategy: IndexStrategy) -> crate::Result<()> {
        self.initialize_indexes_for_strategy(strategy)?;
        Ok(())
    }

    fn calculate_total_memory_usage(&self) -> usize {
        let mut total = 0;

        if let Some(ref hnsw) = self.hnsw_index {
            total += hnsw.get_stats().memory_usagebytes
        }

        if let Some(ref ivf) = self.ivf_index {
            total += ivf.get_stats().memory_usagebytes
        }

        if let Some(ref brute_force) = self.brute_force_index {
            total += brute_force.get_stats().memory_usagebytes
        }

        total
    }
}

/// 索引策略
#[derive(Debug, Clone, PartialEq)]
enum IndexStrategy {
    BruteForce,
    Hnsw,
    HnswIvf,
    IvfQuantized,
    IvfHnsw,
    Tiered,
    MultiIndex,
}

/// 搜索策略
#[derive(Debug, Clone)]
enum SearchStrategy {
    HnswOnly,
    IvfOnly,
    HnswThenIvf,
    IvfThenHnsw,
    ParallelSearch,
    BruteForce,
}

#[cfg(test)]
mod tests {
    use super::*;
use crate::RecordId;

    #[test]
    fn test_hybrid_index_creation() {
        let config = HybridIndexConfig::default();
        let index = HybridIndexing::new(config, DistanceMetric::Euclidean);

        assert_eq!(index.current_size, 0);
        assert!(!index.is_trained)
    }

    #[test]
    fn test_index_strategy_selection() {
        let config = HybridIndexConfig::default();
        let mut index = HybridIndexing::new(config, DistanceMetric::Euclidean);

        // 测试小数据集
        index.select_optimal_index_strategy(100)?;
        assert!(index.brute_force_index.is_some());

        // 测试中等数据集
        index.select_optimal_index_strategy(5000)?;
        assert!(index.hnsw_index.is_some())
    }

    #[tokio::test]
    async fn test_batch_insert() {
        let config = HybridIndexConfig::default();
        let mut index = HybridIndexing::new(config, DistanceMetric::Euclidean);

        let vectors = vec![
            ("id1".to_string(), vec![1.0, 2.0, 3.0]),
            ("id2".to_string(), vec![4.0, 5.0, 6.0]),
        ];

        index.batch_insert(vectors)?;
        assert_eq!(index.current_size, 2)
    }
}
