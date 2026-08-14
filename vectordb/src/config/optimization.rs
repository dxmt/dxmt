use serde::{Deserialize, Serialize};

/// 性能优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct OptimizationConfig {
    /// 查询优化设置
    pub query: QueryOptimizationConfig,
    /// 内存优化设置
    pub memory: MemoryOptimizationConfig,
    /// 索引优化设置
    pub index: IndexOptimizationConfig,
    /// SIMD优化设置
    pub simd: SimdOptimizationConfig,
}

/// 查询优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptimizationConfig {
    /// 启用批量查询
    pub enable_batch_search: bool,
    /// 批量查询的最大批次大小
    pub max_batch_size: usize,
    /// 启用查询结果缓存
    pub enable_result_cache: bool,
    /// 缓存TTL（秒）
    pub cache_ttl_seconds: u64,
    /// 启用查询预热
    pub enable_warmup: bool,
    /// 并行查询线程数
    pub query_parallelism: usize,
}

/// 内存优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// 启用内存池
    pub enable_memory_pool: bool,
    /// 内存池大小（每种类型的缓冲区数量）
    pub pool_size: usize,
    /// 启用向量对齐
    pub enable_vector_alignment: bool,
    /// 对齐边界（字节）
    pub alignment_boundary: usize,
    /// 启用大页内存
    pub enable_huge_pages: bool,
    /// 预分配内存大小（MB）
    pub preallocate_mb: usize,
}

/// 索引优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexOptimizationConfig {
    /// 默认使用优化的HNSW索引
    pub use_optimized_hnsw: bool,
    /// HNSW构建时的连接数
    pub hnsw_max_connections: usize,
    /// HNSW构建时的候选集大小
    pub hnsw_ef_construction: usize,
    /// 启用索引预构建
    pub enable_index_prebuilding: bool,
    /// 启用增量索引更新
    pub enable_incremental_updates: bool,
    /// 索引重建阈值（删除比例）
    pub rebuild_threshold: f32,
}

/// SIMD优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdOptimizationConfig {
    /// 启用SIMD加速
    pub enable_simd: bool,
    /// 强制使用特定指令集
    pub force_instruction_set: Option<String>,
    /// 启用AVX-512（如果可用）
    pub enable_avx512: bool,
    /// SIMD批处理大小
    pub simd_batch_size: usize,
    /// 启用向量归一化优化
    pub enable_normalized_vectors: bool,
}


impl Default for QueryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_batch_search: true,
            max_batch_size: 1000,
            enable_result_cache: true,
            cache_ttl_seconds: 300,
            enable_warmup: true,
            query_parallelism: num_cpus::get(),
        }
    }
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_memory_pool: true,
            pool_size: num_cpus::get() * 4,
            enable_vector_alignment: true,
            alignment_boundary: 32,
            enable_huge_pages: false,
            preallocate_mb: 1024,
        }
    }
}

impl Default for IndexOptimizationConfig {
    fn default() -> Self {
        Self {
            use_optimized_hnsw: true,
            hnsw_max_connections: 48,
            hnsw_ef_construction: 500,
            enable_index_prebuilding: true,
            enable_incremental_updates: true,
            rebuild_threshold: 0.2,
        }
    }
}

impl Default for SimdOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            force_instruction_set: None,
            enable_avx512: false,
            simd_batch_size: 64,
            enable_normalized_vectors: true,
        }
    }
}

impl OptimizationConfig {
    /// 加载优化配置
    pub fn load() -> Self {
        // 可以从环境变量或配置文件加载
        // 这里先返回默认配置
        Self::default()
    }

    /// 创建高性能配置
    pub fn high_performance() -> Self {
        Self {
            query: QueryOptimizationConfig {
                enable_batch_search: true,
                max_batch_size: 10000,
                enable_result_cache: true,
                cache_ttl_seconds: 600,
                enable_warmup: true,
                query_parallelism: num_cpus::get() * 2,
            },
            memory: MemoryOptimizationConfig {
                enable_memory_pool: true,
                pool_size: num_cpus::get() * 8,
                enable_vector_alignment: true,
                alignment_boundary: 64,
                enable_huge_pages: true,
                preallocate_mb: 4096,
            },
            index: IndexOptimizationConfig {
                use_optimized_hnsw: true,
                hnsw_max_connections: 64,
                hnsw_ef_construction: 800,
                enable_index_prebuilding: true,
                enable_incremental_updates: true,
                rebuild_threshold: 0.1,
            },
            simd: SimdOptimizationConfig {
                enable_simd: true,
                force_instruction_set: None,
                enable_avx512: true,
                simd_batch_size: 128,
                enable_normalized_vectors: true,
            },
        }
    }

    /// 创建平衡配置
    pub fn balanced() -> Self {
        Self::default()
    }

    /// 创建低延迟配置
    pub fn low_latency() -> Self {
        Self {
            query: QueryOptimizationConfig {
                enable_batch_search: false,
                max_batch_size: 100,
                enable_result_cache: true,
                cache_ttl_seconds: 60,
                enable_warmup: true,
                query_parallelism: num_cpus::get() / 2,
            },
            memory: MemoryOptimizationConfig {
                enable_memory_pool: true,
                pool_size: num_cpus::get() * 2,
                enable_vector_alignment: true,
                alignment_boundary: 32,
                enable_huge_pages: false,
                preallocate_mb: 512,
            },
            index: IndexOptimizationConfig {
                use_optimized_hnsw: true,
                hnsw_max_connections: 32,
                hnsw_ef_construction: 200,
                enable_index_prebuilding: false,
                enable_incremental_updates: true,
                rebuild_threshold: 0.3,
            },
            simd: SimdOptimizationConfig {
                enable_simd: true,
                force_instruction_set: None,
                enable_avx512: false,
                simd_batch_size: 32,
                enable_normalized_vectors: false,
            },
        }
    }
}
