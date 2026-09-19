use serde::{Deserialize, Serialize};
use std::thread;

/// AMD Turin 8核心性能优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// CPU核心分配配置
    pub cpu_allocation: CpuAllocation,
    /// 内存层级配置
    pub memory_tiers: MemoryTiers,
    /// SIMD优化配置
    pub simd_optimization: SimdOptimization,
    /// 线程亲和性配置
    pub thread_affinity: ThreadAffinityConfig,
}

/// CPU核心分配
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuAllocation {
    /// 查询处理核心数
    pub query_cores: usize,
    /// 索引维护核心数
    pub index_cores: usize,
    /// 系统和网络IO核心数
    pub system_cores: usize,
    /// 核心ID映射
    pub core_mapping: CoreMapping,
}

/// 核心ID映射
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMapping {
    /// 查询处理核心ID列表
    pub query_core_ids: Vec<usize>,
    /// 索引维护核心ID列表
    pub index_core_ids: Vec<usize>,
    /// 系统IO核心ID列表
    pub system_core_ids: Vec<usize>,
}

/// 内存层级配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTiers {
    /// 向量缓存大小 (GB)
    pub vector_cache_gb: usize,
    /// 索引内存大小 (GB)
    pub index_memory_gb: usize,
    /// 系统预留内存 (GB)
    pub system_memory_gb: usize,
    /// 内存大页配置
    pub huge_pages: HugePagesConfig,
}

/// 大页内存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HugePagesConfig {
    /// 是否启用大页
    pub enabled: bool,
    /// 大页大小 (MB)
    pub page_size_mb: usize,
    /// 预分配大页数量
    pub preallocate_pages: usize,
}

/// SIMD优化配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimdOptimization {
    /// 是否启用AVX2
    pub enable_avx2: bool,
    /// 是否启用AVX512 (Turin支持)
    pub enable_avx512: bool,
    /// 向量批处理大小
    pub vector_batch_size: usize,
    /// 自动向量化提示
    pub auto_vectorization: AutoVectorizationConfig,
}

/// 自动向量化配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutoVectorizationConfig {
    /// 编译器优化级别
    pub optimization_level: String,
    /// 目标CPU特性
    pub target_features: Vec<String>,
    /// 循环展开因子
    pub unroll_factor: usize,
}

/// 线程亲和性配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadAffinityConfig {
    /// 是否绑定线程到核心
    pub bind_threads: bool,
    /// NUMA节点亲和性
    pub numa_aware: bool,
    /// 线程池配置
    pub thread_pools: ThreadPoolsConfig,
}

/// 线程池配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolsConfig {
    /// 查询线程池大小
    pub query_pool_size: usize,
    /// 索引线程池大小
    pub index_pool_size: usize,
    /// IO线程池大小
    pub io_pool_size: usize,
    /// 工作窃取策略
    pub work_stealing: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            cpu_allocation: CpuAllocation {
                query_cores: 4,
                index_cores: 2,
                system_cores: 2,
                core_mapping: CoreMapping {
                    query_core_ids: vec![0, 1, 2, 3],
                    index_core_ids: vec![4, 5],
                    system_core_ids: vec![6, 7],
                },
            },
            memory_tiers: MemoryTiers {
                vector_cache_gb: 40,
                index_memory_gb: 15,
                system_memory_gb: 8,
                huge_pages: HugePagesConfig {
                    enabled: true,
                    page_size_mb: 2,
                    preallocate_pages: 20480, // 40GB in 2MB pages
                },
            },
            simd_optimization: SimdOptimization {
                enable_avx2: true,
                enable_avx512: true, // Turin supports AVX-512
                vector_batch_size: 256,
                auto_vectorization: AutoVectorizationConfig {
                    optimization_level: "3".to_string(),
                    target_features: vec![
                        "avx2".to_string(),
                        "avx512f".to_string(),
                        "avx512vl".to_string(),
                        "avx512bw".to_string(),
                        "avx512dq".to_string(),
                        "avx512vnni".to_string(),
                        "fma".to_string(),
                    ],
                    unroll_factor: 4,
                },
            },
            thread_affinity: ThreadAffinityConfig {
                bind_threads: true,
                numa_aware: true,
                thread_pools: ThreadPoolsConfig {
                    query_pool_size: 4,
                    index_pool_size: 2,
                    io_pool_size: 2,
                    work_stealing: true,
                },
            },
        }
    }
}

impl PerformanceConfig {
    /// 从环境变量加载配置
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // CPU核心分配
        if let Ok(query_cores) = crate::config::env::var("DXMT_PERF_QUERY_CORES") {
            if let Ok(cores) = query_cores.parse::<usize>() {
                config.cpu_allocation.query_cores = cores
            }
        }

        if let Ok(index_cores) = crate::config::env::var("DXMT_PERF_INDEX_CORES") {
            if let Ok(cores) = index_cores.parse::<usize>() {
                config.cpu_allocation.index_cores = cores
            }
        }

        if let Ok(system_cores) = crate::config::env::var("DXMT_PERF_SYSTEM_CORES") {
            if let Ok(cores) = system_cores.parse::<usize>() {
                config.cpu_allocation.system_cores = cores
            }
        }

        // 内存配置
        if let Ok(vector_cache) = crate::config::env::var("DXMT_PERF_VECTOR_CACHE_GB") {
            if let Ok(gb) = vector_cache.parse::<usize>() {
                config.memory_tiers.vector_cache_gb = gb
            }
        }

        if let Ok(index_memory) = crate::config::env::var("DXMT_PERF_INDEX_MEMORY_GB") {
            if let Ok(gb) = index_memory.parse::<usize>() {
                config.memory_tiers.index_memory_gb = gb
            }
        }

        // SIMD配置
        if let Ok(enable_avx512) = crate::config::env::var("DXMT_PERF_ENABLE_AVX512") {
            config.simd_optimization.enable_avx512 = enable_avx512 == "true"
        }

        if let Ok(batch_size) = crate::config::env::var("DXMT_PERF_VECTOR_BATCH_SIZE") {
            if let Ok(size) = batch_size.parse::<usize>() {
                config.simd_optimization.vector_batch_size = size
            }
        }

        config
    }

    /// 验证配置的合理性
    pub fn validate(&self) -> crate::Result<()> {
        let total_cores = self.cpu_allocation.query_cores
            + self.cpu_allocation.index_cores
            + self.cpu_allocation.system_cores;

        if total_cores > 8 {
            return Err(crate::Error::Internal(format!(
                "Total allocated cores ({}) exceeds available cores (8)",
                total_cores
            )));
        }

        let total_memory = self.memory_tiers.vector_cache_gb
            + self.memory_tiers.index_memory_gb
            + self.memory_tiers.system_memory_gb;

        if total_memory > 63 {
            return Err(crate::Error::Internal(format!(
                "Total allocated memory ({}GB) exceeds available memory (63GB)",
                total_memory
            )));
        }

        Ok(())
    }

    /// 获取可用的物理核心数
    pub fn available_cores() -> usize {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
    }

    /// 生成优化的编译器标志
    pub fn compiler_flags(&self) -> Vec<String> {
        let mut flags = vec![
            format!(
                "-C opt-level={}",
                self.simd_optimization.auto_vectorization.optimization_level
            ),
            "-C lto=fat".to_string(),
            "-C embed-bitcode=yes".to_string(),
        ];

        // 添加目标CPU特性
        if !self
            .simd_optimization
            .auto_vectorization
            .target_features
            .is_empty()
        {
            let features = self
                .simd_optimization
                .auto_vectorization
                .target_features
                .join(",");
            flags.push(format!("-C target-features=+{}", features))
        }

        // AMD Zen5特定优化
        flags.push("-C target-cpu=znver5".to_string());

        flags
    }
}

/// 性能监控指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// 每秒向量操作数
    pub vector_ops_per_sec: f64,
    /// 缓存命中率
    pub cache_hit_rate: f64,
    /// CPU利用率 (每个核心)
    pub cpu_utilization: Vec<f64>,
    /// 内存带宽使用率
    pub memory_bandwidth_gbps: f64,
    /// 查询延迟统计
    pub query_latency: LatencyStats,
}

/// 延迟统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub avg_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_performance_config() {
        let config = PerformanceConfig::default();
        assert_eq!(config.cpu_allocation.query_cores, 4);
        assert_eq!(config.cpu_allocation.index_cores, 2);
        assert_eq!(config.cpu_allocation.system_cores, 2);
        assert_eq!(config.memory_tiers.vector_cache_gb, 40);
        assert_eq!(config.memory_tiers.index_memory_gb, 15);
        assert_eq!(config.memory_tiers.system_memory_gb, 8)
    }

    #[test]
    fn test_validate_config() {
        let mut config = PerformanceConfig::default();
        assert!(config.validate().is_ok());

        // 测试核心数超限
        config.cpu_allocation.query_cores = 10;
        assert!(config.validate().is_err());

        // 测试内存超限
        config.cpu_allocation.query_cores = 4;
        config.memory_tiers.vector_cache_gb = 100;
        assert!(config.validate().is_err())
    }

    #[test]
    fn test_compiler_flags() {
        let config = PerformanceConfig::default();
        let flags = config.compiler_flags();

        assert!(flags.contains(&"-C opt-level=3".to_string()));
        assert!(flags.contains(&"-C target-cpu=znver5".to_string()));
        assert!(flags.iter().any(|f| f.contains("avx512")))
    }
}
