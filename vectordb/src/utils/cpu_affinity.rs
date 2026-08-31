use crate::config::performance_optimization::PerformanceConfig;

#[cfg(target_os = "linux")]
use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};

/// CPU亲和性管理器
pub struct CpuAffinityManager {
    config: PerformanceConfig,
}

impl CpuAffinityManager {
    pub fn new(config: PerformanceConfig) -> Self {
        Self { config }
    }

    /// 为查询线程设置CPU亲和性
    pub fn set_query_thread_affinity(&self, thread_id: usize) {
        let core_ids = &self.config.cpu_allocation.core_mapping.query_core_ids;
        if core_ids.is_empty() {
            return;
        }

        let core_id = core_ids[thread_id % core_ids.len()];
        self.set_thread_affinity(core_id);
    }

    /// 为索引线程设置CPU亲和性
    pub fn set_index_thread_affinity(&self, thread_id: usize) {
        let core_ids = &self.config.cpu_allocation.core_mapping.index_core_ids;
        if core_ids.is_empty() {
            return;
        }

        let core_id = core_ids[thread_id % core_ids.len()];
        self.set_thread_affinity(core_id);
    }

    /// 为系统IO线程设置CPU亲和性
    pub fn set_system_thread_affinity(&self, thread_id: usize) {
        let core_ids = &self.config.cpu_allocation.core_mapping.system_core_ids;
        if core_ids.is_empty() {
            return;
        }

        let core_id = core_ids[thread_id % core_ids.len()];
        self.set_thread_affinity(core_id);
    }

    /// 设置当前线程的CPU亲和性
    #[cfg(target_os = "linux")]
    fn set_thread_affinity(&self, core_id: usize) {
        unsafe {
            let mut cpu_set = std::mem::zeroed::<cpu_set_t>();
            CPU_ZERO(&mut cpu_set);
            CPU_SET(core_id, &mut cpu_set);

            let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &cpu_set);
            if result != 0 {
                tracing::warn!(
                    "Failed to set CPU affinity for core {}: {}",
                    core_id,
                    std::io::Error::last_os_error()
                )
            } else {
                tracing::debug!("Thread bound to CPU core {}", core_id)
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn set_thread_affinity(&self, core_id: usize) {
        tracing::debug!(
            "CPU affinity not supported on this platform, would bind to core {}",
            core_id
        );
    }

    /// 创建优化的线程池构建器
    pub fn create_query_thread_pool(&self) -> rayon::ThreadPoolBuilder {
        let pool_size = self.config.thread_affinity.thread_pools.query_pool_size;
        let affinity_manager = self.clone();

        rayon::ThreadPoolBuilder::new()
            .num_threads(pool_size)
            .thread_name(|idx| format!("query-worker-{}", idx))
            .start_handler(move |idx| affinity_manager.set_query_thread_affinity(idx))
    }

    /// 创建索引线程池
    pub fn create_index_thread_pool(&self) -> rayon::ThreadPoolBuilder {
        let pool_size = self.config.thread_affinity.thread_pools.index_pool_size;
        let affinity_manager = self.clone();

        rayon::ThreadPoolBuilder::new()
            .num_threads(pool_size)
            .thread_name(|idx| format!("index-worker-{}", idx))
            .start_handler(move |idx| affinity_manager.set_index_thread_affinity(idx))
    }

    /// 创建IO线程池
    pub fn create_io_thread_pool(&self) -> rayon::ThreadPoolBuilder {
        let pool_size = self.config.thread_affinity.thread_pools.io_pool_size;
        let affinity_manager = self.clone();

        rayon::ThreadPoolBuilder::new()
            .num_threads(pool_size)
            .thread_name(|idx| format!("io-worker-{}", idx))
            .start_handler(move |idx| affinity_manager.set_system_thread_affinity(idx))
    }
}

impl Clone for CpuAffinityManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
        }
    }
}

/// NUMA节点信息
#[derive(Debug, Clone)]
pub struct NumaInfo {
    pub nodeid: usize,
    pub cpu_cores: Vec<usize>,
    pub memory_mb: usize,
}

/// NUMA感知的内存分配器
pub struct NumaAwareAllocator {
    numa_nodes: Vec<NumaInfo>,
}

impl NumaAwareAllocator {
    #[cfg(target_os = "linux")]
    pub fn new() -> crate::Result<Self> {
        // 简化的NUMA检测，实际实现需要读取/sys/devices/system/node/
        let numa_nodes = vec![
            NumaInfo {
                nodeid: 0,
                cpu_cores: vec![0, 1, 2, 3],
                memory_mb: 32768, // 32GB
            },
            NumaInfo {
                nodeid: 1,
                cpu_cores: vec![4, 5, 6, 7],
                memory_mb: 32768, // 32GB
            },
        ];

        Ok(Self { numa_nodes })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn new() -> crate::Result<Self> {
        // 非Linux系统假设单NUMA节点
        let numa_nodes = vec![NumaInfo {
            nodeid: 0,
            cpu_cores: (0..8).collect(),
            memory_mb: 65536, // 64GB
        }];

        Ok(Self { numa_nodes })
    }

    /// 获取指定CPU核心所在的NUMA节点
    pub fn get_numa_node_for_core(&self, core_id: usize) -> Option<usize> {
        for node in &self.numa_nodes {
            if node.cpu_cores.contains(&core_id) {
                return Some(node.nodeid);
            }
        }
        None
    }

    /// 为线程分配本地NUMA内存
    pub fn allocate_numa_local(&self, size: usize) -> crate::Result<Vec<u8>> {
        // 简化实现，实际应该使用numa_alloc_local等系统调用
        let vec = vec![0; size];
        Ok(vec)
    }
}

/// 性能计数器
pub struct PerformanceCounters {
    /// 向量操作计数
    pub vector_ops: std::sync::atomic::AtomicU64,
    /// 缓存命中计数
    pub cachehits: std::sync::atomic::AtomicU64,
    /// 缓存未命中计数
    pub cache_misses: std::sync::atomic::AtomicU64,
}

impl Default for PerformanceCounters {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceCounters {
    pub fn new() -> Self {
        Self {
            vector_ops: std::sync::atomic::AtomicU64::new(0),
            cachehits: std::sync::atomic::AtomicU64::new(0),
            cache_misses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn inc_vector_ops(&self, count: u64) {
        self.vector_ops
            .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn inc_cache_hit(&self) {
        self.cachehits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn inc_cache_miss(&self) {
        self.cache_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cachehits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_counters() {
        let counters = PerformanceCounters::new();

        counters.inc_vector_ops(100);
        assert_eq!(
            counters
                .vector_ops
                .load(std::sync::atomic::Ordering::Relaxed),
            100
        );

        counters.inc_cache_hit();
        counters.inc_cache_hit();
        counters.inc_cache_miss();

        let hit_rate = counters.cache_hit_rate();
        assert!((hit_rate - 0.666).abs() < 0.01)
    }

    #[test]
    fn test_numa_allocator() -> Result<(), Box<dyn std::error::Error>> {
        let allocator = NumaAwareAllocator::new()?;

        // 测试核心到NUMA节点的映射
        if let Some(node) = allocator.get_numa_node_for_core(0) {
            assert_eq!(node, 0)
        }

        if let Some(node) = allocator.get_numa_node_for_core(7) {
            assert!(node == 0 || node == 1); // 取决于系统配置
        }
        Ok(())
    }
}
