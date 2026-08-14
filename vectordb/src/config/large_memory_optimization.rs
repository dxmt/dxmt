use crate::config::CacheConfig;
use tracing::info;

/// Optimized cache configuration for high-memory systems (32GB+)
pub fn get_large_memory_cache_config(totalmemorygb: f64) -> CacheConfig {
    let config = if totalmemorygb >= 128.0 {
        // Ultra-high memory systems (128GB+)
        CacheConfig {
            max_size_mb: 20000,  // 20GB cache
            ttl_seconds: 7200,   // 2 hours TTL
            eviction_policy: "lru".to_string(),
        }
    } else if totalmemorygb >= 64.0 {
        // High memory systems (64GB+)
        CacheConfig {
            max_size_mb: 10000,  // 10GB cache
            ttl_seconds: 3600,   // 1 hour TTL
            eviction_policy: "lru".to_string(),
        }
    } else if totalmemorygb >= 32.0 {
        // Medium-high memory systems (32GB+)
        CacheConfig {
            max_size_mb: 5000,   // 5GB cache
            ttl_seconds: 1800,   // 30 minutes TTL
            eviction_policy: "lru".to_string(),
        }
    } else {
        // Standard configuration for smaller systems
        CacheConfig {
            max_size_mb: 1000,   // 1GB cache
            ttl_seconds: 600,    // 10 minutes TTL
            eviction_policy: "lru".to_string(),
        }
    };

    info!(
        "Cache configured for {:.2}GB system: max_size_mb={}, ttl_seconds={}",
        totalmemorygb,
        config.max_size_mb,
        config.ttl_seconds
    );

    config
    }

/// Memory pressure thresholds optimized for system size
pub fn get_optimized_memory_thresholds(totalmemorygb: f64) -> crate::Result<(f64, f64, f64)> {
    let (low, medium, high) = if totalmemorygb >= 128.0 {
        (0.70, 0.85, 0.95) // Can use more memory safely
    } else if totalmemorygb >= 64.0 {
        (0.65, 0.82, 0.92) // Balanced thresholds
    } else if totalmemorygb >= 32.0 {
        (0.60, 0.80, 0.90) // Conservative for medium systems
    } else {
        (0.50, 0.75, 0.85) // Very conservative for small systems
    };

    info!(
        "Memory pressure thresholds for {:.2}GB system: low={:.0}%, medium={:.0}%, high={:.0}%",
        totalmemorygb,
        low * 100.0,
        medium * 100.0,
        high * 100.0
    );

    Ok((low, medium, high))
    }

/// Batch processing configuration based on available memory
pub fn get_optimized_batch_config(cachememorygb: f64) -> BatchConfig {
    let config = if cachememorygb >= 20.0 {
        BatchConfig {
            insert_batch_size: 50_000,
            search_batch_size: 5_000,
            update_batch_size: 25_000,
            delete_batch_size: 25_000,
            parallel_batch_count: 16,
            max_concurrent_batches: 32,
        }
    } else if cachememorygb >= 10.0 {
        BatchConfig {
            insert_batch_size: 20_000,
            search_batch_size: 2_000,
            update_batch_size: 10_000,
            delete_batch_size: 10_000,
            parallel_batch_count: 8,
            max_concurrent_batches: 16,
        }
    } else if cachememorygb >= 4.0 {
        BatchConfig {
            insert_batch_size: 10_000,
            search_batch_size: 1_000,
            update_batch_size: 5_000,
            delete_batch_size: 5_000,
            parallel_batch_count: 4,
            max_concurrent_batches: 8,
        }
    } else {
        BatchConfig {
            insert_batch_size: 1_000,
            search_batch_size: 100,
            update_batch_size: 500,
            delete_batch_size: 500,
            parallel_batch_count: 2,
            max_concurrent_batches: 4,
        }
    };

    info!(
        "Batch processing configured for {:.2}GB cache: insert_batch={}, search_batch={}",
        cachememorygb, config.insert_batch_size, config.search_batch_size
    );

    config
    }

#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub insert_batch_size: usize,
    pub search_batch_size: usize,
    pub update_batch_size: usize,
    pub delete_batch_size: usize,
    pub parallel_batch_count: usize,
    pub max_concurrent_batches: usize,
}

/// Performance tuning recommendations for high-memory systems
pub struct PerformanceTuning {
    pub num_io_threads: usize,
    pub num_search_threads: usize,
    pub vector_operation_threads: usize,
    pub max_concurrent_searches: usize,
    pub max_concurrent_writes: usize,
    pub prefetch_size_mb: usize,
    pub write_buffer_size_mb: usize,
}

pub fn get_performance_tuning(cpucount: usize, memorygb: f64) -> PerformanceTuning {
    let tuning = if memorygb >= 64.0 && cpucount >= 8 {
        PerformanceTuning {
            num_io_threads: cpucount * 2,
            num_search_threads: cpucount,
            vector_operation_threads: cpucount,
            max_concurrent_searches: 2000,
            max_concurrent_writes: 1000,
            prefetch_size_mb: 256,
            write_buffer_size_mb: 2048,
        }
    } else if memorygb >= 32.0 && cpucount >= 4 {
        PerformanceTuning {
            num_io_threads: cpucount + 4,
            num_search_threads: cpucount,
            vector_operation_threads: cpucount,
            max_concurrent_searches: 1000,
            max_concurrent_writes: 500,
            prefetch_size_mb: 128,
            write_buffer_size_mb: 1024,
        }
    } else {
        PerformanceTuning {
            num_io_threads: cpucount.max(4),
            num_search_threads: cpucount,
            vector_operation_threads: cpucount,
            max_concurrent_searches: 500,
            max_concurrent_writes: 250,
            prefetch_size_mb: 64,
            write_buffer_size_mb: 512,
        }
    };

    info!(
        "Performance tuning for {} CPUs, {:.2}GB RAM: {} IO threads, {} search threads",
        cpucount, memorygb, tuning.num_io_threads, tuning.num_search_threads
    );

    tuning
}
