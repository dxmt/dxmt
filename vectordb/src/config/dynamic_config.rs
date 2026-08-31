use crate::config::{AppConfig, MemoryConfig};
// use crate::storage::memorymanager::MemoryConfig;
use crate::config::memory_detector::{calculate_optimal_memory_config, detect_system_memory};
use tracing::info;

/// Apply dynamic memory configuration based on system detection
pub fn apply_dynamic_memory_config(config: &mut AppConfig) -> MemoryConfig {
    // Detect system memory
    let system_info = detect_system_memory();
    let optimal_config = calculate_optimal_memory_config(&system_info);

    // Update AppConfig memory settings
    if let Some(ref mut memory_config) = config.storage.memory {
        let cache_size_mb = (optimal_config.cachememorybytes / (1024 * 1024)) as usize;
        memory_config.max_size_mb = cache_size_mb;

        info!(
            "Updated memory cache size from {} MB to {} MB based on system memory",
            512, // default
            cache_size_mb
        );
    }

    // Update SSD cache size proportionally for large memory systems
    if let Some(ref mut ssd_config) = config.storage.ssd {
        let total_gb = system_info.totalmemorybytes as f64 / (1024.0 * 1024.0 * 1024.0);
        if total_gb >= 32.0 {
            // For large memory systems, increase SSD cache
            ssd_config.max_size_gb = (ssd_config.max_size_gb as f64 * 2.5) as usize;
            info!(
                "Increased SSD cache size to {} GB for high-memory system",
                ssd_config.max_size_gb
            );
        }
    }

    // Create MemoryConfig for memory manager
    let total_gb = system_info.totalmemorybytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let (_pressure_threshold, _emergency_threshold) = if total_gb >= 32.0 {
        (0.85, 0.95)
    } else {
        (0.80, 0.90)
    };

    MemoryConfig {
        max_size_mb: (optimal_config.maxmemorybytes / (1024 * 1024)) as usize,
        eviction_policy: "lru".to_string(),
    }
}

/// Get memory configuration environment variable overrides
pub fn get_memory_env_overrides() -> Option<(u64, u64, u64)> {
    let max_memory_gb = std::env::var("DB_MAX_MEMORY_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())?;

    let cache_percent = std::env::var("DB_CACHE_MEMORY_PERCENT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.35);

    let index_percent = std::env::var("DB_INDEX_MEMORY_PERCENT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.50);

    let max_memorybytes = (max_memory_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let cache_memorybytes = (max_memorybytes as f64 * cache_percent) as u64;
    let index_memorybytes = (max_memorybytes as f64 * index_percent) as u64;

    Some((max_memorybytes, cache_memorybytes, index_memorybytes))
}
