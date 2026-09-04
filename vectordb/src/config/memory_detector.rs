use sysinfo::System;
use tracing::info;

#[derive(Debug, Clone)]
pub struct SystemMemoryInfo {
    pub totalmemorybytes: u64,
    pub availablememorybytes: u64,
    pub cpucount: usize,
}

#[derive(Debug, Clone)]
pub struct OptimizedMemoryConfig {
    pub maxmemorybytes: u64,
    pub cachememorybytes: u64,
    pub indexmemorybytes: u64,
    pub buffermemorybytes: u64,
}

pub fn detect_system_memory() -> SystemMemoryInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    // sysinfo reports bytes. It reported kibibytes up to 0.29, and this multiplied by 1024
    // to convert; on 0.30 that overstated a 33.6 GB machine as 34.4 TB, which then picked the
    // largest branch of every threshold below.
    let total_memory = sys.total_memory();
    let available_memory = sys.available_memory();
    let cpu_count = sys.cpus().len();

    info!(
        "System detected: {} CPUs, {:.2} GB total memory, {:.2} GB available",
        cpu_count,
        total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        available_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    SystemMemoryInfo {
        totalmemorybytes: total_memory,
        availablememorybytes: available_memory,
        cpucount: cpu_count,
    }
}

pub fn calculate_optimal_memory_config(systeminfo: &SystemMemoryInfo) -> OptimizedMemoryConfig {
    let total_gb = systeminfo.totalmemorybytes as f64 / (1024.0 * 1024.0 * 1024.0);

    let os_reserved_gb = match total_gb {
        x if x <= 8.0 => 2.0,
        x if x <= 16.0 => 3.0,
        x if x <= 32.0 => 4.0,
        x if x <= 64.0 => 4.0,
        x if x <= 128.0 => 6.0,
        _ => 8.0,
    };

    let usable_memory_gb = total_gb - os_reserved_gb;
    let usable_memorybytes = (usable_memory_gb * 1024.0 * 1024.0 * 1024.0) as u64;

    let (cache_pct, index_pct, buffer_pct) = if total_gb <= 8.0 {
        (0.15, 0.65, 0.20)
    } else if total_gb <= 32.0 {
        (0.25, 0.60, 0.15)
    } else {
        (0.35, 0.50, 0.15)
    };

    let cache_memorybytes = (usable_memorybytes as f64 * cache_pct) as u64;
    let index_memorybytes = (usable_memorybytes as f64 * index_pct) as u64;
    let buffer_memorybytes = (usable_memorybytes as f64 * buffer_pct) as u64;

    info!(
        "Calculated memory allocation for {:.2} GB system:",
        total_gb
    );
    info!("  OS reserved: {:.2} GB", os_reserved_gb);
    info!("  Max memory: {:.2} GB", usable_memory_gb);
    info!(
        "  Cache memory: {:.2} GB ({}%)",
        cache_memorybytes as f64 / (1024.0 * 1024.0 * 1024.0),
        (cache_pct * 100.0) as u32
    );
    info!(
        "  Index memory: {:.2} GB ({}%)",
        index_memorybytes as f64 / (1024.0 * 1024.0 * 1024.0),
        (index_pct * 100.0) as u32
    );
    info!(
        "  Buffer memory: {:.2} GB ({}%)",
        buffer_memorybytes as f64 / (1024.0 * 1024.0 * 1024.0),
        (buffer_pct * 100.0) as u32
    );

    OptimizedMemoryConfig {
        maxmemorybytes: usable_memorybytes,
        cachememorybytes: cache_memorybytes,
        indexmemorybytes: index_memorybytes,
        buffermemorybytes: buffer_memorybytes,
    }
}

pub fn apply_memory_config_overrides(
    config: &mut crate::config::AppConfig,
    memoryconfig: &OptimizedMemoryConfig,
) {
    if let Some(ref mut memory_config_section) = config.storage.memory {
        memory_config_section.max_size_mb = (memoryconfig.cachememorybytes / (1024 * 1024)) as usize
    }

    if let Some(ref mut ssd_config) = config.storage.ssd {
        let cache_gb = memoryconfig.cachememorybytes as f64 / (1024.0 * 1024.0 * 1024.0);
        if cache_gb >= 10.0 {
            ssd_config.max_size_gb = 50;
        } else if cache_gb >= 4.0 {
            ssd_config.max_size_gb = 20;
        } else {
            ssd_config.max_size_gb = 10;
        }
    }

    info!("Applied dynamic memory configuration overrides");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `sysinfo` reports has to agree with the kernel.
    ///
    /// sysinfo returned kibibytes up to 0.29 and bytes from 0.30. This crate is on 0.30, so
    /// a `* 1024` here overstates memory 1024-fold, and everything downstream sizes itself
    /// from that number.
    #[test]
    fn detected_memory_matches_the_kernel() {
        let proc_total: u64 = std::fs::read_to_string("/proc/meminfo")
            .expect("Linux")
            .lines()
            .find(|l| l.starts_with("MemTotal:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<u64>().ok())
            .expect("MemTotal is there")
            * 1024;

        let detected = detect_system_memory();
        let ratio = detected.totalmemorybytes as f64 / proc_total as f64;
        assert!(
            (0.95..=1.05).contains(&ratio),
            "detect_system_memory says {} bytes, /proc/meminfo says {} -- off by {:.0}x",
            detected.totalmemorybytes,
            proc_total,
            ratio
        );
    }

    #[test]
    fn test_memory_calculation_small_system() {
        let system_info = SystemMemoryInfo {
            totalmemorybytes: 8 * 1024 * 1024 * 1024,
            availablememorybytes: 6 * 1024 * 1024 * 1024,
            cpucount: 4,
        };

        let config = calculate_optimal_memory_config(&system_info);

        assert_eq!(config.maxmemorybytes, 6 * 1024 * 1024 * 1024);
        assert!(
            (config.cachememorybytes as f64 - 0.9 * 1024.0 * 1024.0 * 1024.0).abs()
                < 1024.0 * 1024.0
        );
    }

    #[test]
    fn test_memory_calculation_large_system() {
        let system_info = SystemMemoryInfo {
            totalmemorybytes: 64 * 1024 * 1024 * 1024,
            availablememorybytes: 60 * 1024 * 1024 * 1024,
            cpucount: 8,
        };

        let config = calculate_optimal_memory_config(&system_info);

        assert_eq!(config.maxmemorybytes, 60 * 1024 * 1024 * 1024);
        assert!(
            (config.cachememorybytes as f64 - 21.0 * 1024.0 * 1024.0 * 1024.0).abs()
                < 1024.0 * 1024.0
        );
    }
}
