use super::*;
use std::path::PathBuf;
use num_cpus;
use sysinfo::{System, SystemExt};

/// 零配置默认设置 - 为单机高性能优化
pub struct ZeroConfig;

impl ZeroConfig {
    /// 生成智能默认配置 - 无需任何用户配置
    pub fn generate() -> AppConfig {
        let system = System::new_all();
        let cpu_cores = num_cpus::get();
        let total_memory_mb = (system.total_memory() / 1024 / 1024) as usize;

        // 智能内存分配策略
        // - 给系统留 2GB 或 20% 内存（取较大值）
        let system_reserved_mb = std::cmp::max(2048, total_memory_mb / 5);
        let available_memory_mb = total_memory_mb.saturating_sub(system_reserved_mb);

        // 内存分配比例（优化单机性能）
        // - 60% 用于向量索引和热数据缓存
        // - 30% 用于查询缓存和工作内存
        // - 10% 用于其他操作
        let index_memory_mb = (available_memory_mb * 6) / 10;
        let cache_memory_mb = (available_memory_mb * 3) / 10;

        // 智能数据目录选择
        let data_dir = Self::get_optimal_data_dir();

        AppConfig {
            server: ServerConfig {
                bind_address: "0.0.0.0".to_string(),  // 默认监听所有接口
                port: 9999,  // 统一使用端口9999
                workers: Some(cpu_cores),  // 使用所有CPU核心
            },
            storage: StorageConfig {
                r#type: "hybrid".to_string(),  // 使用混合存储引擎
                memory: Some(MemoryConfig {
                    max_size_mb: index_memory_mb,
                    eviction_policy: "lru".to_string(),
                }),
                ssd: Some(SsdConfig {
                    enabled: true,
                    cache_dir: data_dir.join("cache").to_string_lossy().to_string(),
                    max_size_gb: Self::get_optimal_ssd_cache_size(),
                    compression: "lz4".to_string(),  // 快速压缩
                }),
                object_storage: None,  // 单机模式不需要对象存储
            },
            tables: TableDefaults {
                default_dimension: 384,  // 常见的嵌入维度
                default_distance_metric: "cosine".to_string(),
                max_records_per_table: 10_000_000,  // 单表最大1000万条记录
            },
            cluster: None,  // 单机模式，禁用集群
            processes: None,  // 单机模式，不需要进程管理
        }
    }

    /// 获取最优数据目录
    fn get_optimal_data_dir() -> PathBuf {
        // 优先顺序：
        // 1. 当前目录下的 .dxmt_data
        // 2. 用户主目录下的 .dxmt
        // 3. /var/lib/dxmt (需要权限)
        // 4. /tmp/dxmt (临时，仅用于测试)

        let candidates = vec![
            PathBuf::from("./.dxmt_data"),
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".dxmt"),
            PathBuf::from("/var/lib/dxmt"),
            PathBuf::from("/tmp/dxmt"),
        ];

        for dir in candidates {
            if dir.exists() || std::fs::create_dir_all(&dir).is_ok() {
                return dir;
            }
        }

        // 最后的后备方案
        PathBuf::from("./.dxmt_data")
    }

    /// 获取最优SSD缓存大小
    fn get_optimal_ssd_cache_size() -> usize {
        // 智能检测可用磁盘空间
        // 使用可用空间的 50% 或 100GB，取较小值
        if let Ok(usage) = fs2::available_space(".") {
            let available_gb = usage / (1024 * 1024 * 1024);
            std::cmp::min(available_gb / 2, 100) as usize
        } else {
            10  // 默认 10GB
        }
    }

    /// 创建极简配置 - 最小化资源使用
    pub fn minimal() -> AppConfig {
        AppConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1".to_string(),
                port: 9999,
                workers: Some(2),  // 仅使用2个工作线程
            },
            storage: StorageConfig {
                r#type: "memory".to_string(),  // 纯内存模式
                memory: Some(MemoryConfig {
                    max_size_mb: 512,  // 仅512MB内存
                    eviction_policy: "lru".to_string(),
                }),
                ssd: None,  // 不使用SSD缓存
                object_storage: None,
            },
            tables: TableDefaults {
                default_dimension: 128,
                default_distance_metric: "cosine".to_string(),
                max_records_per_table: 100_000,  // 最大10万条记录
            },
            cluster: None,
            processes: None,
        }
    }

    /// 创建高性能配置 - 最大化单机性能
    pub fn high_performance() -> AppConfig {
        let system = System::new_all();
        let cpu_cores = num_cpus::get();
        let total_memory_mb = (system.total_memory() / 1024 / 1024) as usize;

        AppConfig {
            server: ServerConfig {
                bind_address: "0.0.0.0".to_string(),
                port: 9999,
                workers: Some(cpu_cores * 2),  // 超线程优化
            },
            storage: StorageConfig {
                r#type: "hybrid_optimized".to_string(),
                memory: Some(MemoryConfig {
                    max_size_mb: total_memory_mb * 7 / 10,  // 使用70%内存
                    eviction_policy: "arc".to_string(),  // 自适应缓存
                }),
                ssd: Some(SsdConfig {
                    enabled: true,
                    cache_dir: "/var/lib/dxmt/cache".to_string(),
                    max_size_gb: 500,  // 大SSD缓存
                    compression: "none".to_string(),  // 无压缩以获得最佳性能
                }),
                object_storage: None,
            },
            tables: TableDefaults {
                default_dimension: 768,
                default_distance_metric: "cosine".to_string(),
                max_records_per_table: 100_000_000,  // 1亿条记录
            },
            cluster: None,
            processes: None,
        }
    }
}

/// 配置预设
#[derive(Debug, Clone, Copy)]
pub enum ConfigPreset {
    /// 自动检测最佳配置
    Auto,
    /// 极简配置（低资源消耗）
    Minimal,
    /// 标准配置（平衡）
    Standard,
    /// 高性能配置（最大化性能）
    HighPerformance,
}

impl ConfigPreset {
    pub fn load(self) -> AppConfig {
        match self {
            ConfigPreset::Auto => ZeroConfig::generate(),
            ConfigPreset::Minimal => ZeroConfig::minimal(),
            ConfigPreset::Standard => ZeroConfig::generate(),
            ConfigPreset::HighPerformance => ZeroConfig::high_performance(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_config_generation() {
        let config = ZeroConfig::generate();
        assert_eq!(config.server.port, 9999);
        assert!(config.server.workers.is_some());
        assert!(config.storage.memory.is_some());
    }

    #[test]
    fn test_minimal_config() {
        let config = ZeroConfig::minimal();
        assert_eq!(config.server.workers, Some(2));
        assert_eq!(config.storage.memory.as_ref().unwrap().max_size_mb, 512);
        assert!(config.storage.ssd.is_none());
    }

    #[test]
    fn test_high_performance_config() {
        let config = ZeroConfig::high_performance();
        assert!(config.server.workers.unwrap() > 2);
        assert!(config.storage.ssd.is_some());
    }
}