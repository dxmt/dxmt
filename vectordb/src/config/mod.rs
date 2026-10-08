use serde::{Deserialize, Serialize};

pub mod env;
pub mod layout;
// pub mod eight_core_optimization; // Removed - redundant
pub mod memory_detector;
pub mod performance_optimization;

/// 应用配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub storage: StorageConfig,
    pub tables: TableDefaults,
    pub cluster: Option<ClusterConfig>,
    pub processes: Option<ProcessesConfig>,
}

/// 服务器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub bind_address: String,
    pub port: u16,
    pub workers: Option<usize>,
}

/// 存储配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageConfig {
    pub memory: Option<MemoryConfig>,
    pub ssd: Option<SsdConfig>,
    pub object_storage: Option<ObjectStorageConfig>,
}

/// 内存存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub max_size_mb: usize,
    pub eviction_policy: String,
}

/// 缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub max_size_mb: usize,
    pub ttl_seconds: u64,
    pub eviction_policy: String,
}

/// SSD存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsdConfig {
    pub enabled: bool,
    pub cache_dir: String,
    pub max_size_gb: usize,
}

/// 对象存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStorageConfig {
    pub provider: String,
    pub bucket: String,
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
}

/// 表默认配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableDefaults {
    pub default_dimension: usize,
    pub default_distance_metric: String,
    pub max_records_per_table: usize,
    /// 默认是否启用内存优化 (RaBitQ + Clustering)
    /// true: 启用内存优化，节省 85-90% 内存，召回率 87-95%
    /// false: 禁用优化，保持高精度，召回率 95-99%
    pub default_memory_optimization: bool,
}

/// 集群配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// 是否启用集群模式
    pub enabled: bool,
    /// 节点ID (自动生成或指定)
    pub nodeid: Option<String>,
    /// 节点名称
    pub node_name: Option<String>,
    /// 集群名称
    pub cluster_name: String,
    /// 种子节点列表 (用于发现其他节点)
    pub seed_nodes: Vec<String>,
    /// 是否是种子节点
    pub is_seed: bool,
    /// 心跳间隔(秒)
    pub heartbeat_interval_secs: u64,
    /// 节点超时时间(秒)
    pub node_timeout_secs: u64,
    /// 复制因子
    pub replication_factor: usize,
}

/// 进程管理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessesConfig {
    /// 是否启用进程管理
    pub enabled: bool,
    /// Consul配置
    pub consul: Option<ConsulConfig>,
    /// MinIO配置
    pub minio: Option<MinioConfig>,
    /// 数据目录
    pub data_dir: String,
}

/// Consul进程配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsulConfig {
    /// 是否启用
    pub enabled: bool,
    /// 可执行文件路径
    pub executable: String,
    /// 绑定地址
    pub bind_addr: String,
    /// HTTP端口
    pub httpport: u16,
}

/// MinIO进程配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinioConfig {
    /// 是否启用
    pub enabled: bool,
    /// 可执行文件路径
    pub executable: String,
    /// 服务地址
    pub address: String,
    /// 控制台地址
    pub console_address: String,
}

impl AppConfig {
    /// 加载配置
    pub fn load() -> crate::Result<Self> {
        // 首先尝试加载 .env 文件中的环境变量
        Self::load_env_file();

        // 先尝试加载配置文件，如果失败则使用默认配置
        match Self::load_from_file() {
            Ok(config) => {
                tracing::info!("Configuration loaded from file");
                Ok(config)
            }
            Err(_) => {
                tracing::warn!("Failed to load config file, using default configuration");
                Ok(AppConfig::default())
            }
        }
    }

    /// 加载 .env 文件
    fn load_env_file() {
        // 尝试加载 .env 文件，失败时静默处理
        if dotenv::dotenv().is_err() {
            // .env 文件不存在或无法读取，使用系统环境变量
            tracing::debug!("No .env file found, using system environment variables")
        } else {
            tracing::info!("Environment variables loaded from .env file");
        }
    }

    /// 从文件加载配置
    fn load_from_file() -> crate::Result<Self> {
        let config_str = std::fs::read_to_string("config.toml").map_err(|e| {
            crate::Error::InvalidConfig(format!("Failed to read config file: {}", e))
        })?;

        let config: AppConfig = toml::from_str(&config_str).map_err(|e| {
            crate::Error::InvalidConfig(format!("Failed to parse config file: {}", e))
        })?;

        Ok(config)
    }

    /// 验证配置
    pub fn validate(&self) -> crate::Result<()> {
        if self.server.port == 0 {
            return Err(crate::Error::InvalidConfig(
                "Server port cannot be 0".to_string(),
            ));
        }

        if self.tables.default_dimension == 0 {
            return Err(crate::Error::InvalidConfig(
                "Default dimension cannot be 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        // 从环境变量获取主机地址，如果不存在则使用默认值0.0.0.0
        let host = crate::config::env::var("DXMT_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());

        // 从环境变量获取端口，如果不存在则使用默认值9999
        let port = crate::config::env::var("DXMT_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(9999);

        // 工作线程数
        let workers = crate::config::env::var("DXMT_WORKER_THREADS")
            .ok()
            .and_then(|w| w.parse::<usize>().ok());

        // 存储配置

        let memory_cache_size = crate::config::env::var("DXMT_MEMORY_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(512);

        // `./cache`, relative to wherever the process happened to start, until the layout
        // gave it a home. See `config::layout`.
        let ssd_cache_dir = layout::Layout::from_env()
            .ssd_cache()
            .to_string_lossy()
            .into_owned();

        let ssd_cache_size = crate::config::env::var("DXMT_SSD_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10);

        // 对象存储配置
        let object_storage =
            if let Ok(s3_provider) = crate::config::env::var("DXMT_OBJECT_STORAGE_PROVIDER") {
                let s3_bucket = crate::config::env::var("DXMT_S3_BUCKET")
                    .unwrap_or_else(|_| "vectordb".to_string());

                let s3_region = crate::config::env::var("DXMT_S3_REGION").ok();

                let s3_endpoint = crate::config::env::var("DXMT_S3_ENDPOINT").ok();

                let aws_access_key = crate::config::env::var("DXMT_AWS_ACCESS_KEY_ID").ok();

                let aws_secret_key = crate::config::env::var("DXMT_AWS_SECRET_ACCESS_KEY").ok();

                Some(ObjectStorageConfig {
                    provider: s3_provider,
                    bucket: s3_bucket,
                    region: s3_region.or_else(|| Some("us-east-1".to_string())),
                    endpoint: s3_endpoint,
                    access_key_id: aws_access_key,
                    secret_access_key: aws_secret_key,
                })
            } else {
                None
            };

        // 表默认配置
        let default_dimension = crate::config::env::var("DXMT_DEFAULT_DIMENSION")
            .ok()
            .and_then(|d| d.parse::<usize>().ok())
            .unwrap_or(1536);

        let default_distance_metric = crate::config::env::var("DXMT_DEFAULT_DISTANCE_METRIC")
            .unwrap_or_else(|_| "cosine".to_string());

        let max_records_per_table = crate::config::env::var("DXMT_MAX_RECORDS_PER_TABLE")
            .ok()
            .and_then(|r| r.parse::<usize>().ok())
            .unwrap_or(1_000_000);

        // 内存优化配置（默认启用）
        let default_memory_optimization = crate::config::env::var("DXMT_MEMORY_OPTIMIZATION")
            .ok()
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true);

        // 集群配置
        let cluster =
            if crate::config::env::var("DXMT_CLUSTER_ENABLED").unwrap_or_default() == "true" {
                let node_id = crate::config::env::var("DXMT_NODE_ID").ok();
                let node_name = crate::config::env::var("DXMT_NODE_NAME").ok();
                let cluster_name = crate::config::env::var("DXMT_CLUSTER_NAME")
                    .unwrap_or_else(|_| "dxmt-cluster".to_string());
                let seed_nodes = crate::config::env::var("DXMT_SEED_NODES")
                    .unwrap_or_default()
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
                let is_seed = crate::config::env::var("DXMT_IS_SEED").unwrap_or_default() == "true";
                let heartbeat_interval_secs = crate::config::env::var("DXMT_HEARTBEAT_INTERVAL")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5);
                let node_timeout_secs = crate::config::env::var("DXMT_NODE_TIMEOUT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(30);
                let replication_factor = crate::config::env::var("DXMT_REPLICATION_FACTOR")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2);

                Some(ClusterConfig {
                    enabled: true,
                    nodeid: node_id,
                    node_name,
                    cluster_name,
                    seed_nodes,
                    is_seed,
                    heartbeat_interval_secs,
                    node_timeout_secs,
                    replication_factor,
                })
            } else {
                None
            };

        // 进程管理配置
        let processes = if crate::config::env::var("DXMT_PROCESSES_ENABLED").unwrap_or_default()
            == "true"
        {
            let data_dir = layout::Layout::from_env()
                .processes()
                .to_string_lossy()
                .into_owned();

            // 默认不启用 Consul，除非 DXMT_ENABLE_CONSUL=true
            let consul_enabled = crate::config::env::var("DXMT_ENABLE_CONSUL")
                .unwrap_or_else(|_| "false".to_string())
                == "true"
                || crate::config::env::var("DXMT_CONSUL_ENABLED")
                    .unwrap_or_else(|_| "false".to_string())
                    == "true";

            let consul = if consul_enabled {
                Some(ConsulConfig {
                    enabled: true,
                    executable: crate::config::env::var("DXMT_CONSUL_EXECUTABLE")
                        .unwrap_or_else(|_| "/usr/local/bin/consul".to_string()),
                    bind_addr: crate::config::env::var("DXMT_CONSUL_BIND_ADDR")
                        .unwrap_or_else(|_| "127.0.0.1".to_string()),
                    httpport: crate::config::env::var("DXMT_CONSUL_HTTP_PORT")
                        .ok()
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(8500),
                })
            } else {
                None
            };

            let minio =
                if crate::config::env::var("DXMT_MINIO_ENABLED").unwrap_or_default() == "true" {
                    Some(MinioConfig {
                        enabled: true,
                        executable: crate::config::env::var("DXMT_MINIO_EXECUTABLE")
                            .unwrap_or_else(|_| "/usr/local/bin/minio".to_string()),
                        address: crate::config::env::var("DXMT_MINIO_ADDRESS")
                            .unwrap_or_else(|_| "127.0.0.1:9000".to_string()),
                        console_address: crate::config::env::var("DXMT_MINIO_CONSOLE_ADDRESS")
                            .unwrap_or_else(|_| "127.0.0.1:9001".to_string()),
                    })
                } else {
                    None
                };

            Some(ProcessesConfig {
                enabled: true,
                consul,
                minio,
                data_dir,
            })
        } else {
            None
        };

        Self {
            server: ServerConfig {
                bind_address: format!("{}:{}", host, port),
                port,
                workers,
            },
            storage: StorageConfig {
                memory: Some(MemoryConfig {
                    max_size_mb: memory_cache_size,
                    eviction_policy: "lru".to_string(),
                }),
                ssd: Some(SsdConfig {
                    enabled: true,
                    cache_dir: ssd_cache_dir,
                    max_size_gb: ssd_cache_size,
                }),
                object_storage,
            },
            tables: TableDefaults {
                default_dimension,
                default_distance_metric,
                max_records_per_table,
                default_memory_optimization,
            },
            cluster,
            processes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // 使用全局锁确保环境变量测试串行运行
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_default_config() {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除环境变量以确保测试的一致性
        std::env::remove_var("DXMT_PORT");
        std::env::remove_var("DXMT_HOST");

        let config = AppConfig::default();
        assert_eq!(config.server.port, 9999);
        assert_eq!(config.server.bind_address, "0.0.0.0:9999");
        assert_eq!(config.tables.default_dimension, 1536)
    }

    #[test]
    fn test_config_with_env_port() {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除所有环境变量以确保测试的一致性
        std::env::remove_var("DXMT_HOST");
        std::env::remove_var("DXMT_PORT");

        // 设置环境变量
        std::env::set_var("DXMT_PORT", "9999");

        let config = AppConfig::default();
        assert_eq!(config.server.port, 9999);
        assert_eq!(config.server.bind_address, "0.0.0.0:9999");

        // 清理环境变量
        std::env::remove_var("DXMT_PORT")
    }

    #[test]
    fn test_config_with_env_host() {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除所有环境变量以确保测试的一致性
        std::env::remove_var("DXMT_HOST");
        std::env::remove_var("DXMT_PORT");

        // 设置环境变量
        std::env::set_var("DXMT_HOST", "192.168.1.100");

        let config = AppConfig::default();
        assert_eq!(config.server.bind_address, "192.168.1.100:9999");

        // 清理环境变量
        std::env::remove_var("DXMT_HOST")
    }

    #[test]
    fn test_config_with_both_env_vars() {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除所有环境变量以确保测试的一致性
        std::env::remove_var("DXMT_HOST");
        std::env::remove_var("DXMT_PORT");

        // 设置两个环境变量
        std::env::set_var("DXMT_HOST", "10.0.0.1");
        std::env::set_var("DXMT_PORT", "7777");

        let config = AppConfig::default();
        assert_eq!(config.server.port, 7777);
        assert_eq!(config.server.bind_address, "10.0.0.1:7777");

        // 清理环境变量
        std::env::remove_var("DXMT_HOST");
        std::env::remove_var("DXMT_PORT")
    }

    #[test]
    fn test_config_with_invalid_env_port() {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除所有环境变量以确保测试的一致性
        std::env::remove_var("DXMT_HOST");
        std::env::remove_var("DXMT_PORT");

        // 设置无效的环境变量
        std::env::set_var("DXMT_PORT", "invalid");

        let config = AppConfig::default();
        // 应该回退到默认值
        assert_eq!(config.server.port, 9999);
        assert_eq!(config.server.bind_address, "0.0.0.0:9999");

        // 清理环境变量
        std::env::remove_var("DXMT_PORT")
    }

    #[test]
    fn test_config_validation() {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除环境变量确保使用默认值
        std::env::remove_var("DXMT_HOST");
        std::env::remove_var("DXMT_PORT");

        let config = AppConfig::default();
        assert!(config.validate().is_ok());

        let mut bad_config = config;
        bad_config.server.port = 0;
        assert!(bad_config.validate().is_err())
    }

    #[test]
    fn test_config_with_storage_env_vars() -> Result<(), Box<dyn std::error::Error>> {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除所有环境变量
        clear_all_env_vars();

        // 设置存储相关环境变量
        std::env::set_var("DXMT_MEMORY_CACHE_SIZE", "1024");
        std::env::set_var("DXMT_SSD_CACHE_DIR", "/tmp/cache");
        std::env::set_var("DXMT_SSD_CACHE_SIZE", "20");

        let config = AppConfig::default();
        assert_eq!(config.storage.memory.as_ref().unwrap().max_size_mb, 1024);
        assert_eq!(config.storage.ssd.as_ref().unwrap().cache_dir, "/tmp/cache");
        assert_eq!(config.storage.ssd.as_ref().unwrap().max_size_gb, 20);

        // 清理环境变量
        clear_all_env_vars();
        Ok(())
    }

    #[test]
    fn test_config_with_s3_env_vars() -> Result<(), Box<dyn std::error::Error>> {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除所有环境变量
        clear_all_env_vars();

        // 设置S3相关环境变量
        std::env::set_var("DXMT_OBJECT_STORAGE_PROVIDER", "minio");
        std::env::set_var("DXMT_S3_BUCKET", "my-test-bucket");
        std::env::set_var("DXMT_S3_REGION", "eu-west-1");
        std::env::set_var("DXMT_S3_ENDPOINT", "http://localhost:9000");
        std::env::set_var("DXMT_AWS_ACCESS_KEY_ID", "test_key");
        std::env::set_var("DXMT_AWS_SECRET_ACCESS_KEY", "test_secret");

        let config = AppConfig::default();
        let object_storage = config
            .storage
            .object_storage
            .as_ref()
            .ok_or("object_storage should be configured")?;
        assert_eq!(object_storage.provider, "minio");
        assert_eq!(object_storage.bucket, "my-test-bucket");
        assert_eq!(object_storage.region, Some("eu-west-1".to_string()));
        assert_eq!(
            object_storage.endpoint,
            Some("http://localhost:9000".to_string())
        );
        assert_eq!(object_storage.access_key_id, Some("test_key".to_string()));
        assert_eq!(
            object_storage.secret_access_key,
            Some("test_secret".to_string())
        );

        // 清理环境变量
        clear_all_env_vars();
        Ok(())
    }

    #[test]
    fn test_config_with_table_env_vars() {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除所有环境变量
        clear_all_env_vars();

        // 设置表配置相关环境变量
        std::env::set_var("DXMT_DEFAULT_DIMENSION", "768");
        std::env::set_var("DXMT_DEFAULT_DISTANCE_METRIC", "euclidean");
        std::env::set_var("DXMT_MAX_RECORDS_PER_TABLE", "500000");

        let config = AppConfig::default();
        assert_eq!(config.tables.default_dimension, 768);
        assert_eq!(config.tables.default_distance_metric, "euclidean");
        assert_eq!(config.tables.max_records_per_table, 500000);

        // 清理环境变量
        clear_all_env_vars()
    }

    #[test]
    fn test_config_with_worker_threads() {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除所有环境变量
        clear_all_env_vars();

        // 设置工作线程数
        std::env::set_var("DXMT_WORKER_THREADS", "8");

        let config = AppConfig::default();
        assert_eq!(config.server.workers, Some(8));

        // 清理环境变量
        clear_all_env_vars()
    }

    #[test]
    fn test_config_default_memory_optimization() {
        let _guard = ENV_LOCK.lock().unwrap();

        // 清除所有环境变量
        clear_all_env_vars();

        // 测试默认值（应该为 true）
        let config = AppConfig::default();
        assert!(config.tables.default_memory_optimization);

        // 清理环境变量
        clear_all_env_vars()
    }

    // No unit test here sets DXMT_MEMORY_OPTIMIZATION. `ENV_LOCK` only orders the tests in this
    // module, and `IntelligentIndexedStorage::new` reads the same variable: a storage unit test
    // constructed while one of these held it at "false" would silently get the size-based index
    // selector. The `true`/`false` cases are covered by tests/memory_optimization_config_test.rs,
    // which runs in its own process.

    // 辅助函数：清除所有测试相关的环境变量
    fn clear_all_env_vars() {
        let env_vars = [
            "DXMT_HOST",
            "DXMT_PORT",
            "DXMT_WORKER_THREADS",
            "DXMT_MEMORY_CACHE_SIZE",
            "DXMT_SSD_CACHE_DIR",
            "DXMT_SSD_CACHE_SIZE",
            "DXMT_OBJECT_STORAGE_PROVIDER",
            "DXMT_S3_BUCKET",
            "DXMT_S3_REGION",
            "DXMT_S3_ENDPOINT",
            "DXMT_AWS_ACCESS_KEY_ID",
            "DXMT_AWS_SECRET_ACCESS_KEY",
            "DXMT_DEFAULT_DIMENSION",
            "DXMT_DEFAULT_DISTANCE_METRIC",
            "DXMT_MAX_RECORDS_PER_TABLE",
            "DXMT_MEMORY_OPTIMIZATION",
        ];

        for var in &env_vars {
            std::env::remove_var(var)
        }
    }
}
