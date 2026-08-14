use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Comprehensive multi-modal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalConfig {
    /// Whether multi-modal features are enabled
    pub enabled: bool,

    /// Embedding models configuration
    pub models: HashMap<String, ModelConfig>,

    /// Cache configuration
    pub cache: CacheConfig,

    /// GPU configuration
    pub gpu: GpuConfig,

    /// Performance tuning
    pub performance: PerformanceConfig,

    /// Monitoring configuration
    pub monitoring: MonitoringConfig,

    /// Security settings
    pub security: SecurityConfig,
}

/// Individual model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type (openai, clip, whisper, etc.)
    pub model_type: String,

    /// Specific model name
    pub model_name: String,

    /// Model parameters
    pub parameters: HashMap<String, serde_json::Value>,

    /// Supported modalities
    pub modalities: Vec<String>,

    /// Expected dimension
    pub dimension: usize,

    /// Device preference
    pub device: Option<String>,

    /// Auto-load on startup
    pub auto_load: bool,

    /// Priority for resource allocation
    pub priority: u8,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable caching
    pub enabled: bool,

    /// L1 (memory) cache settings
    pub l1: L1CacheConfig,

    /// L2 (Redis) cache settings
    pub l2: Option<L2CacheConfig>,

    /// L3 (disk) cache settings
    pub l3: Option<L3CacheConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1CacheConfig {
    /// Maximum number of entries
    pub capacity: usize,

    /// Time-to-live in seconds
    pub ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2CacheConfig {
    /// Redis connection URL
    pub redis_url: String,

    /// Time-to-live in seconds
    pub ttl_seconds: u64,

    /// Connection pool size
    pub pool_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L3CacheConfig {
    /// Disk cache directory
    pub cache_dir: PathBuf,

    /// Maximum cache size in MB
    pub max_size_mb: usize,

    /// Time-to-live in seconds
    pub ttl_seconds: u64,
}

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,

    /// GPU memory fraction to use (0.0 - 1.0)
    pub memory_fraction: f32,

    /// Allow GPU memory growth
    pub allow_memory_growth: bool,

    /// Preferred GPU device IDs
    pub device_ids: Vec<usize>,

    /// Fallback to CPU if GPU unavailable
    pub cpu_fallback: bool,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Batch processing settings
    pub batch: BatchConfig,

    /// Concurrent processing limits
    pub concurrency: ConcurrencyConfig,

    /// Timeout settings
    pub timeouts: TimeoutConfig,

    /// Resource limits
    pub limits: ResourceLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchConfig {
    /// Default batch size for text embeddings
    pub text_batch_size: usize,

    /// Default batch size for image embeddings
    pub image_batch_size: usize,

    /// Default batch size for audio embeddings
    pub audio_batch_size: usize,

    /// Maximum batch size allowed
    pub max_batch_size: usize,

    /// Timeout for batch processing (seconds)
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    /// Maximum concurrent embedding requests
    pub max_concurrent_requests: usize,

    /// Maximum concurrent model processes
    pub max_concurrent_processes: usize,

    /// Worker thread pool size
    pub worker_threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Model loading timeout in seconds
    pub model_load_timeout: u64,

    /// Embedding generation timeout in seconds
    pub embedding_timeout: u64,

    /// Process communication timeout in seconds
    pub process_timeout: u64,

    /// Search timeout in seconds
    pub search_timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage per model in MB
    pub max_memory_per_model_mb: usize,

    /// Maximum total memory usage in MB
    pub max_total_memory_mb: usize,

    /// Maximum disk usage for cache in MB
    pub max_disk_cache_mb: usize,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub enabled: bool,

    /// Metrics collection interval in seconds
    pub collection_interval: u64,

    /// Enable performance profiling
    pub profiling: bool,

    /// Health check configuration
    pub health_checks: HealthCheckConfig,

    /// Alerting configuration
    pub alerting: AlertingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval in seconds
    pub interval: u64,

    /// Health check timeout in seconds
    pub timeout: u64,

    /// Enable model health checks
    pub check_models: bool,

    /// Enable cache health checks
    pub check_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,

    /// Error rate threshold for alerts (0.0 - 1.0)
    pub error_rate_threshold: f32,

    /// Latency threshold for alerts in milliseconds
    pub latency_threshold_ms: u64,

    /// Memory usage threshold for alerts (0.0 - 1.0)
    pub memory_threshold: f32,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// API key storage and rotation
    pub api_keys: ApiKeyConfig,

    /// Access control
    pub access_control: AccessControlConfig,

    /// Data privacy settings
    pub privacy: PrivacyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// Encrypt API keys at rest
    pub encrypt_at_rest: bool,

    /// API key rotation interval in days
    pub rotation_interval_days: Option<u32>,

    /// Environment variables for API keys
    pub env_vars: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Enable authentication
    pub authentication_enabled: bool,

    /// Authentication method
    pub auth_method: String,

    /// Enable authorization
    pub authorization_enabled: bool,

    /// Rate limiting per user
    pub rate_limiting: RateLimitConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute per user
    pub requests_per_minute: u32,

    /// Burst allowance
    pub burst_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Log level for sensitive data
    pub sensitive_data_logging: String,

    /// Enable data retention policies
    pub data_retention_enabled: bool,

    /// Data retention period in days
    pub retention_period_days: u32,

    /// Enable data anonymization
    pub anonymization_enabled: bool,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            models: Self::default_models(),
            cache: CacheConfig::default(),
            gpu: GpuConfig::default(),
            performance: PerformanceConfig::default(),
            monitoring: MonitoringConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

impl MultimodalConfig {
    /// Create default model configurations
    fn default_models() -> HashMap<String, ModelConfig> {
        let mut models = HashMap::new();

        // OpenAI text embedding
        models.insert(
            "openai_text".to_string(),
            ModelConfig {
                model_type: "openai".to_string(),
                model_name: "text-embedding-3-small".to_string(),
                parameters: HashMap::from([
                    (
                        "api_key_env".to_string(),
                        serde_json::Value::String("OPENAI_API_KEY".to_string()),
                    ),
                    (
                        "dimension".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(1536)),
                    ),
                    (
                        "max_batch_size".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(2048)),
                    ),
                ]),
                modalities: vec!["text".to_string()],
                dimension: 1536,
                device: None,
                auto_load: true,
                priority: 10,
            },
        );

        // CLIP image-text embedding
        models.insert(
            "clip_multimodal".to_string(),
            ModelConfig {
                model_type: "clip".to_string(),
                model_name: "ViT-B/32".to_string(),
                parameters: HashMap::from([
                    (
                        "device".to_string(),
                        serde_json::Value::String("auto".to_string()),
                    ),
                    (
                        "dimension".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(512)),
                    ),
                    ("normalize".to_string(), serde_json::Value::Bool(true)),
                ]),
                modalities: vec!["text".to_string(), "image".to_string()],
                dimension: 512,
                device: Some("auto".to_string()),
                auto_load: true,
                priority: 8,
            },
        );

        // Whisper audio embedding
        models.insert(
            "whisper_audio".to_string(),
            ModelConfig {
                model_type: "whisper".to_string(),
                model_name: "base".to_string(),
                parameters: HashMap::from([
                    (
                        "device".to_string(),
                        serde_json::Value::String("auto".to_string()),
                    ),
                    (
                        "dimension".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(512)),
                    ),
                    (
                        "transcription_enabled".to_string(),
                        serde_json::Value::Bool(true)),
                ]),
                modalities: vec!["audio".to_string()],
                dimension: 512,
                device: Some("auto".to_string()),
                auto_load: false, // Load on demand due to memory requirements
                priority: 6,
            },
        );

        models
    }

    /// Load configuration from file
    pub fn load_from_file(path: &str) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content).map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> crate::Result<()> {
        let content = toml::to_string_pretty(self).map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::Result<()> {
        // Check if at least one model is enabled
        if !self.models.values().any(|m| m.auto_load) {
            return Err(crate::Error::Internal("At least one model must be set to auto_load".to_string()));
        }

        // Check GPU configuration
        if self.gpu.enabled && (self.gpu.memory_fraction <= 0.0 || self.gpu.memory_fraction > 1.0) {
            return Err(crate::Error::Internal("GPU memory_fraction must be between 0.0 and 1.0".to_string()));
        }

        // Check performance limits
        if self.performance.concurrency.max_concurrent_requests == 0 {
            return Err(crate::Error::Internal("max_concurrent_requests must be greater than 0".to_string()));
        }

        // Check cache configuration
        if self.cache.enabled && self.cache.l1.capacity == 0 {
            return Err(crate::Error::Internal(
                "L1 cache capacity must be greater than 0 when caching is enabled".to_string()
            ))
        }

        Ok(())
    }

    /// Get model configuration by name
    pub fn get_model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.get(name)
    }

    /// Get all auto-load models sorted by priority
    pub fn get_auto_load_models(&self) -> Vec<(&String, &ModelConfig)> {
        let mut models: Vec<_> = self
            .models
            .iter()
            .filter(|(_, config)| config.auto_load)
            .collect();

        // Sort by priority (higher priority first)
        models.sort_by_key(|m| std::cmp::Reverse(m.1.priority));
        models
    }

    /// Create development configuration
    pub fn development() -> Self {
        Self {
            enabled: true,
            cache: CacheConfig {
                enabled: true,
                l1: L1CacheConfig {
                    capacity: 1000,
                    ttl_seconds: 3600,
                },
                l2: None, // No Redis in development
                l3: Some(L3CacheConfig {
                    cache_dir: PathBuf::from("./cache"),
                    max_size_mb: 100,
                    ttl_seconds: 86400,
                }),
            },
            gpu: GpuConfig {
                enabled: false, // CPU-only for development
                memory_fraction: 0.5,
                allow_memory_growth: true,
                device_ids: vec![0],
                cpu_fallback: true,
            },
            performance: PerformanceConfig {
                batch: BatchConfig::default(),
                concurrency: ConcurrencyConfig {
                    max_concurrent_requests: 10,
                    max_concurrent_processes: 2,
                    worker_threads: Some(2),
                },
                ..PerformanceConfig::default()
            },
            monitoring: MonitoringConfig {
                enabled: true,
                profiling: true,
                ..MonitoringConfig::default()
            },
            ..Self::default()
        }
    }

    /// Create production configuration
    pub fn production() -> Self {
        Self {
            enabled: true,
            cache: CacheConfig {
                enabled: true,
                l1: L1CacheConfig {
                    capacity: 50000,
                    ttl_seconds: 3600,
                },
                l2: Some(L2CacheConfig {
                    redis_url: "redis://redis-cluster:6379".to_string(),
                    ttl_seconds: 86400,
                    pool_size: 20,
                }),
                l3: Some(L3CacheConfig {
                    cache_dir: PathBuf::from("/data/cache"),
                    max_size_mb: 10240,  // 10GB
                    ttl_seconds: 604800, // 7 days
                }),
            },
            gpu: GpuConfig {
                enabled: true,
                memory_fraction: 0.8,
                allow_memory_growth: true,
                device_ids: vec![0, 1, 2, 3], // Multi-GPU
                cpu_fallback: true,
            },
            performance: PerformanceConfig {
                batch: BatchConfig::default(),
                concurrency: ConcurrencyConfig {
                    max_concurrent_requests: 1000,
                    max_concurrent_processes: 10,
                    worker_threads: None, // Use default
                },
                ..PerformanceConfig::default()
            },
            monitoring: MonitoringConfig {
                enabled: true,
                profiling: false, // Disabled in production for performance
                alerting: AlertingConfig {
                    enabled: true,
                    error_rate_threshold: 0.05, // 5% error rate
                    latency_threshold_ms: 1000,
                    memory_threshold: 0.85, // 85% memory usage
                },
                ..MonitoringConfig::default()
            },
            security: SecurityConfig {
                api_keys: ApiKeyConfig {
                    encrypt_at_rest: true,
                    rotation_interval_days: Some(90),
                    env_vars: HashMap::from([("openai".to_string(), "OPENAI_API_KEY".to_string())]),
                },
                access_control: AccessControlConfig {
                    authentication_enabled: true,
                    auth_method: "jwt".to_string(),
                    authorization_enabled: true,
                    rate_limiting: RateLimitConfig {
                        requests_per_minute: 1000,
                        burst_size: 100,
                    },
                },
                ..SecurityConfig::default()
            },
            ..Self::default()
        }
    }
}

// Default implementations for all config structs
impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            l1: L1CacheConfig {
                capacity: 10000,
                ttl_seconds: 3600,
            },
            l2: None,
            l3: None,
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_fraction: 0.8,
            allow_memory_growth: true,
            device_ids: vec![0],
            cpu_fallback: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            batch: BatchConfig {
                text_batch_size: 32,
                image_batch_size: 16,
                audio_batch_size: 8,
                max_batch_size: 128,
                timeout_seconds: 30,
            },
            concurrency: ConcurrencyConfig {
                max_concurrent_requests: 100,
                max_concurrent_processes: 4,
                worker_threads: None,
            },
            timeouts: TimeoutConfig {
                model_load_timeout: 300, // 5 minutes
                embedding_timeout: 60,
                process_timeout: 30,
                search_timeout: 10,
            },
            limits: ResourceLimits {
                max_memory_per_model_mb: 4096, // 4GB
                max_total_memory_mb: 16384,    // 16GB
                max_disk_cache_mb: 10240,      // 10GB
            },
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: 60,
            profiling: false,
            health_checks: HealthCheckConfig {
                interval: 30,
                timeout: 10,
                check_models: true,
                check_cache: true,
            },
            alerting: AlertingConfig {
                enabled: false,
                error_rate_threshold: 0.1,
                latency_threshold_ms: 5000,
                memory_threshold: 0.9,
            },
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            api_keys: ApiKeyConfig {
                encrypt_at_rest: false,
                rotation_interval_days: None,
                env_vars: HashMap::new(),
            },
            access_control: AccessControlConfig {
                authentication_enabled: false,
                auth_method: "none".to_string(),
                authorization_enabled: false,
                rate_limiting: RateLimitConfig {
                    requests_per_minute: 60,
                    burst_size: 10,
                },
            },
            privacy: PrivacyConfig {
                sensitive_data_logging: "warn".to_string(),
                data_retention_enabled: false,
                retention_period_days: 30,
                anonymization_enabled: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = MultimodalConfig::default();
        assert!(config.validate().is_ok())
    }

    #[test]
    fn test_development_config() {
        let config = MultimodalConfig::development();
        assert!(config.validate().is_ok());
        assert!(!config.gpu.enabled); // CPU-only for development
        assert!(config.monitoring.profiling); // Profiling enabled in development
    }

    #[test]
    fn test_production_config() {
        let config = MultimodalConfig::production();
        assert!(config.validate().is_ok());
        assert!(config.gpu.enabled); // GPU enabled in production
        assert!(!config.monitoring.profiling); // Profiling disabled in production
        assert!(config.security.api_keys.encrypt_at_rest); // Security enabled
    }

    #[test]
    fn test_auto_load_models() {
        let config = MultimodalConfig::default();
        let auto_load = config.get_auto_load_models();

        // Should have at least OpenAI and CLIP auto-loaded
        assert!(auto_load.len() >= 2);

        // Should be sorted by priority (OpenAI has priority 10, CLIP has 8)
        assert!(auto_load[0].1.priority >= auto_load[1].1.priority)
    }

    #[test]
    fn test_config_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let config = MultimodalConfig::default();
        let toml_str = toml::to_string(&config)?;
        let deserialized: MultimodalConfig = toml::from_str(&toml_str)?;

        assert_eq!(config.enabled, deserialized.enabled);
        assert_eq!(config.models.len(), deserialized.models.len());
        Ok(())
    }
}
