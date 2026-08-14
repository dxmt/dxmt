use serde::{Deserialize, Serialize};

use crate::storage::factory::{
    StorageBackend, OptimizedStorageConfig, PerformanceTargets, 
    OptimizedCostConstraints, CompressionConfig
};
use crate::storage::compression::CompressionLevel;

/// Enhanced storage configuration with optimization features
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct EnhancedStorageConfig {
    /// Base storage configuration
    pub base: crate::config::StorageConfig,
    
    /// Optimization configuration
    pub optimization: OptimizedStorageConfig,
    
    /// Auto-tuning configuration
    pub auto_tuning: AutoTuningConfig,
    
    /// Performance monitoring configuration
    pub monitoring: MonitoringConfig,
    
    /// Migration configuration
    pub migration: MigrationConfig,
}

/// Auto-tuning configuration for storage optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuningConfig {
    /// Enable automatic performance tuning
    pub enabled: bool,
    
    /// Tuning interval in seconds
    pub tuning_interval_secs: u64,
    
    /// Minimum observation period before tuning
    pub min_observation_period_secs: u64,
    
    /// Performance degradation threshold for re-tuning
    pub performance_degradation_threshold: f64,
    
    /// Cost increase threshold for re-tuning
    pub cost_increase_threshold: f64,
    
    /// Learning rate for optimization adjustments
    pub learning_rate: f64,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    
    /// Metrics collection interval in seconds
    pub collection_interval_secs: u64,
    
    /// Metrics retention period in days
    pub retention_days: u32,
    
    /// Enable detailed query tracing
    pub enable_query_tracing: bool,
    
    /// Enable cost tracking
    pub enable_cost_tracking: bool,
    
    /// Performance alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Alert thresholds for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Maximum acceptable P95 latency in ms
    pub max_latency_p95_ms: f64,
    
    /// Minimum acceptable recall rate
    pub min_recall_rate: f64,
    
    /// Maximum acceptable error rate
    pub max_error_rate: f64,
    
    /// Maximum acceptable cost per query
    pub max_cost_per_query: f64,
}

/// Migration configuration for storage backend changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Enable automatic migration
    pub auto_migration_enabled: bool,
    
    /// Migration strategy
    pub strategy: MigrationStrategy,
    
    /// Migration batch size
    pub batch_size: usize,
    
    /// Migration parallelism
    pub parallel_transfers: usize,
    
    /// Verification settings
    pub verification: VerificationConfig,
}

/// Migration strategy options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStrategy {
    /// Migrate all data at once (downtime required)
    BigBang,
    /// Gradual migration with dual-write
    Gradual,
    /// Blue-green deployment style
    BlueGreen,
}

/// Data verification configuration for migrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Enable data integrity verification
    pub enabled: bool,
    
    /// Verification sample rate (0.0 to 1.0)
    pub sample_rate: f64,
    
    /// Enable checksums
    pub enable_checksums: bool,
    
    /// Enable query result verification
    pub enable_query_verification: bool,
}


impl Default for AutoTuningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tuning_interval_secs: 3600, // 1 hour
            min_observation_period_secs: 1800, // 30 minutes
            performance_degradation_threshold: 0.1, // 10% degradation
            cost_increase_threshold: 0.2, // 20% cost increase
            learning_rate: 0.1,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_secs: 60, // 1 minute
            retention_days: 30,
            enable_query_tracing: true,
            enable_cost_tracking: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_latency_p95_ms: 100.0,
            min_recall_rate: 0.95,
            max_error_rate: 0.01, // 1%
            max_cost_per_query: 0.001, // $0.001 per query
        }
    }
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            auto_migration_enabled: false,
            strategy: MigrationStrategy::Gradual,
            batch_size: 1000,
            parallel_transfers: 4,
            verification: VerificationConfig::default(),
        }
    }
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_rate: 0.01, // 1% sampling
            enable_checksums: true,
            enable_query_verification: false,
        }
    }
}

/// Configuration presets for different use cases
pub struct ConfigurationPresets;

impl ConfigurationPresets {
    /// High-performance configuration for latency-sensitive workloads
    pub fn high_performance() -> EnhancedStorageConfig {
        EnhancedStorageConfig {
            optimization: OptimizedStorageConfig {
                backend: StorageBackend::UltraPerformance,
                enable_tiering: true,
                enable_compression: false, // Disable compression for max performance
                enable_mmap: true,
                performance_targets: PerformanceTargets {
                    target_latency_p95_ms: 10.0,
                    target_qps: 10000.0,
                    min_recall_rate: 0.99,
                    target_insert_throughput: 5000.0,
                },
                ..Default::default()
            },
            auto_tuning: AutoTuningConfig {
                enabled: true,
                tuning_interval_secs: 1800, // 30 minutes - more frequent tuning
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    /// Cost-optimized configuration for budget-conscious deployments
    pub fn cost_optimized() -> EnhancedStorageConfig {
        EnhancedStorageConfig {
            optimization: OptimizedStorageConfig {
                backend: StorageBackend::TieredOptimized,
                enable_tiering: true,
                enable_compression: true,
                enable_mmap: true,
                performance_targets: PerformanceTargets {
                    target_latency_p95_ms: 100.0,
                    target_qps: 1000.0,
                    min_recall_rate: 0.95,
                    target_insert_throughput: 500.0,
                },
                cost_constraints: OptimizedCostConstraints {
                    max_storage_cost_per_tb: 20.0,
                    max_compute_cost_per_hour: 5.0,
                    total_budget_limit: 500.0,
                },
                compression: CompressionConfig {
                    level: CompressionLevel::Best, // Maximum compression
                    enable_vector_compression: true,
                    pq_bits_per_component: 4, // Aggressive quantization
                    accuracy_threshold: 0.90, // Lower accuracy for better compression
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    /// Balanced configuration for general-purpose workloads
    pub fn balanced() -> EnhancedStorageConfig {
        EnhancedStorageConfig {
            optimization: OptimizedStorageConfig {
                backend: StorageBackend::IntelligentIndexed,
                enable_tiering: true,
                enable_compression: true,
                enable_mmap: true,
                performance_targets: PerformanceTargets {
                    target_latency_p95_ms: 50.0,
                    target_qps: 2000.0,
                    min_recall_rate: 0.97,
                    target_insert_throughput: 1000.0,
                },
                cost_constraints: OptimizedCostConstraints {
                    max_storage_cost_per_tb: 30.0,
                    max_compute_cost_per_hour: 8.0,
                    total_budget_limit: 800.0,
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    /// Development configuration for local testing
    pub fn development() -> EnhancedStorageConfig {
        EnhancedStorageConfig {
            optimization: OptimizedStorageConfig {
                backend: StorageBackend::Memory,
                enable_tiering: false,
                enable_compression: false,
                enable_mmap: false,
                performance_targets: PerformanceTargets {
                    target_latency_p95_ms: 20.0,
                    target_qps: 100.0,
                    min_recall_rate: 0.90,
                    target_insert_throughput: 100.0,
                },
                ..Default::default()
            },
            auto_tuning: AutoTuningConfig {
                enabled: false, // Disable auto-tuning for development
                ..Default::default()
            },
            monitoring: MonitoringConfig {
                enabled: true,
                collection_interval_secs: 300, // 5 minutes - less frequent
                retention_days: 7,
                enable_query_tracing: true,
                enable_cost_tracking: false,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Configuration validation utilities
pub struct ConfigurationValidator;

impl ConfigurationValidator {
    /// Validate enhanced storage configuration
    pub fn validate(config: &EnhancedStorageConfig) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        
        // Validate performance targets
        if config.optimization.performance_targets.target_latency_p95_ms <= 0.0 {
            errors.push("Target latency must be positive".to_string());
        }
        
        if config.optimization.performance_targets.min_recall_rate < 0.0 || 
           config.optimization.performance_targets.min_recall_rate > 1.0 {
            errors.push("Recall rate must be between 0.0 and 1.0".to_string());
        }
        
        // Validate cost constraints
        if config.optimization.cost_constraints.total_budget_limit <= 0.0 {
            errors.push("Budget limit must be positive".to_string());
        }
        
        // Validate compression settings
        if config.optimization.compression.accuracy_threshold < 0.0 || 
           config.optimization.compression.accuracy_threshold > 1.0 {
            errors.push("Compression accuracy threshold must be between 0.0 and 1.0".to_string());
        }
        
        // Validate auto-tuning settings
        if config.auto_tuning.learning_rate <= 0.0 || config.auto_tuning.learning_rate > 1.0 {
            errors.push("Learning rate must be between 0.0 and 1.0".to_string());
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
    
    /// Check configuration compatibility
    pub fn check_compatibility(config: &EnhancedStorageConfig) -> Vec<String> {
        let mut warnings = Vec::new();
        
        // Check for conflicting settings
        if matches!(config.optimization.backend, StorageBackend::Memory) && 
           config.optimization.enable_tiering {
            warnings.push("Memory backend with tiering enabled may not be optimal".to_string());
        }
        
        if !config.optimization.enable_compression && 
           config.optimization.cost_constraints.max_storage_cost_per_tb < 30.0 {
            warnings.push("Low cost target without compression may be difficult to achieve".to_string());
        }
        
        // Check performance vs cost trade-offs
        if config.optimization.performance_targets.target_latency_p95_ms < 20.0 && 
           config.optimization.cost_constraints.max_compute_cost_per_hour < 10.0 {
            warnings.push("Very low latency target with low compute budget may not be achievable".to_string());
        }
        
        warnings
    }
}

/// Configuration loading utilities
pub struct ConfigurationLoader;

impl ConfigurationLoader {
    /// Load configuration from TOML file
    pub fn from_toml_file(path: &str) -> Result<EnhancedStorageConfig, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: EnhancedStorageConfig = toml::from_str(&content)?;
        
        // Validate loaded configuration
        if let Err(errors) = ConfigurationValidator::validate(&config) {
            return Err(format!("Configuration validation failed: {}", errors.join(", ")).into());
        }
        
        Ok(config)
    }
    
    /// Load configuration from environment variables
    pub fn from_env() -> EnhancedStorageConfig {
        let mut config = EnhancedStorageConfig::default();
        
        // Override with environment variables if present
        if let Ok(backend) = std::env::var("DXMT_STORAGE_BACKEND") {
            config.optimization.backend = match backend.as_str() {
                "memory" => StorageBackend::Memory,
                "intelligent" => StorageBackend::IntelligentIndexed,
                "ultra" => StorageBackend::UltraPerformance,
                "tiered" => StorageBackend::TieredOptimized,
                "auto" => StorageBackend::Auto,
                _ => StorageBackend::Auto,
            };
        }
        
        if let Ok(latency) = std::env::var("DXMT_TARGET_LATENCY_MS") {
            if let Ok(latency_val) = latency.parse::<f64>() {
                config.optimization.performance_targets.target_latency_p95_ms = latency_val;
            }
        }
        
        if let Ok(budget) = std::env::var("DXMT_BUDGET_LIMIT") {
            if let Ok(budget_val) = budget.parse::<f64>() {
                config.optimization.cost_constraints.total_budget_limit = budget_val;
            }
        }
        
        config
    }
    
    /// Create configuration for specific deployment scenario
    pub fn for_scenario(scenario: &str) -> EnhancedStorageConfig {
        match scenario {
            "high-performance" | "gaming" | "real-time" => {
                ConfigurationPresets::high_performance()
            }
            "cost-optimized" | "budget" | "startup" => {
                ConfigurationPresets::cost_optimized()
            }
            "balanced" | "production" | "enterprise" => {
                ConfigurationPresets::balanced()
            }
            "development" | "testing" | "local" => {
                ConfigurationPresets::development()
            }
            _ => {
                println!("Unknown scenario '{}', using balanced configuration", scenario);
                ConfigurationPresets::balanced()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_configuration_presets() {
        let high_perf = ConfigurationPresets::high_performance();
        assert!(matches!(high_perf.optimization.backend, StorageBackend::UltraPerformance));
        assert!(high_perf.optimization.performance_targets.target_latency_p95_ms < 20.0);
        
        let cost_opt = ConfigurationPresets::cost_optimized();
        assert!(matches!(cost_opt.optimization.backend, StorageBackend::TieredOptimized));
        assert!(cost_opt.optimization.enable_compression);
        
        let balanced = ConfigurationPresets::balanced();
        assert!(matches!(balanced.optimization.backend, StorageBackend::IntelligentIndexed));
        
        let dev = ConfigurationPresets::development();
        assert!(matches!(dev.optimization.backend, StorageBackend::Memory));
        assert!(!dev.auto_tuning.enabled);
    }
    
    #[test]
    fn test_configuration_validation() {
        let mut config = EnhancedStorageConfig::default();
        assert!(ConfigurationValidator::validate(&config).is_ok());
        
        // Test invalid configuration
        config.optimization.performance_targets.target_latency_p95_ms = -1.0;
        assert!(ConfigurationValidator::validate(&config).is_err());
    }
    
    #[test]
    fn test_configuration_loader() {
        let config = ConfigurationLoader::from_env();
        assert!(ConfigurationValidator::validate(&config).is_ok());
        
        let scenarios = vec!["high-performance", "cost-optimized", "balanced", "development"];
        for scenario in scenarios {
            let config = ConfigurationLoader::for_scenario(scenario);
            assert!(ConfigurationValidator::validate(&config).is_ok());
        }
    }
}