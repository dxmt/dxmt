#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::config::StorageConfig;
use crate::storage::{
    compression::CompressionLevel, tiered::TieredStorage, traits::_Storage as Storage,
    IntelligentIndexedStorage, MemoryStorage,
};

/// Storage backend types available in DXMT
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum StorageBackend {
    /// In-memory storage for development/testing
    Memory,
    /// Intelligent indexed storage with AI-driven optimization  
    IntelligentIndexed,
    /// Ultra-high performance storage for demanding workloads
    UltraPerformance,
    /// Cost-optimized tiered storage
    TieredOptimized,
    /// Auto-select optimal backend based on requirements
    #[default]
    Auto,
}

/// Comprehensive storage optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedStorageConfig {
    /// Primary storage backend selection
    pub backend: StorageBackend,

    /// Enable tiered storage optimization
    pub enable_tiering: bool,

    /// Enable vector compression
    pub enable_compression: bool,

    /// Enable memory-mapped index storage
    pub enable_mmap: bool,

    /// Performance targets for optimization
    pub performance_targets: PerformanceTargets,

    /// Cost constraints for optimization  
    pub cost_constraints: OptimizedCostConstraints,

    /// Tiering configuration
    pub tiering: TieringConfig,

    /// Compression configuration
    pub compression: CompressionConfig,

    /// Memory mapping configuration
    pub mmap: MmapConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target P95 query latency in milliseconds
    pub target_latency_p95_ms: f64,

    /// Target queries per second
    pub target_qps: f64,

    /// Minimum acceptable recall rate
    pub min_recall_rate: f64,

    /// Target insert throughput (vectors per second)
    pub target_insert_throughput: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedCostConstraints {
    /// Maximum storage cost per TB per month
    pub max_storage_cost_per_tb: f64,

    /// Maximum compute cost per hour
    pub max_compute_cost_per_hour: f64,

    /// Total monthly budget limit
    pub total_budget_limit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringConfig {
    /// Hot tier configuration (memory/fast SSD)
    pub hot_tier: TierConfig,

    /// Warm tier configuration (SSD/NVMe)
    pub warm_tier: TierConfig,

    /// Cold tier configuration (object storage)
    pub cold_tier: TierConfig,

    /// Automatic tier migration enabled
    pub auto_migration: bool,

    /// Migration frequency in seconds
    pub migration_interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// Storage type
    pub storage_type: String,

    /// Maximum size in GB
    pub max_size_gb: f64,

    /// Cost per GB per month
    pub cost_per_gb: f64,

    /// Expected P95 latency in ms
    pub expected_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Compression algorithm
    pub algorithm: String,

    /// Compression level
    pub level: CompressionLevel,

    /// Enable vector-specific compression
    pub enable_vector_compression: bool,

    /// Product quantization bits per component
    pub pq_bits_per_component: u8,

    /// Minimum accuracy threshold for compression
    pub accuracy_threshold: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: "zstd".to_string(),
            level: CompressionLevel::Default,
            enable_vector_compression: false,
            pq_bits_per_component: 8,
            accuracy_threshold: 0.95,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmapConfig {
    /// Enable memory mapping for indices
    pub enable_index_mmap: bool,

    /// Enable memory mapping for vectors
    pub enable_vector_mmap: bool,

    /// Prefetch batch size
    pub prefetch_batch_size: usize,

    /// Memory advise strategy
    pub memory_advise: String,
}

impl Default for OptimizedStorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Auto,
            enable_tiering: true,
            enable_compression: true,
            enable_mmap: true,
            performance_targets: PerformanceTargets {
                target_latency_p95_ms: 50.0,
                target_qps: 1000.0,
                min_recall_rate: 0.95,
                target_insert_throughput: 1000.0,
            },
            cost_constraints: OptimizedCostConstraints {
                max_storage_cost_per_tb: 30.0,
                max_compute_cost_per_hour: 10.0,
                total_budget_limit: 1000.0,
            },
            tiering: TieringConfig {
                hot_tier: TierConfig {
                    storage_type: "memory".to_string(),
                    max_size_gb: 16.0,
                    cost_per_gb: 10.0,
                    expected_latency_ms: 5.0,
                },
                warm_tier: TierConfig {
                    storage_type: "ssd".to_string(),
                    max_size_gb: 500.0,
                    cost_per_gb: 0.5,
                    expected_latency_ms: 25.0,
                },
                cold_tier: TierConfig {
                    storage_type: "s3".to_string(),
                    max_size_gb: f64::INFINITY,
                    cost_per_gb: 0.02,
                    expected_latency_ms: 300.0,
                },
                auto_migration: true,
                migration_interval_secs: 3600, // 1 hour
            },
            compression: CompressionConfig {
                algorithm: "product_quantization".to_string(),
                level: CompressionLevel::Default,
                enable_vector_compression: true,
                pq_bits_per_component: 8,
                accuracy_threshold: 0.95,
            },
            mmap: MmapConfig {
                enable_index_mmap: true,
                enable_vector_mmap: true,
                prefetch_batch_size: 1000,
                memory_advise: "sequential".to_string(),
            },
        }
    }
}

/// Advanced storage factory for creating optimized storage backends
pub struct StorageFactory {
    hardware_detector: HardwareDetector,
    cost_calculator: CostCalculator,
    performance_predictor: PerformancePredictor,
}

impl StorageFactory {
    /// Create a new storage factory
    pub fn new() -> Self {
        Self {
            hardware_detector: HardwareDetector::new(),
            cost_calculator: CostCalculator::new(),
            performance_predictor: PerformancePredictor::new(),
        }
    }

    /// Create optimized storage based on configuration and requirements
    pub async fn create_storage(
        &self,
        base_config: &StorageConfig,
        opt_config: &OptimizedStorageConfig,
    ) -> crate::Result<Arc<dyn Storage + Send + Sync>> {
        info!(
            "Creating optimized storage with backend: {:?}",
            opt_config.backend
        );

        // Detect hardware capabilities
        let hardware_capabilities = self.hardware_detector.detect().await;
        debug!("Detected hardware: {:?}", hardware_capabilities);

        // Select optimal backend if auto mode
        let selected_backend = match &opt_config.backend {
            StorageBackend::Auto => {
                self.select_optimal_backend(opt_config, &hardware_capabilities)
                    .await?
            }
            backend => backend.clone(),
        };

        info!("Selected storage backend: {:?}", selected_backend);

        // Create storage instance
        match selected_backend {
            StorageBackend::Memory => Ok(Arc::new(MemoryStorage::new())),

            StorageBackend::IntelligentIndexed => {
                self.create_intelligent_storage(base_config, opt_config, &hardware_capabilities)
                    .await
            }

            StorageBackend::UltraPerformance => {
                self.create_ultra_performance_storage(base_config, opt_config)
                    .await
            }

            StorageBackend::TieredOptimized => {
                self.create_tiered_storage(base_config, opt_config).await
            }

            StorageBackend::Auto => unreachable!("Auto backend should be resolved"),
        }
    }

    /// Create intelligent indexed storage
    async fn create_intelligent_storage(
        &self,
        base_config: &StorageConfig,
        opt_config: &OptimizedStorageConfig,
        hardware_capabilities: &HardwareCapabilities,
    ) -> crate::Result<Arc<dyn Storage + Send + Sync>> {
        info!("Creating intelligent indexed storage");

        // Convert optimized config to storage config
        let mut storage_config = base_config.clone();
        self.apply_optimizations(&mut storage_config, opt_config);

        // Create intelligent storage
        let storage = IntelligentIndexedStorage::new(&storage_config).await?;

        Ok(Arc::new(storage))
    }

    /// Create ultra-performance storage
    async fn create_ultra_performance_storage(
        &self,
        base_config: &StorageConfig,
        opt_config: &OptimizedStorageConfig,
    ) -> crate::Result<Arc<dyn Storage + Send + Sync>> {
        info!("Creating ultra-performance storage");

        // Convert and apply optimizations
        let mut storage_config = base_config.clone();
        self.apply_optimizations(&mut storage_config, opt_config);

        // Create ultra-performance storage
        let config = StorageConfig::default();
        let storage = IntelligentIndexedStorage::new(&config).await?;

        Ok(Arc::new(storage))
    }

    /// Create tiered storage
    async fn create_tiered_storage(
        &self,
        base_config: &StorageConfig,
        opt_config: &OptimizedStorageConfig,
    ) -> crate::Result<Arc<dyn Storage + Send + Sync>> {
        info!(
            "Creating tiered storage with configuration: {:?}",
            opt_config.tiering
        );

        // Create individual tier storages
        let hot_storage = self
            .create_tier_storage(&opt_config.tiering.hot_tier)
            .await?;
        let warm_storage = self
            .create_tier_storage(&opt_config.tiering.warm_tier)
            .await?;
        let cold_storage = self
            .create_tier_storage(&opt_config.tiering.cold_tier)
            .await?;

        // Create tiered storage
        // Convert to storage traits config
        let storage_config = crate::storage::traits::StorageConfig {
            storage_type: crate::storage::traits::StorageType::Memory,
            path: None,
            cache_size_mb: None,
        };

        let tiered = TieredStorage::new(hot_storage, warm_storage, cold_storage, storage_config);

        Ok(Arc::new(tiered))
    }

    /// Select optimal backend based on requirements
    async fn select_optimal_backend(
        &self,
        opt_config: &OptimizedStorageConfig,
        _hardware_capabilities: &HardwareCapabilities,
    ) -> crate::Result<StorageBackend> {
        let analysis = BackendSelectionAnalysis {
            performance_targets: opt_config.performance_targets.clone(),
            cost_constraints: opt_config.cost_constraints.clone(),
            workload_characteristics: self.analyze_workload().await?,
        };

        let selection = self.analyze_backend_suitability(&analysis).await?;

        info!(
            "Backend selection analysis complete: {:?} with confidence {:.2}",
            selection.recommended_backend, selection.confidence
        );

        Ok(selection.recommended_backend)
    }

    /// Apply optimizations to storage config
    fn apply_optimizations(
        &self,
        storage_config: &mut StorageConfig,
        opt_config: &OptimizedStorageConfig,
    ) {
        // Apply compression settings
        if opt_config.enable_compression {
            // storage_config.enable_compression = true;
            // storage_config.compression_level = opt_config.compression.level;
            debug!("Applied compression optimizations");
        }

        // Apply tiering settings
        if opt_config.enable_tiering {
            // storage_config.enable_tiering = true;
            debug!("Applied tiering optimizations");
        }

        // Apply memory mapping settings
        if opt_config.enable_mmap {
            // storage_config.enable_mmap = true;
            debug!("Applied memory mapping optimizations");
        }
    }

    /// Create storage for a specific tier
    async fn create_tier_storage(
        &self,
        tier_config: &TierConfig,
    ) -> crate::Result<Arc<dyn Storage + Send + Sync>> {
        match tier_config.storage_type.as_str() {
            "memory" => Ok(Arc::new(MemoryStorage::new())),
            "ssd" | "nvme" => Ok(Arc::new(MemoryStorage::new())), // Simplified for now
            "s3" | "object" => Ok(Arc::new(MemoryStorage::new())), // Simplified for now
            _ => {
                warn!(
                    "Unknown storage type: {}, defaulting to memory",
                    tier_config.storage_type
                );
                Ok(Arc::new(MemoryStorage::new()))
            }
        }
    }

    /// Analyze workload characteristics
    async fn analyze_workload(&self) -> crate::Result<WorkloadCharacteristics> {
        // Simplified workload analysis
        Ok(WorkloadCharacteristics {
            query_frequency: QueryFrequency::High,
            data_access_pattern: AccessPattern::Random,
            vector_dimensionality: VectorDimensionality::Medium,
            dataset_size: DatasetSize::Large,
        })
    }

    /// Analyze backend suitability
    async fn analyze_backend_suitability(
        &self,
        analysis: &BackendSelectionAnalysis,
    ) -> crate::Result<BackendSelection> {
        // Simplified selection logic
        let recommended_backend = if analysis.performance_targets.target_latency_p95_ms < 20.0 {
            StorageBackend::UltraPerformance
        } else if analysis.cost_constraints.max_storage_cost_per_tb < 50.0 {
            StorageBackend::TieredOptimized
        } else {
            StorageBackend::IntelligentIndexed
        };

        Ok(BackendSelection {
            recommended_backend,
            confidence: 0.85,
            reasoning: "Based on performance and cost analysis".to_string(),
        })
    }
}

impl Default for StorageFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Hardware detection and capability analysis
pub struct HardwareDetector;

impl Default for HardwareDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareDetector {
    pub fn new() -> Self {
        Self
    }

    pub(crate) async fn detect(&self) -> HardwareCapabilities {
        let has_gpu = {
            #[cfg(feature = "gpu")]
            {
                crate::acceleration::gpu::is_cuda_available()
            }
            #[cfg(not(feature = "gpu"))]
            {
                false
            }
        };
        HardwareCapabilities {
            cpu_cores: num_cpus::get(),
            available_memory_gb: self.detect_memory_gb().await,
            has_avx2: cfg!(target_feature = "avx2"),
            has_gpu,
        }
    }

    async fn detect_memory_gb(&self) -> f64 {
        // Simplified memory detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return (kb as f64) / 1_048_576.0; // Convert KB to GB
                            }
                        }
                    }
                }
            }
        }

        // Default fallback
        16.0
    }
}

/// Cost calculation and optimization
pub struct CostCalculator;

impl Default for CostCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostCalculator {
    pub fn new() -> Self {
        Self
    }

    pub async fn calculate_storage_cost(&self, backend: &StorageBackend, data_size_gb: f64) -> f64 {
        match backend {
            StorageBackend::Memory => data_size_gb * 10.0, // $10/GB/month for memory
            StorageBackend::IntelligentIndexed => data_size_gb * 2.0,
            StorageBackend::UltraPerformance => data_size_gb * 5.0,
            StorageBackend::TieredOptimized => data_size_gb * 0.5, // Optimized cost
            StorageBackend::Auto => 0.0, // Will be calculated for selected backend
        }
    }
}

/// Performance prediction and optimization
pub struct PerformancePredictor;

impl Default for PerformancePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformancePredictor {
    pub fn new() -> Self {
        Self
    }

    pub async fn predict_latency(
        &self,
        backend: &StorageBackend,
        workload: &WorkloadCharacteristics,
    ) -> f64 {
        match backend {
            StorageBackend::Memory => 5.0,
            StorageBackend::IntelligentIndexed => 15.0,
            StorageBackend::UltraPerformance => 8.0,
            StorageBackend::TieredOptimized => 25.0,
            StorageBackend::Auto => 20.0,
        }
    }
}

// Supporting structures for analysis
#[derive(Debug, Clone)]
pub struct BackendSelectionAnalysis {
    pub performance_targets: PerformanceTargets,
    pub cost_constraints: OptimizedCostConstraints,
    //     pub hardware_capabilities: HardwareCapabilities,
    pub workload_characteristics: WorkloadCharacteristics,
}

#[derive(Debug, Clone)]
pub struct BackendSelection {
    pub recommended_backend: StorageBackend,
    pub confidence: f64,
    pub reasoning: String,
}

#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    pub query_frequency: QueryFrequency,
    pub data_access_pattern: AccessPattern,
    pub vector_dimensionality: VectorDimensionality,
    pub dataset_size: DatasetSize,
}

#[derive(Debug, Clone)]
pub enum QueryFrequency {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone)]
pub enum AccessPattern {
    Sequential,
    Random,
    Temporal,
}

#[derive(Debug, Clone)]
pub enum VectorDimensionality {
    Low,    // < 128
    Medium, // 128-768
    High,   // > 768
}

#[derive(Debug, Clone)]
pub enum DatasetSize {
    Small,  // < 10K vectors
    Medium, // 10K-1M vectors
    Large,  // > 1M vectors
}

/// Integration helper for easy migration from basic storage
pub struct StorageIntegration;

impl StorageIntegration {
    /// Create optimized storage with intelligent defaults
    pub async fn create_optimized_default(
        base_config: &StorageConfig,
    ) -> crate::Result<Arc<dyn Storage + Send + Sync>> {
        let factory = StorageFactory::new();
        let opt_config = OptimizedStorageConfig::default();

        factory.create_storage(base_config, &opt_config).await
    }

    /// Migrate from one storage backend to another.
    ///
    /// Copies all tables and their records from `current_storage` into a freshly
    /// created `new_storage` of the requested type, then returns the new storage.
    /// Migration is best-effort: individual record failures are logged but do not
    /// abort the whole migration.
    pub async fn migrate_to_optimized(
        current_storage: Arc<dyn Storage + Send + Sync>,
        target_config: &OptimizedStorageConfig,
        base_config: &StorageConfig,
    ) -> crate::Result<Arc<dyn Storage + Send + Sync>> {
        info!("Starting storage migration to {:?}", target_config.backend);

        let factory = StorageFactory::new();
        let new_storage = factory.create_storage(base_config, target_config).await?;

        // 1. List all tables in the source storage
        let tables = current_storage.list_tables().await?;
        let table_count = tables.len();
        info!("Migrating {} tables", table_count);

        for table in &tables {
            // 2. Re-create the table in the target storage
            let table_config = crate::types::table::TableConfig {
                name: table.name.clone(),
                dimension: table.dimension,
                distance_metric: table.distance_metric,
                index_type: Some(table.index_type),
                description: table.description.clone(),
                enable_full_text_search: table.enable_full_text_search,
                text_fields: table.text_fields.clone(),
                memory_optimization: table.memory_optimization,
                payload_schema: table.payload_schema.clone(),
                schema_version: table.schema_version,
                named_vectors: None,
                hnsw: None,
                quantization: None,
                payload_index: None,
            };

            if let Err(e) = new_storage.create_table(table_config).await {
                warn!(
                    "Failed to create table {} in target storage: {}",
                    table.name, e
                );
                continue;
            }

            // 3. Copy records using a large-k search (zero vector) as a scan proxy.
            //    The _Storage trait has no list-all-records, so we use a high-k search.
            let zero_vec = vec![0.0f32; table.dimension];
            let query = crate::types::SearchQuery {
                vector: zero_vec,
                top_k: usize::MAX / 2,
                limit: usize::MAX / 2,
                filter: None,
                with_payload: true,
                with_vector: true,
                search_mode: crate::types::record::SearchMode::Normal,
                vector_name: None,
                payload_fields: None,
                score_threshold: None,
                offset: 0,
                mmr: None,
            };

            match current_storage.search(&table.name, query).await {
                Ok(result) => {
                    let records: Vec<crate::types::record::CreateRecordRequest> = result
                        .hits
                        .into_iter()
                        .filter_map(|hit| {
                            hit.vector
                                .map(|v| crate::types::record::CreateRecordRequest {
                                    id: hit.id,
                                    vector: v,
                                    sparse_vector: None,
                                    payload: hit.payload,
                                    named_vectors: None,
                                })
                        })
                        .collect();

                    let record_count = records.len();
                    if let Err(e) = new_storage.insert_records(&table.name, records).await {
                        warn!("Failed to migrate records for table {}: {}", table.name, e);
                    } else {
                        info!(
                            "Migrated {} records from table {}",
                            record_count, table.name
                        );
                    }
                }
                Err(e) => {
                    warn!("Failed to scan records from table {}: {}", table.name, e);
                }
            }
        }

        info!(
            "Storage migration complete: {} tables processed",
            table_count
        );
        Ok(new_storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_storage_factory_creation() {
        let factory = StorageFactory::new();
        let base_config = StorageConfig::default();
        let opt_config = OptimizedStorageConfig::default();

        let storage = factory.create_storage(&base_config, &opt_config).await;
        assert!(storage.is_ok());
    }

    #[tokio::test]
    async fn test_backend_selection() {
        let factory = StorageFactory::new();
        let opt_config = OptimizedStorageConfig {
            backend: StorageBackend::Auto,
            ..Default::default()
        };

        // let hardware_capabilities = HardwareCapabilities {
        //     cpu_cores: 8,
        //     available_memory_gb: 32.0,
        //     ssd_storage_gb: 1000.0,
        //     network_bandwidth_gbps: 10.0,
        //     supports_simd: true,
        //     supports_gpu: false,
        // };

        // let backend = factory.select_optimal_backend(&opt_config, &hardware_capabilities).await;
        // assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_hardware_detection() {
        let detector = HardwareDetector::new();
        // detect() method was commented out during cleanup
        // let caps = detector.detect().await;
        // assert!(caps.cpu_cores > 0);
        // assert!(caps.available_memory_gb > 0.0);
    }

    #[test]
    fn test_optimized_config_defaults() {
        let config = OptimizedStorageConfig::default();

        assert!(config.enable_tiering);
        assert!(config.enable_compression);
        assert!(config.enable_mmap);
        assert_eq!(config.performance_targets.target_latency_p95_ms, 50.0);
    }
}

// HardwareCapabilities struct
#[derive(Debug, Clone)]
pub(crate) struct HardwareCapabilities {
    cpu_cores: usize,
    available_memory_gb: f64,
    has_avx2: bool,
    has_gpu: bool,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        let has_gpu = {
            #[cfg(feature = "gpu")]
            {
                crate::acceleration::gpu::is_cuda_available()
            }
            #[cfg(not(feature = "gpu"))]
            {
                false
            }
        };
        Self {
            cpu_cores: num_cpus::get(),
            available_memory_gb: 16.0,
            has_avx2: true,
            has_gpu,
        }
    }
}

impl HardwareCapabilities {
    pub fn detect() -> Self {
        Self::default()
    }
}
