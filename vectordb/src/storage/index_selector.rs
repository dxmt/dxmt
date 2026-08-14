use tracing::info;

#[derive(Debug, Clone)]
pub enum IndexAlgorithm {
    HNSW {
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    },
    HnswPQ {
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        num_subvectors: usize,
        num_centroids: usize,
        enable_reranking: bool,
    },
    QuantizedIVF {
        nlist: usize,
        nprobe: usize,
    },
    BruteForce {
        enable_parallel: bool,
    },
    DiskANN {
        max_degree: usize,
        search_list_size: usize,
    },
    EnhancedDiskANN {
        max_degree: usize,
        search_list_size: usize,
        use_pq: bool,
        num_pq_subvectors: usize,
        cache_size: usize,
    },
    ScaNN {
        num_leaves: usize,
        num_leaves_to_search: usize,
    },
}

#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub dataset_size: usize,
    pub dimension: usize,
    pub distance_metric: crate::types::vector::DistanceMetric,
    pub memory_budget_mb: usize,
    pub performance_requirements: PerformanceRequirements,
    pub cost_constraints: CostConstraints,
    pub hardware_capabilities: HardwareCapabilities,
}

#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    pub target_recall: f32,
    pub max_latency_ms: f32,
    pub min_throughput_qps: f32,
    pub max_query_latency_ms: f32,
    pub min_recall_rate: f32,
    pub target_qps: f32,
    pub min_insert_throughput: f32,
}

#[derive(Debug, Clone)]
pub struct CostConstraints {
    pub max_memory_gb: f32,
    pub max_cpu_cores: usize,
    pub max_cost_per_month: f32,
    pub max_storage_cost_per_tb: f32,
    pub max_compute_cost_per_hour: f32,
    pub total_budget_limit: f32,
}

#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub cpu_cores: usize,
    pub available_memory_gb: f32,
    pub has_avx2: bool,
    pub has_gpu: bool,
    pub ssd_storage_gb: f32,
    pub network_bandwidth_gbps: f32,
    pub supports_simd: bool,
    pub supports_gpu: bool,
}

#[derive(Debug, Clone)]
pub struct IndexDecision {
    pub selected_algorithm: IndexAlgorithm,
    pub confidence: f32,
    pub expected_performance: ExpectedPerformance,
}

#[derive(Debug, Clone)]
pub struct ExpectedPerformance {
    pub recall: f32,
    pub latency_p95_ms: f32,
    pub throughput_qps: f32,
    pub memory_usage_mb: usize,
}

pub struct IntelligentIndexSelector;

impl IntelligentIndexSelector {
    pub fn new() -> Self {
        Self
    }

    pub async fn select_optimal_index(&self, config: &IndexConfig) -> crate::Result<IndexDecision> {
        // For small datasets (< 5,000), use BruteForce (100% recall, simple)
        if config.dataset_size < 5_000 {
            return Ok(IndexDecision {
                selected_algorithm: IndexAlgorithm::BruteForce { enable_parallel: true },
                confidence: 0.95,
                expected_performance: ExpectedPerformance {
                    recall: 1.0,
                    latency_p95_ms: 10.0,
                    throughput_qps: 1000.0,
                    memory_usage_mb: config.dimension * config.dataset_size * 4 / 1024 / 1024,
                },
            });
        }

        let m = 8;
        let ef_construction = 50;
        let ef_search = 50;

        let vector_memory_mb = (config.dimension * config.dataset_size * 4) / 1024 / 1024;
        let graph_memory_mb = (config.dataset_size * m * 2 * 8) / 1024 / 1024;
        let hnsw_total_memory_mb = vector_memory_mb + graph_memory_mb;

        let num_subvectors = if config.dimension.is_multiple_of(120) { 120 } else { config.dimension / 8 };
        let pq_codes_memory_mb = (config.dataset_size * num_subvectors) / 1024 / 1024;
        let codebook_memory_mb = 1;
        let hnswpq_total_memory_mb = pq_codes_memory_mb + graph_memory_mb + codebook_memory_mb;

        let estimated_latency_ms = (config.dataset_size as f32).log2() * 0.5;
        let estimated_qps = 1000.0 / estimated_latency_ms.max(1.0);

        if config.dataset_size >= 1_000_000 || hnswpq_total_memory_mb > config.memory_budget_mb {
            info!(
                "Selecting Enhanced DiskANN: dataset_size={}, HNSW-PQ memory={}MB, budget={}MB",
                config.dataset_size, hnswpq_total_memory_mb, config.memory_budget_mb
            );
            let cache_size = (config.memory_budget_mb * 1024 * 1024 / (config.dimension * 4)).min(100_000);
            return Ok(IndexDecision {
                selected_algorithm: IndexAlgorithm::EnhancedDiskANN {
                    max_degree: 64,
                    search_list_size: 100,
                    use_pq: true,
                    num_pq_subvectors: num_subvectors,
                    cache_size,
                },
                confidence: 0.88,
                expected_performance: ExpectedPerformance {
                    recall: 0.95,
                    latency_p95_ms: estimated_latency_ms + 5.0,
                    throughput_qps: estimated_qps * 0.7,
                    memory_usage_mb: cache_size * config.dimension * 4 / 1024 / 1024,
                },
            });
        }

        if config.dataset_size >= 100_000 || hnsw_total_memory_mb > config.memory_budget_mb {
            info!(
                "Selecting HNSW-PQ: dataset_size={}, HNSW memory={}MB, HNSW-PQ memory={}MB, budget={}MB",
                config.dataset_size, hnsw_total_memory_mb, hnswpq_total_memory_mb, config.memory_budget_mb
            );
            return Ok(IndexDecision {
                selected_algorithm: IndexAlgorithm::HnswPQ {
                    m,
                    ef_construction,
                    ef_search,
                    num_subvectors,
                    num_centroids: 256,
                    enable_reranking: false,
                },
                confidence: 0.90,
                expected_performance: ExpectedPerformance {
                    recall: 0.97,
                    latency_p95_ms: estimated_latency_ms + 1.0,
                    throughput_qps: estimated_qps * 0.9,
                    memory_usage_mb: hnswpq_total_memory_mb,
                },
            });
        }

        info!(
            "Selecting HNSW: dataset_size={}, memory={}MB, budget={}MB",
            config.dataset_size, hnsw_total_memory_mb, config.memory_budget_mb
        );
        Ok(IndexDecision {
            selected_algorithm: IndexAlgorithm::HNSW { m, ef_construction, ef_search },
            confidence: 0.92,
            expected_performance: ExpectedPerformance {
                recall: 0.98,
                latency_p95_ms: estimated_latency_ms,
                throughput_qps: estimated_qps,
                memory_usage_mb: hnsw_total_memory_mb,
            },
        })
    }
}
