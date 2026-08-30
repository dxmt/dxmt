use tracing::info;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    /// IVF whose partitions live in object storage and load on demand.
    ///
    /// Chosen at rebuild time rather than at CREATE TABLE: partitioning needs the vectors to
    /// cluster, and there are none when the table is created.
    PartitionedIvf {
        nlist: usize,
        nprobe: usize,
        max_resident_partitions: usize,
    },

    ScaNN {
        num_leaves: usize,
        num_leaves_to_search: usize,
    },
}

impl IndexAlgorithm {
    /// The `ef_search` this algorithm was configured with, for graph indexes that have one.
    ///
    /// The search path needs this to honour the selector's choice. Without it the parameter is
    /// inert: it reaches `HnswConfig` at build time, but every query used a hardcoded default
    /// instead, so tuning it in the selector changed nothing measurable.
    pub fn configured_ef_search(&self) -> Option<usize> {
        match self {
            IndexAlgorithm::HNSW { ef_search, .. } | IndexAlgorithm::HnswPQ { ef_search, .. } => {
                Some(*ef_search)
            }
            _ => None,
        }
    }

    /// Whether moving from `self` to `want` needs the index thrown away and rebuilt.
    ///
    /// A rebuild is O(n) and, on the `reevaluate_indexes` sweep, happens while the table is
    /// serving. Full equality is the wrong test: the selector derives several parameters from
    /// the current row count, so an equality check makes every sweep after any write rebuild a
    /// table whose algorithm never changed. `PartitionedIvf`'s `nlist` is `sqrt(rows)` — sixty
    /// new rows move it, and the partitioned index has `maintain()` for exactly that drift.
    ///
    /// The parameters that do force a rebuild are the ones baked into the graph as it is
    /// built. `ef_search` is not one of them: it is read per query.
    pub fn requires_rebuild_to_reach(&self, want: &IndexAlgorithm) -> bool {
        match (self, want) {
            (
                IndexAlgorithm::HNSW {
                    m, ef_construction, ..
                },
                IndexAlgorithm::HNSW {
                    m: want_m,
                    ef_construction: want_ef_construction,
                    ..
                },
            ) => m != want_m || ef_construction != want_ef_construction,
            (
                IndexAlgorithm::HnswPQ {
                    m,
                    ef_construction,
                    num_subvectors,
                    num_centroids,
                    ..
                },
                IndexAlgorithm::HnswPQ {
                    m: want_m,
                    ef_construction: want_ef_construction,
                    num_subvectors: want_num_subvectors,
                    num_centroids: want_num_centroids,
                    ..
                },
            ) => {
                m != want_m
                    || ef_construction != want_ef_construction
                    || num_subvectors != want_num_subvectors
                    || num_centroids != want_num_centroids
            }
            // Same algorithm, upkeep-only differences: partition counts drift with every write
            // and are converged by `maintain()`; brute force and the DiskANN family have no
            // build-time parameter the selector varies with size.
            _ => std::mem::discriminant(self) != std::mem::discriminant(want),
        }
    }
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

impl Default for IntelligentIndexSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl IntelligentIndexSelector {
    pub fn new() -> Self {
        Self
    }

    pub async fn select_optimal_index(&self, config: &IndexConfig) -> crate::Result<IndexDecision> {
        // For small datasets (< 5,000), use BruteForce (100% recall, simple)
        if config.dataset_size < 5_000 {
            return Ok(IndexDecision {
                selected_algorithm: IndexAlgorithm::BruteForce {
                    enable_parallel: true,
                },
                confidence: 0.95,
                expected_performance: ExpectedPerformance {
                    recall: 1.0,
                    latency_p95_ms: 10.0,
                    throughput_qps: 1000.0,
                    memory_usage_mb: config.dimension * config.dataset_size * 4 / 1024 / 1024,
                },
            });
        }

        // Graph shape does not depend on row count the way it was written to. Measured on the
        // storage ANN benchmark at 10K×384 (exact cosine ground truth, recall@10): m=8 with
        // ef_construction=100 returned 0.831, while m=16 with ef_construction=200 — what the
        // memory-optimized path has always used — returned 0.945 on the same corpus. A 10K
        // table is not "small enough" to need a sparser graph; what makes a query hard is
        // dimensionality, which this branch never looked at.
        //
        // These now match `select_memory_optimized_index` so the two selection paths do not
        // silently answer with different recall for the same table.
        let m = 16;
        let ef_construction = 200;
        let ef_search = 128;

        let vector_memory_mb = (config.dimension * config.dataset_size * 4) / 1024 / 1024;
        let graph_memory_mb = (config.dataset_size * m * 2 * 8) / 1024 / 1024;
        let hnsw_total_memory_mb = vector_memory_mb + graph_memory_mb;

        let num_subvectors = if config.dimension.is_multiple_of(120) {
            120
        } else {
            config.dimension / 8
        };
        let pq_codes_memory_mb = (config.dataset_size * num_subvectors) / 1024 / 1024;
        let codebook_memory_mb = 1;
        let hnswpq_total_memory_mb = pq_codes_memory_mb + graph_memory_mb + codebook_memory_mb;

        let estimated_latency_ms = (config.dataset_size as f32).log2() * 0.5;
        let estimated_qps = 1000.0 / estimated_latency_ms.max(1.0);

        if config.dataset_size >= 1_000_000 && hnsw_total_memory_mb > config.memory_budget_mb {
            info!(
                "Selecting Enhanced DiskANN: dataset_size={}, HNSW-PQ memory={}MB, budget={}MB",
                config.dataset_size, hnswpq_total_memory_mb, config.memory_budget_mb
            );
            let cache_size =
                (config.memory_budget_mb * 1024 * 1024 / (config.dimension * 4)).min(100_000);
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

        if hnsw_total_memory_mb > config.memory_budget_mb {
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
            selected_algorithm: IndexAlgorithm::HNSW {
                m,
                ef_construction,
                ef_search,
            },
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
