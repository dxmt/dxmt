//! Hybrid RaBitQ + Hierarchical Clustering Index
//!
//! This module combines Phase 2 (RaBitQ binary quantization) and Phase 3 (Hierarchical clustering)
//! to achieve maximum performance:
//! - 100x search space reduction from clustering
//! - 16x memory compression from RaBitQ
//! - Combined: 1600x potential improvement
//!
//! Architecture:
//! ```text
//! Query → Cluster Selection (100x reduction)
//!       → Binary Search in Clusters (10x speedup)
//!       → Optional Reranking (accuracy boost)
//!       → Final Results
//! ```

use crate::search::hierarchical_clustering::{
    HierarchicalClusteringConfig, HierarchicalClusteringIndex,
};
use crate::search::quantization::rabitq::{RaBitQConfig, RaBitQQuantizer};
use crate::types::{vector::DenseVector, DistanceMetric};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Configuration
// ============================================================================

/// Hybrid index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridIndexConfig {
    /// Clustering configuration
    pub clustering_config: HierarchicalClusteringConfig,

    /// RaBitQ binary quantization configuration
    pub rabitq_config: RaBitQConfig,

    /// Whether to store original vectors for reranking
    pub store_original_vectors: bool,

    /// Whether to enable reranking with original vectors
    pub use_reranking: bool,

    /// Reranking factor: rerank top (k * rerank_factor) candidates
    /// Example: k=10, rerank_factor=5 → rerank top 50 candidates to get top 10
    pub rerank_factor: usize,
}

impl Default for HybridIndexConfig {
    fn default() -> Self {
        Self {
            clustering_config: HierarchicalClusteringConfig::default(),
            rabitq_config: RaBitQConfig::default(),
            store_original_vectors: true,
            use_reranking: true,
            rerank_factor: 3,
        }
    }
}

impl HybridIndexConfig {
    /// Memory-optimized preset: minimize memory usage
    pub fn memory_optimized() -> Self {
        Self {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 100,
                num_fine_clusters_per_coarse: 100,
                num_coarse_probes: 5,
                num_fine_probes: 5,
                max_iterations: 20,
                tolerance: 1e-4,
                enable_parallel: true,
            },
            rabitq_config: RaBitQConfig {
                num_codebooks: 16,
                codebook_size: 256,
                training_iterations: 20,
                use_residual: true,
                enable_simd: true,
            },
            store_original_vectors: false, // Don't store originals
            use_reranking: false,
            rerank_factor: 1,
        }
    }

    /// Speed-optimized preset: maximize search speed
    pub fn speed_optimized() -> Self {
        Self {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 1000,
                num_fine_clusters_per_coarse: 100,
                num_coarse_probes: 10,
                num_fine_probes: 20,
                max_iterations: 20,
                tolerance: 1e-4,
                enable_parallel: true,
            },
            rabitq_config: RaBitQConfig {
                num_codebooks: 32,
                codebook_size: 256,
                training_iterations: 20,
                use_residual: true,
                enable_simd: true,
            },
            store_original_vectors: true, // Store for reranking
            use_reranking: true,
            rerank_factor: 5,
        }
    }

    /// Balanced preset: good balance between speed and memory
    pub fn balanced() -> Self {
        Self {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 100,
                num_fine_clusters_per_coarse: 50,
                num_coarse_probes: 10,
                num_fine_probes: 10,
                max_iterations: 20,
                tolerance: 1e-4,
                enable_parallel: true,
            },
            rabitq_config: RaBitQConfig {
                num_codebooks: 16,
                codebook_size: 256,
                training_iterations: 20,
                use_residual: true,
                enable_simd: true,
            },
            store_original_vectors: true,
            use_reranking: true,
            rerank_factor: 3,
        }
    }
}

// ============================================================================
// Data Structures
// ============================================================================

/// Binary code for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryCode {
    /// Vector ID
    pub vector_id: String,

    /// Binary code (RaBitQ quantized)
    pub code: Vec<u8>,
}

/// Statistics for the hybrid index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridIndexStats {
    /// Total number of vectors
    pub total_vectors: usize,

    /// Number of coarse clusters
    pub num_coarse_clusters: usize,

    /// Number of fine clusters
    pub num_fine_clusters: usize,

    /// Total binary codes stored
    pub num_binary_codes: usize,

    /// Memory usage estimate (bytes)
    pub memory_bytes: usize,

    /// Training time (milliseconds)
    pub training_time_ms: u128,

    /// Average search time (milliseconds)
    pub avg_search_time_ms: f64,

    /// Total searches performed
    pub total_searches: usize,
}

impl Default for HybridIndexStats {
    fn default() -> Self {
        Self {
            total_vectors: 0,
            num_coarse_clusters: 0,
            num_fine_clusters: 0,
            num_binary_codes: 0,
            memory_bytes: 0,
            training_time_ms: 0,
            avg_search_time_ms: 0.0,
            total_searches: 0,
        }
    }
}

// ============================================================================
// Hybrid Index
// ============================================================================

/// Hybrid RaBitQ + Hierarchical Clustering Index
pub struct HybridRaBitQClusteringIndex {
    /// Configuration
    config: HybridIndexConfig,

    /// Distance metric
    distance_metric: DistanceMetric,

    /// Vector dimension
    dimension: usize,

    /// Hierarchical clustering index
    clustering: HierarchicalClusteringIndex,

    /// RaBitQ quantizer
    quantizer: RaBitQQuantizer,

    /// Binary codes organized by cluster (coarse_id, fine_id) -> binary codes
    cluster_binary_codes: HashMap<(usize, usize), Vec<BinaryCode>>,

    /// Vector IDs for each cluster (coarse_id, fine_id) -> vector IDs
    cluster_vector_ids: HashMap<(usize, usize), Vec<String>>,

    /// Original vectors (optional, for reranking)
    original_vectors: Option<HashMap<String, DenseVector>>,

    /// Whether the index is trained
    is_trained: bool,

    /// Statistics
    stats: HybridIndexStats,
}

impl HybridRaBitQClusteringIndex {
    /// Create a new hybrid index
    pub fn new(
        config: HybridIndexConfig,
        distance_metric: DistanceMetric,
        dimension: usize,
    ) -> Self {
        let clustering = HierarchicalClusteringIndex::new(
            config.clustering_config.clone(),
            distance_metric,
        );

        let quantizer = RaBitQQuantizer::new(
            config.rabitq_config.clone(),
            distance_metric,
            dimension,
        );

        Self {
            config,
            distance_metric,
            dimension,
            clustering,
            quantizer,
            cluster_binary_codes: HashMap::new(),
            cluster_vector_ids: HashMap::new(),
            original_vectors: None,
            is_trained: false,
            stats: HybridIndexStats::default(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &HybridIndexConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &HybridIndexStats {
        &self.stats
    }

    /// Check if the index is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Train the hybrid index
    pub fn train(&mut self, vectors: &[(String, DenseVector)]) -> crate::Result<()> {
        let train_start = std::time::Instant::now();

        // Step 1: Train hierarchical clustering
        self.clustering.train(vectors)?;

        // Step 2: Train RaBitQ quantizer on all vectors
        let training_vectors: Vec<DenseVector> = vectors.iter().map(|(_, v)| v.clone()).collect();
        self.quantizer.train(&training_vectors)?;

        // Step 3: Assign vectors to clusters and quantize
        for (vector_id, vector) in vectors {
            // Find cluster assignment
            let results = self.clustering.search(vector, 1)?;
            if results.is_empty() {
                continue;
            }

            // Get cluster assignment from clustering index
            if let Some((coarse_id, fine_id)) = self.clustering.get_assignment(vector_id) {
                // Quantize vector to binary code
                let binary_code = self.quantizer.quantize(vector)?;

                // Store binary code
                self.cluster_binary_codes
                    .entry((coarse_id, fine_id))
                    .or_default()
                    .push(BinaryCode {
                        vector_id: vector_id.clone(),
                        code: binary_code,
                    });

                // Store vector ID
                self.cluster_vector_ids
                    .entry((coarse_id, fine_id))
                    .or_default()
                    .push(vector_id.clone());
            }
        }

        // Step 4: Optionally store original vectors for reranking
        if self.config.store_original_vectors {
            self.original_vectors = Some(
                vectors
                    .iter()
                    .map(|(id, vec)| (id.clone(), vec.clone()))
                    .collect(),
            );
        }

        // Update statistics
        let train_time = train_start.elapsed();
        self.stats.total_vectors = vectors.len();
        self.stats.num_coarse_clusters = self.config.clustering_config.num_coarse_clusters;
        self.stats.num_fine_clusters = self.config.clustering_config.num_coarse_clusters
            * self.config.clustering_config.num_fine_clusters_per_coarse;
        self.stats.num_binary_codes = self
            .cluster_binary_codes
            .values()
            .map(|codes| codes.len())
            .sum();
        self.stats.training_time_ms = train_time.as_millis();

        // Estimate memory usage
        self.stats.memory_bytes = self.estimate_memory_usage();

        self.is_trained = true;

        Ok(())
    }

    /// Search for nearest neighbors
    pub fn search(
        &mut self,
        query: &DenseVector,
        k: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        if !self.is_trained {
            return Err(crate::error::Error::InvalidInput(
                "Index must be trained before search".to_string()
            ));
        }

        let search_start = std::time::Instant::now();

        // Step 1: Find nearest clusters using hierarchical clustering
        let num_candidates = k * self.config.rerank_factor;
        let cluster_results = self.clustering.search(query, num_candidates)?;

        // Extract cluster IDs from results
        let mut candidate_cluster_ids = Vec::new();
        for (vector_id, _) in cluster_results.iter() {
            if let Some((coarse_id, fine_id)) = self.clustering.get_assignment(vector_id) {
                if !candidate_cluster_ids.contains(&(coarse_id, fine_id)) {
                    candidate_cluster_ids.push((coarse_id, fine_id));
                }
            }
        }

        // Step 2: Quantize query to binary code
        let query_binary = self.quantizer.quantize(query)?;

        // Step 3: Search within selected clusters using binary codes
        let mut candidates = Vec::new();

        for (coarse_id, fine_id) in candidate_cluster_ids {
            if let Some(binary_codes) = self.cluster_binary_codes.get(&(coarse_id, fine_id)) {
                // Compute Hamming distances
                for binary_code in binary_codes {
                    let hamming_dist = hamming_distance(&query_binary, &binary_code.code);
                    candidates.push((binary_code.vector_id.clone(), hamming_dist as f32));
                }
            }
        }

        // Step 4: Sort by Hamming distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Step 5: Optional reranking with original vectors
        let final_results = if self.config.use_reranking && self.original_vectors.is_some() {
            let rerank_count = (k * self.config.rerank_factor).min(candidates.len());
            let top_candidates = &candidates[..rerank_count];

            self.rerank_with_original_vectors(query, top_candidates)?
        } else {
            candidates
        };

        // Update statistics
        let search_time = search_start.elapsed();
        self.stats.total_searches += 1;
        let total_time = self.stats.avg_search_time_ms * (self.stats.total_searches - 1) as f64
            + search_time.as_secs_f64() * 1000.0;
        self.stats.avg_search_time_ms = total_time / self.stats.total_searches as f64;

        // Return top-k results
        Ok(final_results.into_iter().take(k).collect())
    }

    /// Rerank candidates using original full-precision vectors
    fn rerank_with_original_vectors(
        &self,
        query: &DenseVector,
        candidates: &[(String, f32)],
    ) -> crate::Result<Vec<(String, f32)>> {
        if let Some(originals) = &self.original_vectors {
            let mut reranked = Vec::new();

            for (vector_id, _) in candidates {
                if let Some(original_vector) = originals.get(vector_id) {
                    let distance = self.compute_distance(query, original_vector);
                    reranked.push((vector_id.clone(), distance));
                }
            }

            reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            Ok(reranked)
        } else {
            Ok(candidates.to_vec())
        }
    }

    /// Compute distance between two vectors
    #[inline]
    fn compute_distance(&self, a: &DenseVector, b: &DenseVector) -> f32 {
        match self.distance_metric {
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Dot => dot_product_distance(a, b),
            _ => euclidean_distance(a, b),
        }
    }

    /// Estimate memory usage in bytes
    fn estimate_memory_usage(&self) -> usize {
        let mut total = 0;

        // Binary codes memory
        for codes in self.cluster_binary_codes.values() {
            total += codes.len() * (std::mem::size_of::<BinaryCode>() + codes[0].code.len());
        }

        // Original vectors memory (if stored)
        if let Some(originals) = &self.original_vectors {
            total += originals.len() * (self.dimension * 4 + std::mem::size_of::<String>());
        }

        // Clustering overhead (~5%)
        total += (self.stats.total_vectors as f32 * 0.05) as usize;

        total
    }
}

// ============================================================================
// Distance Functions
// ============================================================================

#[inline]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[inline]
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

#[inline]
fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// Compute Hamming distance between two binary codes
#[inline]
fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as usize)
        .sum()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<(String, DenseVector)> {
        (0..n)
            .map(|i| {
                let vector: DenseVector = (0..dim)
                    .map(|j| ((i * dim + j) as f32 * 0.01) % 10.0)
                    .collect();
                (format!("vec_{}", i), vector)
            })
            .collect()
    }

    #[test]
    fn test_hybrid_index_creation() {
        let config = HybridIndexConfig::balanced();
        let index = HybridRaBitQClusteringIndex::new(config, DistanceMetric::Euclidean, 64);

        assert!(!index.is_trained());
        assert_eq!(index.dimension(), 64);
    }

    #[test]
    fn test_hybrid_index_training() {
        let config = HybridIndexConfig {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 5,
                num_fine_clusters_per_coarse: 5,
                num_coarse_probes: 2,
                num_fine_probes: 3,
                max_iterations: 10,
                tolerance: 1e-4,
                enable_parallel: false,
            },
            rabitq_config: RaBitQConfig {
                num_codebooks: 8,
                codebook_size: 256,
                training_iterations: 10,
                use_residual: true,
                enable_simd: false,
            },
            store_original_vectors: true,
            use_reranking: true,
            rerank_factor: 3,
        };

        let mut index = HybridRaBitQClusteringIndex::new(config, DistanceMetric::Euclidean, 32);
        let vectors = create_test_vectors(100, 32);

        let result = index.train(&vectors);
        assert!(result.is_ok(), "Training should succeed");
        assert!(index.is_trained());

        let stats = index.stats();
        println!("\nHybrid Index Stats:");
        println!("  Total vectors: {}", stats.total_vectors);
        println!("  Coarse clusters: {}", stats.num_coarse_clusters);
        println!("  Fine clusters: {}", stats.num_fine_clusters);
        println!("  Binary codes: {}", stats.num_binary_codes);
        println!("  Training time: {}ms", stats.training_time_ms);
        println!("  Memory: {} bytes ({:.2} MB)", stats.memory_bytes, stats.memory_bytes as f64 / 1_048_576.0);

        assert_eq!(stats.total_vectors, 100);
        assert!(stats.num_binary_codes > 0);
    }

    #[test]
    fn test_hamming_distance() {
        let a = vec![0b11110000, 0b10101010];
        let b = vec![0b11110000, 0b10101010];
        assert_eq!(hamming_distance(&a, &b), 0);

        let c = vec![0b11111111, 0b00000000];
        let d = vec![0b00000000, 0b11111111];
        assert_eq!(hamming_distance(&c, &d), 16);
    }

    #[test]
    fn test_config_presets() {
        let memory_opt = HybridIndexConfig::memory_optimized();
        assert!(!memory_opt.store_original_vectors);
        assert!(!memory_opt.use_reranking);

        let speed_opt = HybridIndexConfig::speed_optimized();
        assert!(speed_opt.store_original_vectors);
        assert!(speed_opt.use_reranking);

        let balanced = HybridIndexConfig::balanced();
        assert!(balanced.store_original_vectors);
        assert!(balanced.use_reranking);
    }

    #[test]
    fn test_hybrid_index_search() {
        let config = HybridIndexConfig {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 5,
                num_fine_clusters_per_coarse: 5,
                num_coarse_probes: 3,
                num_fine_probes: 5,
                max_iterations: 10,
                tolerance: 1e-4,
                enable_parallel: false,
            },
            rabitq_config: RaBitQConfig {
                num_codebooks: 4,
                codebook_size: 256,
                training_iterations: 10,
                use_residual: true,
                enable_simd: false,
            },
            store_original_vectors: true,
            use_reranking: true,
            rerank_factor: 3,
        };

        let mut index = HybridRaBitQClusteringIndex::new(config, DistanceMetric::Euclidean, 32);
        let vectors = create_test_vectors(200, 32);

        // Train
        index.train(&vectors).expect("Training should succeed");

        // Search
        let query = &vectors[0].1;
        let results = index.search(query, 10).expect("Search should succeed");

        println!("\nHybrid Search Results:");
        for (i, (id, dist)) in results.iter().take(5).enumerate() {
            println!("  {}. {}: {:.4}", i + 1, id, dist);
        }

        assert_eq!(results.len(), 10, "Should find 10 results");
        assert_eq!(results[0].0, "vec_0", "First result should be query itself");
        assert!(results[0].1 < 1.0, "Query should match itself closely");

        let stats = index.stats();
        println!("\nSearch Stats:");
        println!("  Avg search time: {:.2}ms", stats.avg_search_time_ms);
        println!("  Total searches: {}", stats.total_searches);

        assert_eq!(stats.total_searches, 1);
    }

    #[test]
    fn test_hybrid_without_reranking() {
        let config = HybridIndexConfig {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 5,
                num_fine_clusters_per_coarse: 5,
                num_coarse_probes: 2,
                num_fine_probes: 3,
                max_iterations: 10,
                tolerance: 1e-4,
                enable_parallel: false,
            },
            rabitq_config: RaBitQConfig {
                num_codebooks: 4,
                codebook_size: 256,
                training_iterations: 10,
                use_residual: true,
                enable_simd: false,
            },
            store_original_vectors: false, // No reranking
            use_reranking: false,
            rerank_factor: 1,
        };

        let mut index = HybridRaBitQClusteringIndex::new(config, DistanceMetric::Cosine, 32);
        let vectors = create_test_vectors(100, 32);

        index.train(&vectors).expect("Training should succeed");

        let query = &vectors[0].1;
        let results = index.search(query, 5).expect("Search should succeed");

        println!("\nHybrid Search (No Reranking):");
        println!("  Results found: {}", results.len());

        assert!(!results.is_empty(), "Should find some results");

        let stats = index.stats();
        println!("  Memory (no reranking): {} bytes ({:.2} MB)",
            stats.memory_bytes,
            stats.memory_bytes as f64 / 1_048_576.0
        );
    }

    #[test]
    fn test_hybrid_memory_savings() {
        let config_with_reranking = HybridIndexConfig {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 5,
                num_fine_clusters_per_coarse: 5,
                num_coarse_probes: 2,
                num_fine_probes: 3,
                max_iterations: 10,
                tolerance: 1e-4,
                enable_parallel: false,
            },
            rabitq_config: RaBitQConfig::default(),
            store_original_vectors: true,
            use_reranking: true,
            rerank_factor: 3,
        };

        let config_without_reranking = HybridIndexConfig {
            store_original_vectors: false,
            use_reranking: false,
            ..config_with_reranking.clone()
        };

        let vectors = create_test_vectors(100, 64);

        // Index with reranking
        let mut index_with = HybridRaBitQClusteringIndex::new(
            config_with_reranking,
            DistanceMetric::Euclidean,
            64
        );
        index_with.train(&vectors).expect("Training should succeed");

        // Index without reranking
        let mut index_without = HybridRaBitQClusteringIndex::new(
            config_without_reranking,
            DistanceMetric::Euclidean,
            64
        );
        index_without.train(&vectors).expect("Training should succeed");

        let mem_with = index_with.stats().memory_bytes;
        let mem_without = index_without.stats().memory_bytes;

        println!("\nMemory Usage Comparison:");
        println!("  With reranking: {:.2} MB", mem_with as f64 / 1_048_576.0);
        println!("  Without reranking: {:.2} MB", mem_without as f64 / 1_048_576.0);
        println!("  Savings: {:.1}%", (1.0 - mem_without as f64 / mem_with as f64) * 100.0);

        assert!(mem_without < mem_with, "Memory without reranking should be less");
    }

    #[test]
    fn test_search_before_training() {
        let config = HybridIndexConfig::balanced();
        let mut index = HybridRaBitQClusteringIndex::new(config, DistanceMetric::Euclidean, 32);

        let query = vec![0.1; 32];
        let result = index.search(&query, 10);

        assert!(result.is_err(), "Search should fail before training");
    }
}
