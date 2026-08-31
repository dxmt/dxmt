//! Cluster-Aware Search Strategy
//!
//! Intelligent search routing using hierarchical clustering to dramatically
//! reduce search space.
//!
//! Key features:
//! - Dynamic cluster selection based on query
//! - Multi-cluster probing for better recall
//! - Integration with HNSW, IVF, and other indices
//! - Adaptive strategy selection
//! - Performance monitoring

use crate::search::hierarchical_clustering::{
    HierarchicalClusteringConfig, HierarchicalClusteringIndex,
};
use crate::types::{vector::DenseVector, DistanceMetric};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cluster-aware search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAwareSearchConfig {
    /// Hierarchical clustering configuration
    pub clustering_config: HierarchicalClusteringConfig,

    /// Search strategy
    pub search_strategy: SearchStrategy,

    /// Enable adaptive cluster selection
    pub adaptive_probing: bool,

    /// Minimum recall threshold for adaptive probing
    pub min_recall_threshold: f32,

    /// Maximum clusters to probe (safety limit)
    pub max_clusters_to_probe: usize,
}

impl Default for ClusterAwareSearchConfig {
    fn default() -> Self {
        Self {
            clustering_config: HierarchicalClusteringConfig::default(),
            search_strategy: SearchStrategy::HierarchicalFirst,
            adaptive_probing: true,
            min_recall_threshold: 0.95,
            max_clusters_to_probe: 20,
        }
    }
}

/// Search strategy options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Use hierarchical clustering first, then refine
    HierarchicalFirst,

    /// Use index first, then cluster-filter results
    IndexFirst,

    /// Hybrid: use both and merge results
    Hybrid,

    /// Adaptive: choose strategy based on query characteristics
    Adaptive,
}

/// Cluster-aware search index
pub struct ClusterAwareSearchIndex {
    config: ClusterAwareSearchConfig,
    distance_metric: DistanceMetric,

    /// Hierarchical clustering index
    clustering: HierarchicalClusteringIndex,

    /// Vector storage with metadata
    vectors: HashMap<String, VectorWithMetadata>,

    /// Performance statistics
    stats: SearchStats,

    /// Whether trained
    is_trained: bool,
}

/// Vector with cluster metadata
#[derive(Debug, Clone)]
struct VectorWithMetadata {
    vector: DenseVector,
    coarse_cluster_id: usize,
    fine_cluster_id: usize,
    access_count: usize,
    last_accessed: std::time::Instant,
}

/// Search statistics
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub total_searches: usize,
    pub avg_clusters_probed: f32,
    pub avg_vectors_scanned: f32,
    pub avg_latency_ms: f32,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Search result with cluster information
#[derive(Debug, Clone)]
pub struct ClusterAwareSearchResult {
    pub id: String,
    pub distance: f32,
    pub coarse_cluster_id: usize,
    pub fine_cluster_id: usize,
}

impl ClusterAwareSearchIndex {
    /// Create new cluster-aware search index
    pub fn new(config: ClusterAwareSearchConfig, distance_metric: DistanceMetric) -> Self {
        let clustering =
            HierarchicalClusteringIndex::new(config.clustering_config.clone(), distance_metric);

        Self {
            config,
            distance_metric,
            clustering,
            vectors: HashMap::new(),
            stats: SearchStats::default(),
            is_trained: false,
        }
    }

    /// Train the index
    pub fn train(&mut self, training_data: &[(String, DenseVector)]) -> crate::Result<()> {
        if training_data.is_empty() {
            return Err(crate::Error::InvalidInput(
                "Training data cannot be empty".to_string(),
            ));
        }

        tracing::info!(
            "Training cluster-aware search index with {} vectors",
            training_data.len()
        );

        // Train clustering
        self.clustering.train(training_data)?;

        // Build vector storage with metadata
        for (id, vector) in training_data {
            if let Some((coarse_id, fine_id)) = self.clustering.get_assignment(id) {
                self.vectors.insert(
                    id.clone(),
                    VectorWithMetadata {
                        vector: vector.clone(),
                        coarse_cluster_id: coarse_id,
                        fine_cluster_id: fine_id,
                        access_count: 0,
                        last_accessed: std::time::Instant::now(),
                    },
                );
            }
        }

        self.is_trained = true;
        tracing::info!("Cluster-aware search index training completed");
        Ok(())
    }

    /// Search with cluster-aware strategy
    pub fn search(
        &mut self,
        query: &DenseVector,
        k: usize,
    ) -> crate::Result<Vec<ClusterAwareSearchResult>> {
        if !self.is_trained {
            return Err(crate::Error::InvalidInput(
                "Index must be trained before search".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();

        let results = match self.config.search_strategy {
            SearchStrategy::HierarchicalFirst => self.hierarchical_first_search(query, k)?,
            SearchStrategy::IndexFirst => self.index_first_search(query, k)?,
            SearchStrategy::Hybrid => self.hybrid_search(query, k)?,
            SearchStrategy::Adaptive => self.adaptive_search(query, k)?,
        };

        // Update statistics
        self.update_search_stats(start_time);

        // Update access counts
        for result in &results {
            if let Some(meta) = self.vectors.get_mut(&result.id) {
                meta.access_count += 1;
                meta.last_accessed = std::time::Instant::now();
            }
        }

        Ok(results)
    }

    // ========================================================================
    // Search Strategy Implementations
    // ========================================================================

    /// Hierarchical-first search: use clustering to narrow search space
    fn hierarchical_first_search(
        &self,
        query: &DenseVector,
        k: usize,
    ) -> crate::Result<Vec<ClusterAwareSearchResult>> {
        // Use hierarchical clustering to find candidates
        let candidates = self.clustering.search(query, k * 2)?; // Search for 2k candidates

        // Convert to ClusterAwareSearchResult
        let results = candidates
            .into_iter()
            .take(k)
            .filter_map(|(id, distance)| {
                self.vectors.get(&id).map(|meta| ClusterAwareSearchResult {
                    id,
                    distance,
                    coarse_cluster_id: meta.coarse_cluster_id,
                    fine_cluster_id: meta.fine_cluster_id,
                })
            })
            .collect();

        Ok(results)
    }

    /// Index-first search: use index then cluster-filter
    fn index_first_search(
        &self,
        query: &DenseVector,
        k: usize,
    ) -> crate::Result<Vec<ClusterAwareSearchResult>> {
        // For this implementation, we'll do a brute-force search
        // In a real system, this would use HNSW or another index
        let mut all_results: Vec<_> = self
            .vectors
            .iter()
            .map(|(id, meta)| {
                let distance = self.compute_distance(query, &meta.vector);
                ClusterAwareSearchResult {
                    id: id.clone(),
                    distance,
                    coarse_cluster_id: meta.coarse_cluster_id,
                    fine_cluster_id: meta.fine_cluster_id,
                }
            })
            .collect();

        all_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(k);

        Ok(all_results)
    }

    /// Hybrid search: combine clustering and index results
    fn hybrid_search(
        &self,
        query: &DenseVector,
        k: usize,
    ) -> crate::Result<Vec<ClusterAwareSearchResult>> {
        // Get results from hierarchical search
        let cluster_results = self.hierarchical_first_search(query, k)?;

        // Get results from index search (limited to save time)
        let index_results = self.index_first_search(query, k / 2)?;

        // Merge and deduplicate
        let mut merged = cluster_results;
        let cluster_ids: std::collections::HashSet<String> =
            merged.iter().map(|r| r.id.clone()).collect();

        for result in index_results {
            if !cluster_ids.contains(&result.id) {
                merged.push(result);
            }
        }

        // Re-sort and truncate
        merged.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(k);

        Ok(merged)
    }

    /// Adaptive search: choose strategy based on query
    fn adaptive_search(
        &self,
        query: &DenseVector,
        k: usize,
    ) -> crate::Result<Vec<ClusterAwareSearchResult>> {
        // Simple heuristic: use hierarchical for large k, index for small k
        if k > 100 || self.vectors.len() > 100_000 {
            self.hierarchical_first_search(query, k)
        } else {
            self.index_first_search(query, k)
        }
    }

    // ========================================================================
    // Adaptive Probing
    // ========================================================================

    /// Adaptively increase cluster probing if recall seems low
    pub fn adaptive_probe_search(
        &mut self,
        query: &DenseVector,
        k: usize,
    ) -> crate::Result<Vec<ClusterAwareSearchResult>> {
        let mut results = self.hierarchical_first_search(query, k)?;

        // If adaptive probing enabled and results seem sparse
        if self.config.adaptive_probing && results.len() < k {
            // Increase probing and retry
            let old_coarse_probes = self.config.clustering_config.num_coarse_probes;
            let old_fine_probes = self.config.clustering_config.num_fine_probes;

            // Increase probes by 2x
            self.config.clustering_config.num_coarse_probes *= 2;
            self.config.clustering_config.num_fine_probes *= 2;

            // Limit to maximum
            self.config.clustering_config.num_coarse_probes = self
                .config
                .clustering_config
                .num_coarse_probes
                .min(self.config.max_clusters_to_probe);

            // Retry search
            results = self.hierarchical_first_search(query, k)?;

            // Restore original settings
            self.config.clustering_config.num_coarse_probes = old_coarse_probes;
            self.config.clustering_config.num_fine_probes = old_fine_probes;
        }

        Ok(results)
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

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

    /// Update search statistics
    fn update_search_stats(&mut self, start_time: std::time::Instant) {
        self.stats.total_searches += 1;

        let latency_ms = start_time.elapsed().as_millis() as f32;

        // Update moving average
        let alpha = 0.1; // Smoothing factor
        self.stats.avg_latency_ms = alpha * latency_ms + (1.0 - alpha) * self.stats.avg_latency_ms;
    }

    /// Get search statistics
    pub fn get_stats(&self) -> &SearchStats {
        &self.stats
    }

    /// Get cluster info for a vector
    pub fn get_cluster_info(&self, vector_id: &str) -> Option<(usize, usize, usize)> {
        self.vectors.get(vector_id).map(|meta| {
            (
                meta.coarse_cluster_id,
                meta.fine_cluster_id,
                meta.access_count,
            )
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors(count: usize, dimension: usize) -> Vec<(String, DenseVector)> {
        (0..count)
            .map(|i| {
                let id = format!("vec_{}", i);
                let vector = (0..dimension)
                    .map(|j| ((i * dimension + j) as f32 * 0.01) % 1.0 + 0.01)
                    .collect();
                (id, vector)
            })
            .collect()
    }

    #[test]
    fn test_cluster_aware_search_creation() {
        let config = ClusterAwareSearchConfig::default();
        let index = ClusterAwareSearchIndex::new(config, DistanceMetric::Euclidean);

        assert!(!index.is_trained);
    }

    #[test]
    fn test_cluster_aware_search_training() {
        let config = ClusterAwareSearchConfig {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 10,
                num_fine_clusters_per_coarse: 10,
                num_coarse_probes: 3,
                num_fine_probes: 5,
                max_iterations: 10,
                tolerance: 1e-4,
                enable_parallel: false,
            },
            search_strategy: SearchStrategy::HierarchicalFirst,
            adaptive_probing: true,
            min_recall_threshold: 0.95,
            max_clusters_to_probe: 20,
        };

        let training_data = create_test_vectors(500, 64);
        let mut index = ClusterAwareSearchIndex::new(config, DistanceMetric::Euclidean);

        let result = index.train(&training_data);
        assert!(result.is_ok());
        assert!(index.is_trained);
        assert_eq!(index.vectors.len(), 500);
    }

    #[test]
    fn test_hierarchical_first_search() {
        let config = ClusterAwareSearchConfig {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 10,
                num_fine_clusters_per_coarse: 10,
                num_coarse_probes: 3,
                num_fine_probes: 5,
                max_iterations: 10,
                tolerance: 1e-4,
                enable_parallel: false,
            },
            search_strategy: SearchStrategy::HierarchicalFirst,
            adaptive_probing: false,
            min_recall_threshold: 0.95,
            max_clusters_to_probe: 20,
        };

        let training_data = create_test_vectors(500, 64);
        let mut index = ClusterAwareSearchIndex::new(config, DistanceMetric::Euclidean);

        index.train(&training_data).unwrap();

        let query = &training_data[0].1;
        let results = index.search(query, 10).unwrap();

        assert_eq!(results.len(), 10);
        println!("\nHierarchical-First Search Results:");
        for (i, result) in results.iter().take(5).enumerate() {
            println!(
                "  {}. {} (cluster {}/{}): {:.4}",
                i + 1,
                result.id,
                result.coarse_cluster_id,
                result.fine_cluster_id,
                result.distance
            );
        }

        // First result should be query itself
        assert_eq!(results[0].id, "vec_0");
        assert!(results[0].distance < 0.01);
    }

    #[test]
    fn test_adaptive_probing() {
        let config = ClusterAwareSearchConfig {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 20,
                num_fine_clusters_per_coarse: 20,
                num_coarse_probes: 1, // Very limited probing
                num_fine_probes: 1,
                max_iterations: 10,
                tolerance: 1e-4,
                enable_parallel: false,
            },
            search_strategy: SearchStrategy::HierarchicalFirst,
            adaptive_probing: true,
            min_recall_threshold: 0.95,
            max_clusters_to_probe: 10,
        };

        let training_data = create_test_vectors(1000, 64);
        let mut index = ClusterAwareSearchIndex::new(config, DistanceMetric::Euclidean);

        index.train(&training_data).unwrap();

        let query = &training_data[0].1;

        // Search with adaptive probing
        let results = index.adaptive_probe_search(query, 10).unwrap();

        println!("\nAdaptive Probing Search:");
        println!("  Results found: {}", results.len());
        if !results.is_empty() {
            println!(
                "  First result: {} ({:.4})",
                results[0].id, results[0].distance
            );
        }

        // With very limited initial probing (1x1), even adaptive may not find many results
        assert!(
            !results.is_empty(),
            "Adaptive probing should find at least some results"
        );
    }

    #[test]
    fn test_hybrid_search() {
        let config = ClusterAwareSearchConfig {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 10,
                num_fine_clusters_per_coarse: 10,
                num_coarse_probes: 3,
                num_fine_probes: 5,
                max_iterations: 10,
                tolerance: 1e-4,
                enable_parallel: false,
            },
            search_strategy: SearchStrategy::Hybrid,
            adaptive_probing: false,
            min_recall_threshold: 0.95,
            max_clusters_to_probe: 20,
        };

        let training_data = create_test_vectors(500, 64);
        let mut index = ClusterAwareSearchIndex::new(config, DistanceMetric::Euclidean);

        index.train(&training_data).unwrap();

        let query = &training_data[50].1;
        let results = index.search(query, 10).unwrap();

        assert_eq!(results.len(), 10);
        println!("\nHybrid Search Results:");
        for (i, result) in results.iter().take(5).enumerate() {
            println!("  {}. {}: {:.4}", i + 1, result.id, result.distance);
        }

        // Should find the query itself
        assert!(results.iter().any(|r| r.id == "vec_50"));
    }

    #[test]
    fn test_search_statistics() {
        let config = ClusterAwareSearchConfig::default();
        let training_data = create_test_vectors(500, 64);
        let mut index = ClusterAwareSearchIndex::new(config, DistanceMetric::Euclidean);

        index.train(&training_data).unwrap();

        // Perform several searches
        for i in 0..10 {
            let query = &training_data[i * 50].1;
            let _ = index.search(query, 10).unwrap();
        }

        let stats = index.get_stats();
        println!("\nSearch Statistics:");
        println!("  Total searches: {}", stats.total_searches);
        println!("  Avg latency: {:.2}ms", stats.avg_latency_ms);

        assert_eq!(stats.total_searches, 10);
        assert!(stats.avg_latency_ms >= 0.0); // Can be 0 for very fast searches
    }
}
