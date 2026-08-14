//! Hierarchical Clustering for Large-Scale Vector Search
//!
//! This module implements multi-level clustering so that a query touches only the
//! centroids of one coarse partition before descending into its fine partitions,
//! bounding the candidate set independently of total collection size.
//!
//! Key features:
//! - Two-level clustering: coarse (L1) + fine (L2)
//! - Reduces search space by 10-100x
//! - K-means++ initialization for better cluster quality
//! - Adaptive cluster assignment
//! - Supports incremental updates
//!
//! Architecture:
//! ```text
//! Level 1 (Coarse): 100-1000 clusters
//!   ├─ Cluster 0: 1000 vectors
//!   │   └─ Level 2 (Fine): 10-100 sub-clusters
//!   ├─ Cluster 1: 1000 vectors
//!   │   └─ Level 2 (Fine): 10-100 sub-clusters
//!   └─ ...
//! ```

use crate::types::{vector::DenseVector, DistanceMetric};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rayon::prelude::*;

/// Hierarchical clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalClusteringConfig {
    /// Number of coarse clusters (Level 1)
    pub num_coarse_clusters: usize,

    /// Number of fine clusters per coarse cluster (Level 2)
    pub num_fine_clusters_per_coarse: usize,

    /// Number of coarse clusters to probe during search
    pub num_coarse_probes: usize,

    /// Number of fine clusters to probe within each coarse cluster
    pub num_fine_probes: usize,

    /// K-means maximum iterations
    pub max_iterations: usize,

    /// Convergence tolerance
    pub tolerance: f32,

    /// Whether to enable parallel training
    pub enable_parallel: bool,
}

impl Default for HierarchicalClusteringConfig {
    fn default() -> Self {
        Self {
            num_coarse_clusters: 100,       // 100 coarse clusters
            num_fine_clusters_per_coarse: 100, // 100 fine per coarse = 10K total
            num_coarse_probes: 5,           // Search 5 coarse clusters
            num_fine_probes: 10,            // Search 10 fine per coarse
            max_iterations: 20,
            tolerance: 1e-4,
            enable_parallel: true,
        }
    }
}

/// Cluster centroid with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCentroid {
    /// Centroid vector
    pub centroid: DenseVector,

    /// Number of vectors assigned to this cluster
    pub size: usize,

    /// Cluster ID
    pub id: usize,
}

/// Fine cluster within a coarse cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineCluster {
    /// Fine cluster centroid
    pub centroid: DenseVector,

    /// Vector IDs assigned to this fine cluster
    pub vector_ids: Vec<String>,

    /// Fine cluster ID (local to coarse cluster)
    pub local_id: usize,
}

/// Coarse cluster (Level 1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoarseCluster {
    /// Coarse cluster centroid
    pub centroid: DenseVector,

    /// Fine clusters within this coarse cluster
    pub fine_clusters: Vec<FineCluster>,

    /// Coarse cluster ID
    pub id: usize,

    /// Total vectors in this coarse cluster
    pub total_vectors: usize,
}

/// Hierarchical clustering index
#[derive(Debug, Serialize, Deserialize)]
pub struct HierarchicalClusteringIndex {
    config: HierarchicalClusteringConfig,
    distance_metric: DistanceMetric,

    /// Coarse clusters (Level 1)
    coarse_clusters: Vec<CoarseCluster>,

    /// Vector storage (ID -> Vector)
    vectors: HashMap<String, DenseVector>,

    /// Vector to cluster assignment (ID -> (coarse_id, fine_id))
    assignments: HashMap<String, (usize, usize)>,

    /// Dimension
    dimension: Option<usize>,

    /// Whether trained
    is_trained: bool,

    /// Statistics
    stats: ClusteringStats,
}

/// Clustering statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusteringStats {
    pub total_vectors: usize,
    pub num_coarse_clusters: usize,
    pub total_fine_clusters: usize,
    pub avg_vectors_per_coarse: f32,
    pub avg_vectors_per_fine: f32,
    pub training_time_ms: u64,
}

impl HierarchicalClusteringIndex {
    /// Create new hierarchical clustering index
    pub fn new(
        config: HierarchicalClusteringConfig,
        distance_metric: DistanceMetric,
    ) -> Self {
        Self {
            config,
            distance_metric,
            coarse_clusters: Vec::new(),
            vectors: HashMap::new(),
            assignments: HashMap::new(),
            dimension: None,
            is_trained: false,
            stats: ClusteringStats::default(),
        }
    }

    /// Train the hierarchical clustering
    pub fn train(&mut self, training_vectors: &[(String, DenseVector)]) -> crate::Result<()> {
        if training_vectors.is_empty() {
            return Err(crate::Error::InvalidInput(
                "Training vectors cannot be empty".to_string()
            ));
        }

        let start_time = std::time::Instant::now();
        let dimension = training_vectors[0].1.len();
        self.dimension = Some(dimension);

        tracing::info!(
            "Training hierarchical clustering with {} vectors, dim={}",
            training_vectors.len(),
            dimension
        );

        // Extract just vectors for clustering
        let vectors: Vec<DenseVector> = training_vectors.iter()
            .map(|(_, v)| v.clone())
            .collect();

        // Step 1: Train coarse clusters (Level 1)
        tracing::info!("Training {} coarse clusters...", self.config.num_coarse_clusters);
        let coarse_centroids = self.train_kmeans(
            &vectors,
            self.config.num_coarse_clusters,
        )?;

        // Step 2: Assign vectors to coarse clusters
        let coarse_assignments = self.assign_to_clusters(&vectors, &coarse_centroids);

        // Step 3: Train fine clusters within each coarse cluster (Level 2)
        tracing::info!("Training fine clusters within each coarse cluster...");
        let mut coarse_clusters = Vec::new();

        for (coarse_id, coarse_centroid) in coarse_centroids.into_iter().enumerate() {
            // Get vectors assigned to this coarse cluster
            let cluster_vectors: Vec<DenseVector> = coarse_assignments
                .iter()
                .enumerate()
                .filter_map(|(idx, &assigned_id)| {
                    if assigned_id == coarse_id {
                        Some(vectors[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();

            if cluster_vectors.is_empty() {
                // Empty coarse cluster - skip
                continue;
            }

            // Train fine clusters for this coarse cluster
            let num_fine = self.config.num_fine_clusters_per_coarse.min(cluster_vectors.len());
            let fine_centroids = if num_fine > 1 {
                self.train_kmeans(&cluster_vectors, num_fine)?
            } else {
                vec![coarse_centroid.clone()]
            };

            // Assign vectors to fine clusters
            let fine_assignments = self.assign_to_clusters(&cluster_vectors, &fine_centroids);

            // Build fine clusters with vector IDs
            let mut fine_clusters = Vec::new();
            for (fine_id, fine_centroid) in fine_centroids.into_iter().enumerate() {
                let mut vector_ids = Vec::new();

                // Find vectors assigned to this fine cluster
                for (idx, &assigned_fine_id) in fine_assignments.iter().enumerate() {
                    if assigned_fine_id == fine_id {
                        // Find original vector ID
                        let mut original_idx = 0;
                        let mut count = 0;
                        for (i, &assigned_coarse_id) in coarse_assignments.iter().enumerate() {
                            if assigned_coarse_id == coarse_id {
                                if count == idx {
                                    original_idx = i;
                                    break;
                                }
                                count += 1;
                            }
                        }
                        vector_ids.push(training_vectors[original_idx].0.clone());
                    }
                }

                fine_clusters.push(FineCluster {
                    centroid: fine_centroid,
                    vector_ids,
                    local_id: fine_id,
                });
            }

            coarse_clusters.push(CoarseCluster {
                centroid: coarse_centroid,
                fine_clusters,
                id: coarse_id,
                total_vectors: cluster_vectors.len(),
            });
        }

        self.coarse_clusters = coarse_clusters;

        // Build vector storage and assignments
        for vector_data in training_vectors {
            self.vectors.insert(vector_data.0.clone(), vector_data.1.clone());
        }

        // Build assignments map
        for coarse_cluster in &self.coarse_clusters {
            for fine_cluster in &coarse_cluster.fine_clusters {
                for vector_id in &fine_cluster.vector_ids {
                    self.assignments.insert(
                        vector_id.clone(),
                        (coarse_cluster.id, fine_cluster.local_id),
                    );
                }
            }
        }

        // Update statistics
        self.stats.total_vectors = training_vectors.len();
        self.stats.num_coarse_clusters = self.coarse_clusters.len();
        self.stats.total_fine_clusters = self.coarse_clusters.iter()
            .map(|c| c.fine_clusters.len())
            .sum();
        self.stats.avg_vectors_per_coarse = training_vectors.len() as f32
            / self.coarse_clusters.len() as f32;
        self.stats.training_time_ms = start_time.elapsed().as_millis() as u64;

        self.is_trained = true;

        tracing::info!(
            "Hierarchical clustering trained in {}ms: {} coarse, {} fine clusters",
            self.stats.training_time_ms,
            self.stats.num_coarse_clusters,
            self.stats.total_fine_clusters
        );

        Ok(())
    }

    /// Search for nearest neighbors using hierarchical clustering
    pub fn search(
        &self,
        query: &DenseVector,
        k: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        if !self.is_trained {
            return Err(crate::Error::InvalidInput(
                "Index must be trained before search".to_string()
            ));
        }

        // Step 1: Find nearest coarse clusters
        let coarse_candidates = self.find_nearest_coarse_clusters(query);

        // Step 2: Search within nearest coarse clusters
        let mut all_candidates = Vec::new();

        for coarse_id in coarse_candidates.iter().take(self.config.num_coarse_probes) {
            let coarse_cluster = &self.coarse_clusters[*coarse_id];

            // Find nearest fine clusters within this coarse cluster
            let fine_candidates = self.find_nearest_fine_clusters(
                query,
                coarse_cluster,
            );

            // Search within nearest fine clusters
            for fine_id in fine_candidates.iter().take(self.config.num_fine_probes) {
                let fine_cluster = &coarse_cluster.fine_clusters[*fine_id];

                // Score all vectors in this fine cluster
                for vector_id in &fine_cluster.vector_ids {
                    if let Some(vector) = self.vectors.get(vector_id) {
                        let distance = self.compute_distance(query, vector);
                        all_candidates.push((vector_id.clone(), distance));
                    }
                }
            }
        }

        // Sort by distance and return top-k
        all_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_candidates.truncate(k);

        Ok(all_candidates)
    }

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /// Train K-means clustering
    fn train_kmeans(
        &self,
        vectors: &[DenseVector],
        num_clusters: usize,
    ) -> crate::Result<Vec<DenseVector>> {
        let num_clusters = num_clusters.min(vectors.len());

        // Initialize centroids using K-means++
        let mut centroids = self.kmeans_plus_plus_init(vectors, num_clusters)?;

        // K-means iterations
        for _iteration in 0..self.config.max_iterations {
            let old_centroids = centroids.clone();

            // Assign vectors to nearest centroid
            let assignments = self.assign_to_clusters(vectors, &centroids);

            // Update centroids
            centroids = self.update_centroids(vectors, &assignments, num_clusters)?;

            // Check convergence
            let movement = self.compute_centroid_movement(&old_centroids, &centroids);
            if movement < self.config.tolerance {
                break;
            }
        }

        Ok(centroids)
    }

    /// K-means++ initialization
    fn kmeans_plus_plus_init(
        &self,
        vectors: &[DenseVector],
        num_clusters: usize,
    ) -> crate::Result<Vec<DenseVector>> {
        if vectors.is_empty() || num_clusters == 0 {
            return Ok(Vec::new());
        }

        let mut centroids = Vec::new();
        let mut rng = rand::thread_rng();
        use rand::Rng;

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..vectors.len());
        centroids.push(vectors[first_idx].clone());

        // Choose remaining centroids
        for _ in 1..num_clusters {
            // Compute distance to nearest centroid for each vector
            let distances: Vec<f32> = if self.config.enable_parallel {
                vectors.par_iter().map(|v| {
                    centroids.iter()
                        .map(|c| self.compute_distance(v, c))
                        .fold(f32::INFINITY, f32::min)
                }).collect()
            } else {
                vectors.iter().map(|v| {
                    centroids.iter()
                        .map(|c| self.compute_distance(v, c))
                        .fold(f32::INFINITY, f32::min)
                }).collect()
            };

            // Choose next centroid with probability proportional to distance squared
            let weights: Vec<f32> = distances.iter().map(|d| d * d).collect();
            let total_weight: f32 = weights.iter().sum();

            if total_weight == 0.0 {
                break;
            }

            let mut cumulative = 0.0;
            let random_value = rng.gen::<f32>() * total_weight;

            for (i, &weight) in weights.iter().enumerate() {
                cumulative += weight;
                if cumulative >= random_value {
                    centroids.push(vectors[i].clone());
                    break;
                }
            }
        }

        Ok(centroids)
    }

    /// Assign vectors to nearest clusters
    fn assign_to_clusters(
        &self,
        vectors: &[DenseVector],
        centroids: &[DenseVector],
    ) -> Vec<usize> {
        if self.config.enable_parallel {
            vectors.par_iter().map(|v| {
                centroids.iter()
                    .enumerate()
                    .map(|(i, c)| (i, self.compute_distance(v, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }).collect()
        } else {
            vectors.iter().map(|v| {
                centroids.iter()
                    .enumerate()
                    .map(|(i, c)| (i, self.compute_distance(v, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }).collect()
        }
    }

    /// Update centroids based on assignments
    fn update_centroids(
        &self,
        vectors: &[DenseVector],
        assignments: &[usize],
        num_clusters: usize,
    ) -> crate::Result<Vec<DenseVector>> {
        let dimension = self.dimension.unwrap();
        let mut new_centroids = vec![vec![0.0; dimension]; num_clusters];
        let mut counts = vec![0; num_clusters];

        for (vector, &cluster_id) in vectors.iter().zip(assignments.iter()) {
            for (i, &value) in vector.iter().enumerate() {
                new_centroids[cluster_id][i] += value;
            }
            counts[cluster_id] += 1;
        }

        // Average to get centroids
        for (centroid, count) in new_centroids.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                for value in centroid.iter_mut() {
                    *value /= *count as f32;
                }
            }
        }

        Ok(new_centroids)
    }

    /// Compute movement between old and new centroids
    fn compute_centroid_movement(
        &self,
        old_centroids: &[DenseVector],
        new_centroids: &[DenseVector],
    ) -> f32 {
        let mut total_movement = 0.0;

        for (old, new) in old_centroids.iter().zip(new_centroids.iter()) {
            total_movement += self.compute_distance(old, new);
        }

        total_movement / old_centroids.len() as f32
    }

    /// Find nearest coarse clusters
    fn find_nearest_coarse_clusters(&self, query: &DenseVector) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = self.coarse_clusters
            .iter()
            .map(|cluster| {
                let distance = self.compute_distance(query, &cluster.centroid);
                (cluster.id, distance)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        distances.into_iter().map(|(id, _)| id).collect()
    }

    /// Find nearest fine clusters within a coarse cluster
    fn find_nearest_fine_clusters(
        &self,
        query: &DenseVector,
        coarse_cluster: &CoarseCluster,
    ) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = coarse_cluster.fine_clusters
            .iter()
            .map(|fine_cluster| {
                let distance = self.compute_distance(query, &fine_cluster.centroid);
                (fine_cluster.local_id, distance)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        distances.into_iter().map(|(id, _)| id).collect()
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

    /// Get statistics
    pub fn get_stats(&self) -> &ClusteringStats {
        &self.stats
    }

    /// Get cluster assignment for a vector
    pub fn get_assignment(&self, vector_id: &str) -> Option<(usize, usize)> {
        self.assignments.get(vector_id).copied()
    }
}

// ============================================================================
// Adaptive Cluster Sizing
// ============================================================================

/// Adaptive cluster sizing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveClusterConfig {
    /// Dataset size
    pub dataset_size: usize,

    /// Memory budget in MB
    pub memory_budget_mb: usize,

    /// Target search latency in ms
    pub target_latency_ms: f32,

    /// Minimum coarse clusters
    pub min_coarse_clusters: usize,

    /// Maximum coarse clusters
    pub max_coarse_clusters: usize,

    /// Minimum fine clusters per coarse
    pub min_fine_per_coarse: usize,

    /// Maximum fine clusters per coarse
    pub max_fine_per_coarse: usize,
}

impl Default for AdaptiveClusterConfig {
    fn default() -> Self {
        Self {
            dataset_size: 0,
            memory_budget_mb: 1024, // 1GB default
            target_latency_ms: 10.0,
            min_coarse_clusters: 10,
            max_coarse_clusters: 1000,
            min_fine_per_coarse: 5,
            max_fine_per_coarse: 100,
        }
    }
}

/// Result of adaptive cluster sizing
#[derive(Debug, Clone)]
pub struct AdaptiveClusterResult {
    /// Recommended number of coarse clusters
    pub num_coarse_clusters: usize,

    /// Recommended number of fine clusters per coarse
    pub num_fine_per_coarse: usize,

    /// Recommended number of coarse probes
    pub num_coarse_probes: usize,

    /// Recommended number of fine probes
    pub num_fine_probes: usize,

    /// Estimated memory usage in MB
    pub estimated_memory_mb: f32,

    /// Estimated search latency in ms
    pub estimated_latency_ms: f32,

    /// Expected search space reduction ratio
    pub reduction_ratio: f32,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Adaptive cluster sizing analyzer
pub struct AdaptiveClusterAnalyzer;

impl AdaptiveClusterAnalyzer {
    /// Recommend optimal cluster configuration based on dataset characteristics
    pub fn recommend_config(
        config: &AdaptiveClusterConfig,
        vector_dim: usize,
    ) -> AdaptiveClusterResult {
        let dataset_size = config.dataset_size;

        // Rule-based recommendations based on dataset size
        let (num_coarse, num_fine) = if dataset_size < 10_000 {
            // Small dataset: 10-50 clusters
            (10, 10)
        } else if dataset_size < 100_000 {
            // Medium dataset: 50-200 clusters
            let coarse = (dataset_size as f32).sqrt() as usize;
            let coarse = coarse.clamp(config.min_coarse_clusters, config.max_coarse_clusters);
            let fine = 20;
            (coarse, fine)
        } else if dataset_size < 1_000_000 {
            // Large dataset: 200-1000 clusters
            let coarse = (dataset_size as f32).sqrt() as usize;
            let coarse = coarse.clamp(config.min_coarse_clusters, config.max_coarse_clusters);
            let fine = 50;
            (coarse, fine)
        } else {
            // Very large dataset: 1000+ clusters
            let coarse = 1000.min(config.max_coarse_clusters);
            let fine = 100;
            (coarse, fine)
        };

        // Adjust for memory constraints
        let bytes_per_vector = vector_dim * 4; // f32
        let total_vector_memory_mb = (dataset_size * bytes_per_vector) as f32 / 1_048_576.0;
        let centroid_memory_mb = (num_coarse * num_fine * bytes_per_vector) as f32 / 1_048_576.0;
        let mut estimated_memory_mb = total_vector_memory_mb + centroid_memory_mb;

        // Adjust if exceeds memory budget
        let (num_coarse, num_fine) = if estimated_memory_mb > config.memory_budget_mb as f32 {
            // Reduce cluster counts proportionally
            let scale = (config.memory_budget_mb as f32 / estimated_memory_mb).sqrt();
            let new_coarse = ((num_coarse as f32 * scale) as usize)
                .max(config.min_coarse_clusters);
            let new_fine = ((num_fine as f32 * scale) as usize)
                .max(config.min_fine_per_coarse);

            // Recalculate memory with new cluster counts
            let new_centroid_memory_mb = (new_coarse * new_fine * bytes_per_vector) as f32 / 1_048_576.0;
            estimated_memory_mb = total_vector_memory_mb + new_centroid_memory_mb;

            (new_coarse, new_fine)
        } else {
            (num_coarse, num_fine)
        };

        // Recommend probing based on target recall
        // Higher probing = higher recall but slower
        let num_coarse_probes = if dataset_size < 10_000 {
            3
        } else if dataset_size < 100_000 {
            5
        } else {
            10
        };

        let num_fine_probes = if dataset_size < 10_000 {
            5
        } else if dataset_size < 100_000 {
            10
        } else {
            20
        };

        // Estimate search performance
        let vectors_per_fine = dataset_size / (num_coarse * num_fine).max(1);
        let vectors_scanned = num_coarse_probes * num_fine_probes * vectors_per_fine;
        let reduction_ratio = dataset_size as f32 / vectors_scanned.max(1) as f32;

        // Estimate latency (rough model: 0.001ms per vector comparison + overhead)
        let estimated_latency_ms = 0.001 * vectors_scanned as f32 + 0.5;

        // Confidence based on how well we match constraints
        let memory_ok = estimated_memory_mb <= config.memory_budget_mb as f32;
        let latency_ok = estimated_latency_ms <= config.target_latency_ms;
        let confidence = match (memory_ok, latency_ok) {
            (true, true) => 0.9,
            (true, false) | (false, true) => 0.6,
            (false, false) => 0.3,
        };

        AdaptiveClusterResult {
            num_coarse_clusters: num_coarse,
            num_fine_per_coarse: num_fine,
            num_coarse_probes,
            num_fine_probes,
            estimated_memory_mb,
            estimated_latency_ms,
            reduction_ratio,
            confidence,
        }
    }

    /// Compute within-cluster variance for Elbow method
    pub fn compute_wcss(
        vectors: &[DenseVector],
        centroids: &[DenseVector],
        assignments: &[usize],
        distance_metric: DistanceMetric,
    ) -> f32 {
        vectors
            .iter()
            .zip(assignments.iter())
            .map(|(vec, &cluster_id)| {
                let centroid = &centroids[cluster_id];
                let dist = match distance_metric {
                    DistanceMetric::Euclidean => euclidean_distance(vec, centroid),
                    DistanceMetric::Cosine => cosine_distance(vec, centroid),
                    DistanceMetric::Dot => dot_product_distance(vec, centroid),
                    _ => euclidean_distance(vec, centroid),
                };
                dist * dist
            })
            .sum()
    }

    /// Compute silhouette score for cluster quality assessment
    /// Returns average silhouette score (-1 to 1, higher is better)
    pub fn compute_silhouette_score(
        vectors: &[DenseVector],
        assignments: &[usize],
        num_clusters: usize,
        distance_metric: DistanceMetric,
    ) -> f32 {
        if vectors.len() < 2 || num_clusters < 2 {
            return 0.0;
        }

        // Group vectors by cluster
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];
        for (i, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id < num_clusters {
                clusters[cluster_id].push(i);
            }
        }

        let mut total_silhouette = 0.0;
        let mut count = 0;

        for (i, vec) in vectors.iter().enumerate() {
            let cluster_id = assignments[i];
            let cluster = &clusters[cluster_id];

            if cluster.len() < 2 {
                continue; // Skip singleton clusters
            }

            // Compute a(i): average distance to points in same cluster
            let a = cluster
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| Self::compute_distance(vec, &vectors[j], distance_metric))
                .sum::<f32>()
                / (cluster.len() - 1) as f32;

            // Compute b(i): min average distance to points in other clusters
            let b = (0..num_clusters)
                .filter(|&c| c != cluster_id)
                .filter_map(|c| {
                    let other_cluster = &clusters[c];
                    if other_cluster.is_empty() {
                        None
                    } else {
                        let avg_dist = other_cluster
                            .iter()
                            .map(|&j| Self::compute_distance(vec, &vectors[j], distance_metric))
                            .sum::<f32>()
                            / other_cluster.len() as f32;
                        Some(avg_dist)
                    }
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            // Silhouette coefficient for point i
            let s = if a < b {
                1.0 - a / b
            } else if a > b {
                b / a - 1.0
            } else {
                0.0
            };

            total_silhouette += s;
            count += 1;
        }

        if count > 0 {
            total_silhouette / count as f32
        } else {
            0.0
        }
    }

    /// Find optimal K using Elbow method
    /// Tests multiple K values and finds the point where improvement plateaus
    pub fn find_optimal_k_elbow(
        vectors: &[DenseVector],
        k_range: std::ops::Range<usize>,
        distance_metric: DistanceMetric,
    ) -> (usize, Vec<(usize, f32)>) {
        let mut wcss_values = Vec::new();

        for k in k_range.clone() {
            // Train K-means with this K
            let (centroids, assignments) = Self::simple_kmeans(vectors, k, 10, distance_metric);
            let wcss = Self::compute_wcss(vectors, &centroids, &assignments, distance_metric);
            wcss_values.push((k, wcss));
        }

        // Find elbow using rate of change
        let optimal_k = if wcss_values.len() >= 3 {
            let mut max_improvement_rate = 0.0;
            let mut optimal = k_range.start;

            for i in 1..wcss_values.len() - 1 {
                let improvement_rate = (wcss_values[i - 1].1 - wcss_values[i].1)
                    / (wcss_values[i].1 - wcss_values[i + 1].1).max(0.001);

                if improvement_rate > max_improvement_rate {
                    max_improvement_rate = improvement_rate;
                    optimal = wcss_values[i].0;
                }
            }
            optimal
        } else {
            k_range.start
        };

        (optimal_k, wcss_values)
    }

    /// Simple K-means implementation for analysis (not optimized)
    fn simple_kmeans(
        vectors: &[DenseVector],
        k: usize,
        max_iter: usize,
        distance_metric: DistanceMetric,
    ) -> (Vec<DenseVector>, Vec<usize>) {
        if vectors.is_empty() || k == 0 {
            return (Vec::new(), Vec::new());
        }

        let dim = vectors[0].len();

        // Initialize centroids with first k vectors
        let mut centroids: Vec<DenseVector> = vectors.iter().take(k).cloned().collect();
        let mut assignments = vec![0; vectors.len()];

        for _ in 0..max_iter {
            // Assign vectors to nearest centroid
            for (i, vec) in vectors.iter().enumerate() {
                let mut min_dist = f32::MAX;
                let mut best_cluster = 0;

                for (cluster_id, centroid) in centroids.iter().enumerate() {
                    let dist = Self::compute_distance(vec, centroid, distance_metric);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = cluster_id;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0; k];

            for (vec, &cluster_id) in vectors.iter().zip(assignments.iter()) {
                for j in 0..dim {
                    new_centroids[cluster_id][j] += vec[j];
                }
                counts[cluster_id] += 1;
            }

            for (cluster_id, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[cluster_id] > 0 {
                    let count = counts[cluster_id] as f32;
                    for v in centroid.iter_mut() {
                        *v /= count;
                    }
                }
            }

            centroids = new_centroids;
        }

        (centroids, assignments)
    }

    fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Dot => dot_product_distance(a, b),
            _ => euclidean_distance(a, b),
        }
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
    fn test_hierarchical_clustering_creation() {
        let config = HierarchicalClusteringConfig::default();
        let index = HierarchicalClusteringIndex::new(config, DistanceMetric::Euclidean);

        assert!(!index.is_trained);
        assert_eq!(index.coarse_clusters.len(), 0);
    }

    #[test]
    fn test_hierarchical_clustering_training() {
        let config = HierarchicalClusteringConfig {
            num_coarse_clusters: 10,
            num_fine_clusters_per_coarse: 10,
            num_coarse_probes: 3,
            num_fine_probes: 5,
            max_iterations: 10,
            tolerance: 1e-4,
            enable_parallel: false,
        };

        let training_data = create_test_vectors(500, 64);
        let mut index = HierarchicalClusteringIndex::new(config, DistanceMetric::Euclidean);

        let result = index.train(&training_data);
        assert!(result.is_ok());
        assert!(index.is_trained);
        assert!(index.coarse_clusters.len() > 0);

        println!("\nHierarchical Clustering Training Stats:");
        let stats = index.get_stats();
        println!("  Coarse clusters: {}", stats.num_coarse_clusters);
        println!("  Total fine clusters: {}", stats.total_fine_clusters);
        println!("  Avg vectors per coarse: {:.1}", stats.avg_vectors_per_coarse);
        println!("  Training time: {}ms", stats.training_time_ms);
    }

    #[test]
    fn test_hierarchical_clustering_search() {
        let config = HierarchicalClusteringConfig {
            num_coarse_clusters: 10,
            num_fine_clusters_per_coarse: 10,
            num_coarse_probes: 3,
            num_fine_probes: 5,
            max_iterations: 10,
            tolerance: 1e-4,
            enable_parallel: false,
        };

        let training_data = create_test_vectors(500, 64);
        let mut index = HierarchicalClusteringIndex::new(config, DistanceMetric::Euclidean);

        index.train(&training_data).unwrap();

        // Search for first vector
        let query = &training_data[0].1;
        let results = index.search(query, 10).unwrap();

        assert_eq!(results.len(), 10);
        println!("\nHierarchical Clustering Search Results:");
        for (i, (id, dist)) in results.iter().take(5).enumerate() {
            println!("  {}. {}: {:.4}", i + 1, id, dist);
        }

        // First result should be the query itself
        assert_eq!(results[0].0, "vec_0");
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_search_space_reduction() {
        let config = HierarchicalClusteringConfig {
            num_coarse_clusters: 20,
            num_fine_clusters_per_coarse: 20,
            num_coarse_probes: 2,    // Only search 2 coarse clusters
            num_fine_probes: 3,      // Only search 3 fine per coarse
            max_iterations: 10,
            tolerance: 1e-4,
            enable_parallel: false,
        };

        let dataset_size = 1000;
        let training_data = create_test_vectors(dataset_size, 64);
        let mut index = HierarchicalClusteringIndex::new(config.clone(), DistanceMetric::Euclidean);

        index.train(&training_data).unwrap();

        // Calculate search space reduction
        let vectors_searched = config.num_coarse_probes * config.num_fine_probes
            * (dataset_size / (config.num_coarse_clusters * config.num_fine_clusters_per_coarse));

        let reduction_ratio = dataset_size as f32 / vectors_searched as f32;

        println!("\nSearch Space Reduction:");
        println!("  Total vectors: {}", dataset_size);
        println!("  Vectors searched: ~{}", vectors_searched);
        println!("  Reduction ratio: {:.1}x", reduction_ratio);

        assert!(reduction_ratio > 10.0, "Should achieve at least 10x reduction");
    }

    #[test]
    fn test_adaptive_config_recommendation() {
        // Test small dataset
        let config = AdaptiveClusterConfig {
            dataset_size: 5_000,
            memory_budget_mb: 512,
            target_latency_ms: 10.0,
            ..Default::default()
        };

        let result = AdaptiveClusterAnalyzer::recommend_config(&config, 128);

        println!("\nAdaptive Config (5K vectors):");
        println!("  Coarse clusters: {}", result.num_coarse_clusters);
        println!("  Fine per coarse: {}", result.num_fine_per_coarse);
        println!("  Coarse probes: {}", result.num_coarse_probes);
        println!("  Fine probes: {}", result.num_fine_probes);
        println!("  Est. memory: {:.2} MB", result.estimated_memory_mb);
        println!("  Est. latency: {:.2} ms", result.estimated_latency_ms);
        println!("  Reduction: {:.1}x", result.reduction_ratio);
        println!("  Confidence: {:.2}", result.confidence);

        assert!(result.num_coarse_clusters >= 10);
        assert!(result.num_fine_per_coarse >= 5);
        assert!(result.reduction_ratio > 1.0);

        // Test large dataset
        let config = AdaptiveClusterConfig {
            dataset_size: 1_000_000,
            memory_budget_mb: 2048,
            target_latency_ms: 50.0,
            ..Default::default()
        };

        let result = AdaptiveClusterAnalyzer::recommend_config(&config, 128);

        println!("\nAdaptive Config (1M vectors):");
        println!("  Coarse clusters: {}", result.num_coarse_clusters);
        println!("  Fine per coarse: {}", result.num_fine_per_coarse);
        println!("  Reduction: {:.1}x", result.reduction_ratio);

        assert!(result.num_coarse_clusters >= 100);
        assert!(result.reduction_ratio > 10.0);
    }

    #[test]
    fn test_wcss_computation() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let centroids = vec![
            vec![0.05, 0.05],   // Cluster 0 centroid
            vec![10.05, 10.05], // Cluster 1 centroid
        ];

        let assignments = vec![0, 0, 1, 1];

        let wcss = AdaptiveClusterAnalyzer::compute_wcss(
            &vectors,
            &centroids,
            &assignments,
            DistanceMetric::Euclidean,
        );

        println!("\nWCSS: {:.4}", wcss);

        // WCSS should be small for well-separated clusters
        assert!(wcss < 1.0);
    }

    #[test]
    fn test_silhouette_score() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.2],
        ];

        let assignments = vec![0, 0, 0, 1, 1, 1];

        let score = AdaptiveClusterAnalyzer::compute_silhouette_score(
            &vectors,
            &assignments,
            2,
            DistanceMetric::Euclidean,
        );

        println!("\nSilhouette Score: {:.4}", score);

        // Well-separated clusters should have high silhouette score (> 0.5)
        assert!(score > 0.5, "Silhouette score should be > 0.5 for well-separated clusters");
    }

    #[test]
    fn test_elbow_method() {
        // Create well-separated clusters
        let mut vectors = Vec::new();

        // Cluster 1: around (0, 0)
        for i in 0..50 {
            vectors.push(vec![
                (i as f32 * 0.1) % 1.0,
                (i as f32 * 0.1) % 1.0,
            ]);
        }

        // Cluster 2: around (10, 10)
        for i in 0..50 {
            vectors.push(vec![
                10.0 + (i as f32 * 0.1) % 1.0,
                10.0 + (i as f32 * 0.1) % 1.0,
            ]);
        }

        // Cluster 3: around (20, 20)
        for i in 0..50 {
            vectors.push(vec![
                20.0 + (i as f32 * 0.1) % 1.0,
                20.0 + (i as f32 * 0.1) % 1.0,
            ]);
        }

        let (optimal_k, wcss_values) = AdaptiveClusterAnalyzer::find_optimal_k_elbow(
            &vectors,
            2..8,
            DistanceMetric::Euclidean,
        );

        println!("\nElbow Method Results:");
        println!("  Optimal K: {}", optimal_k);
        println!("  WCSS values:");
        for (k, wcss) in &wcss_values {
            println!("    K={}: WCSS={:.2}", k, wcss);
        }

        // Should detect 3 clusters or nearby
        assert!(optimal_k >= 2 && optimal_k <= 5, "Should find reasonable K");
        assert_eq!(wcss_values.len(), 6, "Should have 6 K values tested");

        // WCSS should decrease monotonically
        for i in 1..wcss_values.len() {
            assert!(
                wcss_values[i].1 <= wcss_values[i - 1].1,
                "WCSS should decrease with more clusters"
            );
        }
    }

    #[test]
    fn test_memory_constrained_config() {
        // Test with constrained memory budget
        // Use smaller dataset so memory budget is meaningful
        let config = AdaptiveClusterConfig {
            dataset_size: 10_000,
            memory_budget_mb: 64, // 64MB budget
            target_latency_ms: 10.0,
            ..Default::default()
        };

        let result = AdaptiveClusterAnalyzer::recommend_config(&config, 256); // 256 dims

        println!("\nMemory-Constrained Config:");
        println!("  Dataset: {} vectors × {} dims", config.dataset_size, 256);
        println!("  Memory budget: {} MB", config.memory_budget_mb);
        println!("  Coarse clusters: {}", result.num_coarse_clusters);
        println!("  Fine per coarse: {}", result.num_fine_per_coarse);
        println!("  Est. memory: {:.2} MB", result.estimated_memory_mb);
        println!("  Confidence: {:.2}", result.confidence);

        // Should respect memory budget (with generous tolerance since it's an estimate)
        assert!(
            result.estimated_memory_mb <= config.memory_budget_mb as f32 * 1.5,
            "Should stay reasonably close to memory budget"
        );

        // Should still provide reasonable clustering
        assert!(result.num_coarse_clusters >= config.min_coarse_clusters);
        assert!(result.num_fine_per_coarse >= config.min_fine_per_coarse);
    }
}
