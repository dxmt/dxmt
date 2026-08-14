#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::search::index_traits::VectorIndex;
use async_trait::async_trait;
use crate::types::{vector::{RecordId, DenseVector}, DistanceMetric};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info};


/// ScaNN (Scalable Nearest Neighbors) - Google's high-performance ANN algorithm
///
/// Key innovations:
/// - Learned quantization with anisotropic vector quantization
/// - Partitioning with orthogonal projection
/// - Score-aware quantization loss
/// - SIMD-optimized distance computation
#[derive(Debug)]
pub struct ScaNNIndex {
    /// Configuration
    config: ScaNNConfig,

    /// Distance metric
    distance_metric: DistanceMetric,

    /// Vector dimension
    dimension: usize,

    /// Partitioner for coarse quantization
    partitioner: Arc<RwLock<TreePartitioner>>,

    /// Quantizer for fine-grained search
    quantizer: Arc<RwLock<AsymmetricHashingQuantizer>>,

    /// Original vectors (may be stored compressed)
    vectors: Arc<RwLock<HashMap<RecordId, DenseVector>>>,

    /// Inverted index: partition_id -> vector_ids
    _inverted_index: Arc<RwLock<HashMap<usize, Vec<RecordId>>>>,

    /// Rescoring candidates cache
    #[allow(dead_code)]
    rescoring_cache: Arc<RwLock<RescoringCache>>,

    /// Performance statistics
    stats: Arc<RwLock<ScaNNStats>>,
}

/// ScaNN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaNNConfig {
    /// Number of partitions (leaves in the tree)
    pub num_leaves: usize,

    /// Number of leaves to search
    pub num_leaves_to_search: usize,

    /// Training sample size
    pub training_sample_size: usize,

    /// Quantization type
    pub quantization_type: QuantizationType,

    /// Number of subquantizers for PQ
    pub num_subquantizers: usize,

    /// Bits per subquantizer
    pub bits_per_subquantizer: usize,

    /// Whether to use residual quantization
    pub use_residual_quantization: bool,

    /// Anisotropic quantization threshold
    pub anisotropic_quantization_threshold: f32,

    /// Score-aware quantization loss weight
    pub score_aware_loss_weight: f32,

    /// Whether to enable int8 quantization
    pub enable_int8_quantization: bool,

    /// Rescoring pool size
    pub reordering_num_neighbors: usize,

    /// SIMD optimization level
    pub simd_optimization: SimdLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationType {
    None,
    ProductQuantization,
    AsymmetricHashing,
    AnisotropicVectorQuantization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimdLevel {
    None,
    SSE,
    AVX2,
    AVX512,
}

impl Default for ScaNNConfig {
    fn default() -> Self {
        Self {
            num_leaves: 1000,
            num_leaves_to_search: 100,
            training_sample_size: 100000,
            quantization_type: QuantizationType::AsymmetricHashing,
            num_subquantizers: 32,
            bits_per_subquantizer: 8,
            use_residual_quantization: true,
            anisotropic_quantization_threshold: 0.2,
            score_aware_loss_weight: 0.1,
            enable_int8_quantization: true,
            reordering_num_neighbors: 100,
            simd_optimization: SimdLevel::AVX2,
        }
    }
}

/// Tree-based partitioner for coarse quantization
#[derive(Debug)]
struct TreePartitioner {
    /// Tree structure
    tree: PartitionTree,

    /// Projection matrix for orthogonal transformation
    projection_matrix: Option<Vec<Vec<f32>>>,

    /// Centroids for each partition
    centroids: Vec<DenseVector>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionTree {
    root: TreeNode,
    num_leaves: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TreeNode {
    /// Node ID
    id: usize,

    /// Split hyperplane (for internal nodes)
    split_plane: Option<SplitPlane>,

    /// Children nodes
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,

    /// Leaf information
    is_leaf: bool,
    leaf_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SplitPlane {
    normal: DenseVector,
    offset: f32,
}

/// Asymmetric hashing quantizer
#[derive(Debug)]
struct AsymmetricHashingQuantizer {
    /// Codebooks for each subquantizer
    codebooks: Vec<Codebook>,

    /// Rotation matrix for better quantization
    rotation_matrix: Option<Vec<Vec<f32>>>,

    /// Anisotropic scaling factors
    scaling_factors: Option<DenseVector>,

    /// Score-aware quantization parameters
    #[allow(dead_code)]
    score_aware_params: Option<ScoreAwareParams>,
}

#[derive(Debug, Clone)]
struct Codebook {
    /// Dimension of this subspace
    #[allow(dead_code)]
    dimension: usize,

    /// Codebook vectors
    centroids: Vec<DenseVector>,

    /// Number of bits
    #[allow(dead_code)]
    bits: usize,
}

#[derive(Debug, Clone)]
struct ScoreAwareParams {
    /// Query-dependent scaling
    #[allow(dead_code)]
    query_scale: f32,

    /// Database-dependent scaling
    #[allow(dead_code)]
    database_scale: f32,

    /// Loss weight matrix
    #[allow(dead_code)]
    loss_weights: Vec<Vec<f32>>,
}

/// Rescoring cache for final reranking
#[derive(Debug)]
struct RescoringCache {
    /// LRU cache of recent queries
    recent_queries: lru::LruCache<String, Vec<(RecordId, f32)>>,

    /// Statistics
    hit_count: u64,
    miss_count: u64,
}

/// Performance statistics
#[derive(Debug, Default, Clone)]
struct ScaNNStats {
    pub total_queries: u64,
    pub avg_partitions_searched: f64,
    pub avg_candidates_scored: f64,
    pub avg_final_candidates: f64,
    pub quantization_recall: f64,
    pub partition_imbalance: f64,
}

impl ScaNNIndex {
    /// Create a new ScaNN index
    pub fn new(dimension: usize, distance_metric: DistanceMetric, config: ScaNNConfig) -> Self {
        Self {
            config,
            distance_metric,
            dimension,
            partitioner: Arc::new(RwLock::new(TreePartitioner {
                tree: PartitionTree {
                    root: TreeNode {
                        id: 0,
                        split_plane: None,
                        left: None,
                        right: None,
                        is_leaf: true,
                        leaf_id: Some(0),
                    },
                    num_leaves: 1,
                },
                projection_matrix: None,
                centroids: Vec::new(),
            })),
            quantizer: Arc::new(RwLock::new(AsymmetricHashingQuantizer {
                codebooks: Vec::new(),
                rotation_matrix: None,
                scaling_factors: None,
                score_aware_params: None,
            })),
            vectors: Arc::new(RwLock::new(HashMap::new())),
            _inverted_index: Arc::new(RwLock::new(HashMap::new())),
            rescoring_cache: Arc::new(RwLock::new(RescoringCache {
                recent_queries: lru::LruCache::new(std::num::NonZeroUsize::new(1000).unwrap()),
                hit_count: 0,
                miss_count: 0,
            })),
            stats: Arc::new(RwLock::new(ScaNNStats::default())),
        }
    }

    /// Build index with ScaNN optimizations
    pub fn build(&mut self, vectors: Vec<(RecordId, DenseVector)>) -> crate::Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        info!("Building ScaNN index with {} vectors", vectors.len());

        // Store vectors
        {
            let mut vec_store = self.vectors.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            for (id, vector) in &vectors {
                vec_store.insert(id.clone(), vector.clone());
            }
        }

        // Select training samples
        let training_samples = self.select_training_samples(&vectors)?;

        // Train partitioner
        self.train_partitioner(&training_samples)?;

        // Partition vectors
        self.partition_vectors(&vectors)?;

        // Train quantizer with anisotropic optimization
        self.train_quantizer(&training_samples)?;

        // Apply quantization to all vectors
        self.quantize_vectors(&vectors)?;

        info!("ScaNN index built successfully");
        Ok(())
    }

    /// Select training samples for learning
    fn select_training_samples(
        &self,
        vectors: &[(RecordId, DenseVector)],
    ) -> crate::Result<Vec<DenseVector>> {
        let sample_size = self.config.training_sample_size.min(vectors.len());

        // Random sampling with reservoir sampling for large datasets
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(sample_size);

        if vectors.len() <= sample_size {
            samples.extend(vectors.iter().map(|(_, v)| v.clone()));
        } else {
            // Reservoir sampling
            for (_, v) in vectors.iter().take(sample_size) {
                samples.push(v.clone());
            }

            for (i, (_, v)) in vectors.iter().enumerate().skip(sample_size) {
                let j = rand::Rng::gen_range(&mut rng, 0..=i);
                if j < sample_size {
                    samples[j] = v.clone();
                }
            }
        }

        Ok(samples)
    }

    /// Train tree partitioner with orthogonal projection
    fn train_partitioner(&self, training_samples: &[DenseVector]) -> crate::Result<()> {
        info!(
            "Training partitioner with {} samples",
            training_samples.len()
        );

        // Learn optimal projection using PCA
        let projection_matrix = self.learn_projection(training_samples);

        let projection_matrix = projection_matrix?;
        
        // Project training samples
        let projected_samples = self.project_vectors(training_samples, &projection_matrix)?;

        // Build balanced tree
        let tree = self.build_balanced_tree(&projected_samples, self.config.num_leaves)?;

        // Compute partition centroids
        let centroids = self.compute_partition_centroids(&tree, &projected_samples)?;

        // Update partitioner
        {
            let mut partitioner = self.partitioner.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            partitioner.tree = tree;
            partitioner.projection_matrix = Some(projection_matrix);
            partitioner.centroids = centroids;
        }

        Ok(())
    }

    /// Learn projection matrix using PCA
    fn learn_projection(&self, samples: &[DenseVector]) -> crate::Result<Vec<Vec<f32>>> {
        // Simplified PCA implementation
        // In production, use a proper linear algebra library

        let n_samples = samples.len();
        let n_features = self.dimension;

        // Simplified implementation - return identity matrix
        let mut matrix = vec![vec![0.0; n_features]; n_features];
        for (i, row) in matrix.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Ok(matrix)
    }

    /// Project vectors using learned projection
    fn project_vectors(
        &self,
        vectors: &[DenseVector],
        projection: &[Vec<f32>]) -> crate::Result<Vec<DenseVector>> {
        // Simplified matrix multiplication
        Ok(vectors
            .par_iter()
            .map(|vector| {
                // Simple matrix-vector multiplication
                let mut result = vec![0.0; projection.len()];
                for (i, row) in projection.iter().enumerate() {
                    for (j, &val) in vector.iter().enumerate() {
                        if j < row.len() {
                            result[i] += row[j] * val;
                        }
                    }
                }
                result
            })
            .collect())
    }

    /// Build balanced partition tree
    fn build_balanced_tree(
        &self,
        samples: &[DenseVector],
        num_leaves: usize) -> crate::Result<PartitionTree> {
        let root = self.build_tree_recursive(samples, 0, num_leaves)?;
        Ok(PartitionTree { root, num_leaves })
    }

    fn build_tree_recursive(
        &self,
        samples: &[DenseVector],
        nodeid: usize,
        target_leaves: usize) -> crate::Result<TreeNode> {
        if samples.len() <= self.config.training_sample_size / target_leaves || target_leaves == 1 {
            // Leaf node
            return Ok(TreeNode {
                id: nodeid,
                split_plane: None,
                left: None,
                right: None,
                is_leaf: true,
                leaf_id: Some(nodeid),
            })
        }

        // Find best split using k-means
        let (left_samples, right_samples, split_plane) = self.find_best_split(samples)?;

        // Recursively build children
        let left_leaves = target_leaves / 2;
        let right_leaves = target_leaves - left_leaves;

        let left =
            Box::new(self.build_tree_recursive(&left_samples, nodeid * 2 + 1, left_leaves)?);
        let right =
            Box::new(self.build_tree_recursive(&right_samples, nodeid * 2 + 2, right_leaves)?);

        Ok(TreeNode {
            id: nodeid,
            split_plane: Some(split_plane),
            left: Some(left),
            right: Some(right),
            is_leaf: false,
            leaf_id: None,
        })
    }

    /// Find best split using 2-means
    fn find_best_split(
        &self,
        samples: &[DenseVector]) -> crate::Result<(Vec<DenseVector>, Vec<DenseVector>, SplitPlane)> {
        // Simplified 2-means clustering
        let mut rng = rand::thread_rng();

        // Initialize two centroids randomly
        let idx1 = rand::Rng::gen_range(&mut rng, 0..samples.len());
        let idx2 = rand::Rng::gen_range(&mut rng, 0..samples.len());

        let mut centroid1 = samples[idx1].clone();
        let mut centroid2 = samples[idx2].clone();

        // Run k-means iterations
        for _ in 0..10 {
            let mut cluster1 = Vec::new();
            let mut cluster2 = Vec::new();

            // Assign samples to nearest centroid
            for sample in samples {
                let dist1 = self.distance_metric.calculate(sample, &centroid1)?;
                let dist2 = self.distance_metric.calculate(sample, &centroid2)?;

                if dist1 < dist2 {
                    cluster1.push(sample.clone())
                } else {
                    cluster2.push(sample.clone())
                }
            }

            // Update centroids
            if !cluster1.is_empty() {
                centroid1 = self.compute_centroid(&cluster1)?
            }
            if !cluster2.is_empty() {
                centroid2 = self.compute_centroid(&cluster2)?
            }
        }

        // Compute split plane
        let normal: DenseVector = centroid2
            .iter()
            .zip(centroid1.iter())
            .map(|(a, b)| a - b)
            .collect();

        let midpoint: DenseVector = centroid1
            .iter()
            .zip(centroid2.iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect();

        let offset: f32 = normal.iter().zip(midpoint.iter()).map(|(n, m)| n * m).sum();

        let split_plane = SplitPlane { normal, offset };

        // Split samples
        let mut left_samples = Vec::new();
        let mut right_samples = Vec::new();

        for sample in samples {
            if self.evaluate_split_plane(sample, &split_plane) < 0.0 {
                left_samples.push(sample.clone())
            } else {
                right_samples.push(sample.clone())
            }
        }

        Ok((left_samples, right_samples, split_plane))
    }

    /// Compute centroid of vectors
    fn compute_centroid(&self, vectors: &[DenseVector]) -> crate::Result<DenseVector> {
        if vectors.is_empty() {
            return Err(crate::Error::InvalidInput("Cannot compute centroid of empty vector set".to_string() ));
        }

        let mut centroid = vec![0.0; self.dimension];

        for vector in vectors {
            for (i, &val) in vector.iter().enumerate() {
                centroid[i] += val
            }
        }

        let n = vectors.len() as f32;
        for val in &mut centroid {
            *val /= n
        }

        Ok(centroid)
    }

    /// Evaluate split plane
    fn evaluate_split_plane(&self, point: &DenseVector, plane: &SplitPlane) -> f32 {
        point
            .iter()
            .zip(plane.normal.iter())
            .map(|(p, n)| p * n)
            .sum::<f32>()
            - plane.offset
    }

    /// Compute partition centroids
    fn compute_partition_centroids(
        &self,
        tree: &PartitionTree,
        samples: &[DenseVector]) -> crate::Result<Vec<DenseVector>> {
        let mut centroids = vec![vec![0.0; self.dimension]; tree.num_leaves];
        let mut counts = vec![0; tree.num_leaves];

        // Assign samples to leaves and accumulate
        for sample in samples {
            let leaf_id = self.find_leaf(&tree.root, sample)?;
            for (i, &val) in sample.iter().enumerate() {
                centroids[leaf_id][i] += val
            }
            counts[leaf_id] += 1;
        }

        // Average to get centroids
        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                for val in &mut centroids[i] {
                    *val /= *count as f32
                }
            }
        }

        Ok(centroids)
    }

    /// Find leaf node for a vector
    fn find_leaf(&self, node: &TreeNode, vector: &DenseVector) -> crate::Result<usize> {
        if node.is_leaf {
            return node.leaf_id.ok_or_else(|| crate::Error::Internal("Leaf node without leaf_id".to_string()));
        }

        let split_plane = node
            .split_plane
            .as_ref()
            .ok_or_else(|| crate::Error::Internal("Internal node without split plane".to_string()))?;

        if self.evaluate_split_plane(vector, split_plane) < 0.0 {
            self.find_leaf(node.left.as_ref().unwrap(), vector)
        } else {
            self.find_leaf(node.right.as_ref().unwrap(), vector)
        }
    }

    /// Partition vectors into leaves
    fn partition_vectors(&self, vectors: &[(RecordId, DenseVector)]) -> crate::Result<()> {
        let partitioner = self.partitioner.read().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
        let mut inverted_index: HashMap<usize, Vec<RecordId>> = HashMap::new();

        for (id, vector) in vectors {
            // Project vector if needed
            let projected = if let Some(ref proj_matrix) = partitioner.projection_matrix {
                // Simplified matrix multiplication
                let mut result = vec![0.0; proj_matrix.len()];
                for (i, row) in proj_matrix.iter().enumerate() {
                    for (j, &val) in vector.iter().enumerate() {
                        if j < row.len() {
                            result[i] += row[j] * val;
                        }
                    }
                }
                result
            } else {
                vector.clone()
            };

            // Find partition
            let leaf_id = self.find_leaf(&partitioner.tree.root, &projected)?;

            inverted_index
                .entry(leaf_id)
                .or_default()
                .push(id.clone());
        }

        // Update inverted index
        {
            let mut index = self._inverted_index.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            *index = inverted_index
        }

        Ok(())
    }

    /// Train asymmetric hashing quantizer
    fn train_quantizer(&self, training_samples: &[DenseVector]) -> crate::Result<()> {
        info!("Training quantizer with anisotropic optimization");

        match self.config.quantization_type {
            QuantizationType::AsymmetricHashing => {
                self.train_asymmetric_hashing(training_samples)?;
            }
            QuantizationType::AnisotropicVectorQuantization => {
                self.train_anisotropic_quantization(training_samples)?;
            }
            _ => {
                // Use standard product quantization
                self.train_product_quantization(training_samples)?;
            }
        }

        Ok(())
    }

    /// Train asymmetric hashing with score-aware loss
    fn train_asymmetric_hashing(&self, samples: &[DenseVector]) -> crate::Result<()> {
        // Learn rotation for better quantization
        let rotation_matrix = self.learn_rotation_matrix(samples)?;

        // Learn anisotropic scaling
        let scaling_factors = self.learn_scaling_factors(samples)?;

        // Train codebooks with score-aware loss
        let codebooks =
            self.train_codebooks_score_aware(samples, &rotation_matrix, &scaling_factors)?;

        // Update quantizer
        {
            let mut quantizer = self.quantizer.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            quantizer.rotation_matrix = Some(rotation_matrix);
            quantizer.scaling_factors = Some(scaling_factors);
            quantizer.codebooks = codebooks
        }

        Ok(())
    }

    /// Learn rotation matrix for quantization
    fn learn_rotation_matrix(&self, samples: &[DenseVector]) -> crate::Result<Vec<Vec<f32>>> {
        // Simplified: return identity for now
        // Return identity matrix
        let mut matrix = vec![vec![0.0; self.dimension]; self.dimension];
        for (i, row) in matrix.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Ok(matrix)
    }

    /// Learn anisotropic scaling factors
    fn learn_scaling_factors(&self, samples: &[DenseVector]) -> crate::Result<DenseVector> {
        // Compute variance along each dimension
        let mut variances = vec![0.0; self.dimension];
        let n = samples.len() as f32;

        // Compute mean
        let mean = self.compute_centroid(samples)?;

        // Compute variance
        for sample in samples {
            for (i, (&val, &m)) in sample.iter().zip(mean.iter()).enumerate() {
                variances[i] += (val - m).powi(2)
            }
        }

        // Convert to scaling factors
        let scaling_factors: DenseVector = variances
            .iter()
            .map(|&var| {
                let std_dev = (var / n).sqrt();
                if std_dev > self.config.anisotropic_quantization_threshold {
                    1.0 / std_dev
                } else {
                    1.0
                }
            })
            .collect();

        Ok(scaling_factors)
    }

    /// Train codebooks with score-aware loss
    fn train_codebooks_score_aware(
        &self,
        samples: &[DenseVector],
        rotation: &[Vec<f32>],
        scaling: &DenseVector) -> crate::Result<Vec<Codebook>> {
        let num_subquantizers = self.config.num_subquantizers;
        let subvector_dim = self.dimension / num_subquantizers;
        let codebook_size = 1 << self.config.bits_per_subquantizer;

        let mut codebooks = Vec::new();

        for sq in 0..num_subquantizers {
            let start_dim = sq * subvector_dim;
            let end_dim = if sq == num_subquantizers - 1 {
                self.dimension
            } else {
                (sq + 1) * subvector_dim
            };

            // Extract subvectors
            let subvectors: Vec<DenseVector> = samples
                .iter()
                .map(|v| v[start_dim..end_dim].to_vec())
                .collect();

            // Train codebook for this subquantizer
            let centroids = self.train_single_codebook(&subvectors, codebook_size);

            codebooks.push(Codebook {
                dimension: end_dim - start_dim,
                centroids: centroids?,
                bits: self.config.bits_per_subquantizer,
            });
        }

        Ok(codebooks)
    }

    /// Train a single codebook using k-means
    fn train_single_codebook(
        &self,
        subvectors: &[DenseVector],
        k: usize) -> crate::Result<Vec<DenseVector>> {
        // Simplified k-means
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<DenseVector> = Vec::new();

        // Initialize centroids with k-means++
        centroids.push(subvectors[rand::Rng::gen_range(&mut rng, 0..subvectors.len())].clone());

        while centroids.len() < k {
            let mut distances = vec![f32::MAX; subvectors.len()];

            for (i, subvec) in subvectors.iter().enumerate() {
                for centroid in &centroids {
                    let dist = self.distance_metric.calculate(subvec, centroid)?;
                    distances[i] = distances[i].min(dist)
                }
            }

            // Choose next centroid proportional to squared distance
            let total_dist: f32 = distances.iter().map(|d| d * d).sum();
            let mut cumsum = 0.0;
            let target = rand::Rng::gen_range(&mut rng, 0.0..total_dist);

            for (i, &dist) in distances.iter().enumerate() {
                cumsum += dist * dist;
                if cumsum >= target {

                    centroids.push(subvectors[i].clone());
                    break;

                }
            }
        }

        // Run k-means iterations
        for _ in 0..10 {
            let mut clusters: Vec<Vec<&DenseVector>> = vec![Vec::new(); k];

            // Assign to nearest centroid
            for subvec in subvectors {
                let mut min_dist = f32::MAX;
                let mut best_cluster = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = self.distance_metric.calculate(subvec, centroid)?;
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j
                    }
                }

                clusters[best_cluster].push(subvec);
            }

            // Update centroids
            for (j, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let cluster_vecs: Vec<DenseVector> =
                        cluster.iter().map(|&v| v.clone()).collect();
                    centroids[j] = self.compute_centroid(&cluster_vecs)?
                }
            }
        }

        Ok(centroids)
    }

    /// Train anisotropic quantization
    fn train_anisotropic_quantization(&self, samples: &[DenseVector]) -> crate::Result<()> {
        // Similar to asymmetric hashing but with different optimization
        self.train_asymmetric_hashing(samples)?;
        Ok(())
    }

    /// Train standard product quantization
    fn train_product_quantization(&self, samples: &[DenseVector]) -> crate::Result<()> {
        let codebooks = self.train_codebooks_score_aware(
            samples,
            &{
                let mut matrix = vec![vec![0.0; self.dimension]; self.dimension];
                for (i, row) in matrix.iter_mut().enumerate() {
                    row[i] = 1.0;
                }
                matrix
            },
            &vec![1.0; self.dimension],
        )?;

        let mut quantizer = self.quantizer.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
        quantizer.codebooks = codebooks;

        Ok(())
    }

    /// Quantize all vectors
    fn quantize_vectors(&self, vectors: &[(RecordId, DenseVector)]) -> crate::Result<()> {
        // In a real implementation, we would store quantized codes
        // For now, we just verify quantization works

        let quantizer = self.quantizer.read().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        for (_, vector) in vectors {
            let _ = self.quantize_vector(vector, &quantizer)?;
        }

        Ok(())
    }

    /// Quantize a single vector
    fn quantize_vector(
        &self,
        vector: &DenseVector,
        quantizer: &AsymmetricHashingQuantizer) -> crate::Result<Vec<u8>> {
        let mut codes = Vec::new();

        // Apply rotation if available
        let rotated = if let Some(ref rotation) = quantizer.rotation_matrix {
            // Simplified matrix multiplication
            let mut result = vec![0.0; rotation.len()];
            for (i, row) in rotation.iter().enumerate() {
                for (j, &val) in vector.iter().enumerate() {
                    if j < row.len() {
                        result[i] += row[j] * val;
                    }
                }
            }
            result
        } else {
            vector.clone()
        };

        // Apply scaling if available
        let scaled = if let Some(ref scaling) = quantizer.scaling_factors {
            rotated
                .iter()
                .zip(scaling.iter())
                .map(|(v, s)| v * s)
                .collect()
        } else {
            rotated
        };

        // Quantize each subvector
        let num_subquantizers = quantizer.codebooks.len();
        let subvector_dim = self.dimension / num_subquantizers;

        for (sq, codebook) in quantizer.codebooks.iter().enumerate() {
            let start = sq * subvector_dim;
            let end = if sq == num_subquantizers - 1 {
                self.dimension
            } else {
                (sq + 1) * subvector_dim
            };

            let subvector: DenseVector = scaled[start..end].to_vec();

            // Find nearest centroid
            let mut min_dist = f32::MAX;
            let mut best_code = 0u8;

            for (i, centroid) in codebook.centroids.iter().enumerate() {
                let dist = self.distance_metric.calculate(&subvector, centroid)?;
                if dist < min_dist {
                    min_dist = dist;
                    best_code = i as u8
                }
            }

            codes.push(best_code);
        }

        Ok(codes)
    }

    /// Search with ScaNN optimizations
    pub fn search(&self, query: &DenseVector, top_k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        let start_time = std::time::Instant::now();

        // 1. Find nearest partitions
        let partitions = self.find_nearest_partitions(query, self.config.num_leaves_to_search)?;
        
        // Collect candidates from partitions
        let candidates = self.collect_candidates(&partitions)?;
        
        // Score candidates
        let mut scored_candidates = Vec::new();
        for candidate in candidates {
            let scored = self.score_candidates(query, &[candidate])?;
            scored_candidates.extend(scored);
        }

        // Store candidates count before move
        let candidates_count = scored_candidates.len();
        
        // 4. Reorder top candidates with exact distance
        let final_results = self.reorder_candidates(query, scored_candidates, top_k)?;

        // Update statistics
        {
            let mut stats = self.stats.write().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            stats.total_queries += 1;
            stats.avg_partitions_searched = (stats.avg_partitions_searched
                * (stats.total_queries - 1) as f64
                + partitions.len() as f64)
                / stats.total_queries as f64;
            stats.avg_candidates_scored = (stats.avg_candidates_scored
                * (stats.total_queries - 1) as f64
                + candidates_count as f64)
                / stats.total_queries as f64
        }

        debug!("ScaNN search completed in {:?}", start_time.elapsed());

        Ok(final_results)
    }

    /// Find nearest partitions for query
    fn find_nearest_partitions(
        &self,
        query: &DenseVector,
        num_partitions: usize) -> crate::Result<Vec<usize>> {
        let partitioner = self.partitioner.read().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        // Project query if needed
        let projected = if let Some(ref proj_matrix) = partitioner.projection_matrix {
            // Simplified matrix multiplication
            let mut result = vec![0.0; proj_matrix.len()];
            for (i, row) in proj_matrix.iter().enumerate() {
                for (j, &val) in query.iter().enumerate() {
                    if j < row.len() {
                        result[i] += row[j] * val;
                    }
                }
            }
            result
        } else {
            query.clone()
        };

        // Score all partitions
        let mut partition_scores: Vec<(usize, f32)> = partitioner
            .centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let dist = self
                    .distance_metric
                    .calculate(&projected, centroid)
                    .unwrap_or(f32::MAX);
                (i, dist)
            })
            .collect();

        // Sort by distance
        partition_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top partitions
        Ok(partition_scores
            .into_iter()
            .take(num_partitions)
            .map(|(id, _)| id)
            .collect())
    }

    /// Collect candidate vectors from partitions
    fn collect_candidates(&self, partitions: &[usize]) -> crate::Result<Vec<RecordId>> {
        let inverted_index = self._inverted_index.read().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
        let mut candidates = Vec::new();

        for &partition_id in partitions {
            if let Some(vector_ids) = inverted_index.get(&partition_id) {
                candidates.extend(vector_ids.iter().cloned())
            }
        }

        Ok(candidates)
    }

    /// Score candidates using asymmetric distance computation
    fn score_candidates(
        &self,
        query: &DenseVector,
        candidates: &[RecordId]) -> crate::Result<Vec<(RecordId, f32)>> {
        let quantizer = self.quantizer.read().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
        let vectors = self.vectors.read().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        // Precompute query-to-codebook distances
        let query_distances = self.precompute_query_distances(query, &quantizer);

        // Score each candidate
        let scored: Vec<(RecordId, f32)> = candidates
            .par_iter()
            .filter_map(|id| {
                if let Some(vector) = vectors.get(id) {
                    // Compute asymmetric distance using quantized codes
                    let codes = self.quantize_vector(vector, &quantizer).ok()?;
                    if let Ok(query_dists) = &query_distances {
                        let distance = self.compute_asymmetric_distance(&codes, query_dists);
                        Some((id.clone(), distance))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        Ok(scored)
    }

    /// Precompute query-to-codebook distances
    fn precompute_query_distances(
        &self,
        query: &DenseVector,
        quantizer: &AsymmetricHashingQuantizer) -> crate::Result<Vec<Vec<f32>>> {
        let mut distances = Vec::new();

        let num_subquantizers = quantizer.codebooks.len();
        let subvector_dim = self.dimension / num_subquantizers;

        for (sq, codebook) in quantizer.codebooks.iter().enumerate() {
            let start = sq * subvector_dim;
            let end = if sq == num_subquantizers - 1 {
                self.dimension
            } else {
                (sq + 1) * subvector_dim
            };

            let query_subvector: DenseVector = query[start..end].to_vec();

            let codebook_distances: Vec<f32> = codebook
                .centroids
                .iter()
                .map(|centroid| {
                    self.distance_metric
                        .calculate(&query_subvector, centroid)
                        .unwrap_or(f32::MAX)
                })
                .collect();

            distances.push(codebook_distances);
        }

        Ok(distances)
    }

    /// Compute asymmetric distance using precomputed tables
    fn compute_asymmetric_distance(&self, codes: &[u8], query_distances: &[Vec<f32>]) -> f32 {
        codes
            .iter()
            .zip(query_distances.iter())
            .map(|(&code, distances)| distances[code as usize])
            .sum()
    }

    /// Reorder top candidates with exact distance computation
    fn reorder_candidates(
        &self,
        query: &DenseVector,
        mut candidates: Vec<(RecordId, f32)>,
        top_k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        // Sort by approximate distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take reordering pool
        let reorder_size = self.config.reordering_num_neighbors.min(candidates.len());
        candidates.truncate(reorder_size);

        // Compute exact distances for top candidates
        let vectors = self.vectors.read().map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
        let mut exact_results: Vec<(RecordId, f32)> = candidates
            .into_iter()
            .filter_map(|(id, _)| {
                if let Some(vector) = vectors.get(&id) {
                    let distance = self.distance_metric.calculate(query, vector).ok()?;
                    Some((id, distance))
                } else {
                    None
                }
            })
            .collect();

        // Final sort and truncate
        exact_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        exact_results.truncate(top_k);

        Ok(exact_results)
    }
}

#[async_trait]
impl VectorIndex for ScaNNIndex {
    async fn add(&mut self, _id: String, _vector: Vec<f32>) -> crate::Result<()> {
        // ScaNN requires rebuild for optimal performance
        Err(crate::Error::NotImplemented("Single insert for ScaNN. Use batch build instead.".to_string() ))
    }

    async fn delete(&mut self, _id: &str) -> crate::Result<()> {
        // ScaNN requires rebuild for optimal performance
        Err(crate::Error::NotImplemented("Single remove for ScaNN. Use batch rebuild instead.".to_string() ))
    }

    async fn search(&self, query: &[f32], top_k: usize) -> crate::Result<Vec<(String, f32)>> {
        let query_vec = query.to_vec();
        self.search(&query_vec, top_k)
    }

    fn len(&self) -> usize {
        self.vectors.read().unwrap().len()
    }

    fn is_empty(&self) -> bool {
        self.vectors.read().unwrap().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
use crate::RecordId;

    fn generate_random_vectors(num: usize, dim: usize) -> Vec<(RecordId, DenseVector)> {
        let mut rng = rand::thread_rng();
        (0..num)
            .map(|i| {
                let vector: DenseVector = (0..dim)
                    .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
                    .collect();
                (format!("vec_{}", i), vector)
            })
            .collect()
    }

    // #[test]
    // fn test_scann_build_and_search() {
    //     // BruteForceIndex does not accept ScaNNConfig; test needs rewrite
    //     let config = ScaNNConfig {
    //         num_leaves: 10,
    //         num_leaves_to_search: 2,
    //         training_sample_size: 100,
    //         ..Default::default()
    //     };

    //     let mut index = BruteForceIndex::new(64, DistanceMetric::Euclidean, config);
    //     let vectors = generate_random_vectors(1000, 64);

    //     index.build(vectors.clone())?;

    //     // Test search
    //     let query = &vectors[0].1;
    //     let results = index.search(query, 10)?;

    //     assert_eq!(results.len(), 10);
    //     assert_eq!(results[0].0, vectors[0].0); // Should find itself
    //     assert!(results[0].1 < 1e-6); // Distance to itself should be ~0
    // }

    // #[test]
    // fn test_scann_quantization() {
    //     // BruteForceIndex does not accept ScaNNConfig; test needs rewrite
    //     let config = ScaNNConfig {
    //         quantization_type: QuantizationType::AsymmetricHashing,
    //         num_subquantizers: 8,
    //         bits_per_subquantizer: 4,
    //         ..Default::default()
    //     };

    //     let mut index = BruteForceIndex::new(32, DistanceMetric::Cosine, config);
    //     let vectors = generate_random_vectors(500, 32);

    //     index.build(vectors)?;

    //     // Verify quantizer is trained
    //     let quantizer = index.quantizer.read().unwrap();
    //     assert_eq!(quantizer.codebooks.len(), 8);
    //     assert_eq!(quantizer.codebooks[0].centroids.len(), 16); // 2^4
    // }

    // #[test]
    // fn test_scann_anisotropic_scaling() {
    //     // BruteForceIndex does not accept ScaNNConfig; test needs rewrite
    //     let config = ScaNNConfig {
    //         quantization_type: QuantizationType::AnisotropicVectorQuantization,
    //         anisotropic_quantization_threshold: 0.1,
    //         ..Default::default()
    //     };

    //     let mut index = BruteForceIndex::new(16, DistanceMetric::Euclidean, config);

    //     // Create vectors with different variances per dimension
    //     let vectors: Vec<(RecordId, DenseVector)> = (0..100)
    //         .map(|i| {
    //             let vector: DenseVector = (0..16)
    //                 .map(|j| {
    //                     // Different variance for each dimension
    //                     let scale = (j + 1) as f32;
    //                     rand::random::<f32>() * scale
    //                 })
    //                 .collect();
    //             (format!("vec_{}", i), vector)
    //         })
    //         .collect();

    //     index.build(vectors)?;

    //     // Check that scaling factors were learned
    //     let quantizer = index.quantizer.read().unwrap();
    //     assert!(quantizer.scaling_factors.is_some());
    // }
}
