// Enhanced Product Quantization with optimizations from FAISS
// Includes: Optimized K-means training, Polysemous codes, Residual quantization

use crate::types::vector::DenseVector;
use anyhow::Result;
use rand::prelude::*;
use rayon::prelude::*;

/// Configuration for Enhanced PQ
#[derive(Debug, Clone)]
pub struct EnhancedPQConfig {
    /// Number of subquantizers
    pub num_subquantizers: usize,
    /// Number of centroids per subquantizer
    pub num_centroids: usize,
    /// Number of bits per subquantizer
    pub bits: usize,
    /// Use polysemous codes
    pub use_polysemous: bool,
    /// Use residual quantization
    pub use_residual: bool,
    /// K-means iterations
    pub kmeans_iterations: usize,
    /// Number of K-means attempts
    pub kmeans_attempts: usize,
    /// Number of threads
    pub num_threads: usize,
}

impl Default for EnhancedPQConfig {
    fn default() -> Self {
        Self {
            num_subquantizers: 8,
            num_centroids: 256,
            bits: 8,
            use_polysemous: false,
            use_residual: false,
            kmeans_iterations: 25,
            kmeans_attempts: 3,
            num_threads: num_cpus::get(),
        }
    }
}

/// Enhanced Product Quantizer
pub struct EnhancedPQ {
    /// Configuration
    config: EnhancedPQConfig,

    /// Dimension of vectors
    dimension: usize,

    /// Dimension of each subvector
    subvector_dim: usize,

    /// Centroids for each subquantizer
    /// Shape: [num_subquantizers, num_centroids, subvector_dim]
    centroids: Vec<Vec<DenseVector>>,

    /// Polysemous permutation for each subquantizer
    polysemous_permutation: Option<Vec<Vec<usize>>>,

    /// Residual quantizer for refinement
    residual_quantizer: Option<Box<EnhancedPQ>>,

    /// Training statistics
    training_stats: TrainingStats,

    /// Is trained flag
    trained: bool,
}

/// Training statistics
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    pub training_time_ms: u64,
    pub final_distortion: f32,
    pub iterations_performed: usize,
}

impl EnhancedPQ {
    /// Create new Enhanced PQ
    pub fn new(dimension: usize, config: EnhancedPQConfig) -> Result<Self> {
        if !dimension.is_multiple_of(config.num_subquantizers) {
            anyhow::bail!(
                "Dimension {} must be divisible by num_subquantizers {}",
                dimension,
                config.num_subquantizers
            );
        }

        let subvector_dim = dimension / config.num_subquantizers;
        let num_centroids = config.num_centroids;
        let num_subquantizers = config.num_subquantizers;

        Ok(Self {
            config,
            dimension,
            subvector_dim,
            centroids: vec![vec![vec![0.0; subvector_dim]; num_centroids]; num_subquantizers],
            polysemous_permutation: None,
            residual_quantizer: None,
            training_stats: TrainingStats::default(),
            trained: false,
        })
    }

    /// Optimized training with balanced K-means
    pub fn train_optimized(&mut self, data: &[DenseVector]) -> Result<()> {
        let start = std::time::Instant::now();

        // Validate input
        if data.is_empty() {
            anyhow::bail!("Training data is empty");
        }

        // Train each subquantizer in parallel
        let results: Vec<_> = (0..self.config.num_subquantizers)
            .into_par_iter()
            .map(|subq| {
                let subvectors = self.extract_subvectors(data, subq);
                self.train_subquantizer(&subvectors)
            })
            .collect();

        // Collect results
        for (subq, result) in results.into_iter().enumerate() {
            let (centroids, distortion) = result?;
            self.centroids[subq] = centroids;
            self.training_stats.final_distortion += distortion;
        }

        // Optimize polysemous order if enabled
        if self.config.use_polysemous {
            self.optimize_polysemous_codes(data)?;
        }

        // Train residual quantizer if enabled
        if self.config.use_residual {
            self.train_residual_quantizer(data)?;
        }

        self.trained = true;
        self.training_stats.training_time_ms = start.elapsed().as_millis() as u64;

        Ok(())
    }

    /// Extract subvectors for a subquantizer
    fn extract_subvectors(&self, data: &[DenseVector], subq: usize) -> Vec<DenseVector> {
        let start = subq * self.subvector_dim;
        let end = (subq + 1) * self.subvector_dim;

        data.iter()
            .map(|vec| vec[start..end].to_vec())
            .collect()
    }

    /// Train a single subquantizer using balanced K-means
    fn train_subquantizer(&self, data: &[DenseVector]) -> Result<(Vec<DenseVector>, f32)> {
        let mut best_centroids = vec![vec![0.0; self.subvector_dim]; self.config.num_centroids];
        let mut best_distortion = f32::MAX;

        for _ in 0..self.config.kmeans_attempts {
            let (centroids, distortion) = self.balanced_kmeans(data)?;

            if distortion < best_distortion {
                best_distortion = distortion;
                best_centroids = centroids;
            }
        }

        Ok((best_centroids, best_distortion))
    }

    /// Balanced K-means clustering
    fn balanced_kmeans(&self, data: &[DenseVector]) -> Result<(Vec<DenseVector>, f32)> {
        let k = self.config.num_centroids;
        let n = data.len();

        // Initialize centroids using K-means++
        let mut centroids = self.kmeans_plus_plus_init(data, k)?;
        let mut assignments = vec![0; n];
        let mut distortion = 0.0f32;

        for _ in 0..self.config.kmeans_iterations {
            // Assign points to nearest centroids
            distortion = 0.0;
            for (i, point) in data.iter().enumerate() {
                let (nearest, dist) = self.find_nearest_centroid(point, &centroids);
                assignments[i] = nearest;
                distortion += dist;
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0; self.subvector_dim]; k];
            let mut counts = vec![0; k];

            for (i, point) in data.iter().enumerate() {
                let cluster = assignments[i];
                for (j, &val) in point.iter().enumerate() {
                    new_centroids[cluster][j] += val;
                }
                counts[cluster] += 1;
            }

            // Average centroids
            for (i, count) in counts.iter().enumerate() {
                if *count > 0 {
                    let count_f = *count as f32;
                    for v in &mut new_centroids[i] {
                        *v /= count_f;
                    }
                } else {
                    // Reinitialize empty clusters
                    let random_idx = thread_rng().gen_range(0..n);
                    new_centroids[i] = data[random_idx].clone();
                }
            }

            centroids = new_centroids;
        }

        Ok((centroids, distortion))
    }

    /// K-means++ initialization
    fn kmeans_plus_plus_init(&self, data: &[DenseVector], k: usize) -> Result<Vec<DenseVector>> {
        let mut rng = thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..data.len());
        centroids.push(data[first_idx].clone());

        // Choose remaining centroids
        for _ in 1..k {
            let mut distances = vec![f32::MAX; data.len()];

            // Compute distances to nearest existing centroid
            for (i, point) in data.iter().enumerate() {
                for centroid in &centroids {
                    let dist = self.compute_l2_distance(point, centroid);
                    distances[i] = distances[i].min(dist);
                }
            }

            // Choose next centroid with probability proportional to squared distance
            let total: f32 = distances.iter().map(|d| d * d).sum();
            let mut threshold = rng.gen::<f32>() * total;

            for (i, &dist) in distances.iter().enumerate() {
                threshold -= dist * dist;
                if threshold <= 0.0 {
                    centroids.push(data[i].clone());
                    break;
                }
            }
        }

        Ok(centroids)
    }

    /// Find nearest centroid
    fn find_nearest_centroid(&self, point: &[f32], centroids: &[DenseVector]) -> (usize, f32) {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (i, centroid) in centroids.iter().enumerate() {
            let dist = self.compute_l2_distance(point, centroid.as_slice());
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        (best_idx, best_dist)
    }

    /// Compute L2 distance
    #[inline]
    fn compute_l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    /// Optimize polysemous codes order
    fn optimize_polysemous_codes(&mut self, data: &[DenseVector]) -> Result<()> {
        // This optimizes the order of centroids to enable early termination
        // during search while maintaining good recall

        let mut permutations = Vec::with_capacity(self.config.num_subquantizers);

        for subq in 0..self.config.num_subquantizers {
            // Compute centroid frequencies
            let subvectors = self.extract_subvectors(data, subq);
            let mut frequencies = vec![0; self.config.num_centroids];

            for subvec in &subvectors {
                let (nearest, _) = self.find_nearest_centroid(subvec, &self.centroids[subq]);
                frequencies[nearest] += 1;
            }

            // Sort by frequency (most common first)
            let mut indexed: Vec<_> = frequencies.iter().enumerate().collect();
            indexed.sort_by_key(|(_, &freq)| std::cmp::Reverse(freq));

            let permutation: Vec<_> = indexed.iter().map(|(idx, _)| *idx).collect();
            permutations.push(permutation);
        }

        self.polysemous_permutation = Some(permutations);
        Ok(())
    }

    /// Train residual quantizer
    fn train_residual_quantizer(&mut self, data: &[DenseVector]) -> Result<()> {
        // Compute residuals after first stage quantization
        let mut residuals = Vec::with_capacity(data.len());

        for vec in data {
            let codes = self.encode_single(vec)?;
            let reconstructed = self.decode_single(&codes)?;

            let residual: Vec<f32> = vec
                .iter()
                .zip(reconstructed.iter())
                .map(|(v, r)| v - r)
                .collect();

            residuals.push(residual);
        }

        // Train residual quantizer with smaller configuration
        let mut residual_config = self.config.clone();
        residual_config.use_residual = false; // Avoid infinite recursion
        residual_config.use_polysemous = false;

        let mut residual_pq = EnhancedPQ::new(self.dimension, residual_config)?;
        residual_pq.train_optimized(&residuals)?;

        self.residual_quantizer = Some(Box::new(residual_pq));
        Ok(())
    }

    /// Encode a single vector
    pub fn encode_single(&self, vector: &DenseVector) -> Result<Vec<u8>> {
        if !self.trained {
            anyhow::bail!("PQ must be trained before encoding");
        }

        let mut codes = vec![0u8; self.config.num_subquantizers];

        for (subq, code) in codes.iter_mut().enumerate() {
            let start = subq * self.subvector_dim;
            let end = (subq + 1) * self.subvector_dim;
            let subvector = &vector[start..end];

            let (nearest, _) = self.find_nearest_centroid(subvector, &self.centroids[subq]);
            *code = nearest as u8;
        }

        Ok(codes)
    }

    /// Decode a single vector
    pub fn decode_single(&self, codes: &[u8]) -> Result<DenseVector> {
        let mut reconstructed = vec![0.0; self.dimension];

        for (subq, &code_byte) in codes.iter().enumerate() {
            let code = code_byte as usize;
            let centroid = &self.centroids[subq][code];

            let start = subq * self.subvector_dim;
            for (i, &val) in centroid.iter().enumerate() {
                reconstructed[start + i] = val;
            }
        }

        Ok(reconstructed)
    }

    /// Encode batch of vectors
    pub fn encode_batch(&self, vectors: &[DenseVector]) -> Result<Vec<Vec<u8>>> {
        vectors
            .par_iter()
            .map(|v| self.encode_single(v))
            .collect()
    }

    /// Decode batch of codes
    pub fn decode_batch(&self, codes: &[Vec<u8>]) -> Result<Vec<DenseVector>> {
        codes.par_iter().map(|c| self.decode_single(c)).collect()
    }

    /// Get training status
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get training statistics
    pub fn get_stats(&self) -> &TrainingStats {
        &self.training_stats
    }

    /// Get centroid for a specific subquantizer and code
    pub fn get_centroid(&self, subq: usize, code: usize) -> &[f32] {
        &self.centroids[subq][code]
    }
}

/// Scalar Quantization support
pub struct ScalarQuantizer {
    bits: u8,
    trained_min: Vec<f32>,
    trained_max: Vec<f32>,
    dimension: usize,
}

impl ScalarQuantizer {
    pub fn new(dimension: usize, bits: u8) -> Result<Self> {
        if bits != 4 && bits != 8 && bits != 16 {
            anyhow::bail!("Scalar quantization only supports 4, 8, or 16 bits");
        }

        Ok(Self {
            bits,
            trained_min: vec![0.0; dimension],
            trained_max: vec![0.0; dimension],
            dimension,
        })
    }

    /// Train scalar quantizer
    pub fn train(&mut self, data: &[DenseVector]) -> Result<()> {
        for i in 0..self.dimension {
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;

            for vec in data {
                min_val = min_val.min(vec[i]);
                max_val = max_val.max(vec[i]);
            }

            self.trained_min[i] = min_val;
            self.trained_max[i] = max_val;
        }

        Ok(())
    }

    /// Encode with scalar quantization
    pub fn encode(&self, vector: &DenseVector) -> Vec<u8> {
        let levels = (1 << self.bits) - 1;
        let mut encoded = Vec::with_capacity(vector.len());

        for (i, &v) in vector.iter().enumerate() {
            let normalized = (v - self.trained_min[i])
                / (self.trained_max[i] - self.trained_min[i]);
            let quantized = (normalized * levels as f32).round() as u8;
            encoded.push(quantized.min(levels as u8));
        }

        encoded
    }

    /// Decode scalar quantized vector
    pub fn decode(&self, codes: &[u8]) -> DenseVector {
        let levels = (1 << self.bits) - 1;
        let mut decoded = Vec::with_capacity(self.dimension);

        for (i, &code) in codes.iter().enumerate() {
            let normalized = code as f32 / levels as f32;
            let value = self.trained_min[i]
                + normalized * (self.trained_max[i] - self.trained_min[i]);
            decoded.push(value);
        }

        decoded
    }
}

// #[cfg(test)]
// #[path = "enhanced_pq_tests.rs"]
// mod tests;