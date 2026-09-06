use crate::types::{DenseVector, DistanceMetric};
use serde::{Deserialize, Serialize};

/// Product Quantization (PQ) for vector compression
///
/// PQ divides high-dimensional vectors into subvectors and quantizes each subspace separately.
/// This achieves significant memory reduction (4-32x) with minimal accuracy loss (<1%).
///
/// Example: 960-dim vector
/// - Original: 960 × 4 bytes = 3840 bytes
/// - PQ8 (120 subvectors × 8 dims): 120 × 1 byte = 120 bytes
/// - Compression: 32x
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Number of subvectors (e.g., 8, 16, 32)
    pub num_subvectors: usize,

    /// Dimension of each subvector
    pub subvector_dim: usize,

    /// Number of centroids per subspace (typically 256 for 8-bit encoding)
    pub num_centroids: usize,

    /// Codebooks: [num_subvectors][num_centroids][subvector_dim]
    /// Each codebook contains the centroid vectors for one subspace
    pub codebooks: Vec<Vec<DenseVector>>,

    /// Distance metric
    pub distance_metric: DistanceMetric,
}

/// Compressed vector representation using PQ codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCode {
    /// Subvector codes (each code is 0-255 for 256 centroids)
    pub codes: Vec<u8>,
}

/// Precomputed lookup table for Asymmetric Distance Computation (ADC)
///
/// Built once per query from the query vector and the codebooks.
/// Layout: table[subspace_idx * num_centroids + centroid_idx] = distance
/// Scoring a PQ code becomes O(num_subvectors) table lookups, eliminating
/// the inner O(subvector_dim) distance computation for every candidate.
pub struct AdcLookupTable {
    /// Flat table: num_subvectors × num_centroids entries
    pub table: Vec<f32>,
    pub num_subvectors: usize,
    pub num_centroids: usize,
}

impl AdcLookupTable {
    /// Score a single PQ code using the precomputed table — O(num_subvectors)
    #[inline]
    pub fn score(&self, code: &PQCode) -> f32 {
        let mut total = 0.0f32;
        for (sub_idx, &c) in code.codes.iter().enumerate() {
            total += self.table[sub_idx * self.num_centroids + c as usize];
        }
        total
    }
}

impl ProductQuantizer {
    /// Whether codebooks have been trained on representative vectors.
    ///
    /// A newly-created quantizer has zero-filled codebooks. Encoding with
    /// those codebooks is technically possible, but every vector receives the
    /// same code and ADC reranking degenerates into an arbitrary tie-break.
    pub fn is_trained(&self) -> bool {
        self.codebooks
            .iter()
            .flatten()
            .flatten()
            .any(|value| *value != 0.0)
    }

    /// Create a new Product Quantizer
    ///
    /// # Arguments
    /// * `dimension` - Total vector dimension
    /// * `num_subvectors` - Number of subvectors (e.g., 8 for PQ8)
    /// * `num_centroids` - Number of centroids per subspace (typically 256)
    pub fn new(
        dimension: usize,
        num_subvectors: usize,
        num_centroids: usize,
        distance_metric: DistanceMetric,
    ) -> crate::Result<Self> {
        if !dimension.is_multiple_of(num_subvectors) {
            return Err(crate::Error::InvalidInput(format!(
                "Dimension {} must be divisible by num_subvectors {}",
                dimension, num_subvectors
            )));
        }

        let subvector_dim = dimension / num_subvectors;

        Ok(Self {
            num_subvectors,
            subvector_dim,
            num_centroids,
            codebooks: vec![vec![vec![0.0; subvector_dim]; num_centroids]; num_subvectors],
            distance_metric,
        })
    }

    /// Train the quantizer using k-means on training data
    ///
    /// # Arguments
    /// * `training_data` - Training vectors to build codebooks
    /// * `num_iterations` - Number of k-means iterations (default: 25)
    pub fn train(
        &mut self,
        training_data: &[DenseVector],
        num_iterations: usize,
    ) -> crate::Result<()> {
        if training_data.is_empty() {
            return Err(crate::Error::InvalidInput(
                "Training data is empty".to_string(),
            ));
        }

        let dimension = training_data[0].len();
        if dimension != self.num_subvectors * self.subvector_dim {
            return Err(crate::Error::InvalidInput(format!(
                "Training data dimension {} doesn't match expected {}",
                dimension,
                self.num_subvectors * self.subvector_dim
            )));
        }

        tracing::info!(
            "Training PQ with {} subvectors, {} centroids, {} training vectors",
            self.num_subvectors,
            self.num_centroids,
            training_data.len()
        );

        // Each subspace is an independent k-means, so they train in parallel. The k-means++
        // seeding below is O(k^2 n) per subspace; sequentially, 96 subspaces x 256 centroids x
        // 65,536 training vectors did not finish inside ten minutes on one core.
        use rayon::prelude::*;
        let subvector_dim = self.subvector_dim;
        self.codebooks = (0..self.num_subvectors)
            .into_par_iter()
            .map(|subspace_idx| {
                let start_dim = subspace_idx * subvector_dim;
                let end_dim = start_dim + subvector_dim;
                let subvectors: Vec<DenseVector> = training_data
                    .iter()
                    .map(|v| v[start_dim..end_dim].to_vec())
                    .collect();
                let centroids = self.kmeans_clustering(&subvectors, num_iterations)?;
                tracing::debug!(
                    "Trained subspace {} ({} centroids)",
                    subspace_idx,
                    self.num_centroids
                );
                Ok(centroids)
            })
            .collect::<crate::Result<Vec<_>>>()?;

        tracing::info!("PQ training completed successfully");
        Ok(())
    }

    /// K-means clustering for one subspace
    fn kmeans_clustering(
        &self,
        subvectors: &[DenseVector],
        num_iterations: usize,
    ) -> crate::Result<Vec<DenseVector>> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let n = subvectors.len();
        let k = self.num_centroids.min(n); // Can't have more centroids than data points

        // Initialize centroids randomly (k-means++)
        let mut centroids = Vec::with_capacity(k);

        // First centroid: random data point
        centroids.push(subvectors.choose(&mut rng).unwrap().clone());

        // k-means++ initialization: choose remaining centroids with probability proportional to distance
        for _ in 1..k {
            let mut distances: Vec<f32> = subvectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .map(|c| self.compute_subvector_distance(v, c))
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                })
                .collect();

            let sum: f32 = distances.iter().sum();
            if sum == 0.0 {
                // All points are already centroids
                break;
            }

            // Normalize to probabilities
            for d in &mut distances {
                *d /= sum;
            }

            // Choose next centroid based on weighted probability
            let mut cumulative = 0.0;
            let random_val: f32 = rand::random();
            for (idx, &prob) in distances.iter().enumerate() {
                cumulative += prob;
                if cumulative >= random_val {
                    centroids.push(subvectors[idx].clone());
                    break;
                }
            }
        }

        // Lloyd's algorithm iterations
        for iteration in 0..num_iterations {
            // Assignment step: assign each vector to nearest centroid
            let mut assignments = vec![0usize; n];
            for (idx, subvector) in subvectors.iter().enumerate() {
                let mut min_dist = f32::INFINITY;
                let mut best_centroid = 0;

                for (c_idx, centroid) in centroids.iter().enumerate() {
                    let dist = self.compute_subvector_distance(subvector, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_centroid = c_idx;
                    }
                }

                assignments[idx] = best_centroid;
            }

            // Update step: recompute centroids
            let mut new_centroids = vec![vec![0.0; self.subvector_dim]; k];
            let mut counts = vec![0usize; k];

            for (idx, &assignment) in assignments.iter().enumerate() {
                for (dim, &val) in subvectors[idx].iter().enumerate() {
                    new_centroids[assignment][dim] += val;
                }
                counts[assignment] += 1;
            }

            // Average to get new centroids
            let mut changed = false;
            for c_idx in 0..k {
                if counts[c_idx] > 0 {
                    let count_f = counts[c_idx] as f32;
                    for v in &mut new_centroids[c_idx] {
                        *v /= count_f;
                    }
                    if !Self::vectors_equal(&centroids[c_idx], &new_centroids[c_idx]) {
                        changed = true;
                    }
                    centroids[c_idx] = new_centroids[c_idx].clone();
                } else {
                    // Empty cluster: reinitialize with random point
                    centroids[c_idx] = subvectors.choose(&mut rng).unwrap().clone();
                    changed = true;
                }
            }

            // Early stopping if converged
            if !changed && iteration > 5 {
                tracing::debug!("K-means converged at iteration {}", iteration);
                break;
            }
        }

        Ok(centroids)
    }

    /// Encode a vector into PQ codes
    pub fn encode(&self, vector: &DenseVector) -> crate::Result<PQCode> {
        if vector.len() != self.num_subvectors * self.subvector_dim {
            return Err(crate::Error::InvalidInput(format!(
                "Vector dimension {} doesn't match expected {}",
                vector.len(),
                self.num_subvectors * self.subvector_dim
            )));
        }

        let mut codes = Vec::with_capacity(self.num_subvectors);

        for subspace_idx in 0..self.num_subvectors {
            let start_dim = subspace_idx * self.subvector_dim;
            let end_dim = start_dim + self.subvector_dim;
            let subvector = &vector[start_dim..end_dim];

            // Find nearest centroid
            let mut min_dist = f32::INFINITY;
            let mut best_code = 0u8;

            for (centroid_idx, centroid) in self.codebooks[subspace_idx].iter().enumerate() {
                let dist = self.compute_subvector_distance(subvector, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_code = centroid_idx as u8;
                }
            }

            codes.push(best_code);
        }

        Ok(PQCode { codes })
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, code: &PQCode) -> crate::Result<DenseVector> {
        if code.codes.len() != self.num_subvectors {
            return Err(crate::Error::InvalidInput(format!(
                "Code length {} doesn't match expected {}",
                code.codes.len(),
                self.num_subvectors
            )));
        }

        let mut vector = Vec::with_capacity(self.num_subvectors * self.subvector_dim);

        for (subspace_idx, &code_val) in code.codes.iter().enumerate() {
            let centroid = &self.codebooks[subspace_idx][code_val as usize];
            vector.extend_from_slice(centroid);
        }

        Ok(vector)
    }

    /// Build an ADC lookup table for a query vector.
    ///
    /// Call once before scanning candidates; each candidate is then scored via
    /// `AdcLookupTable::score()` in O(num_subvectors) instead of O(num_subvectors × subvector_dim).
    pub fn build_lookup_table(&self, query: &DenseVector) -> crate::Result<AdcLookupTable> {
        if query.len() != self.num_subvectors * self.subvector_dim {
            return Err(crate::Error::InvalidInput(format!(
                "Query dimension {} does not match PQ dimension {}",
                query.len(),
                self.num_subvectors * self.subvector_dim
            )));
        }

        let mut table = vec![0.0f32; self.num_subvectors * self.num_centroids];

        for sub_idx in 0..self.num_subvectors {
            let start = sub_idx * self.subvector_dim;
            let end = start + self.subvector_dim;
            let query_sub = &query[start..end];

            for (c_idx, centroid) in self.codebooks[sub_idx].iter().enumerate() {
                table[sub_idx * self.num_centroids + c_idx] =
                    self.compute_subvector_distance(query_sub, centroid);
            }
        }

        Ok(AdcLookupTable {
            table,
            num_subvectors: self.num_subvectors,
            num_centroids: self.num_centroids,
        })
    }

    /// Compute asymmetric distance (query vector vs. PQ code)
    /// This is faster than decoding and more accurate
    pub fn compute_asymmetric_distance(
        &self,
        query: &DenseVector,
        code: &PQCode,
    ) -> crate::Result<f32> {
        if query.len() != self.num_subvectors * self.subvector_dim {
            return Err(crate::Error::InvalidInput(
                "Query dimension mismatch".to_string(),
            ));
        }
        if code.codes.len() != self.num_subvectors {
            return Err(crate::Error::InvalidInput(
                "Code length mismatch".to_string(),
            ));
        }

        let mut total_distance = 0.0;

        for subspace_idx in 0..self.num_subvectors {
            let start_dim = subspace_idx * self.subvector_dim;
            let end_dim = start_dim + self.subvector_dim;
            let query_subvector = &query[start_dim..end_dim];

            let code_val = code.codes[subspace_idx] as usize;
            let centroid = &self.codebooks[subspace_idx][code_val];

            let dist = self.compute_subvector_distance(query_subvector, centroid);
            total_distance += dist;
        }

        Ok(total_distance)
    }

    /// Compute distance between two subvectors
    fn compute_subvector_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.distance_metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                1.0 - (dot / (norm_a * norm_b + 1e-8))
            }
            DistanceMetric::Dot => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
            _ => {
                // Fallback to Euclidean for other metrics
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
        }
    }

    /// Check if two vectors are approximately equal
    fn vectors_equal(a: &[f32], b: &[f32]) -> bool {
        const EPSILON: f32 = 1e-6;
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < EPSILON)
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Codebooks: num_subvectors × num_centroids × subvector_dim × 4 bytes
        self.num_subvectors * self.num_centroids * self.subvector_dim * 4
    }

    /// Get compressed vector size in bytes
    pub fn compressed_size(&self) -> usize {
        // Each code is 1 byte per subvector
        self.num_subvectors
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.num_subvectors * self.subvector_dim * 4; // float32
        let compressed_size = self.num_subvectors; // uint8 codes
        original_size as f32 / compressed_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_creation() {
        let pq = ProductQuantizer::new(128, 8, 256, DistanceMetric::Euclidean).unwrap();
        assert_eq!(pq.num_subvectors, 8);
        assert_eq!(pq.subvector_dim, 16);
        assert_eq!(pq.num_centroids, 256);
    }

    #[test]
    fn test_pq_training() {
        let mut pq = ProductQuantizer::new(64, 8, 16, DistanceMetric::Euclidean).unwrap();

        // Generate random training data
        let training_data: Vec<DenseVector> = (0..100)
            .map(|_| (0..64).map(|_| rand::random::<f32>()).collect())
            .collect();

        pq.train(&training_data, 10).unwrap();

        // Check that codebooks are populated
        assert_eq!(pq.codebooks.len(), 8);
        assert_eq!(pq.codebooks[0].len(), 16);
    }

    #[test]
    fn test_pq_encode_decode() {
        let mut pq = ProductQuantizer::new(64, 8, 16, DistanceMetric::Euclidean).unwrap();

        let training_data: Vec<DenseVector> = (0..100)
            .map(|_| (0..64).map(|_| rand::random::<f32>()).collect())
            .collect();

        pq.train(&training_data, 10).unwrap();

        // Test encode/decode
        let test_vector: DenseVector = (0..64).map(|_| rand::random::<f32>()).collect();
        let code = pq.encode(&test_vector).unwrap();
        let reconstructed = pq.decode(&code).unwrap();

        assert_eq!(reconstructed.len(), 64);
        assert_eq!(code.codes.len(), 8);
    }

    #[test]
    fn test_compression_ratio() {
        let pq = ProductQuantizer::new(960, 120, 256, DistanceMetric::Euclidean).unwrap();

        // Original: 960 × 4 bytes = 3840 bytes
        // Compressed: 120 × 1 byte = 120 bytes
        // Ratio: 3840 / 120 = 32x
        assert_eq!(pq.compression_ratio(), 32.0);
    }

    #[test]
    fn test_adc_lookup_table_matches_asymmetric_distance() {
        let mut pq = ProductQuantizer::new(64, 8, 16, DistanceMetric::Euclidean).unwrap();

        let training_data: Vec<DenseVector> = (0..100)
            .map(|_| (0..64).map(|_| rand::random::<f32>()).collect())
            .collect();
        pq.train(&training_data, 10).unwrap();

        let query: DenseVector = (0..64).map(|_| rand::random::<f32>()).collect();
        let test_vec: DenseVector = (0..64).map(|_| rand::random::<f32>()).collect();
        let code = pq.encode(&test_vec).unwrap();

        let direct = pq.compute_asymmetric_distance(&query, &code).unwrap();
        let lut = pq.build_lookup_table(&query).unwrap();
        let via_lut = lut.score(&code);

        assert!(
            (direct - via_lut).abs() < 1e-4,
            "ADC direct={direct} vs LUT={via_lut} should match"
        );
    }

    #[test]
    fn test_adc_lookup_table_speedup_consistency() {
        // Verify that scanning many candidates via LUT gives the same ranking as direct ADC
        let mut pq = ProductQuantizer::new(64, 8, 16, DistanceMetric::Euclidean).unwrap();
        let training_data: Vec<DenseVector> = (0..100)
            .map(|_| (0..64).map(|_| rand::random::<f32>()).collect())
            .collect();
        pq.train(&training_data, 10).unwrap();

        let query: DenseVector = (0..64).map(|_| rand::random::<f32>()).collect();
        let lut = pq.build_lookup_table(&query).unwrap();

        let candidates: Vec<DenseVector> = (0..50)
            .map(|_| (0..64).map(|_| rand::random::<f32>()).collect())
            .collect();
        let codes: Vec<_> = candidates.iter().map(|v| pq.encode(v).unwrap()).collect();

        // Both methods should produce the same top-1
        let direct_scores: Vec<f32> = codes
            .iter()
            .map(|c| pq.compute_asymmetric_distance(&query, c).unwrap())
            .collect();
        let lut_scores: Vec<f32> = codes.iter().map(|c| lut.score(c)).collect();

        let best_direct = direct_scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let best_lut = lut_scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            best_direct, best_lut,
            "Top-1 ranking must agree between direct ADC and LUT"
        );
    }
}
