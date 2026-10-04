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

        // More than `TRAINING_POINTS_PER_CENTROID` points a centroid adds time and not
        // codebook quality (FAISS caps at 256 a centroid and warns below 39); a caller that
        // hands over 65,536 rows for 256 centroids trains on a stride through 32,768 of them.
        let cap = self.num_centroids * Self::TRAINING_POINTS_PER_CENTROID;
        let stride = training_data.len().div_ceil(cap).max(1);
        let training_data: Vec<&DenseVector> = training_data.iter().step_by(stride).collect();

        // Each subspace is an independent k-means, so they train in parallel.
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

    /// Training points a centroid gets at most; see `train`.
    const TRAINING_POINTS_PER_CENTROID: usize = 128;
    /// Lloyd's stops once fewer than this share of points changed cluster in a round: the
    /// codebook has settled to within the noise of a different random seed by then.
    const CONVERGED_SHARE: f64 = 0.001;

    /// K-means clustering for one subspace.
    ///
    /// Two things this used to do were measured, on a 200K x 25 load, as 62% of every CPU
    /// sample the *server* took while the table built -- more than the graph. The k-means++
    /// seeding recomputed each point's distance to every centroid chosen so far for every
    /// new centroid, O(k^2 n), where keeping each point's running minimum and updating it
    /// against the newest centroid alone is O(k n): 128x fewer distances at k = 256. And
    /// Lloyd's step scored every point against every centroid through a generic
    /// iterator-chain distance that, for cosine, took two square roots per pair; the norms
    /// of the points never change and the centroids' change once an iteration. Convergence
    /// was "every centroid coordinate within 1e-6 of last time", which a 65,536-point
    /// k-means rarely reaches, so the loop almost always ran all 25 rounds; it stops now when
    /// no point changes cluster, which is the condition the next round would prove anyway.
    ///
    /// The distance is the one `compute_subvector_distance` defines, computed in place, so
    /// the codebooks this produces are the ones the old loop produced for the same seeds.
    fn kmeans_clustering(
        &self,
        subvectors: &[DenseVector],
        num_iterations: usize,
    ) -> crate::Result<Vec<DenseVector>> {
        use rand::seq::SliceRandom;
        use rand::{thread_rng, Rng};

        let mut rng = thread_rng();
        let n = subvectors.len();
        let d = self.subvector_dim;
        let k = self.num_centroids.min(n); // Can't have more centroids than data points

        // Contiguous copies: `n` rows of `d` floats, and the per-row norm for cosine.
        let flat: Vec<f32> = subvectors.iter().flat_map(|v| v.iter().copied()).collect();
        let cosine = matches!(self.distance_metric, DistanceMetric::Cosine);
        let dot_metric = matches!(self.distance_metric, DistanceMetric::Dot);
        let norms: Vec<f32> = if cosine {
            (0..n)
                .map(|i| {
                    flat[i * d..(i + 1) * d]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };
        // The distance `compute_subvector_distance` defines, with the point's norm supplied.
        let dist = |p: &[f32], p_norm: f32, c: &[f32], c_norm: f32| -> f32 {
            if cosine {
                let mut dot = 0.0f32;
                for (x, y) in p.iter().zip(c) {
                    dot += x * y;
                }
                1.0 - (dot / (p_norm * c_norm + 1e-8))
            } else if dot_metric {
                let mut dot = 0.0f32;
                for (x, y) in p.iter().zip(c) {
                    dot += x * y;
                }
                -dot
            } else {
                let mut sum = 0.0f32;
                for (x, y) in p.iter().zip(c) {
                    let diff = x - y;
                    sum += diff * diff;
                }
                sum
            }
        };
        let norm_of = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let point = |i: usize| &flat[i * d..(i + 1) * d];
        let point_norm = |i: usize| if cosine { norms[i] } else { 0.0 };

        // Set when the data runs out of distinct points before `k` is reached; the empty
        // clusters that necessarily follow must then be left alone rather than reseeded.
        let mut saturated = false;

        // k-means++: the first centroid is a random point; each later one is drawn with
        // probability proportional to the point's distance to its nearest centroid so far.
        // `min_d` is that nearest distance, updated against the newest centroid only.
        let mut centroids: Vec<DenseVector> = Vec::with_capacity(k);
        let first = rng.gen_range(0..n);
        centroids.push(point(first).to_vec());
        let mut min_d: Vec<f32> = {
            let c = point(first);
            let cn = point_norm(first);
            (0..n)
                .map(|i| dist(point(i), point_norm(i), c, cn))
                .collect()
        };
        for _ in 1..k {
            let sum: f32 = min_d.iter().sum();
            if sum == 0.0 {
                // Every point coincides with a centroid: the data holds fewer distinct
                // points than `k`, so there is nothing left to seed. Pad to `k` so Lloyd's
                // step below, which indexes `centroids[c_idx]` for `c_idx` in `0..k`, is
                // always in bounds; the duplicates are inert, since a point assigns to the
                // first centroid it ties with and the rest stay empty.
                saturated = true;
                while centroids.len() < k {
                    centroids.push(subvectors[centroids.len() % n].clone());
                }
                break;
            }
            // Compared against `random * sum` rather than normalising the weights: a
            // 65,536-term f32 prefix sum of normalised weights can end at 0.9999 and a draw
            // above that selected nothing, leaving the seeding one centroid short. If
            // rounding still leaves the target unreached, the last point is the answer.
            let target = rand::random::<f32>() * sum;
            let mut cumulative = 0.0f32;
            let mut chosen = n - 1;
            for (idx, &weight) in min_d.iter().enumerate() {
                cumulative += weight;
                if cumulative >= target {
                    chosen = idx;
                    break;
                }
            }
            let c = point(chosen).to_vec();
            let cn = point_norm(chosen);
            for (i, slot) in min_d.iter_mut().enumerate() {
                let dd = dist(point(i), point_norm(i), &c, cn);
                if dd < *slot {
                    *slot = dd;
                }
            }
            centroids.push(c);
        }
        // Guaranteed by construction above, both on the normal path and on the padded one.
        // It was a `debug_assert!` and release builds compile those out, so the invariant it
        // documented was not the invariant that shipped.
        assert_eq!(
            centroids.len(),
            k,
            "k-means++ must seed exactly k centroids"
        );

        // Lloyd's algorithm.
        let mut assignments = vec![usize::MAX; n];
        let mut centroid_norms = vec![0.0f32; k];
        for iteration in 0..num_iterations {
            if cosine {
                for (c_idx, c) in centroids.iter().enumerate() {
                    centroid_norms[c_idx] = norm_of(c);
                }
            }
            // Assignment step. A one-dimensional euclidean subspace -- the layout every table
            // under 64 dimensions gets -- is a sorted number line, and the nearest of 256
            // centroids is a binary search, not 256 subtractions.
            let mut moved = 0usize;
            if d == 1 && !cosine && !dot_metric {
                let mut order: Vec<(f32, usize)> = centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (c[0], i))
                    .collect();
                order.sort_by(|a, b| a.0.total_cmp(&b.0));
                let keys: Vec<f32> = order.iter().map(|(v, _)| *v).collect();
                for i in 0..n {
                    let x = flat[i];
                    let pos = keys.partition_point(|&c| c < x);
                    let best = match (pos.checked_sub(1), keys.get(pos)) {
                        (Some(lo), Some(&hi)) if (x - keys[lo]).abs() <= (hi - x).abs() => {
                            order[lo].1
                        }
                        (Some(lo), None) => order[lo].1,
                        (_, Some(_)) => order[pos].1,
                        (None, None) => 0,
                    };
                    if assignments[i] != best {
                        assignments[i] = best;
                        moved += 1;
                    }
                }
            } else {
                for (i, slot) in assignments.iter_mut().enumerate() {
                    let p = point(i);
                    let pn = point_norm(i);
                    let mut best = 0usize;
                    let mut best_d = f32::INFINITY;
                    for (c_idx, c) in centroids.iter().enumerate() {
                        let dd = dist(p, pn, c, centroid_norms[c_idx]);
                        if dd < best_d {
                            best_d = dd;
                            best = c_idx;
                        }
                    }
                    if *slot != best {
                        *slot = best;
                        moved += 1;
                    }
                }
            }
            // Update step.
            let mut sums = vec![0.0f32; k * d];
            let mut counts = vec![0usize; k];
            for (i, &a) in assignments.iter().enumerate() {
                let p = point(i);
                let acc = &mut sums[a * d..(a + 1) * d];
                for (s, x) in acc.iter_mut().zip(p) {
                    *s += x;
                }
                counts[a] += 1;
            }
            for c_idx in 0..k {
                if counts[c_idx] > 0 {
                    let inv = 1.0 / counts[c_idx] as f32;
                    for (dst, s) in centroids[c_idx]
                        .iter_mut()
                        .zip(&sums[c_idx * d..(c_idx + 1) * d])
                    {
                        *dst = s * inv;
                    }
                } else if !saturated {
                    // Empty cluster: reinitialise with a random point. Worth doing while the
                    // data has more distinct points than centroids, because it escapes a poor
                    // seeding; it also means the next round's assignments can change, so
                    // convergence is not claimed on this round.
                    centroids[c_idx] = subvectors.choose(&mut rng).unwrap().clone();
                    moved += 1;
                }
                // When the data has fewer distinct points than `k`, empty clusters are
                // unavoidable and reseeding them only churns. Leave them.
            }
            if (moved as f64) < (n as f64 * Self::CONVERGED_SHARE).max(1.0) {
                tracing::debug!(
                    "K-means converged at iteration {} ({} of {} points moved)",
                    iteration,
                    moved,
                    n
                );
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

    /// Per-subspace term of the ADC table and of k-means assignment.
    ///
    /// Euclidean is the *squared* distance. The table sums one term per subspace, and a sum
    /// of squared subspace distances is the squared distance of the whole vector; a sum of
    /// square roots is not, and does not rank the same way. Measured on 1M Cohere 768-d with
    /// a full scan over the codes, the square-root version scored 0.5-1.9 points lower
    /// recall@10 before rerank (48, 96 and 192 subspaces). k-means is unaffected: argmin is
    /// the same under the square.
    fn compute_subvector_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.distance_metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>(),
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

    /// Training data with fewer distinct points per subspace than there are centroids.
    ///
    /// This is not a corner case invented for a test: SIFT is 128 dimensions of small
    /// integers, our default splits it into 32 subspaces of 4 dimensions, and one of those
    /// subspaces held 86 distinct points against 256 centroids. Seeding stopped at 86 while
    /// Lloyd's step indexed `centroids[c_idx]` for `c_idx` in `0..k`, and the server aborted
    /// -- in release, where the `debug_assert!` that documented the invariant was compiled
    /// out. The benchmark case that found it is CapacityDim128, which we had never run.
    #[test]
    fn training_survives_fewer_distinct_points_than_centroids() {
        const DIM: usize = 128;
        const SUBS: usize = 32;
        const K: usize = 256;
        // Small integers, exactly the shape that collapses a 4-dimension subspace.
        let training: Vec<DenseVector> = (0..4096)
            .map(|i: usize| (0..DIM).map(|d| ((i * 31 + d * 7) % 11) as f32).collect())
            .collect();

        let mut pq = ProductQuantizer::new(DIM, SUBS, K, DistanceMetric::Euclidean).unwrap();
        pq.train(&training, 10)
            .expect("training must not panic or fail");

        // Every subspace must offer the full codebook, or the ADC table and the one-byte
        // codes disagree about how many centroids a subspace has.
        assert_eq!(pq.codebooks.len(), SUBS);
        for (sub, book) in pq.codebooks.iter().enumerate() {
            assert_eq!(book.len(), K, "subspace {sub} has {} centroids", book.len());
        }

        // And it still encodes and decodes every vector it was trained on.
        for v in training.iter().take(64) {
            let code = pq.encode(v).expect("encode");
            assert_eq!(code.codes.len(), SUBS);
            let back = pq.decode(&code).expect("decode");
            assert_eq!(back.len(), DIM);
        }
    }

    /// The degenerate extreme: one distinct point, many centroids.
    #[test]
    fn training_survives_a_single_distinct_point() {
        let training: Vec<DenseVector> = (0..512).map(|_| vec![1.0f32; 32]).collect();
        let mut pq = ProductQuantizer::new(32, 8, 64, DistanceMetric::Euclidean).unwrap();
        pq.train(&training, 5)
            .expect("training must not panic or fail");
        for book in &pq.codebooks {
            assert_eq!(book.len(), 64);
        }
        let code = pq.encode(&training[0]).expect("encode");
        assert_eq!(code.codes.len(), 8);
    }

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
