use crate::search::hnsw::{HnswConfig, HnswIndex};
use crate::search::quantization::product_quantization::{AdcLookupTable, PQCode, ProductQuantizer};
use crate::types::{DenseVector, DistanceMetric, RecordId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// HNSW index with Product Quantization for memory efficiency
///
/// This combines HNSW's O(log N) search with PQ's memory compression.
/// Memory reduction: 4-32x with < 1% recall loss
///
/// Example memory savings for 1M vectors × 960 dims:
/// - Without PQ: ~3.7 GB
/// - With PQ8: ~120 MB (32x compression)
#[derive(Serialize, Deserialize)]
pub struct HnswPqIndex {
    /// HNSW index structure (stores graph topology)
    hnsw: HnswIndex,

    /// Product Quantizer
    pq: ProductQuantizer,

    /// Compressed vectors (PQ codes)
    compressed_vectors: HashMap<RecordId, PQCode>,

    /// Original vectors for high-precision reranking (optional)
    /// Only store top-k candidates to save memory
    original_vectors: Option<HashMap<RecordId, DenseVector>>,

    /// Configuration
    config: HnswPqConfig,

    /// Distance metric
    distance_metric: DistanceMetric,

    /// Vector dimension
    #[allow(dead_code)]
    dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswPqConfig {
    /// HNSW parameters
    pub hnsw_config: HnswConfig,

    /// Number of subvectors for PQ
    pub num_subvectors: usize,

    /// Number of centroids per subspace (typically 256 for 8-bit)
    pub num_centroids: usize,

    /// Whether to store original vectors for reranking
    pub enable_reranking: bool,

    /// Reranking factor: retrieve k * rerank_factor candidates, then rerank
    pub rerank_factor: usize,
}

impl Default for HnswPqConfig {
    fn default() -> Self {
        Self {
            hnsw_config: HnswConfig::default(),
            num_subvectors: 120, // For 960-dim: 120 × 8 = 960
            num_centroids: 256,  // 8-bit encoding
            enable_reranking: false,
            rerank_factor: 3,
        }
    }
}

impl HnswPqIndex {
    /// Fewest training vectors per centroid. k-means over 256 centroids fitted to 300
    /// points describes those points, not the distribution; at 16 apiece the default
    /// 256-centroid codebook wants 4,096 vectors, and a smaller codebook scales down with it
    /// rather than being held to a number chosen for a bigger one.
    const MIN_VECTORS_PER_CENTROID: usize = 16;

    /// Most vectors to train on. Training is k-means over every subspace and its cost is
    /// linear in this; past a few tens of thousands the codebooks stop moving.
    const MAX_TRAINING_VECTORS: usize = 65_536;

    /// Create a new HNSW-PQ index
    pub fn new(
        dimension: usize,
        distance_metric: DistanceMetric,
        config: Option<HnswPqConfig>,
    ) -> crate::Result<Self> {
        let config = config.unwrap_or_default();

        let pq = ProductQuantizer::new(
            dimension,
            config.num_subvectors,
            config.num_centroids,
            distance_metric,
        )?;

        let hnsw = HnswIndex::new(distance_metric, Some(config.hnsw_config.clone()));

        Ok(Self {
            hnsw,
            pq,
            compressed_vectors: HashMap::new(),
            original_vectors: if config.enable_reranking {
                Some(HashMap::new())
            } else {
                None
            },
            config,
            distance_metric,
            dimension,
        })
    }

    /// Train the PQ quantizer on sample data
    ///
    /// Should be called before inserting vectors, with representative sample
    pub fn train(&mut self, training_data: &[DenseVector]) -> crate::Result<()> {
        tracing::info!("Training PQ quantizer with {} vectors", training_data.len());

        self.pq.train(training_data, 25)?;

        tracing::info!("PQ training completed");
        Ok(())
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        // Encode vector with PQ
        let pq_code = self.pq.encode(&vector)?;

        // Insert into HNSW using original vector (for graph construction accuracy)
        self.hnsw.insert(id.clone(), vector.clone())?;

        // Store compressed version
        self.compressed_vectors.insert(id.clone(), pq_code);

        // Optionally store original for reranking
        if let Some(ref mut orig_vecs) = self.original_vectors {
            orig_vecs.insert(id, vector);
        }

        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        // Determine candidate size for reranking
        let candidate_k = if self.config.enable_reranking {
            k * self.config.rerank_factor
        } else {
            k
        };

        // HNSW graph traversal to get candidates
        let candidates = self.hnsw.search(query, candidate_k)?;

        // Storage can create an HNSW-PQ index before a representative training
        // sample is available.  Zero-filled codebooks assign the same PQ code
        // to every row, so reranking with ADC would destroy the valid HNSW
        // ordering and return arbitrary records.  Until training is performed,
        // preserve the graph distances; callers still get a meaningful ANN
        // result and a later trained index automatically uses PQ reranking.
        if !self.pq.is_trained() {
            return Ok(candidates.into_iter().take(k).collect());
        }

        // Rerank using PQ asymmetric distance or original vectors
        let mut results = if self.config.enable_reranking {
            self.rerank_with_original(query, candidates)?
        } else {
            self.rerank_with_pq(query, candidates)?
        };

        // Sort by distance and return top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    /// Rerank candidates using PQ asymmetric distance with precomputed ADC lookup table.
    ///
    /// The lookup table is built once per query (O(num_subvectors × num_centroids × subvector_dim))
    /// and each candidate is scored in O(num_subvectors) instead of O(num_subvectors × subvector_dim),
    /// giving a speedup proportional to subvector_dim (typically 8x for 960-dim PQ120).
    fn rerank_with_pq(
        &self,
        query: &DenseVector,
        candidates: Vec<(RecordId, f32)>,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        // Build lookup table once for this query
        let lut: AdcLookupTable = self.pq.build_lookup_table(query)?;

        let mut results = Vec::with_capacity(candidates.len());
        for (id, _hnsw_dist) in candidates {
            if let Some(pq_code) = self.compressed_vectors.get(&id) {
                results.push((id, lut.score(pq_code)));
            }
        }

        Ok(results)
    }

    /// Rerank candidates using original vectors (higher accuracy)
    fn rerank_with_original(
        &self,
        query: &DenseVector,
        candidates: Vec<(RecordId, f32)>,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        if let Some(ref orig_vecs) = self.original_vectors {
            let mut results = Vec::with_capacity(candidates.len());

            for (id, _hnsw_dist) in candidates {
                if let Some(orig_vec) = orig_vecs.get(&id) {
                    let distance = self.compute_distance(query, orig_vec);
                    results.push((id, distance));
                }
            }

            Ok(results)
        } else {
            // Fallback to PQ if original vectors not available
            self.rerank_with_pq(query, candidates)
        }
    }

    /// Compute distance between two vectors
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
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
                // Fallback to Euclidean
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
        }
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let num_vectors = self.compressed_vectors.len();

        // PQ compressed vectors
        let pq_memory = num_vectors * self.pq.compressed_size();

        // PQ codebooks
        let codebook_memory = self.pq.memory_usage();

        // HNSW graph structure (approximate)
        let hnsw_memory = num_vectors * (self.config.hnsw_config.m * 2 * 8); // Edges + metadata

        // Original vectors (if stored)
        let original_memory = if let Some(ref orig) = self.original_vectors {
            orig.len() * 960 * 4 // dimension × float32
        } else {
            0
        };

        let total = pq_memory + codebook_memory + hnsw_memory + original_memory;

        // Compression ratio: original vector bytes vs PQ-encoded bytes (excluding shared HNSW graph)
        let uncompressed_vector_memory = num_vectors * self.dimension * 4;
        let compressed_vector_memory = (pq_memory + original_memory).max(1);

        MemoryStats {
            total_bytes: total,
            pq_codes_bytes: pq_memory,
            codebook_bytes: codebook_memory,
            hnsw_graph_bytes: hnsw_memory,
            original_vectors_bytes: original_memory,
            num_vectors,
            compression_ratio: uncompressed_vector_memory as f32 / compressed_vector_memory as f32,
        }
    }

    /// Get number of indexed vectors
    pub fn len(&self) -> usize {
        self.compressed_vectors.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.compressed_vectors.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_bytes: usize,
    pub pq_codes_bytes: usize,
    pub codebook_bytes: usize,
    pub hnsw_graph_bytes: usize,
    pub original_vectors_bytes: usize,
    pub num_vectors: usize,
    pub compression_ratio: f32,
}

impl MemoryStats {
    pub fn total_mb(&self) -> f32 {
        self.total_bytes as f32 / 1024.0 / 1024.0
    }

    pub fn total_gb(&self) -> f32 {
        self.total_bytes as f32 / 1024.0 / 1024.0 / 1024.0
    }
}

// Implement VectorIndex trait for HNSW-PQ
#[async_trait::async_trait]
impl crate::search::index_traits::VectorIndex for HnswPqIndex {
    async fn add(&mut self, id: String, vector: Vec<f32>) -> crate::Result<()> {
        self.insert(id, vector)
    }

    /// Train the codebooks on the data, then insert it.
    ///
    /// Without this override the trait's default `bulk_build` calls `add` in a loop and
    /// nothing ever calls `train`: `HnswPqIndex::train` had no caller outside its own tests.
    /// An untrained quantizer has zero-filled codebooks, so every vector encodes to the same
    /// code; `search` notices and falls back to the graph's own ordering, which is why this
    /// returned correct results while the compression it was chosen for did not happen. The
    /// index is selected when a table's HNSW graph would exceed its memory budget, so the
    /// saving is the entire reason it is there.
    ///
    /// Training needs enough vectors to place `num_centroids` centroids per subspace; below
    /// that, k-means would fit codebooks to noise, and a fallback that is merely slower beats
    /// a codebook that is wrong. The sample is a fixed stride rather than a random draw so
    /// two builds of the same data produce the same codebooks.
    async fn bulk_build(&mut self, vectors: Vec<(String, Vec<f32>)>) -> crate::Result<()> {
        let wanted = self
            .config
            .num_centroids
            .saturating_mul(Self::MIN_VECTORS_PER_CENTROID);
        if vectors.len() >= wanted {
            let stride = (vectors.len() / Self::MAX_TRAINING_VECTORS).max(1);
            let sample: Vec<DenseVector> = vectors
                .iter()
                .step_by(stride)
                .take(Self::MAX_TRAINING_VECTORS)
                .map(|(_, v)| v.clone())
                .collect();
            self.train(&sample)?;
        } else {
            tracing::info!(
                "HNSW-PQ: {} vectors is below the {} needed to train {} centroids per \
                 subspace; leaving the quantizer untrained (search falls back to the graph \
                 ordering, and the memory saving does not happen)",
                vectors.len(),
                wanted,
                self.config.num_centroids
            );
        }
        for (id, vector) in vectors {
            self.insert(id, vector)?;
        }
        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> crate::Result<Vec<(String, f32)>> {
        self.search(&query.to_vec(), k)
    }

    async fn delete(&mut self, id: &str) -> crate::Result<()> {
        // Delete from HNSW graph structure
        self.hnsw.delete(id).await?;

        // Remove compressed vector
        self.compressed_vectors.remove(id);

        // Remove original vector if stored for reranking
        if let Some(ref mut orig_vecs) = self.original_vectors {
            orig_vecs.remove(id);
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn snapshot_to(&self, path: &std::path::Path) -> crate::Result<bool> {
        crate::search::index_traits::SerializableIndex::save(self, path)?;
        Ok(true)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

impl crate::search::index_traits::SerializableIndex for HnswPqIndex {
    fn save(&self, path: &std::path::Path) -> crate::Result<()> {
        // The codebooks travel with the codes. Without them the stored PQ codes are just
        // centroid indices into a table that no longer exists, and every distance computed
        // after a load would be wrong while still looking like a number.
        crate::search::index_persistence::write_index(path, "hnsw-pq", self)
    }

    fn load(path: &std::path::Path) -> crate::Result<Self> {
        crate::search::index_persistence::read_index(path, "hnsw-pq")
    }

    fn stats(&self) -> crate::search::index_traits::IndexStats {
        let codes_bytes = self.compressed_vectors.len() * self.config.num_subvectors;
        let codebook_bytes = self.config.num_subvectors
            * self.config.num_centroids
            * (self.dimension / self.config.num_subvectors.max(1))
            * std::mem::size_of::<f32>();
        let originals_bytes = self
            .original_vectors
            .as_ref()
            .map(|m| m.len() * self.dimension * std::mem::size_of::<f32>())
            .unwrap_or(0);

        crate::search::index_traits::IndexStats {
            total_vectors: self.compressed_vectors.len(),
            index_size_bytes: codes_bytes + codebook_bytes + originals_bytes,
            build_time_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic corpus where every vector is distinct.
    ///
    /// An earlier version of this gave each vector one of 16 centres, which made the nearest
    /// neighbour a tie among hundreds of near-identical rows and the top-1 assertion a coin
    /// flip. Distinct vectors are the point: a quantizer that collapses them is exactly what
    /// these tests are looking for.
    fn corpus(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0
        };
        (0..n)
            .map(|i| (format!("v{i}"), (0..dim).map(|_| next()).collect()))
            .collect()
    }

    #[tokio::test]
    async fn bulk_build_trains_the_quantizer() {
        use crate::search::index_traits::VectorIndex;
        let config = HnswPqConfig {
            num_subvectors: 8,
            num_centroids: 16,
            ..HnswPqConfig::default()
        };
        let mut index = HnswPqIndex::new(16, DistanceMetric::Euclidean, Some(config)).unwrap();
        assert!(
            !index.pq.is_trained(),
            "a fresh quantizer starts with zero-filled codebooks"
        );

        // 16 centroids x MIN_VECTORS_PER_CENTROID is 256, so this trains.
        let data = corpus(600, 16);
        index.bulk_build(data.clone()).await.unwrap();

        assert!(
            index.pq.is_trained(),
            "bulk_build must train the codebooks; an untrained quantizer encodes every \
             vector to the same code and the memory saving this index exists for does not \
             happen"
        );
        assert_eq!(index.len(), 600);

        // Distinct vectors must not all collapse onto one code -- which is exactly what an
        // untrained codebook does.
        let first = index
            .compressed_vectors
            .values()
            .next()
            .expect("codes were stored")
            .clone();
        assert!(
            index
                .compressed_vectors
                .values()
                .any(|c| c.codes != first.codes),
            "every vector encoded to the same PQ code"
        );

        let hits = VectorIndex::search(&index, &data[0].1, 5).await.unwrap();
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].0, "v0", "the query's own vector should rank first");
    }

    #[tokio::test]
    async fn bulk_build_leaves_a_small_corpus_untrained_and_still_answers() {
        use crate::search::index_traits::VectorIndex;
        let config = HnswPqConfig {
            num_subvectors: 8,
            num_centroids: 16,
            ..HnswPqConfig::default()
        };
        let mut index = HnswPqIndex::new(16, DistanceMetric::Euclidean, Some(config)).unwrap();

        // Under the 256 that 16 centroids want: fitting them here would describe the sample
        // rather than the distribution, so the quantizer stays untrained and search falls
        // back to the graph ordering -- slower than PQ, but correct.
        let data = corpus(200, 16);
        index.bulk_build(data.clone()).await.unwrap();

        assert!(!index.pq.is_trained());
        assert_eq!(index.len(), 200);
        let hits = VectorIndex::search(&index, &data[7].1, 3).await.unwrap();
        assert_eq!(hits[0].0, "v7");
    }

    #[test]
    fn test_hnsw_pq_creation() {
        // For 128 dimensions, use 64 subvectors (128/64 = 2)
        let config = HnswPqConfig {
            num_subvectors: 64,
            ..HnswPqConfig::default()
        };
        let index = HnswPqIndex::new(128, DistanceMetric::Euclidean, Some(config)).unwrap();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_pq_insert_search() {
        // For 64 dimensions, use 32 subvectors (64/32 = 2)
        let config = HnswPqConfig {
            num_subvectors: 32,
            ..HnswPqConfig::default()
        };
        let mut index = HnswPqIndex::new(64, DistanceMetric::Euclidean, Some(config)).unwrap();

        // Generate training data
        let training_data: Vec<DenseVector> = (0..100)
            .map(|_| (0..64).map(|_| rand::random::<f32>()).collect())
            .collect();

        // Train PQ
        index.train(&training_data).unwrap();

        // Insert vectors
        for (i, vec) in training_data.iter().take(50).enumerate() {
            index.insert(format!("vec_{}", i), vec.clone()).unwrap();
        }

        assert_eq!(index.len(), 50);

        // Search
        let query: DenseVector = (0..64).map(|_| rand::random::<f32>()).collect();
        let results = index.search(&query, 10).unwrap();

        assert!(results.len() <= 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_untrained_pq_preserves_hnsw_ordering() {
        // Storage may build an HNSW-PQ index before a training sample exists.
        // The zero-filled codebooks must not collapse every ADC score to the
        // same value and turn search into arbitrary insertion order.
        let config = HnswPqConfig {
            num_subvectors: 4,
            hnsw_config: HnswConfig::from_params(16, 100, 100),
            ..HnswPqConfig::default()
        };
        let mut index = HnswPqIndex::new(8, DistanceMetric::Euclidean, Some(config)).unwrap();
        let vectors: Vec<DenseVector> = (0..40)
            .map(|i| {
                let mut v = vec![10.0; 8];
                v[0] = i as f32;
                v
            })
            .collect();
        for (i, vector) in vectors.iter().enumerate() {
            index.insert(format!("v{i}"), vector.clone()).unwrap();
        }

        assert!(!index.pq.is_trained());
        let results = index.search(&vectors[0], 1).unwrap();
        assert_eq!(results.first().map(|(id, _)| id.as_str()), Some("v0"));
    }

    #[test]
    fn test_memory_stats() {
        let mut index = HnswPqIndex::new(960, DistanceMetric::Euclidean, None).unwrap();

        let training_data: Vec<DenseVector> = (0..100)
            .map(|_| (0..960).map(|_| rand::random::<f32>()).collect())
            .collect();

        index.train(&training_data).unwrap();

        for (i, vec) in training_data.iter().enumerate() {
            index.insert(format!("vec_{}", i), vec.clone()).unwrap();
        }

        let stats = index.memory_stats();
        println!("Memory stats: {:?}", stats);

        // Should have significant compression
        assert!(stats.compression_ratio > 1.0);
        assert!(stats.total_mb() < 10.0); // 100 vectors should be < 10MB with PQ
    }
}
