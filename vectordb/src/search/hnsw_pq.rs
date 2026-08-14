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
