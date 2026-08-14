//! Inverted File with Product Quantization (IVF-PQ) Vector Index
//! 
//! A memory-efficient vector index using:
//! - Inverted File structure for coarse quantization (k-means clustering)
//! - Product Quantization for fine-grained compression
//! - Asymmetric Distance Computation (ADC) for fast search
//! - Persistent storage with incremental updates

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Arc, RwLock, Mutex};
use serde::{Serialize, Deserialize};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::types::{Result, VectorDbError};

/// IVF-PQ configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPqConfig {
    /// Number of inverted lists (coarse quantizers)
    pub n_lists: usize,
    /// Number of subquantizers for product quantization
    pub n_subquantizers: usize,
    /// Number of bits per subquantizer (2^bits centroids per subquantizer)
    pub bits_per_subquantizer: usize,
    /// Number of k-means iterations for training
    pub kmeans_iterations: usize,
    /// Number of lists to search during query
    pub n_probe: usize,
    /// Random seed for reproducible training
    pub seed: u64,
    /// Enable polysemous codes for better recall
    pub use_polysemous: bool,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            n_lists: 1024,
            n_subquantizers: 8,
            bits_per_subquantizer: 8, // 256 centroids per subquantizer
            kmeans_iterations: 25,
            n_probe: 8,
            seed: 42,
            use_polysemous: false,
        }
    }
}

/// Vector identifier
pub type VectorId = u32;

/// Compressed vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVector {
    pub id: VectorId,
    /// Coarse quantizer assignment (inverted list ID)
    pub list_id: usize,
    /// Product quantization codes (one per subquantizer)
    pub pq_codes: Vec<u8>,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Coarse quantizer (k-means centroid)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoarseQuantizer {
    /// Centroid vector
    centroid: Vec<f32>,
    /// Assigned vectors in this cluster
    vectors: Vec<CompressedVector>,
}

/// Product quantizer for one subspace
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProductQuantizer {
    /// Centroids for this subquantizer [n_centroids x subvector_dim]
    centroids: Vec<Vec<f32>>,
    /// Dimension of subvectors
    subvector_dim: usize,
}

impl ProductQuantizer {
    fn new(subvector_dim: usize, n_centroids: usize) -> Self {
        Self {
            centroids: vec![vec![0.0; subvector_dim]; n_centroids],
            subvector_dim,
        }
    }
    
    /// Train quantizer on subvectors
    fn train(&mut self, subvectors: &[Vec<f32>], iterations: usize, rng: &mut ChaCha8Rng) {
        if subvectors.is_empty() {
            return;
        }
        
        let n_centroids = self.centroids.len();
        
        // Initialize centroids randomly
        for centroid in &mut self.centroids {
            let idx = rng.gen_range(0..subvectors.len());
            centroid.clone_from(&subvectors[idx]);
        }
        
        // K-means iterations
        for _ in 0..iterations {
            let mut assignments = vec![Vec::new(); n_centroids];
            
            // Assign vectors to closest centroids
            for subvector in subvectors {
                let mut best_dist = f32::INFINITY;
                let mut best_centroid = 0;
                
                for (i, centroid) in self.centroids.iter().enumerate() {
                    let dist = euclidean_distance(subvector, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_centroid = i;
                    }
                }
                
                assignments[best_centroid].push(subvector);
            }
            
            // Update centroids
            for (i, assigned_vectors) in assignments.iter().enumerate() {
                if !assigned_vectors.is_empty() {
                    for j in 0..self.subvector_dim {
                        let sum: f32 = assigned_vectors.iter()
                            .map(|v| v[j])
                            .sum();
                        self.centroids[i][j] = sum / assigned_vectors.len() as f32;
                    }
                }
            }
        }
    }
    
    /// Encode subvector to quantization code
    fn encode(&self, subvector: &[f32]) -> u8 {
        let mut best_dist = f32::INFINITY;
        let mut best_code = 0u8;
        
        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist = euclidean_distance(subvector, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_code = i as u8;
            }
        }
        
        best_code
    }
    
    /// Decode quantization code to centroid
    fn decode(&self, code: u8) -> &Vec<f32> {
        &self.centroids[code as usize]
    }
    
    /// Compute distance table for ADC
    fn compute_distance_table(&self, query_subvector: &[f32]) -> Vec<f32> {
        self.centroids.iter()
            .map(|centroid| euclidean_distance(query_subvector, centroid))
            .collect()
    }
}

/// IVF-PQ index implementation
pub struct IvfPqIndex {
    config: IvfPqConfig,
    dimension: usize,
    
    /// Coarse quantizers (inverted lists)
    coarse_quantizers: Arc<RwLock<Vec<CoarseQuantizer>>>,
    
    /// Product quantizers for each subspace
    product_quantizers: Arc<RwLock<Vec<ProductQuantizer>>>,
    
    /// Trained flag
    is_trained: Arc<RwLock<bool>>,
    
    /// Next available vector ID
    next_id: Arc<Mutex<VectorId>>,
    
    /// Random number generator
    rng: Arc<Mutex<ChaCha8Rng>>,
    
    /// File path for persistence
    file_path: Option<String>,
}

impl IvfPqIndex {
    /// Create new IVF-PQ index
    pub fn new(dimension: usize, config: IvfPqConfig) -> Result<Self> {
        if dimension % config.n_subquantizers != 0 {
            return Err(VectorDbError::InvalidDimension {
                expected: dimension,
                actual: dimension,
            });
        }
        
        let subvector_dim = dimension / config.n_subquantizers;
        let n_centroids = 1 << config.bits_per_subquantizer; // 2^bits
        
        let coarse_quantizers = vec![
            CoarseQuantizer {
                centroid: vec![0.0; dimension],
                vectors: Vec::new(),
            };
            config.n_lists
        ];
        
        let product_quantizers = (0..config.n_subquantizers)
            .map(|_| ProductQuantizer::new(subvector_dim, n_centroids))
            .collect();
        
        let rng = ChaCha8Rng::seed_from_u64(config.seed);
        
        Ok(Self {
            config,
            dimension,
            coarse_quantizers: Arc::new(RwLock::new(coarse_quantizers)),
            product_quantizers: Arc::new(RwLock::new(product_quantizers)),
            is_trained: Arc::new(RwLock::new(false)),
            next_id: Arc::new(Mutex::new(0)),
            rng: Arc::new(Mutex::new(rng)),
            file_path: None,
        })
    }
    
    /// Train index on training vectors
    pub fn train(&self, training_vectors: &[Vec<f32>]) -> Result<()> {
        if training_vectors.is_empty() {
            return Err(VectorDbError::InvalidInput("No training vectors provided".to_string()));
        }
        
        // Validate dimensions
        for vector in training_vectors {
            if vector.len() != self.dimension {
                return Err(VectorDbError::InvalidDimension {
                    expected: self.dimension,
                    actual: vector.len(),
                });
            }
        }
        
        let mut rng = self.rng.lock().unwrap();
        
        // 1. Train coarse quantizers (k-means on full vectors)
        self.train_coarse_quantizers(training_vectors, &mut rng)?;
        
        // 2. Train product quantizers (k-means on subvectors)
        self.train_product_quantizers(training_vectors, &mut rng)?;
        
        *self.is_trained.write().unwrap() = true;
        
        Ok(())
    }
    
    /// Add vector to index (requires trained index)
    pub fn add(&self, vector: Vec<f32>, metadata: Option<serde_json::Value>) -> Result<VectorId> {
        if !*self.is_trained.read().unwrap() {
            return Err(VectorDbError::InvalidInput("Index must be trained before adding vectors".to_string()));
        }
        
        if vector.len() != self.dimension {
            return Err(VectorDbError::InvalidDimension {
                expected: self.dimension,
                actual: vector.len(),
            });
        }
        
        let id = {
            let mut next_id = self.next_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };
        
        // Find closest coarse quantizer
        let list_id = self.find_closest_coarse_quantizer(&vector)?;
        
        // Encode with product quantizers
        let pq_codes = self.encode_vector(&vector)?;
        
        let compressed_vector = CompressedVector {
            id,
            list_id,
            pq_codes,
            metadata,
        };
        
        // Add to inverted list
        {
            let mut quantizers = self.coarse_quantizers.write().unwrap();
            quantizers[list_id].vectors.push(compressed_vector);
        }
        
        Ok(id)
    }
    
    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>> {
        if !*self.is_trained.read().unwrap() {
            return Err(VectorDbError::InvalidInput("Index must be trained before searching".to_string()));
        }
        
        if query.len() != self.dimension {
            return Err(VectorDbError::InvalidDimension {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        
        // Find n_probe closest coarse quantizers
        let probe_lists = self.find_probe_lists(query)?;
        
        // Compute distance tables for ADC
        let distance_tables = self.compute_distance_tables(query)?;
        
        // Search in selected lists
        let mut candidates = BinaryHeap::new();
        
        let quantizers = self.coarse_quantizers.read().unwrap();
        
        for &list_id in &probe_lists {
            for compressed_vector in &quantizers[list_id].vectors {
                let distance = self.compute_adc_distance(&compressed_vector.pq_codes, &distance_tables);
                
                if candidates.len() < k {
                    candidates.push(Reverse((distance, compressed_vector.id)));
                } else if let Some(Reverse((worst_dist, _))) = candidates.peek() {
                    if distance < *worst_dist {
                        candidates.pop();
                        candidates.push(Reverse((distance, compressed_vector.id)));
                    }
                }
            }
        }
        
        // Convert to sorted results
        let mut results: Vec<_> = candidates.into_iter()
            .map(|Reverse((dist, id))| (id, dist))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        Ok(results)
    }
    
    /// Save index to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        let mut writer = BufWriter::new(file);
        
        // Write header
        let header = IvfPqHeader {
            dimension: self.dimension,
            config: self.config.clone(),
            is_trained: *self.is_trained.read().unwrap(),
            next_id: *self.next_id.lock().unwrap(),
        };
        
        bincode::serialize_into(&mut writer, &header)
            .map_err(|e| VectorDbError::SerializationError(e.to_string()))?;
        
        // Write coarse quantizers
        let coarse_quantizers = self.coarse_quantizers.read().unwrap();
        bincode::serialize_into(&mut writer, &*coarse_quantizers)
            .map_err(|e| VectorDbError::SerializationError(e.to_string()))?;
        
        // Write product quantizers
        let product_quantizers = self.product_quantizers.read().unwrap();
        bincode::serialize_into(&mut writer, &*product_quantizers)
            .map_err(|e| VectorDbError::SerializationError(e.to_string()))?;
        
        writer.flush()?;
        Ok(())
    }
    
    /// Load index from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);
        
        // Read header
        let header: IvfPqHeader = bincode::deserialize_from(&mut reader)
            .map_err(|e| VectorDbError::SerializationError(e.to_string()))?;
        
        let mut index = Self::new(header.dimension, header.config)?;
        index.file_path = Some(path.as_ref().to_string_lossy().to_string());
        
        // Read coarse quantizers
        let coarse_quantizers: Vec<CoarseQuantizer> = bincode::deserialize_from(&mut reader)
            .map_err(|e| VectorDbError::SerializationError(e.to_string()))?;
        *index.coarse_quantizers.write().unwrap() = coarse_quantizers;
        
        // Read product quantizers
        let product_quantizers: Vec<ProductQuantizer> = bincode::deserialize_from(&mut reader)
            .map_err(|e| VectorDbError::SerializationError(e.to_string()))?;
        *index.product_quantizers.write().unwrap() = product_quantizers;
        
        *index.is_trained.write().unwrap() = header.is_trained;
        *index.next_id.lock().unwrap() = header.next_id;
        
        Ok(index)
    }
    
    /// Get index statistics
    pub fn stats(&self) -> IvfPqStats {
        let quantizers = self.coarse_quantizers.read().unwrap();
        
        let total_vectors: usize = quantizers.iter()
            .map(|q| q.vectors.len())
            .sum();
        
        let mut list_sizes = Vec::new();
        let mut non_empty_lists = 0;
        
        for quantizer in quantizers.iter() {
            let size = quantizer.vectors.len();
            list_sizes.push(size);
            if size > 0 {
                non_empty_lists += 1;
            }
        }
        
        let avg_list_size = if non_empty_lists > 0 {
            total_vectors as f64 / non_empty_lists as f64
        } else {
            0.0
        };
        
        // Estimate compression ratio
        let original_size = total_vectors * self.dimension * 4; // 4 bytes per float
        let compressed_size = total_vectors * self.config.n_subquantizers; // 1 byte per subquantizer
        let compression_ratio = if compressed_size > 0 {
            original_size as f64 / compressed_size as f64
        } else {
            0.0
        };
        
        IvfPqStats {
            total_vectors,
            n_lists: quantizers.len(),
            non_empty_lists,
            avg_list_size,
            compression_ratio,
            is_trained: *self.is_trained.read().unwrap(),
        }
    }
    
    // Helper methods
    
    fn train_coarse_quantizers(&self, vectors: &[Vec<f32>], rng: &mut ChaCha8Rng) -> Result<()> {
        let mut quantizers = self.coarse_quantizers.write().unwrap();
        
        // Initialize centroids randomly
        for quantizer in quantizers.iter_mut() {
            let idx = rng.gen_range(0..vectors.len());
            quantizer.centroid.clone_from(&vectors[idx]);
        }
        
        // K-means iterations
        for _ in 0..self.config.kmeans_iterations {
            let mut assignments = vec![Vec::new(); quantizers.len()];
            
            // Assign vectors to closest centroids
            for vector in vectors {
                let mut best_dist = f32::INFINITY;
                let mut best_quantizer = 0;
                
                for (i, quantizer) in quantizers.iter().enumerate() {
                    let dist = euclidean_distance(vector, &quantizer.centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_quantizer = i;
                    }
                }
                
                assignments[best_quantizer].push(vector);
            }
            
            // Update centroids
            for (i, assigned_vectors) in assignments.iter().enumerate() {
                if !assigned_vectors.is_empty() {
                    for j in 0..self.dimension {
                        let sum: f32 = assigned_vectors.iter()
                            .map(|v| v[j])
                            .sum();
                        quantizers[i].centroid[j] = sum / assigned_vectors.len() as f32;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn train_product_quantizers(&self, vectors: &[Vec<f32>], rng: &mut ChaCha8Rng) -> Result<()> {
        let mut pq = self.product_quantizers.write().unwrap();
        let subvector_dim = self.dimension / self.config.n_subquantizers;
        
        for (subq_idx, quantizer) in pq.iter_mut().enumerate() {
            let start_dim = subq_idx * subvector_dim;
            let end_dim = start_dim + subvector_dim;
            
            // Extract subvectors for this subquantizer
            let subvectors: Vec<Vec<f32>> = vectors.iter()
                .map(|v| v[start_dim..end_dim].to_vec())
                .collect();
            
            // Train this subquantizer
            quantizer.train(&subvectors, self.config.kmeans_iterations, rng);
        }
        
        Ok(())
    }
    
    fn find_closest_coarse_quantizer(&self, vector: &[f32]) -> Result<usize> {
        let quantizers = self.coarse_quantizers.read().unwrap();
        
        let mut best_dist = f32::INFINITY;
        let mut best_idx = 0;
        
        for (i, quantizer) in quantizers.iter().enumerate() {
            let dist = euclidean_distance(vector, &quantizer.centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        
        Ok(best_idx)
    }
    
    fn find_probe_lists(&self, query: &[f32]) -> Result<Vec<usize>> {
        let quantizers = self.coarse_quantizers.read().unwrap();
        
        let mut distances: Vec<(f32, usize)> = quantizers.iter()
            .enumerate()
            .map(|(i, q)| (euclidean_distance(query, &q.centroid), i))
            .collect();
        
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        Ok(distances.iter()
            .take(self.config.n_probe)
            .map(|(_, idx)| *idx)
            .collect())
    }
    
    fn encode_vector(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let pq = self.product_quantizers.read().unwrap();
        let subvector_dim = self.dimension / self.config.n_subquantizers;
        
        let mut codes = Vec::new();
        
        for (subq_idx, quantizer) in pq.iter().enumerate() {
            let start_dim = subq_idx * subvector_dim;
            let end_dim = start_dim + subvector_dim;
            let subvector = &vector[start_dim..end_dim];
            
            let code = quantizer.encode(subvector);
            codes.push(code);
        }
        
        Ok(codes)
    }
    
    fn compute_distance_tables(&self, query: &[f32]) -> Result<Vec<Vec<f32>>> {
        let pq = self.product_quantizers.read().unwrap();
        let subvector_dim = self.dimension / self.config.n_subquantizers;
        
        let mut distance_tables = Vec::new();
        
        for (subq_idx, quantizer) in pq.iter().enumerate() {
            let start_dim = subq_idx * subvector_dim;
            let end_dim = start_dim + subvector_dim;
            let query_subvector = &query[start_dim..end_dim];
            
            let distance_table = quantizer.compute_distance_table(query_subvector);
            distance_tables.push(distance_table);
        }
        
        Ok(distance_tables)
    }
    
    fn compute_adc_distance(&self, pq_codes: &[u8], distance_tables: &[Vec<f32>]) -> f32 {
        pq_codes.iter()
            .zip(distance_tables.iter())
            .map(|(&code, table)| table[code as usize])
            .sum()
    }
}

/// Index file header for persistence
#[derive(Debug, Serialize, Deserialize)]
struct IvfPqHeader {
    dimension: usize,
    config: IvfPqConfig,
    is_trained: bool,
    next_id: VectorId,
}

/// IVF-PQ index statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct IvfPqStats {
    pub total_vectors: usize,
    pub n_lists: usize,
    pub non_empty_lists: usize,
    pub avg_list_size: f64,
    pub compression_ratio: f64,
    pub is_trained: bool,
}

/// Helper function for Euclidean distance
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ivf_pq_basic_operations() {
        let config = IvfPqConfig {
            n_lists: 4,
            n_subquantizers: 2,
            bits_per_subquantizer: 2, // 4 centroids per subquantizer
            kmeans_iterations: 5,
            n_probe: 2,
            seed: 42,
            use_polysemous: false,
        };
        
        let index = IvfPqIndex::new(4, config).unwrap();
        
        // Training vectors
        let training_vectors = vec![
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        
        // Train index
        index.train(&training_vectors).unwrap();
        
        // Add vectors
        for (i, vector) in training_vectors.iter().enumerate() {
            let metadata = Some(serde_json::json!({"index": i}));
            index.add(vector.clone(), metadata).unwrap();
        }
        
        // Search
        let results = index.search(&[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        
        // Stats
        let stats = index.stats();
        assert_eq!(stats.total_vectors, 5);
        assert!(stats.is_trained);
        assert!(stats.compression_ratio > 1.0);
    }
    
    #[test]
    fn test_ivf_pq_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let index_path = temp_dir.path().join("test_ivf_pq.idx");
        
        // Create and train index
        {
            let config = IvfPqConfig::default();
            let index = IvfPqIndex::new(8, config).unwrap();
            
            let training_vectors: Vec<Vec<f32>> = (0..100)
                .map(|i| (0..8).map(|j| (i + j) as f32 * 0.1).collect())
                .collect();
            
            index.train(&training_vectors).unwrap();
            
            for vector in &training_vectors[..10] {
                index.add(vector.clone(), None).unwrap();
            }
            
            index.save_to_file(&index_path).unwrap();
        }
        
        // Load and verify
        {
            let loaded_index = IvfPqIndex::load_from_file(&index_path).unwrap();
            let stats = loaded_index.stats();
            assert_eq!(stats.total_vectors, 10);
            assert!(stats.is_trained);
        }
    }
    
    #[test]
    fn test_product_quantizer() {
        let mut pq = ProductQuantizer::new(2, 4);
        
        let subvectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        pq.train(&subvectors, 10, &mut rng);
        
        // Test encoding/decoding
        let code = pq.encode(&[1.0, 0.0]);
        let decoded = pq.decode(code);
        assert_eq!(decoded.len(), 2);
        
        // Test distance table
        let distances = pq.compute_distance_table(&[0.5, 0.5]);
        assert_eq!(distances.len(), 4);
    }
}
