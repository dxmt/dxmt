// LSH (Locality Sensitive Hashing) implementation
// Supports both standard LSH and Multi-Probe LSH for improved recall

use crate::types::vector::DenseVector;
use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// LSH configuration
#[derive(Debug, Clone)]
pub struct LSHConfig {
    /// Number of hash tables
    pub num_tables: usize,
    /// Number of hash functions per table
    pub num_hash_functions: usize,
    /// Number of probes for multi-probe LSH
    pub num_probes: usize,
    /// Bucket width for hash functions
    pub bucket_width: f32,
    /// Use multi-probe LSH
    pub use_multi_probe: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for LSHConfig {
    fn default() -> Self {
        Self {
            num_tables: 10,
            num_hash_functions: 12,
            num_probes: 3,
            bucket_width: 1.0,
            use_multi_probe: true,
            seed: None,
        }
    }
}

/// One LSH hash table: hash value -> ids of the vectors that landed in that bucket.
type HashBucket = Arc<RwLock<HashMap<u64, Vec<usize>>>>;

/// LSH index for approximate nearest neighbor search
pub struct LSHIndex {
    /// Configuration
    config: LSHConfig,

    /// Dimension of vectors
    #[allow(dead_code)]
    dimension: usize,

    /// Hash tables: table_id -> (hash_value -> vector_ids)
    hash_tables: Vec<HashBucket>,

    /// Random projections for each table
    projections: Vec<Array2<f32>>,

    /// Random offsets for each table
    offsets: Vec<Array1<f32>>,

    /// Stored vectors
    vectors: Arc<RwLock<Vec<DenseVector>>>,

    /// Vector IDs
    ids: Arc<RwLock<Vec<usize>>>,

    /// Number of indexed vectors
    num_vectors: usize,
}

impl LSHIndex {
    /// Create new LSH index
    pub fn new(dimension: usize, config: LSHConfig) -> Result<Self> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        // Generate random projections
        let projections = Self::generate_projections(
            &mut rng,
            dimension,
            config.num_tables,
            config.num_hash_functions,
        )?;

        // Generate random offsets
        let offsets = Self::generate_offsets(
            &mut rng,
            config.num_tables,
            config.num_hash_functions,
            config.bucket_width,
        )?;

        // Initialize hash tables
        let hash_tables = (0..config.num_tables)
            .map(|_| Arc::new(RwLock::new(HashMap::new())))
            .collect();

        Ok(Self {
            config,
            dimension,
            hash_tables,
            projections,
            offsets,
            vectors: Arc::new(RwLock::new(Vec::new())),
            ids: Arc::new(RwLock::new(Vec::new())),
            num_vectors: 0,
        })
    }

    /// Generate random projections using normal distribution
    fn generate_projections(
        rng: &mut StdRng,
        dimension: usize,
        num_tables: usize,
        num_hash_functions: usize,
    ) -> Result<Vec<Array2<f32>>> {
        let normal = Normal::new(0.0, 1.0)?;
        let mut projections = Vec::with_capacity(num_tables);

        for _ in 0..num_tables {
            let mut projection = Array2::zeros((num_hash_functions, dimension));

            for i in 0..num_hash_functions {
                for j in 0..dimension {
                    projection[[i, j]] = normal.sample(rng) as f32;
                }
            }

            projections.push(projection);
        }

        Ok(projections)
    }

    /// Generate random offsets
    fn generate_offsets(
        rng: &mut StdRng,
        num_tables: usize,
        num_hash_functions: usize,
        bucket_width: f32,
    ) -> Result<Vec<Array1<f32>>> {
        let mut offsets = Vec::with_capacity(num_tables);

        for _ in 0..num_tables {
            let mut offset = Array1::zeros(num_hash_functions);

            for i in 0..num_hash_functions {
                offset[i] = rng.gen::<f32>() * bucket_width;
            }

            offsets.push(offset);
        }

        Ok(offsets)
    }

    /// Build index from vectors
    pub fn build(&mut self, vectors: &[DenseVector]) -> Result<()> {
        // Clear existing data
        self.clear();

        // Build hash tables in parallel
        let hash_entries: Vec<Vec<(usize, u64, usize)>> = vectors
            .par_iter()
            .enumerate()
            .map(|(vec_id, vector)| {
                let mut entries = Vec::with_capacity(self.config.num_tables);

                for table_id in 0..self.config.num_tables {
                    let hash_value = self.compute_hash(vector, table_id);
                    entries.push((table_id, hash_value, vec_id));
                }

                entries
            })
            .collect();

        // Update hash tables
        for entries in hash_entries {
            for (table_id, hash_value, vec_id) in entries {
                let mut table = self.hash_tables[table_id].write().unwrap();
                table.entry(hash_value).or_default().push(vec_id);
            }
        }

        // Store vectors
        let mut stored_vectors = self.vectors.write().unwrap();
        let mut stored_ids = self.ids.write().unwrap();

        for (i, vec) in vectors.iter().enumerate() {
            stored_vectors.push(vec.clone());
            stored_ids.push(i);
        }

        self.num_vectors = vectors.len();

        Ok(())
    }

    /// Add a single vector to the index
    pub fn add(&mut self, vector: DenseVector) -> Result<usize> {
        let vec_id = self.num_vectors;

        // Compute hashes and update tables
        for table_id in 0..self.config.num_tables {
            let hash_value = self.compute_hash(&vector, table_id);

            let mut table = self.hash_tables[table_id].write().unwrap();
            table.entry(hash_value).or_default().push(vec_id);
        }

        // Store vector
        let mut stored_vectors = self.vectors.write().unwrap();
        let mut stored_ids = self.ids.write().unwrap();

        stored_vectors.push(vector);
        stored_ids.push(vec_id);

        self.num_vectors += 1;

        Ok(vec_id)
    }

    /// Compute hash value for a vector in a specific table
    fn compute_hash(&self, vector: &DenseVector, table_id: usize) -> u64 {
        let projection = &self.projections[table_id];
        let offset = &self.offsets[table_id];
        let vec_array = Array1::from_vec(vector.clone());

        // Project vector
        let projected = projection.dot(&vec_array);

        // Apply hash function
        let mut hash_value = 0u64;

        for i in 0..self.config.num_hash_functions {
            let hash_component = ((projected[i] + offset[i]) / self.config.bucket_width) as i32;
            hash_value = hash_value
                .wrapping_mul(31)
                .wrapping_add(hash_component as u64);
        }

        hash_value
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &DenseVector, k: usize) -> Result<Vec<(usize, f32)>> {
        if self.config.use_multi_probe {
            self.multi_probe_search(query, k)
        } else {
            self.standard_search(query, k)
        }
    }

    /// Standard LSH search
    fn standard_search(&self, query: &DenseVector, k: usize) -> Result<Vec<(usize, f32)>> {
        let mut candidates = HashSet::new();

        // Query each table
        for table_id in 0..self.config.num_tables {
            let hash_value = self.compute_hash(query, table_id);

            let table = self.hash_tables[table_id].read().unwrap();
            if let Some(bucket) = table.get(&hash_value) {
                candidates.extend(bucket.iter().cloned());
            }
        }

        // Compute exact distances for candidates
        self.rerank_candidates(query, candidates, k)
    }

    /// Multi-probe LSH search
    fn multi_probe_search(&self, query: &DenseVector, k: usize) -> Result<Vec<(usize, f32)>> {
        let mut candidates = HashSet::new();

        // Query each table with multiple probes
        for table_id in 0..self.config.num_tables {
            let probes = self.generate_probes(query, table_id);

            let table = self.hash_tables[table_id].read().unwrap();
            for probe_hash in probes.iter().take(self.config.num_probes) {
                if let Some(bucket) = table.get(probe_hash) {
                    candidates.extend(bucket.iter().cloned());
                }
            }
        }

        // Compute exact distances for candidates
        self.rerank_candidates(query, candidates, k)
    }

    /// Generate probe sequence for multi-probe LSH
    fn generate_probes(&self, query: &DenseVector, table_id: usize) -> Vec<u64> {
        let projection = &self.projections[table_id];
        let offset = &self.offsets[table_id];
        let vec_array = Array1::from_vec(query.clone());

        // Project query
        let projected = projection.dot(&vec_array);

        // Generate perturbations
        let mut probes = Vec::new();
        let base_hash = self.compute_hash(query, table_id);
        probes.push(base_hash);

        // Generate neighboring hashes by perturbing hash components
        for delta in &[-1i32, 1i32] {
            for i in 0..self.config.num_hash_functions {
                let mut perturbed = projected.clone();
                perturbed[i] += *delta as f32 * self.config.bucket_width;

                // Compute hash for perturbed projection
                let mut hash_value = 0u64;
                for j in 0..self.config.num_hash_functions {
                    let hash_component =
                        ((perturbed[j] + offset[j]) / self.config.bucket_width) as i32;
                    hash_value = hash_value
                        .wrapping_mul(31)
                        .wrapping_add(hash_component as u64);
                }

                probes.push(hash_value);
            }
        }

        // Sort by distance from original hash
        probes.sort_by_key(|&h| (h as i64 - base_hash as i64).abs());
        probes
    }

    /// Rerank candidates by exact distance
    fn rerank_candidates(
        &self,
        query: &DenseVector,
        candidates: HashSet<usize>,
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let stored_vectors = self.vectors.read().unwrap();
        let stored_ids = self.ids.read().unwrap();

        let mut distances: Vec<(usize, f32)> = candidates
            .into_iter()
            .filter_map(|idx| {
                if idx < stored_vectors.len() {
                    let dist = Self::compute_l2_distance(query, &stored_vectors[idx]);
                    Some((stored_ids[idx], dist))
                } else {
                    None
                }
            })
            .collect();

        distances.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        Ok(distances)
    }

    /// Compute L2 distance
    #[inline]
    fn compute_l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Clear the index
    pub fn clear(&mut self) {
        for table in &self.hash_tables {
            table.write().unwrap().clear();
        }
        self.vectors.write().unwrap().clear();
        self.ids.write().unwrap().clear();
        self.num_vectors = 0;
    }

    /// Get statistics
    pub fn get_stats(&self) -> LSHStats {
        let mut total_buckets = 0;
        let mut total_entries = 0;
        let mut max_bucket_size = 0;

        for table in &self.hash_tables {
            let table_read = table.read().unwrap();
            total_buckets += table_read.len();

            for bucket in table_read.values() {
                total_entries += bucket.len();
                max_bucket_size = max_bucket_size.max(bucket.len());
            }
        }

        LSHStats {
            num_vectors: self.num_vectors,
            num_tables: self.config.num_tables,
            total_buckets,
            total_entries,
            avg_bucket_size: if total_buckets > 0 {
                total_entries as f32 / total_buckets as f32
            } else {
                0.0
            },
            max_bucket_size,
        }
    }
}

/// LSH statistics
#[derive(Debug, Clone)]
pub struct LSHStats {
    pub num_vectors: usize,
    pub num_tables: usize,
    pub total_buckets: usize,
    pub total_entries: usize,
    pub avg_bucket_size: f32,
    pub max_bucket_size: usize,
}

/// Binary LSH for Hamming distance
pub struct BinaryLSH {
    #[allow(dead_code)]
    config: LSHConfig,
    #[allow(dead_code)]
    dimension: usize,
    hash_tables: Vec<HashMap<u64, Vec<usize>>>,
    hash_functions: Vec<Vec<usize>>,
    binary_vectors: Vec<Vec<u8>>,
    ids: Vec<usize>,
}

impl BinaryLSH {
    /// Create new binary LSH
    pub fn new(dimension: usize, config: LSHConfig) -> Result<Self> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        // Generate random bit sampling positions
        let mut hash_functions = Vec::with_capacity(config.num_tables);

        for _ in 0..config.num_tables {
            let mut positions: Vec<usize> = (0..dimension).collect();
            positions.shuffle(&mut rng);
            positions.truncate(config.num_hash_functions);
            hash_functions.push(positions);
        }

        let num_tables = config.num_tables;
        Ok(Self {
            config,
            dimension,
            hash_tables: vec![HashMap::new(); num_tables],
            hash_functions,
            binary_vectors: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// Build index from binary vectors
    pub fn build(&mut self, vectors: &[Vec<u8>]) -> Result<()> {
        self.binary_vectors = vectors.to_vec();
        self.ids = (0..vectors.len()).collect();

        for (vec_id, vector) in vectors.iter().enumerate() {
            for (table_id, positions) in self.hash_functions.iter().enumerate() {
                let hash_value = self.compute_binary_hash(vector, positions);
                self.hash_tables[table_id]
                    .entry(hash_value)
                    .or_default()
                    .push(vec_id);
            }
        }

        Ok(())
    }

    /// Compute hash for binary vector
    fn compute_binary_hash(&self, vector: &[u8], positions: &[usize]) -> u64 {
        let mut hash = 0u64;

        for (i, &pos) in positions.iter().enumerate() {
            if i >= 64 {
                break; // Only use first 64 bits for hash
            }

            let byte_idx = pos / 8;
            let bit_idx = pos % 8;

            if byte_idx < vector.len() {
                let bit = (vector[byte_idx] >> bit_idx) & 1;
                hash |= (bit as u64) << i;
            }
        }

        hash
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[u8], k: usize) -> Result<Vec<(usize, u32)>> {
        let mut candidates = HashSet::new();

        // Query each table
        for (table_id, positions) in self.hash_functions.iter().enumerate() {
            let hash_value = self.compute_binary_hash(query, positions);

            if let Some(bucket) = self.hash_tables[table_id].get(&hash_value) {
                candidates.extend(bucket.iter().cloned());
            }
        }

        // Compute exact Hamming distances
        let mut distances: Vec<(usize, u32)> = candidates
            .into_iter()
            .map(|idx| {
                let dist = Self::hamming_distance(query, &self.binary_vectors[idx]);
                (self.ids[idx], dist)
            })
            .collect();

        distances.sort_unstable_by_key(|&(_, d)| d);
        distances.truncate(k);

        Ok(distances)
    }

    /// Compute Hamming distance
    fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }
}

// #[cfg(test)]
// #[path = "lsh_tests.rs"]
// mod tests;
