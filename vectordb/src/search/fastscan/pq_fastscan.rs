// PQFastScan - Fast Product Quantization with SIMD optimization.
// Codes are stored transposed within each block so that one SIMD lane maps to one
// candidate, letting a single shuffle resolve a sub-quantizer for the whole block.

use super::{AlignedVec, FastScanIndex, FASTSCAN_BLOCK_SIZE};
use crate::search::quantization::product_quantizer::ProductQuantizer;
use crate::types::vector::DenseVector;
use anyhow::Result;
use rayon::prelude::*;


/// Configuration for PQFastScan
#[derive(Debug, Clone)]
pub struct PQFastScanConfig {
    /// Number of subquantizers
    pub num_subquantizers: usize,
    /// Number of bits per subquantizer (4 or 8)
    pub bits_per_subquantizer: usize,
    /// Block size for batching
    pub block_size: usize,
    /// Use AVX512 if available
    pub use_avx512: bool,
    /// Number of threads for parallel processing
    pub num_threads: usize,
}

impl Default for PQFastScanConfig {
    fn default() -> Self {
        Self {
            num_subquantizers: 8,
            bits_per_subquantizer: 4,
            block_size: FASTSCAN_BLOCK_SIZE,
            use_avx512: false,
            num_threads: num_cpus::get(),
        }
    }
}

/// PQFastScan index with 4-bit packed codes
pub struct PQFastScan {
    /// Product quantizer
    pq: ProductQuantizer,

    /// Packed codes (4-bit codes packed into u64)
    /// Layout: [block_id][subquantizer][packed_codes]
    packed_codes: Vec<u64>,

    /// Original vector IDs
    ids: Vec<usize>,

    /// Number of vectors
    num_vectors: usize,

    /// Configuration
    config: PQFastScanConfig,

    /// Dimension of original vectors
    dimension: usize,

    /// Precomputed lookup tables cache
    #[allow(dead_code)]
    lookup_cache: Option<Vec<AlignedVec<f32>>>,
}

impl PQFastScan {
    /// Create new PQFastScan index
    pub fn new(dimension: usize, config: PQFastScanConfig) -> Result<Self> {
        let pq = ProductQuantizer::new(
            dimension,
            config.num_subquantizers,
            config.bits_per_subquantizer,
        )?;

        Ok(Self {
            pq,
            packed_codes: Vec::new(),
            ids: Vec::new(),
            num_vectors: 0,
            config,
            dimension,
            lookup_cache: None,
        })
    }

    /// Train the quantizer
    pub fn train(&mut self, training_vectors: &[DenseVector]) -> Result<()> {
        self.pq.train(training_vectors)?;
        Ok(())
    }

    /// Add vectors to the index
    pub fn add(&mut self, vectors: &[DenseVector]) -> Result<()> {
        if !self.pq.is_trained() {
            anyhow::bail!("PQ must be trained before adding vectors");
        }

        let start_id = self.num_vectors;

        // Encode vectors with PQ
        let codes = self.pq.encode_batch(vectors)?;

        // Pack codes into blocks
        let packed = self.pack_codes(&codes);
        self.packed_codes.extend(packed);

        // Store IDs
        for i in 0..vectors.len() {
            self.ids.push(start_id + i);
        }

        self.num_vectors += vectors.len();
        Ok(())
    }

    /// Pack 4-bit codes into u64 blocks
    fn pack_codes(&self, codes: &[Vec<u8>]) -> Vec<u64> {
        let num_blocks = codes.len().div_ceil(self.config.block_size);
        let mut packed = Vec::with_capacity(num_blocks * self.config.num_subquantizers);

        for block_idx in 0..num_blocks {
            let block_start = block_idx * self.config.block_size;
            let block_end = ((block_idx + 1) * self.config.block_size).min(codes.len());

            for subq in 0..self.config.num_subquantizers {
                let packed_val = codes[block_start..block_end]
                    .iter()
                    .enumerate()
                    .fold(0u64, |acc, (i, vec_codes)| {
                        let code = vec_codes[subq] as u64;
                        acc | (code & 0xF) << (i * 4)
                    });
                packed.push(packed_val);
            }
        }

        packed
    }

    /// Compute lookup tables for a query
    fn compute_lookup_table(&self, query: &DenseVector) -> AlignedVec<f32> {
        let mut lut = AlignedVec::new(
            self.config.num_subquantizers * (1 << self.config.bits_per_subquantizer),
            0.0f32,
        );

        let subvector_dim = self.dimension / self.config.num_subquantizers;

        for subq in 0..self.config.num_subquantizers {
            let start = subq * subvector_dim;
            let end = (subq + 1) * subvector_dim;
            let subvector = &query[start..end];

            for code in 0..(1 << self.config.bits_per_subquantizer) {
                let centroid = self.pq.get_centroid(subq, code);
                let dist = Self::compute_l2_distance(subvector, centroid);
                lut.as_mut_slice()[subq * (1 << self.config.bits_per_subquantizer) + code] = dist;
            }
        }

        lut
    }

    /// Compute L2 distance between two vectors
    #[inline]
    fn compute_l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    /// Search with SIMD optimization
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn search_simd(&self, query_lut: &AlignedVec<f32>, k: usize) -> Vec<(usize, f32)> {
        let num_blocks = self.num_vectors.div_ceil(self.config.block_size);
        let mut distances = vec![0.0f32; self.num_vectors];

        // Process each block
        for block_idx in 0..num_blocks {
            let block_start = block_idx * self.config.block_size;
            let block_end = ((block_idx + 1) * self.config.block_size).min(self.num_vectors);
            let block_size = block_end - block_start;

            // Compute distances for this block
            for subq in 0..self.config.num_subquantizers {
                let packed_code = self.packed_codes[block_idx * self.config.num_subquantizers + subq];
                let lut_offset = subq * (1 << self.config.bits_per_subquantizer);

                // Unpack and accumulate distances
                for i in 0..block_size {
                    let code = ((packed_code >> (i * 4)) & 0xF) as usize;
                    distances[block_start + i] += query_lut.as_slice()[lut_offset + code];
                }
            }
        }

        // Find k nearest
        self.find_k_nearest(&distances, k)
    }

    /// Find k nearest neighbors from distances
    fn find_k_nearest(&self, distances: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut indexed_distances: Vec<(usize, f32)> = distances
            .iter()
            .enumerate()
            .map(|(i, &d)| (self.ids[i], d))
            .collect();

        indexed_distances.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        indexed_distances.truncate(k);
        indexed_distances
    }
}

impl FastScanIndex for PQFastScan {
    fn search_batch(
        &self,
        queries: &[DenseVector],
        k: usize,
        num_threads: Option<usize>,
    ) -> Vec<Vec<(usize, f32)>> {
        let num_threads = num_threads.unwrap_or(self.config.num_threads);

        // Parallel search
        queries
            .par_chunks(queries.len() / num_threads + 1)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|query| {
                        let lut = self.compute_lookup_table(query);
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            self.search_simd(&lut, k)
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        self.search_scalar(&lut, k)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn precompute_query_tables(&self, queries: &[DenseVector]) -> Vec<AlignedVec<f32>> {
        queries
            .par_iter()
            .map(|q| self.compute_lookup_table(q))
            .collect()
    }

    fn optimal_batch_size(&self) -> usize {
        self.config.block_size
    }
}

// Scalar fallback for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
impl PQFastScan {
    fn search_scalar(&self, query_lut: &AlignedVec<f32>, k: usize) -> Vec<(usize, f32)> {
        let num_blocks = self.num_vectors.div_ceil(self.config.block_size);
        let mut distances = vec![0.0f32; self.num_vectors];

        for block_idx in 0..num_blocks {
            let block_start = block_idx * self.config.block_size;
            let block_end = ((block_idx + 1) * self.config.block_size).min(self.num_vectors);
            let block_size = block_end - block_start;

            for subq in 0..self.config.num_subquantizers {
                let packed_code = self.packed_codes[block_idx * self.config.num_subquantizers + subq];
                let lut_offset = subq * (1 << self.config.bits_per_subquantizer);

                for i in 0..block_size {
                    let code = ((packed_code >> (i * 4)) & 0xF) as usize;
                    distances[block_start + i] += query_lut.as_slice()[lut_offset + code];
                }
            }
        }

        self.find_k_nearest(&distances, k)
    }
}