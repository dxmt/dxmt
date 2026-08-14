// FastScan implementation inspired by FAISS
// Provides SIMD-optimized batch distance computation for Product Quantization

use crate::types::vector::DenseVector;

pub mod pq_fastscan;
// pub mod simd_utils; // Removed - redundant
pub mod lookup_tables;

// #[cfg(test)]
// mod tests;

pub use pq_fastscan::{PQFastScan, PQFastScanConfig};
pub use lookup_tables::LookupTableBuilder;

/// Aligned vector for SIMD operations
#[repr(align(32))]
pub struct AlignedVec<T> {
    data: Vec<T>,
}

impl<T: Clone> AlignedVec<T> {
    pub fn new(size: usize, default: T) -> Self {
        let mut data = Vec::with_capacity(size);
        data.resize(size, default);
        Self { data }
    }

    pub fn from_vec(vec: Vec<T>) -> Self {
        Self { data: vec }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// FastScan trait for batch operations
pub trait FastScanIndex: Send + Sync {
    /// Batch search with SIMD optimization
    fn search_batch(
        &self,
        queries: &[DenseVector],
        k: usize,
        num_threads: Option<usize>,
    ) -> Vec<Vec<(usize, f32)>>;

    /// Precompute lookup tables for queries
    fn precompute_query_tables(&self, queries: &[DenseVector]) -> Vec<AlignedVec<f32>>;

    /// Get optimal batch size for this index
    fn optimal_batch_size(&self) -> usize {
        32 // Default: 32 vectors per batch
    }
}

/// Block size for FastScan (matches FAISS default)
pub const FASTSCAN_BLOCK_SIZE: usize = 32;

/// SIMD lane width for AVX2
pub const AVX2_LANE_WIDTH: usize = 8;

/// SIMD lane width for AVX512
pub const AVX512_LANE_WIDTH: usize = 16;