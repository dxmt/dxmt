// Lookup table management for FastScan

use super::AlignedVec;
use crate::types::vector::DenseVector;
use std::sync::Arc;

/// Builder for lookup tables
pub struct LookupTableBuilder {
    dimension: usize,
    num_subquantizers: usize,
    centroid_count: usize,
}

impl LookupTableBuilder {
    pub fn new(dimension: usize, num_subquantizers: usize, centroid_count: usize) -> Self {
        Self {
            dimension,
            num_subquantizers,
            centroid_count,
        }
    }

    /// Build lookup table for a query vector
    pub fn build_table(
        &self,
        query: &DenseVector,
        centroids: &[Vec<DenseVector>],
    ) -> AlignedVec<f32> {
        let subvector_dim = self.dimension / self.num_subquantizers;
        let table_size = self.num_subquantizers * self.centroid_count;
        let mut table = AlignedVec::new(table_size, 0.0f32);

        for (subq, centroid_group) in centroids.iter().enumerate() {
            let start = subq * subvector_dim;
            let end = (subq + 1) * subvector_dim;
            let subquery = &query[start..end];

            for (centroid_id, centroid) in centroid_group.iter().enumerate() {
                let distance = Self::compute_distance(subquery, centroid);
                table.as_mut_slice()[subq * self.centroid_count + centroid_id] = distance;
            }
        }

        table
    }

    /// Build multiple tables in parallel
    pub fn build_tables_batch(
        &self,
        queries: &[DenseVector],
        centroids: &[Vec<DenseVector>],
    ) -> Vec<AlignedVec<f32>> {
        use rayon::prelude::*;

        queries
            .par_iter()
            .map(|query| self.build_table(query, centroids))
            .collect()
    }

    /// Compute L2 distance
    #[inline]
    fn compute_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }
}

/// Cache for lookup tables
pub struct LookupTableCache {
    cache: lru::LruCache<u64, Arc<AlignedVec<f32>>>,
}

impl LookupTableCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: lru::LruCache::new(capacity.try_into().unwrap()),
        }
    }

    pub fn get(&mut self, key: u64) -> Option<Arc<AlignedVec<f32>>> {
        self.cache.get(&key).cloned()
    }

    pub fn insert(&mut self, key: u64, table: AlignedVec<f32>) {
        self.cache.put(key, Arc::new(table));
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
