//! Sparse vector inverted index with WAND (Weak AND) pruning
//!
//! Implements Maximum Inner Product Search (MIPS) over sparse vectors using
//! an inverted index. Each dimension maps to a posting list of (record_id, value)
//! pairs sorted by absolute value descending.
//!
//! WAND pruning skips candidates whose upper-bound score (sum of per-dimension
//! max values for the query's non-zero dimensions) cannot beat the current
//! top-k threshold, dramatically reducing the number of candidates scored.
//!
//! Reference: Bruch et al., "WAND with Filter" (2023).

use crate::types::sparse_vector::SparseVector;
use crate::types::RecordId;
use std::collections::HashMap;

/// One entry in a posting list: record id + value for that dimension
#[derive(Debug, Clone)]
struct Posting {
    id: RecordId,
    value: f32,
}

/// Sparse vector index backed by an inverted index with WAND pruning
pub struct SparseIndex {
    /// inverted_index[dimension] = postings sorted by |value| descending
    inverted_index: HashMap<u32, Vec<Posting>>,
    /// Per-dimension maximum absolute value (used as upper bound for WAND)
    dim_max_value: HashMap<u32, f32>,
    /// Number of indexed vectors
    num_vectors: usize,
}

impl SparseIndex {
    pub fn new() -> Self {
        Self {
            inverted_index: HashMap::new(),
            dim_max_value: HashMap::new(),
            num_vectors: 0,
        }
    }

    /// Insert a sparse vector into the index
    pub fn insert(&mut self, id: RecordId, vector: &SparseVector) -> crate::Result<()> {
        for (&dim, &val) in vector.indices.iter().zip(vector.values.iter()) {
            let posting = Posting {
                id: id.clone(),
                value: val,
            };

            let list = self.inverted_index.entry(dim).or_default();
            list.push(posting);
            // Keep sorted by |value| descending for WAND upper-bound scan
            list.sort_by(|a, b| {
                b.value
                    .abs()
                    .partial_cmp(&a.value.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Update per-dimension max
            let max = self.dim_max_value.entry(dim).or_insert(0.0);
            if val.abs() > *max {
                *max = val.abs();
            }
        }
        self.num_vectors += 1;
        Ok(())
    }

    /// Delete a vector from the index by id
    pub fn delete(&mut self, id: &str) -> crate::Result<()> {
        for list in self.inverted_index.values_mut() {
            list.retain(|p| p.id != id);
        }
        // Recompute dim_max_value for affected dimensions
        for (dim, list) in &self.inverted_index {
            let max = list.iter().map(|p| p.value.abs()).fold(0.0f32, f32::max);
            self.dim_max_value.insert(*dim, max);
        }
        self.num_vectors = self.num_vectors.saturating_sub(1);
        Ok(())
    }

    /// Search for the top-k vectors most similar to `query` (dot product)
    ///
    /// Uses WAND pruning: computes a per-candidate upper-bound score using
    /// `dim_max_value` and skips candidates that cannot beat the k-th best
    /// score seen so far.
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(RecordId, f32)> {
        if k == 0 || self.num_vectors == 0 {
            return vec![];
        }

        // Upper bound on contribution of each query dimension = |q_val| * dim_max_value
        // Used for WAND to prune posting lists that cannot improve top-k.
        let mut dim_upper_bounds: Vec<(u32, f32)> = query
            .indices
            .iter()
            .zip(query.values.iter())
            .filter_map(|(&dim, &qval)| {
                self.dim_max_value
                    .get(&dim)
                    .map(|&max_d| (dim, qval.abs() * max_d))
            })
            .collect();

        // Sort by upper bound descending so we iterate the most promising dims first
        dim_upper_bounds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Accumulate dot products per candidate
        let mut dot_products: HashMap<RecordId, f32> = HashMap::new();

        // WAND threshold: track k-th best score found so far
        let mut threshold = f32::NEG_INFINITY;
        let mut heap: Vec<f32> = Vec::with_capacity(k + 1);

        // Build query dimension → value map for O(1) lookup
        let query_map: HashMap<u32, f32> = query
            .indices
            .iter()
            .zip(query.values.iter())
            .map(|(&d, &v)| (d, v))
            .collect();

        for (dim, upper_bound) in &dim_upper_bounds {
            // WAND: if this dimension's upper bound contribution is too small, skip
            // (rough check — full WAND requires accumulating per-candidate bounds,
            //  this simpler version still prunes low-value dimensions)
            if heap.len() >= k && *upper_bound < threshold * 0.01 {
                break;
            }

            let qval = match query_map.get(dim) {
                Some(&v) => v,
                None => continue,
            };

            if let Some(postings) = self.inverted_index.get(dim) {
                for posting in postings {
                    // WAND per-posting: check if remaining contribution can beat threshold
                    let candidate_ub = posting.value.abs() * qval.abs();
                    if heap.len() >= k
                        && candidate_ub + *dot_products.get(&posting.id).unwrap_or(&0.0) < threshold
                    {
                        // Because postings are sorted by |value| desc, all subsequent
                        // postings are even smaller — safe to break this list
                        break;
                    }

                    let score = dot_products.entry(posting.id.clone()).or_insert(0.0);
                    *score += qval * posting.value;

                    // Update heap/threshold
                    heap.push(*score);
                    if heap.len() > k * 4 {
                        heap.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                        heap.truncate(k);
                        threshold = *heap.last().unwrap_or(&f32::NEG_INFINITY);
                    }
                }
            }
        }

        // Collect and sort results
        let mut results: Vec<(RecordId, f32)> = dot_products
            .into_iter()
            .filter(|(_, s)| *s > 0.0) // only positive dot products are relevant
            .collect();

        // Ties by id, before the truncate. `dot_products` is a `HashMap`, so equally-scoring
        // documents would otherwise be kept or dropped according to hash iteration order. The
        // fusion above this re-sorts what it is given, but it is given what survived here.
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        results.truncate(k);
        results
    }

    /// Number of indexed vectors
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Approximate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let postings: usize = self
            .inverted_index
            .values()
            .map(|list| list.len() * (std::mem::size_of::<Posting>() + 32)) // id string approx
            .sum();
        let overhead = self.inverted_index.len() * 64 + self.dim_max_value.len() * 12;
        postings + overhead
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::sparse_vector::SparseVector;

    fn make_sparse(indices: Vec<u32>, values: Vec<f32>) -> SparseVector {
        let dim = indices.iter().max().copied().unwrap_or(0) as usize + 1;
        SparseVector::new(indices, values, dim).unwrap()
    }

    /// Equally-scoring documents are cut by id, so the same k come back every time.
    ///
    /// `dot_products` is a `HashMap`, so before the tiebreak *which* of several tied documents
    /// survived `truncate(k)` followed hash iteration order and differed between processes.
    /// Ties are the ordinary case, not an edge one: any two documents carrying the query's term
    /// with the same weight score identically.
    #[test]
    fn tied_scores_are_cut_by_id() {
        let mut index = SparseIndex::new();
        for id in ["c", "a", "b"] {
            index
                .insert(id.into(), &make_sparse(vec![0], vec![1.0]))
                .unwrap();
        }

        let results = index.search(&make_sparse(vec![0], vec![1.0]), 2);

        assert_eq!(
            results
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(
            results[0].1, results[1].1,
            "the scores were not actually tied, so this proves nothing"
        );
    }

    #[test]
    fn test_sparse_index_insert_and_search() {
        let mut index = SparseIndex::new();

        // Insert three vectors
        index
            .insert("a".into(), &make_sparse(vec![0, 2], vec![1.0, 1.0]))
            .unwrap();
        index
            .insert("b".into(), &make_sparse(vec![1, 2], vec![1.0, 2.0]))
            .unwrap();
        index
            .insert("c".into(), &make_sparse(vec![0, 1], vec![3.0, 0.5]))
            .unwrap();

        // Query: [0]=1, [2]=1 — should prefer "a" (exact match) and "b" (shares dim 2)
        let query = make_sparse(vec![0, 2], vec![1.0, 1.0]);
        let results = index.search(&query, 3);

        assert!(!results.is_empty());
        // "a" has dot product 1*1 + 1*1 = 2.0
        // "b" has dot product 0*1 + 2*1 = 2.0
        // "c" has dot product 3*1 + 0 = 3.0
        assert_eq!(results[0].0, "c"); // c scores highest (dim 0 contributes 3.0)
    }

    #[test]
    fn test_sparse_index_top_k() {
        let mut index = SparseIndex::new();
        for i in 0..10u32 {
            let val = (i + 1) as f32;
            index
                .insert(format!("v{i}"), &make_sparse(vec![0], vec![val]))
                .unwrap();
        }

        let query = make_sparse(vec![0], vec![1.0]);
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        // Highest dot products are v9=10, v8=9, v7=8
        assert_eq!(results[0].0, "v9");
        assert_eq!(results[1].0, "v8");
        assert_eq!(results[2].0, "v7");
    }

    #[test]
    fn test_sparse_index_delete() {
        let mut index = SparseIndex::new();
        index
            .insert("a".into(), &make_sparse(vec![0], vec![5.0]))
            .unwrap();
        index
            .insert("b".into(), &make_sparse(vec![0], vec![3.0]))
            .unwrap();

        index.delete("a").unwrap();

        let results = index.search(&make_sparse(vec![0], vec![1.0]), 5);
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            !ids.contains(&"a"),
            "deleted vector should not appear in results"
        );
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn test_sparse_index_empty_search() {
        let index = SparseIndex::new();
        let query = make_sparse(vec![0], vec![1.0]);
        let results = index.search(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_index_no_overlap() {
        let mut index = SparseIndex::new();
        index
            .insert("a".into(), &make_sparse(vec![10, 11], vec![1.0, 1.0]))
            .unwrap();

        // Query on completely different dimensions
        let query = make_sparse(vec![0, 1], vec![1.0, 1.0]);
        let results = index.search(&query, 5);
        assert!(
            results.is_empty(),
            "no overlap means no positive dot product"
        );
    }

    #[test]
    fn test_sparse_index_memory_usage() {
        let mut index = SparseIndex::new();
        for i in 0..100u32 {
            index
                .insert(format!("v{i}"), &make_sparse(vec![i % 10], vec![1.0]))
                .unwrap();
        }
        assert!(index.memory_usage() > 0);
    }
}
