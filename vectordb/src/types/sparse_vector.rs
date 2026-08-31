use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sparse vector representation using indices and values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    /// Non-zero indices
    pub indices: Vec<u32>,
    /// Corresponding values
    pub values: Vec<f32>,
    /// Dimension of the full vector
    pub dimension: usize,
}

impl SparseVector {
    /// Maximum number of non-zero elements in a flattened multivector.
    ///
    /// A cap is needed because the length is caller-supplied and allocation follows it; 10M
    /// is far above any legitimate sparse vector and still bounds a hostile request.
    pub const MAX_MULTIVECTOR_FLATTENED_LEN: usize = 10_000_000;

    /// Create a new sparse vector with validation
    pub fn new(indices: Vec<u32>, values: Vec<f32>, dimension: usize) -> crate::Result<Self> {
        let sv = Self {
            indices,
            values,
            dimension,
        };
        sv.validate()?;
        Ok(sv)
    }

    /// Create a new sparse vector without validation (for internal use)
    pub fn new_unchecked(indices: Vec<u32>, values: Vec<f32>, dimension: usize) -> Self {
        Self {
            indices,
            values,
            dimension,
        }
    }

    /// Full structural validation: matching lengths, sorted unique indices, finite values,
    /// and indices inside the declared dimension.
    pub fn validate(&self) -> crate::Result<()> {
        // 1. Length matching (indices.len == values.len)
        if self.indices.len() != self.values.len() {
            return Err(crate::Error::InvalidVector(format!(
                "Sparse vector indices length ({}) must match values length ({})",
                self.indices.len(),
                self.values.len()
            )));
        }

        // 2. Empty check
        if self.indices.is_empty() {
            return Err(crate::Error::InvalidVector(
                "Sparse vector cannot be empty (must have at least one non-zero element)"
                    .to_string(),
            ));
        }

        // 3. Max size check (< 10M)
        if self.indices.len() > Self::MAX_MULTIVECTOR_FLATTENED_LEN {
            return Err(crate::Error::InvalidVector(format!(
                "Sparse vector has {} elements, exceeding maximum of {}",
                self.indices.len(),
                Self::MAX_MULTIVECTOR_FLATTENED_LEN
            )));
        }

        // 4. Sorted indices validation (strictly ascending)
        for i in 1..self.indices.len() {
            if self.indices[i] <= self.indices[i - 1] {
                return Err(crate::Error::InvalidVector(format!(
                    "Sparse vector indices must be strictly ascending (found {} <= {} at position {}). \
                     Use sort_indices() to fix.",
                    self.indices[i], self.indices[i - 1], i
                )));
            }
        }

        // 5. Index bounds check (< dimension)
        for (pos, &idx) in self.indices.iter().enumerate() {
            if (idx as usize) >= self.dimension {
                return Err(crate::Error::InvalidVector(format!(
                    "Index {} at position {} is out of bounds for dimension {}",
                    idx, pos, self.dimension
                )));
            }
        }

        // 6. Value validation (no NaN/Inf, optionally non-negative)
        for (pos, &val) in self.values.iter().enumerate() {
            if val.is_nan() {
                return Err(crate::Error::InvalidVector(format!(
                    "NaN value at position {}",
                    pos
                )));
            }
            if val.is_infinite() {
                return Err(crate::Error::InvalidVector(format!(
                    "Infinite value at position {}: {}",
                    pos, val
                )));
            }
            // Note: We don't enforce non-negative values as some algorithms
            // (e.g., BM25) can produce negative weights
        }

        Ok(())
    }

    /// Sort indices and values to ensure indices are in ascending order
    /// Maintains the pairing between indices and values
    pub fn sort_indices(&mut self) {
        // Create pairs of (index, value)
        let mut pairs: Vec<(u32, f32)> = self
            .indices
            .iter()
            .zip(self.values.iter())
            .map(|(&i, &v)| (i, v))
            .collect();

        // Sort by index
        pairs.sort_unstable_by_key(|&(idx, _)| idx);

        // Rebuild indices and values
        self.indices.clear();
        self.values.clear();

        for (idx, val) in pairs {
            self.indices.push(idx);
            self.values.push(val);
        }
    }

    /// Create from dense vector
    /// Uses new_unchecked since we know indices are sorted and valid
    pub fn from_dense(dense: &[f32], threshold: f32) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in dense.iter().enumerate() {
            if val.abs() > threshold {
                indices.push(i as u32);
                values.push(val);
            }
        }

        // Use new_unchecked since we know indices are sorted (0, 1, 2, ...)
        // and all values are valid
        Self::new_unchecked(indices, values, dense.len())
    }

    /// Convert to dense vector
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.dimension];
        for (idx, val) in self.indices.iter().zip(self.values.iter()) {
            dense[*idx as usize] = *val
        }
        dense
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Compute dot product with another sparse vector
    pub fn dot(&self, other: &SparseVector) -> f32 {
        assert_eq!(
            self.dimension, other.dimension,
            "Vectors must have same dimension"
        );

        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            if self.indices[i] == other.indices[j] {
                result += self.values[i] * other.values[j];
                i += 1;
                j += 1
            } else if self.indices[i] < other.indices[j] {
                i += 1
            } else {
                j += 1
            }
        }

        result
    }

    /// Compute cosine similarity
    pub fn cosine_similarity(&self, other: &SparseVector) -> f32 {
        let dot = self.dot(other);
        let norm1 = self.l2_norm();
        let norm2 = other.l2_norm();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot / (norm1 * norm2)
        }
    }

    /// Compute L2 norm
    pub fn l2_norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Scale vector by scalar
    pub fn scale(&mut self, scalar: f32) {
        for val in &mut self.values {
            *val *= scalar
        }
    }

    /// Add another sparse vector
    pub fn add(&mut self, other: &SparseVector) -> crate::Result<()> {
        if self.dimension != other.dimension {
            return Err(crate::Error::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        let mut result_map: HashMap<u32, f32> = HashMap::new();

        // Add self values
        for (idx, val) in self.indices.iter().zip(self.values.iter()) {
            result_map.insert(*idx, *val);
        }

        // Add other values
        for (idx, val) in other.indices.iter().zip(other.values.iter()) {
            *result_map.entry(*idx).or_insert(0.0) += val;
        }

        // Convert back to sparse format
        let mut indices: Vec<u32> = result_map.keys().copied().collect();
        indices.sort_unstable();

        self.indices = indices;
        self.values = self.indices.iter().map(|idx| result_map[idx]).collect();

        Ok(())
    }
}

/// Hybrid vector combining dense and sparse representations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridVector {
    /// Dense vector component
    pub dense: Vec<f32>,
    /// Sparse vector component
    pub sparse: SparseVector,
}

impl HybridVector {
    /// Create new hybrid vector
    pub fn new(dense: Vec<f32>, sparse: SparseVector) -> Self {
        Self { dense, sparse }
    }

    /// Compute hybrid similarity score
    pub fn hybrid_similarity(
        &self,
        other: &HybridVector,
        dense_weight: f32,
        sparse_weight: f32,
    ) -> f32 {
        let dense_sim = cosine_similarity(&self.dense, &other.dense);
        let sparse_sim = self.sparse.cosine_similarity(&other.sparse);

        dense_weight * dense_sim + sparse_weight * sparse_sim
    }
}

/// Compute cosine similarity between dense vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_creation() {
        let indices = vec![0, 2, 5];
        let values = vec![1.0, 2.0, 3.0];
        let sparse = SparseVector::new(indices.clone(), values.clone(), 10).unwrap();

        assert_eq!(sparse.indices, indices);
        assert_eq!(sparse.values, values);
        assert_eq!(sparse.dimension, 10);
        assert_eq!(sparse.nnz(), 3);
    }

    #[test]
    fn test_sparse_vector_validation() {
        // Valid sparse vector
        assert!(SparseVector::new(vec![0, 2, 5], vec![1.0, 2.0, 3.0], 10).is_ok());

        // Length mismatch
        assert!(SparseVector::new(vec![0, 2], vec![1.0, 2.0, 3.0], 10).is_err());

        // Empty vector
        assert!(SparseVector::new(vec![], vec![], 10).is_err());

        // Unsorted indices
        assert!(SparseVector::new(vec![5, 2, 0], vec![1.0, 2.0, 3.0], 10).is_err());

        // Duplicate indices
        assert!(SparseVector::new(vec![0, 2, 2], vec![1.0, 2.0, 3.0], 10).is_err());

        // Out of bounds index
        assert!(SparseVector::new(vec![0, 2, 15], vec![1.0, 2.0, 3.0], 10).is_err());

        // NaN value
        assert!(SparseVector::new(vec![0, 2, 5], vec![1.0, f32::NAN, 3.0], 10).is_err());

        // Infinite value
        assert!(SparseVector::new(vec![0, 2, 5], vec![1.0, f32::INFINITY, 3.0], 10).is_err());
    }

    #[test]
    fn test_sparse_sort_indices() {
        let mut sparse = SparseVector::new_unchecked(vec![5, 2, 0], vec![3.0, 2.0, 1.0], 10);
        sparse.sort_indices();

        assert_eq!(sparse.indices, vec![0, 2, 5]);
        assert_eq!(sparse.values, vec![1.0, 2.0, 3.0]);

        // Now validation should pass
        assert!(sparse.validate().is_ok());
    }

    #[test]
    fn test_sparse_from_dense() {
        let dense = vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0];
        let sparse = SparseVector::from_dense(&dense, 0.5);

        assert_eq!(sparse.indices, vec![1, 3, 6]);
        assert_eq!(sparse.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sparse_to_dense() {
        let sparse = SparseVector::new(vec![1, 3, 6], vec![1.0, 2.0, 3.0], 8).unwrap();
        let dense = sparse.to_dense();

        assert_eq!(dense, vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_sparse_dot_product() {
        let sparse1 = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 6).unwrap();
        let sparse2 = SparseVector::new(vec![1, 2, 4], vec![4.0, 5.0, 6.0], 6).unwrap();

        let dot = sparse1.dot(&sparse2);
        assert_eq!(dot, 2.0 * 5.0 + 3.0 * 6.0); // 10 + 18 = 28
    }
}
