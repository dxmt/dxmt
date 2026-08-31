// Property-based tests for vector operations
//
// Uses proptest to verify mathematical properties hold across random inputs

use proptest::prelude::*;
use dxmt_vectordb::types::{
    vector::{DistanceMetric, VectorValidator, DenseVector},
    sparse_vector::SparseVector,
};

// Arbitrary vector generator (dimension 2-128, values -1.0 to 1.0)
fn arb_dense_vector(dim: usize) -> impl Strategy<Value = DenseVector> {
    prop::collection::vec(-1.0f32..1.0f32, dim..=dim)
}

// Arbitrary normalized vector generator
fn arb_normalized_vector(dim: usize) -> impl Strategy<Value = DenseVector> {
    arb_dense_vector(dim).prop_map(|mut v| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    })
}

// Arbitrary sparse vector generator
fn arb_sparse_vector(dimension: usize, max_nnz: usize) -> impl Strategy<Value = SparseVector> {
    (1..=max_nnz.min(dimension)).prop_flat_map(move |nnz| {
        // Generate sorted unique indices
        prop::collection::btree_set(0u32..(dimension as u32), nnz..=nnz)
            .prop_flat_map(move |indices_set| {
                let indices: Vec<u32> = indices_set.into_iter().collect();
                let values_strategy = prop::collection::vec(-1.0f32..1.0f32, indices.len());
                (Just(indices), values_strategy, Just(dimension))
            })
            .prop_map(|(indices, values, dim)| {
                SparseVector::new_unchecked(indices, values, dim)
            })
    })
}

// =============================================================================
// Distance Metric Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test: Distance is symmetric: d(a, b) = d(b, a)
    #[test]
    fn test_distance_symmetry_euclidean(
        v1 in arb_dense_vector(64),
        v2 in arb_dense_vector(64),
    ) {
        let d1 = DistanceMetric::Euclidean.calculate(&v1, &v2).unwrap();
        let d2 = DistanceMetric::Euclidean.calculate(&v2, &v1).unwrap();
        prop_assert!((d1 - d2).abs() < 1e-6, "Euclidean distance not symmetric: {} vs {}", d1, d2);
    }

    #[test]
    fn test_distance_symmetry_cosine(
        v1 in arb_normalized_vector(64),
        v2 in arb_normalized_vector(64),
    ) {
        let d1 = DistanceMetric::Cosine.calculate(&v1, &v2).unwrap();
        let d2 = DistanceMetric::Cosine.calculate(&v2, &v1).unwrap();
        prop_assert!((d1 - d2).abs() < 1e-6, "Cosine distance not symmetric: {} vs {}", d1, d2);
    }

    #[test]
    fn test_distance_symmetry_manhattan(
        v1 in arb_dense_vector(64),
        v2 in arb_dense_vector(64),
    ) {
        let d1 = DistanceMetric::Manhattan.calculate(&v1, &v2).unwrap();
        let d2 = DistanceMetric::Manhattan.calculate(&v2, &v1).unwrap();
        prop_assert!((d1 - d2).abs() < 1e-6, "Manhattan distance not symmetric: {} vs {}", d1, d2);
    }

    /// Test: Distance is non-negative: d(a, b) >= 0
    #[test]
    fn test_distance_non_negative_euclidean(
        v1 in arb_dense_vector(64),
        v2 in arb_dense_vector(64),
    ) {
        let d = DistanceMetric::Euclidean.calculate(&v1, &v2).unwrap();
        prop_assert!(d >= 0.0, "Euclidean distance is negative: {}", d);
    }

    #[test]
    fn test_distance_non_negative_manhattan(
        v1 in arb_dense_vector(64),
        v2 in arb_dense_vector(64),
    ) {
        let d = DistanceMetric::Manhattan.calculate(&v1, &v2).unwrap();
        prop_assert!(d >= 0.0, "Manhattan distance is negative: {}", d);
    }

    /// Test: Identity: d(a, a) = 0
    #[test]
    fn test_distance_identity_euclidean(v in arb_dense_vector(64)) {
        let d = DistanceMetric::Euclidean.calculate(&v, &v).unwrap();
        prop_assert!(d < 1e-6, "Euclidean distance to self should be 0, got {}", d);
    }

    #[test]
    fn test_distance_identity_cosine(v in arb_normalized_vector(64)) {
        let d = DistanceMetric::Cosine.calculate(&v, &v).unwrap();
        prop_assert!(d < 1e-6, "Cosine distance to self should be 0, got {}", d);
    }

    /// Test: Triangle inequality: d(a, c) <= d(a, b) + d(b, c)
    #[test]
    fn test_triangle_inequality_euclidean(
        v1 in arb_dense_vector(32),
        v2 in arb_dense_vector(32),
        v3 in arb_dense_vector(32),
    ) {
        let d_ac = DistanceMetric::Euclidean.calculate(&v1, &v3).unwrap();
        let d_ab = DistanceMetric::Euclidean.calculate(&v1, &v2).unwrap();
        let d_bc = DistanceMetric::Euclidean.calculate(&v2, &v3).unwrap();

        prop_assert!(
            d_ac <= d_ab + d_bc + 1e-6,
            "Triangle inequality violated: {} > {} + {}",
            d_ac, d_ab, d_bc
        );
    }

    #[test]
    fn test_triangle_inequality_manhattan(
        v1 in arb_dense_vector(32),
        v2 in arb_dense_vector(32),
        v3 in arb_dense_vector(32),
    ) {
        let d_ac = DistanceMetric::Manhattan.calculate(&v1, &v3).unwrap();
        let d_ab = DistanceMetric::Manhattan.calculate(&v1, &v2).unwrap();
        let d_bc = DistanceMetric::Manhattan.calculate(&v2, &v3).unwrap();

        prop_assert!(
            d_ac <= d_ab + d_bc + 1e-6,
            "Triangle inequality violated: {} > {} + {}",
            d_ac, d_ab, d_bc
        );
    }
}

// =============================================================================
// Vector Normalization Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test: Normalized vector has L2 norm = 1.0
    #[test]
    fn test_normalization_produces_unit_vector(v in arb_dense_vector(64)) {
        // Skip zero vectors (can't be normalized)
        let is_zero = v.iter().all(|&x| x.abs() < 1e-6);
        if is_zero {
            return Ok(());
        }

        let mut v_copy = v.clone();
        VectorValidator::normalize(&mut v_copy).unwrap();

        let norm: f32 = v_copy.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!(
            (norm - 1.0).abs() < 1e-5,
            "Normalized vector should have norm 1.0, got {}",
            norm
        );
    }

    /// Test: Normalizing twice gives the same result
    #[test]
    fn test_normalization_idempotent(v in arb_dense_vector(64)) {
        let is_zero = v.iter().all(|&x| x.abs() < 1e-6);
        if is_zero {
            return Ok(());
        }

        let mut v1 = v.clone();
        VectorValidator::normalize(&mut v1).unwrap();

        let mut v2 = v1.clone();
        VectorValidator::normalize(&mut v2).unwrap();

        for (a, b) in v1.iter().zip(v2.iter()) {
            prop_assert!(
                (a - b).abs() < 1e-6,
                "Normalizing twice should give same result"
            );
        }
    }
}

// =============================================================================
// Sparse Vector Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test: Sparse-dense-sparse roundtrip preserves non-zero values
    #[test]
    fn test_sparse_dense_roundtrip(sv in arb_sparse_vector(100, 20)) {
        let dense = sv.to_dense();
        let sv2 = SparseVector::from_dense(&dense, 0.0);

        prop_assert_eq!(sv.indices, sv2.indices, "Indices should match after roundtrip");
        prop_assert_eq!(sv.values.len(), sv2.values.len(), "Value count should match");

        for (v1, v2) in sv.values.iter().zip(sv2.values.iter()) {
            prop_assert!(
                (v1 - v2).abs() < 1e-6,
                "Values should match after roundtrip: {} vs {}",
                v1, v2
            );
        }
    }

    /// Test: Sparse vector dot product is commutative
    #[test]
    fn test_sparse_dot_commutative(
        sv1 in arb_sparse_vector(100, 20),
        sv2 in arb_sparse_vector(100, 20),
    ) {
        let dot1 = sv1.dot(&sv2);
        let dot2 = sv2.dot(&sv1);

        prop_assert!(
            (dot1 - dot2).abs() < 1e-6,
            "Sparse dot product should be commutative: {} vs {}",
            dot1, dot2
        );
    }

    /// Test: Sparse vector L2 norm is non-negative
    #[test]
    fn test_sparse_l2_norm_non_negative(sv in arb_sparse_vector(100, 20)) {
        let norm = sv.l2_norm();
        prop_assert!(norm >= 0.0, "L2 norm should be non-negative, got {}", norm);
    }

    /// Test: Scaling sparse vector by 0 gives zero norm
    #[test]
    fn test_sparse_scale_by_zero(sv in arb_sparse_vector(100, 20)) {
        let mut sv_copy = sv.clone();
        sv_copy.scale(0.0);

        let norm = sv_copy.l2_norm();
        prop_assert!(
            norm < 1e-6,
            "Scaling by 0 should give zero norm, got {}",
            norm
        );
    }

    /// Test: Scaling preserves indices
    #[test]
    fn test_sparse_scale_preserves_indices(
        sv in arb_sparse_vector(100, 20),
        scalar in -10.0f32..10.0f32,
    ) {
        let original_indices = sv.indices.clone();
        let mut sv_copy = sv.clone();
        sv_copy.scale(scalar);

        prop_assert_eq!(
            sv_copy.indices, original_indices,
            "Scaling should preserve indices"
        );
    }
}

// =============================================================================
// Vector Validation Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test: Valid vectors pass validation
    #[test]
    fn test_valid_vectors_pass_validation(v in arb_dense_vector(64)) {
        // Filter out NaN/Inf which proptest might generate
        let has_invalid = v.iter().any(|&x| x.is_nan() || x.is_infinite());
        if has_invalid {
            return Ok(());
        }

        prop_assert!(
            VectorValidator::validate(&v, 64).is_ok(),
            "Valid vector should pass validation"
        );
    }

    /// Test: Dimension mismatch is detected
    #[test]
    fn test_dimension_mismatch_detected(
        v in arb_dense_vector(64),
        wrong_dim in 1usize..128usize,
    ) {
        // Skip if dimensions actually match
        if wrong_dim == 64 {
            return Ok(());
        }

        let result = VectorValidator::validate(&v, wrong_dim);
        prop_assert!(
            result.is_err(),
            "Dimension mismatch should be detected"
        );
    }
}

#[cfg(test)]
mod standard_tests {
    

    #[test]
    fn test_proptest_config() {
        // Verify proptest configuration
        let cases = std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(100);

        println!("Running property tests with {} cases", cases);
        assert!(cases >= 10, "Should run at least 10 cases");
    }
}
