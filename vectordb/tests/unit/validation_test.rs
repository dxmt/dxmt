/// Test enhanced input validation
use dxmt_vectordb::types::{
    vector::{VectorValidator, ValidationConfig},
    sparse_vector::SparseVector,
};

#[test]
fn test_vector_validation_config() {
    let config = ValidationConfig {
        max_dimension: 100,
        min_dimension: 2,
        allow_zero_vectors: false,
        validate_range: Some((-1.0, 1.0)),
    };

    // Valid vector
    let valid_vec = vec![0.5, -0.5, 0.0];
    assert!(VectorValidator::validate_with_config(&valid_vec, 3, &config).is_ok());

    // Empty vector
    let empty_vec = vec![];
    assert!(VectorValidator::validate_with_config(&empty_vec, 0, &config).is_err());

    // Dimension too small
    let small_dim = vec![0.5];
    assert!(VectorValidator::validate_with_config(&small_dim, 1, &config).is_err());

    // Dimension too large
    let large_dim = vec![0.5; 200];
    assert!(VectorValidator::validate_with_config(&large_dim, 200, &config).is_err());

    // Zero vector (not allowed)
    let zero_vec = vec![0.0, 0.0, 0.0];
    assert!(VectorValidator::validate_with_config(&zero_vec, 3, &config).is_err());

    // Value out of range
    let out_of_range = vec![0.5, 1.5, 0.0];
    assert!(VectorValidator::validate_with_config(&out_of_range, 3, &config).is_err());

    // NaN value
    let nan_vec = vec![0.5, f32::NAN, 0.0];
    assert!(VectorValidator::validate_with_config(&nan_vec, 3, &config).is_err());

    // Infinite value
    let inf_vec = vec![0.5, f32::INFINITY, 0.0];
    assert!(VectorValidator::validate_with_config(&inf_vec, 3, &config).is_err());
}

#[test]
fn test_vector_validation_allow_zero() {
    let config = ValidationConfig {
        max_dimension: 100,
        min_dimension: 1,
        allow_zero_vectors: true,
        validate_range: None,
    };

    // Zero vector should be allowed now
    let zero_vec = vec![0.0, 0.0, 0.0];
    assert!(VectorValidator::validate_with_config(&zero_vec, 3, &config).is_ok());
}

#[test]
fn test_sparse_vector_validation() {
    // Valid sparse vector (sorted indices)
    let valid = SparseVector::new(vec![0, 2, 5], vec![1.0, 2.0, 3.0], 10);
    assert!(valid.is_ok());

    // Length mismatch
    let length_mismatch = SparseVector::new(vec![0, 2], vec![1.0, 2.0, 3.0], 10);
    assert!(length_mismatch.is_err());
    assert!(length_mismatch.unwrap_err().to_string().contains("length"));

    // Empty vector
    let empty = SparseVector::new(vec![], vec![], 10);
    assert!(empty.is_err());
    assert!(empty.unwrap_err().to_string().contains("empty"));

    // Unsorted indices (5 > 2 > 0 is descending, not ascending)
    let unsorted = SparseVector::new(vec![5, 2, 0], vec![1.0, 2.0, 3.0], 10);
    assert!(unsorted.is_err());
    assert!(unsorted.unwrap_err().to_string().contains("ascending"));

    // Duplicate indices (2, 2)
    let duplicate = SparseVector::new(vec![0, 2, 2], vec![1.0, 2.0, 3.0], 10);
    assert!(duplicate.is_err());
    assert!(duplicate.unwrap_err().to_string().contains("ascending"));

    // Out of bounds index (15 >= 10)
    let out_of_bounds = SparseVector::new(vec![0, 2, 15], vec![1.0, 2.0, 3.0], 10);
    assert!(out_of_bounds.is_err());
    assert!(out_of_bounds.unwrap_err().to_string().contains("out of bounds"));

    // NaN value
    let nan_val = SparseVector::new(vec![0, 2, 5], vec![1.0, f32::NAN, 3.0], 10);
    assert!(nan_val.is_err());
    assert!(nan_val.unwrap_err().to_string().contains("NaN"));

    // Infinite value
    let inf_val = SparseVector::new(vec![0, 2, 5], vec![1.0, f32::INFINITY, 3.0], 10);
    assert!(inf_val.is_err());
    assert!(inf_val.unwrap_err().to_string().contains("Infinite"));
}

#[test]
fn test_sparse_vector_sort_indices() {
    // Create unsorted sparse vector using unchecked constructor
    let mut sparse = SparseVector::new_unchecked(vec![5, 2, 0, 7], vec![3.0, 2.0, 1.0, 4.0], 10);

    // Should fail validation before sorting
    assert!(sparse.validate().is_err());

    // Sort indices
    sparse.sort_indices();

    // Should pass validation after sorting
    assert!(sparse.validate().is_ok());

    // Check that indices are sorted
    assert_eq!(sparse.indices, vec![0, 2, 5, 7]);
    assert_eq!(sparse.values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_sparse_vector_max_size() {
    // Create a vector that exceeds MAX_MULTIVECTOR_FLATTENED_LEN (10M)
    let large_size = SparseVector::MAX_MULTIVECTOR_FLATTENED_LEN + 1;
    let indices: Vec<u32> = (0..large_size as u32).collect();
    let values: Vec<f32> = vec![1.0; large_size];

    let result = SparseVector::new(indices, values, large_size * 2);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeding maximum"));
}

#[test]
fn test_sparse_vector_from_dense_and_roundtrip() {
    let dense = vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0];
    let sparse = SparseVector::from_dense(&dense, 0.5);

    // Should be valid (created via from_dense)
    assert!(sparse.validate().is_ok());

    // Check indices and values
    assert_eq!(sparse.indices, vec![1, 3, 6]);
    assert_eq!(sparse.values, vec![1.0, 2.0, 3.0]);

    // Convert back to dense
    let dense2 = sparse.to_dense();
    assert_eq!(dense, dense2);
}
