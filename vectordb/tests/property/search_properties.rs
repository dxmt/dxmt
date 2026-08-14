// Property-based tests for search operations
//
// Verifies search correctness properties across random inputs

use proptest::prelude::*;
use dxmt_vectordb::{
    search::{
        brute_force::BruteForceIndex,
        hnsw::HnswIndex,
    },
    types::vector::DistanceMetric,
};

// Arbitrary vector generator
fn arb_vector(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1.0f32..1.0f32, dim..=dim)
        .prop_map(|mut v| {
            // Normalize to avoid extreme values
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            v
        })
}

// =============================================================================
// BruteForce Index Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test: Search returns exactly k results (or fewer if database is smaller)
    #[test]
    fn test_bruteforce_returns_k_results(
        vectors in prop::collection::vec(arb_vector(32), 10..50),
        query in arb_vector(32),
        k in 1usize..20,
    ) {
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);

        // Insert vectors
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(format!("v{}", i), vec.clone()).unwrap();
        }

        // Search
        let results = index.search_sync(&query, k).unwrap();

        // Should return min(k, vectors.len()) results
        let expected_count = k.min(vectors.len());
        prop_assert_eq!(
            results.len(), expected_count,
            "Should return {} results, got {}",
            expected_count, results.len()
        );
    }

    /// Test: Search results have no duplicate IDs
    #[test]
    fn test_bruteforce_no_duplicate_ids(
        vectors in prop::collection::vec(arb_vector(32), 10..50),
        query in arb_vector(32),
        k in 5usize..20,
    ) {
        
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(format!("v{}", i), vec.clone()).unwrap();
        }

        let results = index.search_sync(&query, k).unwrap();

        // Check for duplicates
        let mut seen_ids = std::collections::HashSet::new();
        for (id, _dist) in &results {
            prop_assert!(
                seen_ids.insert(id),
                "Duplicate ID found: {}",
                id
            );
        }
    }

    /// Test: Search results are sorted by distance (ascending)
    #[test]
    fn test_bruteforce_results_sorted(
        vectors in prop::collection::vec(arb_vector(32), 10..50),
        query in arb_vector(32),
        k in 5usize..20,
    ) {
        
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(format!("v{}", i), vec.clone()).unwrap();
        }

        let results = index.search_sync(&query, k).unwrap();

        // Check that distances are in ascending order
        for i in 1..results.len() {
            let prev_dist = results[i - 1].1;
            let curr_dist = results[i].1;
            prop_assert!(
                prev_dist <= curr_dist + 1e-6,
                "Results not sorted: {} > {} at position {}",
                prev_dist, curr_dist, i
            );
        }
    }

    /// Test: All results are valid vectors from the index
    #[test]
    fn test_bruteforce_results_are_valid(
        vectors in prop::collection::vec(arb_vector(32), 10..50),
        query in arb_vector(32),
        k in 5usize..10,
    ) {
        
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);

        let mut valid_ids = std::collections::HashSet::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = format!("v{}", i);
            valid_ids.insert(id.clone());
            index.insert(id, vec.clone()).unwrap();
        }

        let results = index.search_sync(&query, k).unwrap();

        // All returned IDs should be valid
        for (id, _dist) in &results {
            prop_assert!(
                valid_ids.contains(id),
                "Invalid ID returned: {}",
                id
            );
        }
    }
}

// =============================================================================
// HNSW Index Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Test: HNSW returns exactly k results (or fewer if database is smaller)
    #[test]
    fn test_hnsw_returns_k_results(
        vectors in prop::collection::vec(arb_vector(32), 10..30),
        query in arb_vector(32),
        k in 1usize..15,
    ) {
        
        let mut index = HnswIndex::new(DistanceMetric::Euclidean, None);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(format!("v{}", i), vec.clone()).unwrap();
        }

        let results = index.search(&query, k).unwrap();

        let expected_count = k.min(vectors.len());
        prop_assert_eq!(
            results.len(), expected_count,
            "Should return {} results, got {}",
            expected_count, results.len()
        );
    }

    /// Test: HNSW results have no duplicate IDs
    #[test]
    fn test_hnsw_no_duplicate_ids(
        vectors in prop::collection::vec(arb_vector(32), 10..30),
        query in arb_vector(32),
        k in 5usize..15,
    ) {
        
        let mut index = HnswIndex::new(DistanceMetric::Euclidean, None);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(format!("v{}", i), vec.clone()).unwrap();
        }

        let results = index.search(&query, k).unwrap();

        let mut seen_ids = std::collections::HashSet::new();
        for (id, _dist) in &results {
            prop_assert!(
                seen_ids.insert(id),
                "Duplicate ID found: {}",
                id
            );
        }
    }

    /// Test: HNSW results are sorted by distance
    #[test]
    fn test_hnsw_results_sorted(
        vectors in prop::collection::vec(arb_vector(32), 10..30),
        query in arb_vector(32),
        k in 5usize..15,
    ) {
        
        let mut index = HnswIndex::new(DistanceMetric::Euclidean, None);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(format!("v{}", i), vec.clone()).unwrap();
        }

        let results = index.search(&query, k).unwrap();

        for i in 1..results.len() {
            let prev_dist = results[i - 1].1;
            let curr_dist = results[i].1;
            prop_assert!(
                prev_dist <= curr_dist + 1e-6,
                "Results not sorted: {} > {} at position {}",
                prev_dist, curr_dist, i
            );
        }
    }
}

// =============================================================================
// Cross-Index Consistency Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Test: BruteForce and HNSW both return valid results
    /// Note: HNSW is approximate, so we only verify structural properties
    #[test]
    fn test_index_consistency(
        vectors in prop::collection::vec(arb_vector(32), 30..50),
        query in arb_vector(32),
    ) {
        // Filter out zero or near-zero vectors which can cause inconsistencies
        let vectors: Vec<_> = vectors.into_iter().filter(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            norm > 0.1
        }).collect();

        // Skip if we have too few vectors after filtering
        if vectors.len() < 30 {
            return Ok(());
        }

        // Check query is not near-zero
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm < 0.1 {
            return Ok(());
        }

        // Build both indexes
        let mut brute_force = BruteForceIndex::new(DistanceMetric::Euclidean);
        let mut hnsw = HnswIndex::new(DistanceMetric::Euclidean, None);

        for (i, vec) in vectors.iter().enumerate() {
            let id = format!("v{}", i);
            brute_force.insert(id.clone(), vec.clone()).unwrap();
            hnsw.insert(id, vec.clone()).unwrap();
        }

        let k = 10.min(vectors.len());
        let bf_results = brute_force.search_sync(&query, k).unwrap();
        let hnsw_results = hnsw.search(&query, k).unwrap();

        // Both should return k results
        prop_assert_eq!(bf_results.len(), k, "BruteForce should return k results");
        prop_assert_eq!(hnsw_results.len(), k, "HNSW should return k results");

        // Both should have no duplicates
        let bf_ids: std::collections::HashSet<_> =
            bf_results.iter().map(|(id, _)| id.clone()).collect();
        let hnsw_ids: std::collections::HashSet<_> =
            hnsw_results.iter().map(|(id, _)| id.clone()).collect();

        prop_assert_eq!(bf_ids.len(), k, "BruteForce should have no duplicates");
        prop_assert_eq!(hnsw_ids.len(), k, "HNSW should have no duplicates");

        // Check that HNSW top-1 is reasonably close to BruteForce top-1
        // (They should both find something close, even if not identical)
        let bf_best_dist = bf_results[0].1;
        let hnsw_best_dist = hnsw_results[0].1;

        // HNSW's best result should be within 50% of BruteForce's best
        prop_assert!(
            hnsw_best_dist <= bf_best_dist * 1.5,
            "HNSW best distance {:.4} should be within 50% of BruteForce best {:.4}",
            hnsw_best_dist, bf_best_dist
        );
    }
}

// =============================================================================
// Insert/Delete Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test: Inserting the same ID twice updates the vector
    #[test]
    fn test_insert_updates_vector(
        v1 in arb_vector(32),
        v2 in arb_vector(32),
        query in arb_vector(32),
    ) {
        
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);

        // Insert first vector
        index.insert("test_id".to_string(), v1.clone()).unwrap();
        let _results1 = index.search_sync(&query, 1).unwrap();

        // Insert second vector with same ID
        index.insert("test_id".to_string(), v2.clone()).unwrap();
        let results2 = index.search_sync(&query, 1).unwrap();

        // Should only have one result
        prop_assert_eq!(results2.len(), 1);
        prop_assert_eq!(&results2[0].0, "test_id");

        // Distance should be to v2, not v1
        let expected_dist = DistanceMetric::Euclidean.calculate(&query, &v2).unwrap();
        let actual_dist = results2[0].1;

        prop_assert!(
            (expected_dist - actual_dist).abs() < 1e-6,
            "Distance should be to updated vector: expected {}, got {}",
            expected_dist, actual_dist
        );
    }

    /// Test: Deleting a vector removes it from search results
    #[test]
    fn test_delete_removes_vector(
        vectors in prop::collection::vec(arb_vector(32), 5..15),
        query in arb_vector(32),
    ) {
        
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);

        // Insert vectors
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(format!("v{}", i), vec.clone()).unwrap();
        }

        // Delete the first vector
        index.remove(&"v0".to_string()).unwrap();

        // Search
        let results = index.search_sync(&query, 100).unwrap();

        // Should not contain v0
        for (id, _dist) in &results {
            prop_assert_ne!(id, "v0", "Deleted vector should not appear in results");
        }

        // Should have one fewer result
        prop_assert_eq!(
            results.len(), vectors.len() - 1,
            "Should have {} results after delete, got {}",
            vectors.len() - 1, results.len()
        );
    }
}

#[cfg(test)]
mod standard_tests {
    

    #[test]
    fn test_proptest_config() {
        let cases = std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(50);

        println!("Running search property tests with {} cases", cases);
        assert!(cases >= 10, "Should run at least 10 cases");
    }
}
