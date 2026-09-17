// Simplified Ephemeral VM Lifecycle Testing
//
// Tests basic VM lifecycle scenarios with current API
// Note: Full recovery and snapshot tests require additional infrastructure

use dxmt_vectordb::{
    config::{StorageConfig, MemoryConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchQuery, SearchMode},
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use std::time::Instant;

/// Helper to create test storage
async fn create_test_storage(max_size_mb: usize) -> IntelligentIndexedStorage {
    let config = StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };
    IntelligentIndexedStorage::new(&config).await.expect("Failed to create storage")
}

/// Helper to create test vectors
fn create_test_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let mut vec: Vec<f32> = (0..dim)
                .map(|j| ((i + j) as f32 * 0.1).sin())
                .collect();
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
            vec
        })
        .collect()
}

#[tokio::test]
async fn test_basic_state_recreation() {
    // Test recreating storage state (simulating VM restart)

    let table_config = TableConfig {
        name: "state_test".to_string(),
        dimension: 32,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    };

    let vectors = create_test_vectors(50, 32);

    // Phase 1: Create and populate
    let query_results_phase1 = {
        let storage = create_test_storage(10).await;
        storage.create_table(table_config.clone()).await.unwrap();

        let mut records = Vec::new();
        for (i, vector) in vectors.iter().enumerate() {
            records.push(CreateRecordRequest {
                id: format!("vec_{}", i),
                vector: vector.clone(),
                sparse_vector: None,
                payload: None,
                named_vectors: None,
            });
        }

        storage.insert_records("state_test", records).await.unwrap();

        // Perform search
        let query = SearchQuery {
            vector: vectors[0].clone(),
            top_k: 10,
            filter: None,
            with_payload: false,
            with_vector: false,
            limit: 10,
            search_mode: SearchMode::Normal,
            vector_name: None,
        };

        storage.search("state_test", query).await.unwrap()
    };

    // Phase 2: Recreate storage (simulating VM restart)
    let query_results_phase2 = {
        let storage = create_test_storage(10).await;
        storage.create_table(table_config.clone()).await.unwrap();

        let mut records = Vec::new();
        for (i, vector) in vectors.iter().enumerate() {
            records.push(CreateRecordRequest {
                id: format!("vec_{}", i),
                vector: vector.clone(),
                sparse_vector: None,
                payload: None,
                named_vectors: None,
            });
        }

        storage.insert_records("state_test", records).await.unwrap();

        // Perform same search
        let query = SearchQuery {
            vector: vectors[0].clone(),
            top_k: 10,
            filter: None,
            with_payload: false,
            with_vector: false,
            limit: 10,
            search_mode: SearchMode::Normal,
            vector_name: None,
        };

        storage.search("state_test", query).await.unwrap()
    };

    // Verify results are consistent
    assert_eq!(
        query_results_phase1.hits.len(),
        query_results_phase2.hits.len(),
        "Result count should match"
    );

    assert_eq!(
        query_results_phase1.hits[0].id,
        query_results_phase2.hits[0].id,
        "Top result should match"
    );
}

#[tokio::test]
async fn test_fast_data_loading() {
    // Test that data can be loaded quickly (simulating recovery)

    let table_config = TableConfig {
        name: "recovery_test".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Euclidean,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    };

    let vectors = create_test_vectors(1000, 128);

    // Measure load time
    let storage = create_test_storage(50).await;
    storage.create_table(table_config).await.unwrap();

    let load_start = Instant::now();

    let mut records = Vec::new();
    for (i, vector) in vectors.iter().enumerate() {
        records.push(CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vector.clone(),
            sparse_vector: None,
            payload: None,
            named_vectors: None,
        });
    }

    storage.insert_records("recovery_test", records).await.unwrap();

    let load_time = load_start.elapsed();

    println!("Loaded 1K vectors in {:?}", load_time);

    // Verify data is searchable immediately
    let query = SearchQuery {
        vector: vectors[0].clone(),
        top_k: 10,
        filter: None,
        with_payload: false,
        with_vector: false,
        limit: 10,
        search_mode: SearchMode::Normal,
        vector_name: None,
    };

    let results = storage.search("recovery_test", query).await.unwrap();

    assert!(!results.hits.is_empty(), "Should return results after loading");

    // Load should be reasonably fast (< 120 seconds for 1K vectors in debug builds)
    // Note: This includes both insertion and index building time
    assert!(
        load_time.as_secs() < 120,
        "Loading 1K vectors should be <120s, got {:?}",
        load_time
    );
}

#[tokio::test]
async fn test_index_consistency_after_rebuild() {
    // Test that rebuilt indexes produce consistent results

    let table_config = TableConfig {
        name: "rebuild_test".to_string(),
        dimension: 64,
        distance_metric: DistanceMetric::Euclidean,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    };

    let vectors = create_test_vectors(200, 64);

    // Build index first time
    let original_results = {
        let storage = create_test_storage(10).await;
        storage.create_table(table_config.clone()).await.unwrap();

        let mut records = Vec::new();
        for (i, vector) in vectors.iter().enumerate() {
            records.push(CreateRecordRequest {
                id: format!("vec_{}", i),
                vector: vector.clone(),
                sparse_vector: None,
                payload: None,
                named_vectors: None,
            });
        }

        storage.insert_records("rebuild_test", records).await.unwrap();

        // Query multiple test vectors
        let mut all_results = Vec::new();
        for i in [0, 50, 100, 150] {
            let query = SearchQuery {
                vector: vectors[i].clone(),
                top_k: 10,
                filter: None,
                with_payload: false,
                with_vector: false,
                limit: 10,
                search_mode: SearchMode::Normal,
                vector_name: None,
            };
            let results = storage.search("rebuild_test", query).await.unwrap();
            all_results.push(results);
        }
        all_results
    };

    // Rebuild index (simulate recovery)
    let rebuilt_results = {
        let storage = create_test_storage(10).await;
        storage.create_table(table_config.clone()).await.unwrap();

        let mut records = Vec::new();
        for (i, vector) in vectors.iter().enumerate() {
            records.push(CreateRecordRequest {
                id: format!("vec_{}", i),
                vector: vector.clone(),
                sparse_vector: None,
                payload: None,
                named_vectors: None,
            });
        }

        storage.insert_records("rebuild_test", records).await.unwrap();

        // Same queries
        let mut all_results = Vec::new();
        for i in [0, 50, 100, 150] {
            let query = SearchQuery {
                vector: vectors[i].clone(),
                top_k: 10,
                filter: None,
                with_payload: false,
                with_vector: false,
                limit: 10,
                search_mode: SearchMode::Normal,
                vector_name: None,
            };
            let results = storage.search("rebuild_test", query).await.unwrap();
            all_results.push(results);
        }
        all_results
    };

    // Verify all queries return same results
    assert_eq!(
        original_results.len(),
        rebuilt_results.len(),
        "Should have same number of query results"
    );

    for (query_idx, (orig_query, rebuilt_query)) in original_results
        .iter()
        .zip(rebuilt_results.iter())
        .enumerate()
    {
        assert_eq!(
            orig_query.hits.len(),
            rebuilt_query.hits.len(),
            "Query {} should have same result count",
            query_idx
        );

        // Top result should match
        assert_eq!(
            orig_query.hits[0].id, rebuilt_query.hits[0].id,
            "Query {} top result ID should match",
            query_idx
        );
    }
}

#[test]
fn test_ephemeral_vm_compilation() {
    // Verify ephemeral VM tests compile
    println!("Ephemeral VM tests compiled successfully");
}
