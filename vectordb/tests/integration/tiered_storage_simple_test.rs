// Simplified Tiered Storage QA Tests
//
// Tests basic storage functionality for Memory tier
// Note: Full tiered storage tests require API updates for new SearchQuery interface

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

/// Helper function to create test vectors
fn create_test_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let mut vec: Vec<f32> = (0..dim)
                .map(|j| ((i + j) as f32 * 0.1).sin())
                .collect();

            // Normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
            vec
        })
        .collect()
}

#[tokio::test]
async fn test_basic_insert_and_search() {
    // Test basic insertion and search functionality

    let table_config = TableConfig {
        name: "basic_test".to_string(),
        dimension: 64,
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

    let storage = create_test_storage(10).await;
    storage.create_table(table_config).await.unwrap();

    // Insert vectors
    let vectors = create_test_vectors(100, 64);
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

    storage.insert_records("basic_test", records).await.unwrap();

    // Search
    let query = SearchQuery {
        vector: vectors[0].clone(),
        top_k: 10,
        filter: None,
        with_payload: true,
        with_vector: false,
        limit: 10,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let results = storage.search("basic_test", query).await.unwrap();

    // Verify
    assert!(
        !results.hits.is_empty(),
        "Should return results"
    );

    assert_eq!(
        results.hits[0].id, "vec_0",
        "First result should be exact match"
    );

    // Score should be very high for exact match (close to 1.0 for cosine)
    assert!(
        results.hits[0].score > 0.99,
        "Exact match should have high score, got {}",
        results.hits[0].score
    );
}

#[tokio::test]
#[ignore = "performance benchmark — run with: cargo test --test integration test_memory_tier_performance -- --ignored"]
async fn test_memory_tier_performance() {
    // Test that Memory tier provides fast access

    let table_config = TableConfig {
        name: "perf_test".to_string(),
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

    let storage = create_test_storage(512).await;
    storage.create_table(table_config).await.unwrap();

    // Insert test data
    let vectors = create_test_vectors(1000, 128);
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

    storage.insert_records("perf_test", records).await.unwrap();

    // Warm up
    for _ in 0..5 {
        let query = SearchQuery {
            vector: vectors[0].clone(),
            top_k: 10,
            filter: None,
            with_payload: false,
            with_vector: false,
            limit: 10,
            search_mode: SearchMode::Normal,
            vector_name: None,
            ..Default::default()
        };
        storage.search("perf_test", query).await.unwrap();
    }

    // Measure latency for 100 queries
    let mut total_time = std::time::Duration::ZERO;

    for i in 0..100 {
        let query_vec = &vectors[i % vectors.len()];
        let query = SearchQuery {
            vector: query_vec.clone(),
            top_k: 10,
            filter: None,
            with_payload: false,
            with_vector: false,
            limit: 10,
            search_mode: SearchMode::Normal,
            vector_name: None,
            ..Default::default()
        };

        let start = Instant::now();
        storage.search("perf_test", query).await.unwrap();
        total_time += start.elapsed();
    }

    let avg_latency_ms = total_time.as_secs_f64() * 1000.0 / 100.0;

    println!("Average search latency: {:.3}ms", avg_latency_ms);

    // Memory tier should be reasonably fast (< 100ms average)
    assert!(
        avg_latency_ms < 100.0,
        "Memory tier average latency should be <100ms, got {:.3}ms",
        avg_latency_ms
    );
}

#[tokio::test]
async fn test_data_persistence() {
    // Test that data persists correctly in storage

    let table_config = TableConfig {
        name: "persist_test".to_string(),
        dimension: 64,
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

    let storage = create_test_storage(10).await;
    storage.create_table(table_config).await.unwrap();

    // Insert data
    let vectors = create_test_vectors(50, 64);
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

    storage.insert_records("persist_test", records).await.unwrap();

    // Verify data can be searched multiple times
    for iteration in 0..5 {
        let query = SearchQuery {
            vector: vectors[0].clone(),
            top_k: 10,
            filter: None,
            with_payload: false,
            with_vector: false,
            limit: 10,
            search_mode: SearchMode::Normal,
            vector_name: None,
            ..Default::default()
        };

        let results = storage.search("persist_test", query).await.unwrap();

        assert!(
            !results.hits.is_empty(),
            "Iteration {}: Should return results",
            iteration
        );

        assert_eq!(
            results.hits[0].id, "vec_0",
            "Iteration {}: Should return consistent results",
            iteration
        );
    }
}

#[test]
fn test_tiered_storage_compilation() {
    // Verify tiered storage tests compile
    println!("Tiered storage tests compiled successfully");
}
