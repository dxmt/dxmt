// API Handlers Integration Tests
//
// Tests core REST API functionality including:
// - Table creation and management
// - Record insertion with validation
// - Search operations
// - Error handling

use dxmt_vectordb::{
    api::handlers::AppState,
    api::server_health::ServerHealth,
    config::{StorageConfig, MemoryConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::CreateRecordRequest,
        vector::DistanceMetric,
    },
};
use std::sync::Arc;

/// Helper function to create test AppState
async fn create_test_app_state() -> AppState {
    let config = StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 512,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };

    let storage = IntelligentIndexedStorage::new(&config)
        .await
        .expect("Failed to create storage");

    let server_health = ServerHealth::default();

    AppState {
        storage: Arc::new(storage),
        server_health: Arc::new(server_health),
    }
}

#[tokio::test]
async fn test_create_table_basic() {
    let state = create_test_app_state().await;

    // Create a basic table
    let table_config = dxmt_vectordb::types::table::TableConfig {
        name: "test_table".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        description: Some("Test table".to_string()),
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };

    let result = state.storage.create_table(table_config).await;
    assert!(result.is_ok(), "Failed to create table: {:?}", result.err());

    // Verify table exists
    let table = state.storage.get_table("test_table").await;
    assert!(table.is_ok(), "Table should exist");

    let table = table.unwrap();
    assert_eq!(table.name, "test_table");
    assert_eq!(table.dimension, 128);
    assert_eq!(table.distance_metric, DistanceMetric::Cosine);
}

#[tokio::test]
async fn test_create_table_duplicate() {
    let state = create_test_app_state().await;

    // Test: Duplicate table name
    let table_config = dxmt_vectordb::types::table::TableConfig {
        name: "duplicate_test".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };

    state.storage.create_table(table_config.clone()).await.unwrap();
    let result = state.storage.create_table(table_config).await;
    assert!(result.is_err(), "Should reject duplicate table name");
}

#[tokio::test]
async fn test_insert_records_basic() {
    let state = create_test_app_state().await;

    // Create table
    let table_config = dxmt_vectordb::types::table::TableConfig {
        name: "insert_test".to_string(),
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
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };
    state.storage.create_table(table_config).await.unwrap();

    // Insert records
    let records = vec![
        CreateRecordRequest {
            id: "vec1".to_string(),
            vector: vec![1.0; 64],
            sparse_vector: None,
            payload: Some(serde_json::json!({"label": "A"})),
            named_vectors: None,
            text: None,
        },
        CreateRecordRequest {
            id: "vec2".to_string(),
            vector: vec![2.0; 64],
            sparse_vector: None,
            payload: Some(serde_json::json!({"label": "B"})),
            named_vectors: None,
            text: None,
        },
    ];

    let result = state.storage.insert_records("insert_test", records).await;
    assert!(result.is_ok(), "Failed to insert records: {:?}", result.err());

    let inserted = result.unwrap();
    assert_eq!(inserted.len(), 2, "Should insert 2 records");
    assert_eq!(inserted[0].id, "vec1");
    assert_eq!(inserted[1].id, "vec2");
}

#[tokio::test]
async fn test_insert_records_wrong_dimension() {
    let state = create_test_app_state().await;

    // Create table
    let table_config = dxmt_vectordb::types::table::TableConfig {
        name: "validation_test".to_string(),
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
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };
    state.storage.create_table(table_config).await.unwrap();

    // Test: Wrong dimension
    let records = vec![
        CreateRecordRequest {
            id: "vec1".to_string(),
            vector: vec![1.0; 32], // Wrong dimension (32 instead of 64)
            sparse_vector: None,
            payload: None,
            named_vectors: None,
            text: None,
        },
    ];

    let result = state.storage.insert_records("validation_test", records).await;
    assert!(result.is_err(), "Should reject wrong dimension");
}

#[tokio::test]
async fn test_search_basic() {
    let state = create_test_app_state().await;

    // Create table and insert data
    let table_config = dxmt_vectordb::types::table::TableConfig {
        name: "search_test".to_string(),
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
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };
    state.storage.create_table(table_config).await.unwrap();

    // Insert test vectors
    let records = (0..10)
        .map(|i| CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vec![i as f32; 64],
            sparse_vector: None,
            payload: Some(serde_json::json!({"index": i})),
            named_vectors: None,
            text: None,
        })
        .collect();

    state.storage.insert_records("search_test", records).await.unwrap();

    // Search
    let query = dxmt_vectordb::types::record::SearchQuery {
        vector: vec![5.0; 64],
        top_k: 5,
        filter: None,
        with_payload: true,
        with_vector: false,
        limit: 5,
        search_mode: dxmt_vectordb::types::record::SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = state.storage.search("search_test", query).await;
    assert!(result.is_ok(), "Search should succeed");

    let search_result = result.unwrap();
    assert!(!search_result.hits.is_empty(), "Should return results");
    assert!(search_result.hits.len() <= 5, "Should respect top_k limit");

    // Verify results are sorted by score (lower is better for distance metrics)
    let scores: Vec<f32> = search_result.hits.iter().map(|h| h.score).collect();
    let mut sorted_scores = scores.clone();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(scores, sorted_scores, "Results should be sorted by score");
}

#[tokio::test]
async fn test_search_nonexistent_table() {
    let state = create_test_app_state().await;

    let query = dxmt_vectordb::types::record::SearchQuery {
        vector: vec![1.0; 64],
        top_k: 10,
        filter: None,
        with_payload: false,
        with_vector: false,
        limit: 10,
        search_mode: dxmt_vectordb::types::record::SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = state.storage.search("nonexistent_table", query).await;
    assert!(result.is_err(), "Should fail for nonexistent table");
}

#[tokio::test]
async fn test_delete_record() {
    let state = create_test_app_state().await;

    // Create table and insert data
    let table_config = dxmt_vectordb::types::table::TableConfig {
        name: "delete_test".to_string(),
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
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };
    state.storage.create_table(table_config).await.unwrap();

    let records = vec![
        CreateRecordRequest {
            id: "vec1".to_string(),
            vector: vec![1.0; 64],
            sparse_vector: None,
            payload: None,
            named_vectors: None,
            text: None,
        },
        CreateRecordRequest {
            id: "vec2".to_string(),
            vector: vec![2.0; 64],
            sparse_vector: None,
            payload: None,
            named_vectors: None,
            text: None,
        },
    ];

    state.storage.insert_records("delete_test", records).await.unwrap();

    // Delete one record
    let result = state.storage.delete_records("delete_test", vec!["vec1".to_string()]).await;
    assert!(result.is_ok(), "Delete should succeed");

    // Verify we still have vec2
    let query = dxmt_vectordb::types::record::SearchQuery {
        vector: vec![2.0; 64],
        top_k: 10,
        filter: None,
        with_payload: false,
        with_vector: false,
        limit: 10,
        search_mode: dxmt_vectordb::types::record::SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let search_result = state.storage.search("delete_test", query).await.unwrap();
    assert!(!search_result.hits.is_empty(), "Should still have vec2");
}

#[tokio::test]
async fn test_update_record() {
    let state = create_test_app_state().await;

    // Create table
    let table_config = dxmt_vectordb::types::table::TableConfig {
        name: "update_test".to_string(),
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
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };
    state.storage.create_table(table_config).await.unwrap();

    // Insert initial record
    let records = vec![
        CreateRecordRequest {
            id: "vec1".to_string(),
            vector: vec![1.0; 64],
            sparse_vector: None,
            payload: Some(serde_json::json!({"version": 1})),
            named_vectors: None,
            text: None,
        },
    ];
    state.storage.insert_records("update_test", records).await.unwrap();

    // Update record (using update_record - singular)
    let updated_record = CreateRecordRequest {
        id: "vec1".to_string(),
        vector: vec![2.0; 64],
        sparse_vector: None,
        payload: Some(serde_json::json!({"version": 2})),
        named_vectors: None,
        text: None,
    };

    let result = state.storage.update_record("update_test", &"vec1".to_string(), updated_record).await;
    assert!(result.is_ok(), "Update should succeed");

    // Verify update
    let query = dxmt_vectordb::types::record::SearchQuery {
        vector: vec![2.0; 64],
        top_k: 1,
        filter: None,
        with_payload: true,
        with_vector: false,
        limit: 1,
        search_mode: dxmt_vectordb::types::record::SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let search_result = state.storage.search("update_test", query).await.unwrap();
    assert!(!search_result.hits.is_empty(), "Should find updated record");

    if let Some(payload) = &search_result.hits[0].payload {
        let version = payload.get("version").and_then(|v| v.as_i64());
        assert_eq!(version, Some(2), "Payload should be updated");
    }
}

#[tokio::test]
async fn test_list_tables() {
    let state = create_test_app_state().await;

    // Create multiple tables
    for i in 0..3 {
        let table_config = dxmt_vectordb::types::table::TableConfig {
            name: format!("table_{}", i),
            dimension: 64,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            named_vectors: None,
            payload_schema: None,
            schema_version: 1,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
        };
        state.storage.create_table(table_config).await.unwrap();
    }

    // List tables
    let tables = state.storage.list_tables().await;
    assert!(tables.is_ok(), "List tables should succeed");

    let tables = tables.unwrap();
    assert_eq!(tables.len(), 3, "Should have 3 tables");

    let table_names: Vec<String> = tables.iter().map(|t| t.name.clone()).collect();
    assert!(table_names.contains(&"table_0".to_string()));
    assert!(table_names.contains(&"table_1".to_string()));
    assert!(table_names.contains(&"table_2".to_string()));
}

#[tokio::test]
async fn test_large_batch_insert() {
    let state = create_test_app_state().await;

    // Create table
    let table_config = dxmt_vectordb::types::table::TableConfig {
        name: "batch_test".to_string(),
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
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };
    state.storage.create_table(table_config).await.unwrap();

    // Test: Large valid batch (should succeed)
    let large_batch: Vec<CreateRecordRequest> = (0..1000)
        .map(|i| CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vec![i as f32; 64],
            sparse_vector: None,
            payload: None,
            named_vectors: None,
            text: None,
        })
        .collect();

    let result = state.storage.insert_records("batch_test", large_batch).await;
    assert!(result.is_ok(), "Should accept batch of 1000");

    let inserted = result.unwrap();
    assert_eq!(inserted.len(), 1000, "Should insert all 1000 records");
}

#[tokio::test]
async fn test_server_health_status() {
    let state = create_test_app_state().await;

    // Check health status
    let health_state = state.server_health.current_state().await;

    // Should be healthy initially (not overloaded)
    assert!(
        !matches!(health_state, dxmt_vectordb::api::server_health::HealthState::Overloaded),
        "Server should not be overloaded initially"
    );
}
