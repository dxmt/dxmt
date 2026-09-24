use dxmt_vectordb::config::{MemoryConfig, StorageConfig};
use dxmt_vectordb::storage::IntelligentIndexedStorage;
use dxmt_vectordb::types::record::{
    Condition, CreateRecordRequest, Filter, FilterOperation, SearchMode, SearchQuery,
};
use dxmt_vectordb::types::table::TableConfig;
use dxmt_vectordb::types::vector::{DenseVector, DistanceMetric};

/// Generate random vector with given dimension
fn random_vector(dim: usize) -> DenseVector {
    (0..dim).map(|_| rand::random::<f32>()).collect()
}

/// Helper function to create test storage
async fn create_test_storage() -> IntelligentIndexedStorage {
    let config = StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 1024,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };
    IntelligentIndexedStorage::new(&config)
        .await
        .expect("Failed to create storage")
}

#[tokio::test]
async fn test_pre_filtering_with_highly_selective_filter() {
    let storage = create_test_storage().await;

    // Create table
    let table_config = TableConfig {
        name: "test_pre_filter".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        description: Some("Test PRE-FILTERING".to_string()),
        index_type: None,
        enable_full_text_search: false,
        text_fields: None,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
        payload_index: None,
    };
    storage.create_table(table_config).await.unwrap();

    // Insert 1000 records with different categories
    // Only 1% will have category_rare (10 records)
    let mut records = Vec::new();
    for i in 0..1000 {
        let category = if i % 100 == 0 {
            "category_rare" // 1% - this should trigger PRE-FILTERING
        } else {
            "category_common"
        };

        records.push(CreateRecordRequest {
            id: format!("record_{}", i),
            vector: random_vector(128),
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "category": category,
                "index": i,
            })),
            named_vectors: None,
        });
    }

    storage
        .insert_records("test_pre_filter", records)
        .await
        .unwrap();

    // First, verify records were inserted correctly
    let stats = storage.get_stats().await;
    println!("Storage stats after insert: {:?}", stats);

    // Search with highly selective filter (1% selectivity)
    let query = SearchQuery {
        vector: random_vector(128),
        top_k: 5,
        limit: 5,
        filter: Some(Filter {
            must: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::Equal(serde_json::json!("category_rare")),
            }]),
            must_not: None,
            should: None,
        }),
        with_payload: true,
        with_vector: false,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = storage.search("test_pre_filter", query).await.unwrap();

    // Debug output
    println!(
        "Found {} results, tookms: {}, total: {}",
        result.hits.len(),
        result.tookms,
        result.total
    );

    // Should find results (up to 5, but only 10 total match)
    assert!(
        !result.hits.is_empty(),
        "Should find some results with rare category. Got {} results.",
        result.hits.len()
    );
    assert!(result.hits.len() <= 5, "Should not exceed top_k");

    // Verify all results have the correct category
    for hit in &result.hits {
        if let Some(payload) = &hit.payload {
            let category = payload.get("category").and_then(|v| v.as_str()).unwrap();
            assert_eq!(
                category, "category_rare",
                "All results should have category_rare"
            );
        }
    }

    println!(
        "PRE-FILTERING test passed: found {} results in {} ms",
        result.hits.len(),
        result.tookms
    );
}

#[tokio::test]
async fn test_post_filtering_with_low_selective_filter() {
    let storage = create_test_storage().await;

    // Create table
    let table_config = TableConfig {
        name: "test_post_filter".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        description: Some("Test POST-FILTERING".to_string()),
        index_type: None,
        enable_full_text_search: false,
        text_fields: None,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
        payload_index: None,
    };
    storage.create_table(table_config).await.unwrap();

    // Insert 1000 records with different categories
    // 50% will have category_common (500 records)
    let mut records = Vec::new();
    for i in 0..1000 {
        let category = if i % 2 == 0 {
            "category_common" // 50% - this should trigger POST-FILTERING
        } else {
            "category_other"
        };

        records.push(CreateRecordRequest {
            id: format!("record_{}", i),
            vector: random_vector(128),
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "category": category,
                "index": i,
            })),
            named_vectors: None,
        });
    }

    storage
        .insert_records("test_post_filter", records)
        .await
        .unwrap();

    // Search with low selective filter (50% selectivity)
    let query = SearchQuery {
        vector: random_vector(128),
        top_k: 10,
        limit: 10,
        filter: Some(Filter {
            must: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::Equal(serde_json::json!("category_common")),
            }]),
            must_not: None,
            should: None,
        }),
        with_payload: true,
        with_vector: false,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = storage.search("test_post_filter", query).await.unwrap();

    // Should find 10 results
    assert_eq!(result.hits.len(), 10, "Should find 10 results");

    // Verify all results have the correct category
    for hit in &result.hits {
        if let Some(payload) = &hit.payload {
            let category = payload.get("category").and_then(|v| v.as_str()).unwrap();
            assert_eq!(
                category, "category_common",
                "All results should have category_common"
            );
        }
    }

    println!(
        "POST-FILTERING test passed: found {} results in {} ms",
        result.hits.len(),
        result.tookms
    );
}

#[tokio::test]
async fn test_filter_with_in_operation() {
    let storage = create_test_storage().await;

    // Create table
    let table_config = TableConfig {
        name: "test_in_filter".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        description: Some("Test IN filter".to_string()),
        index_type: None,
        enable_full_text_search: false,
        text_fields: None,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
        payload_index: None,
    };
    storage.create_table(table_config).await.unwrap();

    // Insert 1000 records with categories A, B, C, D
    let mut records = Vec::new();
    for i in 0..1000 {
        let category = match i % 4 {
            0 => "A",
            1 => "B",
            2 => "C",
            _ => "D",
        };

        records.push(CreateRecordRequest {
            id: format!("record_{}", i),
            vector: random_vector(128),
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "category": category,
                "index": i,
            })),
            named_vectors: None,
        });
    }

    storage
        .insert_records("test_in_filter", records)
        .await
        .unwrap();

    // Search with IN filter for categories A and B (50% selectivity)
    let query = SearchQuery {
        vector: random_vector(128),
        top_k: 10,
        limit: 10,
        filter: Some(Filter {
            must: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::In(vec![
                    serde_json::json!("A"),
                    serde_json::json!("B"),
                ]),
            }]),
            must_not: None,
            should: None,
        }),
        with_payload: true,
        with_vector: false,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = storage.search("test_in_filter", query).await.unwrap();

    // Should find 10 results
    assert_eq!(result.hits.len(), 10, "Should find 10 results");

    // Verify all results have category A or B
    for hit in &result.hits {
        if let Some(payload) = &hit.payload {
            let category = payload.get("category").and_then(|v| v.as_str()).unwrap();
            assert!(
                category == "A" || category == "B",
                "All results should have category A or B, got: {}",
                category
            );
        }
    }

    println!(
        "IN filter test passed: found {} results in {} ms",
        result.hits.len(),
        result.tookms
    );
}
