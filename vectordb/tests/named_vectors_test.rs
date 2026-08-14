use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchQuery},
        table::{TableConfig, VectorSpaceConfig},
        vector::DistanceMetric,
    },
};
/// T3 — Named Vectors integration tests
///
/// Verifies that a table with multiple vector spaces (text_embedding + image_embedding)
/// routes inserts and searches independently per named space.
use std::collections::HashMap;

fn make_config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 256,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

/// Create a table with two named vector spaces, insert records with both vector types,
/// then search each space independently and verify the results are consistent.
///
/// Uses Euclidean distance so that proximity is unambiguous (unlike Cosine which
/// ignores magnitude and would give equal similarity to all [t,0,0,0] vectors).
#[tokio::test]
async fn test_named_vectors_independent_search() {
    const DIM_TEXT: usize = 4;
    const DIM_IMAGE: usize = 8;
    const N: usize = 20;

    let storage = IntelligentIndexedStorage::new(&make_config())
        .await
        .unwrap();

    let table = storage
        .create_table(TableConfig {
            name: "multimodal".to_string(),
            dimension: DIM_TEXT,
            distance_metric: DistanceMetric::Euclidean,
            description: Some("multi-vector test table".to_string()),
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: Some(vec![
                VectorSpaceConfig {
                    name: "text_embedding".to_string(),
                    dimension: DIM_TEXT,
                    distance_metric: DistanceMetric::Euclidean,
                },
                VectorSpaceConfig {
                    name: "image_embedding".to_string(),
                    dimension: DIM_IMAGE,
                    distance_metric: DistanceMetric::Euclidean,
                },
            ]),
        })
        .await
        .unwrap();

    assert_eq!(table.named_vectors.as_ref().unwrap().len(), 2);

    // Insert N records.
    // Record i text_embedding: [i, 0, 0, 0] — distinct by L2 distance
    // Record i image_embedding: [0, 0, 0, 0, i, 0, 0, 0]
    let records: Vec<CreateRecordRequest> = (0..N)
        .map(|i| {
            let t = i as f32;
            let mut named = HashMap::new();
            named.insert("text_embedding".to_string(), vec![t, 0.0, 0.0, 0.0]);
            named.insert(
                "image_embedding".to_string(),
                vec![0.0, 0.0, 0.0, 0.0, t, 0.0, 0.0, 0.0],
            );
            CreateRecordRequest {
                id: format!("rec-{}", i),
                vector: vec![t, 0.0, 0.0, 0.0],
                sparse_vector: None,
                named_vectors: Some(named),
                payload: Some(serde_json::json!({"idx": i})),
            }
        })
        .collect();

    storage.insert_records("multimodal", records).await.unwrap();

    // Text search: query = [19.5, 0, 0, 0] → closest is rec-19 (L2=0.5) then rec-18 (L2=1.5)
    let text_query = SearchQuery {
        vector: vec![19.5, 0.0, 0.0, 0.0],
        top_k: 3,
        limit: 3,
        vector_name: Some("text_embedding".to_string()),
        ..Default::default()
    };
    let text_result = storage.search("multimodal", text_query).await.unwrap();
    assert_eq!(
        text_result.hits.len(),
        3,
        "text search should return 3 hits"
    );
    let top_text_id = &text_result.hits[0].id;
    assert_eq!(
        top_text_id, "rec-19",
        "top text hit should be rec-19 (L2 closest)"
    );

    // Image search: query = [0,0,0,0,19.5,0,0,0] → closest is rec-19 in image space
    let image_query = SearchQuery {
        vector: vec![0.0, 0.0, 0.0, 0.0, 19.5, 0.0, 0.0, 0.0],
        top_k: 3,
        limit: 3,
        vector_name: Some("image_embedding".to_string()),
        ..Default::default()
    };
    let image_result = storage.search("multimodal", image_query).await.unwrap();
    assert_eq!(
        image_result.hits.len(),
        3,
        "image search should return 3 hits"
    );
    let top_image_id = &image_result.hits[0].id;
    assert_eq!(
        top_image_id, "rec-19",
        "top image hit should be rec-19 (L2 closest)"
    );

    // Cross-check: text search should NOT find by image coordinate and vice versa
    // The text index has dim=4; a dim-8 vector for image space can't be searched through the text index.
    // Just verify each named index returns the right top record.
    let text_low_query = SearchQuery {
        vector: vec![0.5, 0.0, 0.0, 0.0],
        top_k: 1,
        limit: 1,
        vector_name: Some("text_embedding".to_string()),
        ..Default::default()
    };
    let low_result = storage.search("multimodal", text_low_query).await.unwrap();
    assert_eq!(
        low_result.hits[0].id, "rec-0",
        "closest to [0.5,…] in text space should be rec-0 ([0,…])"
    );
}

/// Verify that an unknown vector_name falls back to the default index gracefully.
#[tokio::test]
async fn test_named_vectors_fallback_to_default() {
    const DIM: usize = 4;

    let storage = IntelligentIndexedStorage::new(&make_config())
        .await
        .unwrap();

    storage
        .create_table(TableConfig {
            name: "fallback_test".to_string(),
            dimension: DIM,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
        })
        .await
        .unwrap();

    storage
        .insert_records(
            "fallback_test",
            vec![CreateRecordRequest {
                id: "r1".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                sparse_vector: None,
                named_vectors: None,
                payload: None,
            }],
        )
        .await
        .unwrap();

    // Query with a vector_name not in the table → falls back to default index
    let q = SearchQuery {
        vector: vec![1.0, 0.0, 0.0, 0.0],
        top_k: 1,
        limit: 1,
        vector_name: Some("nonexistent_space".to_string()),
        ..Default::default()
    };
    let result = storage.search("fallback_test", q).await.unwrap();
    // Should still return the default index result
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].id, "r1");
}
