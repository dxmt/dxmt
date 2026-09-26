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
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
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
                text: None,
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
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
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
                text: None,
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

/// A named space is scored with *its own* metric, not the table's.
///
/// `similarity_from_distance` turns the distance an index returns into the score the API
/// returns, and the formula is per metric: `1 - d` for cosine, `1 / (1 + d)` for euclidean,
/// `-d` for dot. The search path took the *table's* metric for that conversion even when a
/// named space had answered, so every named space whose metric differed from the table's came
/// back with scores computed by the wrong formula.
///
/// What made it survive: the index sorts by distance *before* any of this, so the **order was
/// always right**. Only the numbers were wrong — a `dot` space under a `cosine` table scored
/// `1 - (-dot)`, which is above 1 for every hit and clamps flat to 1.0, and a `euclidean`
/// space scored `1 - d` instead of `1 / (1 + d)`. A caller reading the ranking saw nothing
/// amiss; a caller thresholding on `score_threshold`, or fusing two lists, got nonsense.
///
/// This is checked against the same vectors in a table whose *default* metric is the one under
/// test, because that path was always right and is therefore the only reference that does not
/// assume the thing being tested.
#[tokio::test]
async fn a_named_space_is_scored_with_its_own_metric() {
    use dxmt_vectordb::types::record::SearchMode;

    const VECTORS: [(&str, [f32; 4]); 5] = [
        ("berlin", [0.05, 0.61, 0.76, 0.74]),
        ("london", [0.19, 0.81, 0.75, 0.11]),
        ("moscow", [0.36, 0.55, 0.47, 0.94]),
        ("sanfrancisco", [0.18, 0.01, 0.85, 0.80]),
        ("shanghai", [0.24, 0.18, 0.22, 0.44]),
    ];
    let query = vec![0.35f32, 0.55, 0.47, 0.94];

    for metric in [
        DistanceMetric::Dot,
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
    ] {
        let storage = IntelligentIndexedStorage::new(&make_config())
            .await
            .unwrap();

        // The reference: a table whose own metric is the one under test.
        storage
            .create_table(TableConfig {
                name: "reference".to_string(),
                dimension: 4,
                distance_metric: metric,
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
            })
            .await
            .unwrap();

        // The table under test: a *different* default metric, and a named space carrying the
        // one being checked. Cosine is the odd one out for two of the three runs, and for the
        // cosine run the default is euclidean, so no run can pass by the two agreeing.
        let default_metric = if metric == DistanceMetric::Cosine {
            DistanceMetric::Euclidean
        } else {
            DistanceMetric::Cosine
        };
        storage
            .create_table(TableConfig {
                name: "named".to_string(),
                dimension: 4,
                distance_metric: default_metric,
                description: None,
                index_type: None,
                text_fields: None,
                enable_full_text_search: false,
                memory_optimization: None,
                payload_schema: None,
                schema_version: 1,
                named_vectors: Some(vec![VectorSpaceConfig {
                    name: "space".to_string(),
                    dimension: 4,
                    distance_metric: metric,
                }]),
                hnsw: None,
                quantization: None,
                payload_index: None,
                embedding: None,
            })
            .await
            .unwrap();

        for (id, vector) in VECTORS {
            storage
                .insert_records(
                    "reference",
                    vec![CreateRecordRequest {
                        id: id.to_string(),
                        vector: vector.to_vec(),
                        text: None,
                        sparse_vector: None,
                        named_vectors: None,
                        payload: None,
                    }],
                )
                .await
                .unwrap();
            let mut spaces = HashMap::new();
            spaces.insert("space".to_string(), vector.to_vec());
            storage
                .insert_records(
                    "named",
                    vec![CreateRecordRequest {
                        id: id.to_string(),
                        vector: vector.to_vec(),
                        text: None,
                        sparse_vector: None,
                        named_vectors: Some(spaces),
                        payload: None,
                    }],
                )
                .await
                .unwrap();
        }

        let ask = |table: &'static str, space: Option<&'static str>| {
            let query = query.clone();
            let storage = &storage;
            async move {
                storage
                    .search(
                        table,
                        SearchQuery {
                            vector: query,
                            top_k: 5,
                            limit: 5,
                            vector_name: space.map(str::to_string),
                            search_mode: SearchMode::Accurate,
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap()
                    .hits
                    .into_iter()
                    .map(|hit| (hit.id, hit.score))
                    .collect::<Vec<_>>()
            }
        };

        let reference = ask("reference", None).await;
        let named = ask("named", Some("space")).await;

        assert_eq!(
            reference.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            named.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            "{metric:?}: the two rankings differ"
        );
        for ((id, want), (_, got)) in reference.iter().zip(named.iter()) {
            assert!(
                (want - got).abs() < 1e-5,
                "{metric:?}: '{id}' scored {got} in a named space and {want} in a table of the \
                 same metric. The named space was scored with the table's formula."
            );
        }
        // And the failure this replaces was not subtle once looked at: every dot score was
        // exactly 1.0. A regression that produced constant scores would still pass the
        // ordering check above, so say it outright.
        let distinct: std::collections::HashSet<String> =
            named.iter().map(|(_, s)| format!("{s:.4}")).collect();
        assert!(
            distinct.len() > 1,
            "{metric:?}: every hit scored the same ({named:?}), which is what clamping a \
             mis-converted distance looks like"
        );
    }
}
