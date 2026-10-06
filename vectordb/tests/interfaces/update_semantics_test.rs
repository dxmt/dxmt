//! `apply_update` — the one place an update decides what it keeps — from the storage API, the
//! way gRPC reaches it. `tests/api/test_update_semantics.py` pins the same rules over HTTP.
//!
//! A vector-only update used to replace the payload with nothing: in replace mode `payload:
//! None` was passed straight through as the new payload, so re-embedding a record erased its
//! metadata. Both the HTTP handler and this function had that decision, in separate copies.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::{IntelligentIndexedStorage, RecordUpdate},
    types::{record::CreateRecordRequest, table::TableConfig, vector::DistanceMetric},
};
use serde_json::json;

async fn table_with_one_record(wal: Option<std::path::PathBuf>) -> IntelligentIndexedStorage {
    let storage = IntelligentIndexedStorage::new_with_wal(
        &StorageConfig {
            memory: Some(MemoryConfig {
                max_size_mb: 128,
                eviction_policy: "lru".to_string(),
            }),
            ..Default::default()
        },
        wal,
    )
    .await
    .unwrap();
    storage
        .create_table(TableConfig {
            name: "t".into(),
            dimension: 2,
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
            rerank: None,
        })
        .await
        .unwrap();
    storage
        .insert_records(
            "t",
            vec![CreateRecordRequest {
                id: "a".into(),
                vector: vec![1.0, 0.0],
                sparse_vector: None,
                named_vectors: None,
                payload: Some(json!({"keep": "me", "n": 1})),
                text: None,
            }],
        )
        .await
        .unwrap();
    storage
}

async fn check(storage: &IntelligentIndexedStorage) {
    let id = "a".to_string();

    // Vector only: the payload stays.
    storage
        .apply_update(
            "t",
            &id,
            RecordUpdate {
                vector: Some(vec![0.0, 1.0]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let r = storage.get_record("t", &id).await.unwrap();
    assert_eq!(r.vector, vec![0.0, 1.0]);
    assert_eq!(
        r.payload,
        json!({"keep": "me", "n": 1}),
        "a vector update erased the payload"
    );

    // Nothing at all: nothing changes.
    storage
        .apply_update("t", &id, RecordUpdate::default())
        .await
        .unwrap();
    let r = storage.get_record("t", &id).await.unwrap();
    assert_eq!(
        (r.vector, r.payload),
        (vec![0.0, 1.0], json!({"keep": "me", "n": 1}))
    );

    // Merge: keys added and overwritten, the rest kept.
    storage
        .apply_update(
            "t",
            &id,
            RecordUpdate {
                payload: Some(json!({"n": 2, "new": true})),
                merge: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let r = storage.get_record("t", &id).await.unwrap();
    assert_eq!(r.payload, json!({"keep": "me", "n": 2, "new": true}));

    // Replace with a payload: replaced, and the vector kept.
    storage
        .apply_update(
            "t",
            &id,
            RecordUpdate {
                payload: Some(json!({"only": 1})),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let r = storage.get_record("t", &id).await.unwrap();
    assert_eq!((r.vector, r.payload), (vec![0.0, 1.0], json!({"only": 1})));

    // An empty object is how to clear one.
    storage
        .apply_update(
            "t",
            &id,
            RecordUpdate {
                payload: Some(json!({})),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        storage.get_record("t", &id).await.unwrap().payload,
        json!({})
    );
}

#[tokio::test]
async fn an_update_keeps_what_it_was_not_given_in_memory() {
    check(&table_with_one_record(None).await).await;
}

/// On disk the vector lives in the row file and the payload in the record, so "keep the one
/// not given" reads each from a different place.
#[tokio::test]
async fn an_update_keeps_what_it_was_not_given_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    check(&table_with_one_record(Some(dir.path().join("wal"))).await).await;
}
