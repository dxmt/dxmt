//! Writes are refused before the data directory's disk fills.
//!
//! `DiskGuard`'s unit tests cover the arithmetic; this covers the wiring: the guard is
//! consulted on the write path of a WAL-backed store, the caller is told why and how to change
//! it, and reads and deletes -- the remedy -- still work.
//!
//! `DXMT_MIN_FREE_DISK_MB` is process-global, so this is the only test in this binary.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{record::CreateRecordRequest, table::TableConfig, vector::DistanceMetric},
};
use tempfile::TempDir;

fn storage_config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 1024,
            eviction_policy: "lru".to_string(),
        }),
        ..Default::default()
    }
}

fn table(name: &str, dimension: usize) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension,
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
        rerank: None,
    }
}

fn record(id: &str, vector: Vec<f32>) -> CreateRecordRequest {
    CreateRecordRequest {
        id: id.to_string(),
        vector,
        sparse_vector: None,
        named_vectors: None,
        payload: None,
        text: None,
    }
}

#[tokio::test]
async fn a_write_is_refused_when_the_data_disk_is_under_the_floor() {
    let tmp = TempDir::new().unwrap();
    // A floor no disk can meet; read once, when the store opens.
    std::env::set_var("DXMT_MIN_FREE_DISK_MB", "1000000000000");
    let storage =
        IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(tmp.path().join("wal")))
            .await
            .unwrap();
    std::env::remove_var("DXMT_MIN_FREE_DISK_MB");

    storage.create_table(table("t", 4)).await.unwrap();
    let err = storage
        .insert_records("t", vec![record("a", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .expect_err("a write under the disk floor must be refused");
    let message = err.to_string();
    assert!(
        message.contains("free on the data directory's disk"),
        "{message}"
    );
    assert!(message.contains("DXMT_MIN_FREE_DISK_MB"), "{message}");
    assert_eq!(err.status_code(), 503, "{message}");

    assert!(storage
        .list_tables()
        .await
        .unwrap()
        .iter()
        .any(|t| t.name == "t"));
    storage
        .delete_table("t")
        .await
        .expect("a delete frees space and must still be accepted");
}
