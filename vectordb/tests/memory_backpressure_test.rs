//! Writes are refused before the machine runs out of memory.
//!
//! Loading a 768-dimension dataset into a 62 GB box, the server allocated straight through
//! the end of memory: at roughly 7M records it held 54.9 GB of anonymous memory and the
//! machine stopped responding. ICMP still answered, so the kernel was alive, but sshd could
//! no longer fork, no OOM kill was ever recorded (that box had no swap), and it took a hard
//! reboot. Every write was accepted right up to that point.
//!
//! `MemoryGuard`'s own unit tests cover the arithmetic. What this file covers is the wiring:
//! that the guard is actually consulted on the write path and that the caller is told the
//! write did not happen, rather than finding out when the machine stops answering.
//!
//! `DXMT_MIN_FREE_MEMORY_MB` is process-global, so this is the only test in this binary. A
//! second one would race it however carefully it were written.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{record::CreateRecordRequest, table::TableConfig, vector::DistanceMetric},
};
use tempfile::TempDir;

fn storage_config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
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
async fn a_write_is_refused_when_free_memory_is_under_the_floor() {
    let tmp = TempDir::new().unwrap();

    // A floor no machine can satisfy. An explicitly configured floor is honoured as given --
    // the fraction-of-total cap applies only to the default -- which is what makes this
    // testable at all.
    std::env::set_var("DXMT_MIN_FREE_MEMORY_MB", "1000000000");
    let storage =
        IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(tmp.path().join("wal")))
            .await
            .unwrap();
    std::env::remove_var("DXMT_MIN_FREE_MEMORY_MB");

    storage.create_table(table("t", 4)).await.unwrap();

    let err = storage
        .insert_records("t", vec![record("a", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .expect_err("a write under the memory floor must be refused, not accepted");

    let message = err.to_string();
    assert!(
        message.contains("memory available") && message.contains("floor"),
        "the caller should be told why the write was refused, got: {message}"
    );
    assert!(
        message.contains("DXMT_MIN_FREE_MEMORY_MB"),
        "and how to change it, got: {message}"
    );

    // Reads are unaffected: the table is still there and still answers.
    let tables = storage.list_tables().await.unwrap();
    assert!(
        tables.iter().any(|t| t.name == "t"),
        "refusing writes must not take reads down with them"
    );

    // And deletes still work. This is the point of `ensure_memory_headroom` being separate
    // from `ensure_writable`: freeing memory is the remedy for being out of it, so a floor
    // that blocked deletes would lock the operator out of the only fix. `delete_table` goes
    // through `ensure_writable`, which the floor deliberately does not touch.
    storage
        .delete_table("t")
        .await
        .expect("a delete must still be accepted under the memory floor");
    let after = storage.list_tables().await.unwrap();
    assert!(
        !after.iter().any(|t| t.name == "t"),
        "the delete should have taken effect"
    );
}
