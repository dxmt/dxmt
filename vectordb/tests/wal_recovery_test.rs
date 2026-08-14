//! WAL crash-recovery tests.
//!
//! These replace `tests/integration/wal_recovery_test.rs`, which was disabled in
//! `tests/integration.rs` with the note "WAL configuration has been restructured". It targeted
//! a `StorageConfig` carrying `enable_wal` / `wal_dir` / `wal_batch_size` /
//! `wal_sync_interval_ms` / `wal_consistency_mode`, and a `WalConsistencyMode` type. None of
//! those exist any more, so there was nothing to re-enable — only something to rewrite.
//!
//! Each test drops the storage and builds a fresh one over the same directory. That is the
//! honest simulation of a crash for an in-memory store: the new process starts with nothing and
//! must reconstruct its state from the log alone.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use std::path::Path;
use tempfile::TempDir;

fn storage_config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 1024,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

async fn open(wal_dir: &Path) -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(wal_dir.to_path_buf()))
        .await
        .expect("failed to open storage")
}

fn table(name: &str, dimension: usize, metric: DistanceMetric) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension,
        distance_metric: metric,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    }
}

fn record(id: &str, vector: Vec<f32>) -> CreateRecordRequest {
    CreateRecordRequest {
        id: id.to_string(),
        vector,
        sparse_vector: None,
        named_vectors: None,
        payload: None,
    }
}

fn query(vector: Vec<f32>, top_k: usize) -> SearchQuery {
    SearchQuery {
        vector,
        top_k,
        limit: top_k,
        ..Default::default()
    }
}

#[tokio::test]
async fn records_survive_a_restart() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records(
                "t",
                vec![
                    record("a", vec![1.0, 0.0, 0.0, 0.0]),
                    record("b", vec![0.0, 1.0, 0.0, 0.0]),
                    record("c", vec![0.0, 0.0, 1.0, 0.0]),
                ],
            )
            .await
            .unwrap();
    }

    let recovered = open(&wal_dir).await;
    let tables = recovered.list_tables().await.unwrap();
    assert_eq!(tables.len(), 1, "the table itself should come back");
    assert_eq!(
        tables[0].record_count, 3,
        "record_count should be rebuilt, not left at 0"
    );

    // Searching, not just reading the map: a replay that populates `records` but skips the
    // index leaves the map correct and every query empty, and that is the failure mode worth
    // guarding against.
    let hits = recovered
        .search("t", query(vec![1.0, 0.0, 0.0, 0.0], 3))
        .await
        .unwrap();
    assert_eq!(
        hits.hits.len(),
        3,
        "the index must be repopulated, not just the record map"
    );
    assert_eq!(
        hits.hits[0].id, "a",
        "nearest neighbour should be unchanged by the restart"
    );
}

#[tokio::test]
async fn table_config_survives_a_restart() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 8, DistanceMetric::Euclidean))
            .await
            .unwrap();
    }

    let recovered = open(&wal_dir).await;
    let restored = recovered.get_table("t").await.unwrap();

    // The legacy WalEntry::CreateTable carried only { table, dimension }, so a restored table
    // silently fell back to cosine and answered distance queries wrongly rather than failing.
    // CreateTableWithConfig exists to keep this assertion true.
    assert_eq!(restored.distance_metric, DistanceMetric::Euclidean);
    assert_eq!(restored.dimension, 8);
}

#[tokio::test]
async fn deletes_are_replayed() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 2, DistanceMetric::Cosine))
            .await
            .unwrap();
        storage
            .insert_records(
                "t",
                vec![
                    record("keep", vec![1.0, 0.0]),
                    record("gone", vec![0.0, 1.0]),
                ],
            )
            .await
            .unwrap();
        storage
            .delete_records("t", vec!["gone".to_string()])
            .await
            .unwrap();
    }

    let recovered = open(&wal_dir).await;
    let hits = recovered
        .search("t", query(vec![1.0, 0.0], 10))
        .await
        .unwrap();
    let ids: Vec<_> = hits.hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["keep"],
        "a deleted record must not be resurrected by replay"
    );
}

#[tokio::test]
async fn dropped_tables_are_replayed() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("keep", 2, DistanceMetric::Cosine))
            .await
            .unwrap();
        storage
            .create_table(table("gone", 2, DistanceMetric::Cosine))
            .await
            .unwrap();
        storage.delete_table("gone").await.unwrap();
    }

    let recovered = open(&wal_dir).await;
    let names: Vec<_> = recovered
        .list_tables()
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(names, vec!["keep".to_string()]);
}

#[tokio::test]
async fn upsert_converges_to_the_last_write() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 2, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records("t", vec![record("x", vec![1.0, 0.0])])
            .await
            .unwrap();
        storage
            .insert_records("t", vec![record("x", vec![0.0, 1.0])])
            .await
            .unwrap();
    }

    // Two Insert entries share an id. There is no Update variant in the log: replay relies on
    // the same upsert path the live writes take, so the later entry has to win.
    let recovered = open(&wal_dir).await;
    assert_eq!(recovered.list_tables().await.unwrap()[0].record_count, 1);

    let hits = recovered
        .search("t", query(vec![0.0, 1.0], 1))
        .await
        .unwrap();
    assert_eq!(hits.hits.len(), 1);
    assert_eq!(hits.hits[0].id, "x");
    assert!(
        hits.hits[0].score > 0.99,
        "the surviving vector should be the second write, got score {}",
        hits.hits[0].score
    );
}

#[tokio::test]
async fn replay_is_idempotent() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 2, DistanceMetric::Cosine))
            .await
            .unwrap();
        storage
            .insert_records("t", vec![record("a", vec![1.0, 0.0])])
            .await
            .unwrap();
    }

    // Three restarts in a row. Replay must not append what it reads, or the log grows on every
    // boot and recovery time compounds until startup never finishes.
    let mut sizes = Vec::new();
    for _ in 0..3 {
        let recovered = open(&wal_dir).await;
        assert_eq!(recovered.list_tables().await.unwrap()[0].record_count, 1);
        drop(recovered);
        sizes.push(std::fs::metadata(wal_dir.join("wal.log")).unwrap().len());
    }

    assert_eq!(
        sizes[0], sizes[2],
        "WAL grew across restarts ({:?}) — replay is writing back what it reads",
        sizes
    );
}

#[tokio::test]
async fn without_a_wal_directory_nothing_is_persisted() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        // `new()` is the constructor the other ~40 call sites use. It must stay non-persistent,
        // otherwise every unit test and benchmark starts writing a log and replaying each
        // other's entries.
        let storage = IntelligentIndexedStorage::new(&storage_config())
            .await
            .unwrap();
        storage
            .create_table(table("t", 2, DistanceMetric::Cosine))
            .await
            .unwrap();
        storage
            .insert_records("t", vec![record("a", vec![1.0, 0.0])])
            .await
            .unwrap();
    }

    let recovered = open(&wal_dir).await;
    assert!(
        recovered.list_tables().await.unwrap().is_empty(),
        "new() must not write a WAL"
    );
}

#[tokio::test]
async fn a_snapshot_lets_recovery_skip_the_log_it_covers() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records(
                "t",
                (0..50)
                    .map(|i| record(&format!("r{}", i), vec![i as f32, 0.0, 0.0, 0.0]))
                    .collect(),
            )
            .await
            .unwrap();
        storage.snapshot().await.unwrap();
    }

    let recovered = open(&wal_dir).await;
    assert_eq!(recovered.list_tables().await.unwrap()[0].record_count, 50);

    let hits = recovered
        .search("t", query(vec![7.0, 0.0, 0.0, 0.0], 1))
        .await
        .unwrap();
    assert_eq!(
        hits.hits[0].id, "r7",
        "the restored index must answer, not just exist"
    );
}

#[tokio::test]
async fn writes_after_a_snapshot_are_replayed_on_top_of_it() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records(
                "t",
                vec![
                    record("keep", vec![1.0, 0.0, 0.0, 0.0]),
                    record("gone", vec![2.0, 0.0, 0.0, 0.0]),
                ],
            )
            .await
            .unwrap();
        storage.snapshot().await.unwrap();

        // Both edits land after the snapshot, so they exist only in the WAL tail. If recovery
        // stopped at the snapshot, `late` would vanish and `gone` would come back — and the
        // record count alone would not reveal it, since the two errors cancel out.
        storage
            .insert_records("t", vec![record("late", vec![3.0, 0.0, 0.0, 0.0])])
            .await
            .unwrap();
        storage
            .delete_records("t", vec!["gone".to_string()])
            .await
            .unwrap();
    }

    let recovered = open(&wal_dir).await;
    let hits = recovered
        .search("t", query(vec![0.0, 0.0, 0.0, 0.0], 10))
        .await
        .unwrap();
    let mut ids: Vec<_> = hits.hits.iter().map(|h| h.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec!["keep", "late"]);
}

#[tokio::test]
async fn a_corrupt_snapshot_is_refused_rather_than_silently_ignored() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 2, DistanceMetric::Cosine))
            .await
            .unwrap();
        storage
            .insert_records("t", vec![record("a", vec![1.0, 0.0])])
            .await
            .unwrap();
        storage.snapshot().await.unwrap();
    }

    std::fs::write(wal_dir.join("snapshot").join("manifest"), b"garbage").unwrap();

    // A verified snapshot lets the WAL be compacted, which makes it the only copy of the
    // entries it covers. Falling back to "replay whatever is left" would therefore come up
    // with an empty database and a warning nobody reads. Refusing to start is the only
    // honest response.
    let msg = match IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(wal_dir)).await
    {
        Ok(_) => panic!("a corrupt snapshot must not be silently ignored"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("unreadable"), "unhelpful error: {}", msg);
    assert!(
        msg.contains("lose data"),
        "the error must say what is at stake: {}",
        msg
    );
}

#[tokio::test]
async fn a_snapshot_that_cannot_be_read_back_does_not_compact_the_log() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    let storage = open(&wal_dir).await;
    storage
        .create_table(table("t", 2, DistanceMetric::Cosine))
        .await
        .unwrap();
    storage
        .insert_records("t", vec![record("a", vec![1.0, 0.0])])
        .await
        .unwrap();

    let before = std::fs::metadata(wal_dir.join("wal.log")).unwrap().len();
    storage.snapshot().await.unwrap();
    let after = std::fs::metadata(wal_dir.join("wal.log")).unwrap().len();

    // Compaction is the point of the snapshot; without it the log grows forever.
    assert!(
        after < before,
        "WAL was not compacted after a successful snapshot: {} -> {}",
        before,
        after
    );

    // And the database still comes back.
    drop(storage);
    let recovered = open(&wal_dir).await;
    assert_eq!(recovered.list_tables().await.unwrap()[0].record_count, 1);
}
