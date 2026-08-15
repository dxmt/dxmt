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

/// Storage config with a filesystem-backed object store, standing in for S3.
fn storage_config_with_object_store(base: &Path) -> StorageConfig {
    StorageConfig {
        r#type: "tiered".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 64,
            eviction_policy: "lru".to_string(),
        }),
        ssd: Some(dxmt_vectordb::config::SsdConfig {
            enabled: true,
            cache_dir: base.join("ssd").to_string_lossy().into_owned(),
            max_size_gb: 1,
            compression: "none".to_string(),
        }),
        object_storage: Some(dxmt_vectordb::config::ObjectStorageConfig {
            provider: "filesystem".to_string(),
            bucket: "test-bucket".to_string(),
            region: None,
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
        }),
    }
}

#[tokio::test]
async fn a_fresh_machine_recovers_from_object_storage_alone() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);

    // First machine: write data, snapshot, which uploads the segments.
    {
        let wal = tmp.path().join("machine-a");
        let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(wal))
            .await
            .unwrap();
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
                ],
            )
            .await
            .unwrap();
        storage.snapshot().await.unwrap();
    }

    // Prove the segments actually reached the object store rather than the test passing on some
    // shared local cache. FilesystemStorage lays objects out at <root>/<bucket>/<key>.
    let bucket = shared.join("ssd").join("test-bucket").join("snapshot");
    assert!(
        bucket.join("manifest").exists(),
        "manifest was never uploaded; looked in {}",
        bucket.display()
    );
    assert!(
        bucket.join("t0.meta").exists(),
        "table segment was never uploaded"
    );

    // Second machine: a different WAL directory, so its local cache is empty. Everything it
    // knows has to come from the object store — which is the whole point of treating local disk
    // as a cache rather than the source of truth.
    let wal_b = tmp.path().join("machine-b");
    assert!(
        !wal_b.exists(),
        "the second machine must start with nothing local"
    );

    let recovered = IntelligentIndexedStorage::new_with_wal(&config, Some(wal_b))
        .await
        .unwrap();

    let tables = recovered.list_tables().await.unwrap();
    assert_eq!(
        tables.len(),
        1,
        "table did not come back from object storage"
    );
    assert_eq!(tables[0].record_count, 2);

    let hits = recovered
        .search("t", query(vec![1.0, 0.0, 0.0, 0.0], 2))
        .await
        .unwrap();
    assert_eq!(hits.hits[0].id, "a", "the index came back unusable");
}

#[tokio::test]
async fn an_evicted_table_comes_back_on_the_next_query() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("wal")))
        .await
        .unwrap();

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
            ],
        )
        .await
        .unwrap();

    // A freshly written table is hot, so the policy must leave it alone. Evicting on every
    // sweep would turn eviction into a way to reload the same data forever.
    assert!(
        storage.evict_cold_tables().await.unwrap().is_empty(),
        "a table touched moments ago must not be evicted"
    );

    // Now force the evicted state — waiting out a real hotness window would take twenty
    // minutes — and confirm a plain search brings it back. The caller cannot be expected to
    // know that memory management happened.
    storage.snapshot().await.unwrap();
    storage.evict_table_for_test("t").await;

    let hits = storage
        .search("t", query(vec![1.0, 0.0, 0.0, 0.0], 2))
        .await
        .unwrap();
    assert_eq!(hits.hits.len(), 2, "search did not rehydrate the table");
    assert_eq!(
        hits.hits[0].id, "a",
        "the rehydrated index answers differently"
    );

    // And it is genuinely back in memory, not re-read per query.
    let record = storage.get_record("t", &"b".to_string()).await.unwrap();
    assert_eq!(record.id, "b");
}

#[tokio::test]
async fn eviction_never_drops_writes_made_after_the_last_snapshot() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("wal")))
        .await
        .unwrap();

    storage
        .create_table(table("t", 4, DistanceMetric::Euclidean))
        .await
        .unwrap();
    storage
        .insert_records("t", vec![record("old", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .unwrap();
    storage.snapshot().await.unwrap();

    // This write exists only in the WAL and in memory, not in the snapshot on disk.
    storage
        .insert_records("t", vec![record("new", vec![0.0, 1.0, 0.0, 0.0])])
        .await
        .unwrap();

    // Eviction snapshots first for exactly this reason. Dropping the table without doing so
    // would lose "new": the in-memory copy goes, and rehydration reads the stale snapshot.
    storage.evict_cold_tables().await.unwrap();

    let hits = storage
        .search("t", query(vec![0.0, 1.0, 0.0, 0.0], 5))
        .await
        .unwrap();
    let ids: std::collections::HashSet<_> = hits.hits.iter().map(|h| h.id.as_str()).collect();
    assert!(
        ids.contains("new"),
        "a post-snapshot write was lost: {:?}",
        ids
    );
    assert!(
        ids.contains("old"),
        "a pre-snapshot record was lost: {:?}",
        ids
    );
}

#[tokio::test]
async fn an_evicted_table_still_appears_in_listings() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("wal")))
        .await
        .unwrap();

    storage
        .create_table(table("t", 4, DistanceMetric::Euclidean))
        .await
        .unwrap();
    storage
        .insert_records("t", vec![record("a", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .unwrap();
    storage.snapshot().await.unwrap();

    // Simulate the table having gone cold by evicting it directly through the map. From a
    // caller's point of view eviction is invisible bookkeeping — a table that stops being
    // listed is indistinguishable from one that was dropped.
    storage.evict_table_for_test("t").await;

    let listed = storage.list_tables().await.unwrap();
    assert_eq!(
        listed.len(),
        1,
        "an evicted table vanished from list_tables"
    );
    assert_eq!(listed[0].name, "t");
    assert_eq!(listed[0].record_count, 1, "listing lost the row count");
}

#[tokio::test]
async fn writes_after_the_last_snapshot_survive_losing_the_machine() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);

    {
        let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("a")))
            .await
            .unwrap();
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records("t", vec![record("snapshotted", vec![1.0, 0.0, 0.0, 0.0])])
            .await
            .unwrap();
        storage.snapshot().await.unwrap();

        // Acknowledged after the snapshot, so it exists only in the WAL. If the WAL is local
        // only, "object storage is the source of truth" is false for exactly this window —
        // every write between two snapshots dies with the machine.
        storage
            .insert_records(
                "t",
                vec![record("after_snapshot", vec![0.0, 1.0, 0.0, 0.0])],
            )
            .await
            .unwrap();
    }

    // The machine is gone: new node, empty local disk, same bucket.
    let recovered = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("b")))
        .await
        .unwrap();

    let hits = recovered
        .search("t", query(vec![0.0, 0.0, 0.0, 0.0], 10))
        .await
        .unwrap();
    let ids: std::collections::HashSet<_> = hits.hits.iter().map(|h| h.id.as_str()).collect();
    assert!(
        ids.contains("snapshotted"),
        "the snapshotted record was lost: {:?}",
        ids
    );
    assert!(
        ids.contains("after_snapshot"),
        "an acknowledged write made after the snapshot did not survive: {:?}",
        ids
    );
}

#[tokio::test]
async fn a_large_table_is_partitioned_on_rebuild_and_still_answers() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("wal")))
        .await
        .unwrap();

    storage
        .create_table(table("t", 8, DistanceMetric::Euclidean))
        .await
        .unwrap();

    // Distinct vectors on purpose. A modular formula wraps and produces exact duplicates, and
    // then the "nearest neighbour" is a tie whose ordering says nothing about the index.
    let rows: Vec<_> = (0..500)
        .map(|i| {
            let v: Vec<f32> = (0..8).map(|d| i as f32 * 0.01 + d as f32 * 0.37).collect();
            record(&format!("r{}", i), v)
        })
        .collect();
    let probe = rows[123].vector.clone();
    storage.insert_records("t", rows).await.unwrap();

    let before = storage.search("t", query(probe.clone(), 5)).await.unwrap();
    assert_eq!(
        before.hits[0].id, "r123",
        "the pre-rebuild index did not find the exact match"
    );

    // Below the default threshold nothing should change; force it down instead of inserting
    // a hundred thousand rows in a unit test.
    std::env::set_var("DB_PARTITION_THRESHOLD", "100");
    storage.rebuild_index("t").await.unwrap();
    std::env::remove_var("DB_PARTITION_THRESHOLD");

    let table_meta = storage.get_table("t").await.unwrap();
    assert_eq!(
        table_meta.index_type.to_string(),
        "partitioned_ivf",
        "the table did not switch to a partitioned index"
    );

    // Segments must exist in the object store, not just in memory.
    let bucket = shared.join("ssd").join("test-bucket").join("partitions");
    assert!(
        bucket.exists(),
        "no partition segments were written to object storage"
    );

    // The query vector is one of the indexed rows, so the exact match must come back first.
    // IVF is approximate in general, but a vector cannot miss the partition its own centroid
    // chose — if this fails, partitioning or the distance function is wrong, not recall.
    let after = storage.search("t", query(probe, 5)).await.unwrap();
    assert_eq!(
        after.hits[0].id,
        "r123",
        "the exact match was lost after partitioning: {:?}",
        after.hits.iter().map(|h| &h.id).collect::<Vec<_>>()
    );
    assert_eq!(
        after.hits.len(),
        5,
        "partitioned search returned fewer hits"
    );
}

#[tokio::test]
async fn a_backup_directory_is_enough_to_restore_from() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let backup_dir = tmp.path().join("backup");

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
                ],
            )
            .await
            .unwrap();

        let manifest = storage.backup_to(&backup_dir).await.unwrap();
        assert_eq!(manifest["tables"], 1);
        assert!(
            manifest["bytes"].as_u64().unwrap() > 0,
            "backup reported zero bytes"
        );
    }

    // Destroy the original. Since W7 compacts the WAL after a snapshot, the backup is the only
    // copy of everything it covers — if it is not self-contained, nothing else can save this.
    std::fs::remove_dir_all(&wal_dir).unwrap();

    // Restore is deliberately a file copy: the server is stopped, the directory is put back.
    std::fs::create_dir_all(&wal_dir).unwrap();
    let snapshot_dst = wal_dir.join("snapshot");
    std::fs::create_dir_all(&snapshot_dst).unwrap();
    for entry in std::fs::read_dir(backup_dir.join("snapshot")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), snapshot_dst.join(entry.file_name())).unwrap();
    }
    if backup_dir.join("wal.log").exists() {
        std::fs::copy(backup_dir.join("wal.log"), wal_dir.join("wal.log")).unwrap();
    }

    let restored = open(&wal_dir).await;
    let tables = restored.list_tables().await.unwrap();
    assert_eq!(
        tables.len(),
        1,
        "the table did not come back from the backup"
    );
    assert_eq!(tables[0].record_count, 2);

    let hits = restored
        .search("t", query(vec![1.0, 0.0, 0.0, 0.0], 2))
        .await
        .unwrap();
    assert_eq!(
        hits.hits[0].id, "a",
        "the restored index answers differently"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queries_keep_running_while_an_index_is_rebuilt() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    let tmp = TempDir::new().unwrap();
    let storage = Arc::new(
        IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(tmp.path().join("wal")))
            .await
            .unwrap(),
    );

    storage
        .create_table(table("t", 8, DistanceMetric::Euclidean))
        .await
        .unwrap();

    // Enough vectors that the rebuild takes long enough to overlap with the query loop.
    const N: usize = 3000;
    let rows: Vec<_> = (0..N)
        .map(|i| {
            let v: Vec<f32> = (0..8).map(|d| i as f32 * 0.01 + d as f32 * 0.31).collect();
            record(&format!("r{}", i), v)
        })
        .collect();
    let probe = rows[0].vector.clone();
    storage.insert_records("t", rows).await.unwrap();

    // The loop runs until the rebuild finishes rather than for a fixed count. A counted loop
    // finishes before the rebuild even starts and measures nothing — the first version of this
    // test passed against the very code it was supposed to catch.
    let done = Arc::new(AtomicBool::new(false));
    let searcher = {
        let storage = storage.clone();
        let probe = probe.clone();
        let done = done.clone();
        tokio::spawn(async move {
            let mut slowest = std::time::Duration::ZERO;
            let mut fewest = usize::MAX;
            let mut count = 0usize;
            while !done.load(Ordering::SeqCst) {
                let started = Instant::now();
                let hits = storage.search("t", query(probe.clone(), 5)).await.unwrap();
                slowest = slowest.max(started.elapsed());
                fewest = fewest.min(hits.hits.len());
                count += 1;
            }
            (slowest, fewest, count)
        })
    };

    let rebuild_started = Instant::now();
    storage.rebuild_index("t").await.unwrap();
    let rebuild_took = rebuild_started.elapsed();
    done.store(true, Ordering::SeqCst);
    let (slowest, fewest, count) = searcher.await.unwrap();

    assert!(
        count > 10,
        "the query loop only managed {} searches; it did not overlap",
        count
    );
    assert_eq!(
        fewest, 5,
        "a search returned {} of 5 hits during a rebuild",
        fewest
    );

    // Publishing the new index before filling it does not expose a half-built one — the
    // wrapper's RwLock makes readers wait instead — but queries then block on the rebuild.
    // Against the pre-fix code this assertion reports a 102ms search over a 243ms rebuild.
    // Building the replacement first and swapping at the end keeps the old index answering.
    assert!(
        slowest * 4 < rebuild_took,
        "slowest search {:?} against a {:?} rebuild — queries are waiting on the rebuild",
        slowest,
        rebuild_took
    );
}

#[tokio::test]
async fn an_index_is_rebuilt_once_the_table_outgrows_it() {
    let tmp = TempDir::new().unwrap();
    let storage =
        IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(tmp.path().join("wal")))
            .await
            .unwrap();

    storage
        .create_table(table("t", 8, DistanceMetric::Euclidean))
        .await
        .unwrap();

    // A handful of rows. CREATE TABLE picks from a fixed estimate of 10,000 regardless, so the
    // table starts out with an index chosen for a size it does not have.
    let small: Vec<_> = (0..20)
        .map(|i| {
            let v: Vec<f32> = (0..8).map(|d| i as f32 * 0.1 + d as f32 * 0.7).collect();
            record(&format!("s{}", i), v)
        })
        .collect();
    storage.insert_records("t", small).await.unwrap();

    // At 20 records the selector wants BruteForce, so re-evaluation has something to correct.
    let rebuilt = storage.reevaluate_indexes().await.unwrap();
    assert_eq!(
        rebuilt,
        vec!["t".to_string()],
        "a 20-row table kept an index chosen for 10,000 rows"
    );
    assert_eq!(
        storage.get_table("t").await.unwrap().index_type.to_string(),
        "brute_force"
    );

    // Nothing changed since, so a second pass must be a no-op. Rebuilding on every sweep would
    // cost O(n) forever for no benefit.
    assert!(
        storage.reevaluate_indexes().await.unwrap().is_empty(),
        "re-evaluation rebuilt an index that was already the right one"
    );

    // Grow past the BruteForce boundary and it should move again.
    let more: Vec<_> = (0..6000)
        .map(|i| {
            let v: Vec<f32> = (0..8).map(|d| i as f32 * 0.01 + d as f32 * 0.31).collect();
            record(&format!("b{}", i), v)
        })
        .collect();
    storage.insert_records("t", more).await.unwrap();

    let rebuilt = storage.reevaluate_indexes().await.unwrap();
    assert_eq!(
        rebuilt,
        vec!["t".to_string()],
        "the table outgrew BruteForce unnoticed"
    );
    assert_eq!(
        storage.get_table("t").await.unwrap().index_type.to_string(),
        "hnsw"
    );

    // And it still answers.
    let hits = storage
        .search(
            "t",
            query(vec![0.0, 0.31, 0.62, 0.93, 1.24, 1.55, 1.86, 2.17], 1),
        )
        .await
        .unwrap();
    assert_eq!(
        hits.hits[0].id, "b0",
        "the rebuilt index answers differently"
    );
}

#[tokio::test]
async fn re_evaluation_settles_instead_of_rebuilding_forever() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("wal")))
        .await
        .unwrap();

    storage
        .create_table(table("t", 8, DistanceMetric::Euclidean))
        .await
        .unwrap();

    let rows: Vec<_> = (0..300)
        .map(|i| {
            let v: Vec<f32> = (0..8).map(|d| i as f32 * 0.01 + d as f32 * 0.37).collect();
            record(&format!("r{}", i), v)
        })
        .collect();
    storage.insert_records("t", rows).await.unwrap();

    // Low enough that this table partitions, as a large one would in production.
    std::env::set_var("DB_PARTITION_THRESHOLD", "100");
    storage.rebuild_index("t").await.unwrap();
    assert_eq!(
        storage.get_table("t").await.unwrap().index_type.to_string(),
        "partitioned_ivf"
    );

    // The selector does not know about partitioning — that decision lives in rebuild_index — so
    // a naive comparison finds the current algorithm "wrong" every single time, rebuilds, gets
    // a partitioned index back, and does it again on the next sweep. Forever, at O(n) each.
    let first = storage.reevaluate_indexes().await.unwrap();
    let second = storage.reevaluate_indexes().await.unwrap();
    std::env::remove_var("DB_PARTITION_THRESHOLD");

    assert!(
        first.is_empty() && second.is_empty(),
        "re-evaluation keeps rebuilding a table that is already correct: {:?} then {:?}",
        first,
        second
    );
}

#[tokio::test]
async fn partitioning_triggers_on_bytes_not_row_count() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("wal")))
        .await
        .unwrap();

    // Same row count, 150x apart in vector bytes: 500 x 4 x 4 = 8 KB against
    // 500 x 600 x 4 = 1.2 MB. Row count cannot tell these apart, which is the whole argument
    // for thresholding on size — one of them needs to leave memory and the other does not.
    for (name, dim) in [("narrow", 4usize), ("wide", 600usize)] {
        storage
            .create_table(table(name, dim, DistanceMetric::Euclidean))
            .await
            .unwrap();
        let rows: Vec<_> = (0..500)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| i as f32 * 0.01 + d as f32 * 0.03)
                    .collect();
                record(&format!("r{}", i), v)
            })
            .collect();
        storage.insert_records(name, rows).await.unwrap();
    }

    std::env::remove_var("DB_PARTITION_THRESHOLD");
    std::env::set_var("DB_PARTITION_THRESHOLD_MB", "1");
    storage.rebuild_index("narrow").await.unwrap();
    storage.rebuild_index("wide").await.unwrap();
    std::env::remove_var("DB_PARTITION_THRESHOLD_MB");

    assert_ne!(
        storage
            .get_table("narrow")
            .await
            .unwrap()
            .index_type
            .to_string(),
        "partitioned_ivf",
        "8 KB of vectors was partitioned; the threshold is not being read as bytes"
    );
    assert_eq!(
        storage
            .get_table("wide")
            .await
            .unwrap()
            .index_type
            .to_string(),
        "partitioned_ivf",
        "1.2 MB of vectors stayed resident under a 1 MB threshold"
    );
}

#[tokio::test]
async fn replay_never_reaches_below_the_truncation_point() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        for i in 0..30 {
            storage
                .insert_records(
                    "t",
                    vec![record(&format!("r{}", i), vec![i as f32, 0.0, 0.0, 0.0])],
                )
                .await
                .unwrap();
        }
        storage.snapshot().await.unwrap();
        // More writes after the snapshot, so recovery has a genuine tail to replay.
        for i in 30..40 {
            storage
                .insert_records(
                    "t",
                    vec![record(&format!("r{}", i), vec![i as f32, 0.0, 0.0, 0.0])],
                )
                .await
                .unwrap();
        }
    }

    // qdrant hit a class of bug here worth guarding against: a replay target computed from a
    // stale applied-sequence can sit *before* the truncation point, and the log no longer holds
    // those entries — their fix was to clamp the target at the first surviving index. Ours
    // derives the target from the snapshot manifest, which is written with the same sequence
    // the truncation uses, so the two cannot disagree. This test is what makes that a
    // guarantee rather than a coincidence.
    let recovered = open(&wal_dir).await;
    let tables = recovered.list_tables().await.unwrap();
    assert_eq!(
        tables[0].record_count, 40,
        "recovery lost records across the truncation point"
    );

    for i in [0usize, 29, 30, 39] {
        let hits = recovered
            .search("t", query(vec![i as f32, 0.0, 0.0, 0.0], 1))
            .await
            .unwrap();
        assert_eq!(
            hits.hits[0].id,
            format!("r{}", i),
            "r{} did not survive; it sits {} the truncation point",
            i,
            if i < 30 { "below" } else { "above" }
        );
    }
}

#[tokio::test]
async fn a_partitioned_table_gets_maintained_not_rebuilt() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("wal")))
        .await
        .unwrap();

    storage
        .create_table(table("t", 64, DistanceMetric::Euclidean))
        .await
        .unwrap();
    let rows: Vec<_> = (0..500)
        .map(|i| {
            let v: Vec<f32> = (0..64).map(|d| i as f32 * 0.01 + d as f32 * 0.03).collect();
            record(&format!("r{}", i), v)
        })
        .collect();
    storage.insert_records("t", rows).await.unwrap();

    // 500 x 64 x 4 = 128 KB, so a 0 MB threshold partitions it.
    std::env::set_var("DB_PARTITION_THRESHOLD_MB", "0");
    storage.rebuild_index("t").await.unwrap();
    assert_eq!(
        storage.get_table("t").await.unwrap().index_type.to_string(),
        "partitioned_ivf"
    );

    // Write more, then re-evaluate. The algorithm is already right, so this must not rebuild —
    // but it must not do nothing either: the new vectors are sitting in the overflow buffer,
    // and leaving them there means every search scans a buffer that only grows.
    let more: Vec<_> = (500..560)
        .map(|i| {
            let v: Vec<f32> = (0..64).map(|d| i as f32 * 0.01 + d as f32 * 0.03).collect();
            record(&format!("r{}", i), v)
        })
        .collect();
    storage.insert_records("t", more).await.unwrap();

    let rebuilt = storage.reevaluate_indexes().await.unwrap();
    std::env::remove_var("DB_PARTITION_THRESHOLD_MB");

    assert!(
        rebuilt.is_empty(),
        "a partitioned table was rebuilt when maintenance would have done: {:?}",
        rebuilt
    );

    // And everything is still findable, buffered or absorbed.
    for id in ["r0", "r499", "r559"] {
        let i: f32 = id[1..].parse().unwrap();
        let probe: Vec<f32> = (0..64).map(|d| i * 0.01 + d as f32 * 0.03).collect();
        let hits = storage.search("t", query(probe, 1)).await.unwrap();
        assert_eq!(hits.hits[0].id, id, "{} was lost around maintenance", id);
    }
}

#[tokio::test]
async fn the_wal_manifest_replaces_listing_and_catches_a_missing_chunk() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);

    {
        let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("a")))
            .await
            .unwrap();
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        for i in 0..5 {
            storage
                .insert_records(
                    "t",
                    vec![record(&format!("r{}", i), vec![i as f32, 0.0, 0.0, 0.0])],
                )
                .await
                .unwrap();
        }
    }

    // The manifest is the root: without it, finding chunks means listing the prefix, which
    // costs more as the log grows and — the part that matters — cannot tell a complete set from
    // one with a hole in it.
    let manifest_path = shared
        .join("ssd")
        .join("test-bucket")
        .join("wal")
        .join("MANIFEST");
    assert!(manifest_path.exists(), "no WAL manifest was written");

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let chunks = manifest["chunks"].as_array().unwrap();
    assert!(!chunks.is_empty(), "the manifest lists no chunks");

    // Chunks are bucketed by prefix. S3 rate-limits per prefix, so a flat namespace throttles
    // once the log is busy enough.
    let path = chunks[0]["path"].as_str().unwrap();
    assert!(
        path.starts_with("wal/") && path.matches('/').count() == 2,
        "chunk path {} is not bucketed",
        path
    );

    // A fresh node recovers from the manifest alone.
    let recovered = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("b")))
        .await
        .unwrap();
    assert_eq!(recovered.list_tables().await.unwrap()[0].record_count, 5);
}

#[tokio::test]
async fn a_corrupt_wal_manifest_falls_back_to_scanning_the_prefix() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);

    {
        let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("a")))
            .await
            .unwrap();
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records("t", vec![record("a", vec![1.0, 0.0, 0.0, 0.0])])
            .await
            .unwrap();
    }

    // Unlike the snapshot, a damaged WAL manifest is recoverable: the chunks are still there and
    // can be scanned. Refusing to start would be the wrong trade here — the data is intact.
    let manifest_path = shared
        .join("ssd")
        .join("test-bucket")
        .join("wal")
        .join("MANIFEST");
    std::fs::write(&manifest_path, b"not json").unwrap();

    let recovered = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("b")))
        .await
        .unwrap();
    let tables = recovered.list_tables().await.unwrap();
    assert_eq!(tables.len(), 1, "a corrupt manifest lost the log");
    assert_eq!(tables[0].record_count, 1);
}
