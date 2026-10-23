//! WAL crash-recovery tests.
//!
//! These replace `tests/integration/wal_recovery_test.rs`, which was disabled in
//! `tests/integration.rs` with the note "WAL configuration has been restructured". It targeted
//! a `StorageConfig` carrying `enable_wal` / `wal_dir` / `wal_batch_size` /
//! `wal_sync_interval_ms` / `wal_consistency_mode`, and a `WalConsistencyMode` type. None of
//! those exist any more, so there was nothing to re-enable — only something to rewrite.
//!
//! Several of these tests set `DXMT_PARTITION_THRESHOLD_MB` and friends to force a decision.
//! Environment variables are process-global while cargo runs a binary's tests in parallel
//! threads, so those tests carry `#[serial_test::serial]` — without it one test's threshold
//! silently changes another's outcome, which is the kind of flake that only shows up under a
//! loaded machine.
//!
//! Each test drops the storage and builds a fresh one over the same directory. That is the
//! honest simulation of a crash for an in-memory store: the new process starts with nothing and
//! must reconstruct its state from the log alone.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{
            Condition, CreateRecordRequest, Filter, FilterOperation, FusionMethod,
            HybridSearchRequest, SearchQuery,
        },
        sparse_vector::SparseVector,
        table::{FieldDef, FieldType, TableAlterRequest, TableConfig, VectorSpaceConfig},
        vector::DistanceMetric,
    },
};
use std::collections::HashMap;
use std::path::Path;
use tempfile::TempDir;

fn storage_config() -> StorageConfig {
    StorageConfig {
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

async fn assert_upserted_secondary_indexes(storage: &IntelligentIndexedStorage) {
    let record = storage.get_record("rich", &"x".to_string()).await.unwrap();
    assert_eq!(record.version, 2, "upsert version was not restored");
    assert_eq!(record.payload["tag"], "new");
    assert_eq!(record.sparse_vector.as_ref().unwrap().indices, vec![1]);

    let old_payload = Filter {
        must: Some(vec![Condition {
            key: "tag".to_string(),
            operation: FilterOperation::Equal(serde_json::json!("old")),
        }]),
        must_not: None,
        should: None,
    };
    let old_payload_hits = storage
        .search(
            "rich",
            SearchQuery {
                vector: vec![10.0, 0.0],
                top_k: 5,
                limit: 5,
                filter: Some(old_payload),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(
        old_payload_hits.hits.is_empty(),
        "the old payload value remained in the pre-filter index"
    );

    let old_sparse_hits = storage
        .hybrid_search(
            "rich",
            HybridSearchRequest {
                vector: None,
                sparse_vector: Some(SparseVector::new(vec![0], vec![1.0], 4).unwrap()),
                text: None,
                vector_weight: 1.0,
                fusion_method: Some(FusionMethod::Dbsf),
                rrf_k: 60.0,
                limit: 1,
                filter: None,
                with_payload: false,
                with_vector: false,
                rerank: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(
        old_sparse_hits.hits.first().map(|h| h.id.as_str()),
        Some("y"),
        "the upserted record remained in its old sparse posting list"
    );

    let new_sparse_hits = storage
        .hybrid_search(
            "rich",
            HybridSearchRequest {
                vector: None,
                sparse_vector: Some(SparseVector::new(vec![1], vec![1.0], 4).unwrap()),
                text: None,
                vector_weight: 1.0,
                fusion_method: Some(FusionMethod::Dbsf),
                rrf_k: 60.0,
                limit: 1,
                filter: None,
                with_payload: false,
                with_vector: false,
                rerank: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(
        new_sparse_hits.hits.first().map(|h| h.id.as_str()),
        Some("x")
    );

    let old_named_hits = storage
        .search(
            "rich",
            SearchQuery {
                vector: vec![0.0, 0.0],
                top_k: 1,
                limit: 1,
                vector_name: Some("aux".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        old_named_hits.hits.first().map(|h| h.id.as_str()),
        Some("y"),
        "the old named vector remained searchable after replacement"
    );

    let new_named_hits = storage
        .search(
            "rich",
            SearchQuery {
                vector: vec![10.0, 0.0],
                top_k: 1,
                limit: 1,
                vector_name: Some("aux".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        new_named_hits.hits.first().map(|h| h.id.as_str()),
        Some("x")
    );
}

#[tokio::test]
async fn upsert_replaces_secondary_indexes_and_survives_restart() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    let created_at = {
        let storage = open(&wal_dir).await;
        let mut config = table("rich", 2, DistanceMetric::Euclidean);
        config.named_vectors = Some(vec![VectorSpaceConfig {
            name: "aux".to_string(),
            dimension: 2,
            distance_metric: DistanceMetric::Euclidean,
        }]);
        storage.create_table(config).await.unwrap();

        let mut x_old_named = HashMap::new();
        x_old_named.insert("aux".to_string(), vec![0.0, 0.0]);
        let mut y_named = HashMap::new();
        y_named.insert("aux".to_string(), vec![0.1, 0.0]);

        let mut records = vec![
            CreateRecordRequest {
                id: "x".to_string(),
                vector: vec![0.0, 0.0],
                sparse_vector: Some(SparseVector::new(vec![0], vec![2.0], 4).unwrap()),
                named_vectors: Some(x_old_named),
                payload: Some(serde_json::json!({"tag": "old"})),
                text: None,
            },
            CreateRecordRequest {
                id: "y".to_string(),
                vector: vec![0.1, 0.0],
                sparse_vector: Some(SparseVector::new(vec![0], vec![1.0], 4).unwrap()),
                named_vectors: Some(y_named),
                payload: Some(serde_json::json!({"tag": "keep"})),
                text: None,
            },
        ];
        // More than twenty rows force a one-row stale payload posting into the exact
        // pre-filter branch, where it would otherwise bypass record-level post-filtering.
        records.extend((0..24).map(|i| record(&format!("filler-{i}"), vec![20.0 + i as f32, 0.0])));
        storage.insert_records("rich", records).await.unwrap();

        let mut x_new_named = HashMap::new();
        x_new_named.insert("aux".to_string(), vec![10.0, 0.0]);
        storage
            .insert_records(
                "rich",
                vec![CreateRecordRequest {
                    id: "x".to_string(),
                    vector: vec![10.0, 0.0],
                    sparse_vector: Some(SparseVector::new(vec![1], vec![2.0], 4).unwrap()),
                    named_vectors: Some(x_new_named),
                    payload: Some(serde_json::json!({"tag": "new"})),
                    text: None,
                }],
            )
            .await
            .unwrap();

        assert_upserted_secondary_indexes(&storage).await;
        storage
            .get_record("rich", &"x".to_string())
            .await
            .unwrap()
            .created_at
    };

    let recovered = open(&wal_dir).await;
    assert_upserted_secondary_indexes(&recovered).await;
    assert_eq!(
        recovered
            .get_record("rich", &"x".to_string())
            .await
            .unwrap()
            .created_at,
        created_at,
        "current WAL entries must preserve record timestamps exactly"
    );
}

#[tokio::test]
async fn table_alterations_survive_a_restart() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("schema", 2, DistanceMetric::Cosine))
            .await
            .unwrap();
        storage
            .alter_table(
                "schema",
                TableAlterRequest {
                    description: Some("changed".to_string()),
                    memory_optimization: Some(false),
                    enable_full_text_search: None,
                    add_fields: vec![FieldDef {
                        name: "tenant".to_string(),
                        field_type: FieldType::String,
                        nullable: false,
                        default_value: None,
                    }],
                    drop_fields: Vec::new(),
                    payload_index: None,
                    embedding: None,
                    rerank: None,
                    recall_target: None,
                    text_fields: None,
                },
            )
            .await
            .unwrap();
    }

    let recovered = open(&wal_dir).await;
    let restored = recovered.get_table("schema").await.unwrap();
    assert_eq!(restored.description.as_deref(), Some("changed"));
    assert_eq!(restored.memory_optimization, Some(false));
    assert_eq!(restored.schema_version, 2);
    let fields = restored.payload_schema.unwrap();
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].name, "tenant");
    assert_eq!(fields[0].field_type, FieldType::String);
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

/// A local WAL sync failure is a process-level durability fault, not a transient per-request
/// validation error. The first mutation is not acknowledged; subsequent mutations are rejected
/// until restart so callers never mistake an in-memory write for durable state.
#[cfg(feature = "test-failpoints")]
#[tokio::test]
async fn wal_sync_failure_enters_read_only_mode() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let storage = open(&wal_dir).await;
    storage
        .create_table(table("t", 2, DistanceMetric::Cosine))
        .await
        .unwrap();

    storage.fail_next_wal_sync_for_test().unwrap();
    let first = storage
        .insert_records("t", vec![record("unacknowledged", vec![1.0, 0.0])])
        .await
        .unwrap_err();
    assert!(matches!(
        &first,
        dxmt_vectordb::Error::ServiceUnavailable { .. }
    ));
    assert_eq!(first.status_code(), 503);
    assert!(first.is_retryable());

    let second = storage
        .insert_records("t", vec![record("rejected", vec![0.0, 1.0])])
        .await
        .unwrap_err();
    assert!(matches!(
        second,
        dxmt_vectordb::Error::ServiceUnavailable { .. }
    ));

    // The failed mutation may already have reached memory before the filesystem reported the
    // error. It was not acknowledged, so callers must not rely on it surviving a restart.
    let listed = storage.list_tables().await.unwrap();
    assert_eq!(listed[0].record_count, 1);

    let snapshot = storage.snapshot().await.unwrap_err();
    assert!(matches!(
        snapshot,
        dxmt_vectordb::Error::ServiceUnavailable { .. }
    ));
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

/// Session one snapshots, which compacts the log to nothing. Session two writes. Session
/// three must see that write.
///
/// It did not: an empty log opens with its sequence counter at 0, the write in session two
/// was numbered 0, and session three replayed from the snapshot's sequence and skipped it.
/// The record was gone from the index, and from the record map too unless a row file held it.
#[tokio::test]
async fn a_write_in_the_session_after_a_snapshot_survives_the_next_restart() {
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
                ],
            )
            .await
            .unwrap();
        storage.snapshot().await.unwrap();
    }

    {
        let storage = open(&wal_dir).await;
        storage
            .insert_records("t", vec![record("c", vec![0.0, 0.0, 1.0, 0.0])])
            .await
            .unwrap();
        let hits = storage
            .search("t", query(vec![0.0, 0.0, 1.0, 0.0], 1))
            .await
            .unwrap();
        assert_eq!(hits.hits[0].id, "c", "in the writing session");
    }

    let storage = open(&wal_dir).await;
    assert_eq!(
        storage.get_table("t").await.unwrap().record_count,
        3,
        "the write after the snapshot did not survive a second restart"
    );
    let hits = storage
        .search("t", query(vec![0.0, 0.0, 1.0, 0.0], 1))
        .await
        .unwrap();
    assert_eq!(hits.hits[0].id, "c", "after the second restart");
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
        memory: Some(MemoryConfig {
            max_size_mb: 64,
            eviction_policy: "lru".to_string(),
        }),
        ssd: Some(dxmt_vectordb::config::SsdConfig {
            enabled: true,
            cache_dir: base.join("ssd").to_string_lossy().into_owned(),
            max_size_gb: 1,
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

#[serial_test::serial]
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
    std::env::set_var("DXMT_PARTITION_THRESHOLD", "100");
    storage.rebuild_index("t").await.unwrap();
    std::env::remove_var("DXMT_PARTITION_THRESHOLD");

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

    // Destroy the original. Since the WAL is compacted after a snapshot, the backup is the only
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
    // The row files, which hold the records and vectors of every table since those left
    // the snapshot.
    for sub in ["records", "vectors"] {
        let src = backup_dir.join(sub);
        if !src.exists() {
            continue;
        }
        let dst = wal_dir.join(sub);
        std::fs::create_dir_all(&dst).unwrap();
        for entry in std::fs::read_dir(&src).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), dst.join(entry.file_name())).unwrap();
        }
    }
    assert!(
        manifest_lists_row_files(&backup_dir),
        "the backup manifest does not list the row files"
    );

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

#[tokio::test]
async fn backup_rejects_the_wal_directory_and_its_children() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let storage = open(&wal_dir).await;

    let same = storage.backup_to(&wal_dir).await.unwrap_err();
    assert!(matches!(same, dxmt_vectordb::Error::InvalidInput(_)));

    let nested = storage
        .backup_to(&wal_dir.join("backups/latest"))
        .await
        .unwrap_err();
    assert!(matches!(nested, dxmt_vectordb::Error::InvalidInput(_)));

    // The guard runs before snapshotting, so a rejected destination must not leave staging
    // directories or otherwise mutate the source WAL tree.
    assert!(!wal_dir.join("snapshot").exists());
    assert!(!wal_dir.join("backups").exists());
}

#[tokio::test]
async fn health_snapshot_reports_wal_and_snapshot_progress() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let storage = open(&wal_dir).await;

    let initial = storage.health_snapshot().await;
    assert!(initial.last_wal_sync_at.is_none());
    assert!(initial.last_snapshot_at.is_none());
    assert_eq!(initial.snapshot_failures, 0);

    storage
        .create_table(table("t", 2, DistanceMetric::Cosine))
        .await
        .unwrap();
    let after_write = storage.health_snapshot().await;
    assert!(after_write.last_wal_sync_at.is_some());
    assert!(after_write.last_snapshot_at.is_none());

    storage.snapshot().await.unwrap();
    let after_snapshot = storage.health_snapshot().await;
    assert!(after_snapshot.last_wal_sync_at.is_some());
    assert!(after_snapshot.last_snapshot_at.is_some());
    assert_eq!(after_snapshot.snapshot_failures, 0);
    assert!(after_snapshot.last_snapshot_error.is_none());
}

#[tokio::test]
async fn failed_snapshot_is_visible_without_marking_storage_unready() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let storage = open(&wal_dir).await;
    storage
        .create_table(table("t", 2, DistanceMetric::Cosine))
        .await
        .unwrap();

    // Block the staging directory with a regular file. Snapshot installation has not started,
    // so the existing WAL remains a valid recovery source and readiness must stay green.
    std::fs::write(wal_dir.join("snapshot.staging"), b"blocked").unwrap();
    assert!(storage.snapshot().await.is_err());

    let failed = storage.health_snapshot().await;
    assert!(failed.ready);
    assert_eq!(failed.snapshot_failures, 1);
    assert!(failed.last_snapshot_at.is_none());
    assert!(failed.last_snapshot_error.is_some());

    std::fs::remove_file(wal_dir.join("snapshot.staging")).unwrap();
    storage.snapshot().await.unwrap();
    let recovered = storage.health_snapshot().await;
    assert!(recovered.ready);
    assert_eq!(recovered.snapshot_failures, 1);
    assert!(recovered.last_snapshot_at.is_some());
    assert!(recovered.last_snapshot_error.is_none());
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

    // The size-based selector: at 3,000 rows it wants BruteForce where CREATE TABLE installed
    // HNSW, so the rebuild below has work to do. Under the HNSW-PQ default a 3,000-row table
    // is already what a rebuild would produce and `rebuild_index` returns at once.
    let mut sized = table("t", 8, DistanceMetric::Euclidean);
    sized.memory_optimization = Some(false);
    storage.create_table(sized).await.unwrap();

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
    let baseline_started = Instant::now();
    let _ = storage.search("t", query(probe.clone(), 5)).await.unwrap();
    let baseline = baseline_started.elapsed();
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
    // The count/fewest assertions above detect a full stall. This latency check catches a
    // pathological regression while allowing for slow CI runners whose brute-force baseline is
    // already much higher than a developer laptop's.
    assert!(
        slowest <= baseline.saturating_mul(8) + std::time::Duration::from_millis(500),
        "slowest search {:?} exceeded baseline {:?} while rebuilding in {:?}",
        slowest,
        baseline,
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

    // The size-based selector, explicitly: the memory-optimized default answers HNSW-PQ at
    // every size, so it has no boundary for a table to outgrow.
    let mut sized = table("t", 8, DistanceMetric::Euclidean);
    sized.memory_optimization = Some(false);
    storage.create_table(sized).await.unwrap();

    // A handful of rows. CREATE TABLE picks from a fixed estimate of 10,000 regardless, so the
    // table starts out with an index chosen for a size it does not have.
    let small: Vec<_> = (0..20)
        .map(|i| {
            let v: Vec<f32> = (0..8).map(|d| i as f32 * 0.1 + d as f32 * 0.7).collect();
            record(&format!("s{}", i), v)
        })
        .collect();
    storage.insert_records("t", small).await.unwrap();

    // At 20 records the selector wants BruteForce -- and the sweep leaves the table alone
    // anyway, because it only ever climbs. This used to assert the demotion, and the demotion
    // is what a soak caught costing two full rebuilds on a table that was merely growing: it
    // went HNSW -> BruteForce at 1,000 rows and back at 38,400 ten seconds later. An oversized
    // graph on a 20-row table costs memory nobody can measure; a rebuild each way costs the
    // table. The correction is still available on demand, below.
    assert!(
        storage.reevaluate_indexes().await.unwrap().is_empty(),
        "the sweep demoted a small table instead of leaving its oversized index alone"
    );
    assert_eq!(
        storage.get_table("t").await.unwrap().index_type.to_string(),
        "hnsw",
        "the index chosen from the create-time estimate should still be there"
    );

    // On demand, an explicit rebuild re-selects from scratch and does demote it -- the escape
    // hatch the sweep's one-way rule leaves open.
    storage.rebuild_index("t").await.unwrap();
    assert_eq!(
        storage.get_table("t").await.unwrap().index_type.to_string(),
        "brute_force",
        "an explicit rebuild did not re-select for the size the table actually has"
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

#[serial_test::serial]
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
    std::env::set_var("DXMT_PARTITION_THRESHOLD", "100");
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
    std::env::remove_var("DXMT_PARTITION_THRESHOLD");

    assert!(
        first.is_empty() && second.is_empty(),
        "re-evaluation keeps rebuilding a table that is already correct: {:?} then {:?}",
        first,
        second
    );
}

#[serial_test::serial]
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

    std::env::remove_var("DXMT_PARTITION_THRESHOLD");
    std::env::set_var("DXMT_PARTITION_THRESHOLD_MB", "1");
    storage.rebuild_index("narrow").await.unwrap();
    storage.rebuild_index("wide").await.unwrap();
    std::env::remove_var("DXMT_PARTITION_THRESHOLD_MB");

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

    // A class of bug worth guarding against: a replay target computed from a stale
    // applied-sequence can sit *before* the truncation point, where the log no longer holds
    // those entries. One fix is to clamp the target at the first surviving index. Ours derives
    // the target from the snapshot manifest, which is written with the same sequence the
    // truncation uses, so the two cannot disagree. This test is what makes that a guarantee
    // rather than a coincidence.
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

#[serial_test::serial]
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
    std::env::set_var("DXMT_PARTITION_THRESHOLD_MB", "0");
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
    std::env::remove_var("DXMT_PARTITION_THRESHOLD_MB");

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
#[tokio::test]
async fn table_stats_say_whether_the_table_was_in_memory() {
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

    assert_eq!(
        storage.get_table_stats("t").await.unwrap()["resident"],
        true
    );

    storage.snapshot().await.unwrap();
    storage.evict_table_for_test("t").await;

    // Rehydration is transparent, which is convenient and hides a latency cliff: without this
    // field a caller cannot tell that the next query will pay for a fetch from object storage.
    let stats = storage.get_table_stats("t").await.unwrap();
    assert_eq!(
        stats["resident"], false,
        "an evicted table reported itself resident"
    );

    // And asking must not be what makes it true — the check has to happen before the
    // rehydration that answering the rest of the call performs.
    assert_eq!(stats["record_count"], 1);
    assert_eq!(
        storage.get_table_stats("t").await.unwrap()["resident"],
        true
    );
}

#[tokio::test]
async fn an_evicted_table_survives_a_later_snapshot_and_restart() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let wal_dir = tmp.path().join("wal");

    {
        let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(wal_dir.clone()))
            .await
            .unwrap();
        storage
            .create_table(table("cold", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records("cold", vec![record("a", vec![1.0, 0.0, 0.0, 0.0])])
            .await
            .unwrap();
        storage.snapshot().await.unwrap();

        // Evict it, as the sweep would once it went cold.
        storage.evict_table_for_test("cold").await;

        // Then a later snapshot runs — the periodic one, or a shutdown. Snapshots write what is
        // in memory, and an evicted table is not. If it drops out of the manifest here, the
        // table is gone at the next restart even though nobody deleted it.
        storage.snapshot().await.unwrap();
    }

    let recovered = IntelligentIndexedStorage::new_with_wal(&config, Some(wal_dir))
        .await
        .unwrap();
    let names: Vec<_> = recovered
        .list_tables()
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert!(
        names.contains(&"cold".to_string()),
        "an evicted table vanished after a later snapshot: {:?}",
        names
    );

    let hits = recovered
        .search("cold", query(vec![1.0, 0.0, 0.0, 0.0], 1))
        .await
        .unwrap();
    assert_eq!(hits.hits[0].id, "a", "the evicted table came back empty");
}

#[tokio::test]
#[serial_test::serial]
async fn dropping_a_partitioned_table_removes_its_segments() {
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
    let rows: Vec<_> = (0..300)
        .map(|i| {
            let v: Vec<f32> = (0..64).map(|d| i as f32 * 0.01 + d as f32 * 0.03).collect();
            record(&format!("r{}", i), v)
        })
        .collect();
    storage.insert_records("t", rows).await.unwrap();

    std::env::set_var("DXMT_PARTITION_THRESHOLD_MB", "0");
    storage.rebuild_index("t").await.unwrap();
    std::env::remove_var("DXMT_PARTITION_THRESHOLD_MB");

    let segments = shared.join("ssd").join("test-bucket").join("partitions");
    assert!(segments.exists(), "no segments were written");

    // Dropping the table used to leave every segment behind forever: delete_table touched the
    // in-memory map and the log and nothing else.
    storage.delete_table("t").await.unwrap();

    let left: Vec<_> = walk_files(&segments);
    assert!(
        left.is_empty(),
        "dropping the table left {} segment(s) behind: {:?}",
        left.len(),
        left
    );
}

fn manifest_lists_row_files(backup_dir: &Path) -> bool {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(backup_dir.join("BACKUP.json")).unwrap()).unwrap();
    let files = manifest["files"].as_array().unwrap();
    let has = |suffix: &str| files.iter().any(|f| f.as_str().unwrap().ends_with(suffix));
    has("records/t.rec")
        && has("records/t.rec.log")
        && has("vectors/t.f32")
        && has("vectors/t.f32.rows")
}

fn walk_files(dir: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(walk_files(&path));
            } else {
                out.push(path.display().to_string());
            }
        }
    }
    out
}

#[tokio::test]
#[serial_test::serial]
async fn a_selective_filter_works_on_a_partitioned_table() {
    use dxmt_vectordb::types::record::{Condition, Filter, FilterOperation};

    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("wal")))
        .await
        .unwrap();

    storage
        .create_table(table("t", 32, DistanceMetric::Euclidean))
        .await
        .unwrap();

    // One record in fifty carries the rare tag, so a filter on it is ~2% selective — the range
    // where the strategy switches away from a plain post-filter.
    let rows: Vec<_> = (0..1000)
        .map(|i| {
            let v: Vec<f32> = (0..32).map(|d| i as f32 * 0.01 + d as f32 * 0.03).collect();
            CreateRecordRequest {
                id: format!("r{}", i),
                vector: v,
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::json!({
                    "tag": if i % 50 == 0 { "rare" } else { "common" }
                })),
                text: None,
            }
        })
        .collect();
    storage.insert_records("t", rows).await.unwrap();

    let filter = Filter {
        must: Some(vec![Condition {
            key: "tag".to_string(),
            operation: FilterOperation::Equal(serde_json::json!("rare")),
        }]),
        must_not: None,
        should: None,
    };

    let filtered_query = |v: Vec<f32>| SearchQuery {
        vector: v,
        top_k: 5,
        limit: 5,
        filter: Some(filter.clone()),
        with_payload: true,
        ..Default::default()
    };

    let probe: Vec<f32> = (0..32).map(|d| 5.0 + d as f32 * 0.03).collect();
    let before = storage
        .search("t", filtered_query(probe.clone()))
        .await
        .unwrap();
    assert_eq!(
        before.hits.len(),
        5,
        "the filter found nothing before partitioning"
    );

    // Now partition the table and ask again. The partitioned index does not implement in-graph
    // filtering, so this exercises whatever the storage layer falls back to — a combination
    // nothing else covers.
    std::env::set_var("DXMT_PARTITION_THRESHOLD_MB", "0");
    storage.rebuild_index("t").await.unwrap();
    std::env::remove_var("DXMT_PARTITION_THRESHOLD_MB");
    assert_eq!(
        storage.get_table("t").await.unwrap().index_type.to_string(),
        "partitioned_ivf"
    );

    let after = storage.search("t", filtered_query(probe)).await.unwrap();
    assert_eq!(
        after.hits.len(),
        5,
        "a selective filter returned {} of 5 hits on a partitioned table",
        after.hits.len()
    );
    for hit in &after.hits {
        assert_eq!(
            hit.payload
                .as_ref()
                .and_then(|p| p.get("tag"))
                .and_then(|t| t.as_str()),
            Some("rare"),
            "the filter let a non-matching record through"
        );
    }
    let before_ids: Vec<_> = before.hits.iter().map(|h| &h.id).collect();
    let after_ids: Vec<_> = after.hits.iter().map(|h| &h.id).collect();
    assert_eq!(
        before_ids, after_ids,
        "partitioning changed the filtered answer"
    );
}

#[tokio::test]
#[serial_test::serial]
async fn a_partitioned_table_is_still_partitioned_after_a_restart() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let wal_dir = tmp.path().join("wal");

    {
        let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(wal_dir.clone()))
            .await
            .unwrap();
        storage
            .create_table(table("t", 64, DistanceMetric::Euclidean))
            .await
            .unwrap();
        let rows: Vec<_> = (0..400)
            .map(|i| {
                let v: Vec<f32> = (0..64).map(|d| i as f32 * 0.01 + d as f32 * 0.03).collect();
                record(&format!("r{}", i), v)
            })
            .collect();
        storage.insert_records("t", rows).await.unwrap();

        std::env::set_var("DXMT_PARTITION_THRESHOLD_MB", "0");
        storage.rebuild_index("t").await.unwrap();
        std::env::remove_var("DXMT_PARTITION_THRESHOLD_MB");
        assert_eq!(
            storage.get_table("t").await.unwrap().index_type.to_string(),
            "partitioned_ivf"
        );
        storage.snapshot().await.unwrap();
    }

    // Restoring only knew how to rebuild HNSW and HNSW-PQ from their segments; anything else
    // fell through to "re-insert every record", which silently downgrades a partitioned table
    // to whatever a zero-row table would have been given, orphans its segments, and pays the
    // full rebuild cost that snapshots exist to avoid.
    let recovered = IntelligentIndexedStorage::new_with_wal(&config, Some(wal_dir))
        .await
        .unwrap();

    assert_eq!(
        recovered
            .get_table("t")
            .await
            .unwrap()
            .index_type
            .to_string(),
        "partitioned_ivf",
        "a partitioned table came back as something else"
    );

    let probe: Vec<f32> = (0..64).map(|d| 123.0 * 0.01 + d as f32 * 0.03).collect();
    let hits = recovered.search("t", query(probe, 1)).await.unwrap();
    assert_eq!(
        hits.hits[0].id, "r123",
        "the restored partitioned index answers differently"
    );
}

#[tokio::test]
#[serial_test::serial]
async fn maintenance_after_a_snapshot_does_not_strand_that_snapshot() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let wal_dir = tmp.path().join("wal");

    {
        let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(wal_dir.clone()))
            .await
            .unwrap();
        storage
            .create_table(table("t", 64, DistanceMetric::Euclidean))
            .await
            .unwrap();
        let rows: Vec<_> = (0..600)
            .map(|i| {
                let v: Vec<f32> = (0..64).map(|d| i as f32 * 0.01 + d as f32 * 0.03).collect();
                record(&format!("r{}", i), v)
            })
            .collect();
        storage.insert_records("t", rows).await.unwrap();

        // Held for the whole block: without it re-evaluation decides the table is too small to
        // partition and rebuilds it as HNSW, so maintenance never runs and this tests nothing.
        std::env::set_var("DXMT_PARTITION_THRESHOLD_MB", "0");
        storage.rebuild_index("t").await.unwrap();

        // A snapshot pins this shape of the index: its metadata names the partition ids that
        // existed at this moment.
        storage.snapshot().await.unwrap();

        // Then maintenance runs, as the periodic sweep would. Enough new records that
        // partitions cross the split threshold: a split deletes the old segment and writes two
        // new ones under ids the snapshot has never heard of, so recovering from that snapshot
        // afterwards probes a partition that is gone and never probes the two that replaced it.
        // A missing segment costs recall silently — search warns and carries on.
        //
        // Merges are benign by comparison: the absorbing partition's id is already in the
        // snapshot, so its records are still reachable. Splits are the case that loses data.
        let more: Vec<_> = (600..3600)
            .map(|i| {
                let v: Vec<f32> = (0..64).map(|d| i as f32 * 0.01 + d as f32 * 0.03).collect();
                record(&format!("r{}", i), v)
            })
            .collect();
        storage.insert_records("t", more).await.unwrap();
        storage.reevaluate_indexes().await.unwrap();
        std::env::remove_var("DXMT_PARTITION_THRESHOLD_MB");
    }

    // Crash here: recovery uses the snapshot taken before maintenance ran.
    let recovered = IntelligentIndexedStorage::new_with_wal(&config, Some(wal_dir))
        .await
        .unwrap();

    // Every record must still be findable, from before and after the snapshot alike.
    for i in [0usize, 300, 599, 700, 3599] {
        let probe: Vec<f32> = (0..64).map(|d| i as f32 * 0.01 + d as f32 * 0.03).collect();
        let hits = recovered.search("t", query(probe, 1)).await.unwrap();
        assert_eq!(
            hits.hits[0].id,
            format!("r{}", i),
            "r{} was not found after maintenance ran past a snapshot",
            i
        );
    }
}

/// A snapshot of an HNSW table must contain the HNSW graph, not just the rows.
///
/// `snapshot()` asks each index to write itself and took `false` — "this index type has no
/// on-disk form", which is true of BruteForce — for an answer. `unwrap_or(false)` gave the same
/// answer when the write *failed*, and the only visible consequence was that the next restart
/// rebuilt the graph from the rows: correct, arbitrarily slow, and silent. A 77k-row table took
/// over two minutes to start serving because of this, which is what made it worth a test.
///
/// No `reevaluate_indexes` here on purpose. CREATE TABLE sizes its index against a fixed estimate
/// of 10,000 rows and so hands even a tiny table an HNSW index — which is the whole point of the
/// periodic re-evaluation, and is also what lets this test cost 100 inserts instead of the 6,000
/// it takes to make re-evaluation *choose* HNSW. That version ran for 263s in a debug build.
#[tokio::test]
async fn a_snapshot_of_an_hnsw_table_contains_the_index() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let storage = open(&wal_dir).await;

    // The size-based selector: this test is about plain HNSW's on-disk form. (The default,
    // HNSW-PQ, writes its graph too; that is covered by the disk-index tests.)
    let mut hnsw = table("t", 8, DistanceMetric::Euclidean);
    hnsw.memory_optimization = Some(false);
    storage.create_table(hnsw).await.unwrap();
    assert_eq!(
        storage.get_table("t").await.unwrap().index_type.to_string(),
        "hnsw",
        "the table under test is not the one this test is about"
    );

    let rows: Vec<_> = (0..100)
        .map(|i| {
            let v: Vec<f32> = (0..8).map(|d| i as f32 * 0.01 + d as f32 * 0.31).collect();
            record(&format!("r{}", i), v)
        })
        .collect();
    storage.insert_records("t", rows).await.unwrap();

    storage.snapshot().await.unwrap();

    let idx = wal_dir.join("snapshot").join("t0.idx");
    assert!(
        idx.exists(),
        "the snapshot has no index file, so the next restart rebuilds the graph from scratch"
    );
    assert!(
        idx.metadata().unwrap().len() > 0,
        "the snapshot's index file is empty"
    );
}

/// Concurrent snapshots must not damage each other.
///
/// Four callers can snapshot at once in one process: the periodic timer, the eviction sweep
/// (which snapshots before dropping a table), a rebuild, and `POST /snapshot`. They all built
/// into one shared staging directory, starting with `remove_dir_all(&staging)` — so the second
/// caller deletes the first's half-written files. What made that quiet rather than loud is that
/// the deleted file is usually the index: the write fails, `unwrap_or(false)` records it as
/// "this index type has no on-disk form", the directory is recreated in time for the metadata
/// to land, and the snapshot reports success having silently dropped the graph. The next restart
/// then rebuilds it from the rows — correct, and arbitrarily slow. A 77k-row table took over two
/// minutes to start serving, which is how this was found.
///
/// Multi-threaded on purpose. `snapshot()` does its file I/O synchronously, so on the
/// single-threaded runtime `#[tokio::test]` gives you by default, four spawned snapshots each run
/// to completion without yielding and never overlap at all — the test passes against the bug.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_snapshots_do_not_damage_each_other() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let storage = std::sync::Arc::new(open(&wal_dir).await);

    // Enough tables that writing them all takes long enough for four callers to overlap. One
    // table's snapshot is over too quickly to collide with anything.
    for t in 0..40 {
        let name = format!("t{}", t);
        storage
            .create_table(table(&name, 8, DistanceMetric::Euclidean))
            .await
            .unwrap();
        let rows: Vec<_> = (0..50)
            .map(|i| {
                let v: Vec<f32> = (0..8).map(|d| i as f32 * 0.01 + d as f32).collect();
                record(&format!("r{}", i), v)
            })
            .collect();
        storage.insert_records(&name, rows).await.unwrap();
    }

    for round in 0..6 {
        let mut handles = Vec::new();
        for _ in 0..4 {
            let s = storage.clone();
            handles.push(tokio::spawn(async move { s.snapshot().await }));
        }
        for h in handles {
            let written = h.await.unwrap().unwrap_or_else(|e| {
                panic!(
                    "round {}: a snapshot failed while another was running: {}",
                    round, e
                )
            });
            assert_eq!(
                written, 40,
                "round {}: a snapshot wrote {} of 40 tables",
                round, written
            );
        }
    }

    drop(storage);

    // Everything still has to come back.
    let reopened = open(&wal_dir).await;
    for t in 0..40 {
        let name = format!("t{}", t);
        for i in 0..50 {
            assert!(
                reopened.get_record(&name, &format!("r{}", i)).await.is_ok(),
                "{}/r{} was lost by concurrent snapshots",
                name,
                i
            );
        }
    }
}

/// A table restored from a snapshot keeps the index it had, even when that index has no file.
///
/// BruteForce writes nothing to the snapshot — there is nothing worth writing — so restore
/// re-indexes the rows. It did that into whatever `create_table` picked from its fixed estimate
/// of 10,000 rows, which is HNSW. A 94,000-row BruteForce table therefore came back as HNSW and
/// spent over ten minutes building a graph it never had, before the server could bind its port.
/// The snapshot records the algorithm; the restore path has to use it.
#[tokio::test]
async fn a_table_with_no_index_file_is_restored_with_the_index_it_had() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        // The size-based selector: only it answers BruteForce, which is the index without a
        // file that this test is about.
        let mut sized = table("t", 8, DistanceMetric::Euclidean);
        sized.memory_optimization = Some(false);
        storage.create_table(sized).await.unwrap();
        let rows: Vec<_> = (0..20)
            .map(|i| {
                let v: Vec<f32> = (0..8).map(|d| i as f32 * 0.1 + d as f32 * 0.7).collect();
                record(&format!("r{}", i), v)
            })
            .collect();
        storage.insert_records("t", rows).await.unwrap();

        // 20 rows is squarely BruteForce territory; CREATE TABLE had guessed HNSW. An explicit
        // rebuild, not the periodic sweep: the sweep only climbs the size ladder and would
        // leave the guess in place, which is a different test's subject. This one needs a
        // table that really is on BruteForce.
        storage.rebuild_index("t").await.unwrap();
        assert_eq!(
            storage.get_table("t").await.unwrap().index_type.to_string(),
            "brute_force"
        );
        storage.snapshot().await.unwrap();
    }

    let reopened = open(&wal_dir).await;
    assert_eq!(
        reopened
            .get_table("t")
            .await
            .unwrap()
            .index_type
            .to_string(),
        "brute_force",
        "the restored table was given the index CREATE TABLE guesses for 10,000 rows, not the \
         one the snapshot says it had"
    );

    // And it still answers, so the re-indexing went into the index that was installed.
    let hits = reopened
        .search("t", query(vec![0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9], 1))
        .await
        .unwrap();
    assert_eq!(hits.hits[0].id, "r0");
}

/// A search must not hold the table map's lock while it searches.
///
/// That lock is the one every request on every table goes through, and tokio's `RwLock` is
/// write-preferring: while a writer waits behind a reader, later readers queue behind the writer.
/// So a path that holds it across its own slow work does not merely slow that table down, it
/// stops the server. Measured before the fix, a forced rebuild under load gave a read p99 of
/// 25.6s and a write max of 27.1s with no single holder over a second — the holders were a
/// second each, back to back.
///
/// The assertion is a ratio, not a deadline. An earlier attempt at guarding this compared
/// latencies against a baseline of tens of milliseconds and went red two runs in three on a busy
/// machine; it was deleted. How long the search takes here does not matter, only that the lock
/// was free for almost all of it.
#[tokio::test]
#[serial_test::serial]
async fn a_search_does_not_hold_the_table_lock_while_it_searches() {
    let tmp = TempDir::new().unwrap();
    let storage = open(&tmp.path().join("wal")).await;

    storage
        .create_table(table("t", 128, DistanceMetric::Cosine))
        .await
        .unwrap();

    // Enough rows that a search takes long enough to measure against.
    let rows: Vec<_> = (0..20_000)
        .map(|i| {
            let v: Vec<f32> = (0..128)
                .map(|d| ((i * 37 + d * 11) % 997) as f32 / 997.0)
                .collect();
            record(&format!("r{}", i), v)
        })
        .collect();
    storage.insert_records("t", rows).await.unwrap();

    dxmt_vectordb::storage::intelligent_indexed_storage::reset_max_table_lock_hold();

    // Fifty searches, not five: the denominator grows with every one of them while the
    // numerator is a maximum, so the headroom against a stray scheduling hiccup inside a
    // microsecond-long critical section grows too.
    let started = std::time::Instant::now();
    for _ in 0..50 {
        storage
            .search("t", query((0..128).map(|d| d as f32 / 128.0).collect(), 10))
            .await
            .unwrap();
    }
    let searching = started.elapsed();
    let held = dxmt_vectordb::storage::intelligent_indexed_storage::max_table_lock_hold();

    assert!(
        held * 4 < searching,
        "a search held the table map's lock for {:?} of the {:?} it spent searching; the lock \
         has to be released across the index search, not held through it",
        held,
        searching
    );
}

#[tokio::test]
async fn a_dropped_table_does_not_come_back_from_the_snapshot() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let storage = open(&wal_dir).await;

    storage
        .create_table(table("t", 4, DistanceMetric::Euclidean))
        .await
        .unwrap();
    storage
        .insert_records("t", vec![record("a", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .unwrap();

    // The snapshot is the whole precondition. It puts the table in the manifest, and both
    // `list_tables` and rehydration read non-resident tables from there — correct for an
    // evicted table, wrong for a deleted one. Delete without snapshotting first and nothing
    // is wrong to begin with.
    storage.snapshot().await.unwrap();
    storage.delete_table("t").await.unwrap();

    let listed: Vec<String> = storage
        .list_tables()
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert!(
        !listed.contains(&"t".to_string()),
        "a dropped table is still listed: {:?}",
        listed
    );
    assert!(
        storage.get_table("t").await.is_err(),
        "a dropped table is still readable"
    );

    // The name has to be reusable, at a different dimension too: a client that drops a
    // collection and recreates it with a new shape is the case that found this, and it failed
    // with "Dimension mismatch: expected 1536, got 768" against the resurrected table.
    storage
        .create_table(table("t", 8, DistanceMetric::Cosine))
        .await
        .unwrap();
    storage
        .insert_records("t", vec![record("b", vec![0.0; 8])])
        .await
        .unwrap();
    let fresh = storage.get_table("t").await.unwrap();
    assert_eq!(fresh.dimension, 8, "the old table shadowed the new one");
}

#[tokio::test]
async fn a_dropped_table_stays_dropped_across_a_restart() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records("t", vec![record("a", vec![1.0, 0.0, 0.0, 0.0])])
            .await
            .unwrap();
        storage.snapshot().await.unwrap();
        storage.delete_table("t").await.unwrap();

        // This second snapshot is what made the drop unrecoverable rather than merely
        // visible: it copied the table forward into the new manifest and advanced
        // `wal_next_sequence` past the `DeleteTable` entry, so replay no longer covered the
        // deletion and the table came back with its rows on the next start.
        storage.snapshot().await.unwrap();
    }

    let storage = open(&wal_dir).await;
    let listed: Vec<String> = storage
        .list_tables()
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert!(
        !listed.contains(&"t".to_string()),
        "a dropped table came back after a restart: {:?}",
        listed
    );
    assert!(
        storage.get_table("t").await.is_err(),
        "a dropped table is readable after a restart"
    );
}

/// The snapshot carries the graph, and a restart reads it back rather than rebuilding it.
///
/// This exists because the project's own notes claimed for months that "HNSW and HNSW-PQ have no save/load,
/// so a restart means a full index rebuild" while `tests/index_persistence_test.rs` sat in the
/// tree proving the round trip. What neither the doc nor that file covered was the part an
/// operator cares about: that the *server's* restore path uses it. Measured on 100,000 rows at
/// 128 dimensions, the difference is 0.43 s to start serving against 37.1 s to rebuild, so a
/// regression here is not a detail -- it is a restart that takes minutes.
///
/// `has_index` is what the restore path keys on, and the `.idx` file beside the `.meta` is what
/// it reads. Assert the artifact, then assert the answers survive the trip.
#[tokio::test]
async fn a_snapshot_writes_the_index_and_a_restart_reads_it_back() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");

    // Enough rows that the graph is worth persisting and a wrong answer is not a coin flip.
    const ROWS: usize = 2_000;
    let vector = |i: usize| vec![i as f32, (i % 7) as f32, (i % 13) as f32, 1.0];

    let before = {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records(
                "t",
                (0..ROWS)
                    .map(|i| record(&format!("r{}", i), vector(i)))
                    .collect(),
            )
            .await
            .unwrap();
        storage.snapshot().await.unwrap();
        storage
            .search("t", query(vector(1_234), 10))
            .await
            .unwrap()
            .hits
            .into_iter()
            .map(|h| h.id)
            .collect::<Vec<_>>()
    };

    // The graph, on disk, beside the table metadata. The `.meta` holds the records and is the
    // larger of the two here; what matters is that the `.idx` exists and is not a stub, because
    // an absent one is exactly what "no index persistence" would look like.
    let idx = wal_dir.join("snapshot").join("t0.idx");
    assert!(
        idx.exists(),
        "the snapshot wrote no index file; the restore path will rebuild the graph from the \
         rows on every restart. Contents: {:?}",
        std::fs::read_dir(wal_dir.join("snapshot")).map(|d| d
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .collect::<Vec<_>>())
    );
    let idx_bytes = std::fs::metadata(&idx).unwrap().len();
    assert!(
        idx_bytes > 1_024,
        "the index file is {idx_bytes} bytes for {ROWS} rows, which is not a graph"
    );

    let recovered = open(&wal_dir).await;
    let after: Vec<String> = recovered
        .search("t", query(vector(1_234), 10))
        .await
        .unwrap()
        .hits
        .into_iter()
        .map(|h| h.id)
        .collect();

    assert_eq!(
        after, before,
        "the reloaded index answers a different question than the one that was saved"
    );
    assert_eq!(after[0], "r1234", "the nearest neighbour is the row itself");
}

/// Losing the index file costs time, not data.
///
/// The other half of the pair above, and the reason the restore path logs by name instead of
/// falling through quietly: if the snapshot says it wrote an index and the file will not read
/// back, the server pays the full re-index that the snapshot existed to avoid. That is a
/// correct outcome and a slow one, so it must keep working -- and it must not be how the
/// database behaves normally, which is what the test above pins.
#[tokio::test]
async fn a_missing_index_file_is_rebuilt_from_the_rows_rather_than_lost() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let vector = |i: usize| vec![i as f32, 0.0, 0.0, 1.0];

    {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records(
                "t",
                (0..200)
                    .map(|i| record(&format!("r{}", i), vector(i)))
                    .collect(),
            )
            .await
            .unwrap();
        storage.snapshot().await.unwrap();
    }

    let idx = wal_dir.join("snapshot").join("t0.idx");
    assert!(
        idx.exists(),
        "nothing to remove; the snapshot wrote no index"
    );
    std::fs::remove_file(&idx).unwrap();

    let recovered = open(&wal_dir).await;
    assert_eq!(
        recovered.list_tables().await.unwrap()[0].record_count,
        200,
        "the records live in the `.meta`, not the `.idx`; losing the graph must not lose them"
    );
    let hits = recovered.search("t", query(vector(42), 1)).await.unwrap();
    assert_eq!(
        hits.hits[0].id, "r42",
        "the rebuilt-from-rows index must answer as well as the saved one"
    );
}

/// An update followed by a crash must leave a server that starts, with the update in it.
///
/// It did not. `update_record` logged the record as the map keeps it, and on a disk-backed
/// table the map keeps it without its vector — that went to the row file. Replay then indexed
/// an empty vector; the graph stores vectors in one flat array and appended nothing for it while
/// still giving it a node, so every node after it pointed one vector past its data. In a
/// release build — `panic = "abort"` — the first read of that node ended the process, on every
/// start, because the log does not change: `kill -9` after a single payload-only update, and the
/// database could not come back. Measured end to end before the fix: insert, update, SIGKILL,
/// restart → `range end index 1204 out of range for slice of length 1200` and exit.
///
/// Dropping the storage without a snapshot is the crash: the reopened one has only the log.
#[tokio::test]
async fn an_update_is_replayed_with_its_vector() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    let vector = |i: usize| vec![1.0, i as f32 / 50.0, (i % 7) as f32 / 7.0, 0.25];

    {
        let storage = open(&wal).await;
        storage
            .create_table(table("t", 4, DistanceMetric::Cosine))
            .await
            .unwrap();
        storage
            .insert_records(
                "t",
                (0..50)
                    .map(|i| record(&format!("r{i}"), vector(i)))
                    .collect(),
            )
            .await
            .unwrap();
        // Payload-only updates, the way the HTTP handler sends them: the stored vector is read
        // back and passed through unchanged.
        for i in (0..50).step_by(5) {
            let id = format!("r{i}");
            let current = storage.get_record("t", &id).await.unwrap();
            storage
                .update_record(
                    "t",
                    &id,
                    CreateRecordRequest {
                        id: id.clone(),
                        vector: current.vector,
                        sparse_vector: None,
                        named_vectors: None,
                        payload: Some(serde_json::json!({ "updated": i })),
                        text: None,
                    },
                )
                .await
                .unwrap();
        }
        // Dropped here with no snapshot taken: the crash.
    }

    let storage = open(&wal).await;
    let (all, total) = storage
        .list_records("t", None, 100, 0, false)
        .await
        .unwrap();
    assert_eq!(total, 50);
    assert_eq!(all.len(), 50);

    for i in (0..50).step_by(5) {
        let id = format!("r{i}");
        let restored = storage.get_record("t", &id).await.unwrap();
        assert_eq!(
            restored.payload,
            serde_json::json!({ "updated": i }),
            "{id}"
        );
        assert_eq!(restored.vector, vector(i), "{id} lost its vector in replay");

        // And it is in the graph, where it belongs: its own vector finds it first.
        let hits = storage.search("t", query(vector(i), 1)).await.unwrap().hits;
        assert_eq!(hits.first().map(|h| h.id.as_str()), Some(id.as_str()));
    }

    // Every record is reachable and nothing extra is: a node without a vector would either
    // panic here or come back as a hit with a garbage score.
    let hits = storage
        .search("t", query(vector(3), 50))
        .await
        .unwrap()
        .hits;
    assert_eq!(hits.len(), 50);
    assert!(hits.iter().all(|h| h.score.is_finite()), "{hits:?}");
}

/// After a crash, a filtered list pages in the same order as an unfiltered one.
///
/// Replay re-inserts each logged record through the ordinary insert path, which stamps a new
/// record with the time it is *replayed* — a create request carries no timestamp — and then puts
/// the logged record, with its real `created_at`, into the map. A table the replay itself creates
/// has its payload index built from the start, so that index kept the replay times, and
/// `records/list` with a filter pages by the index's timestamps. A record the log had upserted
/// went through the path that keeps the original time — older than every replay time — and sank
/// to the bottom. The stateful walk found it; this is its minimal form: two bulk inserts that
/// share one id, then a crash.
#[tokio::test]
async fn a_filtered_list_keeps_its_order_across_a_crash() {
    use dxmt_vectordb::types::record::{Condition, Filter, FilterOperation};

    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    let tagged = |id: &str| CreateRecordRequest {
        id: id.to_string(),
        vector: vec![1.0, 0.5],
        sparse_vector: None,
        named_vectors: None,
        payload: Some(serde_json::json!({ "k": 1 })),
        text: None,
    };
    {
        let storage = open(&wal).await;
        storage
            .create_table(table("t", 2, DistanceMetric::Cosine))
            .await
            .unwrap();
        storage
            .insert_records("t", vec![tagged("a"), tagged("b"), tagged("c")])
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        // `b` again: an upsert, which keeps its original created_at.
        storage
            .insert_records("t", vec![tagged("b"), tagged("d")])
            .await
            .unwrap();
        // Dropped without a snapshot: the crash.
    }

    let storage = open(&wal).await;
    let filter = Filter {
        must: Some(vec![Condition {
            key: "k".into(),
            operation: FilterOperation::Equal(serde_json::json!(1)),
        }]),
        must_not: None,
        should: None,
    };
    let ids = |records: Vec<dxmt_vectordb::types::record::Record>| -> Vec<String> {
        records.into_iter().map(|r| r.id).collect()
    };
    let unfiltered = ids(storage
        .list_records("t", None, 10, 0, false)
        .await
        .unwrap()
        .0);
    let filtered = ids(storage
        .list_records("t", Some(&filter), 10, 0, false)
        .await
        .unwrap()
        .0);
    assert_eq!(unfiltered.len(), 4);
    assert_eq!(
        filtered, unfiltered,
        "the filtered list is in replay order, not write order"
    );
}

/// How a local WAL write failed: the bytes reached the file and the fsync failed, or only half
/// of the frame reached it (a full disk).
#[cfg(feature = "test-failpoints")]
#[derive(Clone, Copy, Debug)]
enum Fault {
    Sync,
    Torn,
}

/// The write that was in flight when the WAL failed.
#[cfg(feature = "test-failpoints")]
#[derive(Clone, Copy, Debug)]
enum InFlight {
    Insert,
    Update,
    Delete,
}

#[cfg(feature = "test-failpoints")]
const KEPT: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
#[cfg(feature = "test-failpoints")]
const ORIGINAL: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
#[cfg(feature = "test-failpoints")]
const CHANGED: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
#[cfg(feature = "test-failpoints")]
const FRESH: [f32; 4] = [0.0, 0.0, 1.0, 0.0];

/// The ids an exact and an ordinary search put at distance zero from `v`. Euclidean, whose
/// score is `1 / (1 + distance)`, so a stored copy of `v` scores exactly 1 and nothing else
/// does.
#[cfg(feature = "test-failpoints")]
async fn at(storage: &IntelligentIndexedStorage, v: [f32; 4]) -> (Vec<String>, Vec<String>) {
    use dxmt_vectordb::types::record::SearchMode;
    let mut out = Vec::new();
    for mode in [SearchMode::Exact, SearchMode::Normal] {
        let hits = storage
            .search(
                "t",
                SearchQuery {
                    vector: v.to_vec(),
                    top_k: 10,
                    limit: 10,
                    search_mode: mode,
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .hits;
        let mut ids: Vec<String> = hits
            .into_iter()
            .filter(|h| (h.score - 1.0).abs() < 1e-6)
            .map(|h| h.id)
            .collect();
        ids.sort();
        out.push(ids);
    }
    let normal = out.pop().unwrap();
    (out.pop().unwrap(), normal)
}

/// After a restart, the store agrees with itself and with what was acknowledged.
///
/// The write that failed was not acknowledged and the contract does not say whether it
/// survives -- its row may be on disk though its log entry is not, or the reverse. What it
/// does say is that the answer is one answer: `GET`, the exact scan and the graph all hold the
/// write or all do not. Everything acknowledged before it is there; the write refused once the
/// store was read-only is not.
#[cfg(feature = "test-failpoints")]
async fn assert_consistent(storage: &IntelligentIndexedStorage, op: InFlight, label: &str) {
    let get = |id: &'static str| async move { storage.get_record("t", &id.to_string()).await };

    assert_eq!(get("kept").await.unwrap().vector, KEPT.to_vec(), "{label}");
    assert_eq!(
        at(storage, KEPT).await,
        (vec!["kept".into()], vec!["kept".into()]),
        "{label}"
    );
    assert!(
        get("refused").await.is_err(),
        "{label}: a refused write came back"
    );

    match op {
        InFlight::Insert => {
            let held = get("fresh").await.is_ok();
            let expect: Vec<String> = if held { vec!["fresh".into()] } else { vec![] };
            assert_eq!(
                at(storage, FRESH).await,
                (expect.clone(), expect),
                "{label}: GET says {held} and the searches disagree"
            );
        }
        InFlight::Update => {
            let vector = get("a")
                .await
                .expect("an update cannot remove a record")
                .vector;
            let (now, other) = if vector == CHANGED.to_vec() {
                (CHANGED, ORIGINAL)
            } else {
                assert_eq!(
                    vector,
                    ORIGINAL.to_vec(),
                    "{label}: a vector neither write wrote"
                );
                (ORIGINAL, CHANGED)
            };
            assert_eq!(
                at(storage, now).await,
                (vec!["a".into()], vec!["a".into()]),
                "{label}: GET returns {now:?} and the searches do not find it there"
            );
            assert_eq!(
                at(storage, other).await,
                (vec![], vec![]),
                "{label}: GET returns {now:?} and a search finds the record at {other:?}"
            );
        }
        InFlight::Delete => {
            let held = get("a").await.is_ok();
            let expect: Vec<String> = if held { vec!["a".into()] } else { vec![] };
            assert_eq!(
                at(storage, ORIGINAL).await,
                (expect.clone(), expect),
                "{label}: GET says {held} and the searches disagree"
            );
        }
    }

    let stored = storage.list_tables().await.unwrap()[0].record_count;
    assert_eq!(
        storage.get_index_record_count("t").await.unwrap() as u64,
        stored,
        "{label}: the index and the rows hold different numbers of records"
    );
}

#[cfg(feature = "test-failpoints")]
async fn a_failed_write_then_restart(fault: Fault, op: InFlight, snapshot_first: bool) {
    let label = format!("{fault:?} during {op:?}, snapshot first: {snapshot_first}");
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
                    record("kept", KEPT.to_vec()),
                    record("a", ORIGINAL.to_vec()),
                ],
            )
            .await
            .unwrap();
        // With a snapshot in between, the graph that comes back is the snapshot's rather than
        // one rebuilt from the log -- the case where a lost update left the old vector in it.
        if snapshot_first {
            storage.snapshot().await.unwrap();
        }

        match fault {
            Fault::Sync => storage.fail_next_wal_sync_for_test().unwrap(),
            Fault::Torn => storage.tear_next_wal_write_for_test().unwrap(),
        }
        let failed = match op {
            InFlight::Insert => storage
                .insert_records("t", vec![record("fresh", FRESH.to_vec())])
                .await
                .map(|_| ()),
            InFlight::Update => storage
                .update_record("t", &"a".to_string(), record("a", CHANGED.to_vec()))
                .await
                .map(|_| ()),
            InFlight::Delete => storage
                .delete_records("t", vec!["a".to_string()])
                .await
                .map(|_| ()),
        };
        assert_eq!(
            failed.expect_err(&label).status_code(),
            503,
            "{label}: the failed write must not be acknowledged"
        );
        assert!(storage
            .insert_records("t", vec![record("refused", vec![0.5, 0.5, 0.0, 0.0])])
            .await
            .is_err());
    }

    let first = open(&wal_dir).await;
    assert_consistent(&first, op, &format!("{label}, first restart")).await;

    // Writable again, and what it writes now survives the next restart: the sequence numbers
    // after a failed frame do not collide with it.
    first
        .insert_records("t", vec![record("after", vec![0.0, 0.5, 0.5, 0.0])])
        .await
        .unwrap();
    let held_before = first.list_tables().await.unwrap()[0].record_count;
    drop(first);

    let second = open(&wal_dir).await;
    assert_consistent(&second, op, &format!("{label}, second restart")).await;
    assert!(
        second.get_record("t", &"after".to_string()).await.is_ok(),
        "{label}"
    );
    assert_eq!(
        second.list_tables().await.unwrap()[0].record_count,
        held_before,
        "{label}: the second restart changed its mind about the failed write"
    );
}

#[cfg(feature = "test-failpoints")]
#[tokio::test]
async fn a_write_whose_fsync_failed_is_all_there_or_all_gone_after_restart() {
    for op in [InFlight::Insert, InFlight::Update, InFlight::Delete] {
        a_failed_write_then_restart(Fault::Sync, op, false).await;
    }
}

#[cfg(feature = "test-failpoints")]
#[tokio::test]
async fn a_write_torn_by_a_full_disk_is_all_there_or_all_gone_after_restart() {
    for op in [InFlight::Insert, InFlight::Update, InFlight::Delete] {
        a_failed_write_then_restart(Fault::Torn, op, false).await;
    }
}

#[cfg(feature = "test-failpoints")]
#[tokio::test]
async fn a_failed_write_after_a_snapshot_is_all_there_or_all_gone_after_restart() {
    for fault in [Fault::Sync, Fault::Torn] {
        for op in [InFlight::Insert, InFlight::Update, InFlight::Delete] {
            a_failed_write_then_restart(fault, op, true).await;
        }
    }
}

/// A second instance over a directory that is already open is refused, and the directory is
/// free again once the first is gone. Two writers to one log corrupt it; nothing stopped a
/// second `serve`, or `vectordb mcp` beside a running server, from opening it.
#[tokio::test]
async fn a_store_cannot_be_opened_twice_at_once() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let first = open(&wal_dir).await;
    first
        .create_table(table("t", 2, DistanceMetric::Cosine))
        .await
        .unwrap();

    let err = IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(wal_dir.clone()))
        .await
        .err()
        .expect("a second instance opened a store already in use");
    assert!(
        err.to_string().contains("in use by another process"),
        "{err}"
    );

    drop(first);
    let again = open(&wal_dir).await;
    assert_eq!(again.list_tables().await.unwrap().len(), 1);
}

/// A record's version survives a restart that replays it, including an update that keeps the
/// vector (an empty one). The batched replay of 60d76ab brought such a record back one version
/// ahead -- the stateful walk caught it as `r2 version 10 != 9` after `update r2 empty` and a
/// SIGKILL; nothing in the Rust suite compared versions across a replay.
#[tokio::test]
async fn replay_keeps_every_record_version() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let expected = {
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
        storage.snapshot().await.unwrap();
        // After the snapshot, so replay applies them: an upsert, an update with a vector, an
        // update that keeps its vector.
        storage
            .insert_records("t", vec![record("a", vec![1.0, 1.0, 0.0, 0.0])])
            .await
            .unwrap();
        storage
            .update_record("t", &"b".to_string(), record("b", vec![0.0, 1.0, 1.0, 0.0]))
            .await
            .unwrap();
        let mut keep = record("b", Vec::new());
        keep.payload = Some(serde_json::json!({"note": "vector kept"}));
        storage
            .update_record("t", &"b".to_string(), keep)
            .await
            .unwrap();
        let a = storage
            .get_record("t", &"a".to_string())
            .await
            .unwrap()
            .version;
        let b = storage
            .get_record("t", &"b".to_string())
            .await
            .unwrap()
            .version;
        (a, b)
    };

    let recovered = open(&wal_dir).await;
    let a = recovered.get_record("t", &"a".to_string()).await.unwrap();
    let b = recovered.get_record("t", &"b".to_string()).await.unwrap();
    assert_eq!(
        (a.version, b.version),
        expected,
        "versions changed across the replay"
    );
    assert_eq!(b.payload["note"], "vector kept");
    assert_eq!(b.vector, vec![0.0, 1.0, 1.0, 0.0]);
}

/// A replayed run the graph refuses as a whole leaves every record at its logged version.
///
/// The snapshot's graph holds `x`; after it, `x` is upserted, `y` updated, and `x` deleted.
/// Recovery restores that graph, the row files (where `x` is gone), and replays the tail: the
/// run `[x, y]` reaches the processor, which applies the record map, then finds `x` already in
/// the graph and refuses the run -- after `y`'s version was bumped. The retry, one record at a
/// time, used to check versions again, find `y` "ahead of the log", and keep the bump. The
/// stateful walk found it as `r2 version 10 != 9`.
#[tokio::test]
async fn a_refused_replay_run_leaves_versions_as_logged() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    let y_version = {
        let storage = open(&wal_dir).await;
        storage
            .create_table(table("t", 4, DistanceMetric::Euclidean))
            .await
            .unwrap();
        storage
            .insert_records(
                "t",
                vec![
                    record("x", vec![1.0, 0.0, 0.0, 0.0]),
                    record("y", vec![0.0, 1.0, 0.0, 0.0]),
                ],
            )
            .await
            .unwrap();
        storage.snapshot().await.unwrap();
        storage
            .insert_records(
                "t",
                vec![
                    record("x", vec![1.0, 0.5, 0.0, 0.0]),
                    record("y", vec![0.0, 1.0, 0.5, 0.0]),
                ],
            )
            .await
            .unwrap();
        storage
            .delete_records("t", vec!["x".to_string()])
            .await
            .unwrap();
        storage
            .get_record("t", &"y".to_string())
            .await
            .unwrap()
            .version
    };

    let recovered = open(&wal_dir).await;
    let y = recovered.get_record("t", &"y".to_string()).await.unwrap();
    assert_eq!(y.version, y_version, "y came back at another version");
    assert!(recovered.get_record("t", &"x".to_string()).await.is_err());
}

/// Make a directory tree read-only, or writable again, so an upload into it fails.
#[cfg(unix)]
fn set_writable(dir: &Path, writable: bool) {
    use std::os::unix::fs::PermissionsExt;
    let mode = if writable { 0o755 } else { 0o555 };
    for entry in walk(dir) {
        if entry.is_dir() {
            std::fs::set_permissions(&entry, std::fs::Permissions::from_mode(mode)).unwrap();
        }
    }
}

#[cfg(unix)]
fn walk(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = vec![dir.to_path_buf()];
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            if e.path().is_dir() {
                out.extend(walk(&e.path()));
            }
        }
    }
    out
}

#[cfg(unix)]
async fn ids_recovered_from_object_storage_alone(
    config: &StorageConfig,
    fresh_wal: std::path::PathBuf,
) -> Vec<String> {
    let recovered = IntelligentIndexedStorage::new_with_wal(config, Some(fresh_wal))
        .await
        .unwrap();
    let mut ids: Vec<String> = recovered
        .list_records("t", None, 100, 0, false)
        .await
        .unwrap()
        .0
        .into_iter()
        .map(|r| r.id)
        .collect();
    ids.sort();
    ids
}

/// An upload that failed used to be logged and forgotten, leaving the bucket without those
/// entries for good -- invisible to the manifest's own check, since the chunk was never listed.
/// The next write now carries them up with its own.
#[cfg(unix)]
#[tokio::test]
async fn a_wal_upload_that_failed_is_carried_up_by_the_next_write() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let bucket_wal = shared.join("ssd").join("test-bucket").join("wal");
    {
        let storage =
            IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("machine-a")))
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
        assert!(bucket_wal.exists(), "the first chunk was never uploaded");

        set_writable(&bucket_wal, false);
        storage
            .insert_records("t", vec![record("b", vec![0.0, 1.0, 0.0, 0.0])])
            .await
            .expect("durable locally, so the write succeeds");
        let lagging = storage.health_snapshot().await;
        assert!(
            lagging.wal_mirrored_next_sequence < lagging.wal_next_sequence,
            "the frontier should show b missing: {:?} vs {:?}",
            lagging.wal_mirrored_next_sequence,
            lagging.wal_next_sequence
        );

        set_writable(&bucket_wal, true);
        storage
            .insert_records("t", vec![record("c", vec![0.0, 0.0, 1.0, 0.0])])
            .await
            .unwrap();
        let caught_up = storage.health_snapshot().await;
        assert_eq!(
            caught_up.wal_mirrored_next_sequence,
            caught_up.wal_next_sequence
        );
    }
    assert_eq!(
        ids_recovered_from_object_storage_alone(&config, tmp.path().join("machine-b")).await,
        vec!["a", "b", "c"],
        "b's upload failed and nothing carried it up"
    );
}

/// The same, across a restart: entries whose upload failed before shutdown go up at startup.
#[cfg(unix)]
#[tokio::test]
async fn wal_entries_that_never_reached_object_storage_go_up_at_the_next_start() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let bucket_wal = shared.join("ssd").join("test-bucket").join("wal");
    let machine_a = tmp.path().join("machine-a");
    {
        let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(machine_a.clone()))
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
        set_writable(&bucket_wal, false);
        storage
            .insert_records("t", vec![record("b", vec![0.0, 1.0, 0.0, 0.0])])
            .await
            .unwrap();
    }
    set_writable(&bucket_wal, true);
    {
        // The same machine starts again; nothing new is written.
        let storage = IntelligentIndexedStorage::new_with_wal(&config, Some(machine_a))
            .await
            .unwrap();
        let h = storage.health_snapshot().await;
        assert_eq!(
            h.wal_mirrored_next_sequence, h.wal_next_sequence,
            "not caught up at start"
        );
    }
    assert_eq!(
        ids_recovered_from_object_storage_alone(&config, tmp.path().join("machine-b")).await,
        vec!["a", "b"]
    );
}

/// A snapshot whose upload failed used to trim the local log and prune the bucket's WAL chunks
/// below it anyway, leaving a bucket with neither the snapshot nor the entries. Now neither
/// happens until the bucket holds the snapshot, and a fresh machine recovers everything.
#[cfg(unix)]
#[tokio::test]
async fn a_snapshot_that_did_not_reach_object_storage_does_not_prune_it() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let config = storage_config_with_object_store(&shared);
    let bucket = shared.join("ssd").join("test-bucket");
    {
        let storage =
            IntelligentIndexedStorage::new_with_wal(&config, Some(tmp.path().join("machine-a")))
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
        assert!(!bucket.join("snapshot").exists());
        // No snapshot directory can be made in the bucket; the WAL chunks stay deletable.
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&bucket, std::fs::Permissions::from_mode(0o555)).unwrap();
        storage
            .snapshot()
            .await
            .expect("the local snapshot succeeds");
        std::fs::set_permissions(&bucket, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    assert_eq!(
        ids_recovered_from_object_storage_alone(&config, tmp.path().join("machine-b")).await,
        vec!["a", "b"],
        "the bucket lost entries a snapshot it never received was said to cover"
    );
}
