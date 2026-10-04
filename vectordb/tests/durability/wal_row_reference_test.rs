//! The log points at a vector instead of carrying it.
//!
//! With a WAL directory a table's vectors go to `wal/vectors/<table>.f32`, fsynced, before
//! the write is logged. Until 2026-10-04 the log entry then carried the same vector again,
//! so every vector was written twice and the log was almost entirely vectors. An insert or
//! update on such a table now writes `UpsertRecordInRowFile`, a record without its vector
//! plus the row the vector is at; replay reads that row. The claims checked here are the
//! ones replay depends on: the log is small, a restart rebuilds the index from the rows the
//! entries point at, an update is replayed with its *new* vector, and the sequence
//! `insert, delete, insert` of one id -- where the id's current row is not the first entry's
//! row -- comes back as the last insert. A table whose vectors are in memory has no row to
//! point at and still logs the vector.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
    wal::{WalEntry, WalManager},
};

const DIM: usize = 256;
const ROWS: usize = 500;

/// `DXMT_VECTOR_STORE` is read when a table is created; the two tests here want different
/// answers, and tests in one binary share one environment.
static ENV: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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

async fn open(wal: &std::path::Path) -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(wal.to_path_buf()))
        .await
        .expect("open storage")
}

fn table(name: &str) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension: DIM,
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

fn vector(seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (0..DIM).map(|_| rng.gen::<f32>()).collect()
}

fn row(i: usize, seed: u64) -> CreateRecordRequest {
    CreateRecordRequest {
        id: format!("r{i}"),
        vector: vector(seed),
        sparse_vector: None,
        named_vectors: None,
        payload: Some(serde_json::json!({ "seed": seed })),
        text: None,
    }
}

fn rows(from: usize, to: usize) -> Vec<CreateRecordRequest> {
    (from..to).map(|i| row(i, i as u64)).collect()
}

async fn top1(
    storage: &IntelligentIndexedStorage,
    seed: u64,
    mode: SearchMode,
) -> Option<(String, f32)> {
    storage
        .search(
            "t",
            SearchQuery {
                vector: vector(seed),
                top_k: 1,
                limit: 1,
                search_mode: mode,
                with_payload: false,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits
        .first()
        .map(|h| (h.id.clone(), h.score))
}

async fn create(storage: &IntelligentIndexedStorage, vectors_in_memory: bool) {
    let _guard = ENV.lock().await;
    if vectors_in_memory {
        std::env::set_var("DXMT_VECTOR_STORE", "memory");
    } else {
        std::env::remove_var("DXMT_VECTOR_STORE");
    }
    storage.create_table(table("t")).await.unwrap();
    std::env::remove_var("DXMT_VECTOR_STORE");
}

fn upserts(wal: &std::path::Path) -> (usize, usize) {
    let entries = WalManager::new(wal).unwrap().replay_from(0).unwrap();
    let mut by_row = 0;
    let mut with_vector = 0;
    for (_, entry) in entries {
        match entry {
            WalEntry::UpsertRecordInRowFile { record, .. } => {
                assert!(record.vector.is_empty(), "the entry points at the row");
                by_row += 1;
            }
            WalEntry::UpsertRecordV2 { record, .. } => {
                assert_eq!(record.vector.len(), DIM, "the entry carries the vector");
                with_vector += 1;
            }
            _ => {}
        }
    }
    (by_row, with_vector)
}

#[tokio::test]
async fn the_log_points_at_rows_and_a_restart_reads_them() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    let log = wal.join("wal.log");

    {
        let storage = open(&wal).await;
        create(&storage, false).await;
        storage.insert_records("t", rows(0, 300)).await.unwrap();
        storage.insert_records("t", rows(300, ROWS)).await.unwrap();

        // r7 gets a new vector; its old row is no longer what the id resolves to.
        storage
            .update_record("t", &"r7".to_string(), row(7, 10_007))
            .await
            .unwrap();
        // r9 is deleted and written again: the first entry's row is dead, the id's row is
        // the second's. Replay by id would read the second row for the first entry and
        // find nothing for the second, after the delete between them has run.
        storage
            .delete_records("t", vec!["r9".to_string()])
            .await
            .unwrap();
        storage
            .insert_records("t", vec![row(9, 10_009)])
            .await
            .unwrap();
        // r11 stays deleted.
        storage
            .delete_records("t", vec!["r11".to_string()])
            .await
            .unwrap();

        let (by_row, with_vector) = upserts(&wal);
        assert_eq!(
            (by_row, with_vector),
            (ROWS + 2, 0),
            "every insert and the update point at a row"
        );
        let vectors_would_be = ((ROWS + 2) * DIM * 4) as u64;
        let logged = std::fs::metadata(&log).unwrap().len();
        assert!(
            logged < vectors_would_be / 4,
            "the log is {logged} bytes; the vectors alone would have been {vectors_would_be}"
        );
    }

    // No snapshot was taken, so everything the restart knows about the index comes from
    // replaying those entries against the row file.
    let storage = open(&wal).await;
    assert_eq!(
        storage.get_table("t").await.unwrap().record_count,
        (ROWS - 1) as u64
    );
    assert_eq!(
        storage.get_index_record_count("t").await.unwrap(),
        ROWS - 1,
        "the index holds what the rows hold"
    );
    for seed in [0u64, 150, 299, 300, 499] {
        let id = format!("r{seed}");
        for mode in [SearchMode::Exact, SearchMode::Normal] {
            let (got, score) = top1(&storage, seed, mode).await.unwrap();
            assert_eq!(got, id, "{mode:?}");
            assert!(score > 0.999, "{id} at its own vector: score {score}");
        }
    }
    // The update replayed with its new vector, not its first one.
    let (got, score) = top1(&storage, 10_007, SearchMode::Normal).await.unwrap();
    assert_eq!((got.as_str(), score > 0.999), ("r7", true));
    assert_ne!(top1(&storage, 7, SearchMode::Exact).await.unwrap().0, "r7");
    assert_eq!(
        storage
            .get_record("t", &"r7".to_string())
            .await
            .unwrap()
            .version,
        2
    );
    // The re-inserted id replayed as its second write, and `GET` hands back that vector:
    // the replay re-stores the record for its timestamps, and a re-store without the row
    // left the record readable with an empty vector (found by the stateful walk).
    let (got, score) = top1(&storage, 10_009, SearchMode::Normal).await.unwrap();
    assert_eq!((got.as_str(), score > 0.999), ("r9", true));
    for (id, seed) in [("r9", 10_009u64), ("r7", 10_007), ("r0", 0)] {
        let read = storage.get_record("t", &id.to_string()).await.unwrap();
        assert_eq!(
            read.vector,
            vector(seed),
            "{id} read back after the restart"
        );
    }
    assert_ne!(top1(&storage, 9, SearchMode::Exact).await.unwrap().0, "r9");
    // The deleted one is gone in every mode.
    assert_ne!(
        top1(&storage, 11, SearchMode::Exact).await.unwrap().0,
        "r11"
    );
    assert!(storage.get_record("t", &"r11".to_string()).await.is_err());
}

#[tokio::test]
async fn a_table_with_vectors_in_memory_still_logs_them() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    {
        let storage = open(&wal).await;
        create(&storage, true).await;
        storage.insert_records("t", rows(0, 40)).await.unwrap();
        storage
            .update_record("t", &"r3".to_string(), row(3, 10_003))
            .await
            .unwrap();
        assert_eq!(
            upserts(&wal),
            (0, 41),
            "no row file, so the log carries the vectors"
        );
    }
    let storage = open(&wal).await;
    assert_eq!(storage.get_index_record_count("t").await.unwrap(), 40);
    for seed in [0u64, 20, 39, 10_003] {
        let want = if seed == 10_003 {
            "r3"
        } else {
            &format!("r{seed}")
        };
        let (got, score) = top1(&storage, seed, SearchMode::Normal).await.unwrap();
        assert_eq!((got.as_str(), score > 0.999), (want, true));
    }
}

/// Drop a table and create it again under the same name, with no snapshot between. The
/// entries after the drop point at rows of the file the crash left, which is the second
/// table's; a replayed drop that removed it, as a live drop does, would lose every one of
/// them, and a replayed create that started from an empty file would too.
#[tokio::test]
async fn a_table_dropped_and_recreated_in_one_log_comes_back_as_the_second() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    {
        let storage = open(&wal).await;
        create(&storage, false).await;
        storage.insert_records("t", rows(0, 30)).await.unwrap();
        storage.delete_table("t").await.unwrap();
        create(&storage, false).await;
        storage
            .insert_records("t", (0..20).map(|i| row(i, 20_000 + i as u64)).collect())
            .await
            .unwrap();
    }
    let storage = open(&wal).await;
    assert_eq!(storage.get_table("t").await.unwrap().record_count, 20);
    assert_eq!(storage.get_index_record_count("t").await.unwrap(), 20);
    for i in [0u64, 7, 19] {
        let (got, score) = top1(&storage, 20_000 + i, SearchMode::Normal)
            .await
            .unwrap();
        assert_eq!((got, score > 0.999), (format!("r{i}"), true));
        // The first table's vector for the same id is not what the record holds.
        assert!(top1(&storage, i, SearchMode::Exact).await.unwrap().1 < 0.999);
    }
    assert!(storage.get_record("t", &"r25".to_string()).await.is_err());
}
