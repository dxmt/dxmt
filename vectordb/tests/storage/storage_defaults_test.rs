//! What a table gets without any `DB_*` store variable set: records and full vectors in row
//! files under the WAL directory, and an HNSW-PQ index (exact until it has 65,536 rows).
//! `DXMT_RECORD_STORE=memory` / `DXMT_VECTOR_STORE=memory` opt a fresh table out; a table that
//! already has its files keeps them whatever the environment says at restart.
//!
//! The variables are process-global, so every test here is serial and clears what it set.

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

const DIM: usize = 16;
const ROWS: usize = 300;

fn config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 1024,
            eviction_policy: "lru".to_string(),
        }),
        ..Default::default()
    }
}

fn table(name: &str) -> TableConfig {
    TableConfig {
        name: name.to_string(),
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
    }
}

fn vector(i: usize) -> Vec<f32> {
    // A distinct direction per row: a hash of (row, dimension), not a modular ramp that
    // repeats every 97 rows.
    (0..DIM)
        .map(|d| {
            let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (d as u64 + 1);
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            (h & 0xFFFF) as f32 / 65_536.0 + 0.01
        })
        .collect()
}

fn rows() -> Vec<CreateRecordRequest> {
    (0..ROWS)
        .map(|i| CreateRecordRequest {
            id: i.to_string(),
            vector: vector(i),
            sparse_vector: None,
            named_vectors: None,
            payload: Some(serde_json::json!({ "i": i })),
            text: None,
        })
        .collect()
}

async fn open(wal: &Path) -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new_with_wal(&config(), Some(wal.to_path_buf()))
        .await
        .unwrap()
}

fn logs() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
}

fn clear_env() {
    for v in [
        "DXMT_RECORD_STORE",
        "DXMT_VECTOR_STORE",
        "DXMT_VECTOR_F16",
        "DXMT_INDEX_ALGORITHM",
    ] {
        std::env::remove_var(v);
    }
}

fn files(dir: &Path) -> Vec<String> {
    match std::fs::read_dir(dir) {
        Ok(rd) => {
            let mut v: Vec<String> = rd
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            v.sort();
            v
        }
        Err(_) => Vec::new(),
    }
}

async fn top1(storage: &IntelligentIndexedStorage, i: usize) -> String {
    storage
        .search(
            "t",
            SearchQuery {
                vector: vector(i),
                top_k: 1,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits[0]
        .id
        .clone()
}

#[serial_test::serial]
#[tokio::test]
async fn by_default_a_table_puts_records_and_vectors_on_disk_under_an_hnsw_pq_index() {
    clear_env();
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");

    let storage = open(&wal).await;
    storage.create_table(table("t")).await.unwrap();
    storage.insert_records("t", rows()).await.unwrap();

    assert_eq!(
        files(&wal.join("records")),
        vec!["t.rec".to_string(), "t.rec.log".to_string()],
        "records go to a row file without being asked"
    );
    assert_eq!(
        files(&wal.join("vectors")),
        vec!["t.f32".to_string(), "t.f32.rows".to_string()],
        "full vectors go to a single-precision row file without being asked"
    );
    assert_eq!(
        storage.get_table("t").await.unwrap().index_type.to_string(),
        "hnsw_pq",
        "the index is HNSW-PQ from creation, not HNSW to be rebuilt later"
    );

    // Exact below the quantization threshold: the own vector is the top hit every time.
    for i in [0, 17, 299] {
        assert_eq!(top1(&storage, i).await, i.to_string());
    }
    let rec = storage.get_record("t", &"42".to_string()).await.unwrap();
    assert_eq!(rec.vector, vector(42));
    assert_eq!(rec.payload["i"], 42);

    // Growing does not change the algorithm, so the sweep has nothing to rebuild.
    assert!(
        storage.reevaluate_indexes().await.unwrap().is_empty(),
        "re-evaluation rebuilt a table that was created with the index it wants"
    );

    // Restart from WAL and row files.
    drop(storage);
    let storage = open(&wal).await;
    assert_eq!(
        storage.get_table("t").await.unwrap().record_count,
        ROWS as u64
    );
    assert_eq!(top1(&storage, 17).await, "17");
    assert_eq!(
        storage
            .get_record("t", &"42".to_string())
            .await
            .unwrap()
            .vector,
        vector(42)
    );
    clear_env();
}

#[serial_test::serial]
#[tokio::test]
async fn memory_opts_a_fresh_table_out_and_a_table_without_a_wal_directory_is_always_in_memory() {
    clear_env();
    std::env::set_var("DXMT_RECORD_STORE", "memory");
    std::env::set_var("DXMT_VECTOR_STORE", "memory");
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");

    let storage = open(&wal).await;
    storage.create_table(table("t")).await.unwrap();
    storage.insert_records("t", rows()).await.unwrap();
    assert!(files(&wal.join("records")).is_empty(), "no record file");
    assert!(files(&wal.join("vectors")).is_empty(), "no vector file");
    assert_eq!(top1(&storage, 5).await, "5");
    drop(storage);

    clear_env();
    let storage = IntelligentIndexedStorage::new(&config()).await.unwrap();
    storage.create_table(table("t")).await.unwrap();
    storage.insert_records("t", rows()).await.unwrap();
    assert_eq!(top1(&storage, 5).await, "5");
    assert!(
        !tmp.path().join("records").exists() && !tmp.path().join("vectors").exists(),
        "an instance without a WAL directory has nowhere to put row files"
    );
    clear_env();
}

#[serial_test::serial]
#[tokio::test]
async fn a_table_with_files_on_disk_keeps_them_when_the_environment_says_memory() {
    clear_env();
    logs();
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");

    let storage = open(&wal).await;
    storage.create_table(table("t")).await.unwrap();
    storage.insert_records("t", rows()).await.unwrap();
    storage.snapshot().await.unwrap();
    drop(storage);
    assert_eq!(files(&wal.join("records")).len(), 2);

    // The snapshot of a disk table carries no records; they are in the row file. Coming back
    // under `memory` must still find them.
    std::env::set_var("DXMT_RECORD_STORE", "memory");
    std::env::set_var("DXMT_VECTOR_STORE", "memory");
    let storage = open(&wal).await;
    assert_eq!(
        storage.get_table("t").await.unwrap().record_count,
        ROWS as u64,
        "records lost by switching the default under a table that has a row file"
    );
    assert_eq!(top1(&storage, 123).await, "123");
    assert_eq!(
        storage
            .get_record("t", &"123".to_string())
            .await
            .unwrap()
            .vector,
        vector(123)
    );

    // New writes keep going to the files it has.
    storage
        .insert_records(
            "t",
            vec![CreateRecordRequest {
                id: "new".to_string(),
                vector: vector(1000),
                sparse_vector: None,
                named_vectors: None,
                payload: None,
                text: None,
            }],
        )
        .await
        .unwrap();
    assert_eq!(top1(&storage, 1000).await, "new", "before restart");
    assert_eq!(
        storage
            .get_record("t", &"new".to_string())
            .await
            .unwrap()
            .vector,
        vector(1000),
        "before restart"
    );
    drop(storage);
    let storage = open(&wal).await;
    let after = storage.get_record("t", &"new".to_string()).await;
    assert!(after.is_ok(), "after restart: {after:?}");
    assert_eq!(after.unwrap().vector, vector(1000), "after restart vector");
    assert_eq!(top1(&storage, 1000).await, "new");
    assert_eq!(
        storage.get_table("t").await.unwrap().record_count,
        ROWS as u64 + 1
    );
    clear_env();
}

#[serial_test::serial]
#[tokio::test]
async fn a_fresh_create_does_not_inherit_a_dropped_tables_files() {
    clear_env();
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");

    let storage = open(&wal).await;
    storage.create_table(table("t")).await.unwrap();
    storage.insert_records("t", rows()).await.unwrap();
    storage.delete_table("t").await.unwrap();
    assert!(
        files(&wal.join("records")).is_empty(),
        "dropping removes the row files"
    );

    // Same name, memory mode: nothing on disk to resurrect, and none created.
    std::env::set_var("DXMT_RECORD_STORE", "memory");
    std::env::set_var("DXMT_VECTOR_STORE", "memory");
    storage.create_table(table("t")).await.unwrap();
    assert_eq!(storage.get_table("t").await.unwrap().record_count, 0);
    assert!(files(&wal.join("records")).is_empty());
    clear_env();
}

#[serial_test::serial]
#[tokio::test]
async fn an_insert_after_the_last_snapshot_is_searchable_after_a_restart() {
    clear_env();
    logs();
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");

    let storage = open(&wal).await;
    storage.create_table(table("t")).await.unwrap();
    storage.insert_records("t", rows()).await.unwrap();
    storage.snapshot().await.unwrap();
    storage
        .insert_records(
            "t",
            vec![CreateRecordRequest {
                id: "new".to_string(),
                vector: vector(1000),
                sparse_vector: None,
                named_vectors: None,
                payload: None,
                text: None,
            }],
        )
        .await
        .unwrap();
    assert_eq!(top1(&storage, 1000).await, "new", "before restart");
    drop(storage);

    let storage = open(&wal).await;
    assert_eq!(
        storage.get_table("t").await.unwrap().record_count,
        ROWS as u64 + 1
    );
    assert_eq!(top1(&storage, 1000).await, "new", "after restart");
    clear_env();
}
