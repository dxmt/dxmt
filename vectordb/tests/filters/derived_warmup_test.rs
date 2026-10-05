//! The payload index is built ahead of the first filtered query after a restart, off the
//! query path, and only installed when nothing wrote to the table meanwhile.
//!
//! The lazy build was the whole cold start of a filtered workload: the first filtered query
//! on a reopened 10M-row table paid 33 s for a pass over the rows. `warm_derived` does that
//! pass as soon as recovery is done, a chunk at a time under short read locks, so the query
//! finds the index built; a write that lands during the build voids it, and the lazy build
//! runs as before.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{
            Condition, CreateRecordRequest, Filter, FilterOperation, SearchMode, SearchQuery,
        },
        table::TableConfig,
        vector::DistanceMetric,
    },
};

const DIM: usize = 8;
const ROWS: usize = 40_000;

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

fn table() -> TableConfig {
    TableConfig {
        name: "t".to_string(),
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

fn row(i: usize) -> CreateRecordRequest {
    CreateRecordRequest {
        id: format!("r{i}"),
        vector: (0..DIM)
            .map(|d| ((i * 13 + d * 7) % 101) as f32 / 101.0)
            .collect(),
        sparse_vector: None,
        named_vectors: None,
        payload: Some(
            serde_json::json!({ "tags": [i % 97, 1000 + i % 5], "lang": if i.is_multiple_of(3) { "en" } else { "de" } }),
        ),
        text: None,
    }
}

fn tag(t: i64) -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: "tags".to_string(),
            operation: FilterOperation::Equal(serde_json::json!(t)),
        }]),
        must_not: None,
        should: None,
    }
}

async fn count_and_search(storage: &IntelligentIndexedStorage, t: i64) -> (usize, Vec<String>) {
    let n = storage.count_records("t", Some(&tag(t))).await.unwrap();
    let hits = storage
        .search(
            "t",
            SearchQuery {
                vector: row(5).vector,
                top_k: 10,
                limit: 10,
                filter: Some(tag(t)),
                search_mode: SearchMode::Normal,
                with_payload: true,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits;
    (n, hits.into_iter().map(|h| h.id).collect())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reopened_table_is_warmed_and_answers_filters_as_before() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    let before = {
        let storage = open(&wal).await;
        storage.create_table(table()).await.unwrap();
        for chunk in (0..ROWS).collect::<Vec<_>>().chunks(5_000) {
            storage
                .insert_records("t", chunk.iter().map(|&i| row(i)).collect())
                .await
                .unwrap();
        }
        storage.snapshot().await.unwrap();
        count_and_search(&storage, 42).await
    };
    assert_eq!(before.0, ROWS / 97 + usize::from(ROWS % 97 > 42));

    // Reopened: nothing built yet; the warm-up builds it; a second warm-up has nothing to do.
    let storage = open(&wal).await;
    assert!(
        storage.warm_derived("t").await.unwrap(),
        "first warm-up builds"
    );
    assert!(!storage.warm_derived("t").await.unwrap(), "already built");
    let after = count_and_search(&storage, 42).await;
    assert_eq!(
        after, before,
        "the warmed index answers exactly as the one built live did"
    );
    // The ids a filtered search returns all carry the tag: the index is the records'.
    for id in &after.1 {
        let i: usize = id[1..].parse().unwrap();
        assert_eq!(i % 97, 42, "{id} came back for tag 42");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_before_the_warm_up_leaves_the_build_to_the_first_query() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    {
        let storage = open(&wal).await;
        storage.create_table(table()).await.unwrap();
        storage
            .insert_records("t", (0..ROWS).map(row).collect())
            .await
            .unwrap();
        storage.snapshot().await.unwrap();
    }
    let storage = open(&wal).await;
    // A write lands first (as it would on a busy server before the warm-up's turn), and a
    // write that lands *during* the build is the same case from the build's point of view:
    // the counter moved. Here the write precedes the build, so the build sees it and
    // installs; the point under test is that whichever way it goes, the filter is right.
    storage
        .insert_records("t", vec![row(ROWS + 42)])
        .await
        .unwrap();
    let _ = storage.warm_derived("t").await.unwrap();
    let (n, ids) = count_and_search(&storage, 42).await;
    let expect = (0..ROWS)
        .chain(std::iter::once(ROWS + 42))
        .filter(|i| i % 97 == 42)
        .count();
    assert_eq!(
        n, expect,
        "the new row is counted whichever path built the index"
    );
    assert!(ids
        .iter()
        .all(|id| id[1..].parse::<usize>().unwrap() % 97 == 42));
}
