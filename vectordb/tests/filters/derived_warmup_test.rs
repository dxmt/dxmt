//! The payload index is built ahead of the first filtered query after a restart, off the
//! query path, and only installed when nothing wrote to the table meanwhile.
//!
//! The lazy build was the whole cold start of a filtered workload: the first filtered query
//! on a reopened 10M-row table paid 33 s for a pass over the rows. `warm_derived` does that
//! pass as soon as recovery is done, a chunk at a time under short read locks, so the query
//! finds the index built. A write that lands during the build is caught up before the index is
//! installed; it used to void the build, which on a server taking writes was every build.

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
        rerank: None,
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
#[serial_test::serial] // the table-lock high-water mark below is process-wide
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
#[serial_test::serial] // the table-lock high-water mark below is process-wide
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

/// The tag values of every live record, read off the rows rather than any index.
async fn truth(storage: &IntelligentIndexedStorage, t: i64) -> usize {
    let (records, _) = storage
        .list_records("t", None, 1_000_000, 0, false)
        .await
        .unwrap();
    records
        .iter()
        .filter(|r| {
            r.payload["tags"]
                .as_array()
                .is_some_and(|tags| tags.iter().any(|v| v.as_i64() == Some(t)))
        })
        .count()
}

/// Writes during the warm-up -- inserts, upserts that change a filtered field, deletes -- do
/// not void it, and the index it installs answers exactly as the rows do.
///
/// A write used to abandon the build and leave it to the first filtered query, which on a
/// server that takes writes is every build: the warm-up never finished and the first filter
/// after every restart paid for the whole table.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial]
async fn a_warm_up_under_writes_is_kept_and_answers_as_the_rows_do() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    {
        let storage = open(&wal).await;
        storage.create_table(table()).await.unwrap();
        for chunk in (0..ROWS).collect::<Vec<_>>().chunks(5_000) {
            storage
                .insert_records("t", chunk.iter().map(|&i| row(i)).collect())
                .await
                .unwrap();
        }
        storage.snapshot().await.unwrap();
    }
    let storage = Arc::new(open(&wal).await);

    let stop = Arc::new(AtomicBool::new(false));
    let writes = Arc::new(AtomicUsize::new(0));
    let writer = {
        let (storage, stop, writes) =
            (Arc::clone(&storage), Arc::clone(&stop), Arc::clone(&writes));
        tokio::spawn(async move {
            let mut n = 0usize;
            while !stop.load(Ordering::Relaxed) {
                // A new row; an existing row moved to tag 42 (an upsert changing the filtered
                // field); an existing row deleted.
                storage
                    .insert_records("t", vec![row(ROWS + n)])
                    .await
                    .unwrap();
                writes.fetch_add(1, Ordering::Relaxed);
                let mut moved = row(n * 7 % ROWS);
                moved.payload = Some(serde_json::json!({ "tags": [42, 1000], "lang": "en" }));
                storage.insert_records("t", vec![moved]).await.unwrap();
                writes.fetch_add(1, Ordering::Relaxed);
                storage
                    .delete_records("t", vec![format!("r{}", (n * 11 + 3) % ROWS)])
                    .await
                    .unwrap();
                writes.fetch_add(1, Ordering::Relaxed);
                n += 1;
                tokio::task::yield_now().await;
            }
        })
    };

    // The writer is running before the build starts, so writes land throughout it.
    while writes.load(Ordering::Relaxed) == 0 {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    let before = writes.load(Ordering::Relaxed);
    let installed = storage.warm_derived("t").await.unwrap();
    let during = writes.load(Ordering::Relaxed) - before;
    stop.store(true, Ordering::Relaxed);
    writer.await.unwrap();

    assert!(
        installed,
        "{during} writes landed during the warm-up and it was not installed"
    );
    assert!(
        during > 0,
        "nothing wrote during the build, so this tests nothing"
    );
    for t in [42, 7, 96, 1000, 1003] {
        let indexed = storage.count_records("t", Some(&tag(t))).await.unwrap();
        assert_eq!(indexed, truth(&storage, t).await, "tag {t}");
    }
}

/// The first filtered query on a cold table builds the index without holding the table map's
/// lock through the build.
///
/// It used to build inside `IntelligentTableData::derived`, under the map's read lock: 27 s at
/// 10M rows during which a writer queued for the write lock, and tokio's write-preferring lock
/// queued every later reader behind it -- every request to every table.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial]
async fn the_first_filtered_query_does_not_hold_the_table_lock_through_the_build() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    {
        let storage = open(&wal).await;
        storage.create_table(table()).await.unwrap();
        for chunk in (0..ROWS).collect::<Vec<_>>().chunks(5_000) {
            storage
                .insert_records("t", chunk.iter().map(|&i| row(i)).collect())
                .await
                .unwrap();
        }
        storage.snapshot().await.unwrap();
    }
    let storage = open(&wal).await;

    dxmt_vectordb::storage::intelligent_indexed_storage::reset_max_table_lock_hold();
    let started = std::time::Instant::now();
    let n = storage.count_records("t", Some(&tag(42))).await.unwrap();
    let took = started.elapsed();
    let held = dxmt_vectordb::storage::intelligent_indexed_storage::max_table_lock_hold();

    assert_eq!(n, (0..ROWS).filter(|i| i % 97 == 42).count());
    assert!(
        held * 4 < took,
        "the first filtered query held the table map's lock for {held:?} of the {took:?} it \
         took; the build has to happen outside it"
    );
}

/// The derived indexes a clean shutdown writes are read back at the next start instead of
/// rebuilt, answer exactly as the rows do, and are used once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial]
async fn derived_indexes_written_at_shutdown_are_read_back_instead_of_rebuilt() {
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
        let page = storage
            .list_records("t", Some(&tag(1003)), 25, 0, false)
            .await
            .unwrap()
            .0
            .into_iter()
            .map(|r| r.id)
            .collect::<Vec<_>>();
        storage.snapshot().await.unwrap();
        assert_eq!(storage.persist_derived().await.unwrap(), 1);
        page
    };
    let derived_dir = wal.join("derived");
    assert_eq!(std::fs::read_dir(&derived_dir).unwrap().count(), 1);

    let storage = open(&wal).await;
    assert_eq!(
        std::fs::read_dir(&derived_dir).unwrap().count(),
        0,
        "the file is used once and deleted"
    );
    assert!(
        !storage.warm_derived("t").await.unwrap(),
        "the indexes were rebuilt rather than read back"
    );
    for t in [42, 7, 96, 1000, 1003] {
        assert_eq!(
            storage.count_records("t", Some(&tag(t))).await.unwrap(),
            truth(&storage, t).await,
            "tag {t}"
        );
    }
    let after: Vec<String> = storage
        .list_records("t", Some(&tag(1003)), 25, 0, false)
        .await
        .unwrap()
        .0
        .into_iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(
        after, before,
        "a filtered page comes back in the same order"
    );
}

/// A file from an earlier shutdown is not used once the records have moved on: here it is put
/// back after a later session wrote a row, and the next start rebuilds instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial]
async fn derived_indexes_from_before_a_later_write_are_not_used() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    {
        let storage = open(&wal).await;
        storage.create_table(table()).await.unwrap();
        storage
            .insert_records("t", (0..ROWS).map(row).collect())
            .await
            .unwrap();
        storage.count_records("t", Some(&tag(42))).await.unwrap();
        storage.snapshot().await.unwrap();
        storage.persist_derived().await.unwrap();
    }
    let derived_dir = wal.join("derived");
    let file = std::fs::read_dir(&derived_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let saved = dir.path().join("saved.bin");
    std::fs::copy(&file, &saved).unwrap();
    {
        // A later session, ended by a crash rather than a clean shutdown, writes a row.
        let storage = open(&wal).await;
        storage
            .insert_records("t", vec![row(ROWS + 42 * 97)])
            .await
            .unwrap();
    }
    std::fs::copy(&saved, &file).unwrap();

    let storage = open(&wal).await;
    assert!(
        storage.warm_derived("t").await.unwrap(),
        "a stale file was installed instead of a rebuild"
    );
    let expect = (0..ROWS)
        .chain(std::iter::once(ROWS + 42 * 97))
        .filter(|i| i % 97 == 42)
        .count();
    assert_eq!(
        storage.count_records("t", Some(&tag(42))).await.unwrap(),
        expect
    );
}
