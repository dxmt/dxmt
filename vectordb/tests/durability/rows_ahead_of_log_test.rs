//! Rows the index does not hold, after a restart.
//!
//! A write reaches the row file, fsynced, before its WAL entry is appended, so a crash in
//! that window -- or a backup whose `records/` copy was taken after a write and whose
//! `wal.log` copy before it -- leaves a record in the store that neither the snapshot's index
//! nor the log knows about. `GET` returns it and the exact scan finds it, and no ordinary
//! search ever will. The window is reproduced here exactly: snapshot, write more rows, cut
//! `wal.log` back to the length it had at the snapshot, reopen.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};

const DIM: usize = 16;

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
        rerank: None,
    }
}

fn vector(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| ((i * 31 + d * 17) % 97) as f32 / 97.0 + i as f32 * 0.001)
        .collect()
}

fn rows(from: usize, to: usize) -> Vec<CreateRecordRequest> {
    (from..to)
        .map(|i| CreateRecordRequest {
            id: format!("r{i}"),
            vector: vector(i),
            sparse_vector: None,
            named_vectors: None,
            payload: None,
            text: None,
        })
        .collect()
}

async fn top1(storage: &IntelligentIndexedStorage, i: usize, mode: SearchMode) -> Option<String> {
    storage
        .search(
            "t",
            SearchQuery {
                vector: vector(i),
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
        .map(|h| h.id.clone())
}

#[tokio::test]
async fn rows_the_log_never_saw_are_indexed_on_restart() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    let log = wal.join("wal.log");

    {
        let storage = open(&wal).await;
        storage.create_table(table("t")).await.unwrap();
        storage.insert_records("t", rows(0, 300)).await.unwrap();
        storage.snapshot().await.unwrap();
        let at_snapshot = std::fs::metadata(&log).map(|m| m.len()).unwrap_or(0);

        // The writes the log is about to lose. Their rows are fsynced and their WAL entries
        // are appended after; cutting the log back to the snapshot is the crash between.
        storage.insert_records("t", rows(300, 340)).await.unwrap();
        assert_eq!(storage.get_index_record_count("t").await.unwrap(), 340);
        drop(storage);
        let f = std::fs::OpenOptions::new().write(true).open(&log).unwrap();
        f.set_len(at_snapshot).unwrap();
        f.sync_all().unwrap();
    }

    let storage = open(&wal).await;
    let stored = storage.get_table("t").await.unwrap().record_count;
    assert_eq!(
        stored, 340,
        "the row files hold every write; that is what they are for"
    );
    assert_eq!(
        storage.get_index_record_count("t").await.unwrap(),
        340,
        "the index has to hold what the rows hold"
    );
    for i in [300, 320, 339] {
        assert_eq!(
            top1(&storage, i, SearchMode::Exact).await.as_deref(),
            Some(format!("r{i}").as_str())
        );
        assert_eq!(
            top1(&storage, i, SearchMode::Normal).await.as_deref(),
            Some(format!("r{i}").as_str()),
            "r{i} is in the table and an ordinary search cannot find it"
        );
    }
    // And a second restart has nothing left to reconcile.
    drop(storage);
    let storage = open(&wal).await;
    assert_eq!(storage.get_index_record_count("t").await.unwrap(), 340);
}

/// A backup taken while writes land restores to one consistent point inside the window.
///
/// `backup_to` copies the snapshot, then the row files, then `wal.log`, with inserts
/// arriving throughout. The claims a restore can make are: every record it holds is in the
/// index and found by an ordinary search (no row the log never saw, see the test above), the
/// record count is not below what the table held when the backup started, and not above
/// what it held when the backup finished. Five backups, each against a writer that does not
/// pause, so the copy order is exercised against writes at different points.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_backup_under_writes_restores_to_a_point_inside_the_window() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    let backups = dir.path().join("backups");
    std::fs::create_dir_all(&backups).unwrap();
    std::env::set_var("DXMT_BACKUP_ROOT", &backups);

    let storage = std::sync::Arc::new(open(&wal).await);
    storage.create_table(table("t")).await.unwrap();
    storage.insert_records("t", rows(0, 200)).await.unwrap();

    let writer = {
        let storage = std::sync::Arc::clone(&storage);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = std::sync::Arc::clone(&stop);
        let handle = tokio::spawn(async move {
            let mut next = 200usize;
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                storage
                    .insert_records("t", rows(next, next + 5))
                    .await
                    .unwrap();
                next += 5;
            }
            next
        });
        (stop, handle)
    };

    let mut windows = Vec::new();
    for n in 0..5 {
        let before = storage.get_table("t").await.unwrap().record_count;
        let dest = backups.join(format!("b{n}"));
        storage.backup_to(&dest).await.unwrap();
        let after = storage.get_table("t").await.unwrap().record_count;
        windows.push((dest, before, after));
    }
    writer.0.store(true, std::sync::atomic::Ordering::Relaxed);
    let written = writer.1.await.unwrap();
    drop(storage);
    assert!(
        windows.iter().any(|(_, before, after)| after > before),
        "no write landed during any backup, so nothing below is tested: {windows:?}"
    );

    for (dest, before, after) in windows {
        let restored = open(&dest).await;
        let held = restored.get_table("t").await.unwrap().record_count;
        assert!(
            before <= held && held <= after,
            "{}: restored {held} rows, the backup window was {before}..={after}",
            dest.display()
        );
        assert_eq!(
            restored.get_index_record_count("t").await.unwrap() as u64,
            held,
            "{}: every restored row is in the index",
            dest.display()
        );
        // The last row the restore holds is found by an ordinary search; the first row
        // past it is not there at all, in any mode.
        let last = held as usize - 1;
        assert_eq!(
            top1(&restored, last, SearchMode::Normal).await.as_deref(),
            Some(format!("r{last}").as_str())
        );
        if (held as usize) < written {
            assert_ne!(
                top1(&restored, held as usize, SearchMode::Exact)
                    .await
                    .as_deref(),
                Some(format!("r{held}").as_str()),
                "{}: a row past the backup's point is present",
                dest.display()
            );
        }
    }
}
