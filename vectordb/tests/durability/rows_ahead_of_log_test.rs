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
#[serial_test::serial] // both set DXMT_BACKUP_ROOT
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

async fn ids(storage: &IntelligentIndexedStorage) -> Vec<String> {
    let (records, _) = storage
        .list_records("t", None, 100_000, 0, false)
        .await
        .unwrap();
    let mut ids: Vec<String> = records.into_iter().map(|r| r.id).collect();
    ids.sort_by_key(|id| id[1..].parse::<usize>().unwrap());
    ids
}

fn copy_files(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::copy(&path, to.join(path.file_name().unwrap())).unwrap();
        }
    }
}

/// The tree a backup used to be able to produce: its own snapshot, and a `wal.log` copied after
/// a second snapshot had compacted the log past the first one's replay point.
///
/// `backup_to` took a snapshot and then copied, with no lock held across the copy, so the
/// periodic snapshot could run in between. The entries between the two replay points were then
/// in neither half of the backup, and replay started from the first snapshot's point without
/// noticing that the log began later. Inserts in that gap come back from the row files; a
/// delete has nothing to come back from, so the restore holds a record the source had deleted.
#[tokio::test]
async fn a_log_that_starts_after_the_snapshot_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    let assembled = dir.path().join("assembled");

    {
        let storage = open(&wal).await;
        storage.create_table(table("t")).await.unwrap();
        storage.insert_records("t", rows(0, 10)).await.unwrap();
        storage.snapshot().await.unwrap();
        copy_files(&wal.join("snapshot"), &assembled.join("snapshot"));

        storage
            .delete_records("t", vec!["r3".to_string()])
            .await
            .unwrap();
        storage.insert_records("t", rows(10, 12)).await.unwrap();
        storage.snapshot().await.unwrap();
        storage.insert_records("t", rows(12, 13)).await.unwrap();
        drop(storage);
    }
    for sub in ["records", "vectors"] {
        copy_files(&wal.join(sub), &assembled.join(sub));
    }
    std::fs::copy(wal.join("wal.log"), assembled.join("wal.log")).unwrap();

    let err = IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(assembled))
        .await
        .err()
        .expect("a log with a hole in it restored without complaint");
    let message = err.to_string();
    assert!(
        message.contains("snapshot") && message.contains("wal.log"),
        "the refusal should say which two files disagree: {message}"
    );
}

/// Backups taken while a writer inserts and deletes and the periodic snapshot keeps running.
///
/// The writer keeps a sliding window: insert five rows at the top, delete the five at the
/// bottom, so every state it passes through is a contiguous run of ids 200 to 205 long. A
/// restore that holds anything else -- a deleted row back, or a run with a hole in it -- is
/// not a point the source was ever at. The snapshotter is what made this fail: a snapshot
/// between the backup's own and its copy of `wal.log` compacted the log the backup needed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial] // both set DXMT_BACKUP_ROOT
async fn a_backup_under_deletes_and_snapshots_restores_to_a_state_the_source_was_in() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    let backups = dir.path().join("backups");
    std::fs::create_dir_all(&backups).unwrap();
    std::env::set_var("DXMT_BACKUP_ROOT", &backups);

    let storage = Arc::new(open(&wal).await);
    storage.create_table(table("t")).await.unwrap();
    storage.insert_records("t", rows(0, 200)).await.unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (storage, stop) = (Arc::clone(&storage), Arc::clone(&stop));
        tokio::spawn(async move {
            let mut next = 200usize;
            while !stop.load(Ordering::Relaxed) {
                storage
                    .insert_records("t", rows(next, next + 5))
                    .await
                    .unwrap();
                let gone = (next - 200..next - 195).map(|i| format!("r{i}")).collect();
                storage.delete_records("t", gone).await.unwrap();
                next += 5;
            }
        })
    };
    let snapshotter = {
        let (storage, stop) = (Arc::clone(&storage), Arc::clone(&stop));
        tokio::spawn(async move {
            let mut taken = 0usize;
            while !stop.load(Ordering::Relaxed) {
                storage.snapshot().await.unwrap();
                taken += 1;
                tokio::task::yield_now().await;
            }
            taken
        })
    };

    let mut taken = Vec::new();
    for n in 0..10 {
        let dest = backups.join(format!("b{n}"));
        storage.backup_to(&dest).await.unwrap();
        taken.push(dest);
    }
    stop.store(true, Ordering::Relaxed);
    writer.await.unwrap();
    assert!(
        snapshotter.await.unwrap() > 0,
        "no snapshot ran beside the backups"
    );
    drop(storage);

    for dest in taken {
        let label = dest.display().to_string();
        let restored = IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(dest))
            .await
            .unwrap_or_else(|e| panic!("{label}: the backup does not open: {e}"));
        let held = ids(&restored).await;
        let numbers: Vec<usize> = held.iter().map(|id| id[1..].parse().unwrap()).collect();
        let (lo, hi) = (numbers[0], numbers[numbers.len() - 1] + 1);
        assert_eq!(
            numbers,
            (lo..hi).collect::<Vec<_>>(),
            "{label}: the restored ids are not one contiguous run"
        );
        assert!(
            (200..=205).contains(&(hi - lo)),
            "{label}: {} rows, r{lo}..r{hi}; the source never held that",
            hi - lo
        );
        assert_eq!(
            restored.get_index_record_count("t").await.unwrap(),
            held.len(),
            "{label}: the index and the rows disagree"
        );
        for i in [lo, hi - 1] {
            assert_eq!(
                top1(&restored, i, SearchMode::Normal).await.as_deref(),
                Some(format!("r{i}").as_str()),
                "{label}"
            );
        }
        if lo > 0 {
            assert_ne!(
                top1(&restored, lo - 1, SearchMode::Normal).await.as_deref(),
                Some(format!("r{}", lo - 1).as_str()),
                "{label}: a deleted row is back"
            );
        }
    }
}

/// A snapshot asked for while a backup is copying waits for the copy.
///
/// Deterministic version of the race above: the backup is stopped between its snapshot and its
/// copy, and a delete and a second snapshot are issued meanwhile. Before `backup_to` held the
/// snapshot lock across the copy, that snapshot ran at once and compacted the log past the
/// backup's replay point, so the backup's `wal.log` started later than its snapshot -- which
/// recovery now refuses, so this fails at the open rather than with a deleted row restored.
#[cfg(feature = "test-failpoints")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial] // sets DXMT_BACKUP_ROOT
async fn a_snapshot_during_a_backup_waits_for_the_copy() {
    use std::sync::Arc;
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("wal");
    let backups = dir.path().join("backups");
    std::fs::create_dir_all(&backups).unwrap();
    std::env::set_var("DXMT_BACKUP_ROOT", &backups);
    let dest = backups.join("b");

    let storage = Arc::new(open(&wal).await);
    storage.create_table(table("t")).await.unwrap();
    storage.insert_records("t", rows(0, 20)).await.unwrap();

    let (reached, resume) = storage.pause_next_backup_for_test();
    let backup = {
        let (storage, dest) = (Arc::clone(&storage), dest.clone());
        tokio::spawn(async move { storage.backup_to(&dest).await })
    };
    reached.await.unwrap();

    storage
        .delete_records("t", vec!["r3".to_string()])
        .await
        .unwrap();
    let snapshot = {
        let storage = Arc::clone(&storage);
        tokio::spawn(async move { storage.snapshot().await })
    };
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !snapshot.is_finished(),
        "a snapshot ran in the middle of a backup's copy"
    );

    resume.send(()).unwrap();
    backup.await.unwrap().unwrap();
    snapshot.await.unwrap().unwrap();
    drop(storage);

    let restored = IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(dest))
        .await
        .expect("the backup does not open");
    let held = ids(&restored).await;
    assert!(!held.contains(&"r3".to_string()), "a deleted row is back");
    assert_eq!(held.len(), 19);
    assert_eq!(restored.get_index_record_count("t").await.unwrap(), 19);
}
