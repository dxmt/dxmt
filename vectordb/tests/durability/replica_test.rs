//! A read replica follows the store a primary mirrors to object storage.
//!
//! Both run in this process against one filesystem-backed object store standing in for S3,
//! each with its own data directory. What matters: the replica has the primary's data, applies
//! what the primary writes afterwards, refuses writes, never writes the bucket (two writers
//! would overwrite each other's WAL manifest), and says so when it has fallen too far behind
//! to catch up from the WAL chunks.

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

fn config(base: &Path) -> StorageConfig {
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
            bucket: "bucket".to_string(),
            region: None,
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
            prefix: None,
        }),
    }
}

fn table() -> TableConfig {
    TableConfig {
        name: "t".to_string(),
        dimension: 4,
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

fn record(id: &str, x: f32) -> CreateRecordRequest {
    CreateRecordRequest {
        id: id.to_string(),
        vector: vec![x, 0.0, 0.0, 0.0],
        sparse_vector: None,
        named_vectors: None,
        payload: Some(serde_json::json!({"v": x})),
        text: None,
    }
}

async fn ids(storage: &IntelligentIndexedStorage) -> Vec<String> {
    let mut ids: Vec<String> = storage
        .list_records("t", None, 1000, 0, false)
        .await
        .unwrap()
        .0
        .into_iter()
        .map(|r| r.id)
        .collect();
    ids.sort();
    ids
}

fn manifest(base: &Path) -> Vec<u8> {
    std::fs::read(base.join("ssd").join("bucket").join("wal").join("MANIFEST"))
        .expect("the primary uploaded a WAL manifest")
}

#[tokio::test]
async fn a_replica_follows_the_primary_and_refuses_writes() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let primary = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("primary")))
        .await
        .unwrap();
    primary.create_table(table()).await.unwrap();
    primary
        .insert_records("t", vec![record("a", 1.0), record("b", 2.0)])
        .await
        .unwrap();

    let replica = IntelligentIndexedStorage::new_replica(&cfg, tmp.path().join("replica"))
        .await
        .unwrap();
    assert!(replica.is_replica());
    assert_eq!(
        ids(&replica).await,
        vec!["a", "b"],
        "recovered from the bucket"
    );
    let before = manifest(&shared);

    // Every write is refused, with the code that says where to send it.
    let refusals = [
        replica
            .insert_records("t", vec![record("x", 9.0)])
            .await
            .map(|_| ())
            .unwrap_err(),
        replica
            .delete_records("t", vec!["a".into()])
            .await
            .map(|_| ())
            .unwrap_err(),
        replica
            .create_table(TableConfig {
                name: "u".into(),
                ..table()
            })
            .await
            .map(|_| ())
            .unwrap_err(),
        replica.delete_table("t").await.unwrap_err(),
        replica.snapshot().await.map(|_| ()).unwrap_err(),
    ];
    for err in refusals {
        assert_eq!(err.status_code(), 403, "{err}");
        assert!(err.to_string().contains("replica"), "{err}");
    }

    // The primary writes on; the replica applies it.
    primary
        .insert_records("t", vec![record("c", 3.0)])
        .await
        .unwrap();
    primary.delete_records("t", vec!["a".into()]).await.unwrap();
    primary
        .insert_records("t", vec![record("b", 20.0)])
        .await
        .unwrap();
    let applied = replica.follow_object_storage().await.unwrap();
    assert!(applied >= 3, "applied {applied}");
    assert_eq!(ids(&replica).await, vec!["b", "c"]);
    let b = replica.get_record("t", &"b".to_string()).await.unwrap();
    assert_eq!(b.payload["v"], 20.0, "the update reached the replica");
    let hits = replica
        .search(
            "t",
            SearchQuery {
                vector: vec![3.0, 0.0, 0.0, 0.0],
                top_k: 1,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        hits.hits[0].id, "c",
        "the replica's index has the new record"
    );
    assert_eq!(
        replica.follow_object_storage().await.unwrap(),
        0,
        "nothing new"
    );

    // The replica wrote nothing to the bucket: the manifest is exactly the primary's.
    let primary_next = primary.health_snapshot().await.wal_mirrored_next_sequence;
    let replica_next = replica
        .health_snapshot()
        .await
        .replica_applied_next_sequence;
    assert_eq!(replica_next, primary_next, "caught up");
    assert_ne!(manifest(&shared), before, "the primary's writes moved it");
    drop(replica);
    let after_replica = manifest(&shared);
    primary
        .insert_records("t", vec![record("d", 4.0)])
        .await
        .unwrap();
    assert_ne!(
        manifest(&shared),
        after_replica,
        "and only the primary moves it"
    );
}

#[tokio::test]
async fn a_replica_too_far_behind_says_so_and_a_restart_recovers_it() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let primary = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("primary")))
        .await
        .unwrap();
    // No retention window: every snapshot prunes the chunks it covers at once.
    primary.set_wal_retention(std::time::Duration::ZERO);
    primary.create_table(table()).await.unwrap();
    primary
        .insert_records("t", vec![record("a", 1.0)])
        .await
        .unwrap();

    let replica = IntelligentIndexedStorage::new_replica(&cfg, tmp.path().join("replica"))
        .await
        .unwrap();
    assert!(replica.health_snapshot().await.ready);

    // The replica does not follow while the primary writes, snapshots (pruning the chunks
    // the replica has not applied) and writes again.
    primary
        .insert_records("t", vec![record("b", 2.0)])
        .await
        .unwrap();
    primary.snapshot().await.unwrap();
    primary
        .insert_records("t", vec![record("c", 3.0)])
        .await
        .unwrap();

    assert_eq!(replica.follow_object_storage().await.unwrap(), 0);
    let health = replica.health_snapshot().await;
    assert!(!health.ready, "a stale replica must not report ready");
    assert!(
        health
            .replica_stale
            .as_deref()
            .unwrap_or("")
            .contains("restart"),
        "{:?}",
        health.replica_stale
    );
    assert_eq!(
        ids(&replica).await,
        vec!["a"],
        "it applied nothing out of order"
    );

    // A restart recovers from the primary's snapshot and the chunks after it.
    drop(replica);
    let restarted = IntelligentIndexedStorage::new_replica(&cfg, tmp.path().join("replica-2"))
        .await
        .unwrap();
    assert_eq!(ids(&restarted).await, vec!["a", "b", "c"]);
    assert!(restarted.health_snapshot().await.ready);
}

#[tokio::test]
async fn a_replica_needs_object_storage() {
    let tmp = TempDir::new().unwrap();
    let mut cfg = config(tmp.path());
    cfg.object_storage = None;
    let err = IntelligentIndexedStorage::new_replica(&cfg, tmp.path().join("r"))
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(err.to_string().contains("object storage"), "{err}");
}

#[tokio::test]
async fn a_replica_refuses_a_directory_holding_someone_else_s_data() {
    let tmp = TempDir::new().unwrap();
    let cfg = config(&tmp.path().join("shared"));
    let primary_dir = tmp.path().join("primary");
    {
        let primary = IntelligentIndexedStorage::new_with_wal(&cfg, Some(primary_dir.clone()))
            .await
            .unwrap();
        primary.create_table(table()).await.unwrap();
        primary
            .insert_records("t", vec![record("a", 1.0)])
            .await
            .unwrap();
    }
    let before: Vec<_> = std::fs::read_dir(&primary_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .collect();
    let err = IntelligentIndexedStorage::new_replica(&cfg, primary_dir.clone())
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(err.to_string().contains("not a replica's"), "{err}");
    let after: Vec<_> = std::fs::read_dir(&primary_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .collect();
    assert_eq!(
        before.len(),
        after.len(),
        "the primary's directory was touched"
    );
}

#[tokio::test]
async fn a_stale_replica_restarted_on_its_own_directory_starts_clean() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let primary = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("primary")))
        .await
        .unwrap();
    primary.set_wal_retention(std::time::Duration::ZERO);
    primary.create_table(table()).await.unwrap();
    primary
        .insert_records("t", vec![record("a", 1.0)])
        .await
        .unwrap();
    let replica_dir = tmp.path().join("replica");
    let replica = IntelligentIndexedStorage::new_replica(&cfg, replica_dir.clone())
        .await
        .unwrap();
    primary
        .insert_records("t", vec![record("b", 2.0)])
        .await
        .unwrap();
    primary.snapshot().await.unwrap();
    primary
        .insert_records("t", vec![record("c", 3.0)])
        .await
        .unwrap();
    replica.follow_object_storage().await.unwrap();
    assert!(!replica.health_snapshot().await.ready);
    drop(replica);

    // The same directory: emptied and rebuilt from the bucket, not mixed with the old state.
    let restarted = IntelligentIndexedStorage::new_replica(&cfg, replica_dir)
        .await
        .unwrap();
    assert_eq!(ids(&restarted).await, vec!["a", "b", "c"]);
    assert!(restarted.health_snapshot().await.ready);
}

#[tokio::test]
async fn a_replica_is_promoted_by_starting_a_primary_on_its_directory() {
    std::env::set_var("DXMT_PRIMARY_CHECK_MS", "0");
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let replica_dir = tmp.path().join("replica");
    {
        let primary =
            IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("primary")))
                .await
                .unwrap();
        primary.create_table(table()).await.unwrap();
        primary
            .insert_records("t", vec![record("a", 1.0), record("b", 2.0)])
            .await
            .unwrap();
        let replica = IntelligentIndexedStorage::new_replica(&cfg, replica_dir.clone())
            .await
            .unwrap();
        assert_eq!(ids(&replica).await, vec!["a", "b"]);
        // The primary is lost here.
    }
    let promoted = IntelligentIndexedStorage::new_with_wal(&cfg, Some(replica_dir))
        .await
        .unwrap();
    assert!(!promoted.is_replica());
    assert_eq!(ids(&promoted).await, vec!["a", "b"]);
    promoted
        .insert_records("t", vec![record("c", 3.0)])
        .await
        .unwrap();

    let follower = IntelligentIndexedStorage::new_replica(&cfg, tmp.path().join("replica-2"))
        .await
        .unwrap();
    assert_eq!(ids(&follower).await, vec!["a", "b", "c"]);
    promoted
        .insert_records("t", vec![record("d", 4.0)])
        .await
        .unwrap();
    follower.follow_object_storage().await.unwrap();
    assert_eq!(ids(&follower).await, vec!["a", "b", "c", "d"]);
}

#[tokio::test]
async fn a_primary_that_another_has_replaced_stops_writing() {
    std::env::set_var("DXMT_PRIMARY_CHECK_MS", "0");
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let first = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("one")))
        .await
        .unwrap();
    first.create_table(table()).await.unwrap();
    first
        .insert_records("t", vec![record("a", 1.0)])
        .await
        .unwrap();

    // A second primary takes the same bucket (say, a replica promoted while the first was only
    // unreachable, not dead).
    let second = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("two")))
        .await
        .unwrap();
    assert_eq!(ids(&second).await, vec!["a"]);

    let err = first
        .insert_records("t", vec![record("x", 9.0)])
        .await
        .map(|_| ())
        .unwrap_err();
    assert_eq!(err.status_code(), 503, "{err}");
    assert!(err.to_string().contains("another primary"), "{err}");
    let health = first.health_snapshot().await;
    assert!(!health.ready, "a replaced primary must not report ready");
    // Every later write is refused too, without asking the bucket again.
    assert!(first
        .insert_records("t", vec![record("y", 8.0)])
        .await
        .is_err());

    second
        .insert_records("t", vec![record("b", 2.0)])
        .await
        .unwrap();
    let replica = IntelligentIndexedStorage::new_replica(&cfg, tmp.path().join("replica"))
        .await
        .unwrap();
    assert_eq!(
        ids(&replica).await,
        vec!["a", "b"],
        "only the current primary's writes"
    );
}

#[tokio::test]
async fn a_replaced_primary_refuses_to_start_on_its_old_directory() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let one = tmp.path().join("one");
    let first = IntelligentIndexedStorage::new_with_wal(&cfg, Some(one.clone()))
        .await
        .unwrap();
    first.create_table(table()).await.unwrap();
    first
        .insert_records("t", vec![record("a", 1.0)])
        .await
        .unwrap();
    drop(first);
    // Restarting where it left off is the ordinary case and must keep working.
    let first = IntelligentIndexedStorage::new_with_wal(&cfg, Some(one.clone()))
        .await
        .unwrap();
    drop(first);

    // The first primary is gone; another machine takes over the bucket and writes.
    let two = tmp.path().join("two");
    let second = IntelligentIndexedStorage::new_with_wal(&cfg, Some(two.clone()))
        .await
        .unwrap();
    second
        .insert_records("t", vec![record("b", 2.0)])
        .await
        .unwrap();

    // The first comes back -- a supervisor restarting it, say. Its log stops before `b`, and
    // taking the bucket back would depose the primary that wrote `b`.
    let err = IntelligentIndexedStorage::new_with_wal(&cfg, Some(one.clone()))
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(err.to_string().contains("since"), "{err}");
    second
        .insert_records("t", vec![record("c", 3.0)])
        .await
        .expect("the current primary is undisturbed");
    drop(second);

    // The way back for that machine: an empty directory, which recovers from the bucket.
    let again = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("one-new")))
        .await
        .unwrap();
    assert_eq!(ids(&again).await, vec!["a", "b", "c"]);
    // And the machine that just handed over may not take it back on its old directory either.
    drop(again);
    assert!(IntelligentIndexedStorage::new_with_wal(&cfg, Some(two))
        .await
        .is_err());
}

#[tokio::test]
async fn a_replica_behind_a_snapshot_catches_up_from_the_retained_chunks() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let primary = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("primary")))
        .await
        .unwrap();
    primary.set_wal_retention(std::time::Duration::from_secs(3600));
    primary.create_table(table()).await.unwrap();
    primary
        .insert_records("t", vec![record("a", 1.0)])
        .await
        .unwrap();
    let replica = IntelligentIndexedStorage::new_replica(&cfg, tmp.path().join("replica"))
        .await
        .unwrap();

    // The same sequence that makes a replica stale without a window: it does not follow while
    // the primary writes, snapshots, and writes again -- twice.
    for (id, x) in [("b", 2.0), ("c", 3.0)] {
        primary
            .insert_records("t", vec![record(id, x)])
            .await
            .unwrap();
        primary.snapshot().await.unwrap();
    }
    primary
        .insert_records("t", vec![record("d", 4.0)])
        .await
        .unwrap();

    assert!(replica.follow_object_storage().await.unwrap() > 0);
    assert_eq!(ids(&replica).await, vec!["a", "b", "c", "d"]);
    let health = replica.health_snapshot().await;
    assert!(health.ready && health.replica_stale.is_none());
}

/// A manifest write that fails must leave nothing behind in memory: the retry rewrites the
/// chunk under the same key with more entries, and a stale entry for the first attempt made
/// the manifest list one object twice, with two ranges. GCS's per-object write limit is what
/// made manifest writes fail in practice; an unopenable lock file does the same here while
/// chunk writes, in their own directory, still succeed.
#[cfg(unix)]
#[tokio::test]
async fn a_manifest_write_that_failed_leaves_no_trace_in_the_manifest() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let primary = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("primary")))
        .await
        .unwrap();
    primary.create_table(table()).await.unwrap();
    primary
        .insert_records("t", vec![record("a", 1.0)])
        .await
        .unwrap();

    // The manifest is written under a lock file beside it and replaced by rename; a lock
    // file that cannot be opened for writing is what makes that write fail.
    let path = shared
        .join("ssd")
        .join("bucket")
        .join("wal")
        .join("MANIFEST.lock");
    assert!(path.exists(), "the manifest is written conditionally here");
    let set_mode = |mode| std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode));
    set_mode(0o444).unwrap();
    if std::fs::OpenOptions::new().write(true).open(&path).is_ok() {
        set_mode(0o644).unwrap();
        eprintln!("skipped: this user can write a read-only file (root?)");
        return;
    }
    let before = manifest(&shared);
    primary
        .insert_records("t", vec![record("b", 2.0)])
        .await
        .expect("the write is durable locally; the mirror's failure is not the caller's");
    assert_eq!(
        manifest(&shared),
        before,
        "the manifest write was meant to fail"
    );
    set_mode(0o644).unwrap();
    primary
        .insert_records("t", vec![record("c", 3.0)])
        .await
        .unwrap();

    let m: serde_json::Value = serde_json::from_slice(&manifest(&shared)).unwrap();
    let chunks = m["chunks"].as_array().unwrap();
    let mut paths: Vec<&str> = chunks.iter().map(|c| c["path"].as_str().unwrap()).collect();
    paths.dedup();
    assert_eq!(paths.len(), chunks.len(), "an object listed twice: {m}");
    for pair in chunks.windows(2) {
        assert_eq!(
            pair[1]["first_seq"].as_u64(),
            pair[0]["last_seq"].as_u64().map(|s| s + 1),
            "chunks must be contiguous and not overlap: {m}"
        );
    }
    let live: u64 = chunks.iter().map(|c| c["entries"].as_u64().unwrap()).sum();
    assert_eq!(
        live + m["pruned_entries"].as_u64().unwrap(),
        m["total_entries"].as_u64().unwrap(),
        "{m}"
    );
    let replica = IntelligentIndexedStorage::new_replica(&cfg, tmp.path().join("replica"))
        .await
        .unwrap();
    assert_eq!(ids(&replica).await, vec!["a", "b", "c"]);
}

/// A write logged before a snapshot and mirrored after it is still uploaded as a chunk. The
/// snapshot holds it, so recovery never missed it; a replica follows chunks, and the mirror
/// used to skip straight to the snapshot's point, leaving a hole every following replica fell
/// into -- on GCS, whose slow uploads make that window most snapshots, once a minute.
#[cfg(unix)]
#[tokio::test]
async fn a_write_mirrored_after_a_snapshot_still_reaches_the_replicas() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let primary = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("primary")))
        .await
        .unwrap();
    primary.create_table(table()).await.unwrap();
    primary
        .insert_records("t", vec![record("a", 1.0)])
        .await
        .unwrap();
    let replica = IntelligentIndexedStorage::new_replica(&cfg, tmp.path().join("replica"))
        .await
        .unwrap();

    // `b` is logged and not mirrored (its manifest write fails), then a snapshot is taken.
    // The manifest is written under a lock file beside it and replaced by rename; a lock
    // file that cannot be opened for writing is what makes that write fail.
    let path = shared
        .join("ssd")
        .join("bucket")
        .join("wal")
        .join("MANIFEST.lock");
    assert!(path.exists(), "the manifest is written conditionally here");
    let set_mode = |mode| std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode));
    set_mode(0o444).unwrap();
    if std::fs::OpenOptions::new().write(true).open(&path).is_ok() {
        set_mode(0o644).unwrap();
        eprintln!("skipped: this user can write a read-only file (root?)");
        return;
    }
    let before = manifest(&shared);
    primary
        .insert_records("t", vec![record("b", 2.0)])
        .await
        .unwrap();
    assert_eq!(
        manifest(&shared),
        before,
        "the manifest write was meant to fail"
    );
    // Local-only: publishing a snapshot is a manifest write too.
    primary.snapshot().await.unwrap();
    set_mode(0o644).unwrap();
    primary
        .insert_records("t", vec![record("c", 3.0)])
        .await
        .unwrap();

    assert!(replica.follow_object_storage().await.unwrap() > 0);
    let health = replica.health_snapshot().await;
    assert!(health.replica_stale.is_none(), "{:?}", health.replica_stale);
    assert_eq!(ids(&replica).await, vec!["a", "b", "c"]);
}

/// With a store that honours conditional writes (the filesystem one does, as S3 and MinIO do),
/// the manifest is a lock: a replaced primary is refused on its very next write, not within
/// `DXMT_PRIMARY_CHECK_MS`, and the refusal says the manifest changed under it.
#[tokio::test]
async fn a_replaced_primary_is_refused_at_its_next_write_by_the_manifest() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let first = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("one")))
        .await
        .unwrap();
    first.create_table(table()).await.unwrap();
    first
        .insert_records("t", vec![record("a", 1.0)])
        .await
        .unwrap();
    let second = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("two")))
        .await
        .unwrap();

    let err = first
        .insert_records("t", vec![record("x", 9.0)])
        .await
        .map(|_| ())
        .unwrap_err();
    assert_eq!(err.status_code(), 503, "{err}");
    assert!(err.to_string().contains("manifest changed"), "{err}");

    second
        .insert_records("t", vec![record("b", 2.0)])
        .await
        .unwrap();
    let replica = IntelligentIndexedStorage::new_replica(&cfg, tmp.path().join("replica"))
        .await
        .unwrap();
    assert_eq!(ids(&replica).await, vec!["a", "b"]);
}

/// A replaced primary that never writes again -- idle, so nothing told it -- still takes
/// periodic snapshots. Publishing one is a conditional write of the manifest, so it fails, and
/// the bucket keeps the new primary's snapshot: a machine recovering from it has `b`.
#[tokio::test]
async fn a_replaced_primary_cannot_publish_a_snapshot() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let cfg = config(&shared);
    let first = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("one")))
        .await
        .unwrap();
    first.create_table(table()).await.unwrap();
    first
        .insert_records("t", vec![record("a", 1.0)])
        .await
        .unwrap();
    let second = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("two")))
        .await
        .unwrap();
    second
        .insert_records("t", vec![record("b", 2.0)])
        .await
        .unwrap();
    second.snapshot().await.unwrap();
    let published: serde_json::Value = serde_json::from_slice(&manifest(&shared)).unwrap();

    // The old one's snapshot is written locally and then refused at the bucket.
    let _ = first.snapshot().await;
    let after: serde_json::Value = serde_json::from_slice(&manifest(&shared)).unwrap();
    assert_eq!(after["snapshot"], published["snapshot"], "{after}");
    assert!(first
        .insert_records("t", vec![record("y", 7.0)])
        .await
        .is_err());

    drop(second);
    let fresh = IntelligentIndexedStorage::new_with_wal(&cfg, Some(tmp.path().join("three")))
        .await
        .unwrap();
    assert_eq!(ids(&fresh).await, vec!["a", "b"]);
}

fn config_with_prefix(base: &Path, prefix: &str) -> StorageConfig {
    let mut cfg = config(base);
    if let Some(o) = cfg.object_storage.as_mut() {
        o.prefix = Some(prefix.to_string());
    }
    cfg
}

/// Two servers sharing one bucket under different prefixes -- what a cluster's shards are --
/// see only their own: each recovers its own tables, a replica follows its own prefix, and the
/// objects are where the prefix says.
#[tokio::test]
async fn servers_under_different_prefixes_share_a_bucket_without_seeing_each_other() {
    let tmp = TempDir::new().unwrap();
    let shared = tmp.path().join("shared");
    let a = config_with_prefix(&shared, "shards/a");
    let b = config_with_prefix(&shared, "shards/b");
    for (cfg, dir, id) in [(&a, "pa", "in-a"), (&b, "pb", "in-b")] {
        let primary = IntelligentIndexedStorage::new_with_wal(cfg, Some(tmp.path().join(dir)))
            .await
            .unwrap();
        primary.create_table(table()).await.unwrap();
        primary
            .insert_records("t", vec![record(id, 1.0)])
            .await
            .unwrap();
        primary.snapshot().await.unwrap();
        primary
            .insert_records("t", vec![record(&format!("{id}-tail"), 2.0)])
            .await
            .unwrap();
    }
    let bucket = shared.join("ssd").join("bucket");
    assert!(bucket.join("shards/a/wal/MANIFEST").exists());
    assert!(bucket.join("shards/b/wal/MANIFEST").exists());
    assert!(!bucket.join("wal").exists(), "nothing at the bucket's root");

    let fresh_a = IntelligentIndexedStorage::new_with_wal(&a, Some(tmp.path().join("fa")))
        .await
        .unwrap();
    assert_eq!(ids(&fresh_a).await, vec!["in-a", "in-a-tail"]);
    let replica_b = IntelligentIndexedStorage::new_replica(&b, tmp.path().join("rb"))
        .await
        .unwrap();
    assert_eq!(ids(&replica_b).await, vec!["in-b", "in-b-tail"]);
}
