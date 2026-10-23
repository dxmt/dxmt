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
