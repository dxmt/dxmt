//! Records and vectors on disk (`DXMT_RECORD_STORE=disk` + `DXMT_VECTOR_STORE=disk`), end to end.
//!
//! A quantized table keeps its graph on codes and its records without vectors; the row
//! file is what the rerank, `get_record` and `with_vector` read, and what a restart rebuilds
//! from. Each of those is exercised here, plus an update and a delete.

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

const ROWS: usize = 6_000;
const DIM: usize = 64;
const K: usize = 10;

fn config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 4096,
            eviction_policy: "lru".to_string(),
        }),
        ..Default::default()
    }
}

fn vectors() -> Vec<Vec<f32>> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let mut cstate = 0x00C0_FFEE_u64;
    let mut cnext = move || {
        cstate ^= cstate << 13;
        cstate ^= cstate >> 7;
        cstate ^= cstate << 17;
        (cstate >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let centres: Vec<Vec<f32>> = (0..40)
        .map(|_| (0..DIM).map(|_| cnext()).collect())
        .collect();
    (0..ROWS)
        .map(|i| {
            let c = &centres[i % 40];
            (0..DIM).map(|d| c[d] + 0.15 * next()).collect()
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn brute_force_where(vs: &[Vec<f32>], q: &[f32], keep: impl Fn(usize) -> bool) -> Vec<String> {
    let mut scored: Vec<(usize, f32)> = (0..ROWS)
        .filter(|&i| keep(i))
        .map(|i| (i, cosine(&vs[i], q)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
        .into_iter()
        .take(K)
        .map(|(i, _)| i.to_string())
        .collect()
}

fn recall(found: &[String], truth: &[String]) -> f64 {
    found.iter().filter(|id| truth.contains(id)).count() as f64 / truth.len() as f64
}

fn brute_force(vs: &[Vec<f32>], q: &[f32]) -> Vec<String> {
    let mut scored: Vec<(usize, f32)> = (0..ROWS).map(|i| (i, cosine(&vs[i], q))).collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
        .into_iter()
        .take(K)
        .map(|(i, _)| i.to_string())
        .collect()
}

async fn open(wal: &Path) -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new_with_wal(&config(), Some(wal.to_path_buf()))
        .await
        .unwrap()
}

async fn top_ids(storage: &IntelligentIndexedStorage, q: &[f32]) -> Vec<String> {
    storage
        .search(
            "t",
            SearchQuery {
                vector: q.to_vec(),
                top_k: K,
                limit: K,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits
        .iter()
        .map(|h| h.id.clone())
        .collect()
}

#[tokio::test]
async fn a_table_with_records_on_disk_searches_serves_and_restarts() {
    std::env::set_var("DXMT_INDEX_ALGORITHM", "hnsw-pq");
    std::env::set_var("DXMT_VECTOR_STORE", "disk");
    std::env::set_var("DXMT_RECORD_STORE", "disk");
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");
    let vs = vectors();

    let storage = open(&wal).await;
    storage
        .create_table(TableConfig {
            name: "t".to_string(),
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
            rerank: None,
        })
        .await
        .unwrap();
    for chunk in (0..ROWS).collect::<Vec<_>>().chunks(2_000) {
        let batch: Vec<CreateRecordRequest> = chunk
            .iter()
            .map(|&i| CreateRecordRequest {
                id: i.to_string(),
                vector: vs[i].clone(),
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::json!({ "bucket": i % 5, "rare": i % 100 })),
                text: None,
            })
            .collect();
        storage.insert_records("t", batch).await.unwrap();
    }
    storage.rebuild_index("t").await.unwrap();

    // The row file exists and holds every row; that is where the vectors went.
    let files: Vec<_> = std::fs::read_dir(wal.join("vectors"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    let data = files
        .iter()
        .find(|p| p.extension().is_some_and(|e| e == "f32"))
        .expect("a .f32 row file");
    assert_eq!(
        std::fs::metadata(data).unwrap().len(),
        (ROWS * DIM * 4) as u64,
        "one row per record"
    );

    let rec_files: Vec<_> = std::fs::read_dir(wal.join("records"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert!(
        rec_files
            .iter()
            .any(|p| p.extension().is_some_and(|e| e == "rec")),
        "a .rec record file"
    );
    assert!(
        rec_files
            .iter()
            .any(|p| p.to_string_lossy().ends_with(".rec.log")),
        "a .rec.log row log"
    );

    // Payload comes back from the record file, and a post-filter reads it there too.
    let rec = storage.get_record("t", &"4321".to_string()).await.unwrap();
    assert_eq!(rec.payload["bucket"], 4321 % 5);
    let filtered = storage
        .search(
            "t",
            SearchQuery {
                vector: vs[7].clone(),
                top_k: K,
                limit: K,
                with_payload: true,
                filter: Some(dxmt_vectordb::types::record::Filter {
                    must: Some(vec![dxmt_vectordb::types::record::Condition {
                        key: "bucket".to_string(),
                        operation: dxmt_vectordb::types::record::FilterOperation::Equal(
                            serde_json::json!(2),
                        ),
                    }]),
                    must_not: None,
                    should: None,
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(filtered.hits.len(), K);
    assert!(filtered
        .hits
        .iter()
        .all(|h| h.id.parse::<usize>().unwrap() % 5 == 2));
    assert!(filtered
        .hits
        .iter()
        .all(|h| h.payload.as_ref().unwrap()["bucket"] == 2));

    // A 1%-selectivity filter goes through the code scan: the 60 matching ids are scored
    // from the resident codes and reranked exactly, with no graph walk.
    let qi = 4321;
    let rare = storage
        .search(
            "t",
            SearchQuery {
                vector: vs[qi].clone(),
                top_k: K,
                limit: K,
                filter: Some(dxmt_vectordb::types::record::Filter {
                    must: Some(vec![dxmt_vectordb::types::record::Condition {
                        key: "rare".to_string(),
                        operation: dxmt_vectordb::types::record::FilterOperation::Equal(
                            serde_json::json!(7),
                        ),
                    }]),
                    must_not: None,
                    should: None,
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let found: Vec<String> = rare.hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(found.len(), K, "a 1% filter must still fill the top-{K}");
    assert!(found
        .iter()
        .all(|id| id.parse::<usize>().unwrap() % 100 == 7));
    let truth = brute_force_where(&vs, &vs[qi], |i| i % 100 == 7);
    let r = recall(&found, &truth);
    assert!(r >= 0.9, "1%-filter recall@{K}: {r:.3}");

    // Search: exact rerank from the file, own vector first, recall against brute force.
    let queries: Vec<usize> = (0..50).map(|i| i * 113 % ROWS).collect();
    let mut hits = 0.0;
    let mut self_first = 0;
    for &qi in &queries {
        let found = top_ids(&storage, &vs[qi]).await;
        assert_eq!(found.len(), K);
        let truth = brute_force(&vs, &vs[qi]);
        hits += found.iter().filter(|id| truth.contains(id)).count() as f64 / K as f64;
        if found[0] == qi.to_string() {
            self_first += 1;
        }
    }
    let recall = hits / queries.len() as f64;
    assert!(
        recall >= 0.9,
        "recall@{K} with vectors on disk: {recall:.3}"
    );
    assert!(self_first >= 48, "{self_first}/50 own-vector-first");

    // get_record and with_vector read the row back.
    let rec = storage.get_record("t", &"1234".to_string()).await.unwrap();
    assert_eq!(rec.vector, vs[1234]);
    let hit = storage
        .search(
            "t",
            SearchQuery {
                vector: vs[1234].clone(),
                top_k: 1,
                limit: 1,
                with_vector: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(hit.hits[0].id, "1234");
    assert_eq!(hit.hits[0].vector.as_deref(), Some(vs[1234].as_slice()));

    // An update appends a row and the search follows the new vector.
    let moved = vs[4321].iter().map(|x| -x).collect::<Vec<f32>>();
    storage
        .update_record(
            "t",
            &"77".to_string(),
            CreateRecordRequest {
                id: "77".to_string(),
                vector: moved.clone(),
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::json!({ "bucket": 0 })),
                text: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(
        storage
            .get_record("t", &"77".to_string())
            .await
            .unwrap()
            .vector,
        moved
    );
    assert_eq!(top_ids(&storage, &moved).await[0], "77");

    // A delete stops the id from being served.
    storage
        .delete_records("t", vec!["77".to_string()])
        .await
        .unwrap();
    assert!(storage.get_record("t", &"77".to_string()).await.is_err());
    assert_ne!(top_ids(&storage, &moved).await[0], "77");

    // Restart from snapshot + WAL + row file: same answers, same vectors.
    storage.snapshot().await.unwrap();
    let before: Vec<Vec<String>> = {
        let mut v = Vec::new();
        for &qi in queries.iter().take(10) {
            v.push(top_ids(&storage, &vs[qi]).await);
        }
        v
    };
    drop(storage);
    let storage = open(&wal).await;
    for (&qi, expected) in queries.iter().take(10).zip(&before) {
        assert_eq!(
            &top_ids(&storage, &vs[qi]).await,
            expected,
            "query {qi} after restart"
        );
    }
    assert_eq!(
        storage
            .get_record("t", &"1234".to_string())
            .await
            .unwrap()
            .vector,
        vs[1234]
    );
    assert!(storage.get_record("t", &"77".to_string()).await.is_err());
    let back = storage.get_record("t", &"1234".to_string()).await.unwrap();
    assert_eq!(back.payload["bucket"], 1234 % 5);
    let table = storage.get_table("t").await.unwrap();
    assert_eq!(
        table.record_count,
        (ROWS - 1) as u64,
        "one record was deleted"
    );
}
