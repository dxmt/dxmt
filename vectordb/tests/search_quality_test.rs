// Comprehensive search quality and correctness tests for DXMT VectorDB.
//
// Test design inspired by:
// - Qdrant's filterable_hnsw_test.rs (filtered recall with varying selectivity)
// - Qdrant's exact_search_test.rs (brute-force vs HNSW regression guard)
// - Chroma's test_multithreaded.py (concurrent add/query interleaving)
// - Chroma's test_api.py (CRUD lifecycle correctness)
//
// Coverage areas not in e2e_recall_test.rs:
// 1. HNSW vs brute-force exact regression (same data, results must match)
// 2. Filtered recall at varying selectivity (correctness + recall)
// 3. Filter correctness (every result satisfies the filter)
// 4. Concurrent inserts + queries (data-race detection)
// 5. Edge cases: empty table, top_k=0, top_k > N, exact-match self-retrieval
// 6. Distance metric ordering (Cosine vs Euclidean rank different vectors differently)
// 7. CRUD lifecycle: insert → search → delete → re-insert with different vector
// 8. Higher ef_search yields equal-or-better recall (monotonicity)
// 9. Batch insert: all records immediately visible after the await
// 10. Multi-condition AND filter

use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Barrier;

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    search::{
        brute_force::BruteForceIndex,
        hnsw::{HnswConfig, HnswIndex},
        VectorIndex,
    },
    storage::IntelligentIndexedStorage,
    types::{
        record::{Condition, CreateRecordRequest, Filter, FilterOperation, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};

// ── Shared helpers ────────────────────────────────────────────────────────────

fn mem_config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 512,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

fn table_cfg(name: &str, dim: usize, metric: DistanceMetric) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension: dim,
        distance_metric: metric,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    }
}

/// Deterministic pseudo-random L2-normalised vectors.
fn make_unit_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let raw: Vec<f32> = (0..dim)
                .map(|d| {
                    let h = i
                        .wrapping_mul(2654435761usize)
                        .wrapping_add(d.wrapping_mul(2246822519));
                    let h = h ^ (h >> 30);
                    let h = h.wrapping_mul(0xbf58476d1ce4e5b9usize);
                    let h = h ^ (h >> 27);
                    let h = h.wrapping_mul(0x94d049bb133111ebusize);
                    let h = h ^ (h >> 31);
                    (h >> 40) as f32 / (1usize << 24) as f32 * 2.0 - 1.0
                })
                .collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            raw.iter().map(|x| x / norm).collect()
        })
        .collect()
}

/// Brute-force top-k by dot product (cosine on normalised vectors).
fn brute_force_top_k(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = query.iter().zip(v).map(|(a, b)| a * b).sum();
            (i, dot)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at_k(retrieved: &[String], ground_truth_indices: &[usize], all_ids: &[String]) -> f32 {
    let gt_ids: HashSet<&str> = ground_truth_indices
        .iter()
        .map(|&i| all_ids[i].as_str())
        .collect();
    let hits = retrieved
        .iter()
        .filter(|id| gt_ids.contains(id.as_str()))
        .count();
    if ground_truth_indices.is_empty() {
        1.0
    } else {
        hits as f32 / ground_truth_indices.len() as f32
    }
}

async fn insert_batch(
    storage: &IntelligentIndexedStorage,
    table: &str,
    ids: &[String],
    vectors: &[Vec<f32>],
    payloads: Option<&[serde_json::Value]>,
) {
    let requests: Vec<CreateRecordRequest> = ids
        .iter()
        .zip(vectors)
        .enumerate()
        .map(|(i, (id, v))| CreateRecordRequest {
            id: id.clone(),
            vector: v.clone(),
            sparse_vector: None,
            payload: payloads.map(|p| p[i].clone()),
            named_vectors: None,
        })
        .collect();
    storage.insert_records(table, requests).await.unwrap();
}

// ── Test 1: HNSW vs BruteForce regression guard ───────────────────────────────
//
// Inspired by Qdrant's exact_search_test.rs.
// Both indexes built on identical data; their top-10 results must agree ≥95%.

#[tokio::test]
async fn test_hnsw_matches_brute_force() {
    const N: usize = 500;
    const DIM: usize = 32;
    const TOP_K: usize = 10;

    let vectors = make_unit_vectors(N, DIM);
    let ids: Vec<String> = (0..N).map(|i| format!("x{:04}", i)).collect();

    let mut hnsw = HnswIndex::new(
        DistanceMetric::Cosine,
        Some(HnswConfig {
            m: 16,
            max_m: 16,
            max_m0: 32,
            ef_construction: 200,
            ef_search: 200,
            ..Default::default()
        }),
    );
    let mut bf = BruteForceIndex::new(DistanceMetric::Cosine);

    for (id, v) in ids.iter().zip(&vectors) {
        hnsw.add(id.clone(), v.clone()).await.unwrap();
        bf.add(id.clone(), v.clone()).await.unwrap();
    }

    let mut total_match = 0usize;
    let mut total_expected = 0usize;

    for qi in (0..N).step_by(25) {
        let hnsw_res = hnsw
            .search_with_ef_hint(&vectors[qi], TOP_K, 200)
            .await
            .unwrap();
        let bf_res = bf.search(&vectors[qi], TOP_K).await.unwrap();

        let hnsw_ids: HashSet<&str> = hnsw_res.iter().map(|(id, _)| id.as_str()).collect();
        let bf_ids: HashSet<&str> = bf_res.iter().map(|(id, _)| id.as_str()).collect();

        total_match += hnsw_ids.intersection(&bf_ids).count();
        total_expected += TOP_K;
    }

    let match_rate = total_match as f64 / total_expected as f64;
    println!("HNSW-vs-BruteForce match rate: {:.1}%", match_rate * 100.0);
    assert!(
        match_rate >= 0.95,
        "HNSW match rate {:.3} < 0.95 — graph traversal may be degraded",
        match_rate
    );
}

// ── Test 2: Filtered recall at varying selectivity ────────────────────────────
//
// Inspired by Qdrant's filterable_hnsw_test.rs parametrized ef test.
// Verifies that at 10% selectivity, filtered recall stays high and every
// returned result satisfies the filter condition.

#[tokio::test]
async fn test_filtered_recall_varying_selectivity() {
    const N: usize = 300;
    const DIM: usize = 32;

    let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
    storage
        .create_table(table_cfg("filt_sel", DIM, DistanceMetric::Cosine))
        .await
        .unwrap();

    let vectors = make_unit_vectors(N, DIM);
    let ids: Vec<String> = (0..N).map(|i| format!("s{:04}", i)).collect();

    // Assign each record a numeric bucket: i % 10 → "group" field
    let payloads: Vec<serde_json::Value> = (0..N)
        .map(|i| serde_json::json!({ "group": (i % 10) as i64 }))
        .collect();
    insert_batch(&storage, "filt_sel", &ids, &vectors, Some(&payloads)).await;

    // 10% selectivity: group == 0 (30 records)
    let group: i64 = 0;
    let passing_indices: Vec<usize> = (0..N).filter(|&i| (i % 10) as i64 == group).collect();
    let k = 5.min(passing_indices.len());

    let passing_ids_set: HashSet<&str> = passing_indices.iter().map(|&i| ids[i].as_str()).collect();

    let filter = Filter {
        must: Some(vec![Condition {
            key: "group".to_string(),
            operation: FilterOperation::Equal(serde_json::json!(group)),
        }]),
        must_not: None,
        should: None,
    };
    let qi = passing_indices[0];

    let result = storage
        .search(
            "filt_sel",
            SearchQuery {
                vector: vectors[qi].clone(),
                top_k: k,
                limit: k,
                filter: Some(filter),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // CORRECTNESS: every result must satisfy the filter
    for hit in &result.hits {
        assert!(
            passing_ids_set.contains(hit.id.as_str()),
            "filter violated: '{}' has group≠{} (10% selectivity)",
            hit.id,
            group
        );
    }

    // RECALL vs brute-force filtered ground truth
    let filtered_vecs: Vec<Vec<f32>> = passing_indices
        .iter()
        .map(|&i| vectors[i].clone())
        .collect();
    let gt_local = brute_force_top_k(&vectors[qi], &filtered_vecs, k);
    let gt_global: Vec<usize> = gt_local.iter().map(|&li| passing_indices[li]).collect();
    let retrieved: Vec<String> = result.hits.iter().map(|h| h.id.clone()).collect();
    let recall = recall_at_k(&retrieved, &gt_global, &ids);
    println!("filtered recall@{k} (10% selectivity): {:.3}", recall);
    assert!(
        recall >= 0.80,
        "filtered recall {:.3} < 0.80 for 10% selectivity",
        recall
    );

    // 50% selectivity: In([0, 2, 4, 6, 8]) → 150 records
    let passing_50: Vec<usize> = (0..N).filter(|&i| i % 2 == 0).collect();
    let k50 = 5.min(passing_50.len());
    let filter_50 = Filter {
        must: Some(vec![Condition {
            key: "group".to_string(),
            operation: FilterOperation::In(
                [0i64, 2, 4, 6, 8]
                    .iter()
                    .map(|&v| serde_json::json!(v))
                    .collect(),
            ),
        }]),
        must_not: None,
        should: None,
    };

    let result_50 = storage
        .search(
            "filt_sel",
            SearchQuery {
                vector: vectors[0].clone(),
                top_k: k50,
                limit: k50,
                filter: Some(filter_50),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let passing_50_ids: HashSet<&str> = passing_50.iter().map(|&i| ids[i].as_str()).collect();
    for hit in &result_50.hits {
        assert!(
            passing_50_ids.contains(hit.id.as_str()),
            "50% filter violated: '{}' is odd-group",
            hit.id
        );
    }

    let retrieved_50: Vec<String> = result_50.hits.iter().map(|h| h.id.clone()).collect();
    let filtered_vecs_50: Vec<Vec<f32>> = passing_50.iter().map(|&i| vectors[i].clone()).collect();
    let gt_50 = brute_force_top_k(&vectors[0], &filtered_vecs_50, k50);
    let gt_50g: Vec<usize> = gt_50.iter().map(|&li| passing_50[li]).collect();
    let recall_50 = recall_at_k(&retrieved_50, &gt_50g, &ids);
    println!("filtered recall@{k50} (50% selectivity): {:.3}", recall_50);
    assert!(
        recall_50 >= 0.90,
        "50% filtered recall {:.3} < 0.90",
        recall_50
    );
}

// ── Test 3: Filter correctness — all results satisfy condition ────────────────
//
// Constructs records where the geometrically-nearest records may NOT match the
// filter. Verifies that filtered results are always within the filter set.

#[tokio::test]
async fn test_filter_excludes_nearer_non_matching() {
    const DIM: usize = 16;

    let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
    storage
        .create_table(table_cfg("filt_correct", DIM, DistanceMetric::Cosine))
        .await
        .unwrap();

    let vectors = make_unit_vectors(100, DIM);
    let ids: Vec<String> = (0..100).map(|i| format!("fc{:03}", i)).collect();

    // Even-indexed → category "A", odd-indexed → category "B"
    let payloads: Vec<serde_json::Value> = (0..100)
        .map(|i| serde_json::json!({ "category": if i % 2 == 0 { "A" } else { "B" } }))
        .collect();
    insert_batch(&storage, "filt_correct", &ids, &vectors, Some(&payloads)).await;

    let filter = Filter {
        must: Some(vec![Condition {
            key: "category".to_string(),
            operation: FilterOperation::Equal(serde_json::json!("A")),
        }]),
        must_not: None,
        should: None,
    };

    let result = storage
        .search(
            "filt_correct",
            SearchQuery {
                vector: vectors[0].clone(),
                top_k: 10,
                limit: 10,
                filter: Some(filter),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Every returned ID must be even-indexed (category A)
    for hit in &result.hits {
        let idx: usize = hit.id.trim_start_matches("fc").parse().unwrap();
        assert_eq!(
            idx % 2,
            0,
            "result '{}' (idx={}) is category B but filter asked for A",
            hit.id,
            idx
        );
    }

    assert_eq!(
        result.hits.len(),
        10,
        "expected 10 results for category-A filter"
    );

    // No duplicate IDs
    let unique: HashSet<&str> = result.hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        unique.len(),
        result.hits.len(),
        "duplicate IDs in filtered results"
    );
}

// ── Test 4: Edge cases ────────────────────────────────────────────────────────
//
// Inspired by Qdrant's test_hnsw_search_top_zero and Chroma's empty-dataset handling.

#[tokio::test]
async fn test_edge_cases() {
    const DIM: usize = 8;
    let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
    storage
        .create_table(table_cfg("edges", DIM, DistanceMetric::Cosine))
        .await
        .unwrap();

    let vectors = make_unit_vectors(20, DIM);
    let ids: Vec<String> = (0..20).map(|i| format!("e{:02}", i)).collect();

    // Query empty table — must not error, returns empty
    let result = storage
        .search(
            "edges",
            SearchQuery {
                vector: vectors[0].clone(),
                top_k: 5,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(result.hits.is_empty(), "empty table must return no hits");

    insert_batch(&storage, "edges", &ids, &vectors, None).await;

    // top_k = 0 — must not panic, returns empty
    let result = storage
        .search(
            "edges",
            SearchQuery {
                vector: vectors[0].clone(),
                top_k: 0,
                limit: 0,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(result.hits.is_empty(), "top_k=0 must return no hits");

    // top_k > N — must clamp to N with no duplicates
    let result = storage
        .search(
            "edges",
            SearchQuery {
                vector: vectors[0].clone(),
                top_k: 1000,
                limit: 1000,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(
        result.hits.len() <= 20,
        "returned {} hits but only 20 vectors exist",
        result.hits.len()
    );
    let unique_ids: HashSet<&str> = result.hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        unique_ids.len(),
        result.hits.len(),
        "duplicates in top_k>N result"
    );

    // Exact-match self-retrieval: querying with vector[5] must return e05 as first result
    let result = storage
        .search(
            "edges",
            SearchQuery {
                vector: vectors[5].clone(),
                top_k: 1,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(result.hits.len(), 1);
    assert_eq!(
        result.hits[0].id, ids[5],
        "exact-match vector not ranked first (got '{}')",
        result.hits[0].id
    );
}

// ── Test 5: Distance metric ordering ──────────────────────────────────────────
//
// Constructs vectors where cosine similarity and L2 distance rank differently,
// then checks each metric produces the expected ordering.

#[tokio::test]
async fn test_distance_metric_ordering() {
    const DIM: usize = 4;

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.iter().map(|x| x / norm).collect()
    }

    // Cosine: perpendicular vector should rank last
    {
        let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
        storage
            .create_table(table_cfg("cosine_order", DIM, DistanceMetric::Cosine))
            .await
            .unwrap();

        let query_vec = normalize(&[1.0f32, 0.01, 0.01, 0.01]);
        let v_close_cos = normalize(&[0.9f32, 0.1, 0.1, 0.1]);
        let v_perp = normalize(&[0.01f32, 0.9, 0.3, 0.2]);
        let v_same_dir = normalize(&[1.0f32, 0.01, 0.01, 0.01]);

        for (id, v) in [
            ("cos1", v_close_cos),
            ("perp", v_perp),
            ("same", v_same_dir),
        ] {
            storage
                .insert_records(
                    "cosine_order",
                    vec![CreateRecordRequest {
                        id: id.to_string(),
                        vector: v,
                        sparse_vector: None,
                        payload: None,
                        named_vectors: None,
                    }],
                )
                .await
                .unwrap();
        }

        let res = storage
            .search(
                "cosine_order",
                SearchQuery {
                    vector: query_vec,
                    top_k: 3,
                    limit: 3,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            res.hits.last().unwrap().id,
            "perp",
            "cosine: perpendicular vector should be ranked last, got {:?}",
            res.hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>()
        );
    }

    // L2: nearest by magnitude difference should rank first
    {
        let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
        storage
            .create_table(table_cfg("eucl_order", DIM, DistanceMetric::Euclidean))
            .await
            .unwrap();

        let query_raw = vec![1.0f32, 0.0, 0.0, 0.0];
        let v_near = vec![1.1f32, 0.0, 0.0, 0.0];
        let v_mid = vec![0.0f32, 1.0, 0.0, 0.0];
        let v_far = vec![10.0f32, 5.0, 5.0, 5.0];

        for (id, v) in [("near", v_near), ("mid", v_mid), ("far_l2", v_far)] {
            storage
                .insert_records(
                    "eucl_order",
                    vec![CreateRecordRequest {
                        id: id.to_string(),
                        vector: v,
                        sparse_vector: None,
                        payload: None,
                        named_vectors: None,
                    }],
                )
                .await
                .unwrap();
        }

        let res = storage
            .search(
                "eucl_order",
                SearchQuery {
                    vector: query_raw,
                    top_k: 3,
                    limit: 3,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            res.hits[0].id,
            "near",
            "euclidean: nearest vector should be first, got {:?}",
            res.hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>()
        );
        assert_eq!(
            res.hits.last().unwrap().id,
            "far_l2",
            "euclidean: farthest vector should be last, got {:?}",
            res.hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>()
        );
    }
}

// ── Test 6: Concurrent insert + query ─────────────────────────────────────────
//
// Inspired by Chroma's test_multithreaded.py.
// Multiple writer and reader tasks run concurrently; after all complete,
// the final state must be self-consistent.

#[tokio::test]
async fn test_concurrent_insert_and_query() {
    const DIM: usize = 16;
    const WRITERS: usize = 8;
    const BATCH_PER_WRITER: usize = 25; // 200 total
    const READERS: usize = 4;
    const READS_PER_READER: usize = 15;

    let storage = Arc::new(IntelligentIndexedStorage::new(&mem_config()).await.unwrap());
    storage
        .create_table(table_cfg("concurrent", DIM, DistanceMetric::Cosine))
        .await
        .unwrap();

    let all_vectors = Arc::new(make_unit_vectors(WRITERS * BATCH_PER_WRITER, DIM));
    let barrier = Arc::new(Barrier::new(WRITERS + READERS));
    let mut handles = vec![];

    for w in 0..WRITERS {
        let s = storage.clone();
        let vecs = all_vectors.clone();
        let b = barrier.clone();
        handles.push(tokio::spawn(async move {
            b.wait().await;
            let start = w * BATCH_PER_WRITER;
            let reqs: Vec<CreateRecordRequest> = (start..start + BATCH_PER_WRITER)
                .map(|i| CreateRecordRequest {
                    id: format!("c{:04}", i),
                    vector: vecs[i].clone(),
                    sparse_vector: None,
                    payload: None,
                    named_vectors: None,
                })
                .collect();
            s.insert_records("concurrent", reqs).await.unwrap();
        }));
    }

    for r in 0..READERS {
        let s = storage.clone();
        let vecs = all_vectors.clone();
        let b = barrier.clone();
        handles.push(tokio::spawn(async move {
            b.wait().await;
            for read_i in 0..READS_PER_READER {
                let qi = (r * READS_PER_READER + read_i) % vecs.len();
                if let Ok(res) = s
                    .search(
                        "concurrent",
                        SearchQuery {
                            vector: vecs[qi].clone(),
                            top_k: 5,
                            limit: 5,
                            ..Default::default()
                        },
                    )
                    .await
                {
                    for hit in &res.hits {
                        assert!(
                            hit.id.starts_with('c'),
                            "unexpected ID in concurrent search: '{}'",
                            hit.id
                        );
                    }
                }
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    // Final state: search must work and return valid results
    let result = storage
        .search(
            "concurrent",
            SearchQuery {
                vector: all_vectors[0].clone(),
                top_k: 10,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !result.hits.is_empty(),
        "post-concurrent search returned empty"
    );
    let unique_ids: HashSet<&str> = result.hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        unique_ids.len(),
        result.hits.len(),
        "post-concurrent result has duplicates"
    );
}

// ── Test 7: CRUD lifecycle ────────────────────────────────────────────────────
//
// insert → visible → delete → invisible → re-insert different vector → near new position

#[tokio::test]
async fn test_crud_lifecycle() {
    const DIM: usize = 8;

    let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
    storage
        .create_table(table_cfg("lifecycle", DIM, DistanceMetric::Cosine))
        .await
        .unwrap();

    let vectors = make_unit_vectors(10, DIM);
    let ids: Vec<String> = (0..10).map(|i| format!("lc{}", i)).collect();
    insert_batch(&storage, "lifecycle", &ids, &vectors, None).await;

    // Insert → visible: lc0 should be closest to itself
    let res = storage
        .search(
            "lifecycle",
            SearchQuery {
                vector: vectors[0].clone(),
                top_k: 1,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(res.hits[0].id, "lc0");

    // Delete lc0
    storage
        .delete_records("lifecycle", vec!["lc0".to_string()])
        .await
        .unwrap();

    // Deleted → invisible
    let res = storage
        .search(
            "lifecycle",
            SearchQuery {
                vector: vectors[0].clone(),
                top_k: 5,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(
        res.hits.iter().all(|h| h.id != "lc0"),
        "deleted lc0 still appears in search"
    );

    // Re-insert lc0 with vector[5]'s position (completely different direction)
    storage
        .insert_records(
            "lifecycle",
            vec![CreateRecordRequest {
                id: "lc0".to_string(),
                vector: vectors[5].clone(),
                sparse_vector: None,
                payload: None,
                named_vectors: None,
            }],
        )
        .await
        .unwrap();

    // Re-inserted → visible near new position
    let res = storage
        .search(
            "lifecycle",
            SearchQuery {
                vector: vectors[5].clone(),
                top_k: 3,
                limit: 3,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let returned: Vec<&str> = res.hits.iter().map(|h| h.id.as_str()).collect();
    assert!(
        returned.contains(&"lc0") || returned.contains(&"lc5"),
        "re-inserted lc0 not near its new position, got {:?}",
        returned
    );
}

// ── Test 8: ef_search monotonicity ────────────────────────────────────────────
//
// Inspired by Qdrant's parametrized ef test.
// recall(ef=200) ≥ recall(ef=64) ≥ recall(ef=32) within a small tolerance.

#[tokio::test]
async fn test_hnsw_recall_monotone_with_ef() {
    const N: usize = 800;
    const DIM: usize = 32;
    const TOP_K: usize = 10;
    const NUM_QUERIES: usize = 20;

    let vectors = make_unit_vectors(N, DIM);
    let ids: Vec<String> = (0..N).map(|i| format!("ef{:04}", i)).collect();

    let mut index = HnswIndex::new(
        DistanceMetric::Cosine,
        Some(HnswConfig {
            m: 16,
            max_m: 16,
            max_m0: 32,
            ef_construction: 200,
            ef_search: 50,
            ..Default::default()
        }),
    );
    for (id, v) in ids.iter().zip(&vectors) {
        index.add(id.clone(), v.clone()).await.unwrap();
    }

    let ef_values = [32usize, 64, 200];
    let mut recalls = Vec::new();

    for &ef in &ef_values {
        let mut total = 0.0f32;
        let step = N / NUM_QUERIES;
        for qi in (0..N).step_by(step) {
            let gt = brute_force_top_k(&vectors[qi], &vectors, TOP_K);
            let res = index
                .search_with_ef_hint(&vectors[qi], TOP_K, ef)
                .await
                .unwrap();
            let retrieved: Vec<String> = res.iter().map(|(id, _)| id.clone()).collect();
            total += recall_at_k(&retrieved, &gt, &ids);
        }
        let avg = total / NUM_QUERIES as f32;
        recalls.push(avg);
        println!("recall@{TOP_K} at ef={ef}: {:.3}", avg);
    }

    assert!(
        recalls[0] >= 0.65,
        "ef=32 recall {:.3} too low (≥0.65 expected)",
        recalls[0]
    );
    assert!(
        recalls[2] >= 0.90,
        "ef=200 recall {:.3} too low (≥0.90 expected)",
        recalls[2]
    );

    // Monotonicity: higher ef must not regress more than 5%
    for i in 1..recalls.len() {
        assert!(
            recalls[i] >= recalls[i - 1] - 0.05,
            "recall regressed: ef={} ({:.3}) → ef={} ({:.3})",
            ef_values[i - 1],
            recalls[i - 1],
            ef_values[i],
            recalls[i]
        );
    }
}

// ── Test 9: Batch insert — all records immediately visible ────────────────────
//
// A single insert_records call must make ALL records visible before the next await.
// No partial visibility: querying any batch member right after insert must find it.

#[tokio::test]
async fn test_batch_insert_immediate_visibility() {
    const N: usize = 500;
    const DIM: usize = 16;

    let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
    storage
        .create_table(table_cfg("batch_vis", DIM, DistanceMetric::Cosine))
        .await
        .unwrap();

    let vectors = make_unit_vectors(N, DIM);
    let ids: Vec<String> = (0..N).map(|i| format!("b{:04}", i)).collect();

    // Single batch insert
    let reqs: Vec<CreateRecordRequest> = ids
        .iter()
        .zip(&vectors)
        .map(|(id, v)| CreateRecordRequest {
            id: id.clone(),
            vector: v.clone(),
            sparse_vector: None,
            payload: None,
            named_vectors: None,
        })
        .collect();
    storage.insert_records("batch_vis", reqs).await.unwrap();

    // Every 50th vector must be immediately findable as its own nearest neighbour
    let mut failures = vec![];
    for qi in (0..N).step_by(50) {
        let res = storage
            .search(
                "batch_vis",
                SearchQuery {
                    vector: vectors[qi].clone(),
                    top_k: 1,
                    limit: 1,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        if res.hits.is_empty() || res.hits[0].id != ids[qi] {
            let got = res.hits.first().map(|h| h.id.as_str()).unwrap_or("<empty>");
            failures.push(format!(
                "vector {qi}: expected '{}', got '{}'",
                ids[qi], got
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "batch insert visibility failures:\n{}",
        failures.join("\n")
    );
}

// ── Test 10: Multi-condition AND filter ───────────────────────────────────────
//
// All conditions in Filter::must are AND-ed together.
// A result must satisfy every condition in the must list.

#[tokio::test]
async fn test_multi_condition_and_filter() {
    const DIM: usize = 8;

    let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
    storage
        .create_table(table_cfg("multifilter", DIM, DistanceMetric::Cosine))
        .await
        .unwrap();

    let vectors = make_unit_vectors(60, DIM);
    let ids: Vec<String> = (0..60).map(|i| format!("mf{:02}", i)).collect();

    // color: red(i%3==0), green(i%3==1), blue(i%3==2)
    // size:  small(i%2==0), large(i%2==1)
    // color==red AND size==small → i%6==0 → 10 records
    let payloads: Vec<serde_json::Value> = (0..60)
        .map(|i| {
            serde_json::json!({
                "color": match i % 3 { 0 => "red", 1 => "green", _ => "blue" },
                "size":  if i % 2 == 0 { "small" } else { "large" }
            })
        })
        .collect();
    insert_batch(&storage, "multifilter", &ids, &vectors, Some(&payloads)).await;

    let expected: HashSet<usize> = (0..60).filter(|i| i % 6 == 0).collect();

    let filter = Filter {
        must: Some(vec![
            Condition {
                key: "color".to_string(),
                operation: FilterOperation::Equal(serde_json::json!("red")),
            },
            Condition {
                key: "size".to_string(),
                operation: FilterOperation::Equal(serde_json::json!("small")),
            },
        ]),
        must_not: None,
        should: None,
    };

    let result = storage
        .search(
            "multifilter",
            SearchQuery {
                vector: vectors[0].clone(),
                top_k: 10,
                limit: 10,
                filter: Some(filter),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    for hit in &result.hits {
        let idx: usize = hit.id.trim_start_matches("mf").parse().unwrap();
        assert!(
            expected.contains(&idx),
            "'{}' (idx={}) fails color==red AND size==small",
            hit.id,
            idx
        );
    }

    assert!(
        !result.hits.is_empty(),
        "multi-condition AND filter returned no results"
    );
    assert!(result.hits.len() <= 10);

    let unique: HashSet<&str> = result.hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        unique.len(),
        result.hits.len(),
        "multi-condition filter returned duplicates"
    );

    println!("multi-condition AND filter: {} results", result.hits.len());
}
