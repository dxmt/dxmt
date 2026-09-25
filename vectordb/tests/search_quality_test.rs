// Comprehensive search quality and correctness tests for DXMT VectorDB.
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
        record::{
            Condition, CreateRecordRequest, Filter, FilterOperation, FusionMethod,
            HybridSearchRequest, SearchQuery,
        },
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
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
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
            text: None,
        })
        .collect();
    storage.insert_records(table, requests).await.unwrap();
}

// ── Test 1: HNSW vs BruteForce regression guard ───────────────────────────────
//
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
                        text: None,
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
                        text: None,
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
                    text: None,
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
                text: None,
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
            text: None,
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

// ── Test 11: hybrid search ranks the same way plain search does ───────────────
//
// The index answers in distances and the fusion ranks by score, so the dense leg has to be
// converted on the way in. It was not: `fuse_dbsf` normalises "higher is better", so raw
// distances put the nearest vector last. Every fusion method is checked, and against plain
// `search` rather than a hardcoded order, because the bug is precisely a disagreement between
// the two paths that both claim to answer "what is closest to this vector".

#[tokio::test]
async fn test_hybrid_search_orders_like_plain_search() {
    let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
    storage
        .create_table(table_cfg("hybrid_order", 4, DistanceMetric::Cosine))
        .await
        .unwrap();

    // Four one-hot vectors: for a one-hot query the nearest neighbour is unambiguous.
    let vectors: Vec<Vec<f32>> = (0..4)
        .map(|i| (0..4).map(|d| if d == i { 1.0 } else { 0.0 }).collect())
        .collect();
    let ids: Vec<String> = (0..4).map(|i| format!("v{}", i)).collect();
    insert_batch(&storage, "hybrid_order", &ids, &vectors, None).await;

    for (i, query) in vectors.iter().enumerate() {
        let plain = storage
            .search(
                "hybrid_order",
                SearchQuery {
                    vector: query.clone(),
                    top_k: 4,
                    limit: 4,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            plain.hits[0].id, ids[i],
            "plain search did not return the exact match first"
        );

        for fusion in [
            FusionMethod::Dbsf,
            FusionMethod::WeightedSum,
            FusionMethod::ReciprocalRankFusion,
        ] {
            let hybrid = storage
                .hybrid_search(
                    "hybrid_order",
                    HybridSearchRequest {
                        vector: Some(query.clone()),
                        sparse_vector: None,
                        text: None,
                        vector_weight: 0.7,
                        fusion_method: fusion,
                        rrf_k: 60.0,
                        limit: 4,
                        filter: None,
                        with_payload: false,
                        with_vector: false,
                    },
                )
                .await
                .unwrap();

            assert_eq!(
                hybrid.hits[0].id, ids[i],
                "hybrid search with {:?} ranked {} first for a query equal to {}",
                fusion, hybrid.hits[0].id, ids[i]
            );
            let plain_order: Vec<&str> = plain.hits.iter().map(|h| h.id.as_str()).collect();
            let hybrid_order: Vec<&str> = hybrid.hits.iter().map(|h| h.id.as_str()).collect();
            assert_eq!(
                hybrid_order[0], plain_order[0],
                "hybrid ({:?}) and plain search disagree on the nearest neighbour",
                fusion
            );
        }
    }
}

// ── Test 12: the selector's ef_search is readable by the search path ──────────
//
// `ef_search` is chosen by the index selector and passed into `HnswConfig` at build time, but
// the *search* path computed its traversal budget from a hardcoded `let default_ef = 100`. The
// parameter was inert: changing it in the selector moved no measurable number, and `SearchMode`
// scaled a constant instead of the index's own setting.
//
// The end-to-end guard is in `tests/storage_ann_benchmark_test.rs`, which compares recall under
// Fast and Accurate on a corpus hard enough for the budget to matter — at this file's scale
// every mode returns exact results, so a recall assertion here would pass no matter what the
// plumbing did. What is worth checking cheaply is the accessor that carries the value across.

#[tokio::test]
async fn test_selected_algorithm_exposes_ef_search() {
    let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
    let mut cfg = table_cfg("ef_exposed", 16, DistanceMetric::Cosine);
    cfg.memory_optimization = Some(true);
    storage.create_table(cfg).await.unwrap();

    let vectors = make_unit_vectors(64, 16);
    let ids: Vec<String> = (0..64).map(|i| format!("x{i:03}")).collect();
    insert_batch(&storage, "ef_exposed", &ids, &vectors, None).await;

    let decision = storage
        .describe_index("ef_exposed")
        .await
        .expect("table should report its index decision");
    let ef = decision.configured_ef_search().expect(
        "an adaptively selected HNSW index must report the ef_search it was configured with",
    );
    assert!(
        ef > 10,
        "ef_search of {ef} is at or below top_k; the traversal budget would be meaningless"
    );

    // And a search still works with it — that the budget is applied is what the ANN benchmark
    // measures; that it does not break the query path is checked here.
    let result = storage
        .search(
            "ef_exposed",
            SearchQuery {
                vector: vectors[0].clone(),
                top_k: 5,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(result.hits[0].id, ids[0]);
}

// ── Test 13: a filtered graph search reaches candidates it did not start on ───
//
// `search_layer` seeds the traversal from the entry point. It gated *both* the exploration
// queue and the result set on the filter, so when the entry point did not match — which, for a
// selective filter, is nearly always — the queue started empty, the loop never ran, and the
// search returned nothing. Measured on 20K records: at 2% selectivity 97.6% of queries came
// back empty despite 400 matching records, and at 1% it was 98.4%. The rate is just
// `1 - selectivity`: the probability that the one entry point happened to pass the filter.
//
// A single-id filter makes the assertion exact — whatever the graph does, the only admissible
// answer is that id — and repeating over many ids means the entry point is not one of them.

#[tokio::test]
async fn test_filtered_graph_search_does_not_need_a_matching_entry_point() {
    const N: usize = 1_000;
    const DIM: usize = 24;

    let vectors = make_unit_vectors(N, DIM);
    let ids: Vec<String> = (0..N).map(|i| format!("g{i:04}")).collect();

    let mut index = HnswIndex::new(
        DistanceMetric::Cosine,
        Some(HnswConfig::from_params(16, 200, 128)),
    );
    for (id, vector) in ids.iter().zip(&vectors) {
        index.insert(id.clone(), vector.clone()).unwrap();
    }

    let mut empty = 0usize;
    let trials = 40;
    for trial in 0..trials {
        let target = trial * (N / trials) + 3;
        let filter: HashSet<String> = std::iter::once(ids[target].clone()).collect();
        // Query from somewhere else entirely, so the answer cannot come from the query being
        // its own nearest neighbour.
        let query = &vectors[(target + N / 2) % N];

        let hits = index.search_with_filter(query, 5, 128, &filter).unwrap();
        if hits.is_empty() {
            empty += 1;
            continue;
        }
        assert_eq!(
            hits.len(),
            1,
            "a one-id filter returned {} hits: {hits:?}",
            hits.len()
        );
        assert_eq!(
            hits[0].0, ids[target],
            "filtered search returned a non-matching id"
        );
    }

    assert_eq!(
        empty, 0,
        "{empty}/{trials} filtered searches returned nothing; a candidate is unreachable unless \
         the graph entry point happens to match the filter"
    );
}

// ── Test 14: every filter operator returns a full page ────────────────────────
//
// The amplified fetch that feeds post-filtering is sized from an *estimated* selectivity, and
// the string estimates are not estimates: `Contains` is hardcoded to 0.25 and `EndsWith` to
// 0.15 whatever the data holds. A `Contains` whose true selectivity was 1% fetched 240 rows for
// a top-20 and returned 3 — correct rows, too few of them, no error. No constant fixes that,
// because the error is in the input to the constant; the bounded refetch does.
//
// Checked against a reference predicate rather than a golden list, so the assertion is
// "everything returned matches, and enough was returned", for each operator in the enum.

#[tokio::test]
async fn test_every_filter_operator_returns_a_full_page() {
    const N: usize = 3_000;
    const DIM: usize = 8;
    const TOP_K: usize = 20;
    let words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"];

    let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
    let mut cfg = table_cfg("ops", DIM, DistanceMetric::Cosine);
    cfg.memory_optimization = Some(true);
    storage.create_table(cfg).await.unwrap();

    let num_of = |i: usize| (i % 100) as i64;
    let tag_of = |i: usize| words[(num_of(i) as usize) % 7];
    let name_of = |i: usize| format!("rec-{}-x", num_of(i));

    let vectors: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 31 + d * 17) % 1000) as f32) / 1000.0 - 0.5)
                .collect()
        })
        .collect();
    let ids: Vec<String> = (0..N).map(|i| format!("o{i:05}")).collect();
    let payloads: Vec<serde_json::Value> = (0..N)
        .map(|i| serde_json::json!({ "num": num_of(i), "tag": tag_of(i), "name": name_of(i) }))
        .collect();
    insert_batch(&storage, "ops", &ids, &vectors, Some(&payloads)).await;

    #[allow(clippy::type_complexity)]
    let cases: Vec<(&str, &str, FilterOperation, Box<dyn Fn(usize) -> bool>)> = vec![
        (
            "Equal",
            "num",
            FilterOperation::Equal(serde_json::json!(42)),
            Box::new(move |i| num_of(i) == 42),
        ),
        (
            "NotEqual",
            "num",
            FilterOperation::NotEqual(serde_json::json!(42)),
            Box::new(move |i| num_of(i) != 42),
        ),
        (
            "GreaterThan",
            "num",
            FilterOperation::GreaterThan(serde_json::json!(95)),
            Box::new(move |i| num_of(i) > 95),
        ),
        (
            "GreaterThanOrEqual",
            "num",
            FilterOperation::GreaterThanOrEqual(serde_json::json!(95)),
            Box::new(move |i| num_of(i) >= 95),
        ),
        (
            "LessThan",
            "num",
            FilterOperation::LessThan(serde_json::json!(4)),
            Box::new(move |i| num_of(i) < 4),
        ),
        (
            "LessThanOrEqual",
            "num",
            FilterOperation::LessThanOrEqual(serde_json::json!(4)),
            Box::new(move |i| num_of(i) <= 4),
        ),
        (
            "In",
            "num",
            FilterOperation::In(vec![
                serde_json::json!(1),
                serde_json::json!(2),
                serde_json::json!(3),
            ]),
            Box::new(move |i| (1..=3).contains(&num_of(i))),
        ),
        (
            "NotIn",
            "num",
            FilterOperation::NotIn(vec![serde_json::json!(1), serde_json::json!(2)]),
            Box::new(move |i| !(1..=2).contains(&num_of(i))),
        ),
        (
            "Equal(string)",
            "tag",
            FilterOperation::Equal(serde_json::json!("alpha")),
            Box::new(move |i| tag_of(i) == "alpha"),
        ),
        (
            "Contains",
            "name",
            FilterOperation::Contains("-7-".to_string()),
            Box::new(move |i| name_of(i).contains("-7-")),
        ),
        (
            "StartsWith",
            "name",
            FilterOperation::StartsWith("rec-1".to_string()),
            Box::new(move |i| name_of(i).starts_with("rec-1")),
        ),
        (
            "EndsWith",
            "name",
            FilterOperation::EndsWith("9-x".to_string()),
            Box::new(move |i| name_of(i).ends_with("9-x")),
        ),
    ];

    for (label, key, operation, predicate) in cases {
        let expected: HashSet<String> = (0..N)
            .filter(|&i| predicate(i))
            .map(|i| ids[i].clone())
            .collect();
        assert!(
            expected.len() >= TOP_K,
            "{label}: the corpus must hold more than one page of matches to test this"
        );

        let result = storage
            .search(
                "ops",
                SearchQuery {
                    vector: vectors[0].clone(),
                    top_k: TOP_K,
                    limit: TOP_K,
                    filter: Some(Filter {
                        must: Some(vec![Condition {
                            key: key.to_string(),
                            operation,
                        }]),
                        must_not: None,
                        should: None,
                    }),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        for hit in &result.hits {
            assert!(
                expected.contains(&hit.id),
                "{label}: returned '{}', which does not match the filter",
                hit.id
            );
        }
        assert_eq!(
            result.hits.len(),
            TOP_K,
            "{label}: asked for {TOP_K} and {} records match, but only {} came back",
            expected.len(),
            result.hits.len()
        );
    }
}

// ── Test 15: tied scores are cut by id, not by hash order ─────────────────────
//
// Six identical vectors score identically, and `limit` is smaller than the number of ties, so
// the truncate decides *which* records come back rather than only how they are arranged. RRF
// builds its combined scores in a `HashMap`, and the dense leg's rank list — the one the RRF
// ranks are read off — was sorted with no tiebreak, so equally-scored documents could survive
// or be cut according to hash iteration order, differing between calls and between fusion
// methods. The ids are asserted, not their count: a caller paging through equal-scoring results
// is what a wrong answer here actually costs.

#[tokio::test]
async fn test_tied_scores_are_cut_by_id() {
    let storage = IntelligentIndexedStorage::new(&mem_config()).await.unwrap();
    storage
        .create_table(table_cfg("tied_scores", 4, DistanceMetric::Cosine))
        .await
        .unwrap();

    let ids: Vec<String> = (0..6).map(|i| format!("t{i}")).collect();
    let vectors: Vec<Vec<f32>> = (0..6).map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect();
    insert_batch(&storage, "tied_scores", &ids, &vectors, None).await;

    for fusion in [
        FusionMethod::Dbsf,
        FusionMethod::WeightedSum,
        FusionMethod::ReciprocalRankFusion,
    ] {
        let mut answers = Vec::new();
        for _ in 0..5 {
            let result = storage
                .hybrid_search(
                    "tied_scores",
                    HybridSearchRequest {
                        vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
                        sparse_vector: None,
                        text: None,
                        vector_weight: 0.7,
                        fusion_method: fusion,
                        rrf_k: 60.0,
                        limit: 3,
                        filter: None,
                        with_payload: false,
                        with_vector: false,
                    },
                )
                .await
                .unwrap();
            answers.push(
                result
                    .hits
                    .iter()
                    .map(|h| h.id.clone())
                    .collect::<Vec<String>>(),
            );
        }

        assert_eq!(
            answers[0],
            ["t0", "t1", "t2"],
            "{fusion:?} did not keep the three smallest ids out of six tied records"
        );
        assert!(
            answers.iter().all(|answer| *answer == answers[0]),
            "{fusion:?} answered differently across calls: {answers:?}"
        );
    }
}
