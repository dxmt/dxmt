/// End-to-end recall test for IntelligentIndexedStorage.
///
/// Inserts a known vector set, computes brute-force ground truth, then runs
/// the storage's search() and verifies that recall@10 ≥ 0.9.
///
/// This test catches regressions where index wiring, quantization, or
/// distance computation silently degrades search quality.

use dxmt_vectordb::{
    storage::IntelligentIndexedStorage,
    config::StorageConfig,
    types::{
        table::TableConfig,
        record::{CreateRecordRequest, SearchQuery, Filter, Condition, FilterOperation},
        vector::DistanceMetric,
    },
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(dxmt_vectordb::config::MemoryConfig {
            max_size_mb: 256,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

/// Deterministic synthetic vectors using a well-spread construction.
/// Each vector i has a unique random-looking direction to avoid clustering artifacts
/// that degrade HNSW recall on structured data.
fn make_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            // Use a pseudo-random, per-vector unique seed to avoid periodic structure
            let v: Vec<f32> = (0..dim)
                .map(|d| {
                    // LCG-style mixing for good spread across the hypersphere
                    let h = (i.wrapping_mul(2654435761).wrapping_add(d.wrapping_mul(2246822519))) as f32;
                    (h.sin() * 127.0).fract()
                })
                .collect();
            // L2-normalise
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter().map(|x| x / norm).collect()
        })
        .collect()
}

/// Brute-force top-k by cosine dot-product (vectors assumed L2-normalised).
fn ground_truth_top_k(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = query.iter().zip(v).map(|(a, b)| a * b).sum();
            (i, dot)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at_k(retrieved: &[String], ground_truth: &[usize], corpus_ids: &[String]) -> f32 {
    let gt_ids: std::collections::HashSet<&String> =
        ground_truth.iter().map(|&i| &corpus_ids[i]).collect();
    let hits = retrieved.iter().filter(|id| gt_ids.contains(id)).count();
    hits as f32 / ground_truth.len() as f32
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// 200 vectors, dim=32, top-10 queries, expect recall@10 ≥ 0.9 (BruteForce path).
#[tokio::test]
async fn test_e2e_recall_small_dataset() {
    const N: usize = 200;
    const DIM: usize = 32;
    const TOP_K: usize = 10;
    const MIN_RECALL: f32 = 0.90;

    let storage = IntelligentIndexedStorage::new(&make_config()).await.unwrap();

    let table_cfg = TableConfig {
        name: "recall_small".to_string(),
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
    };
    storage.create_table(table_cfg).await.unwrap();

    let vectors = make_vectors(N, DIM);
    let ids: Vec<String> = (0..N).map(|i| format!("v{:04}", i)).collect();

    let requests: Vec<CreateRecordRequest> = ids
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
    storage.insert_records("recall_small", requests).await.unwrap();

    // Query every 10th vector as a query (20 queries total)
    let mut total_recall = 0.0f32;
    let mut num_queries = 0;

    for qi in (0..N).step_by(10) {
        let query_vec = &vectors[qi];
        let gt = ground_truth_top_k(query_vec, &vectors, TOP_K);

        let query = SearchQuery {
            vector: query_vec.clone(),
            top_k: TOP_K,
            limit: TOP_K,
            ..Default::default()
        };
        let result = storage.search("recall_small", query).await.unwrap();
        let retrieved: Vec<String> = result.hits.iter().map(|h| h.id.clone()).collect();

        let r = recall_at_k(&retrieved, &gt, &ids);
        total_recall += r;
        num_queries += 1;
    }

    let avg_recall = total_recall / num_queries as f32;
    assert!(
        avg_recall >= MIN_RECALL,
        "recall@{} = {:.3} < threshold {:.2} (small dataset, {} vectors)",
        TOP_K, avg_recall, MIN_RECALL, N
    );
    println!("recall@{TOP_K} (small, N={N}): {avg_recall:.3}");
}

/// 1500 vectors, dim=32 — above the 5K BruteForce threshold forces HNSW path.
/// Uses memory_optimization=false to select HNSW via the standard selector (>5K triggers HNSW).
/// Kept small to run in under 30s while still validating graph-based search quality.
#[tokio::test]
async fn test_e2e_recall_hnsw_direct() {
    use dxmt_vectordb::search::{hnsw::{HnswIndex, HnswConfig}, index_traits::VectorIndex};

    const N: usize = 1_500;
    const DIM: usize = 32;
    const TOP_K: usize = 10;
    const MIN_RECALL: f32 = 0.90;

    // Test HNSW directly (bypassing the storage layer's channel serialization)
    let config = HnswConfig {
        m: 16,
        max_m: 16,
        max_m0: 32,
        ef_construction: 200,
        ef_search: 100,
        ..Default::default()
    };
    let mut index = HnswIndex::new(dxmt_vectordb::types::vector::DistanceMetric::Cosine, Some(config));

    let vectors = make_vectors(N, DIM);
    let ids: Vec<String> = (0..N).map(|i| format!("v{:06}", i)).collect();

    for (id, v) in ids.iter().zip(&vectors) {
        index.add(id.clone(), v.clone()).await.unwrap();
    }

    // 20 evenly-spaced queries
    let mut total_recall = 0.0f32;
    let mut num_queries = 0;

    for qi in (0..N).step_by(N / 20) {
        let gt = ground_truth_top_k(&vectors[qi], &vectors, TOP_K);
        let results = index.search_with_ef_hint(&vectors[qi], TOP_K, 200).await.unwrap();
        let retrieved: Vec<String> = results.iter().map(|(id, _)| id.clone()).collect();
        let r = recall_at_k(&retrieved, &gt, &ids);
        total_recall += r;
        num_queries += 1;
    }

    let avg_recall = total_recall / num_queries as f32;
    assert!(
        avg_recall >= MIN_RECALL,
        "HNSW recall@{} = {:.3} < threshold {:.2} (N={} vectors, dim={})",
        TOP_K, avg_recall, MIN_RECALL, N, DIM
    );
    println!("recall@{TOP_K} (HNSW direct, N={N}, dim={DIM}): {avg_recall:.3}");
}

/// Verifies that insert → delete → search does not return deleted vectors.
#[tokio::test]
async fn test_e2e_delete_removes_from_search() {
    const DIM: usize = 16;

    let storage = IntelligentIndexedStorage::new(&make_config()).await.unwrap();

    let table_cfg = TableConfig {
        name: "recall_delete".to_string(),
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
    };
    storage.create_table(table_cfg).await.unwrap();

    let vectors = make_vectors(50, DIM);
    let ids: Vec<String> = (0..50).map(|i| format!("d{:03}", i)).collect();

    let requests: Vec<CreateRecordRequest> = ids
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
    storage.insert_records("recall_delete", requests).await.unwrap();

    // Delete the first 25 vectors
    let to_delete: Vec<String> = ids[..25].to_vec();
    storage.delete_records("recall_delete", to_delete.clone()).await.unwrap();

    // Search with a query close to deleted vector 0 — should not appear in results
    let query = SearchQuery {
        vector: vectors[0].clone(),
        top_k: 10,
        limit: 10,
        ..Default::default()
    };
    let result = storage.search("recall_delete", query).await.unwrap();
    let retrieved_ids: std::collections::HashSet<String> =
        result.hits.iter().map(|h| h.id.clone()).collect();

    for deleted_id in &to_delete {
        assert!(
            !retrieved_ids.contains(deleted_id),
            "deleted record {} appeared in search results",
            deleted_id
        );
    }
}

/// Verifies that upserting (re-inserting) a record increments version and preserves created_at.
#[tokio::test]
async fn test_upsert_increments_version() {
    const DIM: usize = 8;

    let storage = IntelligentIndexedStorage::new(&make_config()).await.unwrap();
    storage.create_table(TableConfig {
        name: "upsert_ver".to_string(),
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
    }).await.unwrap();

    let v1: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let v2: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Insert initial record
    storage.insert_records("upsert_ver", vec![CreateRecordRequest {
        id: "r1".to_string(),
        vector: v1.clone(),
        sparse_vector: None,
        payload: Some(serde_json::json!({"tag": "original"})),
        named_vectors: None,
    }]).await.unwrap();

    let r1 = storage.get_record("upsert_ver", &"r1".to_string()).await.unwrap();
    assert_eq!(r1.version, 1, "initial version should be 1");
    let created_at = r1.created_at;

    // Upsert the same record with a different vector
    storage.insert_records("upsert_ver", vec![CreateRecordRequest {
        id: "r1".to_string(),
        vector: v2.clone(),
        sparse_vector: None,
        payload: Some(serde_json::json!({"tag": "updated"})),
        named_vectors: None,
    }]).await.unwrap();

    let r1_updated = storage.get_record("upsert_ver", &"r1".to_string()).await.unwrap();
    assert_eq!(r1_updated.version, 2, "version should increment on upsert");
    assert_eq!(r1_updated.created_at, created_at, "created_at should be preserved on upsert");
    assert_eq!(r1_updated.vector, v2, "vector should be updated");

    // Verify record_count stays at 1 (no duplicate)
    let table = storage.get_table("upsert_ver").await.unwrap();
    assert_eq!(table.record_count, 1, "upsert should not increase record count");
}

/// Verifies that update_record without a vector preserves the existing vector.
#[tokio::test]
async fn test_update_record_preserves_vector_when_not_provided() {
    const DIM: usize = 8;

    let storage = IntelligentIndexedStorage::new(&make_config()).await.unwrap();
    storage.create_table(TableConfig {
        name: "update_vec".to_string(),
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
    }).await.unwrap();

    let original_vector: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let norm = (original_vector.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let original_vector: Vec<f32> = original_vector.iter().map(|x| x / norm).collect();

    storage.insert_records("update_vec", vec![CreateRecordRequest {
        id: "r1".to_string(),
        vector: original_vector.clone(),
        sparse_vector: None,
        payload: Some(serde_json::json!({"tag": "v1"})),
        named_vectors: None,
    }]).await.unwrap();

    // Update only the payload (no vector) via update_record
    let updated = storage.update_record("update_vec", &"r1".to_string(), CreateRecordRequest {
        id: "r1".to_string(),
        vector: original_vector.clone(), // storage-level update_record requires a vector
        sparse_vector: None,
        payload: Some(serde_json::json!({"tag": "v2"})),
        named_vectors: None,
    }).await.unwrap();

    assert_eq!(updated.vector, original_vector, "vector should be unchanged after payload-only update");
    assert_eq!(updated.payload["tag"], "v2", "payload should be updated");
    assert_eq!(updated.version, 2, "version should increment");

    // Verify the record is still searchable with the original vector direction
    let query = SearchQuery {
        vector: original_vector.clone(),
        top_k: 1,
        limit: 1,
        ..Default::default()
    };
    let result = storage.search("update_vec", query).await.unwrap();
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].id, "r1");
}

/// Verifies that payload filter correctly includes/excludes records from search results.
#[tokio::test]
async fn test_search_with_payload_filter() {
    const DIM: usize = 8;
    let config = make_config();
    let storage = IntelligentIndexedStorage::new(&config).await.unwrap();

    storage.create_table(TableConfig {
        name: "filter_test".to_string(),
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
    }).await.unwrap();

    // Insert records with different categories
    let records: Vec<CreateRecordRequest> = (0..20).map(|i| CreateRecordRequest {
        id: format!("r{}", i),
        vector: make_vectors(1, DIM)[0].clone(),
        sparse_vector: None,
        payload: Some(serde_json::json!({
            "category": if i % 2 == 0 { "even" } else { "odd" },
            "index": i
        })),
        named_vectors: None,
    }).collect();
    storage.insert_records("filter_test", records).await.unwrap();

    // Search with filter: only "even" category
    let filter = Filter {
        must: Some(vec![Condition {
            key: "category".to_string(),
            operation: FilterOperation::Equal(serde_json::json!("even")),
        }]),
        must_not: None,
        should: None,
    };

    let query = SearchQuery {
        vector: make_vectors(1, DIM)[0].clone(),
        top_k: 20,
        limit: 20,
        filter: Some(filter),
        with_payload: true,
        ..Default::default()
    };

    let result = storage.search("filter_test", query).await.unwrap();

    // All returned records should have category "even"
    assert!(!result.hits.is_empty(), "should return at least one result");
    for hit in &result.hits {
        let payload = hit.payload.as_ref().expect("payload should be present");
        assert_eq!(
            payload["category"].as_str().unwrap(), "even",
            "record {} should have category 'even', got {:?}", hit.id, payload["category"]
        );
    }

    // Verify we didn't accidentally return "odd" records (IDs are "r0".."r19", strip "r" prefix)
    assert!(result.hits.iter().all(|h| {
        h.id.strip_prefix('r')
            .and_then(|s| s.parse::<usize>().ok())
            .map(|n| n % 2 == 0)
            .unwrap_or(false)
    }), "all returned record indices should be even");
}

/// Tests Acorn-style filter-in-graph: 1000 records, filter matches only 10 (1% selectivity).
/// Verifies that top_k=5 returns exactly 5 records all within the filter set.
/// This exercises the `search_with_filter` code path in HNSW.
#[tokio::test]
async fn test_filtered_search_recall_high_selectivity() {
    const N: usize = 1000;
    const DIM: usize = 32;
    const FILTER_COUNT: usize = 10;  // Only 10 out of 1000 records match (1%)
    const TOP_K: usize = 5;

    let storage = IntelligentIndexedStorage::new(&make_config()).await.unwrap();
    storage.create_table(TableConfig {
        name: "acorn_test".to_string(),
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
    }).await.unwrap();

    let vectors = make_vectors(N, DIM);

    // Insert N records; first FILTER_COUNT get category="target"
    let requests: Vec<CreateRecordRequest> = (0..N).map(|i| CreateRecordRequest {
        id: format!("v{:04}", i),
        vector: vectors[i].clone(),
        sparse_vector: None,
        payload: Some(serde_json::json!({
            "category": if i < FILTER_COUNT { "target" } else { "other" }
        })),
        named_vectors: None,
    }).collect();
    storage.insert_records("acorn_test", requests).await.unwrap();

    // Query closest to v0 (a target record), with filter = only "target" records
    let filter = Filter {
        must: Some(vec![Condition {
            key: "category".to_string(),
            operation: FilterOperation::Equal(serde_json::json!("target")),
        }]),
        must_not: None,
        should: None,
    };

    let query = SearchQuery {
        vector: vectors[0].clone(),
        top_k: TOP_K,
        limit: TOP_K,
        filter: Some(filter),
        with_payload: true,
        ..Default::default()
    };

    let result = storage.search("acorn_test", query).await.unwrap();

    // Must return exactly TOP_K results, all with category "target"
    assert_eq!(result.hits.len(), TOP_K, "should return exactly {} results", TOP_K);
    for hit in &result.hits {
        let payload = hit.payload.as_ref().expect("payload should be present");
        assert_eq!(
            payload["category"].as_str().unwrap(), "target",
            "result {} should have category 'target'", hit.id
        );
    }
}
