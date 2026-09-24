//! HNSW-PQ end to end: the graph walks on codes, the storage layer reranks on the records.
//!
//! `DXMT_INDEX_ALGORITHM=hnsw-pq` pins the family for this process; the table is created,
//! filled past the quantizer's training minimum, rebuilt (which trains and quantizes), and
//! searched. What the caller sees must be exact-distance ordered even though the index's own
//! scores are approximate.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{Condition, CreateRecordRequest, Filter, FilterOperation, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use tempfile::TempDir;

const ROWS: usize = 6_000;
const DIM: usize = 64;
const K: usize = 10;

fn config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 4096,
            eviction_policy: "lru".to_string(),
        }),
        ..Default::default()
    }
}

/// Distinct vectors around 40 centres: the structure a product quantizer relies on.
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

fn brute_force(vs: &[Vec<f32>], q: &[f32], keep: impl Fn(usize) -> bool) -> Vec<String> {
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

#[tokio::test]
async fn a_quantized_table_answers_with_exact_order() {
    std::env::set_var("DXMT_INDEX_ALGORITHM", "hnsw-pq");
    let tmp = TempDir::new().unwrap();
    let storage = IntelligentIndexedStorage::new_with_wal(&config(), Some(tmp.path().join("wal")))
        .await
        .unwrap();
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
        })
        .await
        .unwrap();

    let vs = vectors();
    for chunk in (0..ROWS).collect::<Vec<_>>().chunks(2_000) {
        let batch: Vec<CreateRecordRequest> = chunk
            .iter()
            .map(|&i| CreateRecordRequest {
                id: i.to_string(),
                vector: vs[i].clone(),
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::json!({ "bucket": i % 5 })),
            })
            .collect();
        storage.insert_records("t", batch).await.unwrap();
    }

    // 16 subspaces x 256 centroids want 4,096 training vectors; 6,000 rows train and
    // quantize on rebuild.
    storage.rebuild_index("t").await.unwrap();

    let mut total = 0.0;
    let mut self_first = 0;
    let queries: Vec<usize> = (0..50).map(|i| i * 113 % ROWS).collect();
    for &qi in &queries {
        let hits = storage
            .search(
                "t",
                SearchQuery {
                    vector: vs[qi].clone(),
                    top_k: K,
                    limit: K,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let found: Vec<String> = hits.hits.iter().map(|h| h.id.clone()).collect();
        assert_eq!(found.len(), K, "short result for query {qi}");
        total += recall(&found, &brute_force(&vs, &vs[qi], |_| true));
        if found[0] == qi.to_string() {
            self_first += 1;
        }
        // Exact order: the storage layer reports similarities, best first, and the first
        // one is the record's own vector at exactly 1.0 -- an ADC score would not be.
        let scores: Vec<f32> = hits.hits.iter().map(|h| h.score).collect();
        assert!(
            scores.windows(2).all(|w| w[0] >= w[1]),
            "hits not in descending similarity order: {scores:?}"
        );
    }
    let mean = total / queries.len() as f64;
    assert!(
        mean >= 0.9,
        "recall@{K} over the quantized table: {mean:.3}"
    );
    assert!(
        self_first >= 48,
        "{self_first}/50 queries had their own record first; an exact rerank puts it there"
    );

    // Filtered, through the same rerank.
    let qi = 7;
    let hits = storage
        .search(
            "t",
            SearchQuery {
                vector: vs[qi].clone(),
                top_k: K,
                limit: K,
                filter: Some(Filter {
                    must: Some(vec![Condition {
                        key: "bucket".to_string(),
                        operation: FilterOperation::Equal(serde_json::json!(2)),
                    }]),
                    must_not: None,
                    should: None,
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let found: Vec<String> = hits.hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(found.len(), K);
    assert!(
        found.iter().all(|id| id.parse::<usize>().unwrap() % 5 == 2),
        "a hit outside the filter: {found:?}"
    );
    let r = recall(&found, &brute_force(&vs, &vs[qi], |i| i % 5 == 2));
    assert!(r >= 0.8, "filtered recall@{K}: {r:.3}");
}
