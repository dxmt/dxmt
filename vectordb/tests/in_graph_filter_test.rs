//! A selective filter must not fall off a cliff because its matching set is one row too big.
//!
//! `Performance768D1M99P` passes 1% of the rows -- exactly 10,000 of them -- and the in-graph
//! filter was gated on `cardinality.exp < 10000`, so it missed by one and took the fallback:
//! an *unfiltered* walk with `k = limit / selectivity * 3`, which makes `ef` 30,000 where an
//! unfiltered query uses 256. Measured: 47.9 QPS at a 207.7 ms p99, against 2,307 QPS and
//! 5.0 ms with no filter at all.
//!
//! Widening the gate to admit it was tried and taken back out: measured on that case, the
//! in-graph walk ran at 26.7 QPS against the over-fetching path's 76.9, because it visits
//! about `ef / selectivity` nodes and materialises the whole matching id set on every query.
//! The cap stays at 10,000 and `IN_GRAPH_FILTER_MAX_IDS` carries the numbers.
//!
//! What this file pins is that the two paths agree. The in-graph one has no top-up if it
//! comes back short -- the retry in phase 3 only covers post-filtering -- so both must return
//! a full k, both must return only rows that pass, and neither may be the less accurate.
//!
//! `DXMT_IN_GRAPH_FILTER_MAX_IDS` is process-global, so this is the only test in this binary.

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

const ROWS: usize = 60_000;
const DIM: usize = 16;
/// 4% of the rows: under the 5% the in-graph branch requires, and the 2,400 matching ids are
/// comfortably over `EXACT_FILTER_SCAN_MAX`, so this exercises the traversal rather than the
/// exact-scan shortcut.
const MATCH_EVERY: usize = 25;
const K: usize = 20;

fn config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 4096,
            eviction_policy: "lru".to_string(),
        }),
        ..Default::default()
    }
}

/// Deterministic and all distinct. Vectors built from `i % n` would make the true nearest
/// neighbours a tie among hundreds of identical rows, and any recall number taken over that
/// measures the tie-break rather than the search.
fn vectors() -> Vec<Vec<f32>> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    (0..ROWS)
        .map(|_| (0..DIM).map(|_| next()).collect())
        .collect()
}

fn matching(i: usize) -> bool {
    i.is_multiple_of(MATCH_EVERY)
}

/// Cosine, not the raw dot product. The table is cosine, so ranking the ground truth by dot
/// ranks by magnitude as well as direction -- these vectors have varying norms, and an
/// earlier version of this file did exactly that and reported recall 0.65 for a search that
/// was returning the right rows.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// The right answer, computed directly over the rows that pass the filter.
fn brute_force(vs: &[Vec<f32>], q: &[f32]) -> Vec<String> {
    let mut scored: Vec<(usize, f32)> = (0..ROWS)
        .filter(|&i| matching(i))
        .map(|i| (i, cosine(&vs[i], q)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
        .into_iter()
        .take(K)
        .map(|(i, _)| i.to_string())
        .collect()
}

fn selective_query(vector: Vec<f32>) -> SearchQuery {
    SearchQuery {
        vector,
        top_k: K,
        limit: K,
        filter: Some(Filter {
            must: Some(vec![Condition {
                key: "bucket".to_string(),
                operation: FilterOperation::Equal(serde_json::json!(0)),
            }]),
            must_not: None,
            should: None,
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn a_selective_filter_returns_a_full_result_set_on_either_path() {
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
            payload_index: None,
            embedding: None,
        })
        .await
        .unwrap();

    let vs = vectors();
    for chunk in (0..ROWS).collect::<Vec<_>>().chunks(5_000) {
        let batch: Vec<CreateRecordRequest> = chunk
            .iter()
            .map(|&i| CreateRecordRequest {
                id: i.to_string(),
                vector: vs[i].clone(),
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::json!({
                    "bucket": if matching(i) { 0 } else { 1 + (i % 7) }
                })),
                text: None,
            })
            .collect();
        storage.insert_records("t", batch).await.unwrap();
    }

    let query = vs[7].clone();
    let truth = brute_force(&vs, &query);

    // Cap 1 forces the fallback: an unfiltered walk with an amplified fetch, then a filter.
    std::env::set_var("DXMT_IN_GRAPH_FILTER_MAX_IDS", "1");
    let fallback = storage
        .search("t", selective_query(query.clone()))
        .await
        .unwrap();
    // Raised so this matching set is admitted and the filter is applied during traversal.
    // The shipped default is 10,000, which this table's 2,400 matching rows sit under anyway;
    // setting it explicitly keeps the test honest about which path it is exercising.
    std::env::set_var("DXMT_IN_GRAPH_FILTER_MAX_IDS", "100000");
    let in_graph = storage
        .search("t", selective_query(query.clone()))
        .await
        .unwrap();
    std::env::remove_var("DXMT_IN_GRAPH_FILTER_MAX_IDS");

    for (name, hits) in [("fallback", &fallback), ("in-graph", &in_graph)] {
        assert_eq!(
            hits.hits.len(),
            K,
            "{name} returned {} of {K} results; a selective filter must not come back short",
            hits.hits.len()
        );
        for h in &hits.hits {
            let i: usize = h.id.parse().expect("ids are row numbers");
            assert!(
                matching(i),
                "{name} returned row {i}, which does not pass the filter"
            );
        }
    }

    let recall = |hits: &dxmt_vectordb::types::record::SearchResult| {
        let got: std::collections::HashSet<&str> =
            hits.hits.iter().map(|h| h.id.as_str()).collect();
        truth.iter().filter(|id| got.contains(id.as_str())).count() as f64 / K as f64
    };
    let (r_fallback, r_in_graph) = (recall(&fallback), recall(&in_graph));
    println!("recall: in-graph {r_in_graph:.3}, amplified fallback {r_fallback:.3}");
    assert!(
        r_in_graph >= 0.8,
        "in-graph recall {r_in_graph:.2}, fallback recall {r_fallback:.2}, \
         against brute force over the matching rows"
    );
    assert!(
        r_in_graph >= r_fallback - 0.1,
        "the in-graph path is meant to be cheaper, not worse: {r_in_graph:.2} against the \
         fallback's {r_fallback:.2}"
    );
}
