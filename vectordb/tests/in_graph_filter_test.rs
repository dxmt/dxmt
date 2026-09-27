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
//!
//! **What the counters found when they were added (2026-09-27).** Everything above describes a
//! comparison this file was not making. With the default index, both searches took
//! `FILTER-CODE-SCAN` -- one path against itself -- and the assertions below passed anyway.
//!
//! There are five branches, not the two this file assumed. The planner tries them in order, and the code-scan gate does not consult
//! `DXMT_IN_GRAPH_FILTER_MAX_IDS`: it fires whenever selectivity is under 5% and the match set
//! is at most `CODE_SCAN_MAX` (100,000). The in-graph branch is tried afterwards and needs
//! `cardinality.exp < IN_GRAPH_FILTER_MAX_IDS` (10,000). Reaching it therefore needs a match
//! set both larger than 100,000 and smaller than 10,000, which is nothing --
//! *unless the index has no `scan_ids`*. `HnswPqIndex` implements it and returns `Some`
//! unconditionally; `HnswIndex` does not, so the trait default returns `None` and the planner
//! falls through.
//!
//! So: **on the default index (HNSW-PQ at every size) the in-graph branch never runs.** It is
//! reachable only with `DXMT_INDEX_ALGORITHM=hnsw`. That is not necessarily wrong -- code-scan
//! superseded it and measured far better, which is why it exists -- but it was not written down
//! anywhere, and the test named after the branch was not reaching it.
//!
//! This file now pins both facts: the default index takes code-scan, and the in-graph branch
//! works when it is reached. Each search asserts which counter moved, so a future change to the
//! branch order fails here instead of quietly retiring a path.

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

/// A table of `ROWS` rows with the filter field, on whatever index the environment selects.
///
/// Taken out of the test body so the same corpus can be built twice, once per index family:
/// the branch a filtered search takes depends on whether the index has a `scan_ids`, so the
/// index is the variable and everything else has to be held still.
async fn build(tmp: &TempDir, vs: &[Vec<f32>]) -> IntelligentIndexedStorage {
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
    storage
}

#[tokio::test]
async fn a_selective_filter_returns_a_full_result_set_on_either_path() {
    let tmp = TempDir::new().unwrap();
    let vs = vectors();
    let storage = build(&tmp, &vs).await;

    let query = vs[7].clone();
    let truth = brute_force(&vs, &query);

    // Which branch answered, read rather than assumed. The comments below say which path each
    // search is meant to take; until these counters existed, nothing checked it, and a
    // comparison between two paths that were secretly the same path passes.
    let paths = || async { storage.health_snapshot().await.filter_paths };

    // The default index first. `DXMT_IN_GRAPH_FILTER_MAX_IDS` has no effect here at all --
    // the code-scan gate does not read it -- which is the thing this file used to miss.
    let before = paths().await;
    let code_scan = storage
        .search("t", selective_query(query.clone()))
        .await
        .unwrap();
    let after = paths().await;
    assert_eq!(
        (
            after.code_scan - before.code_scan,
            after.in_graph - before.in_graph,
            after.post_filter - before.post_filter
        ),
        (1, 0, 0),
        "on HNSW-PQ a 4% filter is answered by scanning the codes, whatever the in-graph \
         cap says; it went {before:?} -> {after:?}"
    );

    // Cap 1 with the same index changes nothing, which is worth pinning rather than assuming.
    std::env::set_var("DXMT_IN_GRAPH_FILTER_MAX_IDS", "1");
    let before = paths().await;
    let _ = storage
        .search("t", selective_query(query.clone()))
        .await
        .unwrap();
    let after = paths().await;
    std::env::remove_var("DXMT_IN_GRAPH_FILTER_MAX_IDS");
    assert_eq!(
        after.code_scan - before.code_scan,
        1,
        "the in-graph cap does not gate the code-scan branch, so this is still code-scan: \
         {before:?} -> {after:?}"
    );
    // A plain HNSW has no `scan_ids`, so the planner falls through to the two branches this
    // file is named after. Same rows, same query, same filter -- only the index differs.
    let hnsw_tmp = TempDir::new().unwrap();
    std::env::set_var("DXMT_INDEX_ALGORITHM", "hnsw");
    let hnsw = build(&hnsw_tmp, &vs).await;
    let hnsw_paths = || async { hnsw.health_snapshot().await.filter_paths };

    let before = hnsw_paths().await;
    let in_graph = hnsw
        .search("t", selective_query(query.clone()))
        .await
        .unwrap();
    let after = hnsw_paths().await;
    assert_eq!(
        (
            after.in_graph - before.in_graph,
            after.code_scan - before.code_scan
        ),
        (1, 0),
        "a plain HNSW has no code scan, so a 4% filter is the in-graph walk: \
         {before:?} -> {after:?}"
    );

    // Cap 1 shuts the in-graph branch and leaves the amplified unfiltered walk.
    std::env::set_var("DXMT_IN_GRAPH_FILTER_MAX_IDS", "1");
    let before = hnsw_paths().await;
    let fallback = hnsw
        .search("t", selective_query(query.clone()))
        .await
        .unwrap();
    let after = hnsw_paths().await;
    std::env::remove_var("DXMT_IN_GRAPH_FILTER_MAX_IDS");
    std::env::remove_var("DXMT_INDEX_ALGORITHM");
    // 4% selectivity lands in the 5-30% branch's neighbour, `PRE-FILTERING`, once in-graph is
    // shut: both amplify an unfiltered fetch, and which one runs is decided at 30%.
    assert_eq!(
        (
            after.pre_filter - before.pre_filter,
            after.in_graph - before.in_graph
        ),
        (1, 0),
        "with the cap at 1 and no code scan this is the amplified pre-filter fetch: \
         {before:?} -> {after:?}"
    );

    for (name, hits) in [
        ("code-scan", &code_scan),
        ("fallback", &fallback),
        ("in-graph", &in_graph),
    ] {
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
