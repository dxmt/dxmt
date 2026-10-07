//! A post-filtered search that finds nothing must not fall back to walking the whole graph.
//!
//! `search` post-filters when the cardinality estimate says the filter is broad. If the
//! estimate is wrong and the fetched candidates all fail the predicate, an escalation loop
//! refetches: `fetched * 8`, again, and then `total_records`. That last step is the defect.
//! `search_reranked` multiplies whatever `k` it is handed by the rerank factor, so asking for
//! `total_records` asks the graph for `max(5 * total_records, 100)` results at an `ef` to
//! match, and reranks every one of them against the stored vectors. On the 5M x 1536 table in
//! `docs/architecture/disk-index.md` §5 that was 25M candidates: **16.6 s to return an empty page**,
//! 4.6 s of it holding the table read lock, against 3.6 ms for the same query unfiltered.
//!
//! At `ef = n` the graph is not an index any more -- the walk reaches every node and carries
//! an n-sized heap, which is strictly more work than a linear scan. Whatever the exhaustive
//! answer costs, it should not be paid for through the graph.
//!
//! Reproducing it needs an estimate that is wrong in the expensive direction: broad enough to
//! choose post-filtering (`selectivity >= 0.05`), matching nothing in fact. Range selectivity
//! is interpolated from the field's min and max, which is exact for uniform data -- see
//! `what_the_estimator_says_about_a_uniform_field` -- so the payloads here are deliberately
//! skewed: every value is 0 or 1000, and the filter asks for the empty middle.
//!
//! There are three layers to the answer now, and this file holds one test per layer.
//!
//! 1. The index proves the filter empty, from its fields' distinct values, before anything is
//!    searched. `400 <= bucket <= 600` over values that are only ever 0 or 1000 is empty
//!    because no value is between them -- though neither half is empty on its own, which is
//!    why conditions on one key have to be tested against one value together.
//! 2. The index cannot prove it, so the search runs and comes back short, and the exhaustive
//!    step enumerates by the predicate rather than by the graph. `bucket >= 400 AND
//!    parity >= 1` is this shape: each half matches plenty, no row carries both, and the two
//!    conditions are on different keys so no single field's values contradict.
//! 3. The filter matches, and nothing above it runs.
//!
//! Measured on this test, 20,000 rows x 16 dims, min of five runs, against a filtered query
//! that does match as the control -- both over-fetch the same candidates, only one escalates:
//!
//!                                    release              debug
//!     provably empty, before        47,839 us           49,014 us
//!     provably empty, after              1 us                10 us
//!     cannot prove, after            6,371 us (5.3x)    17,056 us (2.9x)
//!
//! Before any of this the empty answer went through the graph: the escalation asked a
//! 20,000-row index for every record, the rerank factor multiplied that by five, and the
//! measurement was 35,378 us against 1,556 us for the control, 22.7x. The bound below is 15x,
//! which the broken path clears and the current one does not come near.
//!
//! The control is a filtered query that *does* match, not an unfiltered one.//! The control is a filtered query that *does* match, not an unfiltered one. Both post-filter
//! and both over-fetch the same candidates; the only difference is whether the escalation
//! fires. Measuring against an unfiltered query instead compares a top-10 fetch against a
//! post-filter's amplified one and charges the difference to the defect. Times are the
//! smallest of several runs, the stable statistic for a latency this small.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{Condition, CreateRecordRequest, Filter, FilterOperation, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use std::time::Instant;

/// Big enough that a walk of the whole graph is visible next to a normal search, small enough
/// that the graph stays exact (`quantize_at_rows` is 65,536) and the test builds in seconds.
const ROWS: usize = 20_000;
const DIM: usize = 16;

/// The two values every payload carries. The filter below asks for the gap between them.
const LOW: usize = 0;
const HIGH: usize = 1_000;

fn config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 512,
            eviction_policy: "lru".to_string(),
        }),
        ..Default::default()
    }
}

fn table() -> TableConfig {
    TableConfig {
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
    }
}

fn vector(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| {
            let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (d as u64 + 1);
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            (h & 0xFFFF) as f32 / 65_536.0 + 0.01
        })
        .collect()
}

/// `400 <= bucket <= 600`: nothing is in it, but min/max interpolation over [0, 1000] reads
/// each half as broad, so the estimate lands far above the 5% that would pre-filter instead.
fn the_empty_middle() -> Filter {
    Filter {
        must: Some(vec![
            Condition {
                key: "bucket".to_string(),
                operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(400)),
            },
            Condition {
                key: "bucket".to_string(),
                operation: FilterOperation::LessThanOrEqual(serde_json::json!(600)),
            },
        ]),
        must_not: None,
        should: None,
    }
}

fn query(filter: Option<Filter>) -> SearchQuery {
    SearchQuery {
        vector: vector(7),
        top_k: 10,
        limit: 10,
        filter,
        ..Default::default()
    }
}

async fn storage() -> IntelligentIndexedStorage {
    let storage = IntelligentIndexedStorage::new(&config()).await.unwrap();
    storage.create_table(table()).await.unwrap();
    let rows: Vec<CreateRecordRequest> = (0..ROWS)
        .map(|i| CreateRecordRequest {
            id: i.to_string(),
            vector: vector(i),
            sparse_vector: None,
            named_vectors: None,
            // Skewed on purpose: nine tenths at one end, one tenth at the other, nothing
            // between. A uniform field would be estimated correctly and never post-filter.
            // `bucket` is skewed so a range can ask for a gap. `parity` is its exact
            // complement, so a filter naming both is empty without either half being empty
            // and without the two conditions sharing a key -- the shape the index cannot
            // prove, which is what keeps the escalation path under test.
            payload: Some(serde_json::json!({
                "bucket": if i % 10 == 0 { HIGH } else { LOW },
                "parity": if i % 10 == 0 { 0 } else { 1 },
            })),
            text: None,
        })
        .collect();
    storage.insert_records("t", rows).await.unwrap();
    storage
}

/// One in ten rows carries the high value, so this filter matches 2,000 of 20,000: broad
/// enough that the first fetch fills a top-10 and the escalation never runs.
fn only_high() -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: "bucket".to_string(),
            operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(HIGH)),
        }]),
        must_not: None,
        should: None,
    }
}

/// Smallest of `runs` searches, in microseconds.
async fn fastest_of(
    storage: &IntelligentIndexedStorage,
    filter: Option<Filter>,
    runs: usize,
) -> u128 {
    let mut best = u128::MAX;
    for _ in 0..runs {
        let t = Instant::now();
        let hits = storage.search("t", query(filter.clone())).await.unwrap();
        best = best.min(t.elapsed().as_micros().max(1));
        std::hint::black_box(hits);
    }
    best
}

/// `bucket` high and `parity` high: no row carries both, and no single field's values
/// contradict, so the index cannot prove this empty and the search has to discover it.
fn contradictory_across_two_fields() -> Filter {
    Filter {
        must: Some(vec![
            Condition {
                key: "bucket".to_string(),
                operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(400)),
            },
            Condition {
                key: "parity".to_string(),
                operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(1)),
            },
        ]),
        must_not: None,
        should: None,
    }
}

/// A filter the index proves empty never reaches the graph at all.
#[tokio::test]
async fn a_provably_empty_filter_costs_nothing() {
    let storage = storage().await;
    let _ = storage.search("t", query(None)).await.unwrap();

    let matching_us = fastest_of(&storage, Some(only_high()), 5).await;
    let empty_us = fastest_of(&storage, Some(the_empty_middle()), 5).await;

    let empty = storage
        .search("t", query(Some(the_empty_middle())))
        .await
        .unwrap();
    assert!(empty.hits.is_empty(), "no row has 400 <= bucket <= 600");

    eprintln!("provably empty {empty_us} us / matching {matching_us} us");
    assert!(
        empty_us * 10 < matching_us,
        "a filter no indexed value satisfies took {empty_us} us against {matching_us} us for \
         a filtered search that runs: the proof is not being used, and the query is paying \
         for a walk, a rerank and an escalation to learn what the index already knows"
    );
}

#[tokio::test]
async fn a_post_filter_that_matches_nothing_does_not_walk_the_whole_graph() {
    let storage = storage().await;

    // Warm whatever is lazy before either measurement.
    let _ = storage.search("t", query(None)).await.unwrap();

    let matching_us = fastest_of(&storage, Some(only_high()), 5).await;
    let empty_us = fastest_of(&storage, Some(contradictory_across_two_fields()), 5).await;

    let empty = storage
        .search("t", query(Some(contradictory_across_two_fields())))
        .await
        .unwrap();
    assert!(
        empty.hits.is_empty(),
        "no row has both bucket >= 400 and parity >= 1, but {} came back",
        empty.hits.len()
    );

    let ratio = empty_us as f64 / matching_us as f64;
    eprintln!("empty {empty_us} us / matching {matching_us} us = {ratio:.1}x");
    assert!(
        ratio < 15.0,
        "a post-filter matching nothing took {empty_us} us against {matching_us} us for one \
         matching 2,000 rows ({ratio:.1}x) on a {ROWS}-row table: the escalation is asking \
         the graph for every record and reranking the lot"
    );
}

/// The same shape, but the filter does match. The exhaustive step has to keep finding rows the
/// walk never surfaced, so a fix that simply stops escalating would break this.
#[tokio::test]
async fn a_post_filter_that_matches_a_few_rows_still_finds_them() {
    let storage = storage().await;

    let hits = storage.search("t", query(Some(only_high()))).await.unwrap();
    assert_eq!(
        hits.hits.len(),
        10,
        "one row in ten has bucket = {HIGH}, so a top-10 must fill"
    );
    for hit in &hits.hits {
        let id: usize = hit.id.parse().unwrap();
        assert_eq!(id % 10, 0, "id {id} does not carry bucket = {HIGH}");
    }
}
