//! A filter on a field no record carries must return nothing, and must cost nothing.
//!
//! The cardinality estimator used to answer "half the table" for a field it had never seen,
//! calling that a conservative estimate. It is the opposite: the payload index holds every
//! field any record has written, so a missing key is the answer, not missing information.
//!
//! What the wrong answer cost, measured on a 5M x 1536 table (design/DISK_INDEX.md 5.5.1):
//! the search read 50% selectivity as a broad filter, chose post-filtering, fetched 600
//! candidates, matched none of them, and took **16.6 s** to return an empty result while
//! holding the table read lock for 4.6 s. The same query with no filter takes 3.6 ms. The
//! benchmark case that exposed it ran with the harness's payload column switched off, which
//! is a configuration mistake -- but a query that cannot match anything should say so
//! immediately whatever the reason, and holding a write-preferring lock for 4.6 s while
//! doing it stalls every other request on the server.
//!
//! What this file does *not* catch is that defect. Post-filtering 600 candidates that all
//! fail still returns the right empty page, so these tests pass on the broken estimate too;
//! at this size the wasted work is invisible. The guard for the route the search takes is
//! `an_absent_field_estimates_zero_not_half_the_table` in `storage::payload_index`. These
//! tests pin the answers, which is worth pinning separately.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{Condition, CreateRecordRequest, Filter, FilterOperation, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};

const ROWS: usize = 3_000;
const DIM: usize = 16;

fn config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
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

fn ge(key: &str, value: usize) -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: key.to_string(),
            operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(value)),
        }]),
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

async fn storage_with(payload: impl Fn(usize) -> serde_json::Value) -> IntelligentIndexedStorage {
    let storage = IntelligentIndexedStorage::new(&config()).await.unwrap();
    storage.create_table(table()).await.unwrap();
    let rows: Vec<CreateRecordRequest> = (0..ROWS)
        .map(|i| CreateRecordRequest {
            id: i.to_string(),
            vector: vector(i),
            sparse_vector: None,
            named_vectors: None,
            payload: Some(payload(i)),
        })
        .collect();
    storage.insert_records("t", rows).await.unwrap();
    storage
}

#[tokio::test]
async fn a_filter_on_a_field_no_record_has_returns_nothing() {
    // Payloads that carry a different field, which is what an unfiltered load writes.
    let storage = storage_with(|i| serde_json::json!({ "other": i })).await;

    let hits = storage
        .search("t", query(Some(ge("id_num", 1))))
        .await
        .unwrap();
    assert!(
        hits.hits.is_empty(),
        "a filter on an absent field matched {} rows",
        hits.hits.len()
    );

    // The same table answers an unfiltered query normally, so the empty result is the
    // filter's doing and not a broken table.
    let unfiltered = storage.search("t", query(None)).await.unwrap();
    assert_eq!(unfiltered.hits.len(), 10);

    // And a filter on the field that does exist still works.
    let present = storage
        .search("t", query(Some(ge("other", 0))))
        .await
        .unwrap();
    assert_eq!(present.hits.len(), 10, "every row has other >= 0");
}

#[tokio::test]
async fn a_range_that_excludes_every_row_returns_nothing() {
    let storage = storage_with(|i| serde_json::json!({ "id_num": i })).await;

    // Above every value in the table.
    let above = storage
        .search("t", query(Some(ge("id_num", ROWS * 10))))
        .await
        .unwrap();
    assert!(
        above.hits.is_empty(),
        "a range above the maximum matched {} rows",
        above.hits.len()
    );

    // A range that admits everything returns a full page, and only matching rows.
    let all = storage
        .search("t", query(Some(ge("id_num", 0))))
        .await
        .unwrap();
    assert_eq!(all.hits.len(), 10);

    // A selective range returns only rows that pass it.
    let tail = storage
        .search("t", query(Some(ge("id_num", ROWS - 50))))
        .await
        .unwrap();
    assert!(
        !tail.hits.is_empty(),
        "the last 50 rows should be reachable"
    );
    for h in &tail.hits {
        let id: usize = h.id.parse().expect("numeric id");
        assert!(id >= ROWS - 50, "row {id} does not pass the filter");
    }
}
