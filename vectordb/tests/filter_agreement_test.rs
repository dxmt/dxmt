//! A filter means one thing, whichever endpoint answers it.
//!
//! `records/list` and several search paths — a selective filtered search, `search_mode:
//! "exact"`, the k-NN graph — take the payload index's answer as final. Everything else walks
//! the graph and asks `Record::matches_filter`. So wherever the two disagreed, the same filter
//! returned different records depending on how it was asked, and a randomised comparison found
//! them disagreeing in five places, most of them on nearly every query:
//!
//! * `NotEqual` dropped every record *lacking* the field, and every record holding an array.
//! * `id` was answered from a payload field of that name rather than the record's own id.
//! * Numbers were keyed at six decimal places and saturated at ±9.2e12, so every microsecond
//!   timestamp was one key and `Equal` on one matched them all.
//! * `5` and `5.0` were equal to the index and unequal to `matches_filter`.
//!
//! This checks the endpoints a user calls against the records themselves: every record is
//! fetched unfiltered and `matches_filter` decides which should come back.
//!
//! Not covered, deliberately: a filter whose *target* is an array or an object
//! (`tags == ["a","b"]`). The index holds no such values and cannot answer it; making it able
//! to would mean serialising every array in every payload into a key.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{
            Condition, CreateRecordRequest, Filter, FilterOperation, SearchMode, SearchQuery,
        },
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use std::collections::BTreeSet;

const ROWS: usize = 400;
const DIM: usize = 8;

fn vector(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| (((i * 31 + d * 7) % 97) as f32 / 97.0) + 0.01)
        .collect()
}

/// Every shape the comparison found a disagreement on, and some that never had one.
fn payload(i: usize) -> serde_json::Value {
    if i.is_multiple_of(23) {
        return serde_json::Value::Null; // no fields at all
    }
    let mut p = serde_json::Map::new();
    if !i.is_multiple_of(7) {
        p.insert("topic".into(), ["alpha", "beta", "gamma"][i % 3].into());
    }
    if i.is_multiple_of(5) {
        p.insert("tags".into(), serde_json::json!(["a", "b"]));
    }
    // Alternately an integer and the same number as a float.
    let n = (i % 10) as i64;
    p.insert(
        "n".into(),
        if i.is_multiple_of(2) {
            n.into()
        } else {
            (n as f64).into()
        },
    );
    p.insert(
        "us".into(),
        (1_700_000_000_000_000i64 + (i % 4) as i64).into(),
    );
    if i.is_multiple_of(11) {
        p.insert("id".into(), "not-the-record-id".into());
    }
    serde_json::Value::Object(p)
}

fn one(key: &str, operation: FilterOperation) -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: key.into(),
            operation,
        }]),
        must_not: None,
        should: None,
    }
}

fn filters() -> Vec<(&'static str, Filter)> {
    use serde_json::json;
    use FilterOperation::*;
    vec![
        ("topic != alpha", one("topic", NotEqual(json!("alpha")))),
        (
            "topic not in [alpha]",
            one("topic", NotIn(vec![json!("alpha")])),
        ),
        ("tags != a", one("tags", NotEqual(json!("a")))),
        ("tags not in [a]", one("tags", NotIn(vec![json!("a")]))),
        ("id == 42", one("id", Equal(json!("42")))),
        (
            "id in [1, 2, 99]",
            one("id", In(vec![json!("1"), json!("2"), json!("99")])),
        ),
        ("id != 42", one("id", NotEqual(json!("42")))),
        ("id starts with 3", one("id", StartsWith("3".into()))),
        ("n == 4", one("n", Equal(json!(4)))),
        ("n == 4.0", one("n", Equal(json!(4.0)))),
        ("n != 4.0", one("n", NotEqual(json!(4.0)))),
        ("n in [3, 5.0]", one("n", In(vec![json!(3), json!(5.0)]))),
        (
            "us == …002",
            one("us", Equal(json!(1_700_000_000_000_002i64))),
        ),
        (
            "us > …001",
            one("us", GreaterThan(json!(1_700_000_000_000_001i64))),
        ),
        (
            "must_not topic != beta",
            Filter {
                must: None,
                must_not: Some(vec![Condition {
                    key: "topic".into(),
                    operation: NotEqual(json!("beta")),
                }]),
                should: None,
            },
        ),
    ]
}

#[tokio::test]
async fn list_and_exact_search_agree_with_the_records() {
    let storage = IntelligentIndexedStorage::new(&StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 256,
            eviction_policy: "lru".to_string(),
        }),
        ..Default::default()
    })
    .await
    .unwrap();
    storage
        .create_table(TableConfig {
            name: "t".into(),
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
        })
        .await
        .unwrap();
    let rows = (0..ROWS)
        .map(|i| CreateRecordRequest {
            id: i.to_string(),
            vector: vector(i),
            sparse_vector: None,
            named_vectors: None,
            payload: Some(payload(i)),
        })
        .collect();
    storage.insert_records("t", rows).await.unwrap();

    let (all, total) = storage
        .list_records("t", None, ROWS * 2, 0, false)
        .await
        .unwrap();
    assert_eq!(total, ROWS);

    let mut disagreements = Vec::new();
    for (name, filter) in filters() {
        let want: BTreeSet<String> = all
            .iter()
            .filter(|r| r.matches_filter(&filter))
            .map(|r| r.id.clone())
            .collect();

        let (listed, listed_total) = storage
            .list_records("t", Some(&filter), ROWS * 2, 0, false)
            .await
            .unwrap();
        let listed: BTreeSet<String> = listed.into_iter().map(|r| r.id).collect();
        if listed != want || listed_total != want.len() {
            disagreements.push(format!(
                "records/list  {name}: {} returned, {} expected",
                listed.len(),
                want.len()
            ));
        }

        let searched = storage
            .search(
                "t",
                SearchQuery {
                    vector: vector(1),
                    top_k: ROWS,
                    limit: ROWS,
                    filter: Some(filter.clone()),
                    search_mode: SearchMode::Exact,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let searched: BTreeSet<String> = searched.hits.into_iter().map(|h| h.id).collect();
        if searched != want {
            disagreements.push(format!(
                "exact search  {name}: {} returned, {} expected",
                searched.len(),
                want.len()
            ));
        }
    }
    assert!(
        disagreements.is_empty(),
        "the endpoints disagree with the records:\n  {}",
        disagreements.join("\n  ")
    );
}
