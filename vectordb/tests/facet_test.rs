//! Counting the values of a field, and being honest about what the count leaves out.
//!
//! The payload index is already a value→records map, so a facet is a walk of postings rather
//! than of rows. The cost of that is what this file is mostly about: **an index-backed facet
//! loses whatever the index does not hold as a count of zero, not as an absence.** A record
//! whose value for the field is an object is in no bucket; so is one with no value at all; and
//! a chart drawn from the buckets alone shows a corpus that is not the corpus.
//!
//! So the result carries `covered` and `matched`, and their difference is exactly the records
//! the facet cannot describe. Several of the tests here exist only to pin that number, because
//! it is the one a future change would quietly stop maintaining while the counts stayed right.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{Condition, CreateRecordRequest, Filter, FilterOperation},
        table::{PayloadIndexPolicy, TableConfig},
        vector::DistanceMetric,
    },
};
use serde_json::json;

const DIM: usize = 4;

fn config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 128,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

fn table(name: &str, payload_index: Option<PayloadIndexPolicy>) -> TableConfig {
    TableConfig {
        name: name.to_string(),
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
        payload_index,
        embedding: None,
    }
}

fn record(id: &str, payload: serde_json::Value) -> CreateRecordRequest {
    CreateRecordRequest {
        id: id.to_string(),
        vector: vec![1.0, 0.0, 0.0, 0.0],
        text: None,
        sparse_vector: None,
        named_vectors: None,
        payload: Some(payload),
    }
}

async fn open() -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new(&config()).await.unwrap()
}

fn equals(key: &str, value: serde_json::Value) -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: key.to_string(),
            operation: FilterOperation::Equal(value),
        }]),
        must_not: None,
        should: None,
    }
}

/// `[(value, count)]`, as returned.
fn pairs(result: &dxmt_vectordb::storage::payload_index::FacetResult) -> Vec<(String, usize)> {
    result
        .hits
        .iter()
        .map(|hit| (hit.value.to_string(), hit.count))
        .collect()
}

async fn library() -> IntelligentIndexedStorage {
    let storage = open().await;
    storage.create_table(table("docs", None)).await.unwrap();
    storage
        .insert_records(
            "docs",
            vec![
                record("a", json!({"topic": "rust", "year": 2024})),
                record("b", json!({"topic": "rust", "year": 2024})),
                record("c", json!({"topic": "rust", "year": 2025})),
                record("d", json!({"topic": "go", "year": 2025})),
                record("e", json!({"topic": "sql", "year": 2023})),
            ],
        )
        .await
        .unwrap();
    storage
}

#[tokio::test]
async fn values_are_counted_most_common_first() {
    let storage = library().await;
    let facet = storage.facet("docs", "topic", None, 10).await.unwrap();

    assert_eq!(
        pairs(&facet),
        vec![
            ("\"rust\"".to_string(), 3),
            ("\"go\"".to_string(), 1),
            ("\"sql\"".to_string(), 1),
        ],
        "counts descending, then value ascending so a tie is not arbitrary"
    );
    assert_eq!(facet.distinct_values, 3);
    assert_eq!((facet.covered, facet.matched), (5, 5));
}

#[tokio::test]
async fn numbers_are_counted_as_numbers_and_come_back_as_numbers() {
    // The numeric half of the index is a sorted array rather than a value→records map, so this
    // is a different code path from the keyword half and returns 2024 rather than "2024".
    let storage = library().await;
    let facet = storage.facet("docs", "year", None, 10).await.unwrap();
    assert_eq!(
        pairs(&facet),
        vec![
            ("2024".to_string(), 2),
            ("2025".to_string(), 2),
            ("2023".to_string(), 1),
        ]
    );
    assert!(facet.hits[0].value.is_number(), "{:?}", facet.hits[0].value);
}

#[tokio::test]
async fn a_filter_narrows_the_counts_and_the_denominator_with_them() {
    let storage = library().await;
    let facet = storage
        .facet("docs", "topic", Some(&equals("year", json!(2025))), 10)
        .await
        .unwrap();
    assert_eq!(
        pairs(&facet),
        vec![("\"go\"".to_string(), 1), ("\"rust\"".to_string(), 1)]
    );
    assert_eq!(
        (facet.covered, facet.matched),
        (2, 2),
        "`matched` is the filter's result, not the table"
    );
}

#[tokio::test]
async fn what_the_facet_cannot_describe_is_reported_rather_than_dropped() {
    // The whole point of `covered`. Four records, two of which this field cannot be counted
    // for: one has no value, and one has an object, which the index records as presence only.
    // Both are missing from every bucket, and `hits` alone says a corpus of two.
    let storage = open().await;
    storage.create_table(table("docs", None)).await.unwrap();
    storage
        .insert_records(
            "docs",
            vec![
                record("counted", json!({"topic": "rust"})),
                record("also", json!({"topic": "go"})),
                record("absent", json!({"other": 1})),
                record("object", json!({"topic": {"name": "nested"}})),
            ],
        )
        .await
        .unwrap();

    let facet = storage.facet("docs", "topic", None, 10).await.unwrap();
    assert_eq!(facet.hits.len(), 2);
    assert_eq!(facet.matched, 4);
    assert_eq!(
        facet.covered, 2,
        "two records hold a value this facet can count, and the other two have to be visible \
         in the answer or the chart drawn from it silently omits half the corpus"
    );
}

#[tokio::test]
async fn an_array_counts_in_every_bucket_and_covers_its_record_once() {
    // A record tagged ["rag", "llm"] belongs in both buckets — that is what a facet over a
    // multi-valued field means — so the counts sum to more than the records. `covered` is a
    // set of records for exactly that reason; summing the counts would report 5 of 3.
    let storage = open().await;
    storage.create_table(table("docs", None)).await.unwrap();
    storage
        .insert_records(
            "docs",
            vec![
                record("a", json!({"tags": ["rag", "llm"]})),
                record("b", json!({"tags": ["rag"]})),
                record("c", json!({"tags": ["llm", "rust"]})),
            ],
        )
        .await
        .unwrap();

    let facet = storage.facet("docs", "tags", None, 10).await.unwrap();
    assert_eq!(
        pairs(&facet),
        vec![
            ("\"llm\"".to_string(), 2),
            ("\"rag\"".to_string(), 2),
            ("\"rust\"".to_string(), 1),
        ]
    );
    assert_eq!(facet.hits.iter().map(|h| h.count).sum::<usize>(), 5);
    assert_eq!((facet.covered, facet.matched), (3, 3));
}

#[tokio::test]
async fn the_page_is_the_top_n_and_says_how_many_there_were() {
    let storage = open().await;
    storage.create_table(table("docs", None)).await.unwrap();
    // Twenty values with counts 1..20, so the top three are unambiguous.
    let mut records = Vec::new();
    for value in 1..=20usize {
        for copy in 0..value {
            records.push(record(
                &format!("v{value}_{copy}"),
                json!({"bucket": format!("b{value:02}")}),
            ));
        }
    }
    let total = records.len();
    storage.insert_records("docs", records).await.unwrap();

    let facet = storage.facet("docs", "bucket", None, 3).await.unwrap();
    assert_eq!(
        pairs(&facet),
        vec![
            ("\"b20\"".to_string(), 20),
            ("\"b19\"".to_string(), 19),
            ("\"b18\"".to_string(), 18),
        ]
    );
    assert_eq!(
        facet.distinct_values, 20,
        "the page is three of twenty, and a caller has to be able to tell"
    );
    assert_eq!((facet.covered, facet.matched), (total, total));
}

#[tokio::test]
async fn ties_are_broken_the_same_way_every_time() {
    // A facet page that changes between identical calls is a bug report from someone
    // comparing two screenshots. Fifteen values all with one record, asked for five.
    let storage = open().await;
    storage.create_table(table("docs", None)).await.unwrap();
    let records: Vec<_> = (0..15)
        .map(|i| record(&format!("r{i}"), json!({"k": format!("v{i:02}")})))
        .collect();
    storage.insert_records("docs", records).await.unwrap();

    let first = pairs(&storage.facet("docs", "k", None, 5).await.unwrap());
    let second = pairs(&storage.facet("docs", "k", None, 5).await.unwrap());
    assert_eq!(first, second);
    assert_eq!(
        first,
        vec![
            ("\"v00\"".to_string(), 1),
            ("\"v01\"".to_string(), 1),
            ("\"v02\"".to_string(), 1),
            ("\"v03\"".to_string(), 1),
            ("\"v04\"".to_string(), 1),
        ],
        "all counts tie, so the value decides, ascending"
    );
}

#[tokio::test]
async fn a_field_the_table_does_not_index_is_refused_not_counted_as_zero() {
    // Same reasoning as the filter refusal: the index is the only thing that can answer, and
    // an unindexed field would answer every value with nothing — which reads as "the corpus
    // has none of these" rather than "this server cannot tell you".
    let storage = open().await;
    storage
        .create_table(table(
            "docs",
            Some(PayloadIndexPolicy {
                only: None,
                except: Some(vec!["topic".to_string()]),
            }),
        ))
        .await
        .unwrap();
    storage
        .insert_records("docs", vec![record("a", json!({"topic": "rust"}))])
        .await
        .unwrap();

    let err = storage.facet("docs", "topic", None, 10).await.unwrap_err();
    assert_eq!(err.status_code(), 400, "{err}");
    assert!(err.to_string().contains("topic"), "{err}");
}

#[tokio::test]
async fn a_field_nothing_has_is_an_empty_facet_and_not_an_error() {
    // Distinct from the case above, and the distinction is the point: "nobody wrote this
    // field" is an answer, and "this server does not index it" is not.
    let storage = library().await;
    let facet = storage
        .facet("docs", "nobody_wrote_this", None, 10)
        .await
        .unwrap();
    assert!(facet.hits.is_empty());
    assert_eq!((facet.covered, facet.matched), (0, 5));
}

#[tokio::test]
async fn a_nested_path_is_faceted_like_it_is_filtered() {
    let storage = open().await;
    storage.create_table(table("docs", None)).await.unwrap();
    storage
        .insert_records(
            "docs",
            vec![
                record("a", json!({"meta": {"lang": "en"}})),
                record("b", json!({"meta": {"lang": "en"}})),
                record("c", json!({"meta": {"lang": "zh"}})),
            ],
        )
        .await
        .unwrap();

    let facet = storage.facet("docs", "meta.lang", None, 10).await.unwrap();
    assert_eq!(
        pairs(&facet),
        vec![("\"en\"".to_string(), 2), ("\"zh\"".to_string(), 1)]
    );
}

#[tokio::test]
async fn a_facet_under_a_geo_filter() {
    // The two features landed together and the combination is the one a shop finder asks:
    // what kinds of place are near here.
    use dxmt_vectordb::types::record::{GeoPoint, GeoRadius};
    let storage = open().await;
    storage.create_table(table("docs", None)).await.unwrap();
    storage
        .insert_records(
            "docs",
            vec![
                record(
                    "gate",
                    json!({"at": {"lat": 52.5163, "lon": 13.3777}, "kind": "cafe"}),
                ),
                record(
                    "next",
                    json!({"at": {"lat": 52.5164, "lon": 13.3778}, "kind": "cafe"}),
                ),
                record(
                    "bank",
                    json!({"at": {"lat": 52.5165, "lon": 13.3779}, "kind": "bank"}),
                ),
                record(
                    "munich",
                    json!({"at": {"lat": 48.1351, "lon": 11.5820}, "kind": "cafe"}),
                ),
            ],
        )
        .await
        .unwrap();

    let near = Filter {
        must: Some(vec![Condition {
            key: "at".to_string(),
            operation: FilterOperation::GeoRadius(GeoRadius {
                center: GeoPoint {
                    lat: 52.5163,
                    lon: 13.3777,
                },
                radius_m: 1_000.0,
            }),
        }]),
        must_not: None,
        should: None,
    };
    let facet = storage
        .facet("docs", "kind", Some(&near), 10)
        .await
        .unwrap();
    assert_eq!(
        pairs(&facet),
        vec![("\"cafe\"".to_string(), 2), ("\"bank\"".to_string(), 1)]
    );
    assert_eq!((facet.covered, facet.matched), (3, 3));
}

#[tokio::test]
async fn the_counts_follow_an_update_and_a_delete() {
    let storage = library().await;

    storage
        .apply_update(
            "docs",
            &"a".to_string(),
            dxmt_vectordb::storage::RecordUpdate {
                payload: Some(json!({"topic": "go", "year": 2024})),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let facet = storage.facet("docs", "topic", None, 10).await.unwrap();
    assert_eq!(
        pairs(&facet),
        vec![
            ("\"go\"".to_string(), 2),
            ("\"rust\"".to_string(), 2),
            ("\"sql\"".to_string(), 1),
        ],
        "the old value was not dropped from its bucket"
    );

    storage
        .delete_records("docs", vec!["e".to_string()])
        .await
        .unwrap();
    let facet = storage.facet("docs", "topic", None, 10).await.unwrap();
    assert_eq!(
        pairs(&facet),
        vec![("\"go\"".to_string(), 2), ("\"rust\"".to_string(), 2)]
    );
    assert_eq!((facet.covered, facet.matched), (4, 4));
}
