//! Input a vector database has no business refusing.
//!
//! Both cases here were found by loading somebody else's example rather than by review, and
//! both had the same shape: a validation rule wider than its own justification, refusing
//! something ordinary, with every synthetic test in the tree passing because a generated
//! corpus never contains the awkward case.
//!
//! The lesson is not the two rules. It is that a rule which *rejects* is only ever exercised
//! by input nobody generates, so the tests that would catch an over-strict one have to be
//! written deliberately. That is what this file is.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchQuery},
        table::TableConfig,
        vector::{DistanceMetric, VectorValidator},
    },
};

fn config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 128,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

fn table(name: &str, metric: DistanceMetric, dimension: usize) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension,
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

/// The origin is a point. A euclidean table has to take it.
///
/// GIST-960-euclidean, the dataset `ann-benchmarks` publishes ground truth for and the one
/// Epsilla's own example script loads, holds **ten all-zero descriptors** among its million
/// training vectors — 7030 is the first. Loading it into a euclidean table stopped there with
/// "Zero vectors are not allowed". `ValidationConfig::allow_zero_vectors` existed for exactly
/// this and no caller had ever set it.
#[test]
fn a_zero_vector_is_a_point_under_euclidean_and_a_mistake_under_cosine() {
    let zero = vec![0.0f32; 8];
    for metric in [DistanceMetric::Euclidean, DistanceMetric::Manhattan] {
        assert!(
            VectorValidator::validate_for_metric(&zero, 8, metric).is_ok(),
            "{metric:?} refused the origin, which is an ordinary point under it"
        );
    }
    for metric in [DistanceMetric::Cosine, DistanceMetric::Dot] {
        let err = VectorValidator::validate_for_metric(&zero, 8, metric).unwrap_err();
        assert!(
            err.to_string().contains("no direction"),
            "{metric:?} should refuse it and say why: {err}"
        );
    }
}

#[tokio::test]
async fn a_euclidean_table_stores_and_finds_the_origin() {
    // Not just accepted by the validator: stored, indexed, and returned by a search, which is
    // the part a validator test cannot show.
    let storage = IntelligentIndexedStorage::new(&config()).await.unwrap();
    storage
        .create_table(table("gistish", DistanceMetric::Euclidean, 4))
        .await
        .unwrap();
    storage
        .insert_records(
            "gistish",
            vec![
                CreateRecordRequest {
                    id: "origin".to_string(),
                    vector: vec![0.0, 0.0, 0.0, 0.0],
                    text: None,
                    sparse_vector: None,
                    named_vectors: None,
                    payload: None,
                },
                CreateRecordRequest {
                    id: "far".to_string(),
                    vector: vec![9.0, 9.0, 9.0, 9.0],
                    text: None,
                    sparse_vector: None,
                    named_vectors: None,
                    payload: None,
                },
            ],
        )
        .await
        .unwrap();

    let hits = storage
        .search(
            "gistish",
            SearchQuery {
                vector: vec![0.1, 0.1, 0.1, 0.1],
                top_k: 2,
                limit: 2,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits;
    assert_eq!(
        hits.first().map(|h| h.id.as_str()),
        Some("origin"),
        "the origin is the nearer of the two and has to come back first: {hits:?}"
    );
}

/// A table's name becomes a file name, and that is the whole of the rule.
///
/// `my-collection` is how most people name one of these and was refused; so were `2024_logs`
/// and `_staging`. A hyphen and a leading digit are ordinary in a file name on every
/// filesystem, and the justification never covered them.
#[test]
fn table_names_people_actually_use_are_accepted() {
    for name in [
        "my-collection",
        "2024_logs",
        "_staging",
        "docs-v2",
        "a",
        "A1-b2_c3",
    ] {
        assert!(
            dxmt_vectordb::utils::validate_table_name(name).is_ok(),
            "'{name}' was refused, and a file can be called that"
        );
    }
}

/// And the rule still keeps out what it exists to keep out.
#[test]
fn a_table_name_still_cannot_be_a_path_or_a_flag() {
    for (name, why) in [
        ("../escape", "a relative path"),
        ("a/b", "a path separator"),
        ("a\\b", "a windows path separator"),
        (".", "the current directory"),
        ("..", "the parent directory"),
        ("-rf", "a command-line flag"),
        ("table name", "a space"),
        ("表名", "non-ascii, which a filesystem may normalise"),
        ("", "empty"),
    ] {
        assert!(
            dxmt_vectordb::utils::validate_table_name(name).is_err(),
            "'{name}' was accepted and it is {why}"
        );
    }
    let long = "a".repeat(64);
    assert!(dxmt_vectordb::utils::validate_table_name(&long).is_err());
}

/// A name with a hyphen has to survive the whole round trip, not just the name check: it goes
/// into a URL path, and it goes onto the filesystem.
#[tokio::test]
async fn a_hyphenated_table_works_end_to_end() {
    let storage = IntelligentIndexedStorage::new(&config()).await.unwrap();
    storage
        .create_table(table("my-collection", DistanceMetric::Cosine, 4))
        .await
        .unwrap();
    storage
        .insert_records(
            "my-collection",
            vec![CreateRecordRequest {
                id: "a".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                text: None,
                sparse_vector: None,
                named_vectors: None,
                payload: None,
            }],
        )
        .await
        .unwrap();
    let hits = storage
        .search(
            "my-collection",
            SearchQuery {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                top_k: 1,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits;
    assert_eq!(hits.len(), 1);
    storage.delete_table("my-collection").await.unwrap();
}
