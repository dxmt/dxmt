//! `search_mode: "exact"` is the ground truth the other four modes approximate.
//!
//! The four ef-scaling modes answer between 0.95 and 0.99 recall depending on the corpus, and
//! until this mode existed there was no way to say which number was right for *your* data: the
//! benchmark harness measures recall against datasets with published ground truth, and the
//! whole question is about the data someone actually loaded. Exhaustive search is the only
//! answer that has no recall in it at all.
//!
//! So these tests do not check that exact search is fast, or that it agrees with HNSW. They
//! check that it is *correct* — against a brute-force scan computed here in the test, which is
//! the only reference that does not beg the question — and that the things which make it safe
//! to expose (the row cap, the refusal for named spaces) actually hold.

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

const DIM: usize = 16;

fn storage_config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 1024,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

async fn open(wal: Option<std::path::PathBuf>) -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new_with_wal(&storage_config(), wal)
        .await
        .expect("failed to open storage")
}

fn table(name: &str) -> TableConfig {
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
        payload_index: None,
        embedding: None,
    }
}

/// Deterministic, and deliberately not clustered: a corpus where every vector is far from
/// every other makes any index look perfect, which would let a broken exact scan pass by
/// agreeing with a graph that was also right.
fn vector(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| {
            let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (d as u64 + 1);
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            (h & 0xFFFF) as f32 / 65_536.0
        })
        .collect()
}

fn record(i: usize) -> CreateRecordRequest {
    CreateRecordRequest {
        id: format!("r{i}"),
        vector: vector(i),
        sparse_vector: None,
        named_vectors: None,
        payload: Some(serde_json::json!({ "bucket": i % 10, "n": i })),
        text: None,
    }
}

fn query(vector: Vec<f32>, top_k: usize, mode: SearchMode) -> SearchQuery {
    SearchQuery {
        vector,
        top_k,
        limit: top_k,
        search_mode: mode,
        ..Default::default()
    }
}

/// Cosine distance, computed the slow obvious way, so the reference is not the thing under test.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (na * nb)
}

/// The true top-k over `rows`, by the same tiebreak the server uses (distance, then id).
fn ground_truth(q: &[f32], rows: usize, k: usize) -> Vec<String> {
    let mut scored: Vec<(String, f32)> = (0..rows)
        .map(|i| (format!("r{i}"), cosine_distance(q, &vector(i))))
        .collect();
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

#[tokio::test]
async fn exact_returns_the_true_nearest_neighbours() {
    const ROWS: usize = 3_000;
    let storage = open(None).await;
    storage.create_table(table("t")).await.unwrap();
    storage
        .insert_records("t", (0..ROWS).map(record).collect())
        .await
        .unwrap();

    // Not a stored vector: a query that happens to be a record is the easy case, and the one
    // a broken implementation is most likely to get right by accident.
    let q: Vec<f32> = vector(7)
        .iter()
        .zip(vector(19))
        .map(|(a, b)| (a + b) / 2.0)
        .collect();

    let hits = storage
        .search("t", query(q.clone(), 10, SearchMode::Exact))
        .await
        .unwrap();
    let got: Vec<String> = hits.hits.iter().map(|h| h.id.clone()).collect();

    assert_eq!(
        got,
        ground_truth(&q, ROWS, 10),
        "exact search returned something other than the true ten nearest, in the true order"
    );
    // Scores are similarities, descending, so the list must be non-increasing. An exact scan
    // that returned the right ids in the wrong order would still be wrong for anything that
    // reads rank.
    for pair in hits.hits.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "hits are not ordered: {:?} then {:?}",
            pair[0],
            pair[1]
        );
    }
}

#[tokio::test]
async fn a_record_is_its_own_nearest_neighbour_under_exact() {
    const ROWS: usize = 500;
    let storage = open(None).await;
    storage.create_table(table("t")).await.unwrap();
    storage
        .insert_records("t", (0..ROWS).map(record).collect())
        .await
        .unwrap();

    for id in [0usize, 137, 499] {
        let hits = storage
            .search("t", query(vector(id), 3, SearchMode::Exact))
            .await
            .unwrap();
        assert_eq!(
            hits.hits[0].id,
            format!("r{id}"),
            "a record used as its own query must come back first"
        );
        assert!(
            (hits.hits[0].score - 1.0).abs() < 1e-4,
            "cosine similarity with itself should be 1.0, got {}",
            hits.hits[0].score
        );
    }
}

#[tokio::test]
async fn exact_respects_a_filter_and_still_answers_exactly() {
    const ROWS: usize = 2_000;
    let storage = open(None).await;
    storage.create_table(table("t")).await.unwrap();
    storage
        .insert_records("t", (0..ROWS).map(record).collect())
        .await
        .unwrap();

    let q = vector(11);
    let mut with_filter = query(q.clone(), 5, SearchMode::Exact);
    with_filter.filter = Some(Filter {
        must: Some(vec![Condition {
            key: "bucket".to_string(),
            operation: FilterOperation::Equal(serde_json::json!(3)),
        }]),
        must_not: None,
        should: None,
    });

    let hits = storage.search("t", with_filter).await.unwrap();
    let got: Vec<String> = hits.hits.iter().map(|h| h.id.clone()).collect();

    // The same computation, restricted to the bucket the filter names.
    let mut scored: Vec<(String, f32)> = (0..ROWS)
        .filter(|i| i % 10 == 3)
        .map(|i| (format!("r{i}"), cosine_distance(&q, &vector(i))))
        .collect();
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let expected: Vec<String> = scored.into_iter().take(5).map(|(id, _)| id).collect();

    assert_eq!(got, expected, "a filtered exact search is still exact");
    assert!(
        got.iter()
            .all(|id| id[1..].parse::<usize>().unwrap() % 10 == 3),
        "the filter was not applied: {got:?}"
    );
}

#[tokio::test]
async fn the_approximate_modes_are_measured_against_exact_rather_than_assumed() {
    // The reason the mode exists, exercised the way the dashboard will exercise it: run the
    // same query twice and compare. This asserts the comparison is *possible* and sane, not
    // that HNSW hits a particular recall -- a threshold here would be measuring the graph's
    // parameters, which `make bench-recall` is for.
    const ROWS: usize = 2_000;
    let storage = open(None).await;
    storage.create_table(table("t")).await.unwrap();
    storage
        .insert_records("t", (0..ROWS).map(record).collect())
        .await
        .unwrap();

    let mut total = 0.0;
    let probes = 20;
    for i in 0..probes {
        let q = vector(i * 97);
        let exact = storage
            .search("t", query(q.clone(), 10, SearchMode::Exact))
            .await
            .unwrap();
        let ann = storage
            .search("t", query(q, 10, SearchMode::Normal))
            .await
            .unwrap();

        let truth: std::collections::HashSet<&str> =
            exact.hits.iter().map(|h| h.id.as_str()).collect();
        let found = ann
            .hits
            .iter()
            .filter(|h| truth.contains(h.id.as_str()))
            .count();
        total += found as f32 / exact.hits.len() as f32;
    }
    let recall = total / probes as f32;

    assert!(
        recall > 0.5,
        "recall against exact came out at {recall}, which is low enough that the comparison \
         itself is probably broken rather than the index being bad"
    );
    assert!(
        recall <= 1.0,
        "recall of {recall} is impossible; the two runs are not comparing the same thing"
    );
}

#[tokio::test]
async fn exact_refuses_rather_than_scanning_past_its_cap() {
    // The cap is what makes the mode safe to put behind a button: at 100M rows the same
    // request is minutes of full-file reads, and a caller cannot tell that from a hang.
    const ROWS: usize = 300;
    std::env::set_var("DXMT_EXACT_SCAN_MAX", "100");

    let storage = open(None).await;
    storage.create_table(table("t")).await.unwrap();
    storage
        .insert_records("t", (0..ROWS).map(record).collect())
        .await
        .unwrap();

    let err = storage
        .search("t", query(vector(1), 5, SearchMode::Exact))
        .await
        .expect_err("300 rows against a 100-row cap must be refused");
    let message = err.to_string();
    assert!(
        message.contains("300") && message.contains("100"),
        "the refusal should say how many rows it would have read and what the limit is: {message}"
    );

    // A filter that brings the candidate set under the cap makes the same query answerable --
    // the cap is measured against what will be read, not against the table.
    let mut narrowed = query(vector(1), 5, SearchMode::Exact);
    narrowed.filter = Some(Filter {
        must: Some(vec![Condition {
            key: "bucket".to_string(),
            operation: FilterOperation::Equal(serde_json::json!(1)),
        }]),
        must_not: None,
        should: None,
    });
    let hits = storage.search("t", narrowed).await.expect(
        "30 matching rows is under the cap; a selective filter is exactly when exact is cheap",
    );
    assert_eq!(hits.hits.len(), 5);

    std::env::remove_var("DXMT_EXACT_SCAN_MAX");
}

#[tokio::test]
async fn exact_is_exact_with_the_vectors_in_a_row_file() {
    // The production shape, and the one with the real work in it: with a WAL directory the
    // vectors live in `wal/vectors/<table>.f32` and the records carry none, so the scan reads
    // rows from a mapped file in sorted chunks rather than walking an in-process map. Same
    // answers, different code — which is precisely the pair that `tests/api/conftest.py`
    // exists to keep honest for the HTTP layer.
    const ROWS: usize = 2_000;
    let tmp = tempfile::TempDir::new().unwrap();
    let wal = tmp.path().join("wal");

    let storage = open(Some(wal.clone())).await;
    storage.create_table(table("t")).await.unwrap();
    storage
        .insert_records("t", (0..ROWS).map(record).collect())
        .await
        .unwrap();

    // Without this the test would pass while measuring the in-memory path again.
    assert!(
        wal.join("vectors").join("t.f32").exists(),
        "the vectors should be in a row file; this test is about reading them from one"
    );

    let q: Vec<f32> = vector(3)
        .iter()
        .zip(vector(41))
        .map(|(a, b)| (a + b) / 2.0)
        .collect();
    let hits = storage
        .search("t", query(q.clone(), 10, SearchMode::Exact))
        .await
        .unwrap();
    let got: Vec<String> = hits.hits.iter().map(|h| h.id.clone()).collect();

    assert_eq!(
        got,
        ground_truth(&q, ROWS, 10),
        "the row-file scan disagrees with a brute-force pass computed in the test"
    );
}

#[tokio::test]
async fn exact_says_what_it_cannot_do_for_a_named_vector_space() {
    // A named space has its own file and its own dimension. Scanning the default space and
    // calling the answer exact would be worse than refusing, because the caller asked for the
    // one mode whose whole value is that its answer has nothing approximate in it.
    let storage = open(None).await;
    storage.create_table(table("t")).await.unwrap();
    storage
        .insert_records("t", (0..50).map(record).collect())
        .await
        .unwrap();

    let mut q = query(vector(1), 5, SearchMode::Exact);
    q.vector_name = Some("title".to_string());
    let err = storage.search("t", q).await.expect_err("should refuse");
    let message = err.to_string();
    assert!(
        message.contains("named vector"),
        "the refusal should name the reason: {message}"
    );
    assert!(
        message.contains("accurate"),
        "and say what to use instead: {message}"
    );
}

/// A vector pointing away from the query scores its real, negative cosine similarity.
///
/// The distance-to-score conversion clamped the cosine *distance* to [0, 1], so every
/// similarity below zero was reported as 0.0 — the exact opposite of the query scored the same
/// as a vector at right angles to it, and anything re-ranking by score lost their order. The
/// stateful walk caught it comparing exact search against a brute-force cosine.
#[tokio::test]
async fn a_negative_similarity_is_reported_as_it_is() {
    let storage = open(None).await;
    storage.create_table(table("neg")).await.unwrap();
    let q: Vec<f32> = (0..DIM).map(|d| if d == 0 { 1.0 } else { 0.0 }).collect();
    let at = |cos: f32| -> Vec<f32> {
        // A unit vector at a chosen angle from `q`, in the plane of the first two axes.
        let sin = (1.0 - cos * cos).max(0.0).sqrt();
        (0..DIM)
            .map(|d| match d {
                0 => cos,
                1 => sin,
                _ => 0.0,
            })
            .collect()
    };
    let wanted = [
        ("same", 1.0),
        ("right_angle", 0.0),
        ("away", -0.5),
        ("opposite", -1.0),
    ];
    storage
        .insert_records(
            "neg",
            wanted
                .iter()
                .map(|(id, cos)| CreateRecordRequest {
                    id: id.to_string(),
                    vector: at(*cos),
                    sparse_vector: None,
                    named_vectors: None,
                    payload: None,
                    text: None,
                })
                .collect(),
        )
        .await
        .unwrap();

    let hits = storage
        .search("neg", query(q, 4, SearchMode::Exact))
        .await
        .unwrap()
        .hits;
    let got: Vec<(String, f32)> = hits.into_iter().map(|h| (h.id, h.score)).collect();
    assert_eq!(
        got.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
        ["same", "right_angle", "away", "opposite"],
        "{got:?}"
    );
    for ((_, score), (id, cos)) in got.iter().zip(wanted) {
        assert!(
            (score - cos).abs() < 1e-4,
            "{id}: score {score}, cosine {cos}"
        );
    }
}
