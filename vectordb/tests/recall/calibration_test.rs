//! Recall calibration: a table measures itself against its own exact scan and raises the
//! `normal` search budget until it meets the target.
//!
//! The table here is created with `ef_search: 32`, a budget at which HNSW on 20,000
//! 64-dimension vectors of uniform noise misses much of a top-10 -- the shape of a corpus that needs more
//! budget than it was given, which is the case calibration exists for. The claims checked
//! are the ones a caller relies on: the curve was measured at increasing budgets and the
//! chosen one meets the target on the sample; and after calibration an ordinary `normal`
//! search on *held-out* queries -- not the rows the calibration sampled -- has materially
//! higher recall against the exact scan than before.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::{HnswParams, TableAlterRequest, TableConfig},
        vector::DistanceMetric,
    },
};

const DIM: usize = 64;
const ROWS: usize = 20_000;
const K: usize = 10;
const QUERIES: usize = 40;

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

fn table(name: &str, ef_search: usize) -> TableConfig {
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
        hnsw: Some(HnswParams {
            m: None,
            ef_construction: None,
            ef_search: Some(ef_search),
        }),
        quantization: None,
        payload_index: None,
        embedding: None,
    }
}

/// Deterministic pseudo-random vectors: uniform noise in 64 dimensions is a corpus with no
/// structure for the graph to exploit and distances that crowd together, which is what makes
/// a small `ef` visibly short (hnswlib on the same shape: recall@10 0.12 at ef 10, 0.69 at
/// 128). A hand-rolled hash sat here first and produced vectors the graph found at ef 11 with
/// recall 0.995 -- structured enough to make the test prove nothing.
fn vector(seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (0..DIM).map(|_| rng.gen::<f32>() - 0.5).collect()
}

async fn load(storage: &IntelligentIndexedStorage, name: &str) {
    for chunk in (0..ROWS).collect::<Vec<_>>().chunks(2_000) {
        let batch = chunk
            .iter()
            .map(|&i| CreateRecordRequest {
                id: format!("r{i}"),
                vector: vector(i as u64),
                sparse_vector: None,
                named_vectors: None,
                payload: None,
                text: None,
            })
            .collect();
        storage.insert_records(name, batch).await.unwrap();
    }
}

async fn ids(
    storage: &IntelligentIndexedStorage,
    name: &str,
    q: &[f32],
    mode: SearchMode,
) -> Vec<String> {
    storage
        .search(
            name,
            SearchQuery {
                vector: q.to_vec(),
                top_k: K,
                limit: K,
                search_mode: mode,
                with_payload: false,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits
        .into_iter()
        .map(|h| h.id)
        .collect()
}

/// recall@10 of `normal` against `exact` on held-out queries (seeds the corpus never used).
async fn held_out_recall(storage: &IntelligentIndexedStorage, name: &str) -> f32 {
    let mut hits = 0usize;
    for q in 0..QUERIES {
        let query = vector(1_000_000 + q as u64);
        let truth = ids(storage, name, &query, SearchMode::Exact).await;
        let got = ids(storage, name, &query, SearchMode::Normal).await;
        hits += got.iter().filter(|g| truth.contains(g)).count();
    }
    hits as f32 / (QUERIES * K) as f32
}

#[tokio::test]
async fn calibration_raises_a_short_budget_until_the_target_is_met() {
    let storage = IntelligentIndexedStorage::new(&storage_config())
        .await
        .unwrap();
    storage.create_table(table("cal", 32)).await.unwrap();
    load(&storage, "cal").await;

    let before = held_out_recall(&storage, "cal").await;
    assert!(
        before < 0.8,
        "ef_search 32 was supposed to be visibly short; held-out recall@10 is {before:.3}"
    );

    let cal = storage
        .calibrate_recall("cal", true)
        .await
        .unwrap()
        .expect("a 20,000-row HNSW table has something to calibrate");

    assert_eq!(cal.target, 0.95, "the server default");
    assert!(
        cal.curve.len() >= 2,
        "it should have had to try more than one budget: {cal:?}"
    );
    assert_eq!(
        cal.curve[0].ef_search, 32,
        "the curve starts at the configured budget"
    );
    for w in cal.curve.windows(2) {
        assert_eq!(w[1].ef_search, w[0].ef_search * 2, "budgets double");
        assert!(
            w[1].recall >= w[0].recall - 0.02,
            "recall does not fall as ef rises: {cal:?}"
        );
    }
    assert!(cal.met, "256 (8x) should meet 0.95 on 20,000 rows: {cal:?}");
    assert!(cal.measured >= 0.95);
    assert_eq!(cal.ef_search, cal.curve.last().unwrap().ef_search);
    assert_eq!(cal.sample, 50);
    assert_eq!(cal.at_rows, ROWS as u64);

    // The calibration is what `normal` now searches with, and it shows on queries the
    // calibration never saw.
    let after = held_out_recall(&storage, "cal").await;
    assert!(
        after >= 0.90 && after > before + 0.15,
        "normal after calibration: {after:.3} (before {before:.3}); calibration {cal:?}"
    );
    let stored = storage.get_table("cal").await.unwrap();
    assert_eq!(
        stored.recall.as_ref().map(|c| c.ef_search),
        Some(cal.ef_search)
    );
}

#[tokio::test]
async fn a_table_can_opt_out_and_a_calibration_is_redone_only_when_the_table_doubles() {
    let storage = IntelligentIndexedStorage::new(&storage_config())
        .await
        .unwrap();
    storage.create_table(table("opt", 32)).await.unwrap();
    load(&storage, "opt").await;

    // Opted out: nothing is measured, whatever `force` says.
    storage
        .alter_table(
            "opt",
            TableAlterRequest {
                recall_target: Some(0.0),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(storage
        .calibrate_recall("opt", true)
        .await
        .unwrap()
        .is_none());

    // Back on, with a target of its own. 20,000 rows is the sweep's size floor, so the
    // unforced path (what the sweep runs) measures it; a second unforced pass at the same
    // size is a no-op because the table has not doubled since.
    storage
        .alter_table(
            "opt",
            TableAlterRequest {
                recall_target: Some(0.9),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let cal = storage
        .calibrate_recall("opt", false)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cal.target, 0.9);
    assert!(storage
        .calibrate_recall("opt", false)
        .await
        .unwrap()
        .is_none());

    // Changing the target discards the calibration, so `normal` is back on the configured
    // budget until the next measurement.
    storage
        .alter_table(
            "opt",
            TableAlterRequest {
                recall_target: Some(0.99),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(storage.get_table("opt").await.unwrap().recall.is_none());
}

#[tokio::test]
async fn a_table_whose_budget_already_meets_the_target_keeps_it() {
    let storage = IntelligentIndexedStorage::new(&storage_config())
        .await
        .unwrap();
    storage.create_table(table("fine", 1024)).await.unwrap();
    load(&storage, "fine").await;
    let cal = storage
        .calibrate_recall("fine", true)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cal.curve.len(), 1, "met at the first budget: {cal:?}");
    assert_eq!(cal.ef_search, 1024);
    assert!(cal.met);
}
