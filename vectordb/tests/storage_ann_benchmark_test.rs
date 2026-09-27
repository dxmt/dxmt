#![cfg(feature = "benchmarks")]

//! Storage-level ANN benchmark on a reproducible 10K/64D corpus.
//!
//! The existing index-level tests prove HNSW itself, while this test exercises
//! the production `IntelligentIndexedStorage` path: table creation, batched
//! inserts, automatic index selection, query assembly, and recall against an
//! exact brute-force reference.  It is ignored by default because it is a
//! performance test rather than a unit test.  For an optimized run use the
//! unwind-compatible profile:
//! `cargo test --profile release-unwind --features benchmarks
//! --test storage_ann_benchmark_test -- --ignored --nocapture`.
//! Set `DXMT_ANN_N`, `DXMT_ANN_DIM`, and `DXMT_ANN_NUM_QUERIES` to scale the
//! same test to larger corpora.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::{IndexType, TableConfig},
        vector::DistanceMetric,
    },
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Instant;

const DEFAULT_N: usize = 10_000;
const DEFAULT_DIM: usize = 64;
const DEFAULT_TOP_K: usize = 10;
const DEFAULT_NUM_QUERIES: usize = 100;

/// Which index-selection path to measure.
///
/// `DXMT_MEMORY_OPTIMIZATION` defaults to true, so a table created without an explicit setting
/// goes through `select_memory_optimized_index` (m=16, ef_construction=200, ef_search=50).
/// Setting this to `false` measures `select_optimal_index` instead (m=8, ef_construction=100,
/// ef_search=100 below 100K). The two paths produce materially different recall, so a benchmark
/// that silently pins one of them is measuring something users may never run.
fn env_memory_optimization() -> Option<bool> {
    match std::env::var("DXMT_ANN_MEMORY_OPT").ok().as_deref() {
        Some("true") | Some("1") => Some(true),
        Some("false") | Some("0") => Some(false),
        // Unset: inherit the server default, which is what a user gets.
        _ => None,
    }
}

/// Search effort to measure. `Normal` uses the index's configured `ef_search`; `Accurate`
/// multiplies it by four. Which one is in play changes recall far more than anything else here,
/// so it is reported alongside the number rather than left implicit.
fn env_search_mode() -> SearchMode {
    match std::env::var("DXMT_ANN_SEARCH_MODE").ok().as_deref() {
        Some("fast") => SearchMode::Fast,
        Some("accurate") => SearchMode::Accurate,
        Some("auto") => SearchMode::Auto,
        _ => SearchMode::Normal,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn corpus(seed: u64, count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut vector: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for value in &mut vector {
                *value /= norm;
            }
            vector
        })
        .collect()
}

fn exact_top_k(query: &[f32], vectors: &[Vec<f32>], top_k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            let similarity = query.iter().zip(vector).map(|(a, b)| a * b).sum::<f32>();
            (index, similarity)
        })
        .collect();
    scores.sort_by(|(index_a, score_a), (index_b, score_b)| {
        score_b
            .partial_cmp(score_a)
            .unwrap()
            .then_with(|| index_a.cmp(index_b))
    });
    scores
        .into_iter()
        .take(top_k)
        .map(|(index, _)| index)
        .collect()
}

fn percentile(sorted_samples: &[f64], quantile: f64) -> f64 {
    let index = ((sorted_samples.len() - 1) as f64 * quantile).round() as usize;
    sorted_samples[index]
}

#[tokio::test]
#[ignore = "10K ANN performance benchmark; run with --ignored"]
async fn test_storage_ann_10k_recall_and_latency() {
    let n = env_usize("DXMT_ANN_N", DEFAULT_N);
    let dimension = env_usize("DXMT_ANN_DIM", DEFAULT_DIM);
    let top_k = env_usize("DXMT_ANN_TOP_K", DEFAULT_TOP_K).min(n);
    let query_count = env_usize("DXMT_ANN_NUM_QUERIES", DEFAULT_NUM_QUERIES).min(n);

    let storage = IntelligentIndexedStorage::new(&StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 512,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    })
    .await
    .unwrap();

    storage
        .create_table(TableConfig {
            name: "ann_10k".to_string(),
            dimension,
            distance_metric: DistanceMetric::Cosine,
            description: Some("storage ANN benchmark".to_string()),
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: env_memory_optimization(),
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

    let vectors = corpus(0x4458_4D54_104B_u64, n, dimension);
    let records: Vec<CreateRecordRequest> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| CreateRecordRequest {
            id: format!("v{index:05}"),
            vector: vector.clone(),
            sparse_vector: None,
            payload: None,
            named_vectors: None,
            text: None,
        })
        .collect();

    let build_start = Instant::now();
    storage.insert_records("ann_10k", records).await.unwrap();
    let build_ms = build_start.elapsed().as_secs_f64() * 1_000.0;

    let table = storage.get_table("ann_10k").await.unwrap();
    assert_eq!(table.record_count as usize, n);
    // Not pinned to one algorithm: which one the selector picks is part of what this measures.
    // It must not have fallen back to a linear scan, though, or the numbers below are a
    // brute-force baseline wearing an ANN benchmark's name.
    assert_ne!(
        table.index_type,
        IndexType::BruteForce,
        "adaptive selection fell back to brute force at n={n}; the recall below is meaningless"
    );
    let index_type = table.index_type;

    let search_mode = env_search_mode();
    let mut latencies_ms = Vec::with_capacity(query_count);
    let mut recall_sum = 0.0_f64;
    let query_stride = (n / query_count).max(1);
    for query_index in (0..n).step_by(query_stride).take(query_count) {
        let query_vector = &vectors[query_index];
        let ground_truth = exact_top_k(query_vector, &vectors, top_k);
        let start = Instant::now();
        let result = storage
            .search(
                "ann_10k",
                SearchQuery {
                    vector: query_vector.clone(),
                    top_k,
                    limit: top_k,
                    with_payload: false,
                    with_vector: false,
                    search_mode,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        latencies_ms.push(start.elapsed().as_secs_f64() * 1_000.0);

        let hits: std::collections::HashSet<usize> = result
            .hits
            .iter()
            .filter_map(|hit| hit.id.strip_prefix('v')?.parse::<usize>().ok())
            .collect();
        let matches = ground_truth
            .iter()
            .filter(|index| hits.contains(index))
            .count();
        recall_sum += matches as f64 / top_k as f64;
    }

    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let elapsed_s = latencies_ms.iter().sum::<f64>() / 1_000.0;
    let recall = recall_sum / query_count as f64;
    let qps = query_count as f64 / elapsed_s;
    println!(
        "storage ANN: n={n}, dim={dimension}, index={index_type:?}, \
         memory_opt={:?}, mode={search_mode:?}, build={build_ms:.1} ms, recall@{top_k}={recall:.4}, \
         p50={:.3} ms, p95={:.3} ms, p99={:.3} ms, qps={qps:.1}",
        env_memory_optimization(),
        percentile(&latencies_ms, 0.50),
        percentile(&latencies_ms, 0.95),
        percentile(&latencies_ms, 0.99),
    );
    // 0.90 is the floor for the default corpus (10K x 64), where the measured value is 1.000.
    // Larger or higher-dimensional overrides do not all clear it today — 10K x 768 measures
    // 0.900 and 50K x 128 measures 0.886 with the default `ef_search` of 128, on random unit
    // vectors, which is the hardest synthetic distribution there is. That is a real gap and is
    // tracked in ROADMAP §7 W9; the assertion is left strict so it stays visible rather than
    // being tuned down to whatever the engine currently does.
    assert!(
        recall >= 0.90,
        "storage recall@{top_k} too low: {recall:.4}"
    );
}

/// The traversal budget must actually reach the query.
///
/// `ef_search` is chosen by the index selector and reaches `HnswConfig` at build time, but the
/// search path used to compute its budget from a hardcoded `let default_ef = 100`, so the
/// selector's value was inert and `SearchMode` scaled a constant. Recall as a function of search
/// effort is the observable that proves the budget is live; it only separates on a corpus hard
/// enough to matter, which is why this lives beside the benchmark rather than in the unit tests.
#[tokio::test]
#[ignore = "ANN quality check; run with --ignored under --profile release-unwind"]
async fn test_search_effort_is_live() {
    let n = env_usize("DXMT_ANN_N", 10_000);
    let dimension = env_usize("DXMT_ANN_DIM", 384);
    let top_k = 10;
    let query_count = 50;

    let storage = IntelligentIndexedStorage::new(&StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 512,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    })
    .await
    .unwrap();

    storage
        .create_table(TableConfig {
            name: "effort".to_string(),
            dimension,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: Some(true),
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

    let vectors = corpus(0x4458_4D54_E110_u64, n, dimension);
    storage
        .insert_records(
            "effort",
            vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| CreateRecordRequest {
                    id: format!("v{index:05}"),
                    vector: vector.clone(),
                    sparse_vector: None,
                    payload: None,
                    named_vectors: None,
                    text: None,
                })
                .collect(),
        )
        .await
        .unwrap();

    let configured = storage
        .describe_index("effort")
        .await
        .and_then(|algorithm| algorithm.configured_ef_search());

    let mut recalls = Vec::new();
    for mode in [SearchMode::Fast, SearchMode::Normal, SearchMode::Accurate] {
        let mut sum = 0.0_f64;
        let stride = (n / query_count).max(1);
        for query_index in (0..n).step_by(stride).take(query_count) {
            let truth = exact_top_k(&vectors[query_index], &vectors, top_k);
            let result = storage
                .search(
                    "effort",
                    SearchQuery {
                        vector: vectors[query_index].clone(),
                        top_k,
                        limit: top_k,
                        with_payload: false,
                        with_vector: false,
                        search_mode: mode,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            let hits: std::collections::HashSet<usize> = result
                .hits
                .iter()
                .filter_map(|hit| hit.id.strip_prefix('v')?.parse::<usize>().ok())
                .collect();
            sum += truth.iter().filter(|index| hits.contains(index)).count() as f64 / top_k as f64;
        }
        recalls.push(sum / query_count as f64);
    }

    println!(
        "search effort at n={n}, dim={dimension}, configured ef_search={configured:?}: \
         fast={:.4} normal={:.4} accurate={:.4}",
        recalls[0], recalls[1], recalls[2]
    );

    assert!(
        recalls[1] >= recalls[0] && recalls[2] >= recalls[1],
        "recall fell as search effort rose: {recalls:?}"
    );
    assert!(
        recalls[2] > recalls[0] + 0.01,
        "search effort changed nothing ({recalls:?}) — the configured ef_search is not reaching \
         the query path"
    );
}
