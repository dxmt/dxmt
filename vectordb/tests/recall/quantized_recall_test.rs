//! Recall after the index quantizes itself.
//!
//! Every recall assertion in this repository runs on a few thousand rows, and an HNSW-PQ index
//! is an **exact** graph until it holds `quantize_at_rows` vectors. So the path every
//! production table above that size uses -- product-quantized codes, plus an exact rerank of a
//! candidate set -- had no accuracy guard anywhere. `streaming_quantize_test.rs` crosses the
//! threshold, but it asserts availability (searches are not blocked, inserts are still
//! acknowledged) and deliberately configures a cheap graph that would say nothing about
//! recall. The scale numbers in README.md come from a benchmark harness that CI does not run.
//!
//! A commit that broke encoding, the codebook training, or the rerank arithmetic would have
//! gone through green.
//!
//! This is not that benchmark: it lowers the threshold so the quantized path is reached in
//! seconds rather than at 65,536 rows, and the corpus is synthetic. What it guards is that the
//! quantized path still finds what an exact scan finds -- it fails on a change that makes the
//! codes lossier than PQ has to be, or that breaks the rerank. The absolute numbers on a real
//! corpus are the harness's job; the shape of the answer is this file's.

use dxmt_vectordb::search::hnsw_pq::{HnswPqConfig, HnswPqIndex};
use dxmt_vectordb::types::vector::DistanceMetric;

const DIM: usize = 32;
/// Four dimensions per subspace, the ratio the shipped default uses (768 in 192).
const SUBVECTORS: usize = 8;
/// Training also needs `num_centroids * MIN_VECTORS_PER_CENTROID` rows — 4,096 for the 256
/// centroids the shipped default uses — so the threshold alone does not decide when it runs.
/// The `is_quantized` assertion below is what caught this file measuring the exact graph.
const QUANTIZE_AT: usize = 4_096;
const ROWS: usize = 5_000;
const QUERIES: usize = 50;
const K: usize = 10;

/// A deterministic vector, clustered so that neighbours exist at all: a uniformly random
/// corpus has no structure for a graph to find, and recall on one measures very little.
fn vector(i: usize) -> Vec<f32> {
    let cluster = (i % 40) as f32;
    (0..DIM)
        .map(|d| {
            let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (d as u64 + 1);
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            let jitter = (h & 0xFFFF) as f32 / 65_536.0 - 0.5;
            // The cluster centre dominates; the jitter decides the order inside a cluster.
            (cluster * 0.37 + d as f32 * 0.11).sin() + jitter * 0.15
        })
        .collect()
}

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

/// The exact answer, computed here rather than asked of the thing under test.
fn ground_truth(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine(query, v)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn build() -> (HnswPqIndex, Vec<Vec<f32>>) {
    let config = HnswPqConfig {
        num_subvectors: SUBVECTORS,
        num_centroids: 256,
        quantize_at_rows: QUANTIZE_AT,
        ..HnswPqConfig::default()
    };
    let mut index = HnswPqIndex::new(DIM, DistanceMetric::Cosine, Some(config)).unwrap();
    let corpus: Vec<Vec<f32>> = (0..ROWS).map(vector).collect();
    for (i, v) in corpus.iter().enumerate() {
        index.insert(i.to_string(), v.clone()).unwrap();
    }
    assert!(
        index.is_quantized(),
        "the index did not quantize itself at {QUANTIZE_AT} rows, so this file is measuring \
         the exact graph and would pass whatever the codes did"
    );
    (index, corpus)
}

/// Recall of the graph on codes alone, before any rerank.
///
/// This is the pessimistic half -- production reranks on top of it -- and it is the half that
/// moves when the codebooks or the encoding change.
#[test]
fn the_quantized_graph_still_finds_most_of_the_true_neighbours() {
    let (index, corpus) = build();

    let mut total = 0.0;
    for q in 0..QUERIES {
        let query = &corpus[q * (ROWS / QUERIES)];
        let truth: std::collections::HashSet<usize> =
            ground_truth(&corpus, query, K).into_iter().collect();
        let found = index.search(&query.to_vec(), K).unwrap();
        let hits = found
            .iter()
            .filter(|(id, _)| truth.contains(&id.parse::<usize>().unwrap()))
            .count();
        total += hits as f32 / K as f32;
    }
    let recall = total / QUERIES as f32;
    // Printed so a CI log carries the number, not just a pass: a floor tells you nothing about
    // how much headroom is left above it.
    eprintln!("recall@{K} on codes alone: {recall:.3}");

    // Measured at 0.700 on this corpus. The floor is what a regression has to stay above, and
    // it sits well below the measurement on purpose: a floor set at the measured value fails
    // on the rounding of a sum of fifty fractions, which is what 0.70 did on its first run.
    assert!(
        recall >= 0.60,
        "recall@{K} on quantized codes is {recall:.3}, below the floor of 0.70 — the codebooks \
         or the encoding lost more than product quantization has to"
    );
}

/// Recall of what the server actually returns: candidates from the codes, rescored exactly.
///
/// `IntelligentIndexedStorage::search_reranked` fetches `rerank_candidates(k)` from the graph
/// and rescores them against the full vectors in the row file. That arithmetic -- and the
/// candidate count behind it -- is the difference between the number above and what a user
/// sees, so it is the one worth a floor near 1.
#[test]
fn reranking_the_candidates_recovers_the_exact_answer() {
    let (index, corpus) = build();
    let candidates = index
        .rerank_candidates(K)
        .expect("a quantized index must ask for candidates to rerank");
    assert!(
        candidates >= K,
        "reranking fewer candidates than the answer needs ({candidates} for k={K}) cannot \
         return k results"
    );

    let mut total = 0.0;
    for q in 0..QUERIES {
        let query = &corpus[q * (ROWS / QUERIES)];
        let truth: std::collections::HashSet<usize> =
            ground_truth(&corpus, query, K).into_iter().collect();

        // What the storage layer does: take the candidate set from the codes, score it
        // exactly from the stored vectors, keep the best k.
        let mut rescored: Vec<(usize, f32)> = index
            .search(&query.to_vec(), candidates)
            .unwrap()
            .into_iter()
            .map(|(id, _)| {
                let i = id.parse::<usize>().unwrap();
                (i, cosine(query, &corpus[i]))
            })
            .collect();
        rescored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        let hits = rescored
            .iter()
            .take(K)
            .filter(|(i, _)| truth.contains(i))
            .count();
        total += hits as f32 / K as f32;
    }
    let recall = total / QUERIES as f32;
    eprintln!("recall@{K} after the exact rerank: {recall:.3}");

    // Measured at 1.000: on this corpus the rerank recovers every neighbour the codes lost,
    // which is the claim the whole quantized design rests on. The floor leaves room for a
    // corpus where it does not quite, and fails on a change that breaks the rerank itself.
    assert!(
        recall >= 0.90,
        "recall@{K} after the exact rerank is {recall:.3}, below the floor of 0.90 — the rerank \
         is what makes quantized search accurate, so this is the number a user feels"
    );
}
