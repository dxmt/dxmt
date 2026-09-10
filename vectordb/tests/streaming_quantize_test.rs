//! Crossing the quantization threshold must not freeze the table.
//!
//! An HNSW-PQ index quantizes itself once it has grown past `quantize_at_rows`, which means
//! fitting `num_subvectors` codebooks by k-means. That takes about a minute for the shipped
//! 192-subspace default on 65,536 vectors, and it used to run in `after_batch`, inside the
//! index write lock. Nothing else could read or write the table while it ran.
//!
//! A batch load never notices -- the whole load is one long write. Anything that inserts and
//! searches at the same time notices completely. `StreamingPerformanceCase`, the one standard
//! VectorDBBench case that does both, had never produced a valid result in six attempts; the
//! run that finally showed why measured search p99 **47.7 s** with 200+ insertions backed up,
//! against a server log showing PQ training from 11:14:30 to 11:15:26.
//!
//! `IndexWrapper::quantize_if_wanted` splits it the way the wrapper already splits a batch
//! insert: read the sample, train with no lock held, take the write lock only to install. This
//! test crosses the threshold on one task while another searches in a loop, and asserts no
//! search was blocked for anywhere near the time the training took.

use dxmt_vectordb::search::{
    hnsw_pq::{HnswPqConfig, HnswPqIndex},
    index_wrapper::IndexWrapper,
};
use dxmt_vectordb::types::vector::DistanceMetric;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const DIM: usize = 32;
/// Past `PARALLEL_PLAN_MIN_NODES`, so the batch takes the phased path -- the one the real
/// case takes. Below it the graph inserts serially under the write lock for reasons that
/// have nothing to do with quantizing, and the measurement would be of that instead.
const THRESHOLD: usize = 12_000;
const ROWS: usize = 13_000;

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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn quantizing_at_the_threshold_does_not_block_searches() {
    let config = HnswPqConfig {
        // 8 subspaces of 8 dimensions, the full 256 centroids: enough k-means to take
        // seconds, which is what makes a stall visible at all.
        num_subvectors: 8,
        num_centroids: 256,
        quantize_at_rows: THRESHOLD,
        // A cheap graph: this test is about who holds the lock during training, not about
        // recall, and ef_construction 200 on 13,000 rows in a debug build is minutes.
        hnsw_config: dxmt_vectordb::search::hnsw::HnswConfig::from_params(8, 16, 32),
        ..HnswPqConfig::default()
    };
    let index = HnswPqIndex::new(DIM, DistanceMetric::Cosine, Some(config)).unwrap();
    let wrapper = Arc::new(IndexWrapper::new(Box::new(index)));

    // Fill to just under the threshold, so the next batch is the one that crosses it.
    let below: Vec<(String, Vec<f32>)> = (0..THRESHOLD - 500)
        .map(|i| (i.to_string(), vector(i)))
        .collect();
    wrapper.insert_batch(below).await.unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let searcher = {
        let wrapper = Arc::clone(&wrapper);
        let stop = Arc::clone(&stop);
        tokio::spawn(async move {
            let query = vector(7);
            let mut worst = Duration::ZERO;
            let mut searches = 0u32;
            while !stop.load(Ordering::Relaxed) {
                let t = Instant::now();
                let _ = wrapper.search(&query, 10).await.unwrap();
                worst = worst.max(t.elapsed());
                searches += 1;
                tokio::task::yield_now().await;
            }
            (worst, searches)
        })
    };

    // Cross the threshold. This is the call that used to hold the write lock through training.
    let crossing: Vec<(String, Vec<f32>)> = (THRESHOLD - 500..ROWS)
        .map(|i| (i.to_string(), vector(i)))
        .collect();
    let training = Instant::now();
    wrapper.insert_batch(crossing).await.unwrap();
    let insert_took = training.elapsed();

    stop.store(true, Ordering::Relaxed);
    let (worst_search, searches) = searcher.await.unwrap();

    assert!(
        wrapper.len().await >= ROWS,
        "every row should be in the index"
    );
    // Without this the test can pass for the wrong reason: if the threshold was never
    // crossed there is no training to be blocked by, and the latency assertion below is
    // vacuous. `rerank_candidates` answers `Some` only once the graph holds codes.
    assert!(
        wrapper.rerank_candidates(10).await.is_some(),
        "the index never quantized, so this test measured nothing"
    );
    assert!(
        searches > 0,
        "the searching task never ran, so this proves nothing"
    );

    // The insert that crosses the threshold pays for the training; a concurrent search must
    // not. The bound is a fraction of what the insert itself took, so the test scales with
    // the machine instead of pinning a wall clock.
    assert!(
        worst_search * 4 < insert_took.max(Duration::from_millis(400)),
        "a search waited {worst_search:?} while the threshold crossing took {insert_took:?} \
         over {searches} searches: training is holding the index lock again"
    );
}
