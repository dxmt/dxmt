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
//! insert: read the sample, train with no lock held, take the write lock only to install.
//!
//! That fixed what searches saw and left what writers saw. Training ran on the storage layer's
//! single insertion task, so releasing the lock did not stop it stopping every insert in the
//! process -- there is more than one thing serialising writes. A 24-hour soak at 200 rows/s on
//! eight cores caught it: training took 256 s holding no lock, searches were never slowed, and
//! eight batches of inserts sat in the queue past the client's 60 s timeout, answered when
//! their turn came by `Failed to send insert response: insertion result was dropped`. Training
//! now runs in a spawned task, off the runtime's worker threads, and the crossing insert
//! returns without waiting for it.
//!
//! So this test crosses the threshold and then asserts three things about the window while the
//! training runs: searches are not blocked, inserts are still acknowledged, and the quantizer
//! does eventually install.

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
async fn quantizing_at_the_threshold_blocks_neither_searches_nor_inserts() {
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

    // The crossing insert hands the training to a spawned task and returns. If this is
    // already false, the training finished before the assertion could be made -- which would
    // make everything below vacuous rather than passing.
    assert!(
        wrapper.is_quantizing(),
        "the threshold crossing was not followed by a background training job; either the \
         threshold was never crossed or training finished in under {insert_took:?}"
    );

    // The window under test. Inserts must keep being accepted while the training runs: this
    // is the assertion the soak's eight dropped batches would have failed.
    //
    // Paced, and capped at `WRITES_DURING_TRAINING`. An unpaced loop put 2.4 million rows into
    // the graph while the training ran, and then the install -- which re-encodes every row the
    // graph holds, under the write lock -- took 78 s of the 207 s window. That is a real
    // property (the install is as big as the table it lands on) but it is not this test's, and
    // measuring it here only hides whether an insert was blocked.
    const WRITES_DURING_TRAINING: usize = 200;
    let mut inserts_during_training = 0usize;
    let mut worst_insert = Duration::ZERO;
    let mut next_id = ROWS;
    let waiting = Instant::now();
    while wrapper.is_quantizing() {
        assert!(
            waiting.elapsed() < Duration::from_secs(300),
            "training never finished"
        );
        if inserts_during_training < WRITES_DURING_TRAINING {
            let batch: Vec<(String, Vec<f32>)> = (next_id..next_id + 50)
                .map(|i| (i.to_string(), vector(i)))
                .collect();
            next_id += 50;
            let t = Instant::now();
            wrapper
                .insert_batch(batch)
                .await
                .expect("an insert during training must still be served");
            worst_insert = worst_insert.max(t.elapsed());
            inserts_during_training += 1;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let training_took = waiting.elapsed();

    stop.store(true, Ordering::Relaxed);
    let (worst_search, searches) = searcher.await.unwrap();

    assert!(
        wrapper.len().await >= next_id,
        "every row should be in the index, the ones written during training included"
    );
    // Without this the test can pass for the wrong reason: if the threshold was never
    // crossed there is no training to be blocked by, and the assertions are vacuous.
    // `rerank_candidates` answers `Some` only once the graph holds codes.
    assert!(
        wrapper.rerank_candidates(10).await.is_some(),
        "the index never quantized, so this test measured nothing"
    );
    assert!(
        searches > 0,
        "the searching task never ran, so this proves nothing"
    );
    assert!(
        inserts_during_training > 0,
        "no insert landed during {training_took:?} of training, so nothing was proved about \
         whether one could"
    );

    // Neither a search nor the crossing insert may pay for the training. Both bounds are a
    // fraction of the training time rather than a wall clock, so the test scales with the
    // machine.
    let floor = Duration::from_millis(400);
    assert!(
        worst_search * 4 < training_took.max(floor),
        "a search waited {worst_search:?} of {training_took:?} of training over {searches} \
         searches: training is holding the index lock again"
    );
    assert!(
        insert_took * 2 < training_took.max(floor),
        "the crossing insert took {insert_took:?} against {training_took:?} of training: it \
         is waiting for the training again"
    );
    // Not blocked is stronger than eventually served, and it is the property the soak found
    // missing: the training held no lock and no runtime worker, and inserts still stopped
    // dead, because `plan_batch` and the k-means share rayon's global pool and a worker deep
    // in its chunk of the subspaces does not pick up anything else until it finishes.
    assert!(
        worst_insert * 3 < training_took.max(floor),
        "an insert waited {worst_insert:?} of {training_took:?} of training over \
         {inserts_during_training} inserts: the training is monopolising the thread pool the \
         insert path needs"
    );
}
