//! Where graph construction spends its time, as a flamegraph, measured in-process.
//!
//!     cargo test --profile release-unwind --features benchmarks --test build_profile_test \
//!         -- --ignored --nocapture
//!
//! Writes `DXMT_PROFILE_OUT` (default `target/build_profile.svg`) and prints the build rate.
//! `DXMT_PROFILE_ROWS` (default 100,000) and `DXMT_PROFILE_DIM` (default 100) shape the
//! corpus; uniform noise, which is what hnswlib's reference figure on this box was taken on:
//! 7,963 rows/s for 100K x 100 on 16 threads, 1,239 rows/s at 960.
//!
//! Two layers are timed separately, because the ingest question has two possible answers:
//! `HnswIndex::insert_batch` on its own -- the graph with nothing around it -- and the same
//! rows through `IntelligentIndexedStorage::insert_records`, which adds the record store, the
//! row file, the WAL and the insert processor. The profile covers the graph alone; the
//! storage figure says how much of the gap to the reference is the graph's to close.

use std::time::Instant;

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    search::hnsw::{HnswConfig, HnswIndex},
    storage::IntelligentIndexedStorage,
    types::{record::CreateRecordRequest, table::TableConfig, vector::DistanceMetric},
};
use rand::{Rng, SeedableRng};

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn corpus(rows: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
    (0..rows)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "profile; run with --ignored --nocapture"]
async fn where_the_build_spends_its_time() {
    let rows = env("DXMT_PROFILE_ROWS", 100_000);
    let dim = env("DXMT_PROFILE_DIM", 100);
    let out = std::env::var("DXMT_PROFILE_OUT")
        .unwrap_or_else(|_| "target/build_profile.svg".to_string());
    let vectors = corpus(rows, dim);

    // 1. The graph alone, under the profiler.
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(499)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .expect("profiler");
    let mut index = HnswIndex::new(DistanceMetric::Euclidean, Some(HnswConfig::default()));
    let started = Instant::now();
    for (b, chunk) in vectors.chunks(1_000).enumerate() {
        let batch: Vec<(String, Vec<f32>)> = chunk
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("r{}", b * 1_000 + i), v.clone()))
            .collect();
        index.insert_batch(batch).unwrap();
    }
    let graph_s = started.elapsed().as_secs_f64();
    if let Ok(report) = guard.report().build() {
        let file = std::fs::File::create(&out).expect("profile output");
        report.flamegraph(file).expect("flamegraph");
        // Folded stacks beside it, for self-time by function (`inferno-collapse` shape):
        // one line per distinct stack, root first, then the sample count.
        let folded = format!("{out}.folded");
        let mut lines = Vec::with_capacity(report.data.len());
        for (frames, count) in &report.data {
            let names: Vec<String> = frames
                .frames
                .iter()
                .rev()
                .flat_map(|frame| frame.iter().rev().map(|sym| sym.name()))
                .collect();
            lines.push(format!("{} {}", names.join(";"), count));
        }
        std::fs::write(&folded, lines.join("\n")).expect("folded output");
    }
    drop(guard);
    println!(
        "graph alone: {rows} x {dim} in {graph_s:.1}s = {:.0} rows/s (hnswlib on this box: 7,963 at 100-d, 1,239 at 960-d); profile at {out}",
        rows as f64 / graph_s
    );

    // 2. The same rows through storage.
    let storage = IntelligentIndexedStorage::new(&StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 4096,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    })
    .await
    .unwrap();
    storage
        .create_table(TableConfig {
            name: "p".to_string(),
            dimension: dim,
            distance_metric: DistanceMetric::Euclidean,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            // `DXMT_PROFILE_PQ=0` keeps the table an exact graph, for measuring what the
            // walk on codes costs the build at a given width.
            quantization: (std::env::var("DXMT_PROFILE_PQ").as_deref() == Ok("0")).then_some(
                dxmt_vectordb::types::table::QuantizationParams {
                    num_subvectors: None,
                    num_centroids: None,
                    quantize_at_rows: Some(usize::MAX),
                },
            ),
            payload_index: None,
            embedding: None,
        })
        .await
        .unwrap();
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(499)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .expect("profiler");
    let started = Instant::now();
    for (b, chunk) in vectors.chunks(1_000).enumerate() {
        let batch: Vec<CreateRecordRequest> = chunk
            .iter()
            .enumerate()
            .map(|(i, v)| CreateRecordRequest {
                id: format!("r{}", b * 1_000 + i),
                vector: v.clone(),
                sparse_vector: None,
                named_vectors: None,
                payload: None,
                text: None,
            })
            .collect();
        storage.insert_records("p", batch).await.unwrap();
    }
    let storage_s = started.elapsed().as_secs_f64();
    if let Ok(report) = guard.report().build() {
        let out2 = out.replace(".svg", ".storage.svg");
        let file = std::fs::File::create(&out2).expect("profile output");
        report.flamegraph(file).expect("flamegraph");
        let mut lines = Vec::with_capacity(report.data.len());
        for (frames, count) in &report.data {
            let names: Vec<String> = frames
                .frames
                .iter()
                .rev()
                .flat_map(|frame| frame.iter().rev().map(|sym| sym.name()))
                .collect();
            lines.push(format!("{} {}", names.join(";"), count));
        }
        std::fs::write(format!("{out2}.folded"), lines.join("\n")).expect("folded output");
    }
    drop(guard);
    println!(
        "through storage (no WAL dir): {rows} x {dim} in {storage_s:.1}s = {:.0} rows/s",
        rows as f64 / storage_s
    );
}
