#![cfg(feature = "benchmarks")]

//! What the filter paths cost at scale, measured rather than reasoned about.
//!
//! Three questions this repository could not answer, each about a feature whose *correctness*
//! is covered elsewhere and whose *price* was never taken:
//!
//! 1. **A geographic filter** walks a band of latitudes and tests each point exactly. That is
//!    bounded by the band, not by the table — but by how much, at what radius, on how many
//!    rows? And the geo half of the index is 24 bytes a point on top of what the same value
//!    already costs in the numeric half.
//! 2. **A facet** walks postings rather than rows, and computes `covered` as a *set* of record
//!    ordinals. On a large table that set is a real allocation, and the cost depends on the
//!    field's cardinality in a way the shape of the code does not make obvious.
//! 3. **Does a filter make a search worse?** The graph is walked with a filter, or the
//!    candidate set is scored exactly, depending on the estimated selectivity, and the worry
//!    was that somewhere on that curve the answer comes back short or wrong. Asked as a
//!    *relative* question — filtered against unfiltered on the same corpus, both against
//!    `SearchMode::Exact` — because that is the question, and because the absolute recall of a
//!    synthetic corpus is a number about the corpus. The absolute figures this project quotes
//!    come from the benchmark harness on Cohere and Bioasq, and belong there.
//!
//! **Read `quantized` before reading any recall row.** Codebook training runs in the
//! background after the rows cross the threshold, so two runs at different scales can be
//! asking different engines while reporting the same index configuration. Measured here:
//! `false` at 100,000 and at 300,000 rows, `true` at a million. With it true this corpus
//! scores about 0.49 and with it false 1.000 — which is what a corpus of four thousand tight
//! clusters does to product quantization (250 near-identical members per cluster share a
//! code), not a statement about the engine on real embeddings.
//!
//! ```bash
//! DXMT_SCALE_N=1000000 cargo test --profile release-unwind --features benchmarks \
//!   --test scale_filters_test -- --ignored --nocapture
//! ```
//!
//! `DXMT_SCALE_N` (default 50,000) and `DXMT_SCALE_DIM` (default 64) scale it. The defaults
//! are small enough to run in under a minute so the harness itself stays exercised; the
//! numbers worth quoting come from a run at a million.
//!
//! **Measurement traps this harness avoids**, each of which has cost this repository a wrong
//! conclusion before:
//!
//! * The corpus is built *before* the clock starts on any query, and every vector is distinct.
//!   A corpus of repeated vectors makes recall meaningless — a thousand records tie at 1.0 and
//!   the approximate and exact answers pick different tens of them.
//! * Every query is run `REPEATS` times and the **median** reported, because the first touch
//!   of a cold structure is not the steady state anyone will see.
//! * Resident memory is sampled from `/proc/self/status`, not from a `pgrep` of a command line
//!   that would match this test's own process.

use std::time::{Duration, Instant};

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{
            Condition, CreateRecordRequest, Filter, FilterOperation, GeoPoint, GeoRadius,
            SearchMode, SearchQuery,
        },
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use serde_json::json;

const DEFAULT_N: usize = 50_000;
const DEFAULT_DIM: usize = 64;
const BATCH: usize = 5_000;
const REPEATS: usize = 7;
/// Distinct values of the medium-cardinality facet field.
const ZONES: usize = 2_000;
/// Distinct values of the low-cardinality one.
const KINDS: [&str; 8] = [
    "cafe", "bank", "shop", "park", "school", "clinic", "hotel", "garage",
];
/// City centres the corpus clusters around, so a radius selects a realistic fraction rather
/// than a uniform slice of the planet.
const CITIES: [(f64, f64); 8] = [
    (52.5163, 13.3777),   // Berlin
    (48.8566, 2.3522),    // Paris
    (51.5074, -0.1278),   // London
    (40.7128, -74.0060),  // New York
    (35.6762, 139.6503),  // Tokyo
    (-33.8688, 151.2093), // Sydney
    (37.7749, -122.4194), // San Francisco
    (1.3521, 103.8198),   // Singapore
];

fn env(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Resident set size in megabytes, read from this process rather than found by pattern.
fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

/// A deterministic pseudo-random stream. Not a good generator; a reproducible one.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn storage_config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 8_192,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

fn table(dimension: usize) -> TableConfig {
    TableConfig {
        name: "scale".to_string(),
        dimension,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        // The shipped defaults on purpose: these numbers are meant to describe what an
        // operator gets, not what a tuned graph could do.
        hnsw: None,
        // `DXMT_SCALE_EXACT_INDEX=1` pushes the quantization threshold above the corpus, so
        // the table stays an exact HNSW graph. It exists to tell two explanations of a low
        // recall apart: a corpus the graph finds hard, and product quantization losing more
        // than expected. Nothing else here can distinguish them.
        quantization: (env("DXMT_SCALE_EXACT_INDEX", 0) == 1).then_some(
            dxmt_vectordb::types::table::QuantizationParams {
                num_subvectors: None,
                num_centroids: None,
                quantize_at_rows: Some(usize::MAX),
                fast_scan: None,
            },
        ),
        payload_index: None,
        embedding: None,
        rerank: None,
    }
}

/// The median of `REPEATS` runs of an async `f`, and whatever the last run returned.
///
/// Async rather than `block_on` inside the test's own runtime: blocking a worker thread on a
/// future that then wants a lock held by that runtime is a deadlock waiting for a contended
/// moment, and a measurement harness that hangs is worse than one that is slightly slower.
async fn median<T, F, Fut>(mut f: F) -> (Duration, T)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let mut times = Vec::with_capacity(REPEATS);
    let mut last = None;
    for _ in 0..REPEATS {
        let start = Instant::now();
        last = Some(f().await);
        times.push(start.elapsed());
    }
    times.sort();
    (times[REPEATS / 2], last.unwrap())
}

fn geo_filter(lat: f64, lon: f64, radius_m: f64) -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: "at".to_string(),
            operation: FilterOperation::GeoRadius(GeoRadius {
                center: GeoPoint { lat, lon },
                radius_m,
            }),
        }]),
        must_not: None,
        should: None,
    }
}

fn kinds_filter(kinds: &[&str]) -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: "kind".to_string(),
            operation: FilterOperation::In(kinds.iter().map(|k| json!(k)).collect()),
        }]),
        must_not: None,
        should: None,
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "scale measurement; run with --ignored --nocapture"]
async fn what_the_filter_paths_cost() {
    let n = env("DXMT_SCALE_N", DEFAULT_N);
    let dim = env("DXMT_SCALE_DIM", DEFAULT_DIM);
    println!("\n=== corpus: {n} rows, {dim} dimensions, shipped index defaults ===");

    let before_rss = rss_mb();
    let storage = IntelligentIndexedStorage::new(&storage_config())
        .await
        .unwrap();
    storage.create_table(table(dim)).await.unwrap();

    // Roughly 250 rows a cluster, which is the density `parallel_build_recall_test` measures
    // 0.984 at. A *fixed* cluster count does not survive being scaled: sixty clusters over a
    // million rows is sixteen thousand near-identical vectors each, the top ten among them is
    // close to arbitrary, and the unfiltered control came back at 0.535 -- measuring the
    // corpus rather than the index. This is the third corpus written here and the first that
    // can answer the question; the two before it are recorded above and in the control row,
    // because "recall is low" and "recall cannot be measured on this" look identical.
    let clusters = (n / 250).max(8);
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let started = Instant::now();
    let mut generating = Duration::ZERO;
    for start in (0..n).step_by(BATCH) {
        let t = Instant::now();
        let mut batch = Vec::with_capacity(BATCH);
        for i in start..(start + BATCH).min(n) {
            let city = i % CITIES.len();
            // A cluster about 50 km across, so a 1 km radius is selective and a 100 km one is
            // not -- both ends of the question.
            let (clat, clon) = CITIES[city];
            let lat = (clat + (rng.next() - 0.5) * 0.9).clamp(-89.0, 89.0);
            let lon = (clon + (rng.next() - 0.5) * 0.9).clamp(-179.0, 179.0);
            // The generator `parallel_build_recall_test` measures 0.984 with, rather than one
            // of my own. The first version here put 125,000 vectors per cluster a hair apart
            // in 64 dimensions, and the *unfiltered* control came back at recall 0.705: on a
            // corpus of near-ties the top ten is close to arbitrary and recall measures the
            // corpus, not the index. Sixty clusters with real spread is what makes the number
            // mean something. (The geographic clustering is separate and stays at eight
            // cities, because a radius has to select a realistic fraction.)
            let cluster = (i % clusters) as f32;
            let vector: Vec<f32> = (0..dim)
                .map(|d| (cluster * 0.29 + d as f32 * 0.07).sin() + (rng.next() as f32 - 0.5) * 0.2)
                .collect();
            batch.push(CreateRecordRequest {
                id: format!("r{i}"),
                vector,
                text: None,
                sparse_vector: None,
                named_vectors: None,
                payload: Some(json!({
                    "at": {"lat": lat, "lon": lon},
                    "kind": KINDS[i % KINDS.len()],
                    "zone": i % ZONES,
                })),
            });
        }
        generating += t.elapsed();
        storage.insert_records("scale", batch).await.unwrap();
    }
    let build = started.elapsed();
    println!(
        "build: {:.1} s total, of which {:.1} s was this harness generating rows \
         ({:.0} rows/s of server time)",
        build.as_secs_f64(),
        generating.as_secs_f64(),
        n as f64 / (build - generating).as_secs_f64()
    );

    // The payload index is built on the first read that needs it, not at insert, so the first
    // filtered query pays for it. Timed on its own rather than hidden inside a query median.
    let t = Instant::now();
    let warm = storage
        .count_records("scale", Some(&kinds_filter(&["cafe"])))
        .await
        .unwrap();
    println!(
        "payload index built on first use: {:.2} s ({warm} of {n} matched)",
        t.elapsed().as_secs_f64()
    );

    // The report comes back as the JSON the endpoint serves, which is what an operator would
    // be reading, so this prints the same numbers they would see.
    // Which index the selector landed on, printed because it is the first thing that explains
    // a recall number and the last thing anyone remembers to check: the choice depends on the
    // row count, so two runs at different scales are not running the same engine.
    println!("index: {:?}", storage.describe_index("scale").await);
    // Whether the codebooks have actually landed. Training runs in the background after the
    // rows cross the threshold, so two runs at different scales can be asking *different
    // engines* while reporting the same index configuration -- the small one still an exact
    // graph, the large one on codes. A recall comparison that does not check this is
    // comparing two things it has not identified.
    let stats = storage.get_table_stats("scale").await.unwrap();
    println!("quantized: {}", stats["quantized"]);

    let report = storage.payload_index_report("scale").await.unwrap();
    println!("\n--- what the index holds ---");
    for field in report["fields"].as_array().into_iter().flatten() {
        let get = |name: &str| field[name].as_u64().unwrap_or(0);
        println!(
            "  {:<10} distinct {:>8}  keyword rows {:>9}  numeric {:>9}  geo {:>9}  ~{:>7.1} MB",
            field["field"].as_str().unwrap_or("?"),
            get("distinct_values"),
            get("keyword_rows"),
            get("numeric_entries"),
            get("geo_points"),
            get("estimated_bytes") as f64 / 1e6
        );
    }
    println!(
        "  process RSS: {:.0} MB (was {:.0} MB before the corpus)",
        rss_mb(),
        before_rss
    );

    println!("\n--- a geographic filter, by radius (median of {REPEATS}) ---");
    let (clat, clon) = CITIES[0];
    for radius_m in [1_000.0, 10_000.0, 100_000.0, 1_000_000.0] {
        let filter = geo_filter(clat, clon, radius_m);
        let (elapsed, matched) =
            median(|| async { storage.count_records("scale", Some(&filter)).await.unwrap() }).await;
        println!(
            "  radius {:>9.0} m: {:>8.2} ms, {matched:>8} of {n} matched ({:.2}%)",
            radius_m,
            elapsed.as_secs_f64() * 1000.0,
            matched as f64 * 100.0 / n as f64
        );
    }

    println!("\n--- a facet, by the field's cardinality (median of {REPEATS}) ---");
    for (key, expected) in [("kind", KINDS.len()), ("zone", ZONES)] {
        let (elapsed, facet) =
            median(|| async { storage.facet("scale", key, None, 20).await.unwrap() }).await;
        println!(
            "  {key:<6} {:>5} distinct: {:>8.2} ms, covered {}/{}",
            facet.distinct_values,
            elapsed.as_secs_f64() * 1000.0,
            facet.covered,
            facet.matched
        );
        assert_eq!(
            facet.distinct_values, expected,
            "the corpus is not as described"
        );
    }
    // Under a filter, which adds one evaluation of the filter and an intersection per posting.
    let near = geo_filter(clat, clon, 10_000.0);
    for key in ["kind", "zone"] {
        let (elapsed, facet) =
            median(|| async { storage.facet("scale", key, Some(&near), 20).await.unwrap() }).await;
        println!(
            "  {key:<6} under a 10 km radius: {:>8.2} ms, covered {}/{}",
            elapsed.as_secs_f64() * 1000.0,
            facet.covered,
            facet.matched
        );
    }

    println!("\n--- a filtered search against the exact answer, by selectivity ---");
    println!("    (recall@10 of SearchMode::Normal against SearchMode::Exact, 20 queries)");
    println!(
        "    Read these against the control, not against 1.0: the absolute number belongs to\n\
         \x20   the corpus (and to `quantized` above), and the question is whether a filter\n\
         \x20   makes the answer worse than no filter does."
    );
    let mut probe = Rng(0x9E37_79B9_7F4A_7C15);
    for kinds in [
        &[] as &[&str],
        &KINDS[0..1],
        &KINDS[0..2],
        &KINDS[0..4],
        &KINDS[0..8],
    ] {
        let filter = kinds_filter(kinds);
        let mut hits = 0usize;
        let mut expected = 0usize;
        let mut short = 0usize;
        let mut total_ms = 0.0;
        for _ in 0..20 {
            // A query near a cluster rather than on a stored vector: an exact hit on a row
            // the graph contains is a weaker question, since it only has to return the node
            // it starts from.
            let cluster = (probe.next() * clusters as f64) as f32;
            let query: Vec<f32> = (0..dim)
                .map(|d| {
                    (cluster * 0.29 + d as f32 * 0.07).sin() + (probe.next() as f32 - 0.5) * 0.2
                })
                .collect();
            let ask = |mode| SearchQuery {
                vector: query.clone(),
                top_k: 10,
                limit: 10,
                filter: (!kinds.is_empty()).then(|| filter.clone()),
                search_mode: mode,
                ..Default::default()
            };
            let truth: std::collections::HashSet<String> = storage
                .search("scale", ask(SearchMode::Exact))
                .await
                .unwrap()
                .hits
                .into_iter()
                .map(|h| h.id)
                .collect();
            let t = Instant::now();
            let got = storage
                .search("scale", ask(SearchMode::Normal))
                .await
                .unwrap()
                .hits;
            total_ms += t.elapsed().as_secs_f64() * 1000.0;
            expected += truth.len();
            hits += got.iter().filter(|h| truth.contains(&h.id)).count();
            // Separate from recall: a page that is *short* is a different failure from one
            // that is full of the wrong records, and only one of them looks like a bug.
            if got.len() < truth.len() {
                short += 1;
            }
        }
        let label = if kinds.is_empty() {
            "  no filter        (control)".to_string()
        } else {
            format!(
                "  {} of 8 kinds ({:>5.1}% of rows)",
                kinds.len(),
                kinds.len() as f64 * 100.0 / KINDS.len() as f64
            )
        };
        println!(
            "{label:<32}: recall@10 {:.3}, {short}/20 pages short, {:>7.2} ms",
            hits as f64 / expected.max(1) as f64,
            total_ms / 20.0
        );
    }
    println!();
}
