//! Does a budget calibrated on queries the table makes for itself hold for the queries a
//! caller sends? Measured on ann-benchmarks corpora against the ground truth they ship.
//!
//! Recall calibration (`calibrate_recall`) has no held-out queries, so it makes its own: a
//! point 0.35 of the way between two stored rows. This harness loads a corpus, then walks
//! the same graph at a series of budgets with four query sets -- the dataset's real test
//! queries scored against its `neighbors`, the calibration's 0.35 interpolation, the
//! midpoint, and stored rows themselves -- the last three scored against the exact scan.
//! The gap between the first column and the second, at the budget the calibration picks,
//! is what a caller who trusts the calibration gets.
//!
//! Ignored: needs the raw arrays `scripts/ann_to_raw.py` writes from an ann-benchmarks
//! hdf5 file (`<set>.train.f32`, `.test.f32`, `.neighbors.i32`, `.meta.json`).
//!
//!   DXMT_ANN_DIR=/data/ann DXMT_ANN_SET=glove-100-angular \
//!     cargo test --profile release-unwind --test calibration_transfer_test -- --ignored --nocapture
//!
//! Measured 2026-10-04 (16 cores, release-unwind, in-process; the table in
//! docs/SUPPORT_MATRIX.md under "recall calibration"): on glove-25, glove-100 and lastfm-64
//! the 0.35 interpolation reads 0.7 to 3.5 points *below* the real queries at every budget,
//! the midpoint up to 11 points below, and stored rows 1 to 8 points above. The
//! calibration's claim is therefore a floor on what a caller gets, which is the side to
//! err on, and 0.35 is the closest of the three.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use rand::{Rng, SeedableRng};

/// A query and the ids it should return.
type Labelled = (Vec<f32>, Vec<String>);

const K: usize = 10;
const REAL_QUERIES: usize = 1_000;
const SYNTHETIC_QUERIES: usize = 300;
const BUDGETS: [usize; 6] = [32, 64, 128, 256, 512, 1024];

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn read_i32(path: &std::path::Path) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn query(v: &[f32], mode: SearchMode) -> SearchQuery {
    SearchQuery {
        vector: v.to_vec(),
        top_k: K,
        limit: K,
        search_mode: mode,
        with_payload: false,
        ..Default::default()
    }
}

async fn ids_at(
    storage: &IntelligentIndexedStorage,
    v: &[f32],
    ef: Option<usize>,
    mode: SearchMode,
) -> Vec<String> {
    storage
        .search_with_budget("t", query(v, mode), ef)
        .await
        .unwrap()
        .hits
        .into_iter()
        .map(|h| h.id)
        .collect()
}

fn recall(got: &[String], truth: &[String]) -> f32 {
    got.iter().filter(|g| truth.contains(g)).count() as f32 / truth.len() as f32
}

fn find_key<'a>(v: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    match v {
        serde_json::Value::Object(m) => m
            .get(key)
            .or_else(|| m.values().find_map(|x| find_key(x, key))),
        serde_json::Value::Array(a) => a.iter().find_map(|x| find_key(x, key)),
        _ => None,
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn calibration_transfer_on_a_real_corpus() {
    let dir = std::path::PathBuf::from(std::env::var("DXMT_ANN_DIR").expect("DXMT_ANN_DIR"));
    let set = std::env::var("DXMT_ANN_SET").expect("DXMT_ANN_SET");
    let base = dir.join(&set);
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{}.meta.json", base.display())).unwrap())
            .unwrap();
    let n = meta["n"].as_u64().unwrap() as usize;
    let dim = meta["dim"].as_u64().unwrap() as usize;
    let nq = meta["nq"].as_u64().unwrap() as usize;
    let metric = match meta["metric"].as_str().unwrap() {
        "angular" => DistanceMetric::Cosine,
        "euclidean" => DistanceMetric::Euclidean,
        other => panic!("metric {other}"),
    };
    let train = read_f32(&std::path::PathBuf::from(format!(
        "{}.train.f32",
        base.display()
    )));
    let test = read_f32(&std::path::PathBuf::from(format!(
        "{}.test.f32",
        base.display()
    )));
    let neighbors = read_i32(&std::path::PathBuf::from(format!(
        "{}.neighbors.i32",
        base.display()
    )));
    assert_eq!(train.len(), n * dim);
    assert_eq!(test.len(), nq * dim);
    assert_eq!(neighbors.len(), nq * K);
    let row = |i: usize| &train[i * dim..(i + 1) * dim];

    let storage = IntelligentIndexedStorage::new(&StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 65_536,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    })
    .await
    .unwrap();
    storage
        .create_table(TableConfig {
            name: "t".to_string(),
            dimension: dim,
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
        })
        .await
        .unwrap();

    let started = std::time::Instant::now();
    for chunk in (0..n).collect::<Vec<_>>().chunks(5_000) {
        let batch = chunk
            .iter()
            .map(|&i| CreateRecordRequest {
                id: format!("r{i}"),
                vector: row(i).to_vec(),
                sparse_vector: None,
                named_vectors: None,
                payload: None,
                text: None,
            })
            .collect();
        storage.insert_records("t", batch).await.unwrap();
        if chunk[0] % 200_000 == 0 && chunk[0] > 0 {
            println!(
                "  {} rows, {:.0} rows/s",
                chunk[0],
                chunk[0] as f64 / started.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "{set}: {n} x {dim} loaded in {:.0}s ({:.0} rows/s)",
        started.elapsed().as_secs_f64(),
        n as f64 / started.elapsed().as_secs_f64()
    );
    // The default index quantizes in the background past 65,536 rows; measuring before it
    // has is measuring a different engine.
    let waited = std::time::Instant::now();
    loop {
        let stats = storage.get_table_stats("t").await.unwrap();
        if find_key(&stats, "quantized").and_then(|q| q.as_bool()) == Some(true) {
            break;
        }
        if waited.elapsed().as_secs() > 1_800 {
            panic!("not quantized after 30 minutes: {stats}");
        }
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }
    println!(
        "quantized after a further {:.0}s",
        waited.elapsed().as_secs_f64()
    );

    // Query sets. Real: the dataset's, with its ground truth. Synthetic: against the exact
    // scan, as the calibration scores its own.
    let real: Vec<Labelled> = (0..REAL_QUERIES.min(nq))
        .map(|q| {
            (
                test[q * dim..(q + 1) * dim].to_vec(),
                neighbors[q * K..(q + 1) * K]
                    .iter()
                    .map(|&i| format!("r{i}"))
                    .collect(),
            )
        })
        .collect();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
    let pairs: Vec<(usize, usize)> = (0..SYNTHETIC_QUERIES)
        .map(|_| (rng.gen_range(0..n), rng.gen_range(0..n)))
        .collect();
    let lerp = |a: usize, b: usize, t: f32| -> Vec<f32> {
        row(a)
            .iter()
            .zip(row(b))
            .map(|(x, y)| x + t * (y - x))
            .collect()
    };
    let mut synthetic: Vec<(&str, Vec<Labelled>)> = Vec::new();
    for (name, t) in [
        ("interp 0.35", 0.35f32),
        ("midpoint", 0.5),
        ("self (row)", 0.0),
    ] {
        let mut set = Vec::with_capacity(pairs.len());
        for &(a, b) in &pairs {
            let q = lerp(a, b, t);
            let truth = ids_at(&storage, &q, None, SearchMode::Exact).await;
            set.push((q, truth));
        }
        synthetic.push((name, set));
    }

    println!(
        "\n| ef | real queries vs ground truth | interp 0.35 | midpoint | self (row) | p50 real |"
    );
    println!("| --- | --- | --- | --- | --- | --- |");
    let mut real_curve = Vec::new();
    for &ef in &BUDGETS {
        let mut r = 0f32;
        let mut lat = Vec::with_capacity(real.len());
        for (q, truth) in &real {
            let t0 = std::time::Instant::now();
            let got = ids_at(&storage, q, Some(ef), SearchMode::Normal).await;
            lat.push(t0.elapsed().as_secs_f64() * 1e3);
            r += recall(&got, truth);
        }
        let real_r = r / real.len() as f32;
        real_curve.push((ef, real_r));
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut cols = Vec::new();
        for (_, set) in &synthetic {
            let mut s = 0f32;
            for (q, truth) in set {
                s += recall(
                    &ids_at(&storage, q, Some(ef), SearchMode::Normal).await,
                    truth,
                );
            }
            cols.push(format!("{:.4}", s / set.len() as f32));
        }
        println!(
            "| {ef} | {real_r:.4} | {} | {:.2} ms |",
            cols.join(" | "),
            lat[lat.len() / 2]
        );
    }

    // What the product's own calibration picks, and what real queries get there.
    let cal = storage
        .calibrate_recall("t", true)
        .await
        .unwrap()
        .expect("a calibration");
    println!(
        "\ncalibration: target {} -> ef_search {} (sample recall {:.4}, met {}), curve {:?}",
        cal.target,
        cal.ef_search,
        cal.measured,
        cal.met,
        cal.curve
            .iter()
            .map(|p| (p.ef_search, p.recall))
            .collect::<Vec<_>>()
    );
    let mut r = 0f32;
    for (q, truth) in &real {
        r += recall(
            &ids_at(&storage, q, Some(cal.ef_search), SearchMode::Normal).await,
            truth,
        );
    }
    println!(
        "real queries at the calibrated ef_search {}: recall@10 {:.4} (calibration claimed {:.4})",
        cal.ef_search,
        r / real.len() as f32,
        cal.measured
    );
}
