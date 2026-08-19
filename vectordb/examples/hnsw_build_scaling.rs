//! How does HNSW build time scale with the number of vectors?
//!
//! Not a benchmark — a shape check. Doubling the input should roughly double the time. Anything
//! close to quadrupling means an O(n) step hides inside the per-insert path, which is a different
//! and much cheaper fix than parallelising the build.
//!
//! An example rather than a test because `[profile.release]` sets `panic = "abort"`, which
//! `cargo test --release` refuses to link against, and a debug-build measurement of index
//! construction is worthless.
//!
//!     cargo run --release --example hnsw_build_scaling

use dxmt_vectordb::search::hnsw::HnswIndex;
use dxmt_vectordb::types::vector::DistanceMetric;

fn main() {
    let dim: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(128);

    println!(
        "dim={} data={}",
        dim,
        if std::env::var("CLUSTERED").is_ok() {
            "clustered"
        } else {
            "uniform"
        }
    );
    let mut previous: Option<(usize, f64)> = None;
    for n in [5_000usize, 10_000, 20_000, 40_000] {
        let mut index = HnswIndex::new(DistanceMetric::Cosine, None);
        // Uniform random, deterministically seeded. An earlier version of this used
        // `(i * 31 + d * 17) % 1000`, which produces heavily clustered near-duplicates: the graph
        // degenerates, layer searches terminate early, and the measured rate came out several
        // times higher than the same build achieves on real data. A construction benchmark whose
        // vectors are not spread out is measuring the wrong thing.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 8_388_608.0 - 1.0
        };
        // Uniform random is the worst case for ANN and, for the neighbour-selection heuristic
        // specifically, the one distribution where it provably has nothing to do: the heuristic
        // exists to keep long edges between clusters, and there are no clusters. Real embeddings
        // are clustered, so measure both.
        let clustered = std::env::var("CLUSTERED").is_ok();
        let raw: Vec<Vec<f32>> = if clustered {
            let centres: Vec<Vec<f32>> = (0..50)
                .map(|_| (0..dim).map(|_| next()).collect())
                .collect();
            (0..n)
                .map(|i| {
                    let c = &centres[i % centres.len()];
                    c.iter().map(|x| x + next() * 0.15).collect()
                })
                .collect()
        } else {
            (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
        };

        let started = std::time::Instant::now();
        for (i, v) in raw.iter().cloned().enumerate() {
            index.insert(format!("r{}", i), v).unwrap();
        }
        let secs = started.elapsed().as_secs_f64();
        let scaling = previous
            .map(|(pn, ps)| {
                format!(
                    "{:.2}x time for {:.0}x rows",
                    secs / ps,
                    n as f64 / pn as f64
                )
            })
            .unwrap_or_else(|| "-".to_string());
        println!(
            "n={:>6}  {:>7.2}s  {:>8.0} inserts/s  {:>8.0} us/insert   {}",
            n,
            secs,
            n as f64 / secs,
            secs * 1e6 / n as f64,
            scaling
        );
        previous = Some((n, secs));

        // Recall against exhaustive search. Build time is only interesting if the graph it
        // produces still answers correctly, and the layer assignment this measures was changed —
        // a faster build that finds worse neighbours is not an improvement.
        let queries: Vec<Vec<f32>> = (0..50)
            .map(|_| (0..dim).map(|_| next()).collect())
            .collect();
        let mut hits = 0usize;
        let mut total = 0usize;
        let mut truths: Vec<std::collections::HashSet<String>> = Vec::new();
        for q in &queries {
            let mut exact: Vec<(usize, f32)> = raw
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_distance(q, v)))
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let truth: std::collections::HashSet<String> = exact
                .iter()
                .take(10)
                .map(|(i, _)| format!("r{}", i))
                .collect();

            let got = index.search(q, 10).unwrap();
            hits += got.iter().filter(|(id, _)| truth.contains(id)).count();
            total += 10;
            truths.push(truth);
        }
        println!(
            "           recall@10 = {:.3} over 50 queries",
            hits as f64 / total as f64
        );

        // Does more search effort recover it? If recall climbs with ef the graph is sound and the
        // default is simply too small; if it plateaus, the graph itself is not connected well
        // enough for a search to reach the true neighbours however long it looks.
        for ef in [100usize, 200, 500, 1000] {
            let mut h = 0usize;
            for (q, truth) in queries.iter().zip(&truths) {
                let got = index.search_with_ef(q, 10, ef).unwrap();
                h += got.iter().filter(|(id, _)| truth.contains(id)).count();
            }
            print!("  ef={}:{:.3}", ef, h as f64 / (queries.len() * 10) as f64);
        }
        println!();
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}
