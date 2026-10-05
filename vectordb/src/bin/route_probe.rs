//! The two measurements that decide the disk-index route.
//!
//! Every route that puts vectors on disk traverses on a compressed copy and keeps the full
//! vectors for reranking. So before choosing between IVF-PQ, DiskANN and HNSW+quantization the
//! questions are (1) how much recall the compressed copy loses on *our* data, and (2) how far a
//! partition index (no graph) trails a graph at the same candidate budget. Both are cheap to
//! measure and either answer can eliminate a route. This binary measures them on the
//! VectorDBBench Cohere 768-d set, against a brute-force truth computed here rather than the
//! shipped `neighbors.parquet` so the metric and the ordering are beyond doubt.
//!
//!     route_probe <dir> [stages=truth,pq,sq,ivf,hnsw] [n=1000000] [queries=1000] [k=10] [pq=96] [build=decoded] [quantize=1] [quantize_at=N]
//!
//! `<dir>` holds `train.f32` (row-major float32), `train_ids.i64`, `test.f32`, `test_ids.i64`,
//! `neighbors.i64` (100 ids per query) as written by `convert.py`. `<dir>=synthetic` makes a
//! small random corpus so the binary can be smoke-tested where the dataset is not.
//!
//! The PQ stage uses the crate's `ProductQuantizer` on purpose — the number that matters is
//! what the code we would ship loses, not what a textbook PQ loses. Its ADC table is scored two
//! ways: as the crate builds it, and with the per-subspace distances squared, because a sum of
//! square roots does not rank the way a sum of squares does (see `scan_pq`).

use dxmt_vectordb::search::hnsw::{HnswConfig, HnswIndex};
use dxmt_vectordb::search::quantization::product_quantization::ProductQuantizer;
use dxmt_vectordb::search::simd_distance::dot_product;
use dxmt_vectordb::types::DistanceMetric;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;

const DIM: usize = 768;

/// `k` of recall@k. VectorDBBench scores recall@100, so 10 (the default) and 100 both matter.
static K_OF: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// `pq=<m>`: the code size (bytes per vector) the graph walk, the IVF stage and the decoded
/// build use. 96 is the default; at recall@100 the 96-byte code needs a 1,000-candidate
/// rerank where 192 bytes need 200, so both are worth walking with.
static WALK_M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
fn walk_m() -> usize {
    *WALK_M.get().unwrap_or(&96)
}
#[allow(non_snake_case)]
fn K() -> usize {
    *K_OF.get().unwrap_or(&10)
}

/// How many walk/scan candidates to rerank exactly, per `K()`.
fn rerank_at() -> Vec<usize> {
    let k = K();
    [1, 2, 5, 10, 20, 50].iter().map(|f| f * k).collect()
}

fn scan_cap() -> usize {
    *rerank_at().last().unwrap()
}

struct Data {
    n: usize,
    /// Row-major, every row unit length. Cosine on unit vectors is a dot product, and the L2
    /// ordering the PQ codebooks are trained under agrees with it.
    train: Vec<f32>,
    queries: Vec<Vec<f32>>,
    /// The shipped ground truth mapped to row numbers, for a cross-check only.
    shipped_truth: Vec<Vec<u32>>,
}

impl Data {
    fn row(&self, i: usize) -> &[f32] {
        &self.train[i * DIM..(i + 1) * DIM]
    }
}

fn normalize(v: &mut [f32]) {
    let norm = dot_product(v, v).sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Streamed rather than `fs::read` + convert: at 10M x 768 the byte copy is 30 GB, and holding
/// it next to the float copy while converting took the process to 108 GB on a 125 GB box.
fn read_f32(path: &str) -> Vec<f32> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let len = file.metadata().unwrap().len() as usize;
    let mut out: Vec<f32> = Vec::with_capacity(len / 4);
    let mut buf = vec![0u8; 64 << 20];
    loop {
        let got = file.read(&mut buf).unwrap();
        if got == 0 {
            break;
        }
        out.extend(
            buf[..got - got % 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
    }
    out
}

fn read_i64(path: &str) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn load(dir: &str, n: usize, nq: usize) -> Data {
    if dir == "synthetic" {
        return synthetic(n, nq);
    }
    let t = Instant::now();
    let mut train = read_f32(&format!("{dir}/train.f32"));
    train.truncate(n * DIM);
    let n = train.len() / DIM;
    train.par_chunks_mut(DIM).for_each(normalize);
    let ids = read_i64(&format!("{dir}/train_ids.i64"));
    let row_of: HashMap<i64, u32> = ids
        .iter()
        .take(n)
        .enumerate()
        .map(|(row, &id)| (id, row as u32))
        .collect();

    let test = read_f32(&format!("{dir}/test.f32"));
    let mut queries: Vec<Vec<f32>> = test
        .chunks_exact(DIM)
        .take(nq)
        .map(|q| q.to_vec())
        .collect();
    queries.iter_mut().for_each(|q| normalize(q));

    let nb = read_i64(&format!("{dir}/neighbors.i64"));
    let per = nb.len() / (test.len() / DIM);
    let shipped_truth = nb
        .chunks_exact(per)
        .take(nq)
        .map(|ids| {
            ids.iter()
                .filter_map(|id| row_of.get(id).copied())
                .take(K())
                .collect()
        })
        .collect();
    println!(
        "loaded {} rows x {} dims, {} queries, {} shipped neighbours/query, in {:.1}s {}",
        n,
        DIM,
        queries.len(),
        per,
        t.elapsed().as_secs_f64(),
        mem_tag()
    );
    Data {
        n,
        train,
        queries,
        shipped_truth,
    }
}

fn synthetic(n: usize, nq: usize) -> Data {
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let mut train: Vec<f32> = (0..n * DIM).map(|_| next()).collect();
    train.par_chunks_mut(DIM).for_each(normalize);
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|_| {
            let mut q: Vec<f32> = (0..DIM).map(|_| next()).collect();
            normalize(&mut q);
            q
        })
        .collect();
    Data {
        n,
        train,
        queries,
        shipped_truth: vec![Vec::new(); nq],
    }
}

/// Exact top-k by dot product over the whole corpus.
fn exact_truth(d: &Data) -> Vec<Vec<u32>> {
    let t = Instant::now();
    let truth: Vec<Vec<u32>> = d
        .queries
        .par_iter()
        // A bounded heap rather than scoring into an n-long vector per query: the 1M run's
        // resident size grew by 7.5 GB across this stage -- a thousand 8 MB vectors the
        // allocator never handed back -- and the 10M version of that is what the box cannot
        // hold.
        .map(|q| top_by(0..d.n as u32, K(), |i| dot_product(q, d.row(i as usize))))
        .collect();
    let agree: usize = truth
        .iter()
        .zip(&d.shipped_truth)
        .map(|(a, b)| a.iter().filter(|x| b.contains(x)).count())
        .sum();
    println!(
        "truth: exact top-{} for {} queries in {:.1}s; overlap with shipped neighbours {:.4} {}",
        K(),
        d.queries.len(),
        t.elapsed().as_secs_f64(),
        agree as f64 / (K() * d.queries.len()) as f64,
        mem_tag()
    );
    truth
}

fn recall(found: &[Vec<u32>], truth: &[Vec<u32>]) -> f64 {
    let hit: usize = found
        .iter()
        .zip(truth)
        .map(|(f, t)| f.iter().filter(|x| t.contains(x)).count())
        .sum();
    hit as f64 / (K() * truth.len()) as f64
}

/// Smallest-first heap capped at `cap`, keyed on a score where larger is better.
#[derive(PartialEq)]
struct Cand(f32, u32);
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // Reverse so BinaryHeap::peek is the worst kept candidate.
        o.0.total_cmp(&self.0)
    }
}

fn top_by<F: Fn(u32) -> f32>(ids: impl Iterator<Item = u32>, cap: usize, score: F) -> Vec<u32> {
    let mut heap = BinaryHeap::with_capacity(cap + 1);
    for i in ids {
        let s = score(i);
        if heap.len() < cap {
            heap.push(Cand(s, i));
        } else if heap.peek().map(|w| s > w.0).unwrap_or(false) {
            heap.pop();
            heap.push(Cand(s, i));
        }
    }
    let mut v: Vec<Cand> = heap.into_vec();
    v.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    v.into_iter().map(|c| c.1).collect()
}

fn rerank(d: &Data, q: &[f32], cands: &[u32], r: usize) -> Vec<u32> {
    top_by(cands.iter().take(r).copied(), K(), |i| {
        dot_product(q, d.row(i as usize))
    })
}

/// Every `stride`-th row, the way `HnswPqIndex::bulk_build` samples its training set.
fn sample(d: &Data, want: usize) -> Vec<Vec<f32>> {
    let stride = (d.n / want).max(1);
    (0..d.n)
        .step_by(stride)
        .take(want)
        .map(|i| d.row(i).to_vec())
        .collect()
}

struct PqCodes {
    m: usize,
    codes: Vec<u8>,
}

impl PqCodes {
    fn code(&self, i: u32) -> &[u8] {
        &self.codes[i as usize * self.m..(i as usize + 1) * self.m]
    }
}

/// Train the crate's quantizer and encode the corpus.
fn build_pq(d: &Data, m: usize) -> (ProductQuantizer, PqCodes) {
    let t = Instant::now();
    let mut pq = ProductQuantizer::new(DIM, m, 256, DistanceMetric::Euclidean).unwrap();
    let training = sample(d, 65_536);
    pq.train(&training, 25).unwrap();
    let train_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    // One reused 768-float buffer per 4,096-row chunk instead of a fresh `to_vec` per row: ten
    // million 3 KB allocations churned through 32 malloc arenas is what the allocator held on
    // to when the 10M run reached 108 GB resident.
    let codes: Vec<u8> = d
        .train
        .par_chunks(DIM * 4096)
        .map(|rows| {
            let mut buf = Vec::with_capacity(DIM);
            let mut out = Vec::with_capacity(rows.len() / DIM * m);
            for row in rows.chunks_exact(DIM) {
                buf.clear();
                buf.extend_from_slice(row);
                out.extend(pq.encode(&buf).unwrap().codes);
            }
            out
        })
        .collect::<Vec<Vec<u8>>>()
        .concat();
    println!(
        "pq m={m} ({m} B/vector, {:.0}x): trained on {} in {:.1}s, encoded {} in {:.1}s {}",
        (DIM * 4) as f64 / m as f64,
        training.len(),
        train_s,
        d.n,
        t.elapsed().as_secs_f64(),
        mem_tag()
    );
    (pq, PqCodes { m, codes })
}

/// ADC table as the crate builds it (per-subspace L2, i.e. a square root per subspace) and
/// with each entry squared, which is the quantity whose sum ranks like the true distance.
fn luts(pq: &ProductQuantizer, q: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let table = pq.build_lookup_table(&q.to_vec()).unwrap().table;
    let squared = table.iter().map(|x| x * x).collect();
    (table, squared)
}

#[inline]
fn adc(lut: &[f32], code: &[u8]) -> f32 {
    let mut s = 0.0f32;
    for (sub, &c) in code.iter().enumerate() {
        s += lut[sub * 256 + c as usize];
    }
    -s
}

fn stage_pq(d: &Data, truth: &[Vec<u32>]) -> Option<(ProductQuantizer, PqCodes)> {
    let mut keep = None;
    for m in [48usize, 96, 192] {
        let (pq, codes) = build_pq(d, m);
        let t = Instant::now();
        // Per query: top-500 by corrected ADC, top-10 by the crate's ADC.
        let per_query: Vec<(Vec<u32>, Vec<u32>)> = d
            .queries
            .par_iter()
            .map(|q| {
                let (asis, sq) = luts(&pq, q);
                let corrected = top_by(0..d.n as u32, scan_cap(), |i| adc(&sq, codes.code(i)));
                let crate_lut = top_by(0..d.n as u32, K(), |i| adc(&asis, codes.code(i)));
                (corrected, crate_lut)
            })
            .collect();
        let scan_ms = t.elapsed().as_secs_f64() * 1000.0 / d.queries.len() as f64
            * rayon::current_num_threads() as f64;
        let no_rerank: Vec<Vec<u32>> = per_query.iter().map(|(c, _)| c[..K()].to_vec()).collect();
        let crate_lut: Vec<Vec<u32>> = per_query.iter().map(|(_, c)| c.clone()).collect();
        print!(
            "  full scan {:.1} ms/query/thread; recall crate-lut {:.4}, squared-lut {:.4}; rerank",
            scan_ms,
            recall(&crate_lut, truth),
            recall(&no_rerank, truth)
        );
        for r in rerank_at() {
            let reranked: Vec<Vec<u32>> = d
                .queries
                .par_iter()
                .zip(&per_query)
                .map(|(q, (c, _))| rerank(d, q, c, r))
                .collect();
            print!(" R{r}={:.4}", recall(&reranked, truth));
        }
        println!(" {}", mem_tag());
        if m == walk_m() {
            keep = Some((pq, codes));
        }
    }
    keep
}

/// 8-bit scalar quantization with one global range, the shape Qdrant ships by default.
fn stage_sq(d: &Data, truth: &[Vec<u32>]) {
    let t = Instant::now();
    let (lo, hi) = d
        .train
        .par_iter()
        .fold(
            || (f32::MAX, f32::MIN),
            |(lo, hi), &x| (lo.min(x), hi.max(x)),
        )
        .reduce(|| (f32::MAX, f32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));
    let step = (hi - lo) / 255.0;
    let codes: Vec<u8> = d
        .train
        .par_iter()
        .map(|&x| ((x - lo) / step).round().clamp(0.0, 255.0) as u8)
        .collect();
    println!(
        "sq8 ({} B/vector, 4x): range [{lo:.3}, {hi:.3}], encoded in {:.1}s",
        DIM,
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let per_query: Vec<Vec<u32>> = d
        .queries
        .par_iter()
        .map(|q| {
            // dot(q, lo + step*c) = lo*sum(q) + step*dot(q, c); the constant does not rank.
            top_by(0..d.n as u32, scan_cap(), |i| {
                let c = &codes[i as usize * DIM..(i as usize + 1) * DIM];
                q.iter().zip(c).map(|(a, &b)| a * b as f32).sum::<f32>()
            })
        })
        .collect();
    let scan_ms = t.elapsed().as_secs_f64() * 1000.0 / d.queries.len() as f64
        * rayon::current_num_threads() as f64;
    let no_rerank: Vec<Vec<u32>> = per_query.iter().map(|c| c[..K()].to_vec()).collect();
    print!(
        "  full scan {:.1} ms/query/thread; recall {:.4}; rerank",
        scan_ms,
        recall(&no_rerank, truth)
    );
    for r in rerank_at() {
        let reranked: Vec<Vec<u32>> = d
            .queries
            .par_iter()
            .zip(&per_query)
            .map(|(q, c)| rerank(d, q, c, r))
            .collect();
        print!(" R{r}={:.4}", recall(&reranked, truth));
    }
    println!(" {}", mem_tag());
}

/// Spherical k-means on a sample, then assign the whole corpus.
fn build_ivf(d: &Data, nlist: usize) -> (Vec<Vec<f32>>, Vec<Vec<u32>>) {
    let t = Instant::now();
    let training = sample(d, (nlist * 64).min(d.n));
    let mut centroids: Vec<Vec<f32>> = training
        .iter()
        .step_by(training.len() / nlist)
        .take(nlist)
        .cloned()
        .collect();
    let nearest = |v: &[f32], cs: &[Vec<f32>]| -> usize {
        let mut best = (f32::MIN, 0);
        for (j, c) in cs.iter().enumerate() {
            let s = dot_product(v, c);
            if s > best.0 {
                best = (s, j);
            }
        }
        best.1
    };
    for _ in 0..20 {
        let assign: Vec<usize> = training
            .par_iter()
            .map(|v| nearest(v, &centroids))
            .collect();
        let mut sums = vec![vec![0.0f32; DIM]; nlist];
        let mut counts = vec![0usize; nlist];
        for (v, &a) in training.iter().zip(&assign) {
            for (s, x) in sums[a].iter_mut().zip(v) {
                *s += x;
            }
            counts[a] += 1;
        }
        for (j, s) in sums.iter_mut().enumerate() {
            if counts[j] > 0 {
                normalize(s);
                centroids[j] = s.clone();
            }
        }
    }
    let train_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let assign: Vec<u32> = (0..d.n)
        .into_par_iter()
        .map(|i| nearest(d.row(i), &centroids) as u32)
        .collect();
    let mut lists = vec![Vec::new(); nlist];
    for (i, &a) in assign.iter().enumerate() {
        lists[a as usize].push(i as u32);
    }
    let mut sizes: Vec<usize> = lists.iter().map(|l| l.len()).collect();
    sizes.sort_unstable();
    println!(
        "ivf nlist={nlist}: k-means on {} in {:.1}s, assigned {} in {:.1}s; list size min {} median {} max {} empty {}",
        training.len(),
        train_s,
        d.n,
        t.elapsed().as_secs_f64(),
        sizes[0],
        sizes[nlist / 2],
        sizes[nlist - 1],
        sizes.iter().filter(|&&s| s == 0).count()
    );
    (centroids, lists)
}

fn stage_ivf(
    d: &Data,
    truth: &[Vec<u32>],
    pq: Option<&PqCodes>,
    pq_model: Option<&ProductQuantizer>,
) {
    for nlist in [1024usize, 4096] {
        let (centroids, lists) = build_ivf(d, nlist);
        for nprobe in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            if nprobe > nlist {
                break;
            }
            let t = Instant::now();
            let per_query: Vec<(Vec<u32>, usize)> = d
                .queries
                .par_iter()
                .map(|q| {
                    let probe = top_by(0..nlist as u32, nprobe, |j| {
                        dot_product(q, &centroids[j as usize])
                    });
                    let ids = probe
                        .iter()
                        .flat_map(|&j| lists[j as usize].iter().copied());
                    let scanned: usize = probe.iter().map(|&j| lists[j as usize].len()).sum();
                    (
                        top_by(ids, K(), |i| dot_product(q, d.row(i as usize))),
                        scanned,
                    )
                })
                .collect();
            let flat_ms = t.elapsed().as_secs_f64() * 1000.0 / d.queries.len() as f64
                * rayon::current_num_threads() as f64;
            let flat: Vec<Vec<u32>> = per_query.iter().map(|(f, _)| f.clone()).collect();
            let scanned =
                per_query.iter().map(|(_, s)| *s).sum::<usize>() as f64 / per_query.len() as f64;
            print!(
                "  nprobe={nprobe:<3} scanned {:>7.0} ({:.2}%) flat: recall {:.4} {:.2} ms/thread",
                scanned,
                100.0 * scanned / d.n as f64,
                recall(&flat, truth),
                flat_ms
            );
            if let (Some(codes), Some(model)) = (pq, pq_model) {
                let t = Instant::now();
                let with_pq: Vec<Vec<Vec<u32>>> = d
                    .queries
                    .par_iter()
                    .map(|q| {
                        let (_, sq) = luts(model, q);
                        let probe = top_by(0..nlist as u32, nprobe, |j| {
                            dot_product(q, &centroids[j as usize])
                        });
                        let ids = probe
                            .iter()
                            .flat_map(|&j| lists[j as usize].iter().copied());
                        let cands = top_by(ids, scan_cap(), |i| adc(&sq, codes.code(i)));
                        rerank_at()
                            .iter()
                            .map(|&r| rerank(d, q, &cands, r))
                            .collect()
                    })
                    .collect();
                let pq_ms = t.elapsed().as_secs_f64() * 1000.0 / d.queries.len() as f64
                    * rayon::current_num_threads() as f64;
                print!(" | pq{}+rerank {:.2} ms/thread:", walk_m(), pq_ms);
                for (ri, r) in rerank_at().iter().enumerate() {
                    let found: Vec<Vec<u32>> = with_pq.iter().map(|v| v[ri].clone()).collect();
                    print!(" R{r}={:.4}", recall(&found, truth));
                }
            }
            println!(" {}", mem_tag());
        }
    }
}

/// Peak resident size so far, so a log shows which stage set it.
fn hwm_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|k| k.parse::<f64>().ok())
            })
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn mem_tag() -> String {
    format!("[rss {:.0} MB, peak {:.0} MB]", rss_mb(), hwm_mb())
}

fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|p| p.parse::<f64>().ok())
        })
        .map(|pages| pages * 4096.0 / 1_048_576.0)
        .unwrap_or(0.0)
}

/// `build=decoded` builds the graph from vectors reconstructed out of their PQ codes instead
/// of the originals. With the full vectors on disk a 100M build cannot score every candidate
/// neighbour exactly, so the graph would be built on approximate distances; this is the
/// cheapest stand-in for that (symmetric PQ distance, slightly worse than ADC) that needs no
/// change to the insert path.
fn stage_hnsw(
    d: &Data,
    truth: &[Vec<u32>],
    pq96: Option<&(ProductQuantizer, PqCodes)>,
    decoded_build: bool,
    quantize_graph: bool,
    quantize_at: usize,
) {
    use dxmt_vectordb::search::quantization::product_quantization::PQCode;
    let before = rss_mb();
    let t = Instant::now();
    let mut index = HnswIndex::new(
        DistanceMetric::Cosine,
        Some(HnswConfig::from_params(16, 200, 256)),
    );
    // `quantize_at=N`: the server's streaming build -- exact until N rows, then the graph
    // is quantized and every later row is planned with ADC against the codes and pruned
    // with the code-to-code table. The 10M server run lost five points of recall against
    // the probe's exact-then-quantize build; this reproduces the server's build in-process.
    let mut quantized_during_build = false;
    let mut start = 0;
    while start < d.n {
        if quantize_at > 0 && !quantized_during_build && start >= quantize_at {
            let Some((pq, _)) = pq96 else {
                panic!("quantize_at needs the pq stage's model");
            };
            let t = Instant::now();
            index.quantize(pq.clone(), None).unwrap();
            quantized_during_build = true;
            println!(
                "quantized the graph at {start} rows in {:.1}s; the remaining {} rows build on codes",
                t.elapsed().as_secs_f64(),
                d.n - start
            );
        }
        let mut end = (start + 50_000).min(d.n);
        if quantize_at > 0 && start < quantize_at {
            end = end.min(quantize_at); // quantize exactly at the requested row
        }
        let chunk: Vec<(String, Vec<f32>)> = (start..end)
            .map(|i| {
                let v = match (decoded_build, pq96) {
                    (true, Some((pq, codes))) => pq
                        .decode(&PQCode {
                            codes: codes.code(i as u32).to_vec(),
                        })
                        .unwrap(),
                    _ => d.row(i).to_vec(),
                };
                (i.to_string(), v)
            })
            .collect();
        index.insert_batch(chunk).unwrap();
        start = end;
    }
    println!(
        "hnsw m=16 ef_c=200 built from {} vectors: built {} in {:.1}s, rss +{:.0} MB ({:.0} B/vector incl. the {} B vector; graph accounts for {:.0} B/vector)",
        if decoded_build { "PQ-decoded" } else { "exact" },
        d.n,
        t.elapsed().as_secs_f64(),
        rss_mb() - before,
        (rss_mb() - before) * 1_048_576.0 / d.n as f64,
        DIM * 4,
        index.resident_bytes_without_vectors() as f64 / d.n as f64
    );
    for ef in [32usize, 64, 128, 256, 512, 1024, 2048] {
        if quantized_during_build {
            break; // no exact walk to measure: the graph already holds codes
        }
        let t = Instant::now();
        let found: Vec<Vec<u32>> = d
            .queries
            .par_iter()
            .map(|q| {
                index
                    .search_with_ef(q, K(), ef)
                    .unwrap()
                    .into_iter()
                    .map(|(id, _)| id.parse::<u32>().unwrap())
                    .collect()
            })
            .collect();
        let par_ms = t.elapsed().as_secs_f64() * 1000.0 / d.queries.len() as f64;
        let t = Instant::now();
        for q in d.queries.iter().take(200) {
            index.search_with_ef(q, K(), ef).unwrap();
        }
        let single_ms = t.elapsed().as_secs_f64() * 1000.0 / 200.0;
        println!(
            "  ef={ef:<3} recall {:.4}  {:.3} ms/query single-thread, {:.3} ms/query wall over {} threads",
            recall(&found, truth),
            single_ms,
            par_ms,
            rayon::current_num_threads()
        );
    }
    // The graph's own quantized path (`HnswIndex::quantize` + `search_with_ef`), timed as
    // the server runs it, with the codes' size from `pq=`. This is the number the SIMD
    // ADC work is judged on.
    if quantize_graph || quantized_during_build {
        let Some((pq, _)) = pq96 else {
            return;
        };
        let t = Instant::now();
        if !quantized_during_build {
            index.quantize(pq.clone(), None).unwrap();
        }
        println!(
            "quantized the graph in {:.1}s; graph accounts for {:.0} B/vector {}",
            t.elapsed().as_secs_f64(),
            index.resident_bytes_without_vectors() as f64 / d.n as f64,
            mem_tag()
        );
        for ef in [256usize, 512, 1024, 2048] {
            let r = 500.min(ef);
            let t = Instant::now();
            let walks: Vec<Vec<u32>> = d
                .queries
                .par_iter()
                .map(|q| {
                    index
                        .search_with_ef(q, r, ef)
                        .unwrap()
                        .into_iter()
                        .map(|(id, _)| id.parse::<u32>().unwrap())
                        .collect()
                })
                .collect();
            let par_ms = t.elapsed().as_secs_f64() * 1000.0 / d.queries.len() as f64;
            let t = Instant::now();
            for q in d.queries.iter().take(200) {
                index.search_with_ef(q, r, ef).unwrap();
            }
            let single_ms = t.elapsed().as_secs_f64() * 1000.0 / 200.0;
            let reranked: Vec<Vec<u32>> = d
                .queries
                .par_iter()
                .zip(&walks)
                .map(|(q, w)| rerank(d, q, w, r))
                .collect();
            println!(
                "  quantized-graph ef={ef:<4} R{r}: recall {:.4}  {:.3} ms/query single-thread, {:.3} ms/query wall over {} threads",
                recall(&reranked, truth),
                single_ms,
                par_ms,
                rayon::current_num_threads()
            );
        }
        return;
    }

    // Route C as it would actually run: the graph walked on ADC over 96-byte codes with the
    // full vectors consulted only to rerank the walk's best `R`. The PQ stage's full-scan
    // rerank numbers are this measurement's upper bound; the gap between the two is what the
    // walk loses by being steered by approximate distances.
    let Some((pq, codes)) = pq96 else {
        return;
    };
    let row_of = |idx: usize| index.node_id(idx).parse::<u32>().unwrap();
    for ef in [64usize, 128, 256, 512, 1024, 2048] {
        let t = Instant::now();
        let walks: Vec<Vec<u32>> = d
            .queries
            .par_iter()
            .map(|q| {
                let (_, sq) = luts(pq, q);
                index
                    .search_by_score(|idx| -adc(&sq, codes.code(row_of(idx))), ef)
                    .into_iter()
                    .map(|(idx, _)| row_of(idx))
                    .collect()
            })
            .collect();
        let walk_ms = t.elapsed().as_secs_f64() * 1000.0 / d.queries.len() as f64
            * rayon::current_num_threads() as f64;
        let no_rerank: Vec<Vec<u32>> = walks
            .iter()
            .map(|w| w[..K().min(w.len())].to_vec())
            .collect();
        print!(
            "  pq{}-walk ef={ef:<4} {:.3} ms/query/thread; recall no-rerank {:.4}; rerank",
            walk_m(),
            walk_ms,
            recall(&no_rerank, truth)
        );
        for r in rerank_at() {
            if r > ef {
                break;
            }
            let reranked: Vec<Vec<u32>> = d
                .queries
                .par_iter()
                .zip(&walks)
                .map(|(q, w)| rerank(d, q, w, r))
                .collect();
            print!(" R{r}={:.4}", recall(&reranked, truth));
        }
        println!(" {}", mem_tag());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| "synthetic".into());
    let mut stages: HashSet<String> = ["truth", "pq", "sq", "ivf", "hnsw"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut n = 1_000_000;
    let mut nq = 1000;
    let mut decoded_build = false;
    let mut quantize_graph = false;
    let mut quantize_at = 0usize;
    for a in &args[2..] {
        if let Some(v) = a.strip_prefix("stages=") {
            stages = v.split(',').map(|s| s.to_string()).collect();
        } else if let Some(v) = a.strip_prefix("n=") {
            n = v.parse().unwrap();
        } else if let Some(v) = a.strip_prefix("queries=") {
            nq = v.parse().unwrap();
        } else if let Some(v) = a.strip_prefix("k=") {
            K_OF.set(v.parse().unwrap()).unwrap();
        } else if let Some(v) = a.strip_prefix("pq=") {
            WALK_M.set(v.parse().unwrap()).unwrap();
        } else if a == "build=decoded" {
            decoded_build = true;
        } else if a == "quantize=1" {
            quantize_graph = true;
        } else if let Some(v) = a.strip_prefix("quantize_at=") {
            quantize_at = v.parse().unwrap();
        }
    }
    if dir == "synthetic" && n == 1_000_000 {
        n = 20_000;
        nq = 100;
    }
    println!(
        "route_probe: {} threads, recall@{}, rss {:.0} MB",
        rayon::current_num_threads(),
        K(),
        rss_mb()
    );
    let d = load(&dir, n, nq);
    let truth = exact_truth(&d);
    let mut pq96 = if stages.contains("pq") {
        stage_pq(&d, &truth)
    } else {
        None
    };
    if pq96.is_none() && (stages.contains("hnsw") || stages.contains("ivf")) {
        pq96 = Some(build_pq(&d, walk_m()));
    }
    if stages.contains("sq") {
        stage_sq(&d, &truth);
    }
    if stages.contains("ivf") {
        stage_ivf(
            &d,
            &truth,
            pq96.as_ref().map(|p| &p.1),
            pq96.as_ref().map(|p| &p.0),
        );
    }
    if stages.contains("hnsw") {
        stage_hnsw(
            &d,
            &truth,
            pq96.as_ref(),
            decoded_build,
            quantize_graph,
            quantize_at,
        );
    }
    println!("done {}", mem_tag());
}
