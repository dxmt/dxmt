//! The big-ann-benchmarks filtered track (NeurIPS'23): YFCC-10M, 192-d CLIP embeddings as
//! `u8`, every row carrying a set of integer tags, 100K queries each with one or two tags
//! that a result must all carry, and a published ground truth (k=10, euclidean). The
//! track's score is QPS at recall@10 >= 0.9 on the full 10M; a subset has no published
//! truth, so below 10M this scores against the table's own exact scan instead.
//!
//! Ignored: needs the track's files in `DXMT_YFCC_DIR` (`base.10M.u8bin`,
//! `base.metadata.10M.spmat`, `query.public.100K.u8bin`,
//! `query.metadata.public.100K.spmat`, `GT.public.ibin`). `DXMT_YFCC_ROWS` (default all),
//! `DXMT_YFCC_QUERIES` (default 1,000), `DXMT_YFCC_DIR_WAL` (default a temp dir: rows go to
//! row files, as on a server).
//!
//!   DXMT_YFCC_DIR=/data/yfcc cargo test --profile release-unwind --test yfcc_filtered_test \
//!     -- --ignored --nocapture

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{
            Condition, CreateRecordRequest, Filter, FilterOperation, SearchMode, SearchQuery,
        },
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

const K: usize = 10;

fn read_all(path: &Path) -> Vec<u8> {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut v = Vec::new();
    f.read_to_end(&mut v).unwrap();
    v
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn i32_at(b: &[u8], at: usize) -> i32 {
    i32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn i64_at(b: &[u8], at: usize) -> i64 {
    i64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// `.u8bin`: `n: u32, d: u32`, then `n * d` bytes.
fn read_u8bin(path: &Path, limit: Option<usize>) -> (usize, usize, Vec<u8>) {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut head = [0u8; 8];
    f.read_exact(&mut head).unwrap();
    let n = u32_at(&head, 0) as usize;
    let d = u32_at(&head, 4) as usize;
    let n = limit.map_or(n, |l| l.min(n));
    let mut data = vec![0u8; n * d];
    f.read_exact(&mut data).unwrap();
    (n, d, data)
}

/// `.spmat`: CSR, `rows: i64, cols: i64, nnz: i64`, `indptr: i64[rows+1]`,
/// `indices: i32[nnz]`, `data: f32[nnz]`. The tags of row `i` are `indices[indptr[i]..indptr[i+1]]`.
fn read_spmat(path: &Path, limit: Option<usize>) -> Vec<Vec<i64>> {
    let b = read_all(path);
    let rows = i64_at(&b, 0) as usize;
    let nnz = i64_at(&b, 16) as usize;
    let indptr_at = 24;
    let indices_at = indptr_at + 8 * (rows + 1);
    assert!(
        indices_at + 4 * nnz <= b.len(),
        "{} is shorter than its header says",
        path.display()
    );
    let rows = limit.map_or(rows, |l| l.min(rows));
    (0..rows)
        .map(|i| {
            let s = i64_at(&b, indptr_at + 8 * i) as usize;
            let e = i64_at(&b, indptr_at + 8 * (i + 1)) as usize;
            (s..e)
                .map(|j| i32_at(&b, indices_at + 4 * j) as i64)
                .collect()
        })
        .collect()
}

/// `.ibin` ground truth: `nq: u32, k: u32`, `ids: i32[nq*k]`, `dists: f32[nq*k]`.
fn read_gt(path: &Path, limit: usize) -> Vec<Vec<i32>> {
    let b = read_all(path);
    let nq = u32_at(&b, 0) as usize;
    let k = u32_at(&b, 4) as usize;
    (0..limit.min(nq))
        .map(|q| {
            (0..k.min(K))
                .map(|j| i32_at(&b, 8 + 4 * (q * k + j)))
                .collect()
        })
        .collect()
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

fn env(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn tag_filter(tags: &[i64]) -> Filter {
    Filter {
        must: Some(
            tags.iter()
                .map(|&t| Condition {
                    key: "tags".to_string(),
                    operation: FilterOperation::Equal(serde_json::json!(t)),
                })
                .collect(),
        ),
        must_not: None,
        should: None,
    }
}

fn query(v: Vec<f32>, filter: Filter, mode: SearchMode) -> SearchQuery {
    SearchQuery {
        vector: v,
        top_k: K,
        limit: K,
        filter: Some(filter),
        search_mode: mode,
        with_payload: false,
        ..Default::default()
    }
}

fn recall(got: &[String], truth: &[String]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    got.iter().filter(|g| truth.contains(g)).count() as f32 / truth.len() as f32
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn yfcc_filtered_track() {
    if std::env::var("DXMT_YFCC_LOG").is_ok() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("dxmt_vectordb=info")
            .with_writer(std::io::stderr)
            .try_init();
    }
    let dir = std::path::PathBuf::from(std::env::var("DXMT_YFCC_DIR").expect("DXMT_YFCC_DIR"));
    let limit = env("DXMT_YFCC_ROWS");
    let nq = env("DXMT_YFCC_QUERIES").unwrap_or(1_000);

    let (n, dim, base) = read_u8bin(&dir.join("base.10M.u8bin"), limit);
    let tags = read_spmat(&dir.join("base.metadata.10M.spmat"), Some(n));
    let (qn, qd, qvecs) = read_u8bin(&dir.join("query.public.100K.u8bin"), Some(nq));
    let qtags = read_spmat(&dir.join("query.metadata.public.100K.spmat"), Some(qn));
    assert_eq!(dim, qd);
    let full = limit.is_none_or(|l| l >= 10_000_000);
    let gt = if full {
        read_gt(&dir.join("GT.public.ibin"), qn)
    } else {
        Vec::new()
    };
    let tags_per_row = tags.iter().map(Vec::len).sum::<usize>() as f64 / n as f64;
    println!(
        "yfcc: {n} x {dim}, {tags_per_row:.1} tags per row; {qn} queries, {} with two tags; truth: {}",
        qtags.iter().filter(|t| t.len() == 2).count(),
        if full { "published GT" } else { "exact scan (subset)" }
    );

    let wal = std::env::var("DXMT_YFCC_DIR_WAL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("dxmt-yfcc-{}", std::process::id())));
    let storage = Arc::new(
        IntelligentIndexedStorage::new_with_wal(
            &StorageConfig {
                memory: Some(MemoryConfig {
                    max_size_mb: 65_536,
                    eviction_policy: "lru".to_string(),
                }),
                ssd: None,
                object_storage: None,
            },
            Some(wal.clone()),
        )
        .await
        .unwrap(),
    );
    // A WAL directory that already holds the table (a previous run with DXMT_YFCC_DIR_WAL)
    // is reopened as it is: the load is the long part, and the snapshot taken after it makes
    // the reopen seconds rather than minutes.
    let already = storage
        .get_table("t")
        .await
        .ok()
        .map(|t| t.record_count as usize);
    if already == Some(n) {
        println!("reopened {n} rows from {}", wal.display());
    } else {
        storage
            .create_table(TableConfig {
                name: "t".to_string(),
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
                quantization: None,
                payload_index: None,
                embedding: None,
            })
            .await
            .unwrap();

        let started = std::time::Instant::now();
        const BATCH: usize = 5_000;
        for start in (0..n).step_by(BATCH) {
            let end = (start + BATCH).min(n);
            let batch = (start..end)
                .map(|i| CreateRecordRequest {
                    id: i.to_string(),
                    vector: base[i * dim..(i + 1) * dim]
                        .iter()
                        .map(|&b| b as f32)
                        .collect(),
                    sparse_vector: None,
                    named_vectors: None,
                    payload: Some(serde_json::json!({ "tags": tags[i] })),
                    text: None,
                })
                .collect();
            storage.insert_records("t", batch).await.unwrap();
            if start > 0 && start % 1_000_000 == 0 {
                println!(
                    "  {start} rows, {:.0} rows/s",
                    start as f64 / started.elapsed().as_secs_f64()
                );
            }
        }
        println!(
            "loaded {n} rows in {:.0}s ({:.0} rows/s)",
            started.elapsed().as_secs_f64(),
            n as f64 / started.elapsed().as_secs_f64()
        );
        if std::env::var("DXMT_YFCC_DIR_WAL").is_ok() {
            let t = std::time::Instant::now();
            storage.snapshot().await.unwrap();
            println!("snapshot for reopening: {:.0}s", t.elapsed().as_secs_f64());
        }
    }
    let waited = std::time::Instant::now();
    loop {
        let stats = storage.get_table_stats("t").await.unwrap();
        if find_key(&stats, "quantized").and_then(|q| q.as_bool()) == Some(true) || n <= 65_536 {
            break;
        }
        assert!(
            waited.elapsed().as_secs() < 3_600,
            "not quantized after an hour: {stats}"
        );
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    }
    println!(
        "quantized after a further {:.0}s",
        waited.elapsed().as_secs_f64()
    );
    if std::env::var("DXMT_YFCC_WARM").as_deref() == Ok("1") {
        let t = std::time::Instant::now();
        let warmed = storage.warm_derived("t").await.unwrap();
        println!(
            "warm_derived: {warmed} in {:.1}s",
            t.elapsed().as_secs_f64()
        );
    }
    if std::env::var("DXMT_YFCC_FAST_SCAN").as_deref() == Ok("1") {
        let t = std::time::Instant::now();
        let built = storage.ensure_fast_scan("t").await.unwrap();
        let stats = storage.get_table_stats("t").await.unwrap();
        println!(
            "fast-scan: built {built} in {:.1}s; stats say {:?}",
            t.elapsed().as_secs_f64(),
            find_key(&stats, "fast_scan")
        );
    }

    let qvec = |q: usize| -> Vec<f32> {
        qvecs[q * dim..(q + 1) * dim]
            .iter()
            .map(|&b| b as f32)
            .collect()
    };
    let ids = |r: dxmt_vectordb::types::record::SearchResult| -> Vec<String> {
        r.hits.into_iter().map(|h| h.id).collect()
    };

    // The product's cold start: a `normal` filtered query, which builds the payload index
    // if the warm-up has not, and reads a rerank pool of rows -- not the exact scan below,
    // which reads the whole row file and measures page cache, not the index.
    let t0 = std::time::Instant::now();
    let _ = storage
        .search(
            "t",
            query(qvec(0), tag_filter(&qtags[0]), SearchMode::Normal),
        )
        .await
        .unwrap();
    println!(
        "first filtered query at normal (payload index build if not warmed): {:.2}s",
        t0.elapsed().as_secs_f64()
    );
    let mut truth: Vec<Vec<String>> = Vec::with_capacity(qn);
    let exact_started = std::time::Instant::now();
    for q in 0..qn {
        if full {
            truth.push(
                gt[q]
                    .iter()
                    .filter(|&&i| i >= 0)
                    .map(|i| i.to_string())
                    .collect(),
            );
        } else {
            let r = storage
                .search(
                    "t",
                    query(qvec(q), tag_filter(&qtags[q]), SearchMode::Exact),
                )
                .await
                .unwrap();
            truth.push(ids(r));
        }
    }
    if !full {
        println!(
            "exact filtered scan, {qn} queries: p50 {:.1} ms",
            exact_started.elapsed().as_secs_f64() * 1e3 / qn as f64
        );
    }

    // Where the time goes at `normal`, by branch and by how many rows the filter matches:
    // each query's branch is read off the counters, its match count from `count_records`.
    {
        let buckets = ["<2K", "2K-10K", "10K-100K", "100K-500K", ">=500K"];
        let bucket = |m: usize| match m {
            0..=1_999 => 0,
            2_000..=9_999 => 1,
            10_000..=99_999 => 2,
            100_000..=499_999 => 3,
            _ => 4,
        };
        let names = [
            "exact_scan",
            "code_scan",
            "in_graph",
            "pre_filter",
            "post_filter",
        ];
        /// Latencies, total matched, total recall, total evaluation ns of one cell.
        type Cell = (Vec<f64>, usize, f32, u64);
        let mut cells: std::collections::BTreeMap<(usize, usize), Cell> =
            std::collections::BTreeMap::new();
        let start = storage.health_snapshot().await.filter_paths;
        for q in 0..qn {
            let f = tag_filter(&qtags[q]);
            let matched = storage.count_records("t", Some(&f)).await.unwrap();
            let before = storage.health_snapshot().await.filter_paths;
            let t = std::time::Instant::now();
            let got = ids(storage
                .search("t", query(qvec(q), f, SearchMode::Normal))
                .await
                .unwrap());
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let after = storage.health_snapshot().await.filter_paths;
            let took = [
                after.exact_scan - before.exact_scan,
                after.code_scan - before.code_scan,
                after.in_graph - before.in_graph,
                after.pre_filter - before.pre_filter,
                after.post_filter - before.post_filter,
            ];
            let branch = took.iter().position(|&d| d > 0).unwrap_or(names.len());
            let cell = cells
                .entry((branch, bucket(matched)))
                .or_insert((Vec::new(), 0, 0.0, 0));
            cell.0.push(ms);
            cell.1 += matched;
            cell.2 += recall(&got, &truth[q]);
            cell.3 += after.eval_ns - before.eval_ns;
        }
        let end = storage.health_snapshot().await.filter_paths;
        println!("\n`normal`, by branch and match set (recall@10, p50, p95, mean matched):");
        println!("| branch | matched | queries | recall | p50 | p95 | mean matched | mean eval |");
        println!("| --- | --- | --- | --- | --- | --- | --- | --- |");
        for ((branch, b), (mut lat, matched, r, eval_ns)) in cells {
            lat.sort_by(|a, c| a.partial_cmp(c).unwrap());
            let n = lat.len();
            println!(
                "| {} | {} | {n} | {:.4} | {:.2} ms | {:.2} ms | {} | {:.2} ms |",
                names.get(branch).copied().unwrap_or("none"),
                buckets[b],
                r / n as f32,
                lat[n / 2],
                lat[n * 95 / 100],
                matched / n,
                eval_ns as f64 / 1e6 / n as f64
            );
        }
        let total_ns = (end.exact_scan_ns
            + end.code_scan_ns
            + end.in_graph_ns
            + end.pre_filter_ns
            + end.post_filter_ns)
            - (start.exact_scan_ns
                + start.code_scan_ns
                + start.in_graph_ns
                + start.pre_filter_ns
                + start.post_filter_ns);
        println!(
            "node map misses: {}",
            end.node_map_misses - start.node_map_misses
        );
        println!(
            "filter evaluation (estimate + id materialisation): {:.1}% of branch time; per branch ms: exact {:.1} code {:.1} in_graph {:.1} pre {:.1} post {:.1}",
            100.0 * (end.eval_ns - start.eval_ns) as f64 / total_ns.max(1) as f64,
            (end.exact_scan_ns - start.exact_scan_ns) as f64 / 1e6,
            (end.code_scan_ns - start.code_scan_ns) as f64 / 1e6,
            (end.in_graph_ns - start.in_graph_ns) as f64 / 1e6,
            (end.pre_filter_ns - start.pre_filter_ns) as f64 / 1e6,
            (end.post_filter_ns - start.post_filter_ns) as f64 / 1e6,
        );
    }

    let before = storage.health_snapshot().await.filter_paths;
    println!("\n| mode | recall@10 | p50 | p95 | 1-thread QPS | 16-thread QPS |");
    println!("| --- | --- | --- | --- | --- | --- |");
    for mode in [
        SearchMode::Fast,
        SearchMode::Normal,
        SearchMode::Accurate,
        SearchMode::Exact,
    ] {
        // On the full set the exact mode scores every row of 10M per query; 50 of them are
        // a check that the published truth and this table agree, not a benchmark.
        let qn = if full && mode == SearchMode::Exact {
            qn.min(50)
        } else {
            qn
        };
        let mut r = 0f32;
        let mut lat = Vec::with_capacity(qn);
        let serial = std::time::Instant::now();
        for q in 0..qn {
            let t = std::time::Instant::now();
            let got = ids(storage
                .search("t", query(qvec(q), tag_filter(&qtags[q]), mode))
                .await
                .unwrap());
            lat.push(t.elapsed().as_secs_f64() * 1e3);
            r += recall(&got, &truth[q]);
        }
        let serial_qps = qn as f64 / serial.elapsed().as_secs_f64();
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Concurrent: the same queries over 16 tasks.
        let par = std::time::Instant::now();
        let mut handles = Vec::new();
        for w in 0..16usize {
            let storage = Arc::clone(&storage);
            let qs: Vec<(Vec<f32>, Filter)> = (w..qn)
                .step_by(16)
                .map(|q| (qvec(q), tag_filter(&qtags[q])))
                .collect();
            handles.push(tokio::spawn(async move {
                for (v, f) in qs {
                    let _ = storage.search("t", query(v, f, mode)).await.unwrap();
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let par_qps = qn as f64 / par.elapsed().as_secs_f64();
        println!(
            "| {mode:?} ({qn} queries) | {:.4} | {:.2} ms | {:.2} ms | {serial_qps:.0} | {par_qps:.0} |",
            r / qn as f32,
            lat[lat.len() / 2],
            lat[lat.len() * 95 / 100]
        );
    }
    let after = storage.health_snapshot().await.filter_paths;
    println!(
        "\nfilter branches taken: exact_scan {} code_scan {} in_graph {} pre_filter {} post_filter {}",
        after.exact_scan - before.exact_scan,
        after.code_scan - before.code_scan,
        after.in_graph - before.in_graph,
        after.pre_filter - before.pre_filter,
        after.post_filter - before.post_filter
    );
    if std::env::var("DXMT_YFCC_DIR_WAL").is_err() {
        drop(storage);
        let _ = std::fs::remove_dir_all(&wal);
    }
}
