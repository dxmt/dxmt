//! `loadgen`, over gRPC, so the two can be compared without the client being the difference.
//!
//! The JSON generator beside this one exists because a client that serialises 768 floats per
//! request is not free, and a server-side change measured through an expensive client arrives
//! damped. The same argument applies to comparing *protocols*: putting a cheap protobuf client
//! against an expensive JSON one measures the clients. This one is built to the same rules --
//! request bodies constructed once, one connection per worker, its own CPU reported -- so that
//! the difference between the two runs is the wire format and the server's parsing of it.
//!
//! One difference that cannot be removed: tonic is async and needs a tokio runtime, where the
//! JSON generator is blocking threads with no runtime at all. A run reports client CPU so that
//! difference is visible rather than assumed away.
//!
//!     loadgen_grpc --queries q.f32 --url http://127.0.0.1:9998 --table bench \
//!                  --dim 768 --k 100 --threads 16 --duration 30

#![cfg(feature = "api-grpc")]

use dxmt_vectordb::api::grpc::proto::vector_db_client::VectorDbClient;
use dxmt_vectordb::api::grpc::proto::SearchRequest;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Args {
    url: String,
    table: String,
    queries: String,
    dim: usize,
    k: u32,
    threads: usize,
    duration: u64,
    warmup: u64,
    query_count: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:9998".to_string(),
            table: "bench".to_string(),
            queries: String::new(),
            dim: 768,
            k: 100,
            threads: 16,
            duration: 30,
            warmup: 5,
            query_count: 1000,
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "loadgen_grpc --queries <file.f32> [--url http://host:port] [--table NAME]\n\
         \x20            [--dim 768] [--k 100] [--threads 16] [--duration 30] [--warmup 5]\n\
         \x20            [--query-count 1000]\n\
         \n\
         The query file is raw little-endian f32, `dim` values per query -- the same file the\n\
         JSON loadgen takes, so the two runs ask the same questions."
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_else(|| usage())
        };
        match argv[i].as_str() {
            "--url" => a.url = next(&mut i),
            "--table" => a.table = next(&mut i),
            "--queries" => a.queries = next(&mut i),
            "--dim" => a.dim = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--k" => a.k = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--threads" => a.threads = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--duration" => a.duration = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--warmup" => a.warmup = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--query-count" => a.query_count = next(&mut i).parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
        i += 1;
    }
    if a.queries.is_empty() {
        usage();
    }
    a
}

/// Seconds of CPU this process has used, user plus system.
fn cpu_seconds() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    let fields: Vec<&str> = stat.split_whitespace().collect();
    let ticks = |i: usize| -> f64 { fields.get(i).and_then(|v| v.parse().ok()).unwrap_or(0.0) };
    // utime and stime, fields 14 and 15 (1-indexed), at the kernel's 100 Hz.
    (ticks(13) + ticks(14)) / 100.0
}

#[tokio::main]
async fn main() {
    let args = parse_args();

    let raw = std::fs::read(&args.queries).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {e}", args.queries);
        std::process::exit(2);
    });
    let floats: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if floats.len() < args.dim {
        eprintln!("{} holds fewer than {} floats", args.queries, args.dim);
        std::process::exit(2);
    }
    let count = (floats.len() / args.dim).min(args.query_count).max(1);

    // Built once, like the JSON generator's byte buffers. prost still encodes on every send,
    // which is the work being measured; what is not re-done is assembling the message.
    let requests: Arc<Vec<SearchRequest>> = Arc::new(
        (0..count)
            .map(|i| SearchRequest {
                table: args.table.clone(),
                vector: floats[i * args.dim..(i + 1) * args.dim].to_vec(),
                top_k: args.k,
                filter: None,
                with_payload: false,
                with_vector: false,
            })
            .collect(),
    );
    let encoded: usize = {
        use prost::Message;
        requests.iter().map(|r| r.encoded_len()).sum::<usize>() / count
    };

    let stop = Arc::new(AtomicBool::new(false));
    let served = Arc::new(AtomicU64::new(0));
    let failed = Arc::new(AtomicU64::new(0));
    let latencies = Arc::new(parking_lot::Mutex::new(Vec::<u64>::with_capacity(1 << 20)));

    let mut workers = Vec::with_capacity(args.threads);
    for worker in 0..args.threads {
        let (url, requests) = (args.url.clone(), Arc::clone(&requests));
        let (stop, served, failed) = (Arc::clone(&stop), Arc::clone(&served), Arc::clone(&failed));
        let latencies = Arc::clone(&latencies);
        workers.push(tokio::spawn(async move {
            let mut client = match VectorDbClient::connect(url.clone()).await {
                Ok(client) => client,
                Err(e) => {
                    eprintln!("worker {worker}: cannot connect to {url}: {e}");
                    return;
                }
            };
            let mut mine = Vec::with_capacity(1 << 16);
            let mut i = worker;
            while !stop.load(Ordering::Relaxed) {
                let request = requests[i % requests.len()].clone();
                i += 1;
                let start = Instant::now();
                match client.search(request).await {
                    Ok(_) => {
                        mine.push(start.elapsed().as_micros() as u64);
                        served.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            latencies.lock().extend(mine);
        }));
    }

    tokio::time::sleep(Duration::from_secs(args.warmup)).await;
    // Warm-up counts are thrown away so the numbers describe the steady state.
    served.store(0, Ordering::Relaxed);
    failed.store(0, Ordering::Relaxed);
    latencies.lock().clear();
    let cpu_before = cpu_seconds();
    let measured = Instant::now();

    tokio::time::sleep(Duration::from_secs(args.duration)).await;
    stop.store(true, Ordering::Relaxed);
    let elapsed = measured.elapsed().as_secs_f64();
    let cpu = cpu_seconds() - cpu_before;
    for worker in workers {
        let _ = worker.await;
    }

    let ok = served.load(Ordering::Relaxed);
    let bad = failed.load(Ordering::Relaxed);
    let mut samples = latencies.lock().clone();
    samples.sort_unstable();
    let at = |q: f64| -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        samples[((samples.len() as f64 * q) as usize).min(samples.len() - 1)] as f64 / 1000.0
    };

    println!("transport     gRPC (protobuf)");
    println!("threads       {}", args.threads);
    println!("request       {encoded} bytes encoded");
    println!("served        {ok} in {elapsed:.1}s   failed {bad}");
    println!("QPS           {:.0}", ok as f64 / elapsed);
    println!(
        "latency       p50 {:.2} ms   p95 {:.2} ms   p99 {:.2} ms",
        at(0.50),
        at(0.95),
        at(0.99)
    );
    println!(
        "client CPU    {:.2} cores ({:.0} us/query)",
        cpu / elapsed,
        cpu * 1e6 / ok.max(1) as f64
    );
}
