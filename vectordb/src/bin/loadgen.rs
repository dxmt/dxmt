//! A load generator cheap enough not to be the thing under test.
//!
//! The benchmark harness and the server share one machine, and a client that serialises 768
//! floats per request in Python is not free: sweeping the client size against the 1M table
//! showed throughput plateau at 6,690 QPS with the box 100% busy, the server pinned at 24.8
//! of 32 cores and **the load generator holding 6.7**. Every
//! server-side improvement then arrives damped -- 6% off the server was 4.7% off the total
//! and showed up as a 3% throughput gain, small enough to be mistaken for noise.
//!
//! Two things make this one cheap:
//!
//! 1. **Request bodies are built once.** The query set is fixed, so each request is a byte
//!    buffer prepared at startup and written unchanged. The server still parses the same
//!    JSON, so nothing about its work changes; the client simply stops re-doing work whose
//!    answer it already has.
//! 2. **No async runtime and no HTTP crate.** One blocking connection per thread, kept
//!    alive, and just enough of HTTP/1.1 to write a request and find the end of a response.
//!
//! It reports its own CPU alongside the throughput, so a run states plainly how much of the
//! machine it took rather than leaving that to be guessed.
//!
//! ```text
//! loadgen --url http://127.0.0.1:9999/api/v1 --table T --queries test.f32 \
//!         --dim 768 --k 100 --threads 16 --duration 30
//! ```
//!
//! Note what this does *not* do: it does not check recall. Comparisons that are published
//! should keep running on unmodified VectorDBBench, where both engines pay the same client
//! tax; this is for measuring our own changes without that tax hiding them.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Args {
    host: String,
    port: u16,
    base: String,
    table: String,
    queries: String,
    dim: usize,
    k: usize,
    mode: Option<String>,
    threads: usize,
    duration: f64,
    warmup: f64,
    limit_queries: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 9999,
            base: "/api/v1".to_string(),
            table: "VectorDBBenchCollection".to_string(),
            queries: String::new(),
            dim: 768,
            k: 100,
            mode: None,
            threads: 16,
            duration: 30.0,
            warmup: 5.0,
            limit_queries: 1000,
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "loadgen --queries <file.f32> [--url http://host:port/api/v1] [--table NAME]\n\
         \x20       [--dim 768] [--k 100] [--mode fast|normal|accurate]\n\
         \x20       [--threads 16] [--duration 30] [--warmup 5] [--query-count 1000]\n\
         \n\
         The query file is raw little-endian f32, `dim` values per query."
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
            "--url" => {
                let url = next(&mut i);
                let rest = url.strip_prefix("http://").unwrap_or_else(|| {
                    eprintln!("only http:// is supported: {url}");
                    std::process::exit(2);
                });
                let (authority, path) = match rest.find('/') {
                    Some(p) => (&rest[..p], rest[p..].trim_end_matches('/').to_string()),
                    None => (rest, String::new()),
                };
                let (h, p) = match authority.rsplit_once(':') {
                    Some((h, p)) => (h.to_string(), p.parse().unwrap_or_else(|_| usage())),
                    None => (authority.to_string(), 80u16),
                };
                a.host = h;
                a.port = p;
                a.base = path;
            }
            "--table" => a.table = next(&mut i),
            "--queries" => a.queries = next(&mut i),
            "--dim" => a.dim = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--k" => a.k = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--mode" => a.mode = Some(next(&mut i)),
            "--threads" => a.threads = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--duration" => a.duration = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--warmup" => a.warmup = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--query-count" => a.limit_queries = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown argument: {other}");
                usage();
            }
        }
        i += 1;
    }
    if a.queries.is_empty() || a.threads == 0 {
        usage();
    }
    a
}

/// Self CPU (user + system) in seconds, for reporting what the client itself cost.
///
/// `/proc/self/stat`'s second field is the command name in parentheses and may contain
/// spaces, so the fields are counted from the last `)` rather than from the start.
fn self_cpu_secs() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return f64::NAN;
    };
    let Some((_, after)) = stat.rsplit_once(')') else {
        return f64::NAN;
    };
    // After `)` comes state (index 0), so utime is index 11 and stime index 12.
    let f: Vec<&str> = after.split_whitespace().collect();
    let ticks = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (ticks(11) + ticks(12)) / 100.0
}

/// The request bytes for one query, built once.
fn build_request(a: &Args, vector: &[f32]) -> Vec<u8> {
    use std::fmt::Write as _;
    let mut body = String::with_capacity(vector.len() * 12 + 96);
    body.push_str("{\"vector\":[");
    for (i, v) in vector.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        // Same shape the SDK sends; the server parses f64 either way.
        let _ = write!(body, "{v}");
    }
    body.push_str("],\"top_k\":");
    body.push_str(&a.k.to_string());
    body.push_str(",\"limit\":");
    body.push_str(&a.k.to_string());
    body.push_str(",\"filter\":null");
    if let Some(mode) = &a.mode {
        body.push_str(",\"search_mode\":\"");
        body.push_str(mode);
        body.push('"');
    }
    body.push('}');

    let mut req = Vec::with_capacity(body.len() + 200);
    write!(
        req,
        "POST {}/tables/{}/search HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: keep-alive\r\n\r\n",
        a.base,
        a.table,
        a.host,
        a.port,
        body.len()
    )
    .expect("writing to a Vec cannot fail");
    req.extend_from_slice(body.as_bytes());
    req
}

/// Read one HTTP/1.1 response, returning its status and body length.
fn read_response(
    r: &mut BufReader<TcpStream>,
    scratch: &mut Vec<u8>,
) -> std::io::Result<(u16, usize)> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "server closed the connection",
        ));
    }
    let status: u16 = line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut len: Option<usize> = None;
    let mut chunked = false;
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            break;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            len = v.trim().parse().ok();
        } else if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            chunked = true;
        }
    }
    if chunked {
        return Err(std::io::Error::other(
            "chunked responses are not supported; this client expects Content-Length",
        ));
    }
    let n = len.unwrap_or(0);
    scratch.clear();
    scratch.resize(n, 0);
    r.read_exact(scratch)?;
    Ok((status, n))
}

fn percentile(sorted: &[u32], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx] as f64 / 1000.0
}

fn main() {
    let args = parse_args();

    let raw = std::fs::read(&args.queries).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {e}", args.queries);
        std::process::exit(1);
    });
    let floats: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    if floats.len() < args.dim {
        eprintln!(
            "{} holds {} floats, fewer than one {}-dimension query",
            args.queries,
            floats.len(),
            args.dim
        );
        std::process::exit(1);
    }
    let available = floats.len() / args.dim;
    let count = available.min(args.limit_queries).max(1);
    let requests: Arc<Vec<Vec<u8>>> = Arc::new(
        (0..count)
            .map(|i| build_request(&args, &floats[i * args.dim..(i + 1) * args.dim]))
            .collect(),
    );
    let body_bytes: usize = requests.iter().map(|r| r.len()).sum::<usize>() / count;
    println!(
        "loadgen: {} queries of {} dims, {} B a request, {} threads, {}s warmup + {}s measured, \
         mode {}",
        count,
        args.dim,
        body_bytes,
        args.threads,
        args.warmup,
        args.duration,
        args.mode.as_deref().unwrap_or("server default")
    );

    let addr = format!("{}:{}", args.host, args.port);
    let measuring = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    let started = Instant::now();

    let mut handles = Vec::with_capacity(args.threads);
    for t in 0..args.threads {
        let requests = Arc::clone(&requests);
        let addr = addr.clone();
        let measuring = Arc::clone(&measuring);
        let stop = Arc::clone(&stop);
        handles.push(std::thread::spawn(move || -> (Vec<u32>, u64, u64) {
            let stream = TcpStream::connect(&addr).unwrap_or_else(|e| {
                eprintln!("thread {t}: cannot connect to {addr}: {e}");
                std::process::exit(1);
            });
            stream.set_nodelay(true).ok();
            let mut reader = BufReader::new(stream.try_clone().expect("clone the socket"));
            let mut writer = stream;
            let mut scratch = Vec::new();
            let mut lat: Vec<u32> = Vec::with_capacity(1 << 16);
            let mut bad = 0u64;
            let mut empty = 0u64;
            let mut i = t;
            while !stop.load(Ordering::Relaxed) {
                let req = &requests[i % requests.len()];
                i += 1;
                let start = Instant::now();
                if writer.write_all(req).is_err() {
                    bad += 1;
                    break;
                }
                match read_response(&mut reader, &mut scratch) {
                    Ok((status, n)) => {
                        let took = start.elapsed().as_micros().min(u32::MAX as u128) as u32;
                        if measuring.load(Ordering::Relaxed) {
                            lat.push(took);
                            if status != 200 {
                                bad += 1;
                            }
                            // A 200 with a body too short to hold k hits is a silent
                            // mis-measurement: an empty result costs the server nothing.
                            if n < 32 {
                                empty += 1;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("thread {t}: {e}");
                        bad += 1;
                        break;
                    }
                }
            }
            (lat, bad, empty)
        }));
    }

    std::thread::sleep(Duration::from_secs_f64(args.warmup));
    let cpu_before = self_cpu_secs();
    let measure_start = Instant::now();
    measuring.store(true, Ordering::Relaxed);
    std::thread::sleep(Duration::from_secs_f64(args.duration));
    measuring.store(false, Ordering::Relaxed);
    let measured = measure_start.elapsed().as_secs_f64();
    let cpu = self_cpu_secs() - cpu_before;
    stop.store(true, Ordering::Relaxed);

    let mut lat = Vec::new();
    let (mut bad, mut empty) = (0u64, 0u64);
    for h in handles {
        let (l, b, e) = h.join().expect("a load thread panicked");
        lat.extend(l);
        bad += b;
        empty += e;
    }
    lat.sort_unstable();
    let n = lat.len();
    let qps = n as f64 / measured;
    println!(
        "qps={:.0} n={} dur={:.1}s p50={:.2} p90={:.2} p99={:.2} max={:.2} ms  \
         client_cpu={:.2} cores ({:.0} us/query)  errors={} empty={}  wall={:.1}s",
        qps,
        n,
        measured,
        percentile(&lat, 0.50),
        percentile(&lat, 0.90),
        percentile(&lat, 0.99),
        percentile(&lat, 1.0),
        cpu / measured,
        if n > 0 {
            cpu * 1e6 / n as f64
        } else {
            f64::NAN
        },
        bad,
        empty,
        started.elapsed().as_secs_f64()
    );
    if bad > 0 || empty > 0 {
        // A run that measured errors or empty results is not a throughput measurement.
        std::process::exit(1);
    }
}
