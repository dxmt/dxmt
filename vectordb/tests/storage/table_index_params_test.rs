//! Per-table graph and quantization parameters, and that they survive a restart.
//!
//! `m` and `ef_construction` decide the graph and cannot be recovered afterwards: a search can
//! look harder, but no amount of searching adds edges the build never made. They were a
//! hardcoded 16 and an environment variable, which is the wrong grain — one server holds a
//! 5,000-row lookup table beside a 10M-row corpus, and the environment cannot speak per table.
//!
//! The assertion that matters is not that the fields round-trip: it is that they reach the
//! index. `quantize_at_rows` is the one with a visible effect at a size a test can afford, so
//! it carries that weight here, in both directions — a table that quantizes early, and a table
//! that stays exact because it asked to.

use std::process::{Child, Command, Stdio};

#[path = "../common/http.rs"]
mod http;

struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

fn start(dir: &std::path::Path, port: u16) -> Server {
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DXMT_WAL_DIR", dir.join("wal"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
        .env_remove("DXMT_API_KEY")
        .env_remove("DXMT_HNSW_EF_CONSTRUCTION")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");
    let server = Server { child, port };
    for _ in 0..160 {
        if call(port, "GET", "/live", None).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

fn call(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<serde_json::Value> {
    let (_, text) = http::call(port, method, path, &[], body)?;
    serde_json::from_str(&text).ok()
}

fn fill(port: u16, table: &str, rows: usize) {
    for batch in 0..rows.div_ceil(1000) {
        let records: Vec<serde_json::Value> = (0..1000.min(rows - batch * 1000))
            .map(|i| {
                let id = batch * 1000 + i;
                // Deterministic and spread out: k-means on a single point is not a test of
                // anything, and a fixed corpus keeps the timing of this test stable.
                let vector: Vec<f32> = (0..DIM)
                    .map(|d| {
                        let mut h =
                            (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (d as u64 + 1);
                        h ^= h >> 29;
                        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        (h >> 33) as f32 / u32::MAX as f32
                    })
                    .collect();
                serde_json::json!({"id": format!("r{id}"), "vector": vector})
            })
            .collect();
        let body = serde_json::json!({ "records": records }).to_string();
        call(
            port,
            "POST",
            &format!("/api/v1/tables/{table}/records"),
            Some(&body),
        )
        .expect("insert answered");
    }
}

fn stats(port: u16, table: &str) -> serde_json::Value {
    call(port, "GET", &format!("/api/v1/tables/{table}/stats"), None).expect("stats answered")
        ["data"]
        .clone()
}

const DIM: usize = 32;
/// Above `num_centroids * 16`, which training also requires, and small enough that k-means in
/// a debug build finishes in well under a minute.
// Enough to cross the threshold below and no more. The first version of this test used
// 5,000 rows against 256 centroids, which is a real k-means: 34 s on an idle machine, and
// past the 60 s wait below once `cargo test --tests` ran the other targets alongside it.
// The parameter reaching the index is what is under test, not the size of the codebook.
const ROWS: usize = 1_000;

#[test]
fn quantize_at_rows_decides_when_a_table_leaves_the_exact_graph() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let server = start(dir.path(), port);

    // One table asks to quantize early; the other asks never to, by naming a threshold it will
    // not reach. Same data, same size, same server.
    call(
        port,
        "POST",
        "/api/v1/tables",
        Some(
            r#"{"name":"early","dimension":32,
            "quantization":{"num_subvectors":8,"num_centroids":16,"quantize_at_rows":512}}"#,
        ),
    );
    call(
        port,
        "POST",
        "/api/v1/tables",
        Some(r#"{"name":"exact","dimension":32,"quantization":{"quantize_at_rows":10000000}}"#),
    );
    fill(server.port, "early", ROWS);
    fill(server.port, "exact", ROWS);

    // Training runs off the request path, so the switch happens after the insert is answered.
    let mut quantized = false;
    for _ in 0..120 {
        if stats(server.port, "early")["quantized"] == serde_json::Value::Bool(true) {
            quantized = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    assert!(
        quantized,
        "a table that asked to quantize at 512 rows is still exact at {ROWS}; the parameter \
         did not reach the index"
    );
    assert_eq!(
        stats(server.port, "exact")["quantized"],
        serde_json::Value::Bool(false),
        "a table that asked not to quantize until 10M rows quantized at {ROWS}"
    );
}

#[test]
fn the_parameters_survive_a_restart() {
    // A rebuild after a restart goes through the same decision function as creation, so a
    // table whose parameters were not persisted would come back built to the server's
    // defaults -- silently, and only visibly as recall.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let body = r#"{"name":"tuned","dimension":32,
        "hnsw":{"m":8,"ef_construction":64,"ef_search":48},
        "quantization":{"num_subvectors":8,"num_centroids":128,"quantize_at_rows":4096}}"#;
    {
        let server = start(dir.path(), port);
        call(server.port, "POST", "/api/v1/tables", Some(body));
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
    }

    let server = start(dir.path(), port);
    let table = call(server.port, "GET", "/api/v1/tables/tuned", None).expect("table answered");
    let table = &table["data"]["table"];
    assert_eq!(table["hnsw"]["m"], 8, "{table}");
    assert_eq!(table["hnsw"]["ef_construction"], 64, "{table}");
    assert_eq!(table["hnsw"]["ef_search"], 48, "{table}");
    assert_eq!(table["quantization"]["num_subvectors"], 8, "{table}");
    assert_eq!(table["quantization"]["num_centroids"], 128, "{table}");
    assert_eq!(table["quantization"]["quantize_at_rows"], 4096, "{table}");
}

#[test]
fn a_table_without_parameters_keeps_the_defaults() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let server = start(dir.path(), port);
    call(
        port,
        "POST",
        "/api/v1/tables",
        Some(r#"{"name":"plain","dimension":32}"#),
    );
    fill(server.port, "plain", 1_000);

    let table = call(server.port, "GET", "/api/v1/tables/plain", None).expect("table answered");
    assert!(table["data"]["table"]["hnsw"].is_null(), "{table}");
    assert!(table["data"]["table"]["quantization"].is_null(), "{table}");
    // The default threshold is 65,536, so a thousand rows are still exact.
    assert_eq!(
        stats(server.port, "plain")["quantized"],
        serde_json::Value::Bool(false)
    );
}
