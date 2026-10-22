//! `serve --replica` over HTTP: two processes on one filesystem-backed object store. Writes to
//! the primary appear on the replica; a write to the replica is a 403 naming the primary;
//! /metrics on the replica reports how far it has applied.

use std::path::Path;
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

fn start(data: &Path, bucket_root: &Path, replica: bool) -> Server {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vectordb"));
    cmd.args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()]);
    if replica {
        cmd.arg("--replica").env("DXMT_REPLICA_POLL_MS", "100");
    }
    let child = cmd
        .env("DXMT_ROOT_DIR", data)
        .env_remove("DXMT_WAL_DIR")
        .env("DXMT_SSD_CACHE_DIR", bucket_root)
        .env("DXMT_OBJECT_STORAGE_PROVIDER", "filesystem")
        .env("DXMT_S3_BUCKET", "bucket")
        .env("DXMT_EMBEDDING_AUTODETECT", "0")
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the vectordb binary");
    let server = Server { child, port };
    for _ in 0..160 {
        if http::call(port, "GET", "/live", &[], None).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

fn post(port: u16, path: &str, body: serde_json::Value) -> (u16, String) {
    http::call(port, "POST", path, &[], Some(&body.to_string())).expect("answer")
}

#[test]
fn a_replica_process_serves_the_primary_s_writes_and_refuses_its_own() {
    let tmp = tempfile::tempdir().unwrap();
    let bucket = tmp.path().join("bucket-root");
    let primary = start(&tmp.path().join("primary"), &bucket, false);
    let (status, _) = post(
        primary.port,
        "/api/v1/tables",
        serde_json::json!({"name": "t", "dimension": 4, "distance_metric": "euclidean"}),
    );
    assert_eq!(status, 200);
    let (status, _) = post(
        primary.port,
        "/api/v1/tables/t/records",
        serde_json::json!({"records": [{"id": "a", "vector": [1, 0, 0, 0]}]}),
    );
    assert_eq!(status, 200);

    let replica = start(&tmp.path().join("replica"), &bucket, true);
    let (status, _) = post(
        primary.port,
        "/api/v1/tables/t/records",
        serde_json::json!({"records": [{"id": "b", "vector": [0, 1, 0, 0]}]}),
    );
    assert_eq!(status, 200);

    // The replica polls every 100 ms; give it a few.
    let mut found = false;
    for _ in 0..50 {
        let (status, body) = post(
            replica.port,
            "/api/v1/tables/t/search",
            serde_json::json!({"vector": [0, 1, 0, 0], "top_k": 1}),
        );
        if status == 200 && body.contains("\"b\"") {
            found = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(found, "the replica never served the primary's write");

    let (status, body) = post(
        replica.port,
        "/api/v1/tables/t/records",
        serde_json::json!({"records": [{"id": "x", "vector": [0, 0, 1, 0]}]}),
    );
    assert_eq!(status, 403, "{body}");
    assert!(body.contains("READ_ONLY_REPLICA"), "{body}");

    let (_, metrics) = http::call(replica.port, "GET", "/metrics", &[], None).unwrap();
    assert!(
        metrics.contains("dxmt_replica_applied_next_sequence"),
        "{metrics}"
    );
    assert!(metrics.contains("dxmt_replica_stale 0"), "{metrics}");
    let (status, _) = http::call(replica.port, "GET", "/ready", &[], None).unwrap();
    assert_eq!(status, 200);
}
