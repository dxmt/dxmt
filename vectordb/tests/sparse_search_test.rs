//! Retrieval by sparse vector alone.
//!
//! Sparse vectors could be written here, and were indexed, and could not be searched by. The
//! only path into `SparseIndex` was `POST /search/hybrid`, which refuses a body carrying no
//! dense vector and no text — "At least one of 'vector' or 'text' must be provided" — and needs
//! the `text-bm25` build besides. A table of SPLADE vectors was stored, indexed, and
//! unreachable: accepted and ignored, which is the failure this repository keeps finding.
//!
//! The case worth a test beyond "it returns something" is the restart. The sparse index is
//! derived state, rebuilt on demand from the records rather than restored, and the same
//! omission for the payload index once made every filter match nothing after a restart with no
//! error anywhere. Sparse search has exactly that shape, so it gets exactly that test.

use std::process::{Child, Command, Stdio};

struct Server {
    child: Child,
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
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");
    let server = Server { child };
    for _ in 0..160 {
        if call(port, "GET", "/live", None).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

fn call(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<serde_json::Value> {
    let mut command = Command::new("curl");
    command
        .arg("-s")
        .arg("-m")
        .arg("60")
        .arg("-X")
        .arg(method)
        .arg(format!("http://127.0.0.1:{port}{path}"));
    if body.is_some() {
        command
            .arg("-H")
            .arg("content-type: application/json")
            .arg("--data-binary")
            .arg("@-")
            .stdin(Stdio::piped());
    }
    command.stdout(Stdio::piped());
    let mut child = command.spawn().expect("curl is required for this test");
    if let Some(body) = body {
        use std::io::Write as _;
        child
            .stdin
            .take()
            .expect("stdin was piped")
            .write_all(body.as_bytes())
            .expect("curl took the body");
    }
    let out = child.wait_with_output().expect("curl ran");
    serde_json::from_slice(&out.stdout).ok()
}

fn seed(port: u16, table: &str) {
    let created = call(
        port,
        "POST",
        "/api/v1/tables",
        Some(&serde_json::json!({"name": table, "dimension": 2}).to_string()),
    )
    .expect("create answered");
    assert_eq!(created["success"], true, "create table: {created}");

    // Dimension 5 is the query's dimension, so the dot products are readable by hand:
    // a scores 0.9 against it, b 0.2, c nothing at all.
    let body = serde_json::json!({"records": [
        {"id": "a", "vector": [1.0, 0.0], "payload": {"lang": "en"},
         "sparse_vector": {"indices": [1, 5], "values": [0.5, 0.9], "dimension": 100}},
        {"id": "b", "vector": [0.0, 1.0], "payload": {"lang": "zh"},
         "sparse_vector": {"indices": [5, 7], "values": [0.2, 0.8], "dimension": 100}},
        {"id": "c", "vector": [1.0, 1.0], "payload": {"lang": "en"},
         "sparse_vector": {"indices": [9], "values": [1.0], "dimension": 100}},
        // No sparse vector at all: it must never appear in a sparse result.
        {"id": "dense_only", "vector": [0.5, 0.5], "payload": {"lang": "en"}},
    ]})
    .to_string();
    let inserted = call(
        port,
        "POST",
        &format!("/api/v1/tables/{table}/records"),
        Some(&body),
    )
    .expect("insert answered");
    assert_eq!(inserted["success"], true, "insert: {inserted}");
}

fn sparse(port: u16, table: &str, body: serde_json::Value) -> Vec<(String, f64)> {
    let answer = call(
        port,
        "POST",
        &format!("/api/v1/tables/{table}/search/sparse"),
        Some(&body.to_string()),
    )
    .expect("sparse search answered");
    assert_eq!(answer["success"], true, "sparse search: {answer}");
    answer["data"]["hits"]
        .as_array()
        .expect("hits")
        .iter()
        .map(|h| {
            (
                h["id"].as_str().unwrap_or_default().to_string(),
                h["score"].as_f64().unwrap_or_default(),
            )
        })
        .collect()
}

/// The query: one dimension, weight 1.0, so a hit's score is its own weight there.
fn on_dimension(dim: u32) -> serde_json::Value {
    serde_json::json!({"indices": [dim], "values": [1.0], "dimension": 100})
}

#[test]
fn a_sparse_query_ranks_by_dot_product_and_skips_records_without_one() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "sparse");

    let hits = sparse(
        port,
        "sparse",
        serde_json::json!({"sparse_vector": on_dimension(5), "limit": 10}),
    );
    assert_eq!(
        hits.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"],
        "a holds 0.9 on dimension 5 and b holds 0.2; c and dense_only hold nothing there: {hits:?}"
    );
    assert!(
        (hits[0].1 - 0.9).abs() < 1e-5,
        "the score is the dot product: {hits:?}"
    );
    assert!(
        (hits[1].1 - 0.2).abs() < 1e-5,
        "the score is the dot product: {hits:?}"
    );
}

#[test]
fn a_filter_a_threshold_and_a_projection_apply_as_they_do_on_a_dense_search() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "sparse");

    let filtered = sparse(
        port,
        "sparse",
        serde_json::json!({
            "sparse_vector": on_dimension(5),
            "limit": 10,
            "filter": {"must": [{"key": "lang", "operation": {"type": "Equal", "value": "zh"}}]}
        }),
    );
    assert_eq!(
        filtered
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>(),
        vec!["b"],
        "the filter should remove a, which outscores b: {filtered:?}"
    );

    // A floor on the dot product, the same direction as `score_threshold` on a dense search:
    // higher is nearer, and it only removes.
    let floored = sparse(
        port,
        "sparse",
        serde_json::json!({
            "sparse_vector": on_dimension(5), "limit": 10, "score_threshold": 0.5
        }),
    );
    assert_eq!(
        floored
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>(),
        vec!["a"],
        "b scores 0.2 and is below the floor: {floored:?}"
    );

    let answer = call(
        port,
        "POST",
        "/api/v1/tables/sparse/search/sparse",
        Some(
            &serde_json::json!({
                "sparse_vector": on_dimension(9),
                "limit": 1,
                "payload_fields": ["lang"],
                "with_vector": true
            })
            .to_string(),
        ),
    )
    .expect("sparse search answered");
    let hit = &answer["data"]["hits"][0];
    assert_eq!(hit["id"], "c", "{answer}");
    assert_eq!(
        hit["payload"],
        serde_json::json!({"lang": "en"}),
        "{answer}"
    );
    assert_eq!(hit["vector"], serde_json::json!([1.0, 1.0]), "{answer}");
}

#[test]
fn sparse_search_still_answers_after_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let port = free_port();
        let _server = start(dir.path(), port);
        seed(port, "sparse");
        assert_eq!(
            sparse(
                port,
                "sparse",
                serde_json::json!({"sparse_vector": on_dimension(5), "limit": 10})
            )
            .len(),
            2
        );
    }

    // The sparse index is derived from the records and rebuilt on demand, not restored. When
    // the payload index had the same shape and `restore_table` did not rebuild it, every filter
    // matched nothing after a restart -- a 200 with an empty page, indistinguishable from a
    // filter that genuinely selects nothing. This is that test for the sparse half.
    let port = free_port();
    let _server = start(dir.path(), port);
    let hits = sparse(
        port,
        "sparse",
        serde_json::json!({"sparse_vector": on_dimension(5), "limit": 10}),
    );
    assert_eq!(
        hits.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"],
        "after a restart: {hits:?}"
    );
}
