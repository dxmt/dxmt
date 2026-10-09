//! Grouped search: one document's chunks stop taking the whole page.
//!
//! The corpus here is the case the feature exists for — a document cut into chunks, each chunk
//! its own record. A plain top-k over it returns the same document k times, which is a correct
//! nearest-neighbour answer and a useless retrieval result.
//!
//! What is checked beyond "it groups": that the window is widened until enough groups appear
//! rather than answering short from the first batch, that a record with no value under
//! `group_by` is counted instead of quietly dropped, and that the grouping key does not leak
//! into a payload the caller did not ask for — the server has to read that field to group at
//! all, so it is exactly the kind of thing that ends up in the response by accident.

use std::process::{Child, Command, Stdio};

#[path = "../common/http.rs"]
mod http;

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
    let (_, text) = http::call(port, method, path, &[], body)?;
    serde_json::from_str(&text).ok()
}

/// Twelve documents of eight chunks each, nearest first by document, plus one record with no
/// `doc_id` at all.
///
/// Twelve by eight is chosen against the default window: `groups * group_size * 4` for ten
/// groups of one is 40 candidates, and the first 40 nearest neighbours here cover five
/// documents. Filling ten groups therefore *requires* the widening loop, which is the part a
/// smaller corpus would not exercise.
fn seed(port: u16, table: &str) {
    let created = call(
        port,
        "POST",
        "/api/v1/tables",
        Some(&serde_json::json!({"name": table, "dimension": 2}).to_string()),
    )
    .expect("create answered");
    assert_eq!(created["success"], true, "create table: {created}");

    let mut records = Vec::new();
    for doc in 0..12 {
        for chunk in 0..8 {
            // Each document occupies a band of angles; chunks within a document are adjacent.
            // So the nearest 40 records are the first five documents, whole.
            let offset = doc as f32 * 0.05 + chunk as f32 * 0.001;
            records.push(serde_json::json!({
                "id": format!("d{doc}c{chunk}"),
                "vector": [1.0 - offset, offset],
                "payload": {"doc_id": format!("doc{doc}"), "chunk": chunk, "body": "text"}
            }));
        }
    }
    records.push(serde_json::json!({
        "id": "orphan",
        "vector": [0.9, 0.1],
        "payload": {"body": "no doc_id here"}
    }));
    let body = serde_json::json!({"records": records}).to_string();
    let inserted = call(
        port,
        "POST",
        &format!("/api/v1/tables/{table}/records"),
        Some(&body),
    )
    .expect("insert answered");
    assert_eq!(inserted["success"], true, "insert: {inserted}");
}

fn groups(port: u16, table: &str, body: serde_json::Value) -> serde_json::Value {
    let answer = call(
        port,
        "POST",
        &format!("/api/v1/tables/{table}/search/groups"),
        Some(&body.to_string()),
    )
    .expect("grouped search answered");
    assert_eq!(answer["success"], true, "grouped search: {answer}");
    answer["data"].clone()
}

#[test]
fn grouping_returns_distinct_documents_where_a_plain_search_returns_one() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "chunks");

    // The problem, stated as an assertion: the nearest eight records are one document.
    let plain = call(
        port,
        "POST",
        "/api/v1/tables/chunks/search",
        Some(
            &serde_json::json!({"vector": [1.0, 0.0], "top_k": 8,
                                "payload_fields": ["doc_id"]})
            .to_string(),
        ),
    )
    .expect("search answered");
    let docs: std::collections::BTreeSet<String> = plain["data"]["hits"]
        .as_array()
        .expect("hits")
        .iter()
        .map(|h| {
            h["payload"]["doc_id"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    assert_eq!(
        docs.len(),
        1,
        "a plain top-8 should be one document: {docs:?}"
    );

    // The answer: ten documents, one chunk each, and the widening loop had to run to find
    // them — the first window of 40 candidates only covers five.
    let grouped = groups(
        port,
        "chunks",
        serde_json::json!({"vector": [1.0, 0.0], "group_by": "doc_id",
                           "groups": 10, "group_size": 1, "payload_fields": ["doc_id"]}),
    );
    let ids: Vec<&str> = grouped["groups"]
        .as_array()
        .expect("groups")
        .iter()
        .map(|g| g["group_id"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(ids.len(), 10, "ten groups: {ids:?}");
    assert_eq!(ids[0], "doc0", "best group first: {ids:?}");
    let distinct: std::collections::BTreeSet<&&str> = ids.iter().collect();
    assert_eq!(
        distinct.len(),
        10,
        "each group is a different document: {ids:?}"
    );
    assert!(
        grouped["candidates_fetched"].as_u64().unwrap_or(0) > 40,
        "the window should have widened past its first size: {grouped}"
    );
    for group in grouped["groups"].as_array().expect("groups") {
        assert_eq!(group["hits"].as_array().expect("hits").len(), 1);
        // `matched` counts what fell inside the window, not what the document holds: the
        // nearest document has all eight chunks in it, the tenth only as many as the window
        // reached. Asserting 8 everywhere is what this test did first, and the failure was
        // the feature working -- the widened window ends partway through the last document.
        let matched = group["matched"].as_u64().unwrap_or(0);
        assert!(
            (1..=8).contains(&matched),
            "matched counts the window: {group}"
        );
    }
    assert_eq!(
        grouped["groups"][0]["matched"], 8,
        "the nearest document is wholly inside any window: {grouped}"
    );
}

#[test]
fn group_size_keeps_the_best_of_each_group_and_a_record_without_the_field_is_counted() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "sized");

    let grouped = groups(
        port,
        "sized",
        serde_json::json!({"vector": [1.0, 0.0], "group_by": "doc_id",
                           "groups": 3, "group_size": 2, "payload_fields": ["chunk"]}),
    );
    let first = &grouped["groups"][0];
    let chunks: Vec<i64> = first["hits"]
        .as_array()
        .expect("hits")
        .iter()
        .map(|h| h["payload"]["chunk"].as_i64().unwrap_or(-1))
        .collect();
    assert_eq!(
        chunks,
        vec![0, 1],
        "the two nearest chunks, in order: {first}"
    );

    // The record with no `doc_id` belongs to no group. Reported, because a corpus where the
    // field is missing on half the rows otherwise looks like a half-empty index.
    assert_eq!(grouped["ungrouped"], 1, "the orphan is counted: {grouped}");

    // And the grouping key stays out of a payload that did not ask for it -- the server has to
    // read `doc_id` to group at all, so it is exactly what leaks by accident.
    assert!(
        first["hits"][0]["payload"].get("doc_id").is_none(),
        "doc_id was not requested: {first}"
    );
}

#[test]
fn a_filter_applies_before_the_grouping() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "filtered");

    let grouped = groups(
        port,
        "filtered",
        serde_json::json!({
            "vector": [1.0, 0.0],
            "group_by": "doc_id",
            "groups": 5,
            "group_size": 3,
            "filter": {"must": [{"key": "chunk", "operation": {"type": "LessThan", "value": 2}}]},
            "payload_fields": ["chunk"]
        }),
    );
    for group in grouped["groups"].as_array().expect("groups") {
        assert_eq!(
            group["matched"], 2,
            "only two chunks per document pass the filter: {group}"
        );
        for hit in group["hits"].as_array().expect("hits") {
            assert!(
                hit["payload"]["chunk"].as_i64().unwrap_or(9) < 2,
                "a hit outside the filter: {hit}"
            );
        }
    }
}
