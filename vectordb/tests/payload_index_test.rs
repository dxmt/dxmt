//! Which payload fields a table indexes for filtering, and what happens to a filter on one it
//! does not.
//!
//! Every field is indexed by default, so a filter is fast here without anyone declaring
//! anything — Qdrant and Weaviate both require the declaration. What that costs is uneven: on a
//! 10M-row table one field holding a distinct string per row took 11 s of a 32 s index rebuild
//! against 3.5 s for a four-valued one, and a field nobody filters on pays it for nothing.
//!
//! The part worth testing is not that the policy round-trips. It is the consequence: this
//! engine answers a filter *only* from the index — a field with no postings yields an empty
//! set, not a scan — so an excluded field has to be refused. An empty page and "nothing
//! matched" are the same response, and returning it would be a wrong answer that looks like a
//! right one.

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

/// Two records carrying a `category` anyone filters on and a `notes` nobody does.
fn seed(port: u16, table: &str, policy: serde_json::Value) {
    let mut config = serde_json::json!({"name": table, "dimension": 3});
    if !policy.is_null() {
        config["payload_index"] = policy;
    }
    let created =
        call(port, "POST", "/api/v1/tables", Some(&config.to_string())).expect("create answered");
    assert_eq!(created["success"], true, "create table: {created}");

    let body = serde_json::json!({"records": [
        {"id": "1", "vector": [1.0, 0.0, 0.0], "payload": {"category": "a", "notes": "one"}},
        {"id": "2", "vector": [0.0, 1.0, 0.0], "payload": {"category": "b", "notes": "two"}},
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

fn count_by(port: u16, table: &str, field: &str, value: &str) -> serde_json::Value {
    let body = serde_json::json!({"filter": {"must": [
        {"key": field, "operation": {"type": "Equal", "value": value}}
    ]}})
    .to_string();
    call(
        port,
        "POST",
        &format!("/api/v1/tables/{table}/records/count"),
        Some(&body),
    )
    .expect("count answered")
}

#[test]
fn a_filter_on_an_unindexed_field_is_refused_and_an_indexed_one_answers() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "narrow", serde_json::json!({"only": ["category"]}));

    let indexed = count_by(port, "narrow", "category", "a");
    assert_eq!(indexed["data"]["count"], 1, "indexed field: {indexed}");

    let excluded = count_by(port, "narrow", "notes", "one");
    assert_eq!(excluded["success"], false, "unindexed field: {excluded}");
    let message = excluded["error"]["message"].as_str().unwrap_or_default();
    // The refusal has to name the field and the way back, or it is just as unhelpful as the
    // empty page it replaces.
    assert!(
        message.contains("notes") && message.contains("payload_index/notes"),
        "the refusal should name the field and how to index it: {message}"
    );

    // A search carrying the same filter is refused for the same reason. It is a separate
    // entry point into the same index, and one of them answering empty would be the defect.
    let search = serde_json::json!({
        "vector": [1.0, 0.0, 0.0],
        "top_k": 2,
        "filter": {"must": [{"key": "notes", "operation": {"type": "Equal", "value": "one"}}]}
    })
    .to_string();
    let searched =
        call(port, "POST", "/api/v1/tables/narrow/search", Some(&search)).expect("search answered");
    assert_eq!(searched["success"], false, "search with it: {searched}");
}

#[test]
fn indexing_a_field_makes_its_filter_work_and_dropping_one_takes_it_away() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "moving", serde_json::json!({"only": ["category"]}));

    let put = call(
        port,
        "PUT",
        "/api/v1/tables/moving/payload_index/notes",
        None,
    )
    .expect("put answered");
    assert_eq!(put["success"], true, "put: {put}");

    // Rebuilt from the records, so the field is filterable over rows written before it was
    // indexed -- a payload index that only covered later writes would be a trap.
    let after_put = count_by(port, "moving", "notes", "one");
    assert_eq!(after_put["data"]["count"], 1, "after PUT: {after_put}");

    let deleted = call(
        port,
        "DELETE",
        "/api/v1/tables/moving/payload_index/category",
        None,
    )
    .expect("delete answered");
    assert_eq!(deleted["success"], true, "delete: {deleted}");

    let after_delete = count_by(port, "moving", "category", "a");
    assert_eq!(
        after_delete["success"], false,
        "a dropped field should stop answering: {after_delete}"
    );
    // And the field that is still indexed keeps working, so the rebuild dropped one field
    // rather than the index.
    let still = count_by(port, "moving", "notes", "two");
    assert_eq!(still["data"]["count"], 1, "the other field: {still}");
}

#[test]
fn the_report_lists_the_indexed_fields_and_does_not_build_the_index_to_do_it() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "report", serde_json::Value::Null);

    let report = call(port, "GET", "/api/v1/tables/report/payload_index", None)
        .expect("report answered")["data"]
        .clone();
    assert_eq!(report["indexes_every_field"], true, "default: {report}");
    let fields: Vec<&str> = report["fields"]
        .as_array()
        .expect("fields is an array")
        .iter()
        .map(|f| f["field"].as_str().unwrap_or_default())
        .collect();
    assert!(
        fields.contains(&"category") && fields.contains(&"notes"),
        "every field is indexed by default: {fields:?}"
    );

    // Excluding a field and asking again: the index was dropped by the policy change, and the
    // report says so rather than paying for a rebuild in order to answer.
    call(
        port,
        "DELETE",
        "/api/v1/tables/report/payload_index/notes",
        None,
    )
    .expect("delete answered");
    let after = call(port, "GET", "/api/v1/tables/report/payload_index", None)
        .expect("report answered")["data"]
        .clone();
    assert_eq!(
        after["built"], false,
        "not rebuilt to be described: {after}"
    );

    // One filter later it is built again, without the dropped field.
    assert_eq!(
        count_by(port, "report", "category", "a")["data"]["count"],
        1
    );
    let rebuilt = call(port, "GET", "/api/v1/tables/report/payload_index", None)
        .expect("report answered")["data"]
        .clone();
    let fields: Vec<&str> = rebuilt["fields"]
        .as_array()
        .expect("fields is an array")
        .iter()
        .map(|f| f["field"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(
        fields,
        vec!["category"],
        "the excluded field should cost nothing after the rebuild: {rebuilt}"
    );
}

#[test]
fn the_policy_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    {
        let _server = start(dir.path(), port);
        seed(port, "persist", serde_json::json!({"except": ["notes"]}));
        assert_eq!(
            count_by(port, "persist", "category", "a")["data"]["count"],
            1
        );
    }

    let port = free_port();
    let _server = start(dir.path(), port);
    // Replayed from the WAL: the table was created with the policy, and a policy that came
    // back as "index everything" would answer this filter instead of refusing it.
    let excluded = count_by(port, "persist", "notes", "one");
    assert_eq!(excluded["success"], false, "after restart: {excluded}");
    let report = call(port, "GET", "/api/v1/tables/persist/payload_index", None)
        .expect("report answered")["data"]
        .clone();
    assert_eq!(
        report["except"][0], "notes",
        "policy after restart: {report}"
    );
}
