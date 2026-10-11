//! A deleted record is gone from every path that could return it.
//!
//! `cargo test --test delete_sweep_test` (add `--features text-bm25` for the text and fusion
//! halves, which are skipped without it).
//!
//! A record lands in as many as five places when it is written: the vector index, the payload
//! index that answers filters, the sparse postings, the full-text index, and one index per
//! named vector space. Deleting it has to take it out of all five, and the existing coverage
//! checked two — the record store and the default vector index. The named spaces were missed
//! for exactly that reason, and the symptom was not an empty result but a **wrong** one: the
//! deleted record came back at rank one, with the score of an exact match, because it was still
//! the nearest thing in a graph nobody had told about the delete.
//!
//! So this asks every path in one place. A path added to the server without a line here is a
//! path a delete can be forgotten in.

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
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn start(dir: &std::path::Path, port: u16) -> Server {
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DXMT_WAL_DIR", dir.join("wal"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");

    let server = Server { child, port };
    for _ in 0..200 {
        if call(port, "GET", "/api/v1/health", None).is_some() {
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

/// The ids in an answer, from whichever shape it uses.
fn ids(answer: &serde_json::Value) -> Vec<String> {
    for pointer in ["/data/hits", "/data/records", "/data/ids"] {
        if let Some(items) = answer.pointer(pointer).and_then(|v| v.as_array()) {
            return items
                .iter()
                .map(|item| match item {
                    serde_json::Value::String(id) => id.clone(),
                    other => other["id"].as_str().unwrap_or_default().to_string(),
                })
                .collect();
        }
    }
    panic!("no ids in {answer}");
}

const DOOMED: &str = "doomed";

/// Every path a record can be returned by, asked in a way that would return `doomed` if it were
/// still there: its own vector, its own named vector, a filter it satisfies, its own sparse
/// dimension, a word its text contains.
///
/// Each of those is shared with exactly one other record, `keep1`. A query that only `doomed`
/// could answer would return nothing after the delete whether the index had been *updated* or
/// simply *emptied*, and those are different bugs — the survivor is what tells them apart.
fn every_path(port: u16) -> Vec<(&'static str, Vec<String>)> {
    // `mut` only when the text halves below are compiled in; the attribute keeps a default
    // build from warning about it rather than splitting this into two functions.
    #[cfg_attr(not(feature = "text-bm25"), allow(unused_mut))]
    let mut answers = vec![
        (
            "vector search",
            ids(&call(port, "POST", "/api/v1/tables/t/search",
                Some(r#"{"vector":[1,0,0,0],"top_k":5,"limit":5}"#)).unwrap()),
        ),
        (
            "vector search, exact",
            ids(&call(port, "POST", "/api/v1/tables/t/search",
                Some(r#"{"vector":[1,0,0,0],"top_k":5,"limit":5,"search_mode":"exact"}"#)).unwrap()),
        ),
        (
            "named vector search",
            ids(&call(port, "POST", "/api/v1/tables/t/search",
                Some(r#"{"vector":[0,0,0,1],"top_k":5,"limit":5,"vector_name":"other"}"#)).unwrap()),
        ),
        (
            "filtered search",
            ids(&call(port, "POST", "/api/v1/tables/t/search",
                Some(r#"{"vector":[1,0,0,0],"top_k":5,"limit":5,"filter":{"must":[{"key":"mark","operation":{"type":"Equal","value":"target"}}]}}"#)).unwrap()),
        ),
        (
            "records/list, filtered",
            ids(&call(port, "POST", "/api/v1/tables/t/records/list",
                Some(r#"{"limit":9,"filter":{"must":[{"key":"mark","operation":{"type":"Equal","value":"target"}}]}}"#)).unwrap()),
        ),
        (
            "records/list, unfiltered",
            ids(&call(port, "POST", "/api/v1/tables/t/records/list", Some(r#"{"limit":99}"#)).unwrap()),
        ),
        (
            "knn graph",
            ids(&call(port, "POST", "/api/v1/tables/t/graph",
                Some(r#"{"sample":99,"neighbours":2,"seed":1}"#)).unwrap()),
        ),
    ];

    #[cfg(feature = "text-bm25")]
    {
        answers.push((
            "full-text search",
            ids(&call(
                port,
                "POST",
                "/api/v1/tables/t/search/text",
                Some(r#"{"query":"condemned","limit":9}"#),
            )
            .unwrap()),
        ));
        answers.push((
            "hybrid, sparse half",
            ids(&call(
                port,
                "POST",
                "/api/v1/tables/t/search/hybrid",
                Some(
                    r#"{"vector":[0,0,1,0],"text":"condemned","limit":9,"fusion_method":"dbsf",
                         "sparse_vector":{"indices":[7],"values":[1.0],"dimension":16}}"#,
                ),
            )
            .unwrap()),
        ));
    }

    answers
}

fn seed(port: u16) {
    call(
        port,
        "POST",
        "/api/v1/tables",
        Some(
            r#"{"name":"t","dimension":4,"distance_metric":"cosine",
                "enable_full_text_search":true,
                "text_fields":[{"name":"content","source":{"metadata_field":"content"},"weight":1.0}],
                "named_vectors":[{"name":"other","dimension":4,"distance_metric":"cosine"}]}"#,
        ),
    );
    call(
        port,
        "POST",
        "/api/v1/tables/t/records",
        Some(
            r#"{"records":[
                {"id":"doomed","vector":[1,0,0,0],"named_vectors":{"other":[0,0,0,1]},
                 "sparse_vector":{"indices":[7],"values":[1.0],"dimension":16},
                 "payload":{"mark":"target","content":"a condemned document"}},
                {"id":"keep1","vector":[0,1,0,0],"named_vectors":{"other":[0,0,1,0]},
                 "sparse_vector":{"indices":[7],"values":[0.5],"dimension":16},
                 "payload":{"mark":"target","content":"another condemned document"}},
                {"id":"keep2","vector":[0,0,1,0],"named_vectors":{"other":[0,1,0,0]},
                 "sparse_vector":{"indices":[2],"values":[1.0],"dimension":16},
                 "payload":{"mark":"ordinary","content":"another ordinary document"}}
            ]}"#,
        ),
    );
}

#[test]
fn a_deleted_record_is_gone_from_every_path() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let server = start(dir.path(), port);
    seed(server.port);

    // Every path must find it first, or the check below passes for the wrong reason.
    for (path, found) in every_path(server.port) {
        assert!(
            found.contains(&DOOMED.to_string()),
            "{path} did not return {DOOMED} before it was deleted, so it cannot show that the \
             delete worked: {found:?}"
        );
    }

    let deleted = call(
        server.port,
        "DELETE",
        "/api/v1/tables/t/records",
        Some(r#"{"ids":["doomed"]}"#),
    )
    .unwrap();
    assert_eq!(deleted["data"]["deleted_count"], 1, "{deleted}");

    let mut still_there = Vec::new();
    for (path, found) in every_path(server.port) {
        if found.contains(&DOOMED.to_string()) {
            still_there.push(format!("  {path}: {found:?}"));
        }
        assert!(
            found.contains(&"keep1".to_string()) || found.contains(&"keep2".to_string()),
            "{path} returned nothing at all after the delete, which is a different bug: {found:?}"
        );
    }
    assert!(
        still_there.is_empty(),
        "a deleted record was still returned by:\n{}",
        still_there.join("\n")
    );

    assert_eq!(
        call(server.port, "GET", "/api/v1/tables/t", None).unwrap()["data"]["record_count"],
        2
    );
}

/// Deleting by filter takes the record out of the same five places deleting by id does.
///
/// "Remove everything belonging to tenant X" could not be expressed at all before: the caller
/// paged the table, collected ids and sent them back. The risk in the new path is that it is a
/// *different* path — so it is asked the same question this file asks of every other one.
#[test]
fn a_filter_deletes_from_every_path_too() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let server = start(dir.path(), port);
    seed(server.port);

    // `mark: target` is `doomed` and `keep1`; `keep2` is ordinary and must survive.
    let answer = call(
        server.port,
        "DELETE",
        "/api/v1/tables/t/records",
        Some(
            r#"{"filter":{"must":[{"key":"mark","operation":{"type":"Equal","value":"target"}}]}}"#,
        ),
    )
    .expect("no answer");
    assert_eq!(answer["data"]["deleted_count"], 2, "{answer}");

    let mut still_there = Vec::new();
    for (path, found) in every_path(server.port) {
        for gone in [DOOMED, "keep1"] {
            if found.contains(&gone.to_string()) {
                still_there.push(format!("  {path}: {gone} in {found:?}"));
            }
        }
    }
    assert!(
        still_there.is_empty(),
        "records deleted by filter came back through:\n{}",
        still_there.join("\n")
    );
}

/// The two selectors are exclusive, and an empty body is not "delete everything".
#[test]
fn a_delete_names_ids_or_a_filter_but_not_both_and_not_neither() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let server = start(dir.path(), port);
    seed(server.port);

    for body in [
        r#"{}"#,
        r#"{"ids":["doomed"],"filter":{"must":[{"key":"mark","operation":{"type":"Equal","value":"target"}}]}}"#,
    ] {
        let answer = call(
            server.port,
            "DELETE",
            "/api/v1/tables/t/records",
            Some(body),
        )
        .expect("no answer");
        assert_eq!(answer["success"], false, "{body} was accepted: {answer}");
        assert_eq!(answer["error"]["code"], "INVALID_INPUT", "{answer}");
    }

    // Nothing was deleted by either refusal.
    let (_, all) = every_path(server.port)
        .into_iter()
        .find(|(path, _)| *path == "records/list, unfiltered")
        .expect("the unfiltered list is one of the paths");
    assert_eq!(all.len(), 3, "a refused delete removed something: {all:?}");
}

#[test]
fn it_is_still_gone_after_a_restart() {
    // The indexes are rebuilt from the records on the way back up, so a record the *records*
    // still held would reappear everywhere at once.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    {
        let server = start(dir.path(), port);
        seed(server.port);
        call(
            server.port,
            "DELETE",
            "/api/v1/tables/t/records",
            Some(r#"{"ids":["doomed"]}"#),
        );
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
    }

    let server = start(dir.path(), port);
    let mut still_there = Vec::new();
    for (path, found) in every_path(server.port) {
        if found.contains(&DOOMED.to_string()) {
            still_there.push(format!("  {path}: {found:?}"));
        }
    }
    assert!(
        still_there.is_empty(),
        "a record deleted before the restart came back through:\n{}",
        still_there.join("\n")
    );
}
