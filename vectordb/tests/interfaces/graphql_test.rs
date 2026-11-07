//! `POST /graphql`, against the real binary started with `--graphql`.
//!
//! Every resolver reads the server's state from the GraphQL context, and nothing put it there:
//! until this file existed, a query touching data answered "Data `Arc<AppState>` does not
//! exist" while `{ __typename }` -- the only thing anyone had tried -- worked. There was no test
//! of GraphQL at all, which is how that went unseen.
//!
//! Compiled only with `api-graphql`; the "Feature-gated tests" CI step names it.
#![cfg(feature = "api-graphql")]

use std::process::{Child, Command, Stdio};
use std::time::Duration;

#[path = "../common/http.rs"]
mod http;

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn start(dir: &std::path::Path, api_key: Option<&str>) -> (Server, u16) {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vectordb"));
    cmd.args(["serve", "--graphql", "--port", &port.to_string()])
        .env("DXMT_WAL_DIR", dir.join("wal"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    match api_key {
        Some(k) => cmd.env("DXMT_API_KEY", k),
        None => cmd.env_remove("DXMT_API_KEY"),
    };
    let server = Server(cmd.spawn().expect("spawn vectordb"));
    for _ in 0..160 {
        if http::call(port, "GET", "/live", &[], None).is_some() {
            return (server, port);
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

fn graphql(port: u16, key: Option<&str>, body: serde_json::Value) -> (u16, serde_json::Value) {
    let mut headers = vec![("content-type", "application/json")];
    if let Some(k) = key {
        headers.push(("api-key", k));
    }
    let (status, text) = http::call(port, "POST", "/graphql", &headers, Some(&body.to_string()))
        .expect("no answer from /graphql");
    (
        status,
        serde_json::from_str(&text).unwrap_or(serde_json::Value::Null),
    )
}

#[test]
fn queries_and_mutations_reach_the_storage() {
    let dir = tempfile::tempdir().unwrap();
    let (_server, port) = start(dir.path(), None);

    let (status, created) = graphql(
        port,
        None,
        serde_json::json!({
            "query": "mutation { createTable(input: {name: \"docs\", dimension: 2}) { name dimension } }"
        }),
    );
    assert_eq!(status, 200, "{created}");
    assert_eq!(
        created["data"]["createTable"],
        serde_json::json!({"name": "docs", "dimension": 2}),
        "{created}"
    );

    let (_, listed) = graphql(
        port,
        None,
        serde_json::json!({"query": "{ tables { name dimension } }"}),
    );
    assert!(listed["errors"].is_null(), "{listed}");
    assert_eq!(
        listed["data"]["tables"],
        serde_json::json!([{"name": "docs", "dimension": 2}])
    );

    // The table made through GraphQL is the REST API's table: one store behind both.
    let (status, rest) = http::call(port, "GET", "/api/v1/tables/docs", &[], None).unwrap();
    assert_eq!(status, 200, "{rest}");

    // A batch is an array in and an array out.
    let (_, batch) = graphql(
        port,
        None,
        serde_json::json!([{"query": "{ __typename }"}, {"query": "{ tables { name } }"}]),
    );
    assert_eq!(batch[1]["data"]["tables"][0]["name"], "docs", "{batch}");
}

#[test]
fn the_api_key_guards_graphql_too() {
    let dir = tempfile::tempdir().unwrap();
    let (_server, port) = start(dir.path(), Some("graphql-test-key"));
    let query = serde_json::json!({"query": "{ tables { name } }"});

    let (status, _) = graphql(port, None, query.clone());
    assert_eq!(status, 401, "a query without the key is refused");

    let (status, answer) = graphql(port, Some("graphql-test-key"), query);
    assert_eq!(status, 200, "{answer}");
    assert!(answer["errors"].is_null(), "{answer}");
}
