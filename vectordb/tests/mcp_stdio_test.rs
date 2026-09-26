//! End-to-end test of the MCP stdio transport.
//!
//! This spawns the real binary rather than calling into the library, because most of what can
//! break here lives outside the request handlers: logs leaking onto stdout, missing flushes,
//! the wrong field casing on the wire. A test that called `handle_message` directly would pass
//! while the server was unusable from any actual client.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use tempfile::TempDir;

mod fake_provider;

/// Send `requests` to a fresh MCP server, return one parsed reply per request.
fn exchange(wal_dir: &std::path::Path, requests: &[&str]) -> Vec<serde_json::Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("mcp")
        .arg("--wal-dir")
        .arg(wal_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");

    {
        let stdin = child.stdin.as_mut().unwrap();
        for req in requests {
            writeln!(stdin, "{}", req).unwrap();
        }
    }
    // Dropping stdin closes the pipe, which ends the server's read loop.
    drop(child.stdin.take());

    let stdout = BufReader::new(child.stdout.take().unwrap());
    let replies: Vec<_> = stdout
        .lines()
        .map(|l| l.expect("stdout was not valid UTF-8"))
        .map(|l| {
            serde_json::from_str(&l).unwrap_or_else(|e| {
                // The most likely cause is a log line on stdout, so quote it in the failure.
                panic!("stdout carried a non-JSON line ({}): {:?}", e, l)
            })
        })
        .collect();

    child.wait().unwrap();
    replies
}

fn initialize() -> &'static str {
    r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#
}

#[test]
fn initialize_accepts_the_wire_format_clients_send() {
    let tmp = TempDir::new().unwrap();
    let replies = exchange(&tmp.path().join("wal"), &[initialize()]);

    assert_eq!(replies.len(), 1);
    // MCP is camelCase on the wire. With serde's default snake_case this request failed with
    // "missing field `protocol_version`" — every client would have been rejected at the door.
    assert!(
        replies[0].get("error").is_none(),
        "initialize was rejected: {}",
        replies[0]
    );
    assert!(
        replies[0]["result"]["serverInfo"].is_object(),
        "expected camelCase serverInfo"
    );
}

#[test]
fn every_tool_is_listed_not_just_the_one_that_won_a_race() {
    let tmp = TempDir::new().unwrap();
    let replies = exchange(
        &tmp.path().join("wal"),
        &[
            initialize(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
        ],
    );

    let tools = replies[1]["result"]["tools"]
        .as_array()
        .expect("no tools array");
    let names: Vec<_> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();

    // The catalogue used to be filled by a tokio::spawn from the constructor, so a tools/list
    // arriving first saw whatever that task had managed to push — one tool, in practice.
    assert!(
        names.contains(&"vector_search"),
        "missing vector_search: {:?}",
        names
    );
    assert!(
        names.contains(&"create_table"),
        "missing create_table: {:?}",
        names
    );
    assert!(
        names.len() >= 7,
        "expected the full catalogue, got {:?}",
        names
    );
    assert!(
        tools[0]["inputSchema"].is_object(),
        "schema field must be inputSchema"
    );
}

#[test]
fn a_response_never_carries_both_result_and_error() {
    let tmp = TempDir::new().unwrap();
    let replies = exchange(
        &tmp.path().join("wal"),
        &[
            initialize(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
        ],
    );

    for reply in &replies {
        let has_result = reply.get("result").is_some();
        let has_error = reply.get("error").is_some();
        assert!(
            has_result ^ has_error,
            "JSON-RPC allows exactly one of result/error, got: {}",
            reply
        );
    }
}

#[test]
fn tools_reach_the_storage_layer_and_persist() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");

    let replies = exchange(
        &wal,
        &[
            initialize(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"create_table","arguments":{"name":"t","dimension":4,"metric":"euclidean"}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"insert_vector","arguments":{"table":"t","id":"a","vector":[1,0,0,0]}}}"#,
        ],
    );
    for r in &replies[1..] {
        assert_eq!(
            r["result"]["isError"],
            serde_json::json!(false),
            "tool call failed: {}",
            r
        );
    }

    // Second process, same WAL directory: the MCP server has to be backed by the same durable
    // storage as the HTTP server, not by a private in-memory one.
    let replies = exchange(
        &wal,
        &[
            initialize(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"vector_search","arguments":{"table":"t","vector":[1,0,0,0],"limit":1}}}"#,
        ],
    );
    let text = replies[1]["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("\"id\": \"a\""),
        "vector did not survive the restart: {}",
        text
    );
}

#[test]
fn a_table_name_cannot_put_files_outside_the_data_directory() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");

    // Row files are `<wal>/records/<table>.rec` and `<wal>/vectors/<table>.f32`, joined with no
    // sanitising, and this tool used to accept any string as the table name.
    let replies = exchange(
        &wal,
        &[
            initialize(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"create_table","arguments":{"name":"../../escape","dimension":4}}}"#,
        ],
    );
    let reply = &replies[1];
    let refused =
        reply.get("error").is_some() || reply["result"]["isError"] == serde_json::json!(true);
    assert!(refused, "a table named ../../escape was created: {}", reply);

    let stray: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .filter(|name| name != "wal")
        .collect();
    assert!(
        stray.is_empty(),
        "files written beside the WAL directory: {:?}",
        stray
    );
}

#[test]
fn an_agent_with_no_embedding_model_can_load_a_corpus_and_search_it() {
    // This is the case MCP exists for and the one it could not serve: the caller is a language
    // model driving tools, and it has no way to turn a sentence into 1,536 floats. Every tool
    // here required a vector, so `insert_vector` and `vector_search` were unusable by the only
    // sort of client that speaks this protocol.
    //
    // The provider runs in this process; the server runs as a child and reaches it over
    // loopback, exactly as it would reach a text-embeddings-inference container.
    let (base_url, provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");

    // No `dimension`: the agent names a model and the server asks it how wide it is.
    let create = format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"create_table","arguments":{{"name":"notes","embedding":{{"model":"bag-of-words","base_url":"{base_url}","api_key_env":"DXMT_TEST_KEY_UNSET"}}}}}}}}"#
    );
    let replies = exchange(
        &wal,
        &[
            initialize(),
            &create,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"insert_vector","arguments":{"table":"notes","id":"rust","text":"rust borrows memory instead of collecting it"}}}"#,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"insert_vector","arguments":{"table":"notes","id":"go","text":"go collects garbage instead of borrowing"}}}"#,
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"vector_search","arguments":{"table":"notes","text":"borrows memory","limit":2}}}"#,
        ],
    );
    for r in &replies[1..] {
        assert_eq!(
            r["result"]["isError"],
            serde_json::json!(false),
            "tool call failed: {}",
            r
        );
    }

    // The create reports the width it took from the model, not the one it was given.
    let created = replies[1]["result"]["content"][0]["text"].as_str().unwrap();
    assert!(created.contains("dimension 32"), "{created}");

    let hits = replies[4]["result"]["content"][0]["text"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(hits).unwrap();
    assert_eq!(parsed["hits"][0]["id"], "rust", "ranked: {hits}");
    // The text came back with the hit, which is what makes the result usable as context --
    // otherwise an agent has the ids of passages it cannot read.
    assert_eq!(
        parsed["hits"][0]["payload"]["text"], "rust borrows memory instead of collecting it",
        "{hits}"
    );
    // One width probe, two inserts, one query: the corpus is not re-embedded per search.
    assert_eq!(provider.call_count(), 4);
}

#[test]
fn a_tool_call_with_neither_a_vector_nor_text_says_which_to_send() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");
    let replies = exchange(
        &wal,
        &[
            initialize(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"create_table","arguments":{"name":"t","dimension":4}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"insert_vector","arguments":{"table":"t","id":"a"}}}"#,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"vector_search","arguments":{"table":"t"}}}"#,
        ],
    );
    for r in &replies[2..] {
        let message = serde_json::to_string(&r["error"]).unwrap();
        assert!(
            message.contains("`text`"),
            "the error does not name the alternative: {r}"
        );
    }
}
