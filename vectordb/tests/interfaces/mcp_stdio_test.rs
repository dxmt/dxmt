//! End-to-end test of the MCP stdio transport.
//!
//! This spawns the real binary rather than calling into the library, because most of what can
//! break here lives outside the request handlers: logs leaking onto stdout, missing flushes,
//! the wrong field casing on the wire. A test that called `handle_message` directly would pass
//! while the server was unusable from any actual client.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use tempfile::TempDir;

// Shared module, one directory up now that this test lives in a scenario
// folder. Cargo resolves `mod` relative to the file, so without the path it
// would look for tests/api/fake_provider.rs and fail to compile.
#[path = "../fake_provider/mod.rs"]
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

#[test]
fn hybrid_search_from_text_alone_uses_both_halves() {
    // The tool ran a plain vector search under this name, and answered text alone with an
    // empty page and a note -- and a table `create_table` made had no keyword index to fuse
    // with anyway. An agent has text; this is the call it makes.
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");
    let create = format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"create_table","arguments":{{"name":"notes","embedding":{{"model":"bag-of-words","base_url":"{base_url}","api_key_env":"DXMT_TEST_KEY_UNSET"}}}}}}}}"#
    );
    let loaded = exchange(
        &wal,
        &[
            initialize(),
            &create,
            // Ids chosen so that a tie, which is what a fusion with no text half produces at
            // vector_weight 0, ranks the wrong one first.
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"insert_vector","arguments":{"table":"notes","id":"a_dinner","text":"tomato and egg for dinner tonight"}}}"#,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"insert_vector","arguments":{"table":"notes","id":"z_graph","text":"hnsw builds a graph index for vector search"}}}"#,
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"get_table_info","arguments":{"name":"notes"}}}"#,
        ],
    );
    for r in &loaded[1..] {
        assert_eq!(r["result"]["isError"], serde_json::json!(false), "{r}");
    }
    let info = loaded[4]["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        info.contains("\"enable_full_text_search\": true"),
        "a table that stores text has no keyword index: {info}"
    );

    // A second process: the keyword index is read back from disk, as after any restart.
    let searched = exchange(
        &wal,
        &[
            initialize(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"hybrid_search","arguments":{"table":"notes","text":"graph index","limit":2}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"hybrid_search","arguments":{"table":"notes","text":"graph index","limit":2,"vector_weight":0.0,"fusion_method":"weighted_sum"}}}"#,
        ],
    );
    for (n, r) in searched[1..].iter().enumerate() {
        assert_eq!(r["result"]["isError"], serde_json::json!(false), "{r}");
        let hits: serde_json::Value =
            serde_json::from_str(r["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(hits["hits"][0]["id"], "z_graph", "call {n}: {hits}");
    }
}

#[test]
fn an_agent_can_add_a_long_note_find_a_passage_in_it_and_delete_it() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");
    let note: String = (0..30)
        .map(|i| {
            if i == 22 {
                format!("Sentence {i} is the only one about zebras. ")
            } else {
                format!("Sentence {i} says ownership keeps memory safe. ")
            }
        })
        .collect();
    let create = format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"create_table","arguments":{{"name":"notes","embedding":{{"model":"bag-of-words","base_url":"{base_url}","api_key_env":"DXMT_TEST_KEY_UNSET"}}}}}}}}"#
    );
    let add = serde_json::json!({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "add_document", "arguments": {
            "table": "notes", "id": "rust.md", "text": note, "chunk_size": 300,
            "metadata": {"title": "Rust"}
        }}
    })
    .to_string();
    let added = exchange(&wal, &[initialize(), &create, &add]);
    for r in &added[1..] {
        assert_eq!(r["result"]["isError"], serde_json::json!(false), "{r}");
    }
    let written: serde_json::Value =
        serde_json::from_str(added[2]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    let chunks = written["chunks"].as_u64().unwrap();
    assert!(chunks >= 4, "{written}");

    let searched = exchange(
        &wal,
        &[
            initialize(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"hybrid_search","arguments":{"table":"notes","text":"zebras","limit":1}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"delete_document","arguments":{"table":"notes","id":"rust.md"}}}"#,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"hybrid_search","arguments":{"table":"notes","text":"zebras","limit":1}}}"#,
        ],
    );
    let hit: serde_json::Value = serde_json::from_str(
        searched[1]["result"]["content"][0]["text"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    let top = &hit["hits"][0];
    assert!(
        top["payload"]["text"].as_str().unwrap().contains("zebras"),
        "{hit}"
    );
    assert_eq!(top["payload"]["doc_id"], "rust.md");
    assert_eq!(top["payload"]["title"], "Rust");
    let deleted = searched[2]["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    assert!(
        deleted.contains(&format!("\"deleted_chunks\":{chunks}")),
        "{deleted}"
    );
    let after: serde_json::Value = serde_json::from_str(
        searched[3]["result"]["content"][0]["text"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(after["hits"], serde_json::json!([]), "{after}");
}

#[test]
fn hybrid_search_with_text_on_a_table_without_keyword_search_says_so() {
    // The HTTP route refused this with FULL_TEXT_NOT_ENABLED; the check lived in the handler,
    // so through MCP the text was dropped and the vector ranking came back as the "fusion".
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");
    let replies = exchange(
        &wal,
        &[
            initialize(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"create_table","arguments":{"name":"t","dimension":4}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"insert_vector","arguments":{"table":"t","id":"a","vector":[1,0,0,0]}}}"#,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"hybrid_search","arguments":{"table":"t","vector":[1,0,0,0],"text":"anything"}}}"#,
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"hybrid_search","arguments":{"table":"t","vector":[1,0,0,0]}}}"#,
        ],
    );
    let refused = serde_json::to_string(&replies[3]).unwrap();
    assert!(
        refused.contains("full-text index")
            && replies[3]["result"]["isError"] != serde_json::json!(false),
        "{refused}"
    );
    // A vector alone has nothing to drop, and still works.
    assert_eq!(
        replies[4]["result"]["isError"],
        serde_json::json!(false),
        "{}",
        replies[4]
    );
}

/// `exchange`, with the server's default model pointed at a provider.
fn exchange_with_model(
    wal_dir: &std::path::Path,
    base_url: &str,
    requests: &[String],
) -> Vec<serde_json::Value> {
    exchange_with_model_and_env(wal_dir, base_url, requests, &[])
}

fn exchange_with_model_and_env(
    wal_dir: &std::path::Path,
    base_url: &str,
    requests: &[String],
    env: &[(&str, &str)],
) -> Vec<serde_json::Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("mcp")
        .arg("--wal-dir")
        .arg(wal_dir)
        .env("DXMT_EMBEDDING_MODEL", "bag-of-words")
        .env("DXMT_EMBEDDING_BASE_URL", base_url)
        .env("DXMT_EMBEDDING_AUTODETECT", "0")
        .envs(env.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");
    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "{}", initialize()).unwrap();
        for req in requests {
            writeln!(stdin, "{}", req).unwrap();
        }
    }
    drop(child.stdin.take());
    let replies: Vec<serde_json::Value> = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    child.wait().unwrap();
    replies
}

fn call(id: u64, tool: &str, arguments: serde_json::Value) -> String {
    serde_json::json!({
        "jsonrpc": "2.0", "id": id, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments}
    })
    .to_string()
}

fn content(reply: &serde_json::Value) -> serde_json::Value {
    assert_eq!(
        reply["result"]["isError"],
        serde_json::json!(false),
        "{reply}"
    );
    serde_json::from_str(reply["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
}

/// What a personal agent does with memory, through MCP with no table made by hand: keep facts
/// and preferences, replace a fact that changed without losing it, recall by topic and by kind,
/// audit the list, and forget on request. The memory table appears on first use.
/// With a reranker on the server (`DXMT_RERANK_*`), the memory table is made with it and
/// recall's order is the cross-encoder's. The two memories are built so the two orders
/// disagree: in the bag-of-words space the query sits next to the one that repeats "tomato",
/// while the stand-in reranker counts query words and prefers the one that also has "soil".
#[test]
fn recall_takes_the_rerankers_order_when_the_server_has_one() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let requests = [
        call(
            2,
            "remember",
            serde_json::json!({"id": "near", "text": "tomato tomato tomato tomato tomato tomato garden", "kind": "fact"}),
        ),
        call(
            3,
            "remember",
            serde_json::json!({"id": "apt", "text": "tomato garden soil water", "kind": "fact"}),
        ),
        call(
            4,
            "recall",
            serde_json::json!({"query": "tomato tomato tomato tomato garden soil", "limit": 1}),
        ),
    ];
    let first = |replies: &[serde_json::Value]| {
        content(&replies[3])["memories"][0]["id"]
            .as_str()
            .unwrap()
            .to_string()
    };

    let tmp = TempDir::new().unwrap();
    let plain = exchange_with_model(&tmp.path().join("wal"), &base_url, &requests);
    assert_eq!(first(&plain), "near", "{}", plain[3]);

    let tmp = TempDir::new().unwrap();
    let reranked = exchange_with_model_and_env(
        &tmp.path().join("wal"),
        &base_url,
        &requests,
        &[
            ("DXMT_RERANK_MODEL", "word-overlap"),
            ("DXMT_RERANK_BASE_URL", &base_url),
            ("DXMT_RERANK_API_KEY_ENV", "DXMT_TEST_RERANK_KEY_UNSET"),
        ],
    );
    assert_eq!(first(&reranked), "apt", "{}", reranked[3]);
}

/// Saying a fact twice keeps it once, refreshed and with both calls' tags; an event said twice
/// is two events; and the check can be turned off.
#[test]
fn a_memory_said_twice_is_kept_once_unless_it_is_an_event() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let requests = [
        call(
            2,
            "remember",
            serde_json::json!({"text": "The user has two cats", "kind": "fact", "tags": ["pets"]}),
        ),
        call(
            3,
            "remember",
            serde_json::json!({"text": "the user has two cats.", "kind": "fact", "tags": ["home"]}),
        ),
        call(
            4,
            "remember",
            serde_json::json!({"text": "The user went running", "kind": "event"}),
        ),
        call(
            5,
            "remember",
            serde_json::json!({"text": "The user went running", "kind": "event"}),
        ),
        call(
            6,
            "remember",
            serde_json::json!({"text": "The user has two cats", "kind": "preference"}),
        ),
        call(7, "list_memories", serde_json::json!({})),
    ];

    let tmp = TempDir::new().unwrap();
    let replies = exchange_with_model(&tmp.path().join("wal"), &base_url, &requests);
    let first = content(&replies[1]);
    let again = content(&replies[2]);
    assert_eq!(again["id"], first["id"], "{again}");
    assert_eq!(again["duplicate"], true, "{again}");
    assert_ne!(
        content(&replies[3])["id"],
        content(&replies[4])["id"],
        "an event that happens twice happened twice"
    );
    assert_ne!(
        content(&replies[5])["id"],
        first["id"],
        "the same words as another kind are another memory"
    );
    let listed = content(&replies[6]);
    assert_eq!(listed["total"], 4, "{listed}");
    let cats = listed["memories"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == first["id"])
        .unwrap();
    assert_eq!(cats["tags"], serde_json::json!(["pets", "home"]), "{cats}");
    assert_eq!(
        cats["text"], "The user has two cats",
        "the memory already held is kept"
    );

    let tmp = TempDir::new().unwrap();
    let replies = exchange_with_model_and_env(
        &tmp.path().join("wal"),
        &base_url,
        &requests,
        &[("DXMT_MEMORY_DUPLICATE_SIMILARITY", "off")],
    );
    assert_eq!(content(&replies[6])["total"], 5, "with the check off");
}

// DXMT_MCP_READ_ONLY: an agent that should consult the store and memory but never change them
// is offered only the tools that read, and a write it asks for anyway is refused.
#[test]
fn a_read_only_server_offers_and_runs_only_the_tools_that_read() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");
    // Something to read, written by an ordinary server first.
    exchange_with_model(
        &wal,
        &base_url,
        &[call(
            2,
            "remember",
            serde_json::json!({"id": "bees", "text": "The user keeps bees"}),
        )],
    );
    let replies = exchange_with_model_and_env(
        &wal,
        &base_url,
        &[
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#.to_string(),
            call(
                3,
                "remember",
                serde_json::json!({"text": "The user sells honey"}),
            ),
            call(
                4,
                "delete_vectors",
                serde_json::json!({"table_name": "memory", "ids": ["bees"]}),
            ),
            call(5, "recall", serde_json::json!({"query": "bees"})),
        ],
        &[("DXMT_MCP_READ_ONLY", "1")],
    );
    let mut names: Vec<&str> = replies[1]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    names.sort();
    let mut offered: Vec<&str> = dxmt_vectordb::mcp::server::READ_ONLY_TOOLS
        .iter()
        .copied()
        .filter(|n| names.contains(n))
        .collect();
    offered.sort();
    assert_eq!(names, offered, "only reading tools are listed");
    assert!(names.contains(&"recall") && names.contains(&"vector_search"));
    for refused in [&replies[2], &replies[3]] {
        assert!(
            refused.to_string().contains("read-only"),
            "a write is refused: {refused}"
        );
    }
    assert_eq!(
        content(&replies[4])["memories"][0]["id"],
        "bees",
        "reads work, and nothing was deleted"
    );
}

// A memory is read back by the id `remember` gave, superseded or not; a missing id is an error,
// not an empty object.
#[test]
fn a_memory_is_read_back_by_id() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let replies = exchange_with_model(
        &tmp.path().join("wal"),
        &base_url,
        &[
            call(
                2,
                "remember",
                serde_json::json!({"id": "home", "text": "The user lives in Shanghai", "kind": "fact", "tags": ["profile"], "source": "claude"}),
            ),
            call(
                3,
                "remember",
                serde_json::json!({"text": "The user lives in Berlin now", "kind": "fact", "supersedes": ["home"]}),
            ),
            call(4, "get_memory", serde_json::json!({"id": "home"})),
            call(
                5,
                "get_memory",
                serde_json::json!({"id": "nobody-has-this-id"}),
            ),
        ],
    );
    let home = content(&replies[3]);
    assert_eq!(home["id"], "home");
    assert_eq!(home["text"], "The user lives in Shanghai");
    assert_eq!(home["kind"], "fact");
    assert_eq!(home["tags"], serde_json::json!(["profile"]));
    assert_eq!(home["source"], "claude");
    assert_eq!(
        home["superseded_by"],
        content(&replies[2])["id"],
        "a superseded memory is still readable, with what replaced it: {home}"
    );
    let missing = &replies[4];
    assert!(
        missing.get("error").is_some() || missing["result"]["isError"] == serde_json::json!(true),
        "a missing id is an error: {missing}"
    );
}

// `score` is the fused rank, so the best hit scores near 1.0 however poor a match it is; a
// plugin that deleted "the single match above 0.9" deleted whatever came first. `similarity` is
// the cosine to the query, and says how close the hit actually is.
#[test]
fn recall_reports_how_similar_each_hit_is_not_only_its_rank() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let replies = exchange_with_model(
        &tmp.path().join("wal"),
        &base_url,
        &[
            call(
                2,
                "remember",
                serde_json::json!({"text": "garden roses bloom in june", "tags": ["a"]}),
            ),
            call(
                3,
                "recall",
                serde_json::json!({"query": "garden roses bloom in june", "tags": ["a"]}),
            ),
            call(
                4,
                "recall",
                serde_json::json!({"query": "tax return deadline garden", "tags": ["a"]}),
            ),
        ],
    );
    let same = &content(&replies[2])["memories"][0];
    let far = &content(&replies[3])["memories"][0];
    assert!(
        same["similarity"].as_f64().unwrap() > 0.99,
        "the same words: {same}"
    );
    assert!(
        far["score"].as_f64().unwrap() > 0.9,
        "the only hit ranks first whatever it is: {far}"
    );
    assert!(
        far["similarity"].as_f64().unwrap() < 0.9,
        "and its similarity says it is a poor match: {far}"
    );
}

// An agent that searched once and did not find the answer asked the same thing again and got
// the same ten memories back: there was no way to ask for the next ones. `offset` pages through
// the ranking; the pages do not overlap and together are the one long answer.
#[test]
fn recall_pages_through_the_ranking_with_offset() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let texts = [
        "Nate adopted a dog named Max",
        "Nate has had three turtles for years",
        "Nate bought a new tank for the turtles",
        "Nate walks the dog every morning",
        "Nate thinks about getting a cat",
    ];
    let mut requests: Vec<_> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            call(
                2 + i as u64,
                "remember",
                serde_json::json!({"text": t, "kind": "event", "tags": ["nate"]}),
            )
        })
        .collect();
    let query = |id: u64, limit: u64, offset: u64| {
        call(
            id,
            "recall",
            serde_json::json!({"query": "Nate's pets", "tags": ["nate"], "limit": limit, "offset": offset}),
        )
    };
    requests.push(query(10, 5, 0));
    requests.push(query(11, 2, 0));
    requests.push(query(12, 2, 2));
    requests.push(query(13, 2, 4));
    requests.push(query(14, 2, 6));
    let replies = exchange_with_model(&tmp.path().join("wal"), &base_url, &requests);
    let ids = |v: &serde_json::Value| -> Vec<String> {
        v["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap().to_string())
            .collect()
    };
    // replies[0] answers `initialize`; the five remembers are 1..=5.
    let all = ids(&content(&replies[6]));
    assert_eq!(all.len(), 5, "{all:?}");
    let pages: Vec<String> = (7..10).flat_map(|i| ids(&content(&replies[i]))).collect();
    assert_eq!(pages, all, "the pages, in order, are the long answer");
    assert!(
        ids(&content(&replies[10])).is_empty(),
        "past the end there is nothing"
    );
}

// The server derives memory with a chat model: facts out of a conversation, a person's profile
// out of their facts. Extracting a conversation again under the same prefix replaces its
// facts; the profile is remade in place and found by `recall` with kind `profile`.
#[test]
fn the_server_extracts_facts_and_keeps_a_profile() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let model_env = [
        ("DXMT_MEMORY_LLM_MODEL", "rule-chat"),
        ("DXMT_MEMORY_LLM_BASE_URL", base_url.as_str()),
        ("DXMT_MEMORY_LLM_API_KEY_ENV", "DXMT_TEST_CHAT_KEY_UNSET"),
    ];
    let conversation = "Caroline: I went to a counseling workshop last Friday\nMelanie: we roasted marshmallows while camping";
    let replies = exchange_with_model_and_env(
        &tmp.path().join("wal"),
        &base_url,
        &[
            call(
                2,
                "extract_memories",
                serde_json::json!({"text": conversation, "when": "10:37 am on 27 June, 2023", "tags": ["u1"], "id_prefix": "chat-1"}),
            ),
            // Again, shorter: the set from the first extraction is replaced, not added to.
            call(
                3,
                "extract_memories",
                serde_json::json!({"text": "Caroline: I went to a counseling workshop last Friday and loved it", "when": "10:37 am on 27 June, 2023", "tags": ["u1"], "id_prefix": "chat-1"}),
            ),
            call(4, "list_memories", serde_json::json!({"kind": "fact"})),
            call(
                5,
                "update_profile",
                serde_json::json!({"name": "Caroline", "tags": ["u1"]}),
            ),
            call(
                6,
                "update_profile",
                serde_json::json!({"name": "Caroline", "tags": ["u1"]}),
            ),
            call(
                7,
                "recall",
                serde_json::json!({"query": "Caroline", "kind": "profile", "tags": ["u1"]}),
            ),
            call(8, "update_profile", serde_json::json!({"name": "Nobody"})),
        ],
        &model_env,
    );
    let first = content(&replies[1]);
    let facts = first["facts"].as_array().unwrap();
    assert_eq!(facts.len(), 2, "{first}");
    assert_eq!(facts[0]["id"], "chat-1:F0");
    assert!(
        facts[0]["text"]
            .as_str()
            .unwrap()
            .starts_with("[10:37 am on 27 June, 2023] Caroline said:"),
        "a fact carries the session's date: {first}"
    );
    assert!(
        facts[0]["text"].as_str().unwrap().ends_with("(dated)"),
        "the prompt asked for relative times to be resolved: {first}"
    );

    let listed = content(&replies[3]);
    assert_eq!(
        listed["total"], 1,
        "extracting again replaced the set: {listed}"
    );
    assert!(listed["memories"][0]["text"]
        .as_str()
        .unwrap()
        .contains("loved it"));

    let profile = content(&replies[4]);
    assert_eq!(profile["facts"], 1, "{profile}");
    assert_eq!(content(&replies[5])["id"], profile["id"], "remade in place");
    let recalled = content(&replies[6]);
    let profiles = recalled["memories"].as_array().unwrap();
    assert_eq!(profiles.len(), 1, "one profile, not two: {recalled}");
    assert_eq!(profiles[0]["id"], profile["id"]);
    assert!(profiles[0]["text"]
        .as_str()
        .unwrap()
        .starts_with("Profile of Caroline:"));

    let nobody = &replies[7];
    assert!(
        nobody["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("no facts about 'Nobody'")),
        "{nobody}"
    );

    // Without a memory model, the tools say what to set.
    let tmp = TempDir::new().unwrap();
    let replies = exchange_with_model(
        &tmp.path().join("wal"),
        &base_url,
        &[call(
            2,
            "extract_memories",
            serde_json::json!({"text": conversation}),
        )],
    );
    assert!(
        replies[1].to_string().contains("DXMT_MEMORY_LLM_MODEL"),
        "{}",
        replies[1]
    );
}

// One block for the model, inside a budget: the profile first, then standing preferences and
// instructions, then the rest by relevance -- and never more than the budget.
#[test]
fn context_assembles_profile_preferences_and_memories_inside_a_budget() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let mut requests = vec![
        call(
            2,
            "remember",
            serde_json::json!({"text": "Profile of Nate: owns a dog and three turtles; plays video games", "kind": "profile", "tags": ["nate"]}),
        ),
        call(
            3,
            "remember",
            serde_json::json!({"text": "Nate wants short answers about his pets", "kind": "preference", "tags": ["nate"]}),
        ),
        call(
            4,
            "remember",
            serde_json::json!({"text": "Always mention the turtles by name", "kind": "instruction", "tags": ["nate"]}),
        ),
    ];
    for i in 0..30 {
        requests.push(call(
            5 + i,
            "remember",
            serde_json::json!({"text": format!("Nate's pets story number {i} about the dog and turtles at the lake"), "kind": "event", "tags": ["nate"]}),
        ));
    }
    requests.push(call(
        40,
        "get_context",
        serde_json::json!({"query": "Nate pets dog turtles", "tags": ["nate"], "max_tokens": 200}),
    ));
    requests.push(call(41, "get_context", serde_json::json!({"query": "  "})));
    let replies = exchange_with_model(&tmp.path().join("wal"), &base_url, &requests);
    let block = content(&replies[replies.len() - 2]);
    let text = block["context"].as_str().unwrap();
    let about = text.find("## About the user").expect(text);
    let standing = text.find("## Preferences and instructions").expect(text);
    let relevant = text.find("## Relevant memories").expect(text);
    assert!(about < standing && standing < relevant, "{text}");
    assert!(
        text.contains("[instruction] Always mention the turtles by name"),
        "{text}"
    );
    assert!(block["tokens"].as_u64().unwrap() <= 200, "{block}");
    let events = text.matches("[event]").count();
    assert!(
        events > 0 && events < 30,
        "the budget cut the long tail: {events} events"
    );
    assert_eq!(block["memories"][0]["kind"], "profile", "{block}");
    let empty = &replies[replies.len() - 1];
    assert!(
        empty.to_string().contains("context needs `query`"),
        "{empty}"
    );
}

#[test]
fn an_agent_remembers_recalls_supersedes_and_forgets() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");
    let replies = exchange_with_model(
        &wal,
        &base_url,
        &[
            call(
                2,
                "remember",
                serde_json::json!({"id": "home", "text": "The user lives in Shanghai", "kind": "fact", "tags": ["profile"], "source": "claude"}),
            ),
            call(
                3,
                "remember",
                serde_json::json!({"text": "The user prefers answers in Chinese", "kind": "preference"}),
            ),
            call(
                4,
                "remember",
                serde_json::json!({"text": "The user moved to Berlin and lives in Berlin now", "kind": "fact", "tags": ["profile"], "supersedes": ["home"]}),
            ),
            call(
                5,
                "recall",
                serde_json::json!({"query": "where does the user live", "limit": 5}),
            ),
            call(
                6,
                "recall",
                serde_json::json!({"query": "where does the user live", "limit": 5, "include_superseded": true}),
            ),
            call(
                7,
                "recall",
                serde_json::json!({"query": "the user", "kind": "preference"}),
            ),
            call(
                8,
                "recall",
                serde_json::json!({"query": "the user", "tags": ["profile"]}),
            ),
            call(9, "list_memories", serde_json::json!({})),
        ],
    );
    let kept = content(&replies[1]);
    assert_eq!(kept["id"], "home");
    let moved = content(&replies[3]);
    assert_eq!(moved["superseded"], serde_json::json!(["home"]), "{moved}");
    let moved_id = moved["id"].as_str().unwrap().to_string();

    let ids = |v: &serde_json::Value| -> Vec<String> {
        v["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap().to_string())
            .collect()
    };
    let current = content(&replies[4]);
    assert!(ids(&current).contains(&moved_id), "{current}");
    assert!(
        !ids(&current).contains(&"home".to_string()),
        "a superseded fact is not recalled: {current}"
    );
    let history = content(&replies[5]);
    let old = history["memories"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "home")
        .expect("history includes the old fact");
    assert_eq!(old["superseded_by"], moved_id.as_str());
    assert_eq!(old["source"], "claude");
    assert!(old["at"].as_i64().unwrap() > 1_700_000_000, "{old}");

    let preferences = content(&replies[6]);
    assert_eq!(
        preferences["memories"].as_array().unwrap().len(),
        1,
        "{preferences}"
    );
    assert_eq!(preferences["memories"][0]["kind"], "preference");
    let tagged = content(&replies[7]);
    assert_eq!(
        ids(&tagged),
        vec![moved_id.clone()],
        "only the current profile fact: {tagged}"
    );
    let listed = content(&replies[8]);
    assert_eq!(listed["total"], 3, "{listed}");

    // A second session -- another agent -- sees the same memory, and can forget part of it.
    let pref_id = preferences["memories"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    let later = exchange_with_model(
        &wal,
        &base_url,
        &[
            call(2, "forget", serde_json::json!({"ids": [pref_id]})),
            call(3, "list_memories", serde_json::json!({})),
            call(
                4,
                "remember",
                serde_json::json!({"text": "x", "metadata": {"kind": "sneaky"}}),
            ),
        ],
    );
    assert_eq!(content(&later[1])["forgotten"], 1);
    assert_eq!(content(&later[2])["total"], 2);
    // A tool's refusal comes back as this server's tools all refuse: a JSON-RPC error.
    assert!(
        later[3]["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("metadata may not set `kind`")),
        "metadata may not set a memory's own fields: {}",
        later[3]
    );
}

#[path = "../common/http.rs"]
mod http;

/// Two ways in, one memory. With `vectordb serve` holding the store, `vectordb mcp` on the same
/// store relays to the server instead of failing on the lock; what it remembers is recalled over
/// REST, and `POST /mcp` answers directly too.
#[test]
fn mcp_relays_to_a_running_server_and_shares_its_memory() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("wal");
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    struct Kill(std::process::Child);
    impl Drop for Kill {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let _server = Kill(
        Command::new(env!("CARGO_BIN_EXE_vectordb"))
            .args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()])
            .env("DXMT_WAL_DIR", &wal)
            .env("DXMT_EMBEDDING_MODEL", "bag-of-words")
            .env("DXMT_EMBEDDING_BASE_URL", &base_url)
            .env("DXMT_EMBEDDING_AUTODETECT", "0")
            .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
            .env_remove("DXMT_API_KEY")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while http::call(port, "GET", "/live", &[], None).is_none() {
        assert!(std::time::Instant::now() < deadline, "server never came up");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // MCP over HTTP, directly.
    let (status, listed) = http::call(
        port,
        "POST",
        "/mcp",
        &[],
        Some(r#"{"jsonrpc":"2.0","id":7,"method":"tools/list"}"#),
    )
    .unwrap();
    assert_eq!(status, 200, "{listed}");
    let listed: serde_json::Value = serde_json::from_str(&listed).unwrap();
    assert_eq!(listed["id"], 7);
    assert!(listed["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["name"] == "recall"));
    let (status, _) = http::call(
        port,
        "POST",
        "/mcp",
        &[],
        Some(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#),
    )
    .unwrap();
    assert_eq!(status, 202, "a notification has no reply");

    // The stdio process on the same store: relayed, not refused.
    let mut child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .args(["mcp", "--wal-dir"])
        .arg(&wal)
        .env("DXMT_PORT", port.to_string())
        .env_remove("DXMT_MCP_SERVER")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "{}", initialize()).unwrap();
        writeln!(
            stdin,
            "{}",
            call(
                2,
                "remember",
                serde_json::json!({"text": "The user keeps bees on the roof", "kind": "fact"})
            )
        )
        .unwrap();
    }
    drop(child.stdin.take());
    let replies: Vec<serde_json::Value> = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    child.wait().unwrap();
    assert_eq!(replies.len(), 2, "{replies:?}");
    let kept = content(&replies[1]);

    let (status, recalled) = http::call(
        port,
        "POST",
        "/api/v1/memories/recall",
        &[],
        Some(r#"{"query":"bees","limit":3}"#),
    )
    .unwrap();
    assert_eq!(status, 200, "{recalled}");
    let recalled: serde_json::Value = serde_json::from_str(&recalled).unwrap();
    assert_eq!(
        recalled["data"]["memories"][0]["id"], kept["id"],
        "{recalled}"
    );

    // Relayed with --read-only to the same, writable server: the relay lists only the tools that
    // read and refuses the rest itself; a read goes through.
    let mut child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .args(["mcp", "--read-only", "--wal-dir"])
        .arg(&wal)
        .env("DXMT_PORT", port.to_string())
        .env_remove("DXMT_MCP_SERVER")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "{}", initialize()).unwrap();
        writeln!(stdin, r#"{{"jsonrpc":"2.0","id":2,"method":"tools/list"}}"#).unwrap();
        writeln!(
            stdin,
            "{}",
            call(
                3,
                "remember",
                serde_json::json!({"text": "The user sells honey"})
            )
        )
        .unwrap();
        writeln!(
            stdin,
            "{}",
            call(
                4,
                "recall",
                serde_json::json!({"query": "bees", "limit": 3})
            )
        )
        .unwrap();
    }
    drop(child.stdin.take());
    let replies: Vec<serde_json::Value> = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    child.wait().unwrap();
    let names: Vec<&str> = replies[1]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(
        names.contains(&"recall") && !names.contains(&"remember"),
        "{names:?}"
    );
    assert!(
        replies[2].to_string().contains("read-only"),
        "a write is refused by the relay: {}",
        replies[2]
    );
    assert_eq!(content(&replies[3])["memories"][0]["id"], kept["id"]);
}
