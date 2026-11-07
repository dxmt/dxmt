//! Tenants: per-tenant keys from `DXMT_TENANTS_FILE`, each reaching only its own memories.
//!
//! Every way one tenant could see or change another's memory is tried here and must fail: recall
//! (with and without the other's tag), the duplicate check, an explicit id, `supersedes`,
//! `forget`, the list, context, extraction under the same prefix, and the routes outside memory.
#![cfg(feature = "server-embedding")]

use sha2::{Digest, Sha256};
use std::process::{Command, Stdio};
use std::time::Duration;

#[path = "../fake_provider/mod.rs"]
mod fake_provider;
#[path = "../common/http.rs"]
mod http;

struct Kill(std::process::Child);
impl Drop for Kill {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

const ADMIN: &str = "admin-key-for-tests";
const KEY_A: &str = "tenant-a-key-for-tests";
const KEY_B: &str = "tenant-b-key-for-tests";

fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn serve(dir: &std::path::Path, base_url: &str) -> (Kill, u16) {
    let file = dir.join("tenants.json");
    std::fs::write(
        &file,
        serde_json::json!({"tenants": [
            {"id": "a", "key_sha256": sha256_hex(KEY_A)},
            {"id": "b", "key_sha256": sha256_hex(KEY_B)},
        ]})
        .to_string(),
    )
    .unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let server = Kill(
        Command::new(env!("CARGO_BIN_EXE_vectordb"))
            .args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()])
            .env("DXMT_WAL_DIR", dir.join("wal"))
            .env("DXMT_EMBEDDING_MODEL", "bag-of-words")
            .env("DXMT_EMBEDDING_BASE_URL", base_url)
            .env("DXMT_EMBEDDING_AUTODETECT", "0")
            .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
            .env("DXMT_MEMORY_LLM_MODEL", "rule-chat")
            .env("DXMT_MEMORY_LLM_BASE_URL", base_url)
            .env("DXMT_MEMORY_LLM_API_KEY_ENV", "DXMT_TEST_CHAT_KEY_UNSET")
            .env("DXMT_API_KEY", ADMIN)
            .env("DXMT_TENANTS_FILE", &file)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while http::call(port, "GET", "/live", &[], None).is_none() {
        assert!(std::time::Instant::now() < deadline, "server never came up");
        std::thread::sleep(Duration::from_millis(100));
    }
    (server, port)
}

/// (status, body) for a request with `key`.
fn call(
    port: u16,
    key: Option<&str>,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> (u16, serde_json::Value) {
    let headers: Vec<(&str, &str)> = key.map(|k| vec![("api-key", k)]).unwrap_or_default();
    let text = body.map(|b| b.to_string());
    let reply = http::request(
        port,
        method,
        path,
        &headers,
        text.as_deref().map(str::as_bytes),
        Duration::from_secs(30),
    )
    .expect("no answer");
    let value = serde_json::from_slice(&reply.body).unwrap_or(serde_json::Value::Null);
    (reply.status, value)
}

fn remember(port: u16, key: &str, body: serde_json::Value) -> (u16, serde_json::Value) {
    call(port, Some(key), "POST", "/api/v1/memories", Some(body))
}

fn texts(value: &serde_json::Value) -> Vec<String> {
    value["data"]["memories"]
        .as_array()
        .map(|m| {
            m.iter()
                .map(|m| m["text"].as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn a_tenant_reaches_only_its_own_memories() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = tempfile::TempDir::new().unwrap();
    let (_server, port) = serve(tmp.path(), &base_url);

    // No key, or a wrong one: 401, tenants configured or not.
    assert_eq!(call(port, None, "GET", "/api/v1/memories", None).0, 401);
    assert_eq!(
        call(port, Some("nope"), "GET", "/api/v1/memories", None).0,
        401
    );

    let (status, kept) = remember(
        port,
        KEY_A,
        serde_json::json!({"text": "The user likes green tea", "kind": "preference"}),
    );
    assert_eq!(status, 200, "{kept}");
    let id_a = kept["data"]["id"].as_str().unwrap().to_string();

    // The same words from B: B's own memory, not A's refreshed (and A's id never shown to B).
    let (status, kept_b) = remember(
        port,
        KEY_B,
        serde_json::json!({"text": "The user likes green tea", "kind": "preference"}),
    );
    assert_eq!(status, 200, "{kept_b}");
    assert_ne!(
        kept_b["data"]["id"], id_a,
        "B was handed A's memory: {kept_b}"
    );
    assert_ne!(kept_b["data"]["duplicate"], true, "{kept_b}");
    let id_b = kept_b["data"]["id"].as_str().unwrap().to_string();
    remember(
        port,
        KEY_A,
        serde_json::json!({"text": "Alice keeps bees on the roof", "kind": "fact"}),
    );

    // Recall: B sees only B's, even naming A's tag.
    let (_, recalled) = call(
        port,
        Some(KEY_B),
        "POST",
        "/api/v1/memories/recall",
        Some(serde_json::json!({"query": "bees green tea", "limit": 10})),
    );
    let seen = texts(&recalled);
    assert!(!seen.iter().any(|t| t.contains("bees")), "{recalled}");
    assert_eq!(recalled["data"]["memories"].as_array().unwrap().len(), 1);
    let (_, recalled) = call(
        port,
        Some(KEY_B),
        "POST",
        "/api/v1/memories/recall",
        Some(serde_json::json!({"query": "bees", "tags": ["tenant:a"]})),
    );
    assert!(texts(&recalled).is_empty(), "{recalled}");

    // Writing into A's space: a `tenant:` tag, A's id, superseding A's memory -- all refused.
    let (status, refused) = remember(
        port,
        KEY_B,
        serde_json::json!({"text": "planted", "tags": ["tenant:a"]}),
    );
    assert_eq!(status, 400, "{refused}");
    let (status, refused) = remember(
        port,
        KEY_B,
        serde_json::json!({"text": "overwritten", "id": id_a}),
    );
    assert_eq!(status, 400, "{refused}");
    let (status, refused) = remember(
        port,
        KEY_B,
        serde_json::json!({"text": "superseded", "supersedes": [id_a]}),
    );
    assert_eq!(status, 404, "{refused}");

    // Read by id: A's memory is "not found" for B -- the same answer as no memory -- and there
    // for A.
    let (status, body) = call(
        port,
        Some(KEY_B),
        "GET",
        &format!("/api/v1/memories/{id_a}"),
        None,
    );
    assert_eq!(status, 404, "B read A's memory by id: {body}");
    let (status, body) = call(
        port,
        Some(KEY_A),
        "GET",
        &format!("/api/v1/memories/{id_a}"),
        None,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["data"]["text"], "The user likes green tea");

    // Forget: A's id is not there for B; it is still there for A.
    let (_, forgot) = call(
        port,
        Some(KEY_B),
        "DELETE",
        &format!("/api/v1/memories/{id_a}"),
        None,
    );
    assert_eq!(forgot["data"]["forgotten"], 0, "{forgot}");

    // The list and the context: each tenant's own.
    let (_, listed_b) = call(port, Some(KEY_B), "GET", "/api/v1/memories", None);
    assert_eq!(listed_b["data"]["total"], 1, "{listed_b}");
    let (_, listed_a) = call(port, Some(KEY_A), "GET", "/api/v1/memories", None);
    assert_eq!(listed_a["data"]["total"], 2, "{listed_a}");
    let (_, block) = call(
        port,
        Some(KEY_B),
        "POST",
        "/api/v1/memories/context",
        Some(serde_json::json!({"query": "what does the user like, bees"})),
    );
    let block = block["data"]["context"].as_str().unwrap_or("").to_string();
    assert!(
        block.contains("green tea") && !block.contains("bees"),
        "{block}"
    );

    // Extraction under the same prefix: B's run replaces B's set, never A's.
    for key in [KEY_A, KEY_B] {
        let (status, out) = call(
            port,
            Some(key),
            "POST",
            "/api/v1/memories/extract",
            Some(
                serde_json::json!({"text": "Nate: I adopted a dog named Max", "id_prefix": "chat-1"}),
            ),
        );
        assert_eq!(status, 200, "{out}");
    }
    let (_, listed_a) = call(port, Some(KEY_A), "GET", "/api/v1/memories?kind=fact", None);
    assert!(
        texts(&listed_a).iter().any(|t| t.contains("Max")),
        "B's extraction removed A's facts: {listed_a}"
    );

    // Everything outside memory and the gateway is the admin's.
    for (method, path) in [
        ("GET", "/api/v1/tables"),
        ("GET", "/api/v1/tables/memory"),
        ("POST", "/api/v1/snapshot"),
        ("GET", "/status"),
    ] {
        let (status, body) = call(port, Some(KEY_B), method, path, None);
        assert_eq!(status, 403, "{method} {path}: {body}");
    }
    let (status, _) = call(
        port,
        Some(KEY_B),
        "POST",
        "/mcp",
        Some(serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})),
    );
    assert_eq!(status, 403);

    // The admin sees every memory, both tenants', and the tables.
    let (_, everything) = call(port, Some(ADMIN), "GET", "/api/v1/memories", None);
    assert!(
        everything["data"]["total"].as_u64().unwrap() >= 4,
        "{everything}"
    );
    assert_eq!(
        call(port, Some(ADMIN), "GET", "/api/v1/tables", None).0,
        200
    );
    // And A's memory survived everything B tried.
    let (_, a_mem) = call(
        port,
        Some(KEY_A),
        "POST",
        "/api/v1/memories/recall",
        Some(serde_json::json!({"query": "green tea"})),
    );
    assert!(a_mem["data"]["memories"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["id"] == id_a));
    let _ = id_b;
}

#[test]
fn a_bad_tenants_file_stops_the_server() {
    let tmp = tempfile::TempDir::new().unwrap();
    let file = tmp.path().join("tenants.json");
    std::fs::write(&file, r#"{"tenants":[{"id":"a","key_sha256":"not-hex"}]}"#).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .args(["serve", "--host", "127.0.0.1", "--port", "0"])
        .env("DXMT_WAL_DIR", tmp.path().join("wal"))
        .env("DXMT_TENANTS_FILE", &file)
        .env("DXMT_EMBEDDING_AUTODETECT", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(
        !status.success(),
        "a malformed tenants file must not start a server"
    );
}
