//! A first run with no configuration but an embedding model: the server says where everything
//! is, a table can be created with neither a dimension nor a model, the knowledge-base table
//! is made by its first write, and a text-only hybrid search uses the model as well as the
//! keywords.
//!
//! The model here is `tests/fake_provider` named through `DXMT_EMBEDDING_MODEL`, the same
//! route a user's OpenAI or ollama endpoint takes; detection of a local ollama is switched
//! off so the test means the same on a machine that has one.

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::Ordering;

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::record::CreateRecordRequest,
};

#[path = "../common/http.rs"]
mod http;

#[path = "../fake_provider/mod.rs"]
mod fake_provider;
use fake_provider::Behaviour;

const DIM: usize = 32;

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

fn start(root: &Path, provider_url: &str, stdout: &Path) -> Server {
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()])
        .env("DXMT_ROOT_DIR", root)
        .env_remove("DXMT_WAL_DIR")
        .env_remove("DXMT_KB_TABLE")
        .env("DXMT_EMBEDDING_MODEL", "fake-embedder")
        .env("DXMT_EMBEDDING_BASE_URL", provider_url)
        .env("DXMT_EMBEDDING_API_KEY_ENV", "DXMT_TEST_NO_SUCH_KEY")
        .env("DXMT_EMBEDDING_AUTODETECT", "0")
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .stdout(std::fs::File::create(stdout).unwrap())
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

fn json(port: u16, method: &str, path: &str, body: serde_json::Value) -> serde_json::Value {
    let (_, text) = http::call(port, method, path, &[], Some(&body.to_string())).expect("answer");
    serde_json::from_str(&text).unwrap_or_else(|_| panic!("not JSON: {text}"))
}

fn ids(answer: &serde_json::Value) -> Vec<String> {
    answer["data"]["hits"]
        .as_array()
        .unwrap_or_else(|| panic!("no hits: {answer}"))
        .iter()
        .map(|h| h["id"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn a_first_run_with_a_model_needs_no_configuration_for_notes() {
    let root = tempfile::tempdir().unwrap();
    let stdout = root.path().join("stdout.txt");
    let (url, provider) = fake_provider::serve_on_a_thread(Behaviour::Normal, DIM);
    let server = start(&root.path().join("db"), &url, &stdout);
    let port = server.port;

    // The startup message, on stdout with no RUST_LOG.
    let printed = std::fs::read_to_string(&stdout).unwrap();
    assert!(printed.contains("is running"), "{printed}");
    assert!(
        printed.contains("fake-embedder (DXMT_EMBEDDING_MODEL)"),
        "{printed}"
    );
    assert!(printed.contains("table 'kb'"), "{printed}");
    assert!(printed.contains("/tables/kb/records"), "{printed}");

    // A table with neither a dimension nor a model takes the server's, and its width from it.
    let created = json(
        port,
        "POST",
        "/api/v1/tables",
        serde_json::json!({"name": "plain"}),
    );
    assert_eq!(created["success"], true, "{created}");
    assert_eq!(created["data"]["dimension"], DIM, "{created}");
    assert_eq!(created["data"]["embedding"]["model"], "fake-embedder");
    // One given a dimension stores the caller's vectors and gets no model it did not ask for.
    let sized = json(
        port,
        "POST",
        "/api/v1/tables",
        serde_json::json!({"name": "sized", "dimension": 4}),
    );
    assert_eq!(sized["success"], true, "{sized}");
    assert!(sized["data"]["embedding"].is_null(), "{sized}");

    // The knowledge-base table is made by its first write.
    let inserted = json(
        port,
        "POST",
        "/api/v1/tables/kb/records",
        serde_json::json!({"records": [
            {"id": "graph", "text": "hnsw builds a graph index for vector search"},
            {"id": "dinner", "text": "tomato and egg for dinner tonight"},
            {"id": "rust", "text": "ownership keeps rust memory safe"}
        ]}),
    );
    assert_eq!(inserted["success"], true, "{inserted}");
    let kb = json(port, "GET", "/api/v1/tables/kb", serde_json::json!({}));
    assert_eq!(kb["data"]["table"]["dimension"], DIM, "{kb}");

    // A text-only hybrid search embeds the text: the dense half runs, not the keywords alone.
    std::thread::sleep(std::time::Duration::from_millis(1500)); // the text index commits async
    let before = provider.calls.load(Ordering::SeqCst);
    let found = json(
        port,
        "POST",
        "/api/v1/tables/kb/search/hybrid",
        serde_json::json!({"text": "vector search graph", "limit": 2}),
    );
    assert_eq!(found["success"], true, "{found}");
    assert!(
        provider.calls.load(Ordering::SeqCst) > before,
        "a text-only hybrid search did not embed its text"
    );
    assert_eq!(
        ids(&found).first().map(String::as_str),
        Some("graph"),
        "{found}"
    );
}

#[tokio::test]
async fn without_a_model_the_knowledge_base_table_is_not_made() {
    let dir = tempfile::tempdir().unwrap();
    let storage = IntelligentIndexedStorage::new_with_wal(
        &StorageConfig {
            memory: Some(MemoryConfig {
                max_size_mb: 128,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        },
        Some(dir.path().join("wal")),
    )
    .await
    .unwrap();
    assert!(storage.default_embedding().is_none());
    let err = storage
        .insert_records(
            "kb",
            vec![CreateRecordRequest {
                id: "a".to_string(),
                vector: Vec::new(),
                sparse_vector: None,
                named_vectors: None,
                payload: None,
                text: Some("a note".to_string()),
            }],
        )
        .await
        .expect_err("a table was made with no model to make it with");
    // It says what is missing and how to get it, not `TABLE_NOT_FOUND`, which is what a
    // first run that followed the README without ollama was told.
    assert_eq!(err.status_code(), 400, "{err}");
    assert!(err.to_string().contains("ollama pull bge-m3"), "{err}");
    assert!(storage.list_tables().await.unwrap().is_empty());
}
