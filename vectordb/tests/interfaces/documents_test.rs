//! Documents over HTTP: a long text in, overlapping chunks stored as records, searchable,
//! replaced as a whole, deleted as a whole. The model is `tests/fake_provider` named through
//! `DXMT_EMBEDDING_MODEL`, the route a user's ollama takes.

use std::path::Path;
use std::process::{Child, Command, Stdio};

#[path = "../common/http.rs"]
mod http;

#[path = "../fake_provider/mod.rs"]
mod fake_provider;
use fake_provider::Behaviour;

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

fn start(root: &Path, provider_url: &str) -> Server {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
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

fn call(
    port: u16,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> (u16, serde_json::Value) {
    let body = body.map(|b| b.to_string());
    let (status, text) = http::call(port, method, path, &[], body.as_deref()).expect("answer");
    (
        status,
        serde_json::from_str(&text).unwrap_or_else(|_| panic!("not JSON: {text}")),
    )
}

/// Forty numbered sentences, one of which carries a word nothing else does.
fn long_note(sentences: usize) -> String {
    (0..sentences)
        .map(|i| {
            if i == 31 {
                format!("第{i}句：斑马 zebra 只在这一句里出现。")
            } else {
                format!("第{i}句：向量数据库把文本变成向量，再按相似度检索。")
            }
        })
        .collect()
}

#[test]
fn a_document_is_chunked_searched_replaced_and_deleted_as_one() {
    let root = tempfile::tempdir().unwrap();
    let (url, _provider) = fake_provider::serve_on_a_thread(Behaviour::Normal, 32);
    let server = start(&root.path().join("db"), &url);
    let port = server.port;
    let path = "/api/v1/tables/kb/documents/notes%2Fdb.md";

    // In: 40 sentences of ~26 characters, 200 to a chunk.
    let text = long_note(40);
    let (status, put) = call(
        port,
        "PUT",
        path,
        Some(serde_json::json!({
            "text": text, "chunk_size": 200, "chunk_overlap": 20,
            "metadata": {"title": "数据库笔记"}
        })),
    );
    assert_eq!(status, 200, "{put}");
    let chunks = put["data"]["chunks"].as_u64().unwrap() as usize;
    assert!(chunks >= 5, "{put}");
    assert_eq!(put["data"]["removed"], 0, "{put}");

    // Read back: in order, under the decoded id, with the metadata and no bookkeeping in it.
    let (status, got) = call(port, "GET", path, None);
    assert_eq!(status, 200, "{got}");
    assert_eq!(got["data"]["doc_id"], "notes/db.md");
    assert_eq!(
        got["data"]["metadata"],
        serde_json::json!({"title": "数据库笔记"})
    );
    let stored = got["data"]["chunks"].as_array().unwrap();
    assert_eq!(stored.len(), chunks);
    for (i, c) in stored.iter().enumerate() {
        assert_eq!(c["chunk_index"], i);
        assert_eq!(c["id"], format!("notes/db.md#{i}"));
        assert!(c["text"].as_str().unwrap().chars().count() <= 200);
    }

    // Search finds the chunk holding the rare word, and says which document it belongs to.
    std::thread::sleep(std::time::Duration::from_millis(1500)); // the text index commits async
    let (status, found) = call(
        port,
        "POST",
        "/api/v1/tables/kb/search/hybrid",
        Some(serde_json::json!({"text": "斑马 zebra", "limit": 3})),
    );
    assert_eq!(status, 200, "{found}");
    let top = &found["data"]["hits"][0];
    assert!(
        top["payload"]["text"].as_str().unwrap().contains("zebra"),
        "{found}"
    );
    assert_eq!(top["payload"]["doc_id"], "notes/db.md");
    assert_eq!(top["payload"]["title"], "数据库笔记");

    // Written again, shorter: the chunks past the new count go.
    let (status, again) = call(
        port,
        "PUT",
        path,
        Some(serde_json::json!({"text": long_note(10), "chunk_size": 200, "chunk_overlap": 20})),
    );
    assert_eq!(status, 200, "{again}");
    let now = again["data"]["chunks"].as_u64().unwrap() as usize;
    assert!(now < chunks, "{again}");
    assert_eq!(
        again["data"]["removed"].as_u64().unwrap() as usize,
        chunks - now
    );
    let (_, got) = call(port, "GET", path, None);
    assert_eq!(got["data"]["chunks"].as_array().unwrap().len(), now);
    assert_eq!(
        got["data"]["metadata"],
        serde_json::json!({}),
        "metadata is per write"
    );

    // Out, as a whole.
    let (status, deleted) = call(port, "DELETE", path, None);
    assert_eq!(status, 200, "{deleted}");
    assert_eq!(
        deleted["data"]["deleted_chunks"].as_u64().unwrap() as usize,
        now
    );
    let (status, gone) = call(port, "GET", path, None);
    assert_eq!(status, 404, "{gone}");
}

#[test]
fn what_a_document_write_refuses() {
    let root = tempfile::tempdir().unwrap();
    let (url, _provider) = fake_provider::serve_on_a_thread(Behaviour::Normal, 32);
    let server = start(&root.path().join("db"), &url);
    let port = server.port;

    // A table of caller-computed vectors has no model to embed chunks with.
    let (status, _) = call(
        port,
        "POST",
        "/api/v1/tables",
        Some(serde_json::json!({"name": "raw", "dimension": 4})),
    );
    assert_eq!(status, 200);
    let refusals = [
        (
            "/api/v1/tables/raw/documents/d",
            serde_json::json!({"text": "一些文字。"}),
            "embedding",
        ),
        (
            "/api/v1/tables/kb/documents/d",
            serde_json::json!({"text": "  \n "}),
            "no text",
        ),
        (
            "/api/v1/tables/kb/documents/d",
            serde_json::json!({"text": "文字。", "metadata": {"doc_id": "other"}}),
            "doc_id",
        ),
        (
            "/api/v1/tables/kb/documents/d",
            serde_json::json!({"text": "文字。", "chunk_size": 100, "chunk_overlap": 50}),
            "chunk_overlap",
        ),
    ];
    for (path, body, says) in refusals {
        let (status, answer) = call(port, "PUT", path, Some(body));
        assert_eq!(status, 400, "{path}: {answer}");
        assert!(answer.to_string().contains(says), "{path}: {answer}");
    }
    let (status, answer) = call(port, "GET", "/api/v1/tables/kb/documents/never", None);
    assert_eq!(status, 404, "{answer}");
}
