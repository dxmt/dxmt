//! A provider slower than `DXMT_EMBEDDING_TIMEOUT_SECS` for a batch: the batch is split until
//! it fits, the size that fitted is kept for later writes, and one text too slow on its own
//! fails with a message that says what to change.
//!
//! In a spawned server, because the timeout is read once into the process's HTTP client.

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

fn start(root: &Path) -> Server {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()])
        .env("DXMT_ROOT_DIR", root)
        .env_remove("DXMT_WAL_DIR")
        .env("DXMT_EMBEDDING_AUTODETECT", "0")
        .env("DXMT_EMBEDDING_TIMEOUT_SECS", "1")
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

fn call(port: u16, method: &str, path: &str, body: serde_json::Value) -> (u16, serde_json::Value) {
    let (status, text) =
        http::call(port, method, path, &[], Some(&body.to_string())).expect("answer");
    (status, serde_json::from_str(&text).unwrap())
}

fn notes(prefix: &str, n: usize) -> serde_json::Value {
    serde_json::json!({"records": (0..n)
        .map(|i| serde_json::json!({"id": format!("{prefix}{i}"), "text": format!("note {i}")}))
        .collect::<Vec<_>>()})
}

#[test]
fn a_batch_too_slow_for_the_timeout_is_split_and_the_size_kept() {
    // 300 ms a text against a one-second timeout: eight texts take 2.4 s, four 1.2 s, two fit.
    let (url, provider) = fake_provider::serve_on_a_thread(Behaviour::SlowPerInput(300), 16);
    let root = tempfile::tempdir().unwrap();
    let server = start(&root.path().join("db"));
    let port = server.port;
    let (status, created) = call(
        port,
        "POST",
        "/api/v1/tables",
        serde_json::json!({"name": "slow", "dimension": 16, "embedding": {
            "model": "slow-model", "base_url": url, "api_key_env": "DXMT_TEST_NO_SUCH_KEY"}}),
    );
    assert_eq!(status, 200, "{created}");

    let (status, inserted) = call(port, "POST", "/api/v1/tables/slow/records", notes("a", 8));
    assert_eq!(status, 200, "{inserted}");
    let first: Vec<usize> = provider.batches.lock().unwrap().clone();
    // 8 timed out, then 4, then the 2s went through -- and nothing was sent three times over.
    assert_eq!(&first[..3], &[8, 4, 2], "{first:?}");
    assert!(first.iter().filter(|&&n| n == 8).count() == 1, "{first:?}");

    // The next write starts at the size that fitted: no timeout to pay first.
    provider.batches.lock().unwrap().clear();
    let (status, inserted) = call(port, "POST", "/api/v1/tables/slow/records", notes("b", 8));
    assert_eq!(status, 200, "{inserted}");
    assert_eq!(*provider.batches.lock().unwrap(), vec![2, 2, 2, 2]);

    // Every record has its vector: a search by text finds one.
    let (status, found) = call(
        port,
        "POST",
        "/api/v1/tables/slow/search",
        serde_json::json!({"text": "note 3", "top_k": 1}),
    );
    assert_eq!(status, 200, "{found}");
}

#[test]
fn one_text_too_slow_on_its_own_says_which_setting_to_raise() {
    let (url, _provider) = fake_provider::serve_on_a_thread(Behaviour::SlowPerInput(1500), 16);
    let root = tempfile::tempdir().unwrap();
    let server = start(&root.path().join("db"));
    let port = server.port;
    call(
        port,
        "POST",
        "/api/v1/tables",
        serde_json::json!({"name": "slow", "dimension": 16, "embedding": {
            "model": "slow-model", "base_url": url, "api_key_env": "DXMT_TEST_NO_SUCH_KEY"}}),
    );
    let (status, answer) = call(port, "POST", "/api/v1/tables/slow/records", notes("a", 2));
    assert_ne!(status, 200, "{answer}");
    assert!(
        answer.to_string().contains("DXMT_EMBEDDING_TIMEOUT_SECS"),
        "{answer}"
    );
}
