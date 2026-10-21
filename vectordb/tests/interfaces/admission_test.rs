//! Query admission over HTTP: with one slot and no queue, concurrent searches past the first
//! are answered 429 at once rather than all slowing down together, and `/metrics` counts them.
//!
//! Each search embeds its text through a provider that takes 500 ms, which is what holds the
//! slot long enough for the others to arrive while it is taken.

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
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_MAX_CONCURRENT_QUERIES", "1")
        .env("DXMT_QUERY_QUEUE", "0")
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

#[test]
fn searches_past_the_limit_are_turned_away_with_a_429() {
    let (url, _provider) = fake_provider::serve_on_a_thread(Behaviour::SlowPerInput(500), 16);
    let root = tempfile::tempdir().unwrap();
    let server = start(&root.path().join("db"));
    let port = server.port;
    let body = serde_json::json!({"name": "t", "dimension": 16, "embedding": {
        "model": "slow", "base_url": url, "api_key_env": "DXMT_TEST_NO_SUCH_KEY"}})
    .to_string();
    let (status, _) = http::call(port, "POST", "/api/v1/tables", &[], Some(&body)).unwrap();
    assert_eq!(status, 200);
    // Writes are not admitted here: an insert goes through while queries are limited.
    let records = r#"{"records":[{"id":"a","text":"hello"}]}"#;
    let (status, _) =
        http::call(port, "POST", "/api/v1/tables/t/records", &[], Some(records)).unwrap();
    assert_eq!(status, 200);

    let answers: Vec<(u16, String)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..3)
            .map(|_| {
                scope.spawn(move || {
                    http::call(
                        port,
                        "POST",
                        "/api/v1/tables/t/search",
                        &[],
                        Some(r#"{"text":"hello","top_k":1}"#),
                    )
                    .unwrap()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let ok = answers.iter().filter(|(s, _)| *s == 200).count();
    let refused: Vec<&String> = answers
        .iter()
        .filter(|(s, _)| *s == 429)
        .map(|(_, b)| b)
        .collect();
    assert!(ok >= 1, "{answers:?}");
    assert!(!refused.is_empty(), "no search was refused: {answers:?}");
    assert!(refused[0].contains("TOO_MANY_REQUESTS"), "{}", refused[0]);

    let (_, metrics) = http::call(port, "GET", "/metrics", &[], None).unwrap();
    let rejected: u64 = metrics
        .lines()
        .find_map(|l| l.strip_prefix("dxmt_queries_rejected_total "))
        .expect("the counter is exported")
        .trim()
        .parse()
        .unwrap();
    assert_eq!(rejected as usize, refused.len(), "{metrics}");
}
