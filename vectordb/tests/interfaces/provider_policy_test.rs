//! A table's provider settings cannot read the server's environment out to a URL.
//!
//! The attack: create a table whose embedding names any environment variable as its "key" and
//! any URL as its provider, insert one text, and the server sent that variable's value as a
//! bearer token. Here the "provider" is a listener that records every byte it receives.

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

#[path = "../common/http.rs"]
mod http;

const SECRET: &str = "s3cr3t-value-that-must-not-leave";

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
        .env("DXMT_POLICY_TEST_SECRET", SECRET)
        .env("DXMT_API_KEY", SECRET)
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

/// A "provider" that records each request and answers it with one 4-wide embedding.
fn recorder() -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let mut buf = vec![0u8; 65536];
            let n = stream.read(&mut buf).unwrap_or(0);
            log.lock()
                .unwrap()
                .push(String::from_utf8_lossy(&buf[..n]).to_string());
            let body = r#"{"data":[{"index":0,"embedding":[1,0,0,0]}]}"#;
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
        }
    });
    (url, seen)
}

fn call(port: u16, method: &str, path: &str, body: serde_json::Value) -> (u16, String) {
    http::call(
        port,
        method,
        path,
        &[("api-key", SECRET)],
        Some(&body.to_string()),
    )
    .expect("answer")
}

#[test]
fn a_table_cannot_send_the_server_s_environment_to_a_url() {
    let (url, seen) = recorder();
    let root = tempfile::tempdir().unwrap();
    let server = start(&root.path().join("db"));
    let port = server.port;

    // Any variable, and the server's own API key: both refused before anything is sent.
    for (table, env) in [
        ("arbitrary", "DXMT_POLICY_TEST_SECRET"),
        ("own_key", "DXMT_API_KEY"),
    ] {
        let (status, created) = call(
            port,
            "POST",
            "/api/v1/tables",
            serde_json::json!({"name": table, "dimension": 4, "embedding": {
                "model": "m", "base_url": url, "api_key_env": env}}),
        );
        if status == 200 {
            let (status, answer) = call(
                port,
                "POST",
                &format!("/api/v1/tables/{table}/records"),
                serde_json::json!({"records": [{"id": "a", "text": "hello"}]}),
            );
            assert_ne!(status, 200, "{env}: the insert went through: {answer}");
            assert!(answer.contains("DXMT_ALLOWED_KEY_ENVS"), "{env}: {answer}");
        } else {
            assert!(
                created.contains("DXMT_ALLOWED_KEY_ENVS"),
                "{env}: {created}"
            );
        }
    }
    // A cloud metadata address is refused even with no key to send.
    let (status, created) = call(
        port,
        "POST",
        "/api/v1/tables",
        serde_json::json!({"name": "metadata", "embedding": {
            "model": "m", "base_url": "http://169.254.169.254/v1",
            "api_key_env": "DXMT_POLICY_TEST_UNSET"}}),
    );
    assert_ne!(status, 200, "{created}");
    assert!(created.contains("metadata"), "{created}");

    std::thread::sleep(std::time::Duration::from_millis(300));
    for request in seen.lock().unwrap().iter() {
        assert!(
            !request.contains(SECRET),
            "the secret left the server:\n{request}"
        );
    }
}
