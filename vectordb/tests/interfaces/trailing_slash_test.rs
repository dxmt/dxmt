//! A trailing slash, or a doubled one, reaches the same route as the clean path.
//!
//! The router matches exactly, so `/api/v1/tables/docs/` used to answer an empty 404 beside a
//! working `/api/v1/tables/docs`. `api::path` normalises the path before routing; this checks
//! it on the real binary, where that wrapping happens, and that the console, which *needs* its
//! slash -- `/dashboard` redirects to `/dashboard/` -- is left as it was.

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

fn start(dir: &std::path::Path) -> (Server, u16) {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .args(["serve", "--port", &port.to_string()])
        .env("DXMT_WAL_DIR", dir.join("wal"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
        .env_remove("DXMT_API_KEY")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn vectordb");
    let server = Server(child);
    for _ in 0..160 {
        if http::call(port, "GET", "/live", &[], None).is_some() {
            return (server, port);
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

fn call(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    http::call(
        port,
        method,
        path,
        &[("content-type", "application/json")],
        body,
    )
    .expect("no answer")
}

#[test]
fn a_trailing_slash_reaches_the_same_route() {
    let dir = tempfile::tempdir().unwrap();
    let (_server, port) = start(dir.path());

    let (status, text) = call(
        port,
        "POST",
        "/api/v1/tables/",
        Some(r#"{"name":"docs","dimension":2}"#),
    );
    assert_eq!(status, 200, "create through a trailing slash: {text}");

    for path in [
        "/api/v1/tables/docs/",
        "/api/v1/tables/docs//",
        "/api//v1/tables/docs",
        "/live/",
        "/api/docs/",
    ] {
        let (status, text) = call(port, "GET", path, None);
        assert_eq!(status, 200, "{path}: {text}");
    }

    let (status, text) = call(
        port,
        "POST",
        "/api/v1/tables/docs/records/",
        Some(r#"{"records":[{"id":"a","vector":[1,0]}]}"#),
    );
    assert_eq!(status, 200, "insert: {text}");
    let (status, text) = call(
        port,
        "POST",
        "/api/v1/tables/docs/search/",
        Some(r#"{"vector":[1,0],"limit":1}"#),
    );
    assert_eq!(status, 200, "search: {text}");
    assert!(
        text.contains(r#""id":"a""#),
        "the body survived the rewrite: {text}"
    );

    // The query string is kept: `limit=1` still limits.
    let (status, text) = call(port, "GET", "/api/v1/tables/?limit=1", None);
    assert_eq!(status, 200, "{text}");
    assert!(text.contains(r#""limit":1"#), "{text}");
}

#[test]
fn the_dashboard_keeps_its_slash() {
    let dir = tempfile::tempdir().unwrap();
    let (_server, port) = start(dir.path());
    // `/dashboard` still redirects to `/dashboard/`, and `/dashboard/` is not trimmed back
    // into the redirect. Whether the console is compiled in decides 200 or 404 there; what
    // must not happen is a 308 pointing at itself.
    let reply = http::request(
        port,
        "GET",
        "/dashboard",
        &[],
        None,
        Duration::from_secs(10),
    )
    .expect("no answer");
    assert_eq!(reply.status, 308);
    let reply = http::request(
        port,
        "GET",
        "/dashboard/",
        &[],
        None,
        Duration::from_secs(10),
    )
    .expect("no answer");
    assert_ne!(reply.status, 308, "/dashboard/ redirected again");
}
