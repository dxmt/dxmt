//! Another name for a table, and the swap it exists for.
//!
//! Rebuilding a table with different index parameters means building a second one and moving
//! traffic across. Without an alias that move is a client-side deploy: every caller has to
//! learn the new name at the same moment. With one it is a single request, and a reader
//! arriving mid-swap sees one table or the other rather than neither.
//!
//! The part that needs a test more than the swap does is the *coverage*. An alias resolved on
//! `search` and not on `records/count` is worse than no alias at all — it works until the one
//! call that does not, and the error names a table the caller never asked about. So the first
//! test here walks every table-scoped route the SDKs use, with an alias in the path, and fails
//! on any that cannot find it.

use std::process::{Child, Command, Stdio};

struct Server {
    child: Child,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

fn start(dir: &std::path::Path, port: u16) -> Server {
    std::fs::create_dir_all(dir.join("exports")).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DXMT_WAL_DIR", dir.join("wal"))
        .env("DXMT_BACKUP_ROOT", dir.join("exports"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
        .env_remove("DXMT_API_KEY")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");
    let server = Server { child };
    for _ in 0..160 {
        if call(port, "GET", "/live", None).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

/// The response body and its status, since a route that cannot find the table answers 404 with
/// a body worth reading.
fn request(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut command = Command::new("curl");
    command
        .arg("-s")
        .arg("-m")
        .arg("60")
        .arg("-w")
        .arg("\n%{http_code}")
        .arg("-X")
        .arg(method)
        .arg(format!("http://127.0.0.1:{port}{path}"));
    if body.is_some() {
        command
            .arg("-H")
            .arg("content-type: application/json")
            .arg("--data-binary")
            .arg("@-")
            .stdin(Stdio::piped());
    }
    command.stdout(Stdio::piped());
    let mut child = command.spawn().expect("curl is required for this test");
    if let Some(body) = body {
        use std::io::Write as _;
        child
            .stdin
            .take()
            .expect("stdin was piped")
            .write_all(body.as_bytes())
            .expect("curl took the body");
    }
    let out = child.wait_with_output().expect("curl ran");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let (body, status) = text.rsplit_once('\n').unwrap_or((text.as_str(), "0"));
    (status.parse().unwrap_or(0), body.to_string())
}

fn call(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<serde_json::Value> {
    let (_, text) = request(port, method, path, body);
    serde_json::from_str(&text).ok()
}

fn seed(port: u16, table: &str) {
    let created = call(
        port,
        "POST",
        "/api/v1/tables",
        Some(&serde_json::json!({"name": table, "dimension": 2}).to_string()),
    )
    .expect("create answered");
    assert_eq!(created["success"], true, "create table: {created}");
    let body = serde_json::json!({"records": [
        {"id": "a", "vector": [1.0, 0.0], "payload": {"v": 1}},
        {"id": "b", "vector": [0.0, 1.0], "payload": {"v": 2}},
    ]})
    .to_string();
    let inserted = call(
        port,
        "POST",
        &format!("/api/v1/tables/{table}/records"),
        Some(&body),
    )
    .expect("insert answered");
    assert_eq!(inserted["success"], true, "insert: {inserted}");
}

fn point(port: u16, alias: &str, table: &str) -> serde_json::Value {
    call(
        port,
        "PUT",
        &format!("/api/v1/aliases/{alias}"),
        Some(&serde_json::json!({"table": table}).to_string()),
    )
    .expect("alias answered")
}

/// Every table-scoped route a client reaches for, and a body each will accept.
///
/// Kept beside `tests/api_contract_test.rs`'s list on purpose: that one asserts the routes
/// exist, this one asserts they resolve a name. A route added there and missed here is an alias
/// that works everywhere except one place.
const ROUTES: &[(&str, &str, &str)] = &[
    ("GET", "/api/v1/tables/{t}", ""),
    ("GET", "/api/v1/tables/{t}/stats", ""),
    (
        "POST",
        "/api/v1/tables/{t}/search",
        r#"{"vector":[1,0],"top_k":1}"#,
    ),
    (
        "POST",
        "/api/v1/tables/{t}/search/groups",
        r#"{"vector":[1,0],"group_by":"v","groups":1,"group_size":1}"#,
    ),
    (
        "POST",
        "/api/v1/tables/{t}/search/sparse",
        r#"{"sparse_vector":{"indices":[1],"values":[1.0],"dimension":8},"limit":1}"#,
    ),
    ("POST", "/api/v1/tables/{t}/records/list", r#"{"limit":1}"#),
    ("POST", "/api/v1/tables/{t}/records/count", "{}"),
    ("GET", "/api/v1/tables/{t}/records/a", ""),
    ("GET", "/api/v1/tables/{t}/payload_index", ""),
    (
        "POST",
        "/api/v1/tables/{t}/records",
        r#"{"records":[{"id":"c","vector":[0.5,0.5]}]}"#,
    ),
    (
        "PUT",
        "/api/v1/tables/{t}/records/a",
        r#"{"vector":[1,0],"payload":{"v":1}}"#,
    ),
    ("POST", "/api/v1/tables/{t}/index/rebuild", ""),
    (
        "PATCH",
        "/api/v1/tables/{t}",
        r#"{"description":"through an alias"}"#,
    ),
    ("DELETE", "/api/v1/tables/{t}/records", r#"{"ids":["c"]}"#),
];

#[test]
fn every_table_route_resolves_an_alias() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "real");
    assert_eq!(point(port, "nick", "real")["success"], true);

    let mut broken = Vec::new();
    for (method, template, body) in ROUTES {
        let path = template.replace("{t}", "nick");
        let (status, text) = request(port, method, &path, (!body.is_empty()).then_some(*body));
        // 404 is the failure this is looking for: the route exists (the contract test says so),
        // so a 404 here means it looked the name up without resolving it.
        if status == 404 || text.contains("TABLE_NOT_FOUND") {
            broken.push(format!(
                "{method} {template} -> {status} {}",
                &text[..text.len().min(120)]
            ));
        }
    }
    assert!(
        broken.is_empty(),
        "these routes do not resolve an alias, so the alias works everywhere except here:\n  {}",
        broken.join("\n  ")
    );
}

#[test]
fn moving_an_alias_moves_the_reads() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "v1");
    seed(port, "v2");
    // Make the two tables tell each other apart by size.
    call(
        port,
        "POST",
        "/api/v1/tables/v2/records",
        Some(&serde_json::json!({"records": [{"id": "c", "vector": [0.5, 0.5]}]}).to_string()),
    )
    .expect("insert answered");

    let count = |table: &str| -> u64 {
        call(
            port,
            "POST",
            &format!("/api/v1/tables/{table}/records/count"),
            Some("{}"),
        )
        .expect("count answered")["data"]["count"]
            .as_u64()
            .unwrap_or(0)
    };

    assert_eq!(point(port, "live", "v1")["success"], true);
    assert_eq!(count("live"), 2, "the alias reads v1");
    assert_eq!(point(port, "live", "v2")["success"], true);
    assert_eq!(count("live"), 3, "the same alias now reads v2");
    // And both tables are still there: a swap moves a name, it does not move data.
    assert_eq!(count("v1"), 2);
    assert_eq!(count("v2"), 3);
}

#[test]
fn an_alias_and_a_table_cannot_share_a_name() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "real");
    assert_eq!(point(port, "nick", "real")["success"], true);

    // Which of the two a name meant would otherwise depend on the order of two maps.
    let shadowing = point(port, "real", "real");
    assert_eq!(
        shadowing["success"], false,
        "alias over a table: {shadowing}"
    );

    let colliding = call(
        port,
        "POST",
        "/api/v1/tables",
        Some(&serde_json::json!({"name": "nick", "dimension": 2}).to_string()),
    )
    .expect("create answered");
    assert_eq!(
        colliding["success"], false,
        "table over an alias: {colliding}"
    );

    // Dropping the table drops the alias with it, or the alias would resolve to a name nothing
    // holds and the error would talk about a table the caller never mentioned.
    call(port, "DELETE", "/api/v1/tables/real", None).expect("delete answered");
    let aliases = call(port, "GET", "/api/v1/aliases", None).expect("list answered");
    assert_eq!(aliases["data"]["count"], 0, "left behind: {aliases}");
}

#[test]
fn aliases_survive_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let port = free_port();
        let _server = start(dir.path(), port);
        seed(port, "real");
        assert_eq!(point(port, "nick", "real")["success"], true);
    }

    // They are in their own file, not the WAL, so this is the only thing that checks the file
    // is written and read at all.
    let port = free_port();
    let _server = start(dir.path(), port);
    let counted = call(
        port,
        "POST",
        "/api/v1/tables/nick/records/count",
        Some("{}"),
    )
    .expect("count answered");
    assert_eq!(counted["data"]["count"], 2, "through the alias: {counted}");
    let aliases = call(port, "GET", "/api/v1/aliases", None).expect("list answered");
    assert_eq!(aliases["data"]["aliases"]["nick"], "real", "{aliases}");
}
