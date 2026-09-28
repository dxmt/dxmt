//! One table to a file, and back — on this server or another one.
//!
//! `POST /backup` copies the whole instance and restoring it means stopping the server and
//! replacing its directory. Moving one table somewhere else, or keeping a copy of it before a
//! migration, had no answer short of paging it out through the API and writing the insert loop
//! by hand.
//!
//! The two properties worth holding on to are both ones an export can lose silently. A
//! disk-backed table keeps its vectors in row files and its snapshot records carry none, so an
//! export built the way a snapshot is built would write a file full of vectorless rows that
//! imports into a table nothing can find — the first version of this did exactly that, and it
//! reported `"records": 0` rather than failing. And an import that re-inserted through the API
//! would restamp `created_at` on every row, which silently reorders `records/list` for a caller
//! who never asked for anything to change.

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

fn call(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<serde_json::Value> {
    let mut command = Command::new("curl");
    command
        .arg("-s")
        .arg("-m")
        .arg("60")
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
    serde_json::from_slice(&out.stdout).ok()
}

fn seed(port: u16, table: &str) {
    let created = call(
        port,
        "POST",
        "/api/v1/tables",
        Some(&serde_json::json!({"name": table, "dimension": 3, "hnsw": {"m": 20}}).to_string()),
    )
    .expect("create answered");
    assert_eq!(created["success"], true, "create table: {created}");

    let body = serde_json::json!({"records": [
        {"id": "a", "vector": [1.0, 0.0, 0.0], "payload": {"k": "v1", "n": 1}},
        {"id": "b", "vector": [0.0, 1.0, 0.0], "payload": {"k": "v2", "n": 2}},
        {"id": "c", "vector": [0.0, 0.0, 1.0], "payload": {"k": "v3", "n": 3}},
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

fn record(port: u16, table: &str, id: &str) -> serde_json::Value {
    call(
        port,
        "GET",
        &format!("/api/v1/tables/{table}/records/{id}"),
        None,
    )
    .expect("get answered")["data"]
        .clone()
}

#[test]
fn an_exported_table_comes_back_with_its_vectors_payloads_and_timestamps() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "src");
    let before = record(port, "src", "a");

    let file = dir.path().join("exports").join("src.dxmt");
    let exported = call(
        port,
        "POST",
        "/api/v1/tables/src/export",
        Some(&serde_json::json!({"path": file.display().to_string()}).to_string()),
    )
    .expect("export answered");
    assert_eq!(exported["success"], true, "export: {exported}");
    // The number that was zero when the export read `snapshot_records`, which is empty on a
    // disk-backed table by design.
    assert_eq!(exported["data"]["records"], 3, "export: {exported}");
    assert_eq!(
        exported["data"]["records_without_vectors"], 0,
        "every record's vector should be in the file: {exported}"
    );
    assert!(
        file.is_file(),
        "the export should exist at {}",
        file.display()
    );

    let imported = call(
        port,
        "POST",
        "/api/v1/tables/import",
        Some(&serde_json::json!({"path": file.display().to_string(), "name": "copy"}).to_string()),
    )
    .expect("import answered");
    assert_eq!(imported["success"], true, "import: {imported}");
    assert_eq!(imported["data"]["records"], 3, "import: {imported}");

    let after = record(port, "copy", "a");
    assert_eq!(
        after["created_at"], before["created_at"],
        "created_at was restamped"
    );
    assert_eq!(after["version"], before["version"], "version changed");
    assert_eq!(
        after["payload_json"], before["payload_json"],
        "payload changed"
    );
    assert_eq!(
        after["vector"], before["vector"],
        "the vector did not survive"
    );

    // Searchable, which is the point: the file carries no graph and the import rebuilds one.
    let hits = call(
        port,
        "POST",
        "/api/v1/tables/copy/search",
        Some(&serde_json::json!({"vector": [1.0, 0.0, 0.0], "top_k": 1}).to_string()),
    )
    .expect("search answered");
    assert_eq!(
        hits["data"]["hits"][0]["id"], "a",
        "search on the copy: {hits}"
    );

    // Filterable too, so the payload index was built over the imported rows.
    let counted = call(
        port,
        "POST",
        "/api/v1/tables/copy/records/count",
        Some(
            &serde_json::json!({"filter": {"must": [
                {"key": "k", "operation": {"type": "Equal", "value": "v2"}}
            ]}})
            .to_string(),
        ),
    )
    .expect("count answered");
    assert_eq!(counted["data"]["count"], 1, "filter on the copy: {counted}");

    // And the table's own parameters travelled with it.
    let table = call(port, "GET", "/api/v1/tables/copy", None).expect("get table answered");
    assert_eq!(table["data"]["table"]["hnsw"]["m"], 20, "hnsw: {table}");
}

#[test]
fn an_imported_table_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("exports").join("src.dxmt");
    {
        let port = free_port();
        let _server = start(dir.path(), port);
        seed(port, "src");
        call(
            port,
            "POST",
            "/api/v1/tables/src/export",
            Some(&serde_json::json!({"path": file.display().to_string()}).to_string()),
        )
        .expect("export answered");
        call(
            port,
            "POST",
            "/api/v1/tables/import",
            Some(
                &serde_json::json!({"path": file.display().to_string(), "name": "copy"})
                    .to_string(),
            ),
        )
        .expect("import answered");
    }

    // No snapshot was taken, so what comes back is what the import wrote to the WAL. An import
    // that only touched memory would leave an empty table here.
    let port = free_port();
    let _server = start(dir.path(), port);
    let counted = call(
        port,
        "POST",
        "/api/v1/tables/copy/records/count",
        Some("{}"),
    )
    .expect("count answered");
    assert_eq!(counted["data"]["count"], 3, "after a restart: {counted}");
    let after = record(port, "copy", "a");
    assert_eq!(
        after["vector"][0], 1.0,
        "the vector after a restart: {after}"
    );
}

#[test]
fn an_import_refuses_a_name_in_use_and_replaces_it_when_told_to() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "src");

    let file = dir.path().join("exports").join("src.dxmt");
    call(
        port,
        "POST",
        "/api/v1/tables/src/export",
        Some(&serde_json::json!({"path": file.display().to_string()}).to_string()),
    )
    .expect("export answered");

    // Delete one record so the live table and the file differ, which is what makes the
    // replacement observable.
    call(
        port,
        "DELETE",
        "/api/v1/tables/src/records",
        Some(&serde_json::json!({"ids": ["c"]}).to_string()),
    )
    .expect("delete answered");

    let refused = call(
        port,
        "POST",
        "/api/v1/tables/import",
        Some(&serde_json::json!({"path": file.display().to_string()}).to_string()),
    )
    .expect("import answered");
    assert_eq!(
        refused["success"], false,
        "an import onto a live name: {refused}"
    );
    let message = refused["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("overwrite") && message.contains("name"),
        "the refusal should name both ways out: {message}"
    );

    let replaced = call(
        port,
        "POST",
        "/api/v1/tables/import",
        Some(
            &serde_json::json!({"path": file.display().to_string(), "overwrite": true}).to_string(),
        ),
    )
    .expect("import answered");
    assert_eq!(replaced["success"], true, "overwrite: {replaced}");
    assert_eq!(replaced["data"]["replaced"], true, "overwrite: {replaced}");
    let counted =
        call(port, "POST", "/api/v1/tables/src/records/count", Some("{}")).expect("count answered");
    assert_eq!(
        counted["data"]["count"], 3,
        "the deleted record is back: {counted}"
    );
}

#[test]
fn a_path_outside_the_backup_root_is_refused_in_both_directions() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _server = start(dir.path(), port);
    seed(port, "src");

    // The path comes off the wire and this server has no authentication: unbounded, an export
    // writes a copy of a table anywhere the process can write and an import reads any file it
    // can open. Both are held to DXMT_BACKUP_ROOT, the same root POST /backup obeys.
    let outside = dir.path().join("not-exports.dxmt");
    let export = call(
        port,
        "POST",
        "/api/v1/tables/src/export",
        Some(&serde_json::json!({"path": outside.display().to_string()}).to_string()),
    )
    .expect("export answered");
    assert_eq!(
        export["success"], false,
        "export outside the root: {export}"
    );
    assert!(!outside.exists(), "it should not have been written");

    let import = call(
        port,
        "POST",
        "/api/v1/tables/import",
        Some(&serde_json::json!({"path": "/etc/hostname"}).to_string()),
    )
    .expect("import answered");
    assert_eq!(
        import["success"], false,
        "import from outside the root: {import}"
    );

    // A file inside the root that is not an export is refused on its contents, not swallowed.
    let junk = dir.path().join("exports").join("junk.dxmt");
    std::fs::write(&junk, b"not a dxmt export").unwrap();
    let garbage = call(
        port,
        "POST",
        "/api/v1/tables/import",
        Some(&serde_json::json!({"path": junk.display().to_string()}).to_string()),
    )
    .expect("import answered");
    assert_eq!(garbage["success"], false, "import of junk: {garbage}");
}
