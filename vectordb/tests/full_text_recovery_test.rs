//! Full-text search has to survive a restart, and it did not.
//!
//! `cargo test --features text-bm25 --test full_text_recovery_test`
//!
//! The failure was silent in every way that matters. After a restart the records were all
//! there, the table still reported `enable_full_text_search: true`, `POST /search/text` still
//! answered 200 — and it returned nothing, for any query, forever. Nothing in a log, nothing in
//! an error, and no way for a caller to tell it from "no documents matched".
//!
//! The cause was one line from the symptom. A table's id was `Uuid::new_v4()`, and recovery
//! rebuilds tables through `TableConfig`, which carries no id — so every table was issued a new
//! one on every start. The Tantivy index lives at `{DB_STORAGE_PATH}/dxmt_full_text_{id}`, so
//! the engine opened a different, empty directory each time. On the demo machine that had left
//! **135 orphaned index directories behind five tables**.
//!
//! ## Why this spawns the binary
//!
//! The obvious test — build a storage, drop it, build another over the same directory — cannot
//! work here, and the reason is worth writing down so nobody rewrites it that way. Tantivy's
//! `IndexWriter` holds an exclusive `.tantivy-writer.lock`, and an `IntelligentIndexedStorage`
//! keeps background tasks with their own handle, so dropping the binding does not release it.
//! The second open then fails for a reason that has nothing to do with the bug and would have
//! kept failing after it was fixed. A restart is a new process; the test has to be one too.

#![cfg(feature = "text-bm25")]

use std::process::{Child, Command, Stdio};

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
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Start a server over `dir`, waiting until it answers. `DB_STORAGE_PATH` is set so the
/// full-text index lands inside the temporary directory rather than in `/tmp`, shared with
/// every other thing on the machine — including previous runs of this test.
fn start(dir: &std::path::Path, port: u16) -> Server {
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DB_WAL_DIR", dir.join("wal"))
        .env("DB_STORAGE_PATH", dir)
        .env("DB_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DB_EVICTION_INTERVAL_SECS", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");

    let server = Server { child, port };
    for _ in 0..160 {
        if call(port, "GET", "/api/v1/health", None).is_some() {
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
        .arg("30")
        .arg("-X")
        .arg(method)
        .arg(format!("http://127.0.0.1:{port}{path}"));
    if let Some(body) = body {
        command
            .arg("-H")
            .arg("Content-Type: application/json")
            .arg("-d")
            .arg(body);
    }
    let out = command.output().expect("curl is required for this test");
    serde_json::from_slice(&out.stdout).ok()
}

fn text_hits(port: u16, query: &str) -> usize {
    let body = format!(r#"{{"query":"{query}","limit":10}}"#);
    let reply = call(port, "POST", "/api/v1/tables/t/search/text", Some(&body))
        .unwrap_or_else(|| panic!("no answer for {query}"));
    reply["data"]["hits"]
        .as_array()
        .map(|hits| hits.len())
        .unwrap_or_else(|| panic!("text search answered {reply}"))
}

fn table_id(port: u16) -> String {
    call(port, "GET", "/api/v1/tables/t", None).unwrap()["data"]["table"]["id"]
        .as_str()
        .unwrap()
        .to_string()
}

fn seed(port: u16) {
    call(
        port,
        "POST",
        "/api/v1/tables",
        Some(
            r#"{"name":"t","dimension":4,"distance_metric":"cosine","enable_full_text_search":true}"#,
        ),
    );
    call(
        port,
        "POST",
        "/api/v1/tables/t/records",
        Some(
            r#"{"records":[
                {"id":"a","vector":[1,0,0,0],"payload":{"content":"a distant quasar in the galactic core"}},
                {"id":"b","vector":[0,1,0,0],"payload":{"content":"the compost heap needs more nitrogen"}}
            ]}"#,
        ),
    );
}

#[test]
fn text_search_still_answers_after_a_restart() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    let id_before;
    {
        let server = start(dir.path(), port);
        seed(server.port);
        assert_eq!(text_hits(server.port, "quasar"), 1, "before the restart");
        id_before = table_id(server.port);
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
    }

    let server = start(dir.path(), port);

    assert_eq!(
        call(server.port, "GET", "/api/v1/tables/t", None).unwrap()["data"]["record_count"],
        2,
        "the records came back; this test is about whether the text index did"
    );
    assert_eq!(
        table_id(server.port),
        id_before,
        "the table was issued a new id by the restart, which is what used to send the full-text \
         engine to an empty directory"
    );

    assert_eq!(
        text_hits(server.port, "quasar"),
        1,
        "text search answered nothing after the restart — the index was opened at a new path and \
         found empty, which is exactly what it used to do"
    );
    // The other document, so this is a working index rather than one lucky survivor.
    assert_eq!(text_hits(server.port, "compost"), 1);
    assert_eq!(text_hits(server.port, "nothinglikethis"), 0);
}

#[test]
fn one_table_leaves_one_index_directory_however_often_it_restarts() {
    // The leak the id instability caused, and the cheapest signal that it is gone: five tables
    // on the demo machine had produced 135 of these.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    {
        let server = start(dir.path(), port);
        seed(server.port);
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
    }
    for _ in 0..2 {
        let server = start(dir.path(), port);
        assert_eq!(text_hits(server.port, "quasar"), 1);
    }

    let directories = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("dxmt_full_text_")
        })
        .count();
    assert_eq!(
        directories, 1,
        "three starts left {directories} index directories for one table"
    );
}

#[test]
fn a_recreated_table_does_not_inherit_the_dropped_one_s_documents() {
    // The cost of a stable id: drop and recreate under the same name and the id comes back, so
    // anything keyed by it has to be cleaned up on the way out. Nothing did, which was
    // survivable only while every restart minted a new id and merely leaked a directory.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let server = start(dir.path(), port);

    seed(server.port);
    assert_eq!(text_hits(server.port, "quasar"), 1);

    call(server.port, "DELETE", "/api/v1/tables/t", None);
    call(
        server.port,
        "POST",
        "/api/v1/tables",
        Some(
            r#"{"name":"t","dimension":4,"distance_metric":"cosine","enable_full_text_search":true}"#,
        ),
    );

    assert_eq!(
        text_hits(server.port, "quasar"),
        0,
        "the new table answered with the dropped table's documents"
    );
}

#[test]
fn a_lost_index_directory_is_rebuilt_from_the_records() {
    // The default location for a table's text index is under `DB_STORAGE_PATH`, which defaults
    // to `/tmp` — a directory plenty of systems clear on reboot, while the WAL and the row files
    // beside it survive. The same empty-index state is what every table upgrading past the id
    // fix lands in, since its documents were written under an id that is no longer issued.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    {
        let server = start(dir.path(), port);
        seed(server.port);
        assert_eq!(text_hits(server.port, "quasar"), 1);
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
    }

    // Wipe the index, keep everything else.
    let mut wiped = 0;
    for entry in std::fs::read_dir(dir.path()).unwrap().flatten() {
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with("dxmt_full_text_")
        {
            std::fs::remove_dir_all(entry.path()).unwrap();
            wiped += 1;
        }
    }
    assert_eq!(wiped, 1, "the test did not wipe what it meant to wipe");

    let server = start(dir.path(), port);
    assert_eq!(
        text_hits(server.port, "quasar"),
        1,
        "the index was gone and the records were not; it should have been rebuilt from them"
    );
    assert_eq!(text_hits(server.port, "compost"), 1);
    assert_eq!(text_hits(server.port, "nothinglikethis"), 0);
}
