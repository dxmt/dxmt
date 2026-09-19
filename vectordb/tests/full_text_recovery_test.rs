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
//! one on every start. The Tantivy index is named after that id, so the engine opened a
//! different, empty directory each time. On the demo machine that had left **135 orphaned index
//! directories behind five tables**.
//!
//! These also cover where that index lives: `{DXMT_WAL_DIR}/fulltext/{table_id}`, a sibling of
//! `records/` and `vectors/`. It used to default to `/tmp` — outside the directory the backup
//! procedure copies, and inside the one the operating system empties.
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

/// Start a server over `dir`, waiting until it answers.
///
/// The index lands under this temporary directory because that is where `DXMT_WAL_DIR` puts it,
/// and there is no second way to say otherwise. `DXMT_STORAGE_PATH` was that second way, and it
/// defaulted to `/tmp`.
fn start(dir: &std::path::Path, port: u16) -> Server {
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DXMT_WAL_DIR", dir.join("wal"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
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

/// Where the indexes live: a sibling of `records/` and `vectors/` under the WAL directory.
fn index_root(dir: &std::path::Path) -> std::path::PathBuf {
    dir.join("wal").join("fulltext")
}

fn text_hits(port: u16, query: &str) -> usize {
    text_hits_in(port, "t", query)
}

fn text_hits_in(port: u16, table: &str, query: &str) -> usize {
    let body = format!(r#"{{"query":"{query}","limit":10}}"#);
    let reply = call(
        port,
        "POST",
        &format!("/api/v1/tables/{table}/search/text"),
        Some(&body),
    )
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
    seed_named(port, "t");
}

fn seed_named(port: u16, table: &str) {
    call(
        port,
        "POST",
        "/api/v1/tables",
        Some(&format!(
            r#"{{"name":"{table}","dimension":4,"distance_metric":"cosine","enable_full_text_search":true}}"#
        )),
    );
    call(
        port,
        "POST",
        &format!("/api/v1/tables/{table}/records"),
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

    let directories = std::fs::read_dir(index_root(dir.path()))
        .expect("the index directory should exist beside the WAL")
        .filter_map(|entry| entry.ok())
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
    // An index can go missing while the records do not: the `/tmp` default this used to have,
    // cleared by the operating system; a partial restore; or — for every table upgrading past
    // the id fix, or past the move out of `/tmp` — documents written under a path that is no
    // longer the one being opened. All of them land in this same state, and none of them used
    // to say anything at all.
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
    for entry in std::fs::read_dir(index_root(dir.path())).unwrap().flatten() {
        std::fs::remove_dir_all(entry.path()).unwrap();
        wiped += 1;
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

#[test]
fn the_index_lives_beside_the_wal_not_in_tmp() {
    // It is data — the only copy of the term postings — and it used to default to `/tmp`, which
    // `systemd-tmpfiles` clears on a schedule and many images wipe on boot. It also sat outside
    // `DXMT_WAL_DIR`, so the backup procedure in `docs/operations/backup-restore.md`, which copies
    // that directory, never included it.
    //
    // The table name is unique to this run on purpose. A table id is a function of its name, so
    // a table called `t` would derive the id every other test here derives, and the check for
    // the old location would then be answered by a directory some earlier run left in `/tmp`
    // rather than by anything this one did.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let table = format!("where_{port}");
    let server = start(dir.path(), port);
    seed_named(server.port, &table);
    assert_eq!(text_hits_in(server.port, &table, "quasar"), 1);

    let root = index_root(dir.path());
    let indexes: Vec<_> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("no index directory at {}: {e}", root.display()))
        .flatten()
        .collect();
    assert_eq!(
        indexes.len(),
        1,
        "expected one index under {}",
        root.display()
    );
    assert!(
        indexes[0].path().join("meta.json").exists(),
        "a Tantivy index should have written its meta.json here"
    );

    // And a sibling of the rest of the table's durable state, so one directory is the whole of it.
    for sibling in ["records", "vectors"] {
        assert!(
            dir.path().join("wal").join(sibling).exists(),
            "{sibling}/ should be beside fulltext/"
        );
    }

    // Nothing in the old place, checked by this table's own id.
    let id = call(server.port, "GET", &format!("/api/v1/tables/{table}"), None).unwrap()["data"]
        ["table"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    let old = std::path::PathBuf::from("/tmp").join(format!("dxmt_full_text_{id}"));
    assert!(
        !old.exists(),
        "an index was still written to the old location, {}",
        old.display()
    );
}

#[test]
fn the_index_follows_the_wal_and_nothing_else_moves_it() {
    // `DXMT_STORAGE_PATH` used to point the text index somewhere of its own, defaulting to
    // `/tmp`. It is gone: one directory holds a table's durable state, and a second way to say
    // where part of it lives is a second place for that part to be lost -- which is exactly
    // what happened while the default was `/tmp`.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DXMT_WAL_DIR", dir.path().join("wal"))
        // Set, and ignored. Nothing reads it any more.
        .env("DXMT_STORAGE_PATH", "/tmp/somewhere-else")
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");
    let server = Server { child, port };
    for _ in 0..160 {
        if call(port, "GET", "/api/v1/health", None).is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }

    seed(server.port);
    assert_eq!(text_hits(server.port, "quasar"), 1);
    assert_eq!(
        std::fs::read_dir(index_root(dir.path())).unwrap().count(),
        1,
        "the index is beside the WAL"
    );
    assert!(
        !std::path::Path::new("/tmp/somewhere-else").exists(),
        "DXMT_STORAGE_PATH moved the index; it should read as nothing at all"
    );
}
