//! A backup has to restore. Anything short of that is a directory of files.
//!
//! `POST /backup` answers with a manifest listing what it wrote and a `restore` string telling
//! the operator to replace `DB_WAL_DIR` with the directory. Nothing checked that following
//! those instructions produces a working server with the data still in it, and "the backup
//! reported success" is the kind of assurance that holds until the day it matters. This is the
//! same failure shape as `deleted_count` reporting the requested count rather than the deleted
//! one, but with the discovery deferred to a disaster.
//!
//! Restarting the server is why this is a Rust test and not part of `tests/api/`: that suite
//! attaches to a server someone else started, deliberately, so it cannot stop one.

use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

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

/// A server over `wal_dir`, so a restore can be pointed at a backup directory.
fn start_server_on(wal_dir: &Path) -> Server {
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env("DB_WAL_DIR", wal_dir)
        .env("DB_INDEX_ALGORITHM", "hnsw-pq")
        // A sweep mid-test would make a failure hard to attribute.
        .env("DB_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DB_EVICTION_INTERVAL_SECS", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");

    let server = Server { child, port };
    for _ in 0..160 {
        if curl(server.port, "GET", "/health", None).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {}", server.port);
}

fn curl(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
    let url = if path.starts_with("/health") {
        format!("http://127.0.0.1:{port}{path}")
    } else {
        format!("http://127.0.0.1:{port}/api/v1{path}")
    };
    let mut command = Command::new("curl");
    command
        .arg("-s")
        .arg("-m")
        .arg("30")
        .arg("-o")
        .arg("-")
        .arg("-w")
        .arg("\n%{http_code}")
        .arg("-X")
        .arg(method)
        .arg(&url);
    if let Some(body) = body {
        command
            .arg("-H")
            .arg("Content-Type: application/json")
            .arg("-d")
            .arg(body);
    }
    let output = command.output().expect("curl is required for this test");
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string();
    let (body, status) = text.rsplit_once('\n')?;
    Some((status.trim().parse().ok()?, body.to_string()))
}

fn ok(port: u16, method: &str, path: &str, body: Option<&str>) -> String {
    let (status, text) = curl(port, method, path, body).expect("request failed");
    assert!(
        (200..300).contains(&status),
        "{method} {path} -> {status}\n{text}"
    );
    text
}

/// Copy a directory tree. `fs_extra` is not a dependency and this is eight lines.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

const TABLE: &str = "backup_subject";
const RECORDS: &str = r#"{"records":[
    {"id":"a","vector":[1.0,0.0,0.0,0.0],"payload":{"n":1,"tag":"first"}},
    {"id":"b","vector":[0.0,1.0,0.0,0.0],"payload":{"n":2,"tag":"second"}},
    {"id":"c","vector":[0.0,0.0,1.0,0.0],"payload":{"n":3,"tag":"third"}}
]}"#;

fn seed(port: u16) {
    ok(
        port,
        "POST",
        "/tables",
        Some(&format!(
            r#"{{"name":"{TABLE}","dimension":4,"distance_metric":"cosine"}}"#
        )),
    );
    ok(
        port,
        "POST",
        &format!("/tables/{TABLE}/records"),
        Some(RECORDS),
    );
}

fn search_ids(port: u16) -> Vec<String> {
    let body = ok(
        port,
        "POST",
        &format!("/tables/{TABLE}/search"),
        Some(r#"{"vector":[1.0,0.0,0.0,0.0],"top_k":10}"#),
    );
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    parsed["data"]["hits"]
        .as_array()
        .expect("hits")
        .iter()
        .map(|h| h["id"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn a_backup_restores_into_a_working_server() {
    let original = tempfile::TempDir::new().unwrap();
    let backup = tempfile::TempDir::new().unwrap();
    let backup_dir = backup.path().join("out");

    let before_ids = {
        let server = start_server_on(&original.path().join("wal"));
        seed(server.port);
        let before = search_ids(server.port);
        assert_eq!(before.len(), 3, "the fixture did not land");

        ok(server.port, "POST", "/snapshot", Some("{}"));
        let manifest = ok(
            server.port,
            "POST",
            "/backup",
            Some(&format!(r#"{{"path":"{}"}}"#, backup_dir.to_string_lossy())),
        );
        let parsed: serde_json::Value = serde_json::from_str(&manifest).unwrap();
        assert_eq!(parsed["data"]["tables"], 1);
        assert!(
            parsed["data"]["bytes"].as_u64().unwrap() > 0,
            "a backup of zero bytes is not a backup"
        );
        // The manifest tells the operator to replace DB_WAL_DIR with this directory. The test
        // below does exactly that, so the instruction is what is under test, not a paraphrase.
        assert!(
            parsed["data"]["restore"]
                .as_str()
                .unwrap()
                .contains("DB_WAL_DIR"),
            "the manifest should say how to restore"
        );
        before
    };
    // The first server is gone here: `Drop` killed it, and a restore has to work against a
    // directory nobody is holding open.

    let restored = tempfile::TempDir::new().unwrap();
    let restored_wal = restored.path().join("wal");
    copy_tree(&backup_dir, &restored_wal);

    let server = start_server_on(&restored_wal);

    let tables = ok(server.port, "GET", "/tables", None);
    assert!(
        tables.contains(TABLE),
        "the restored server does not have the table: {tables}"
    );

    assert_eq!(
        search_ids(server.port),
        before_ids,
        "the restored server returned different records"
    );

    let record = ok(
        server.port,
        "GET",
        &format!("/tables/{TABLE}/records/b"),
        None,
    );
    assert!(
        record.contains("second"),
        "the payload did not survive the round trip: {record}"
    );
}

#[test]
fn a_snapshot_survives_a_restart_of_the_same_directory() {
    let dir = tempfile::TempDir::new().unwrap();
    let wal = dir.path().join("wal");

    let before = {
        let server = start_server_on(&wal);
        seed(server.port);
        ok(server.port, "POST", "/snapshot", Some("{}"));
        search_ids(server.port)
    };

    // Same directory, new process. This is the ordinary restart, as distinct from the restore
    // above, and it is the path a snapshot exists for.
    let server = start_server_on(&wal);
    assert_eq!(
        search_ids(server.port),
        before,
        "a restart lost or reordered records"
    );
}

#[test]
fn a_backup_of_a_path_that_cannot_be_written_fails_loudly() {
    let dir = tempfile::TempDir::new().unwrap();
    let server = start_server_on(&dir.path().join("wal"));
    seed(server.port);

    let (status, body) = curl(
        server.port,
        "POST",
        "/backup",
        Some(r#"{"path":"/proc/definitely-not-writable/backup"}"#),
    )
    .expect("request failed");
    assert!(
        !(200..300).contains(&status),
        "a backup to an unwritable path reported success: {status} {body}"
    );
}

#[test]
fn the_manifest_lists_the_files_that_are_there() {
    let dir = tempfile::TempDir::new().unwrap();
    let backup = tempfile::TempDir::new().unwrap();
    let backup_dir = backup.path().join("out");

    let server = start_server_on(&dir.path().join("wal"));
    seed(server.port);
    ok(server.port, "POST", "/snapshot", Some("{}"));

    let manifest = ok(
        server.port,
        "POST",
        "/backup",
        Some(&format!(r#"{{"path":"{}"}}"#, backup_dir.to_string_lossy())),
    );
    let parsed: serde_json::Value = serde_json::from_str(&manifest).unwrap();
    let listed: Vec<String> = parsed["data"]["files"]
        .as_array()
        .expect("files")
        .iter()
        .map(|f| f.as_str().unwrap().to_string())
        .collect();
    assert!(!listed.is_empty(), "the manifest listed no files");

    for name in &listed {
        let path = backup_dir.join(name);
        assert!(
            path.exists(),
            "the manifest lists {name}, which is not there"
        );
    }

    // Existence, not size. An empty `wal.log` is the correct outcome, not a defect: the
    // snapshot taken just above compacts the log, so the data lives in `snapshot/` and the row
    // files and the log is legitimately zero bytes. A first version of this test asserted
    // every listed file was non-empty and failed here, which would have been the wrong thing
    // to "fix" -- the restore in `a_backup_restores_into_a_working_server` proves the backup
    // is complete, and that is the assertion that matters.
    let snapshot_manifest = backup_dir.join("snapshot/manifest");
    assert!(
        snapshot_manifest.exists(),
        "a backup taken after a snapshot must carry the snapshot"
    );
    let mut contents = Vec::new();
    std::fs::File::open(&snapshot_manifest)
        .unwrap()
        .read_to_end(&mut contents)
        .unwrap();
    assert!(!contents.is_empty(), "the snapshot manifest is empty");
}
