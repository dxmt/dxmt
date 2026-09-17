//! Filters have to keep working across a restart, and they did not.
//!
//! `cargo test --test filter_recovery_test`
//!
//! Two indexes are a pure function of the records and are built as records arrive: the payload
//! index that answers `filter`, and the sparse postings that answer a sparse or hybrid query.
//! Neither is written anywhere, and `restore_table` did not rebuild them — so after any restart
//! **every filter matched nothing**. `POST /search` and `records/list` both answered 200 with an
//! empty result, which is indistinguishable from a filter that genuinely selects no rows. No
//! error, no log line, and the records themselves all present and correct.
//!
//! It applied to evicted tables too, since `ensure_resident` rehydrates through the same path.
//!
//! The whole `tests/api/` suite was green through this, and would stay green: it inserts and
//! queries inside one process, so nothing ever asked a restored table to filter. That is the
//! gap this file exists to close, and it is why the test spawns the binary — the bug lives
//! entirely in what a *second* process does with a table it did not create.
//!
//! Deliberately not feature-gated: filtering is in the default build, so this runs in CI's
//! `cargo test --tests` rather than only in the optional-features job.

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
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn start(dir: &std::path::Path, port: u16) -> Server {
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DB_WAL_DIR", dir.join("wal"))
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

const EQUALS_BLUE: &str =
    r#"{"must":[{"key":"colour","operation":{"type":"Equal","value":"blue"}}]}"#;

fn seed(port: u16) {
    call(
        port,
        "POST",
        "/api/v1/tables",
        Some(r#"{"name":"t","dimension":4,"distance_metric":"cosine"}"#),
    );
    // Three blue, two red. The vectors fan out from the query direction so the dense ranking
    // is exactly r0..r4, and each record owns one sparse dimension so a sparse query can name
    // precisely one of them. Both derived indexes then have something to be wrong about.
    let records = [
        (r#"[1,0,0,0]"#, "blue"),
        (r#"[0.9,0.4,0,0]"#, "blue"),
        (r#"[0.7,0.7,0,0]"#, "blue"),
        (r#"[0.4,0.9,0,0]"#, "red"),
        (r#"[0,1,0,0]"#, "red"),
    ];
    let body: Vec<String> = records
        .iter()
        .enumerate()
        .map(|(i, (vector, colour))| {
            format!(
                r#"{{"id":"r{i}","vector":{vector},"payload":{{"colour":"{colour}","n":{i}}},
                    "sparse_vector":{{"indices":[{i}],"values":[1.0],"dimension":16}}}}"#
            )
        })
        .collect();
    call(
        port,
        "POST",
        "/api/v1/tables/t/records",
        Some(&format!(r#"{{"records":[{}]}}"#, body.join(","))),
    );
}

/// How many records `records/list` says match the filter.
fn listed(port: u16, filter: &str) -> u64 {
    let body = format!(r#"{{"limit":10,"filter":{filter}}}"#);
    let reply = call(port, "POST", "/api/v1/tables/t/records/list", Some(&body))
        .expect("no answer from records/list");
    reply["data"]["total"]
        .as_u64()
        .unwrap_or_else(|| panic!("records/list answered {reply}"))
}

/// How many hits a filtered search returns.
fn searched(port: u16, filter: &str) -> usize {
    let body = format!(r#"{{"vector":[1.0,0.1,0.1,0.1],"top_k":10,"limit":10,"filter":{filter}}}"#);
    let reply =
        call(port, "POST", "/api/v1/tables/t/search", Some(&body)).expect("no answer from search");
    reply["data"]["hits"]
        .as_array()
        .map(|hits| hits.len())
        .unwrap_or_else(|| panic!("search answered {reply}"))
}

/// The top hit of a hybrid query whose sparse half names exactly one record.
///
/// Counting hits cannot see this index: `/search/hybrid` requires a dense vector or text
/// alongside the sparse one, so the dense half returns everything either way and only the
/// *order* moves. `r4` is last on the dense ranking and owns sparse dimension 4, so it comes
/// first when the postings are there and stays last when they are not.
fn hybrid_top_hit(port: u16) -> String {
    let body = r#"{"vector":[1,0,0,0],"sparse_vector":{"indices":[4],"values":[1.0],"dimension":16},
                   "limit":5,"fusion_method":"dbsf"}"#;
    let reply = call(port, "POST", "/api/v1/tables/t/search/hybrid", Some(body))
        .expect("no answer from hybrid search");
    reply["data"]["hits"][0]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("hybrid search answered {reply}"))
        .to_string()
}

#[test]
fn a_filter_still_matches_after_a_restart() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    {
        let server = start(dir.path(), port);
        seed(server.port);
        assert_eq!(listed(server.port, EQUALS_BLUE), 3, "before the restart");
        assert_eq!(searched(server.port, EQUALS_BLUE), 3, "before the restart");
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
    }

    let server = start(dir.path(), port);

    assert_eq!(
        call(server.port, "GET", "/api/v1/tables/t", None).unwrap()["data"]["record_count"],
        5,
        "the records came back; this test is about whether the payload index did"
    );
    assert_eq!(
        listed(server.port, EQUALS_BLUE),
        3,
        "records/list returned nothing for a filter that matches three records -- the payload \
         index was empty, and an empty result is indistinguishable from no matches"
    );
    assert_eq!(
        searched(server.port, EQUALS_BLUE),
        3,
        "search returned nothing for the same filter, for the same reason"
    );
    // A filter that really does match nothing still matches nothing.
    assert_eq!(
        listed(
            server.port,
            r#"{"must":[{"key":"colour","operation":{"type":"Equal","value":"chartreuse"}}]}"#
        ),
        0
    );
}

#[test]
fn the_sparse_postings_come_back_too() {
    // The other index built from the records, lost by the same omission and restored by the
    // same rebuild.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    {
        let server = start(dir.path(), port);
        seed(server.port);
        assert_eq!(hybrid_top_hit(server.port), "r4", "before the restart");
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
    }

    let server = start(dir.path(), port);
    assert_eq!(
        hybrid_top_hit(server.port),
        "r4",
        "the sparse half of the query contributed nothing after the restart: r4 is last on the \
         dense ranking and only the postings lift it"
    );
}

#[test]
fn a_write_after_the_restart_is_filterable_alongside_the_restored_records() {
    // The case that made the bug visible in the first place: a record inserted after a restart
    // *was* filterable while everything older was not, because the insert path maintained an
    // index that no longer described the table. Both have to work, and the rebuild must not
    // lose the write that arrived before it.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    {
        let server = start(dir.path(), port);
        seed(server.port);
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
    }

    let server = start(dir.path(), port);
    // Written before anything forces the rebuild, so this record exists only in the records
    // when the indexes are built from them.
    call(
        server.port,
        "POST",
        "/api/v1/tables/t/records",
        Some(
            r#"{"records":[{"id":"late","vector":[0.2,0.2,0.2,0.9],"payload":{"colour":"blue","n":99}}]}"#,
        ),
    );

    assert_eq!(
        listed(server.port, EQUALS_BLUE),
        4,
        "three restored blues plus the one written after the restart"
    );
    assert_eq!(searched(server.port, EQUALS_BLUE), 4);
}

#[test]
fn a_deleted_record_leaves_the_filter_after_a_restart() {
    // The rebuild reads the records, so a record deleted before it happened must not come back
    // through the index -- the failure that a naive "index everything in the snapshot" rebuild
    // would have.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    {
        let server = start(dir.path(), port);
        seed(server.port);
        call(
            server.port,
            "DELETE",
            "/api/v1/tables/t/records",
            Some(r#"{"ids":["r0"]}"#),
        );
        assert_eq!(listed(server.port, EQUALS_BLUE), 2, "before the restart");
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
    }

    let server = start(dir.path(), port);
    assert_eq!(
        listed(server.port, EQUALS_BLUE),
        2,
        "the deleted record came back through the rebuilt payload index"
    );
}
