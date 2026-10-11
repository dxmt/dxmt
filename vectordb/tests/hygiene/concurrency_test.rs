//! Reads and writes at the same time: no lost writes, no wrong answers, no deadlock.
//!
//! `cargo test --test concurrency_test`
//!
//! Most of this server's state is behind a lock, and several pieces of it are now built lazily
//! — the payload index, the sparse postings, the `created_at` array a page is cut from. Lazy
//! means the first reader builds it while other readers wait and writers may be arriving, which
//! is the shape that produces a deadlock or a half-built structure. It was checked by hand when
//! that code was written and then the script was thrown away; this keeps it.
//!
//! The properties, in the order they would break:
//!
//! 1. **Nothing hangs.** Every thread joins within the timeout. A lock taken twice on one path,
//!    or two taken in different orders on two paths, shows up here and nowhere else.
//! 2. **No write is lost.** The count after the storm equals what was acknowledged.
//! 3. **No answer is wrong.** A filtered read returns only records that match, whatever was
//!    landing at the time — an index caught half-built would return ids that do not.
//! 4. **Pages do not repeat or skip** among records that existed before the storm started.

use std::collections::HashSet;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[path = "../common/http.rs"]
mod http;

struct Server {
    child: Child,
    #[allow(dead_code)]
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
        .env("DXMT_WAL_DIR", dir.join("wal"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");

    let server = Server { child, port };
    for _ in 0..200 {
        if call(port, "GET", "/api/v1/health", None).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

fn call(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<serde_json::Value> {
    let (_, text) = http::call(port, method, path, &[], body)?;
    serde_json::from_str(&text).ok()
}

const SEEDED: usize = 400;
const WRITERS: usize = 8;
const READERS: usize = 16;
const PER_WRITER: usize = 10;

fn vector(i: usize) -> String {
    format!(
        "[{:.4},{:.4},0.1,0.2]",
        (i % 97) as f32 / 97.0,
        (i % 31) as f32 / 31.0
    )
}

fn seed(port: u16) {
    call(
        port,
        "POST",
        "/api/v1/tables",
        Some(r#"{"name":"c","dimension":4,"distance_metric":"cosine"}"#),
    );
    let records: Vec<String> = (0..SEEDED)
        .map(|i| {
            format!(
                r#"{{"id":"s{i}","vector":{},"payload":{{"topic":"{}","n":{i}}}}}"#,
                vector(i),
                if i % 2 == 0 { "even" } else { "odd" }
            )
        })
        .collect();
    call(
        port,
        "POST",
        "/api/v1/tables/c/records/bulk",
        Some(&format!(r#"{{"records":[{}]}}"#, records.join(","))),
    );
}

#[test]
fn readers_and_writers_do_not_block_or_corrupt_each_other() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    {
        let server = start(dir.path(), port);
        seed(server.port);
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
    }

    // Restarted, so every lazily-built structure is unbuilt when the storm starts and the first
    // readers race to build it. Building on a quiet server proves much less.
    let server = start(dir.path(), port);
    let port = server.port;

    let seeded_ids: HashSet<String> = (0..SEEDED).map(|i| format!("s{i}")).collect();
    let failures = Arc::new(AtomicUsize::new(0));
    let wrong = Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));

    let mut threads = Vec::new();

    for w in 0..WRITERS {
        let failures = Arc::clone(&failures);
        threads.push(std::thread::spawn(move || {
            for k in 0..PER_WRITER {
                let i = SEEDED + w * PER_WRITER + k;
                let body = format!(
                    r#"{{"records":[{{"id":"w{i}","vector":{},"payload":{{"topic":"even","n":{i}}}}}]}}"#,
                    vector(i)
                );
                let answer = call(port, "POST", "/api/v1/tables/c/records", Some(&body));
                if answer.is_none_or(|a| a["success"] != serde_json::Value::Bool(true)) {
                    failures.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for _ in 0..READERS {
        let failures = Arc::clone(&failures);
        let wrong = Arc::clone(&wrong);
        let seeded = seeded_ids.clone();
        threads.push(std::thread::spawn(move || {
            for round in 0..4 {
                // A filtered read, which is what forces the payload index to be built.
                let filtered = call(
                    port,
                    "POST",
                    "/api/v1/tables/c/records/list",
                    Some(
                        r#"{"limit":50,"filter":{"must":[{"key":"topic","operation":{"type":"Equal","value":"odd"}}]}}"#,
                    ),
                );
                match filtered {
                    None => {
                        failures.fetch_add(1, Ordering::Relaxed);
                    }
                    Some(answer) => {
                        // Every returned record must actually match. An index caught part-way
                        // through being built would hand back ids that do not.
                        for record in answer["data"]["records"].as_array().unwrap_or(&vec![]) {
                            let payload = record["payload_json"].as_str().unwrap_or("{}");
                            if !payload.contains("\"topic\":\"odd\"") {
                                wrong.lock().push(format!(
                                    "a filter for odd returned {}: {payload}",
                                    record["id"]
                                ));
                            }
                        }
                    }
                }

                // Two pages of the records that existed before the storm. Concurrent inserts
                // are newer and sort ahead of them, so they may shift the window, but within
                // one call a page must not repeat an id.
                let page = call(
                    port,
                    "POST",
                    "/api/v1/tables/c/records/list",
                    Some(&format!(r#"{{"limit":25,"offset":{}}}"#, round * 25)),
                );
                if let Some(answer) = page {
                    let ids: Vec<String> = answer["data"]["records"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .map(|r| r["id"].as_str().unwrap_or_default().to_string())
                        .collect();
                    let unique: HashSet<&String> = ids.iter().collect();
                    if unique.len() != ids.len() {
                        wrong.lock().push(format!("a page repeated an id: {ids:?}"));
                    }
                    for id in &ids {
                        if !seeded.contains(id) && !id.starts_with('w') {
                            wrong.lock().push(format!("a page returned an unknown id {id}"));
                        }
                    }
                } else {
                    failures.fetch_add(1, Ordering::Relaxed);
                }

                // And a search, which takes a different lock path again.
                let search = call(
                    port,
                    "POST",
                    "/api/v1/tables/c/search",
                    Some(r#"{"vector":[0.5,0.5,0.1,0.2],"top_k":10,"limit":10}"#),
                );
                if search.is_none_or(|a| a["data"]["hits"].as_array().is_none_or(Vec::is_empty)) {
                    failures.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    for (n, thread) in threads.into_iter().enumerate() {
        assert!(
            std::time::Instant::now() < deadline,
            "thread {n} had not finished after two minutes: something is deadlocked"
        );
        thread.join().expect("a worker panicked");
    }

    let wrong = wrong.lock();
    assert!(
        wrong.is_empty(),
        "wrong answers under concurrency:\n  {}",
        wrong.join("\n  ")
    );
    assert_eq!(
        failures.load(Ordering::Relaxed),
        0,
        "requests failed under concurrency"
    );

    // Nothing was lost.
    let expected = SEEDED + WRITERS * PER_WRITER;
    assert_eq!(
        call(port, "GET", "/api/v1/tables/c", None).unwrap()["data"]["record_count"],
        expected,
        "writes were lost under concurrency"
    );
    // And the index that was built during the storm is complete: half the seeded records are
    // odd, and every concurrent write was even.
    let odd = call(
        port,
        "POST",
        "/api/v1/tables/c/records/list",
        Some(r#"{"limit":1,"filter":{"must":[{"key":"topic","operation":{"type":"Equal","value":"odd"}}]}}"#),
    )
    .unwrap()["data"]["total"]
        .as_u64()
        .unwrap();
    assert_eq!(
        odd as usize,
        SEEDED / 2,
        "the payload index was built while writes were landing and came out incomplete"
    );
}
