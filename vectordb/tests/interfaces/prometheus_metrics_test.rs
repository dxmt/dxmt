//! `GET /metrics` in Prometheus text exposition format, from a running server.
//!
//! A JSON health document is not something a monitoring system can scrape, so until this
//! endpoint existed, putting DXMT under the monitoring every other database assumes meant
//! writing an exporter first. What makes it worth a test of its own is that the format is
//! parsed by machines and nothing else here parses it: a sample value of `false` instead of
//! `0` is accepted by every JSON test in this repository and rejected by every scraper, and
//! that is exactly the bug this test caught on its first run.

use std::process::{Child, Command, Stdio};

struct Server {
    child: Child,
    port: u16,
    _dir: tempfile::TempDir,
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

fn start() -> Server {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DXMT_WAL_DIR", dir.path().join("wal"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
        .env_remove("DXMT_API_KEY")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");
    let server = Server {
        child,
        port,
        _dir: dir,
    };
    for _ in 0..160 {
        if get(port, "/live").is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

fn curl(args: &[&str]) -> Option<String> {
    let out = Command::new("curl")
        .arg("-s")
        .arg("-m")
        .arg("30")
        .args(args)
        .output()
        .expect("curl is required for this test");
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

fn get(port: u16, path: &str) -> Option<String> {
    curl(&[&format!("http://127.0.0.1:{port}{path}")])
}

fn post(port: u16, path: &str, body: &str) -> Option<String> {
    curl(&[
        "-X",
        "POST",
        "-H",
        "content-type: application/json",
        "-d",
        body,
        &format!("http://127.0.0.1:{port}{path}"),
    ])
}

/// Every sample line, as (name-with-labels, value).
fn samples(body: &str) -> Vec<(String, String)> {
    body.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let (name, value) = line.rsplit_once(' ').expect("a sample is `name value`");
            (name.to_string(), value.to_string())
        })
        .collect()
}

fn value_of<'a>(samples: &'a [(String, String)], name: &str) -> &'a str {
    samples
        .iter()
        .find(|(sample, _)| sample == name)
        .map(|(_, value)| value.as_str())
        .unwrap_or_else(|| panic!("no sample named {name} in {samples:?}"))
}

#[test]
fn the_endpoint_is_parseable_prometheus_text() {
    let server = start();
    let port = server.port;
    post(
        port,
        "/api/v1/tables",
        r#"{"name": "metrics_demo", "dimension": 4}"#,
    );
    post(
        port,
        "/api/v1/tables/metrics_demo/records",
        r#"{"records": [{"id": "a", "vector": [1, 0, 0, 0]}, {"id": "b", "vector": [0, 1, 0, 0]}]}"#,
    );

    let body = get(port, "/metrics").expect("no answer from /metrics");
    let samples = samples(&body);
    assert!(!samples.is_empty(), "{body}");

    // Every sample value must parse as a number. A bool rendered as "false" passes a JSON
    // assertion and fails a scrape, which is how this went out the first time.
    for (name, value) in &samples {
        assert!(
            value.parse::<f64>().is_ok(),
            "{name} has a non-numeric value {value:?}; a scraper would reject the whole page"
        );
    }

    // Every sample belongs to a metric that declared its type.
    for (name, _) in &samples {
        let metric = name.split('{').next().unwrap();
        assert!(
            body.contains(&format!("# TYPE {metric} ")),
            "{metric} has samples but no `# TYPE` line"
        );
        assert!(
            body.contains(&format!("# HELP {metric} ")),
            "{metric} has samples but no `# HELP` line"
        );
    }

    assert_eq!(value_of(&samples, "dxmt_up"), "1");
    assert_eq!(value_of(&samples, "dxmt_durability_degraded"), "0");
    assert_eq!(value_of(&samples, "dxmt_tables"), "1");
    assert_eq!(value_of(&samples, "dxmt_records"), "2");
    assert_eq!(value_of(&samples, "dxmt_index_building"), "0");
    assert!(samples
        .iter()
        .any(|(name, _)| name == r#"dxmt_ongoing_operations{kind="insert"}"#));
}

#[test]
fn the_content_type_is_the_one_scrapers_expect() {
    let server = start();
    let headers = curl(&[
        "-D",
        "-",
        "-o",
        "/dev/null",
        &format!("http://127.0.0.1:{}/metrics", server.port),
    ])
    .expect("no answer");
    assert!(
        headers.to_lowercase().contains("text/plain"),
        "a scraper reads the content type before the body: {headers}"
    );
}

#[test]
fn the_durability_frontier_is_exported() {
    // `wal_next_sequence - snapshot_wal_next_sequence` is how much a restart replays, which is
    // the number an operator pages on. It is in the JSON health document; it has to be here too.
    let server = start();
    let port = server.port;
    post(
        port,
        "/api/v1/tables",
        r#"{"name": "wal_demo", "dimension": 2}"#,
    );
    post(
        port,
        "/api/v1/tables/wal_demo/records",
        r#"{"records": [{"id": "a", "vector": [1, 0]}]}"#,
    );

    let body = get(port, "/metrics").expect("no answer");
    let samples = samples(&body);
    let next: u64 = value_of(&samples, "dxmt_wal_next_sequence")
        .parse()
        .unwrap();
    assert!(next > 0, "{body}");
    assert!(
        body.contains("dxmt_last_wal_sync_timestamp_seconds"),
        "a write has been acknowledged, so a sync timestamp must exist: {body}"
    );
}
