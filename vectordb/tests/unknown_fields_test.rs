//! A field the server does not have is an error, on every endpoint that takes a body.
//!
//! serde ignores unknown fields by default, and that default had cost this API real behaviour
//! three times over: `mode` on an insert (ten names, one code path, all of them removed --
//! every SDK went on sending it), `top_k` and `alpha` on a hybrid search (in this crate's own
//! Swagger example, tuning nothing), `named_vectors` on a table that then built no named index
//! and searched the default vector instead. None of them failed. They were accepted, dropped,
//! and answered wrongly.
//!
//! So this walks the endpoints that take a body and sends each one a body it should refuse,
//! because the value of the change is that it holds for *all* of them: one struct missing the
//! attribute is one endpoint back to silently ignoring what it is sent. Each case also checks
//! the good body still works, so "refuses everything" cannot pass as a result.
//!
//! `simple_core.rs` was a third file here, for the zero-config `/collections` API. That was a
//! second data model over the same storage and it is gone; the server has one.

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
        if call(port, "GET", "/live", None).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

fn call(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
    let mut command = Command::new("curl");
    command
        .arg("-s")
        .arg("-m")
        .arg("30")
        .arg("-o")
        .arg("/dev/stdout")
        .arg("-w")
        .arg("\n%{http_code}")
        .arg("-X")
        .arg(method)
        .arg(format!("http://127.0.0.1:{port}{path}"));
    if let Some(body) = body {
        command
            .arg("-H")
            .arg("content-type: application/json")
            .arg("-d")
            .arg(body);
    }
    let out = command.output().expect("curl is required for this test");
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let (body, code) = text.rsplit_once('\n')?;
    Some((code.trim().parse().ok()?, body.to_string()))
}

fn post(port: u16, path: &str, body: serde_json::Value) -> (u16, serde_json::Value) {
    send(port, "POST", path, body)
}

fn send(port: u16, method: &str, path: &str, body: serde_json::Value) -> (u16, serde_json::Value) {
    let (code, text) = call(port, method, path, Some(&body.to_string())).expect("no answer");
    let json = serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
    (code, json)
}

/// The good body, and the same body with one field the server does not have.
fn with_typo(mut body: serde_json::Value, field: &str) -> serde_json::Value {
    body.as_object_mut()
        .unwrap()
        .insert(field.to_string(), serde_json::json!("x"));
    body
}

#[test]
fn every_body_endpoint_refuses_a_field_it_does_not_have() {
    let server = start();
    let port = server.port;
    let table = "unknown_fields";

    // Each case: the path, a body the server accepts, and the field to smuggle in.
    let (code, _) = post(
        port,
        "/api/v1/tables",
        serde_json::json!({"name": table, "dimension": 3}),
    );
    assert_eq!(code, 200, "the good create must work");

    let cases: Vec<(&str, String, serde_json::Value, &str)> = vec![
        (
            "POST",
            "/api/v1/tables".to_string(),
            serde_json::json!({"name": "unknown_fields_2", "dimension": 3}),
            // The field `TableConfig` has and this request did not: it was accepted and
            // dropped, and the table came up with no named vector spaces.
            "dimensions",
        ),
        (
            "POST",
            format!("/api/v1/tables/{table}/records"),
            serde_json::json!({"records": [{"id": "r1", "vector": [1.0, 0.0, 0.0]}]}),
            // What every SDK used to send on every insert.
            "mode",
        ),
        (
            "POST",
            format!("/api/v1/tables/{table}/records/bulk"),
            serde_json::json!({"records": [{"id": "r2", "vector": [1.0, 0.0, 0.0]}]}),
            "mode",
        ),
        (
            "POST",
            format!("/api/v1/tables/{table}/search"),
            serde_json::json!({"vector": [1.0, 0.0, 0.0], "top_k": 1}),
            "ef_search",
        ),
        (
            "POST",
            format!("/api/v1/tables/{table}/records/list"),
            serde_json::json!({"limit": 1}),
            "with_payload",
        ),
        (
            "POST",
            format!("/api/v1/tables/{table}/search/hybrid"),
            serde_json::json!({"vector": [1.0, 0.0, 0.0], "limit": 1}),
            // Both of these were in the Swagger example for this endpoint.
            "alpha",
        ),
        (
            "POST",
            format!("/api/v1/tables/{table}/graph"),
            serde_json::json!({"sample": 2, "neighbours": 1}),
            "neighbors",
        ),
        (
            "DELETE",
            format!("/api/v1/tables/{table}/records"),
            serde_json::json!({"ids": ["nobody"]}),
            "id",
        ),
    ];

    for (method, path, good, field) in cases {
        let (code, body) = send(port, method, &path, good.clone());
        assert!(
            (200..300).contains(&code),
            "{path}: the good body was refused with {code}: {body}"
        );

        let (code, body) = send(port, method, &path, with_typo(good, field));
        assert_eq!(code, 400, "{path}: `{field}` was accepted: {body}");
        assert_eq!(body["success"], false, "{path}: {body}");
        assert_eq!(body["error"]["code"], "INVALID_INPUT", "{path}: {body}");
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains(field),
            "{path}: the message does not name `{field}`: {message}"
        );
    }
}

#[test]
fn an_unknown_query_parameter_is_refused() {
    // `?limitt=1` was accepted and dropped, so the caller got the default page of 50 while
    // believing they had asked for one row. A query string is the easiest place in an HTTP API
    // to make a typo and the least likely place to notice one.
    let server = start();
    let port = server.port;

    let (code, body) = call(port, "GET", "/api/v1/tables?limit=1", None).expect("no answer");
    assert_eq!(code, 200, "the good query must work: {body}");

    let (code, text) = call(port, "GET", "/api/v1/tables?limitt=1", None).expect("no answer");
    assert_eq!(code, 400, "{text}");
    let body: serde_json::Value = serde_json::from_str(&text).expect("the body must be JSON");
    assert_eq!(body["error"]["code"], "INVALID_INPUT");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("limitt"),
        "{body}"
    );
}

#[test]
fn a_filter_condition_refuses_a_field_it_does_not_have() {
    // The nested types matter as much as the top level: `{"key": ..., "op": ...}` for
    // `operation` is exactly the typo a person makes writing a filter by hand.
    let server = start();
    let port = server.port;
    let table = "unknown_fields_filter";
    post(
        port,
        "/api/v1/tables",
        serde_json::json!({"name": table, "dimension": 3}),
    );

    let good = serde_json::json!({
        "limit": 1,
        "filter": {"must": [{"key": "lang", "operation": {"type": "Equal", "value": "en"}}]}
    });
    let (code, body) = post(port, &format!("/api/v1/tables/{table}/records/list"), good);
    assert_eq!(code, 200, "{body}");

    let typo = serde_json::json!({
        "limit": 1,
        "filter": {"must": [{"key": "lang", "operation": {"type": "Equal", "value": "en"}, "boost": 2}]}
    });
    let (code, body) = post(port, &format!("/api/v1/tables/{table}/records/list"), typo);
    assert_eq!(code, 400, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("boost"),
        "{body}"
    );

    let misspelt_list = serde_json::json!({
        "limit": 1,
        "filter": {"musts": [{"key": "lang", "operation": {"type": "Equal", "value": "en"}}]}
    });
    let (code, body) = post(
        port,
        &format!("/api/v1/tables/{table}/records/list"),
        misspelt_list,
    );
    assert_eq!(
        code, 400,
        "a misspelt `must` used to filter nothing: {body}"
    );
}

#[test]
fn a_malformed_body_is_an_error_object_not_plain_text() {
    // axum's own rejection is a 422 with a `text/plain` body, so a client reading `error.code`
    // off every response gets a parse failure for the most ordinary mistake there is.
    let server = start();
    let (code, text) = call(
        server.port,
        "POST",
        "/api/v1/tables",
        Some("{\"name\": \"x\", "),
    )
    .expect("no answer");
    assert_eq!(code, 400);
    let body: serde_json::Value = serde_json::from_str(&text).expect("the body must be JSON");
    assert_eq!(body["error"]["code"], "INVALID_INPUT");
}

/// Every JSON body in the tree goes through `ApiJson`, and its type refuses unknown fields.
///
/// The runtime tests above cover the endpoints that exist today. This one covers the next
/// endpoint somebody adds: `Json<T>` still compiles, and a request type without the attribute
/// still deserialises, so the failure mode of this whole change is silent and arrives later.
/// Reading the source is the only way to catch that before it ships.
#[test]
fn no_handler_takes_a_typed_body_through_plain_json() {
    let mut offenders = Vec::new();
    let mut wanted = Vec::new();
    // Every file under `src/api/`, not a named list. The list was `["handlers.rs",
    // "multimodal.rs"]`, which is the same shape of mistake this test exists to catch: a guard
    // that names a subset stops guarding the moment the subset stops being the whole set. It
    // also broke outright when `multimodal.rs` moved to `planning/` — a source-scanning test
    // should not carry a path that a `git mv` can invalidate.
    for path in walk(&format!("{}/src/api", env!("CARGO_MANIFEST_DIR"))) {
        let file = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let source = std::fs::read_to_string(&path).expect("api source");
        for line in source.lines() {
            let line = line.trim();
            if let Some(rest) = line
                .strip_prefix("Json(")
                .and_then(|l| l.split_once("): Json<"))
            {
                let kind = rest.1.trim_end_matches(">,").trim_end_matches('>');
                // A `serde_json::Value` body has no fields to be unknown; it is read by hand.
                if !kind.contains("serde_json::Value") {
                    offenders.push(format!("{file}: {line}"));
                }
            }
            if let Some(rest) = line
                .strip_prefix("ApiJson(")
                .and_then(|l| l.split_once("): ApiJson<"))
            {
                let kind = rest.1.trim_end_matches(">,").trim_end_matches('>');
                let name = kind.rsplit("::").next().unwrap_or(kind);
                let name = name
                    .trim_start_matches("Vec<")
                    .trim_end_matches('>')
                    .to_string();
                wanted.push(name);
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "these take a typed body through axum's `Json`, whose rejection is a 422 in plain text \
         and which ignores unknown fields; use `ApiJson`:\n  {}",
        offenders.join("\n  ")
    );

    // And each of those types refuses unknown fields.
    let mut sources = String::new();
    for entry in walk(&format!("{}/src", env!("CARGO_MANIFEST_DIR"))) {
        sources.push_str(&std::fs::read_to_string(entry).unwrap_or_default());
    }
    let mut missing = Vec::new();
    for name in wanted {
        // Look for the definition and the attribute above it.
        // `FullTextSearchRequest` names two types: the body this API takes, and the engine's
        // own request inside `search::full_text`. It is enough that one definition carries the
        // attribute; a type with none carrying it is the case worth failing on.
        let mut found = false;
        let mut denied = false;
        for (at, _) in sources.match_indices(&format!("pub struct {name} ")) {
            found = true;
            // The sources have Chinese doc comments, so slicing by byte offset has to land on
            // a character boundary.
            let mut start = at.saturating_sub(400);
            while start < at && !sources.is_char_boundary(start) {
                start += 1;
            }
            if sources[start..at].contains("deny_unknown_fields") {
                denied = true;
            }
        }
        if found && !denied {
            missing.push(name);
        }
    }
    assert!(
        missing.is_empty(),
        "request types without `#[serde(deny_unknown_fields)]`, so a misspelled field in a body \
         is accepted and dropped: {missing:?}"
    );
}

fn walk(dir: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path.to_string_lossy()));
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out
}
