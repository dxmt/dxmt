//! The API key as an HTTP surface, in a real process.
//!
//! The unit tests in `src/api/auth.rs` prove the comparison is right. They cannot prove the
//! middleware is *reached* — a layer merged in the wrong order, or applied after the routes it
//! was meant to cover, compiles and passes every unit test while serving the data to anyone.
//! Only a request to a running server answers that, so this spawns the binary.
//!
//! The two failures worth guarding are opposite in shape. One is a server that looks protected
//! and is not: the key is set, and something still answers. The other is a server nobody can
//! operate: the key is set, and a Kubernetes liveness probe — which cannot carry a header —
//! gets 401, so the orchestrator restarts a perfectly healthy process forever.

use std::process::{Child, Command, Stdio};

#[path = "../common/http.rs"]
mod http;

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

/// Start a server, optionally with a key.
fn start(key: Option<&str>) -> Server {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let mut command = Command::new(env!("CARGO_BIN_EXE_vectordb"));
    command
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DXMT_WAL_DIR", dir.path().join("wal"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    match key {
        Some(key) => command.env("DXMT_API_KEY", key),
        // Removed, not left alone: a developer with one exported would otherwise be running a
        // different test from everyone else, and it would be the one that passes.
        None => command.env_remove("DXMT_API_KEY"),
    };

    let child = command
        .spawn()
        .expect("failed to spawn the vectordb binary");
    let server = Server {
        child,
        port,
        _dir: dir,
    };
    for _ in 0..160 {
        // `/live` needs no key by design, which makes it the one probe usable for readiness
        // here whether or not this server has one.
        if status(port, "GET", "/live", &[]).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

/// The status code, or `None` if nothing answered.
fn status(port: u16, method: &str, path: &str, headers: &[(&str, &str)]) -> Option<u16> {
    call(port, method, path, headers).map(|(code, _)| code)
}

fn call(port: u16, method: &str, path: &str, headers: &[(&str, &str)]) -> Option<(u16, String)> {
    http::call(port, method, path, headers, None)
}

const KEY: &str = "a-test-key-1234";

/// Unset means open. The default has to stay what every other vector database's is, or the
/// first thing a new user meets is a 401 they were never told to expect.
#[test]
fn without_a_key_everything_answers() {
    let server = start(None);
    for path in ["/status", "/health/metrics", "/api/v1/tables"] {
        assert_eq!(
            status(server.port, "GET", path, &[]),
            Some(200),
            "{path} should be open when no key is configured"
        );
    }
}

/// The failure that matters: a server that looks protected and is not.
#[test]
fn with_a_key_the_data_endpoints_refuse() {
    let server = start(Some(KEY));
    for path in ["/api/v1/tables", "/status", "/health/metrics"] {
        let (code, body) = call(server.port, "GET", path, &[]).expect("an answer");
        assert_eq!(code, 401, "{path} answered {code}: {body}");
        // The same envelope as every other error, so a client parsing `error.code` needs no
        // second shape for this one.
        let json: serde_json::Value = serde_json::from_str(&body)
            .unwrap_or_else(|e| panic!("{path} did not answer the standard envelope: {e}: {body}"));
        assert_eq!(json["success"], false, "{body}");
        assert_eq!(json["error"]["code"], "UNAUTHORIZED", "{body}");
        assert!(
            !body.contains(KEY),
            "the rejection quoted the key back: {body}"
        );
    }

    // And a write is refused too. A read-only leak would be bad; a writable one is worse, and
    // POST takes a different path through the router than GET.
    let (code, _) = call(server.port, "POST", "/api/v1/tables", &[]).expect("an answer");
    assert_eq!(code, 401, "POST was not covered by the middleware");
}

#[test]
fn the_right_key_gets_through_by_either_header() {
    let server = start(Some(KEY));
    for header in [
        ("api-key", KEY),
        ("Authorization", "Bearer a-test-key-1234"),
    ] {
        assert_eq!(
            status(server.port, "GET", "/api/v1/tables", &[header]),
            Some(200),
            "{} was not accepted",
            header.0
        );
    }
}

/// Near misses, because "close enough" is the whole failure mode of a hand-written compare.
#[test]
fn a_wrong_key_is_refused_however_close() {
    let server = start(Some(KEY));
    for wrong in [
        "a-test-key-1235",      // one character
        "a-test-key-123",       // a prefix
        "a-test-key-1234-more", // an extension
        "A-TEST-KEY-1234",      // case
        "",
    ] {
        assert_eq!(
            status(server.port, "GET", "/api/v1/tables", &[("api-key", wrong)]),
            Some(401),
            "{wrong:?} was accepted as the key"
        );
    }

    // A bearer token that is not the key, and a scheme that is not Bearer.
    for header in [("Authorization", "Bearer wrong"), ("Authorization", KEY)] {
        assert_eq!(
            status(server.port, "GET", "/api/v1/tables", &[header]),
            Some(401),
            "{header:?} was accepted"
        );
    }
}

/// Surrounding whitespace never reaches us, and that is HTTP, not us.
///
/// RFC 7230 §3.2.4 has the parser strip leading and trailing OWS from a field value, so
/// `api-key: <space>k` and `api-key: k` are the same request by the time any handler runs —
/// measured, not assumed: all four spacings below answer 200. Asserting a 401 for them, which
/// this test first did, is asserting something the protocol makes impossible.
///
/// It also settles a design question. A key is read from the environment with `trim`, and a
/// key configured *with* meaningful trailing whitespace could therefore never be sent at all —
/// so trimming the configured side is not a convenience, it is what keeps the two ends able to
/// agree.
#[test]
fn http_strips_the_whitespace_before_we_ever_see_it() {
    let server = start(Some(KEY));
    for spacing in [
        "a-test-key-1234",
        " a-test-key-1234",
        "a-test-key-1234 ",
        "  a-test-key-1234  ",
    ] {
        assert_eq!(
            status(
                server.port,
                "GET",
                "/api/v1/tables",
                &[("api-key", spacing)]
            ),
            Some(200),
            "{spacing:?} should be the same request as the key itself"
        );
    }
}

/// The other failure: a server nobody can operate.
///
/// A liveness probe cannot carry a header. If it gets 401 the orchestrator restarts a healthy
/// process, forever, and the cause looks nothing like "we turned on authentication".
#[test]
fn the_probes_stay_open_so_an_orchestrator_can_still_work() {
    let server = start(Some(KEY));
    for path in [
        "/health",
        "/live",
        "/ready",
        "/health/live",
        "/health/ready",
        "/api/v1/health",
    ] {
        assert_eq!(
            status(server.port, "GET", path, &[]),
            Some(200),
            "{path} needs no key: a probe cannot send one"
        );
    }
}

/// `/health` is open and `/health/metrics` is not, so the exemption must be an exact match.
#[test]
fn the_exemption_does_not_extend_by_prefix() {
    let server = start(Some(KEY));
    assert_eq!(status(server.port, "GET", "/health", &[]), Some(200));
    assert_eq!(
        status(server.port, "GET", "/health/metrics", &[]),
        Some(401),
        "metrics name tables and rode in on the /health prefix"
    );
}

/// A browser asks permission before it asks for data, and the question must not need a key.
///
/// This is the bug this file did not catch. The middleware was first applied after the CORS
/// layer, which — since `layer` wraps inward-out — put it *outside*, so an `OPTIONS` preflight
/// got a bare 401 with no CORS headers on it. The browser then never sends the real request,
/// and what the operator sees is a CORS error that says nothing about a key. curl sends no
/// preflight, so every test here passed.
#[test]
fn a_cors_preflight_is_answered_without_a_key() {
    let server = start(Some(KEY));
    let headers = [
        ("Origin", "http://localhost:3000"),
        ("Access-Control-Request-Method", "GET"),
        ("Access-Control-Request-Headers", "api-key"),
    ];
    let (code, _) =
        call(server.port, "OPTIONS", "/api/v1/tables", &headers).expect("an answer to preflight");
    assert_eq!(
        code, 200,
        "a preflight needs no key; a browser cannot attach one to it"
    );

    // And the answer has to permit the header we then send, or the request is blocked anyway.
    let allowed = header_value(
        server.port,
        "OPTIONS",
        "/api/v1/tables",
        &headers,
        "access-control-allow-headers",
    )
    .expect("no access-control-allow-headers on the preflight");
    assert!(
        allowed.to_ascii_lowercase().contains("api-key"),
        "the preflight does not allow the header the key travels in: {allowed}"
    );
}

/// One response header, lowercased name, as curl reports it.
fn header_value(
    port: u16,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    wanted: &str,
) -> Option<String> {
    http::request(
        port,
        method,
        path,
        headers,
        None,
        std::time::Duration::from_secs(30),
    )?
    .header(wanted)
    .map(str::to_string)
}

/// A 401 says how to retry. Without it a browser prompt has nothing to show and a client has
/// to guess the scheme.
#[test]
fn the_rejection_names_the_scheme() {
    let server = start(Some(KEY));
    let value = header_value(server.port, "GET", "/status", &[], "www-authenticate")
        .expect("no www-authenticate on the 401");
    assert!(value.starts_with("Bearer"), "{value}");
}

/// A blank key is a mistake, not a secret. Accepting one gives a server that looks protected
/// and is open, which is the worst of the three states.
#[test]
fn a_blank_key_leaves_the_server_open_rather_than_half_protected() {
    let server = start(Some("   "));
    assert_eq!(
        status(server.port, "GET", "/api/v1/tables", &[]),
        Some(200),
        "a whitespace key should configure nothing at all"
    );
}

/// The compatibility promise reaches here too: the same rename fallback, end to end.
#[test]
fn the_legacy_variable_name_still_configures_the_key() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DXMT_WAL_DIR", dir.path().join("wal"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env_remove("DXMT_API_KEY")
        .env("DB_API_KEY", KEY)
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
        if status(port, "GET", "/live", &[]).is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }

    assert_eq!(
        status(server.port, "GET", "/api/v1/tables", &[]),
        Some(401),
        "DB_API_KEY did not take effect, so an upgrade would silently drop authentication"
    );
    assert_eq!(
        status(server.port, "GET", "/api/v1/tables", &[("api-key", KEY)]),
        Some(200)
    );
}
