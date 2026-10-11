//! The dashboard as an HTTP surface: does the bundle reach a browser, and does it stay correct
//! across a redeploy.
//!
//! `cargo test --features api-dashboard --test dashboard_test`
//!
//! Almost every assertion here is about caching or about a wrong answer being loud. Those two
//! are where a served single-page app goes wrong in ways that are invisible from the server
//! side: `index.html` names content-hashed bundles, so a cached copy of it after a redeploy
//! points at files that no longer exist and the page breaks with nothing in the log; and a
//! missing `.js` answered with HTML produces a syntax error in the console rather than a 404,
//! which sends the reader looking in entirely the wrong place.
//!
//! **When the bundle has been built, a failure here is a failure.** `api-dashboard` used to be
//! opt-in, so compiling with it meant intending to ship the console, and this file failed
//! outright when `dashboard/dist/` was absent — a skipped test and an absent one look identical
//! in CI output. The feature is on by default now, so that inference no longer holds: a
//! checkout without Node builds a binary with the routes and none of the bytes, and failing
//! there would mean such a checkout cannot run its own test suite. So the rule is now read off
//! the disk rather than inferred: `dashboard/dist/index.html` present means the bundle was
//! built and everything below must hold; absent means the console was never built here and
//! each test says so and stops.

#![cfg(feature = "api-dashboard")]

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

/// Whether a bundle was built in this checkout. Not whether this binary carries one -- if the
/// two disagree, the binary is stale, and the assertions below say so rather than skipping.
fn bundle_was_built() -> bool {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("dashboard/dist/index.html")
        .exists()
}

/// `return` out of a test when there is no console in this checkout to test.
macro_rules! needs_bundle {
    () => {
        if !bundle_was_built() {
            eprintln!(
                "skipping: dashboard/dist/ has not been built in this checkout. \
                 `make dashboard-build` to build it."
            );
            return;
        }
    };
}

/// `start(true)` asks for the console by name, `start(false)` turns it off with
/// `--no-dashboard`. It is served by default now, so "not asked for" no longer means "absent".
/// A server started the way a person starts one: no flags but the port.
fn start_plain() -> Server {
    start_argv(&[])
}

fn start(with_dashboard: bool) -> Server {
    start_argv(if with_dashboard {
        &["--dashboard"]
    } else {
        &["--no-dashboard"]
    })
}

fn start_argv(extra: &[&str]) -> Server {
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
    command.args(extra);
    let child = command
        .spawn()
        .expect("failed to spawn the vectordb binary");
    let server = Server {
        child,
        port,
        _dir: dir,
    };

    for _ in 0..120 {
        if fetch(server.port, "/health", &[]).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {}", server.port);
}

struct Reply {
    status: u16,
    headers: String,
    body: String,
}

impl Reply {
    fn header(&self, name: &str) -> Option<String> {
        self.headers_all(name).into_iter().next()
    }

    /// Every value of a header, because some arrive more than once.
    ///
    /// `Vary` does: the handler sets `accept-encoding` and the CORS layer appends `origin`,
    /// and both are legal -- a cache combines them. Reading only the first found `origin` and
    /// concluded the encoding was not declared.
    fn headers_all(&self, name: &str) -> Vec<String> {
        let needle = format!("{}:", name.to_ascii_lowercase());
        self.headers
            .lines()
            .filter(|line| line.to_ascii_lowercase().starts_with(&needle))
            .map(|line| line[needle.len()..].trim().to_string())
            .collect()
    }

    fn header_mentions(&self, name: &str, value: &str) -> bool {
        self.headers_all(name)
            .iter()
            .any(|header| header.to_ascii_lowercase().contains(value))
    }
}

fn fetch(port: u16, path: &str, extra: &[&str]) -> Option<Reply> {
    // `extra` is still written as curl's `-H "Name: value"` pairs, which is how the callers
    // read best. No `Accept-Encoding` unless one is given: whether a body arrives encoded is
    // one of the things under test.
    let headers: Vec<(&str, &str)> = extra
        .chunks(2)
        .filter(|pair| pair[0] == "-H")
        .filter_map(|pair| pair[1].split_once(':'))
        .map(|(name, value)| (name.trim(), value.trim()))
        .collect();
    let reply = http::request(
        port,
        "GET",
        path,
        &headers,
        None,
        std::time::Duration::from_secs(15),
    )?;
    Some(Reply {
        status: reply.status,
        headers: reply
            .headers
            .iter()
            .map(|(name, value)| format!("{name}: {value}\n"))
            .collect(),
        body: reply.text(),
    })
}

fn index_of(server: &Server) -> Reply {
    let reply = fetch(server.port, "/dashboard/", &[]).expect("no reply");
    assert_ne!(
        reply.status, 503,
        "the bundle is not embedded in this binary. Run `make dashboard-build` and rebuild \
         with --features api-dashboard. This is a failure rather than a skip because a skipped \
         test and an absent one look the same in a CI log."
    );
    reply
}

/// The first `/dashboard/assets/...` URL the page references.
fn first_asset(body: &str) -> String {
    let start = body
        .find("/dashboard/assets/")
        .expect("index.html references no hashed asset, so nothing was bundled");
    let rest = &body[start..];
    let end = rest
        .find(['"', '\''])
        .expect("an asset URL with no closing quote");
    rest[..end].to_string()
}

#[test]
fn the_page_is_served_and_carries_the_headers_that_make_it_safe() {
    needs_bundle!();
    let server = start(true);
    let reply = index_of(&server);

    assert_eq!(reply.status, 200);
    assert!(reply
        .header("content-type")
        .unwrap()
        .starts_with("text/html"));
    assert!(
        reply.body.contains("id=\"root\""),
        "this is not the app's shell"
    );

    // `script-src 'self'` is the load-bearing one: it forbids a CDN, which is the difference
    // between this page and the Swagger page, which fetches from jsdelivr and is therefore
    // blank on an air-gapped host.
    let csp = reply.header("content-security-policy").expect("no CSP");
    assert!(
        csp.contains("script-src 'self'"),
        "CSP allows scripts from elsewhere: {csp}"
    );
    assert!(csp.contains("frame-ancestors 'none'"));
    assert_eq!(reply.header("x-frame-options").as_deref(), Some("DENY"));
    assert_eq!(
        reply.header("x-content-type-options").as_deref(),
        Some("nosniff")
    );
}

#[test]
fn index_is_revalidated_and_hashed_assets_are_not() {
    needs_bundle!();
    // The whole correctness story for a redeploy. `index.html` names the hashed bundles, so a
    // cached copy points at files the new server does not have -- a page that breaks with no
    // error anywhere and no fix but a hard reload. Relaxing either half brings that back.
    let server = start(true);
    let index = index_of(&server);
    assert_eq!(
        index.header("cache-control").as_deref(),
        Some("no-cache"),
        "index.html must be revalidated"
    );

    let asset = first_asset(&index.body);
    let reply = fetch(server.port, &asset, &[]).expect("no reply");
    assert_eq!(reply.status, 200);
    let cache = reply.header("cache-control").unwrap();
    assert!(
        cache.contains("immutable") && cache.contains("max-age=31536000"),
        "a content-hashed asset should be cacheable forever: {cache}"
    );
    assert!(
        reply.header("content-type").unwrap().contains("javascript"),
        "a .js asset came back as {:?}, which a browser will refuse to execute",
        reply.header("content-type")
    );
}

#[test]
fn a_matching_etag_is_answered_with_304_and_no_body() {
    needs_bundle!();
    let server = start(true);
    let index = index_of(&server);
    let etag = index.header("etag").expect("index.html has no ETag");

    let again = fetch(
        server.port,
        "/dashboard/",
        &["-H", &format!("If-None-Match: {etag}")],
    )
    .unwrap();

    assert_eq!(
        again.status, 304,
        "an unchanged page should not be sent twice"
    );
    assert!(again.body.is_empty(), "a 304 must carry no body");
    // The headers that would have varied must be repeated, or the next request has no ETag to
    // send and `no-cache` costs a full transfer every time instead of one 304.
    assert!(
        again.headers_all("etag").contains(&etag),
        "the 304 dropped the ETag, so the next request has none to send"
    );
    assert_eq!(again.header("cache-control").as_deref(), Some("no-cache"));
}

#[test]
fn a_deep_link_serves_the_app_but_a_missing_file_is_a_404() {
    needs_bundle!();
    let server = start(true);
    let _ = index_of(&server);

    // A client-side route. Reloading on one, or pasting one to a colleague, has to work.
    let deep = fetch(server.port, "/dashboard/tables/mine/search", &[]).unwrap();
    assert_eq!(deep.status, 200);
    assert!(deep
        .header("content-type")
        .unwrap()
        .starts_with("text/html"));

    // But a missing asset must not be answered with the app. Served as HTML, the browser
    // reports a syntax error in what it believes is a script, and the real cause -- a partial
    // or stale embed -- is nowhere in the message.
    let missing = fetch(server.port, "/dashboard/assets/nope-12345678.js", &[]).unwrap();
    assert_eq!(
        missing.status, 404,
        "a missing .js was answered {} — the history fallback is too greedy",
        missing.status
    );
}

#[test]
fn the_bundle_is_compressed_when_the_browser_asks() {
    needs_bundle!();
    let server = start(true);
    let index = index_of(&server);
    let asset = first_asset(&index.body);

    let reply = fetch(server.port, &asset, &["-H", "Accept-Encoding: gzip"]).unwrap();
    assert_eq!(
        reply.header("content-encoding").as_deref(),
        Some("gzip"),
        "the bundle was sent uncompressed; the sub-router's CompressionLayer is not wired"
    );
    assert!(
        reply.header_mentions("vary", "accept-encoding"),
        "without Vary, a shared cache can hand a gzip body to a client that did not ask; got {:?}",
        reply.headers_all("vary")
    );
}

#[test]
fn the_api_is_not_compressed_because_that_is_a_separate_decision() {
    needs_bundle!();
    // The compression layer is scoped to the dashboard on purpose: turning it on globally
    // changes every API response's framing for every SDK and every functional test. If that
    // ever becomes the right call it should be measured, not arrive as a side effect of
    // serving a JavaScript file.
    let server = start(true);
    let reply = fetch(
        server.port,
        "/api/v1/tables",
        &["-H", "Accept-Encoding: gzip"],
    )
    .unwrap();
    assert_eq!(reply.status, 200);
    assert_eq!(
        reply.header("content-encoding"),
        None,
        "the API is now compressed; that is a larger change than this one and needs its own \
         measurement"
    );
}

/// The console is served without being asked for, and `--no-dashboard` takes it away.
///
/// It used to be the other way round -- `--dashboard` to get it -- and the test here asserted
/// that the runtime gate was real, because there was a period when all three optional
/// interfaces were compiled in and unreachable behind a hardcoded `false`. The gate is still
/// real; the default moved. Both directions are asserted, because a default that cannot be
/// turned off is the same defect with the sign flipped.
#[test]
fn the_console_is_on_by_default_and_can_be_turned_off() {
    needs_bundle!();
    let by_default = start_plain();
    assert_eq!(
        fetch(by_default.port, "/dashboard/", &[]).unwrap().status,
        200,
        "the console did not answer on a server started with no flags at all"
    );

    let refused = start(false);
    assert_eq!(
        fetch(refused.port, "/dashboard/", &[]).unwrap().status,
        404,
        "--no-dashboard served the console anyway"
    );
}

#[test]
fn a_path_that_climbs_out_of_the_bundle_is_refused() {
    needs_bundle!();
    let server = start(true);
    let _ = index_of(&server);
    for path in [
        "/dashboard/../../etc/passwd",
        "/dashboard/..%2f..%2fetc%2fpasswd",
        "/dashboard/assets/../../../Cargo.toml",
    ] {
        let reply = fetch(server.port, path, &[]).unwrap();
        assert_ne!(
            reply.status,
            200,
            "{path} was served with status 200 and a {} byte body",
            reply.body.len()
        );
        assert!(
            !reply.body.contains("[package]"),
            "{path} leaked a file from outside the bundle"
        );
    }
}
