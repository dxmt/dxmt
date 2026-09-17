//! The read-only web dashboard, served from bytes compiled into the binary.
//!
//! # Why the assets are embedded rather than read from a directory
//!
//! The product ships as one binary — the Docker image copies `/usr/local/bin/vectordb` and
//! nothing else — so a dashboard that needs a `static/` directory beside it is a dashboard that
//! 404s in the image and in every `cargo install`. `rust-embed` puts the built bundle inside
//! the executable, which keeps that property and makes "binary and assets out of sync"
//! unrepresentable.
//!
//! # Why an empty asset set is allowed to compile, and then refuses to serve
//!
//! `#[allow_missing]` lets the derive compile when `dashboard/dist/` has never been built, which
//! is the state of every clean checkout and of CI's `optional-features` job — that job's whole
//! purpose is to type-check the cfg-gated code, and it must not need Node to do it.
//!
//! The cost of that is a build which links happily and serves a white page. So
//! [`assets_are_embedded`] exists and `server.rs` turns a false into a startup error naming the
//! command to run. Without that guard, `allow_missing` converts an honest compile failure into
//! a silent one, and this file would be the mechanism. **If the guard is ever removed, remove
//! `allow_missing` with it.**
//!
//! # Caching, which is load-bearing rather than an optimisation
//!
//! Vite emits content-hashed files under `assets/` and an `index.html` that names them. So the
//! hashed files are `immutable` — the name changes when the bytes do — and `index.html` is
//! `no-cache`, because a cached copy of it points at bundles that a redeployed server no longer
//! has. That pair is the whole correctness story for a redeploy, and `tests/dashboard_test.rs`
//! pins both halves.

use axum::{
    http::{header, HeaderMap, HeaderValue, StatusCode, Uri},
    response::{IntoResponse, Redirect, Response},
    routing::get,
    Router,
};
use tower_http::{compression::CompressionLayer, trace::TraceLayer};

/// The built single-page app. Empty until `make dashboard-build` has run.
#[derive(rust_embed::Embed)]
#[folder = "dashboard/dist/"]
#[allow_missing = true]
struct DashboardAssets;

/// Whether this binary actually carries a dashboard.
///
/// `server.rs` calls this before serving and refuses to start when it is false. See the module
/// docs for why that refusal is the other half of `allow_missing`.
pub fn assets_are_embedded() -> bool {
    DashboardAssets::get("index.html").is_some()
}

/// The dashboard's routes, generic over the state so this can be merged before `with_state`.
///
/// Both layers are scoped here rather than taken from the server's global stack, for two
/// different reasons.
///
/// `CompressionLayer`, because the bundle is the one response this server sends where transfer
/// size dominates, and because putting it on the global stack would switch every API response
/// to chunked with a `Content-Encoding` — for every SDK and every functional test. That may
/// well be worth doing; it is a separate change with its own measurement.
///
/// `TraceLayer`, because this router is merged *outside* the server's layer stack and would
/// otherwise be the one surface with no request logging. It is outside because the global
/// `CorsLayer` replaces `Vary` wholesale with its own values, which silently undid the
/// `Vary: accept-encoding` that makes the compression above safe for a shared cache. A test
/// caught that; without `Vary`, a cache can hand a gzip body to a client that never asked.
pub fn create_dashboard_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        // Without the trailing slash a relative asset reference would resolve against `/`
        // rather than `/dashboard/`. Redirect rather than serve both, so there is one URL.
        .route(
            "/dashboard",
            get(|| async { Redirect::permanent("/dashboard/") }),
        )
        .route("/dashboard/", get(index))
        .route("/dashboard/*path", get(asset))
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
}

async fn index(headers: HeaderMap) -> Response {
    serve("index.html", &headers)
}

async fn asset(uri: Uri, headers: HeaderMap) -> Response {
    let path = uri.path().trim_start_matches("/dashboard/");

    if DashboardAssets::get(path).is_some() {
        return serve(path, &headers);
    }

    // History fallback: `/dashboard/tables/foo/search` is a client-side route, not a file, and
    // must return the app so a refresh or a pasted link works.
    //
    // Only when the last segment has no dot in it. A missing `.js` or `.css` served as HTML is
    // the worst possible answer — the browser reports a syntax error in what it thinks is a
    // script, and the real problem (a partial or stale embed) is invisible. A 404 says it.
    let last = path.rsplit('/').next().unwrap_or("");
    if last.contains('.') {
        return (StatusCode::NOT_FOUND, "").into_response();
    }
    serve("index.html", &headers)
}

fn serve(path: &str, request: &HeaderMap) -> Response {
    let Some(file) = DashboardAssets::get(path) else {
        // Only reachable for `index.html` on a binary built without the bundle, which
        // `server.rs` refuses to start. Answer honestly rather than with an empty 404.
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "The dashboard was not built into this binary. Run `make dashboard-build` and \
             rebuild with `--features api-dashboard`.",
        )
            .into_response();
    };

    // rust-embed hands us the sha256 of the contents, which is exactly what an ETag wants.
    let etag = format!("\"{}\"", hex16(&file.metadata.sha256_hash()));
    if request
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|candidate| candidate.trim() == etag))
    {
        // 304 must repeat the headers that would have varied, and must carry no body.
        return (StatusCode::NOT_MODIFIED, cache_headers(path, &etag)).into_response();
    }

    let mut headers = cache_headers(path, &etag);
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(file.metadata.mimetype())
            .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream")),
    );
    (headers, file.data.into_owned()).into_response()
}

/// The headers that are the same on a 200 and on a 304.
fn cache_headers(path: &str, etag: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::ETAG,
        HeaderValue::from_str(etag).unwrap_or_else(|_| HeaderValue::from_static("\"\"")),
    );
    // `CompressionLayer` does not vary the ETag by encoding, so the same tag denotes the
    // identity and the compressed entity. `Vary` is what keeps a shared cache from handing a
    // br-encoded body to a client that cannot read one.
    headers.insert(header::VARY, HeaderValue::from_static("accept-encoding"));
    headers.insert(
        header::CACHE_CONTROL,
        if path.starts_with("assets/") {
            // Content-hashed by Vite: a change is a new name, so this copy is good forever.
            HeaderValue::from_static("public, max-age=31536000, immutable")
        } else {
            // `index.html` names the hashed bundles. A cached one points at bundles a
            // redeployed server no longer has, and the page breaks with no way for the user to
            // fix it but a hard reload. `no-cache` means revalidate, not "do not store" — with
            // the ETag above it costs one 304.
            HeaderValue::from_static("no-cache")
        },
    );

    // The page is same-origin with the API and loads nothing from anywhere else. Saying so in a
    // CSP is what makes that a guarantee rather than a habit — and it is the difference between
    // this page and the Swagger one, which fetches from a CDN and is therefore blank on an
    // air-gapped host.
    if !path.starts_with("assets/") {
        headers.insert(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static(
                "default-src 'self'; connect-src 'self'; img-src 'self' data:; \
                 style-src 'self' 'unsafe-inline'; script-src 'self'; font-src 'self' data:; \
                 worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'",
            ),
        );
        headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
        headers.insert(
            header::REFERRER_POLICY,
            HeaderValue::from_static("no-referrer"),
        );
    }
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers
}

/// The first eight bytes of a hash, hex. Enough for an ETag; the whole 32 would be noise.
fn hex16(hash: &[u8; 32]) -> String {
    hash[..8].iter().map(|b| format!("{b:02x}")).collect()
}
