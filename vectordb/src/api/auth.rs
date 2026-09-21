//! A shared secret in front of the HTTP API, off unless `DXMT_API_KEY` is set.
//!
//! This is a seatbelt, not a vault. It is one static key with no rotation, no accounts and no
//! per-table scoping, and it does nothing about an attacker who can read the environment or
//! the process list. What it does is turn an accidentally-reachable port from "anyone who
//! finds it owns the data" into "anyone who finds it gets 401", which is the difference that
//! matters in practice, because the way this goes wrong is almost never a determined attacker
//! — it is a demo box that outlived its firewall rule.
//!
//! The shape follows the field. Qdrant reads `service.api_key` and takes an `api-key` header;
//! Chroma and Weaviate both read a key from the environment; all three default to off. We use
//! the same header name so an existing Qdrant-shaped client needs no new code, and accept
//! `Authorization: Bearer` as well because that is what everything else sends.
//!
//! **The probes are deliberately open.** `/health`, `/live` and `/ready` answer without a key:
//! a Kubernetes liveness probe cannot carry one, and a server that fails its own health check
//! because authentication was switched on is a worse outage than the one the key prevents.
//! They report liveness and durability state, not data. Everything else, `/status` and
//! `/health/metrics` included, needs the key — those name tables.

use axum::{
    body::BoxBody,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use subtle::ConstantTimeEq;

use crate::api::response::{ApiError, ApiResponse};

/// The variable that switches it on.
pub const KEY_VARIABLE: &str = "DXMT_API_KEY";

/// The header a client sends, matching Qdrant's.
pub const HEADER: &str = "api-key";

/// Paths that answer without a key. Exact matches only — no prefix, or `/health/metrics`
/// would ride in on `/health`.
const OPEN: [&str; 8] = [
    "/health",
    "/live",
    "/ready",
    "/health/live",
    "/health/ready",
    "/api/v1/health",
    "/api/v1/health/live",
    "/api/v1/health/ready",
];

/// The configured key, or `None` when the API is open.
#[derive(Clone)]
pub struct ApiKey(std::sync::Arc<String>);

impl ApiKey {
    /// Read the key from the environment.
    ///
    /// Absent or blank means no authentication. A key made only of whitespace is a mistake, not
    /// a secret, and treating it as one would give a server that looks protected and is not.
    pub fn from_env() -> Option<Self> {
        Self::parse(crate::config::env::var(KEY_VARIABLE).ok().as_deref())
    }

    /// The same, from a value rather than the environment, which is what the tests use.
    ///
    /// They used to set `DXMT_API_KEY` and read it back, two tests doing so on parallel threads
    /// with nothing ordering them: one set `"  padded  "`, the other set `""` in between, and
    /// the first found no key at all. It failed once in a full run and passed on every run
    /// after, which is the shape of the same mistake `Layout` made and was fixed for.
    pub fn parse(value: Option<&str>) -> Option<Self> {
        let key = value?.trim();
        if key.is_empty() {
            return None;
        }
        Some(Self(std::sync::Arc::new(key.to_string())))
    }

    /// Whether `candidate` is the key, compared in constant time.
    ///
    /// Not because a remote timing attack on an HTTP string compare is likely — it is not —
    /// but because the alternative costs one line and this is the file people read to decide
    /// whether the rest of the server was written carefully.
    pub fn matches(&self, candidate: &str) -> bool {
        // `ct_eq` is only constant-time over equal-length inputs; the length itself is not a
        // secret worth protecting here.
        self.0.as_bytes().ct_eq(candidate.as_bytes()).into()
    }

    /// What to print at startup. Never the key.
    pub fn describe(&self) -> String {
        format!("{} characters", self.0.chars().count())
    }
}

/// Reject a request that carries no valid key.
pub async fn require_key<B>(
    axum::extract::State(key): axum::extract::State<Option<ApiKey>>,
    request: Request<B>,
    next: Next<B>,
) -> Response {
    let Some(key) = key else {
        return next.run(request).await;
    };

    let path = request.uri().path();
    if OPEN.contains(&path) {
        return next.run(request).await;
    }

    if presented(&request).is_some_and(|presented| key.matches(presented)) {
        return next.run(request).await;
    }

    unauthorized()
}

/// The key a request presents, by either accepted route.
fn presented<B>(request: &Request<B>) -> Option<&str> {
    let headers = request.headers();
    if let Some(value) = headers.get(HEADER).and_then(|v| v.to_str().ok()) {
        return Some(value);
    }
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

/// A 401 in the same envelope as every other error, so a client that already parses
/// `error.code` does not need a second shape for this one.
fn unauthorized() -> Response {
    let body = ApiResponse::error(ApiError {
        code: "UNAUTHORIZED".to_string(),
        message: format!("missing or invalid API key; send it as the `{HEADER}` header"),
        details: None,
    });
    let mut response: Response<BoxBody> = (StatusCode::UNAUTHORIZED, Json(body)).into_response();
    // Says which scheme to retry with, rather than leaving a client to guess. The realm names
    // the product because a browser shows it in the prompt.
    response.headers_mut().insert(
        axum::http::header::WWW_AUTHENTICATE,
        axum::http::HeaderValue::from_static("Bearer realm=\"dxmt\""),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(value: &str) -> ApiKey {
        ApiKey(std::sync::Arc::new(value.to_string()))
    }

    #[test]
    fn a_key_matches_only_itself() {
        let key = key("s3cret");
        assert!(key.matches("s3cret"));
        assert!(!key.matches("s3cres"));
        assert!(!key.matches("s3cret "), "no trimming of what was presented");
        assert!(!key.matches(""));
        assert!(!key.matches("s3cret-and-more"), "a prefix is not the key");
    }

    /// The failure that would matter: a server that looks protected and is not.
    #[test]
    fn a_blank_key_is_no_key() {
        for blank in ["", "   ", "\t\n"] {
            assert!(
                ApiKey::parse(Some(blank)).is_none(),
                "{blank:?} was accepted as a key"
            );
        }
        assert!(ApiKey::parse(None).is_none());
    }

    /// Surrounding whitespace in the environment is almost always the shell's, not the
    /// operator's — but only the configured side is trimmed, never what a client sends.
    #[test]
    fn the_configured_key_is_trimmed() {
        let key = ApiKey::parse(Some("  padded  ")).expect("a key");
        assert!(key.matches("padded"));
        assert!(!key.matches("  padded  "));
    }

    #[test]
    fn the_description_never_contains_the_key() {
        let described = key("s3cret").describe();
        assert!(!described.contains("s3cret"), "{described}");
        assert_eq!(described, "6 characters");
    }

    #[test]
    fn both_header_forms_are_read() {
        let bearer = Request::builder()
            .header(axum::http::header::AUTHORIZATION, "Bearer s3cret")
            .body(())
            .unwrap();
        assert_eq!(presented(&bearer), Some("s3cret"));

        let api_key = Request::builder()
            .header(HEADER, "s3cret")
            .body(())
            .unwrap();
        assert_eq!(presented(&api_key), Some("s3cret"));

        // `api-key` wins, so a proxy's own Authorization header cannot shadow the real one.
        let both = Request::builder()
            .header(HEADER, "s3cret")
            .header(axum::http::header::AUTHORIZATION, "Bearer wrong")
            .body(())
            .unwrap();
        assert_eq!(presented(&both), Some("s3cret"));

        let basic = Request::builder()
            .header(axum::http::header::AUTHORIZATION, "Basic abc")
            .body(())
            .unwrap();
        assert_eq!(presented(&basic), None, "only Bearer is a key");

        let none = Request::builder().body(()).unwrap();
        assert_eq!(presented(&none), None);
    }

    /// Exact matches, so a path that merely starts with an open one is still protected.
    #[test]
    fn the_open_list_does_not_leak_by_prefix() {
        assert!(OPEN.contains(&"/health"));
        assert!(!OPEN.contains(&"/health/metrics"), "it names tables");
        assert!(!OPEN.contains(&"/status"), "it names tables");
        assert!(!OPEN.contains(&"/api/v1/tables"));
    }
}
