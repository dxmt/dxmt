//! A shared secret in front of the HTTP API, off unless `DXMT_API_KEY` is set.
//!
//! This is a seatbelt, not a vault. The admin key is one static key with no rotation, and it
//! does nothing about an attacker who can read the environment or the process list. Tenants
//! (`DXMT_TENANTS_FILE`, below) add per-tenant keys that reach only their own memories; tables
//! have no per-tenant scoping and stay with the admin key. What it does is turn an accidentally-reachable port from "anyone who
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
use sha2::{Digest, Sha256};
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

/// Who a request acts for, inserted into the request by `require_key` for the handlers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Identity {
    /// The admin key, or a server with no keys at all: the whole store.
    Admin,
    /// One tenant: its own memories, through the memory API and the gateway, and nothing else.
    Tenant(String),
}

impl Identity {
    /// The tag a tenant's memories carry (`tenant:<id>`); `None` for the admin.
    pub fn scope(&self) -> Option<String> {
        match self {
            Identity::Admin => None,
            Identity::Tenant(id) => Some(format!(
                "{}{id}",
                crate::storage::agent_memory::TENANT_TAG_PREFIX
            )),
        }
    }
}

/// The scope a handler applies: its request's tenant, or none.
pub fn scope_of(identity: Option<&Identity>) -> Option<String> {
    identity.and_then(Identity::scope)
}

/// One tenant: an id and the SHA-256 of its key. The key itself is never stored or logged.
struct Tenant {
    id: String,
    key_sha256: [u8; 32],
}

/// The tenants in `DXMT_TENANTS_FILE`: `{"tenants": [{"id": "acme", "key_sha256": "<64 hex>"}]}`.
///
/// Per-tenant keys for a server that holds several people's or apps' memory. A tenant's key
/// reaches only the memory API (`/api/v1/memories...`) and the gateway (`/v1/...`), and there
/// only its own memories: the server tags what it writes `tenant:<id>` and filters every read,
/// duplicate check, supersede, forget and extraction on that tag. Tables, records, snapshots,
/// backups, DXQL and MCP stay with the admin key (`DXMT_API_KEY`). With tenants configured, a
/// request with no key is refused even when there is no admin key.
pub struct Tenants(Vec<Tenant>);

/// The file's path.
pub const TENANTS_VARIABLE: &str = "DXMT_TENANTS_FILE";

impl Tenants {
    fn load() -> crate::Result<Self> {
        let Some(path) = crate::config::env::var(TENANTS_VARIABLE)
            .ok()
            .filter(|p| !p.trim().is_empty())
        else {
            return Ok(Self(Vec::new()));
        };
        let text = std::fs::read_to_string(path.trim()).map_err(|e| {
            crate::Error::InvalidConfig(format!("{TENANTS_VARIABLE} ({path}): {e}"))
        })?;
        Self::parse(&text)
    }

    fn parse(text: &str) -> crate::Result<Self> {
        let bad = |why: String| crate::Error::InvalidConfig(format!("{TENANTS_VARIABLE}: {why}"));
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| bad(format!("not JSON: {e}")))?;
        let mut tenants = Vec::new();
        for entry in value["tenants"]
            .as_array()
            .ok_or_else(|| bad("needs a `tenants` array".into()))?
        {
            let id = entry["id"].as_str().unwrap_or("").to_string();
            if id.is_empty()
                || id.len() > 64
                || !id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Err(bad(format!(
                    "tenant id {id:?}: 1-64 of a-z, A-Z, 0-9, '-' and '_'"
                )));
            }
            let hex = entry["key_sha256"].as_str().unwrap_or("");
            let key_sha256 = decode_sha256(hex).ok_or_else(|| {
                bad(format!(
                    "tenant {id}: `key_sha256` must be 64 hex digits (sha256sum of the key)"
                ))
            })?;
            if tenants.iter().any(|t: &Tenant| t.id == id) {
                return Err(bad(format!("tenant {id} appears twice")));
            }
            tenants.push(Tenant { id, key_sha256 });
        }
        Ok(Self(tenants))
    }

    /// The configured tenants, read once. A bad file is a startup error (`server` checks it).
    pub fn global() -> crate::Result<&'static Tenants> {
        static TENANTS: std::sync::OnceLock<Result<Tenants, String>> = std::sync::OnceLock::new();
        match TENANTS.get_or_init(|| Tenants::load().map_err(|e| e.to_string())) {
            Ok(t) => Ok(t),
            Err(e) => Err(crate::Error::InvalidConfig(e.clone())),
        }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The tenant whose key this is, compared in constant time over every tenant.
    fn find(&self, presented: &str) -> Option<&str> {
        let digest: [u8; 32] = Sha256::digest(presented.as_bytes()).into();
        let mut found = None;
        for tenant in &self.0 {
            if bool::from(tenant.key_sha256.ct_eq(&digest)) {
                found = Some(tenant.id.as_str());
            }
        }
        found
    }
}

fn decode_sha256(hex: &str) -> Option<[u8; 32]> {
    let hex = hex.trim();
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(hex.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

/// What a tenant's key reaches: the memory API and the gateway.
fn tenant_may_reach(path: &str) -> bool {
    path == "/api/v1/memories"
        || path.starts_with("/api/v1/memories/")
        || path == "/v1/chat/completions"
        || path == "/v1/models"
}

/// Reject a request that carries no valid key; say who a valid one acts for.
pub async fn require_key<B>(
    axum::extract::State(key): axum::extract::State<Option<ApiKey>>,
    mut request: Request<B>,
    next: Next<B>,
) -> Response {
    let tenants = match Tenants::global() {
        Ok(t) => t,
        Err(_) => return unauthorized(),
    };
    if key.is_none() && tenants.is_empty() {
        request.extensions_mut().insert(Identity::Admin);
        return next.run(request).await;
    }

    let path = request.uri().path().to_string();
    if OPEN.contains(&path.as_str()) {
        return next.run(request).await;
    }

    let Some(presented) = presented(&request).map(str::to_string) else {
        return unauthorized();
    };
    if key.as_ref().is_some_and(|key| key.matches(&presented)) {
        request.extensions_mut().insert(Identity::Admin);
        return next.run(request).await;
    }
    if let Some(tenant) = tenants.find(&presented) {
        if !tenant_may_reach(&path) {
            return forbidden(&path);
        }
        request
            .extensions_mut()
            .insert(Identity::Tenant(tenant.to_string()));
        return next.run(request).await;
    }

    unauthorized()
}

/// A 403 for a tenant's key on a route that is not memory or the gateway.
fn forbidden(path: &str) -> Response {
    let body = ApiResponse::error(ApiError {
        code: "FORBIDDEN".to_string(),
        message: format!(
            "{path}: a tenant's key reaches only the memory API (/api/v1/memories) and the gateway \
             (/v1/chat/completions); everything else needs the admin key"
        ),
        details: None,
    });
    (StatusCode::FORBIDDEN, Json(body)).into_response()
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
    fn tenants_are_read_by_hash_and_found_only_by_their_key() {
        let hash = |k: &str| -> String {
            Sha256::digest(k.as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        };
        let tenants = Tenants::parse(&format!(
            r#"{{"tenants":[{{"id":"acme","key_sha256":"{}"}},{{"id":"b-2","key_sha256":"{}"}}]}}"#,
            hash("acme-key"),
            hash("other")
        ))
        .unwrap();
        assert_eq!(tenants.len(), 2);
        assert_eq!(tenants.find("acme-key"), Some("acme"));
        assert_eq!(tenants.find("other"), Some("b-2"));
        assert_eq!(
            tenants.find("acme-key "),
            None,
            "no trimming of what is presented"
        );
        assert_eq!(tenants.find(""), None);
    }

    #[test]
    fn a_bad_tenants_file_is_refused() {
        let ok = "a".repeat(64);
        for bad in [
            "not json".to_string(),
            r#"{"tenant": []}"#.to_string(),
            format!(r#"{{"tenants":[{{"id":"","key_sha256":"{ok}"}}]}}"#),
            format!(r#"{{"tenants":[{{"id":"has space","key_sha256":"{ok}"}}]}}"#),
            r#"{"tenants":[{"id":"a","key_sha256":"abc"}]}"#.to_string(),
            format!(
                r#"{{"tenants":[{{"id":"a","key_sha256":"{ok}"}},{{"id":"a","key_sha256":"{ok}"}}]}}"#
            ),
        ] {
            assert!(Tenants::parse(&bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn a_tenant_reaches_memory_and_the_gateway_only() {
        for open in [
            "/api/v1/memories",
            "/api/v1/memories/recall",
            "/api/v1/memories/x",
            "/v1/chat/completions",
            "/v1/models",
        ] {
            assert!(tenant_may_reach(open), "{open}");
        }
        for closed in [
            "/api/v1/tables",
            "/api/v1/memoriesx",
            "/mcp",
            "/status",
            "/api/v1/query",
            "/v1/embeddings",
        ] {
            assert!(!tenant_may_reach(closed), "{closed}");
        }
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
