//! One URL per API route, with or without a trailing slash.
//!
//! The router matches paths exactly, so `/api/v1/tables/docs/` answered an empty 404 while
//! `/api/v1/tables/docs` answered the table -- a hand-written curl or a client that joins
//! paths with a trailing `/` got nothing and no reason. This trims the trailing slash before
//! routing, so both reach the same handler.
//!
//! Not under `/dashboard` or `/api/docs`: those are web apps whose pages load assets by
//! relative path, which needs the slash -- `/dashboard` redirects to `/dashboard/` on purpose,
//! and trimming it back would be a redirect loop.

use axum::{
    extract::Request,
    http::{uri::PathAndQuery, Uri},
    middleware::Next,
    response::Response,
    Router, ServiceExt,
};

/// Serve `router` on `listener`, with a trailing slash on an API path trimmed first.
///
/// The trimming wraps the whole router rather than going in through `Router::layer`, which runs
/// after the route has been chosen -- too late to change which one it is.
pub async fn serve(
    listener: tokio::net::TcpListener,
    router: Router,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let app = tower::Layer::layer(&axum::middleware::from_fn(trim), router);
    axum::serve(listener, ServiceExt::<Request>::into_make_service(app))
        .with_graceful_shutdown(shutdown)
        .await
}

async fn trim(mut request: Request, next: Next) -> Response {
    if let Some(uri) = trimmed(request.uri()) {
        *request.uri_mut() = uri;
    }
    next.run(request).await
}

/// The same URI without the trailing slashes on its path, or `None` to leave it alone.
fn trimmed(uri: &Uri) -> Option<Uri> {
    let path = uri.path();
    if path.len() <= 1 || !path.ends_with('/') || is_web_app(path) {
        return None;
    }
    let path = path.trim_end_matches('/');
    let path = if path.is_empty() { "/" } else { path };
    let path_and_query = match uri.query() {
        Some(query) => format!("{path}?{query}"),
        None => path.to_string(),
    };
    let mut parts = uri.clone().into_parts();
    parts.path_and_query = Some(PathAndQuery::try_from(path_and_query).ok()?);
    Uri::from_parts(parts).ok()
}

fn is_web_app(path: &str) -> bool {
    ["/dashboard", "/api/docs"].iter().any(|prefix| {
        path.strip_prefix(prefix)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
    })
}

#[cfg(test)]
mod tests {
    use super::trimmed;

    fn t(uri: &str) -> Option<String> {
        trimmed(&uri.parse().unwrap()).map(|u| u.to_string())
    }

    #[test]
    fn an_api_path_loses_its_trailing_slash_and_keeps_its_query() {
        assert_eq!(t("/api/v1/tables/").as_deref(), Some("/api/v1/tables"));
        assert_eq!(
            t("/api/v1/memories/?user_id=u1&limit=5").as_deref(),
            Some("/api/v1/memories?user_id=u1&limit=5")
        );
        assert_eq!(t("/live//").as_deref(), Some("/live"));
    }

    #[test]
    fn the_root_a_bare_path_and_the_web_apps_are_left_alone() {
        assert_eq!(t("/"), None);
        assert_eq!(t("/api/v1/tables"), None);
        assert_eq!(t("/dashboard/"), None);
        assert_eq!(t("/dashboard/tables/docs/"), None);
        assert_eq!(t("/api/docs/"), None);
    }
}
