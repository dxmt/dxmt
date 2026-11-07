//! One URL per API route, with or without a trailing slash.
//!
//! The router matches paths exactly, so `/api/v1/tables/docs/` answered an empty 404 while
//! `/api/v1/tables/docs` answered the table -- a hand-written curl or a client that joins
//! paths with a trailing `/` got nothing and no reason. This trims the trailing slash before
//! routing, so both reach the same handler.
//!
//! Repeated slashes are collapsed too (`/api//v1/tables`), the usual result of joining a base
//! URL that ends in `/` with a path that starts with one.
//!
//! Not under `/dashboard`: the console loads its assets by relative path, which needs the
//! slash, and `/dashboard` redirects to `/dashboard/` on purpose.

use axum::{
    extract::Request,
    http::{uri::PathAndQuery, Uri},
    middleware::Next,
    response::Response,
    Router, ServiceExt,
};

/// Serve `router` on `listener`, with the request path normalised first.
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

/// The same URI with its path normalised -- repeated slashes collapsed, the trailing one
/// dropped -- or `None` when there is nothing to change.
fn trimmed(uri: &Uri) -> Option<Uri> {
    let path = uri.path();
    let needs_it = (path.len() > 1 && path.ends_with('/')) || path.contains("//");
    if !needs_it || is_dashboard(path) {
        return None;
    }
    let mut normal = String::with_capacity(path.len());
    for segment in path.split('/').filter(|s| !s.is_empty()) {
        normal.push('/');
        normal.push_str(segment);
    }
    if normal.is_empty() {
        normal.push('/');
    }
    if let Some(query) = uri.query() {
        normal.push('?');
        normal.push_str(query);
    }
    let mut parts = uri.clone().into_parts();
    parts.path_and_query = Some(PathAndQuery::try_from(normal).ok()?);
    Uri::from_parts(parts).ok()
}

/// The console loads its assets by relative path, so `/dashboard/` has to keep its slash --
/// and `/dashboard` redirects there on purpose; trimming it back would loop.
fn is_dashboard(path: &str) -> bool {
    path.strip_prefix("/dashboard")
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
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
        assert_eq!(t("/api/docs/").as_deref(), Some("/api/docs"));
    }

    #[test]
    fn repeated_slashes_collapse() {
        assert_eq!(t("/api//v1/tables").as_deref(), Some("/api/v1/tables"));
        assert_eq!(
            t("//api/v1//tables/docs/?limit=1").as_deref(),
            Some("/api/v1/tables/docs?limit=1")
        );
        assert_eq!(t("//").as_deref(), Some("/"));
    }

    #[test]
    fn the_root_a_bare_path_and_the_web_apps_are_left_alone() {
        assert_eq!(t("/"), None);
        assert_eq!(t("/api/v1/tables"), None);
        assert_eq!(t("/dashboard/"), None);
        assert_eq!(t("/dashboard/tables/docs/"), None);
    }
}
