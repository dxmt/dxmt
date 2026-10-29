//! A server as a member of a cluster: `serve --shard <id> --advertise <url>`.
//!
//! It is an ordinary server whose object storage is under `shards/<id>/`; being a member adds
//! two things. It says who it is every few seconds (`cluster/nodes/<node>`), so routers know
//! where the shard's primary and replicas are. And it refuses the requests that change where
//! tables live -- creating, dropping or importing a table -- unless they come through a
//! router, because a table created directly on a node is one no router knows about.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use super::nodes::{self, NodeRecord, Role};
use super::ClusterStore;
use crate::storage::IntelligentIndexedStorage;

/// This server's membership, from the environment `serve --shard` sets.
#[derive(Debug, Clone)]
pub struct Membership {
    pub shard: String,
    pub url: String,
    pub node: String,
}

impl Membership {
    /// `None` outside a cluster (no `DXMT_SHARD`).
    pub fn from_env() -> crate::Result<Option<Self>> {
        let Some(shard) = crate::config::env::var("DXMT_SHARD")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        else {
            return Ok(None);
        };
        if !shard
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(crate::Error::InvalidConfig(format!(
                "shard id '{shard}': letters, digits, '-' and '_' only -- it is a path in the \
                 bucket"
            )));
        }
        let url = crate::config::env::var("DXMT_ADVERTISE_URL")
            .ok()
            .map(|u| u.trim().trim_end_matches('/').to_string())
            .filter(|u| !u.is_empty())
            .ok_or_else(|| {
                crate::Error::InvalidConfig(
                    "a cluster node needs --advertise (DXMT_ADVERTISE_URL): the address routers \
                     reach it at, e.g. http://10.0.0.5:9999"
                        .to_string(),
                )
            })?;
        let node = crate::config::env::var("DXMT_NODE_ID")
            .ok()
            .filter(|n| !n.trim().is_empty())
            .unwrap_or_else(|| nodes::node_id_for(&url));
        Ok(Some(Self { shard, url, node }))
    }

    /// The prefix a shard's store lives under.
    pub fn prefix(&self) -> String {
        format!("shards/{}", self.shard)
    }
}

/// Write this node's record now and every `DXMT_HEARTBEAT_SECS`, for as long as the process
/// lives. A node that cannot reach the bucket stops being written, and routers drop it once
/// the record is older than `DXMT_NODE_TTL_SECS`.
pub fn spawn_heartbeat(
    cluster: Arc<ClusterStore>,
    storage: Arc<IntelligentIndexedStorage>,
    membership: Membership,
) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(nodes::heartbeat_every());
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let ready = storage.health_snapshot().await.ready;
            let record = NodeRecord {
                node: membership.node.clone(),
                shard: membership.shard.clone(),
                role: if storage.is_replica() {
                    Role::Replica
                } else {
                    Role::Primary
                },
                url: membership.url.clone(),
                epoch: storage.primary_epoch(),
                ready,
                heartbeat_at: super::now(),
            };
            if let Err(e) = nodes::write(&cluster, &record).await {
                tracing::warn!("{e}");
            }
        }
    });
}

/// Whether a request changes which tables exist: create, drop, import.
fn changes_placement(method: &Method, path: &str) -> bool {
    let path = path.trim_end_matches('/');
    match *method {
        Method::POST => path == "/api/v1/tables" || path == "/api/v1/tables/import",
        Method::DELETE => path
            .strip_prefix("/api/v1/tables/")
            .is_some_and(|rest| !rest.is_empty() && !rest.contains('/')),
        _ => false,
    }
}

/// Middleware for a cluster node: requests that change placement must carry the router header.
pub async fn require_router_for_placement(request: Request<Body>, next: Next<Body>) -> Response {
    if !changes_placement(request.method(), request.uri().path()) {
        return next.run(request).await;
    }
    let presented = request
        .headers()
        .get(super::ROUTER_HEADER)
        .and_then(|v| v.to_str().ok());
    if presented == Some(super::router_secret().as_str()) {
        return next.run(request).await;
    }
    let body = serde_json::json!({
        "success": false,
        "data": null,
        "error": {
            "code": "USE_THE_ROUTER",
            "message": "this server is a cluster node: tables are created, dropped and \
                        imported through a router (`vectordb router`), which records which \
                        shard holds them",
            "details": null,
        },
    });
    (StatusCode::FORBIDDEN, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn which_requests_change_placement() {
        assert!(changes_placement(&Method::POST, "/api/v1/tables"));
        assert!(changes_placement(&Method::POST, "/api/v1/tables/import"));
        assert!(changes_placement(&Method::DELETE, "/api/v1/tables/t"));
        assert!(!changes_placement(
            &Method::DELETE,
            "/api/v1/tables/t/records"
        ));
        assert!(!changes_placement(
            &Method::POST,
            "/api/v1/tables/t/records"
        ));
        assert!(!changes_placement(&Method::GET, "/api/v1/tables"));
    }
}
