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

use super::lease;
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

/// Whether a replica takes its shard over when the primary's lease runs out:
/// `DXMT_AUTO_FAILOVER`, on unless `0`/`false`.
fn auto_failover() -> bool {
    !crate::config::env::var("DXMT_AUTO_FAILOVER")
        .is_ok_and(|v| matches!(v.trim(), "0" | "false" | "no" | "off"))
}

/// Run this node's part in the cluster, now and every `DXMT_HEARTBEAT_SECS`, for as long as the
/// process lives:
///
/// - **A primary** takes its shard's lease at once (it was started as the primary: that is the
///   operator's choice) and renews it each tick. Writes are allowed only until the end of the
///   term it last secured, counted from before the renewal was sent; a renewal refused because
///   someone else holds the lease stops it for good (`step_down`).
/// - **A replica**, with `DXMT_AUTO_FAILOVER` on, watches the lease. Once the term in it ended
///   more than the grace ago -- the primary has stopped renewing -- it tries to take it; the
///   one replica that does promotes itself (`IntelligentIndexedStorage::promote`), starts the
///   primary's timers through `on_promote`, and carries on as the primary. One that fails to
///   promote gives the lease straight back.
/// - **Both** write the node record routers read (`cluster/nodes/<node>`).
///
/// In a cluster a write is acknowledged only once the bucket has it: failover recovers from
/// the bucket, and a write only this machine had would be lost with it.
pub fn spawn_membership(
    cluster: Arc<ClusterStore>,
    storage: Arc<IntelligentIndexedStorage>,
    membership: Membership,
    on_promote: fn(Arc<IntelligentIndexedStorage>),
) {
    storage.set_ack_after_mirror(true);
    if !storage.is_replica() {
        // No writes until the lease is ours: the first tick takes it.
        storage.hold_write_lease_until(Some(std::time::Instant::now()));
    }
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(nodes::heartbeat_every());
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut held: Option<lease::Held> = None;
        let mut first = true;
        let term = lease::term()
            .to_std()
            .unwrap_or(std::time::Duration::from_secs(10));
        loop {
            ticker.tick().await;
            if storage.is_replica() {
                if auto_failover() && storage.health_snapshot().await.ready {
                    held = try_take_over(&cluster, &storage, &membership, on_promote).await;
                }
            } else {
                let sent = std::time::Instant::now();
                let attempt = match &held {
                    Some(h) => lease::renew(&cluster, &membership.shard, h).await,
                    // Started as the primary, or promoted without a lease in hand: take it.
                    None => {
                        lease::acquire(&cluster, &membership.shard, &membership.node, first).await
                    }
                };
                match attempt {
                    Ok(Some(h)) => {
                        storage.hold_write_lease_until(Some(sent + term));
                        held = Some(h);
                    }
                    Ok(None) => {
                        if held.take().is_some() || !first {
                            storage.step_down(&format!(
                                "another node took shard '{}''s lease",
                                membership.shard
                            ));
                        }
                    }
                    // The bucket could not be reached: the term runs out on its own.
                    Err(e) => tracing::warn!("lease on shard '{}': {e}", membership.shard),
                }
            }
            first = false;
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
                ready: storage.health_snapshot().await.ready,
                heartbeat_at: super::now(),
            };
            if let Err(e) = nodes::write(&cluster, &record).await {
                tracing::warn!("{e}");
            }
        }
    });
}

/// A replica's turn: if the shard's lease has run out, take it and promote. The lease held on
/// success; `None` otherwise.
async fn try_take_over(
    cluster: &ClusterStore,
    storage: &Arc<IntelligentIndexedStorage>,
    membership: &Membership,
    on_promote: fn(Arc<IntelligentIndexedStorage>),
) -> Option<lease::Held> {
    // Only an expired lease: a shard with none has never had a cluster primary (or is being
    // upgraded from before leases), and a replica is no judge of whether its primary is alive.
    let (current, _) = lease::read(cluster, &membership.shard).await.ok()??;
    if super::now() <= current.expires_at + lease::grace() {
        return None;
    }
    // Before the write that takes the lease: the term it records starts no later than this.
    let sent = std::time::Instant::now();
    let held = lease::acquire(cluster, &membership.shard, &membership.node, false)
        .await
        .ok()??;
    tracing::warn!(
        "shard '{}': the primary ({}) let its lease run out; this replica is taking over",
        membership.shard,
        current.node
    );
    match storage.promote().await {
        Ok(()) => {
            let term = lease::term()
                .to_std()
                .unwrap_or(std::time::Duration::from_secs(10));
            // Counted from before the promotion, which took part of the term.
            storage.hold_write_lease_until(Some(sent + term));
            on_promote(Arc::clone(storage));
            Some(held)
        }
        Err(e) => {
            tracing::warn!(
                "shard '{}': promotion failed ({e}); giving the lease back",
                membership.shard
            );
            lease::release(cluster, &membership.shard, &held).await;
            None
        }
    }
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
