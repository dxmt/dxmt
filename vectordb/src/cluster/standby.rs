//! A standby node: `serve --standby --advertise <url>` serves no shard until one is lost, then
//! becomes that shard's primary.
//!
//! A shard's data is in the bucket, so a shard whose primary and replicas are all gone is not
//! lost -- it only needs a process. A standby watches every shard's lease; when one has run out
//! by the grace and `DXMT_STANDBY_DELAY_SECS` more (so that a replica, which has the shard's
//! data already, goes first), it takes the lease and starts as that shard's primary, recovering
//! from the bucket like any new machine. One standby per shard lost; run a few for a cluster
//! that should heal by itself.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::{lease, ClusterStore};
use crate::storage::object::types::ListObjectsOptions;

/// How much longer than a replica a standby waits: `DXMT_STANDBY_DELAY_SECS` (default 5).
pub fn delay() -> chrono::Duration {
    chrono::Duration::seconds(
        crate::config::env::var("DXMT_STANDBY_DELAY_SECS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(5),
    )
}

/// Every shard that has ever had a lease.
async fn shards(cluster: &ClusterStore) -> Vec<String> {
    let Ok(listed) = cluster
        .store()
        .list_objects(&ListObjectsOptions {
            prefix: Some("cluster/shards/".to_string()),
            max_keys: Some(10_000),
            continuation_token: None,
            delimiter: None,
        })
        .await
    else {
        return Vec::new();
    };
    let mut ids: Vec<String> = listed
        .objects
        .iter()
        .filter_map(|o| {
            o.key
                .strip_prefix("cluster/shards/")
                .and_then(|rest| rest.strip_suffix("/LEASE"))
                .map(str::to_string)
        })
        .collect();
    ids.sort();
    ids.dedup();
    ids
}

/// Wait until a shard is lost, take its lease, and return the shard with the lease held.
pub async fn adopt(cluster: &ClusterStore, node: &str) -> crate::Result<(String, lease::Held)> {
    let every = super::nodes::heartbeat_every();
    tracing::info!("standby node '{node}': waiting for a shard to lose all its nodes");
    loop {
        for shard in shards(cluster).await {
            let Ok(Some((current, _))) = lease::read(cluster, &shard).await else {
                continue;
            };
            if super::now() <= current.expires_at + lease::grace() + delay() {
                continue;
            }
            if let Ok(Some(held)) = lease::acquire(cluster, &shard, node, false).await {
                tracing::warn!(
                    "standby node '{node}' is taking over shard '{shard}': its lease (held by \
                     '{}') ran out and no replica took it",
                    current.node
                );
                return Ok((shard, held));
            }
        }
        tokio::time::sleep(every).await;
    }
}

/// Keep renewing a lease until `stop` is set: while the adopted shard is recovering, before the
/// node's own membership loop takes the lease over. Recovery can take longer than a term.
pub fn keep_renewing(
    cluster: Arc<ClusterStore>,
    shard: String,
    held: lease::Held,
) -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    tokio::spawn(async move {
        let mut held = held;
        let mut ticker = tokio::time::interval(super::nodes::heartbeat_every());
        loop {
            ticker.tick().await;
            if flag.load(Ordering::SeqCst) {
                break;
            }
            match lease::renew(&cluster, &shard, &held).await {
                Ok(Some(h)) => held = h,
                Ok(None) => break, // someone else's now; the membership loop will find out
                Err(e) => tracing::warn!("renewing the adopted shard's lease: {e}"),
            }
        }
    });
    stop
}
