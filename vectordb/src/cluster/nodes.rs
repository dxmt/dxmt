//! Who is serving what: `cluster/nodes/<node>`, written by each node every
//! `DXMT_HEARTBEAT_SECS` and read by routers to find each shard's primary and replicas.
//!
//! One object per node rather than one shared list, so heartbeats never contend. A record whose
//! heartbeat is older than `DXMT_NODE_TTL_SECS` is ignored: the node is gone, or cannot reach
//! the bucket, and either way should not be sent requests.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::ClusterStore;
use crate::storage::object::types::ListObjectsOptions;

pub const PREFIX: &str = "cluster/nodes/";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Primary,
    Replica,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeRecord {
    pub node: String,
    pub shard: String,
    pub role: Role,
    /// Where routers reach it: `http://host:port`.
    pub url: String,
    /// A primary's epoch (`wal/PRIMARY`); when two records claim one shard, the higher wins.
    pub epoch: u64,
    /// The node's own readiness: false while recovering, once fenced, once durability failed.
    pub ready: bool,
    pub heartbeat_at: chrono::DateTime<chrono::Utc>,
}

pub fn heartbeat_every() -> std::time::Duration {
    std::time::Duration::from_secs(
        crate::config::env::var("DXMT_HEARTBEAT_SECS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(2),
    )
}

pub fn ttl() -> chrono::Duration {
    chrono::Duration::seconds(
        crate::config::env::var("DXMT_NODE_TTL_SECS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(10),
    )
}

/// A node's id from its address, `host-port`: stable across restarts, distinct per process.
pub fn node_id_for(url: &str) -> String {
    let bare = url
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/');
    bare.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

pub async fn write(cluster: &ClusterStore, record: &NodeRecord) -> crate::Result<()> {
    let bytes = bytes::Bytes::from(
        serde_json::to_vec(record).map_err(|e| crate::Error::Serialization(e.to_string()))?,
    );
    cluster
        .store()
        .put_object(
            &format!("{PREFIX}{}", record.node),
            bytes,
            Some("application/json"),
        )
        .await
        .map_err(|e| crate::Error::Internal(format!("node heartbeat: {e}")))
}

pub async fn remove(cluster: &ClusterStore, node: &str) {
    let _ = cluster
        .store()
        .delete_object(&format!("{PREFIX}{node}"))
        .await;
}

/// Every record in the bucket, live or not.
pub async fn list(cluster: &ClusterStore) -> crate::Result<Vec<NodeRecord>> {
    let listed = cluster
        .store()
        .list_objects(&ListObjectsOptions {
            prefix: Some(PREFIX.to_string()),
            max_keys: Some(10_000),
            continuation_token: None,
            delimiter: None,
        })
        .await
        .map_err(|e| crate::Error::Internal(format!("listing nodes: {e}")))?;
    let mut records = Vec::new();
    for object in listed.objects {
        if let Ok(bytes) = cluster.store().get_object(&object.key).await {
            if let Ok(record) = serde_json::from_slice::<NodeRecord>(&bytes) {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// A shard as routers see it: the primary to write to, and the replicas that may serve reads.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ShardView {
    pub primary: Option<NodeRecord>,
    pub replicas: Vec<NodeRecord>,
}

/// The live nodes by shard. Of several live records claiming to be a shard's primary -- an old
/// one not yet expired beside the one that replaced it -- the highest epoch wins, and only a
/// ready one is used.
pub fn shards(
    records: &[NodeRecord],
    now: chrono::DateTime<chrono::Utc>,
    ttl: chrono::Duration,
) -> BTreeMap<String, ShardView> {
    let mut view: BTreeMap<String, ShardView> = BTreeMap::new();
    for record in records {
        let entry = view.entry(record.shard.clone()).or_default();
        if now - record.heartbeat_at > ttl || !record.ready {
            continue;
        }
        match record.role {
            Role::Primary => {
                if entry
                    .primary
                    .as_ref()
                    .is_none_or(|p| record.epoch > p.epoch)
                {
                    entry.primary = Some(record.clone());
                }
            }
            Role::Replica => entry.replicas.push(record.clone()),
        }
    }
    view
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        node: &str,
        shard: &str,
        role: Role,
        epoch: u64,
        age_s: i64,
        ready: bool,
    ) -> NodeRecord {
        NodeRecord {
            node: node.into(),
            shard: shard.into(),
            role,
            url: format!("http://{node}"),
            epoch,
            ready,
            heartbeat_at: chrono::Utc::now() - chrono::Duration::seconds(age_s),
        }
    }

    #[test]
    fn the_live_ready_newest_primary_wins() {
        let records = vec![
            record("old", "a", Role::Primary, 1, 1, true),
            record("new", "a", Role::Primary, 2, 1, true),
            record("fenced", "a", Role::Primary, 3, 1, false),
            record("gone", "b", Role::Primary, 9, 60, true),
            record("r1", "a", Role::Replica, 0, 1, true),
            record("r-stale", "a", Role::Replica, 0, 60, true),
        ];
        let view = shards(&records, chrono::Utc::now(), chrono::Duration::seconds(10));
        assert_eq!(view["a"].primary.as_ref().unwrap().node, "new");
        assert_eq!(view["a"].replicas.len(), 1);
        assert!(
            view["b"].primary.is_none(),
            "an expired primary is not used"
        );
    }

    #[test]
    fn node_ids_come_from_addresses() {
        assert_eq!(node_id_for("http://10.0.0.5:9999/"), "10-0-0-5-9999");
    }
}
