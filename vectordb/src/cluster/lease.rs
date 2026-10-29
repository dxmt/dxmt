//! Which node is a shard's primary: `cluster/shards/<shard>/LEASE`, held for a term and renewed
//! by its holder, taken by conditional write.
//!
//! The lease is what makes failover automatic and safe at once:
//!
//! - **One holder.** Taking it is a write conditional on the version read, so of two replicas
//!   that see it expire, one wins and the other's write is refused.
//! - **The holder stops first.** A primary renews every heartbeat and refuses writes once its
//!   own term is over (`IntelligentIndexedStorage::hold_write_lease_until`), measured on its own
//!   clock from *before* the renewal was sent. A replica takes the lease only once the term
//!   recorded in it is over by [`grace`] more, on its clock. So a primary cut off from the bucket
//!   has stopped writing before anyone may start, as long as the clocks agree to within the
//!   grace.
//! - **A takeover by hand still works.** A primary started explicitly (`serve --shard` without
//!   `--replica`) takes the lease whatever its state: the operator chose it.

use serde::{Deserialize, Serialize};

use super::ClusterStore;
use crate::storage::object::errors::ObjectStorageError;
use crate::storage::object::traits::PutCondition;

pub fn key(shard: &str) -> String {
    format!("cluster/shards/{shard}/LEASE")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lease {
    pub node: String,
    /// Counts the times the lease changed hands.
    pub term: u64,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

/// A lease this node holds, with the version to renew it against.
#[derive(Debug, Clone)]
pub struct Held {
    pub lease: Lease,
    version: String,
}

/// How long a term lasts: `DXMT_NODE_TTL_SECS` (default 10).
pub fn term() -> chrono::Duration {
    super::nodes::ttl()
}

/// How far past a term's end a replica waits before taking the lease:
/// `DXMT_LEASE_GRACE_SECS` (default 3). The clock skew the cluster tolerates.
pub fn grace() -> chrono::Duration {
    chrono::Duration::seconds(
        crate::config::env::var("DXMT_LEASE_GRACE_SECS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(3),
    )
}

pub async fn read(cluster: &ClusterStore, shard: &str) -> crate::Result<Option<(Lease, String)>> {
    match cluster.store().get_object_versioned(&key(shard)).await {
        Ok((bytes, Some(version))) => {
            let lease = serde_json::from_slice(&bytes).map_err(|e| {
                crate::Error::Internal(format!("{} is not a lease: {e}", key(shard)))
            })?;
            Ok(Some((lease, version)))
        }
        Ok((_, None)) => Err(crate::Error::Internal(
            "the object store gave no version for a lease".to_string(),
        )),
        Err(ObjectStorageError::ObjectNotFound { .. }) => Ok(None),
        Err(e) => Err(crate::Error::Internal(format!(
            "reading {}: {e}",
            key(shard)
        ))),
    }
}

async fn write(
    cluster: &ClusterStore,
    shard: &str,
    lease: &Lease,
    condition: PutCondition,
) -> crate::Result<Option<String>> {
    let bytes = bytes::Bytes::from(
        serde_json::to_vec(lease).map_err(|e| crate::Error::Serialization(e.to_string()))?,
    );
    match cluster
        .store()
        .put_object_if(&key(shard), bytes, Some("application/json"), condition)
        .await
    {
        Ok(version) => Ok(Some(version)),
        Err(ObjectStorageError::PreconditionFailed { .. }) => Ok(None),
        Err(e) => Err(crate::Error::Internal(format!(
            "writing {}: {e}",
            key(shard)
        ))),
    }
}

/// Take the lease for `node`: always when `force`, otherwise only if nobody holds it, this node
/// already does, or the holder's term ended more than [`grace`] ago. `Ok(None)` when someone
/// else holds it, or won the race to take it.
pub async fn acquire(
    cluster: &ClusterStore,
    shard: &str,
    node: &str,
    force: bool,
) -> crate::Result<Option<Held>> {
    let now = super::now();
    let current = read(cluster, shard).await?;
    let (term, condition) = match &current {
        None => (1, PutCondition::IfAbsent),
        Some((lease, version)) => {
            let ours = lease.node == node;
            let over = now > lease.expires_at + grace();
            if !(force || ours || over) {
                return Ok(None);
            }
            (
                if ours { lease.term } else { lease.term + 1 },
                PutCondition::IfMatch(version.clone()),
            )
        }
    };
    let lease = Lease {
        node: node.to_string(),
        term,
        expires_at: now + self::term(),
    };
    Ok(write(cluster, shard, &lease, condition)
        .await?
        .map(|version| Held { lease, version }))
}

/// Extend a held lease by a term. `Ok(None)` when it is no longer this node's: someone took it.
pub async fn renew(
    cluster: &ClusterStore,
    shard: &str,
    held: &Held,
) -> crate::Result<Option<Held>> {
    let lease = Lease {
        expires_at: super::now() + term(),
        ..held.lease.clone()
    };
    Ok(write(
        cluster,
        shard,
        &lease,
        PutCondition::IfMatch(held.version.clone()),
    )
    .await?
    .map(|version| Held { lease, version }))
}

/// Give a lease up now, so a waiting replica need not sit out the rest of the term.
pub async fn release(cluster: &ClusterStore, shard: &str, held: &Held) {
    let lease = Lease {
        expires_at: super::now() - grace() - chrono::Duration::seconds(1),
        ..held.lease.clone()
    };
    let _ = write(
        cluster,
        shard,
        &lease,
        PutCondition::IfMatch(held.version.clone()),
    )
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::object::filesystem::FilesystemStorage;

    fn cluster(dir: &std::path::Path) -> ClusterStore {
        ClusterStore::with_store(Box::new(FilesystemStorage::new(dir, "bucket").unwrap()))
    }

    #[tokio::test]
    async fn one_holder_renewal_and_takeover() {
        let dir = tempfile::tempdir().unwrap();
        let c = cluster(dir.path());
        let held = acquire(&c, "a", "n1", false).await.unwrap().expect("free");
        assert!(
            acquire(&c, "a", "n2", false).await.unwrap().is_none(),
            "held and live"
        );
        let held = renew(&c, "a", &held).await.unwrap().expect("still ours");

        // An operator's primary takes it anyway, and the old holder's renewal then fails.
        let forced = acquire(&c, "a", "n2", true).await.unwrap().expect("forced");
        assert_eq!(forced.lease.term, held.lease.term + 1);
        assert!(renew(&c, "a", &held).await.unwrap().is_none(), "lost");

        // Released, it is free at once.
        release(&c, "a", &forced).await;
        assert!(acquire(&c, "a", "n3", false).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn of_two_racing_for_an_expired_lease_one_wins() {
        let dir = tempfile::tempdir().unwrap();
        let c = std::sync::Arc::new(cluster(dir.path()));
        let held = acquire(&c, "a", "old", false).await.unwrap().unwrap();
        release(&c, "a", &held).await;
        let (x, y) = tokio::join!(acquire(&c, "a", "r1", false), acquire(&c, "a", "r2", false));
        let winners = [x.unwrap().is_some(), y.unwrap().is_some()];
        assert_eq!(winners.iter().filter(|w| **w).count(), 1, "{winners:?}");
    }
}
