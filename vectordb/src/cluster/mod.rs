//! A cluster of DXMT servers sharing one object storage bucket. See `design/DISTRIBUTED.md`.
//!
//! - A **shard** is one ordinary store -- its own WAL, snapshots and manifest lock, a primary
//!   and any number of read replicas -- under `shards/<id>/` in the bucket
//!   (`DXMT_OBJECT_STORAGE_PREFIX`).
//! - Every table belongs to one shard; which one is `cluster/PLACEMENT` ([`placement`]).
//! - Every node says who it is, every few seconds, in `cluster/nodes/<node>` ([`nodes`]).
//! - A **router** ([`router`]) is the one entry point: it serves `/api/v1` and forwards each
//!   request to the shard that holds its table.
//!
//! There is no consensus service. The cluster's own state lives in the bucket and every change
//! to it is a conditional write, so a cluster needs a store that honours them (S3, MinIO, the
//! filesystem provider) -- checked when a node or router starts ([`ClusterStore::open`]).

pub mod lease;
pub mod node;
pub mod nodes;
pub mod placement;
pub mod router;
pub mod transfer;

use crate::storage::object::errors::ObjectStorageError;
use crate::storage::object::traits::{ObjectStorage, PutCondition};

/// The header a router puts on everything it forwards, and that a node in a cluster requires on
/// the requests that change placement: creating and dropping a table. Its value is
/// `DXMT_CLUSTER_SECRET` when one is set.
pub const ROUTER_HEADER: &str = "x-dxmt-router";

/// The bucket's root, where the cluster's own state lives (`cluster/...`), outside every
/// shard's prefix.
pub struct ClusterStore {
    store: Box<dyn ObjectStorage + Send + Sync>,
}

impl ClusterStore {
    /// The bucket the configuration names, at its root -- refused unless it honours
    /// conditional writes, without which two routers placing tables at once would overwrite
    /// each other's placement unseen.
    pub async fn open(config: &crate::config::StorageConfig) -> crate::Result<Self> {
        if config.object_storage.is_none() {
            return Err(crate::Error::InvalidConfig(
                "a cluster lives in object storage: set DXMT_OBJECT_STORAGE_PROVIDER and its \
                 bucket"
                    .to_string(),
            ));
        }
        let store = crate::storage::object::tiered::object_storage_for(config, None)
            .await
            .map_err(|e| crate::Error::InvalidConfig(format!("cluster object storage: {e}")))?;
        let cluster = Self { store };
        if !cluster.honours_conditional_writes().await {
            return Err(crate::Error::InvalidConfig(
                "this object store does not honour conditional writes (GCS's S3 API does not), \
                 and a cluster keeps its placement with them; use S3, MinIO or the filesystem \
                 provider"
                    .to_string(),
            ));
        }
        Ok(cluster)
    }

    /// For tests: a store already built.
    pub fn with_store(store: Box<dyn ObjectStorage + Send + Sync>) -> Self {
        Self { store }
    }

    async fn honours_conditional_writes(&self) -> bool {
        const PROBE: &str = "cluster/CONDITIONAL-WRITE-PROBE";
        let body = || bytes::Bytes::from_static(b"dxmt cluster probe");
        match self
            .store
            .put_object_if(PROBE, body(), None, PutCondition::IfAbsent)
            .await
        {
            Ok(_) | Err(ObjectStorageError::PreconditionFailed { .. }) => {}
            Err(_) => return false,
        }
        matches!(
            self.store
                .put_object_if(PROBE, body(), None, PutCondition::IfMatch("1".to_string()))
                .await,
            Err(ObjectStorageError::PreconditionFailed { .. })
        )
    }

    pub(crate) fn store(&self) -> &(dyn ObjectStorage + Send + Sync) {
        self.store.as_ref()
    }
}

/// `DXMT_CLUSTER_SECRET`, or `"1"` when unset: the router header's expected value.
pub fn router_secret() -> String {
    crate::config::env::var("DXMT_CLUSTER_SECRET")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "1".to_string())
}

/// A path segment as the table name it encodes (`my%2Dtable` is `my-table`).
pub(crate) fn percent_decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Some(b) = segment
                .get(i + 1..i + 3)
                .and_then(|hex| u8::from_str_radix(hex, 16).ok())
            {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| segment.to_string())
}

fn now() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now()
}
