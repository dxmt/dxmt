//! Which shard holds which table: `cluster/PLACEMENT`, one JSON object, changed only by
//! conditional writes ([`update`]), so two routers creating tables at once each see the
//! other's entry rather than overwriting it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::ClusterStore;
use crate::storage::object::errors::ObjectStorageError;
use crate::storage::object::traits::PutCondition;

pub const KEY: &str = "cluster/PLACEMENT";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// A router has claimed the name and is creating the table on its shard. An entry left in
    /// this state for longer than [`ABANDONED_AFTER`] belongs to a router that did not finish,
    /// and the name may be claimed again.
    Creating,
    Ready,
}

/// How long a `Creating` entry is honoured before it counts as abandoned.
pub const ABANDONED_AFTER: chrono::Duration = chrono::Duration::seconds(60);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Placed {
    pub shard: String,
    pub state: State,
    pub since: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    #[serde(default)]
    pub tables: BTreeMap<String, Placed>,
}

impl Placement {
    pub fn shard_of(&self, table: &str) -> Option<&str> {
        self.tables.get(table).map(|p| p.shard.as_str())
    }

    /// Tables per shard, counting every shard in `shards` even when it holds none.
    pub fn load_by_shard<'a>(
        &self,
        shards: impl IntoIterator<Item = &'a str>,
    ) -> BTreeMap<String, usize> {
        let mut load: BTreeMap<String, usize> =
            shards.into_iter().map(|s| (s.to_string(), 0)).collect();
        for placed in self.tables.values() {
            if let Some(n) = load.get_mut(&placed.shard) {
                *n += 1;
            }
        }
        load
    }
}

/// The placement and its version; an empty one, with no version, when there is none yet.
pub async fn load(cluster: &ClusterStore) -> crate::Result<(Placement, Option<String>)> {
    match cluster.store().get_object_versioned(KEY).await {
        Ok((bytes, version)) => {
            let placement = serde_json::from_slice(&bytes)
                .map_err(|e| crate::Error::Internal(format!("{KEY} is not a placement: {e}")))?;
            Ok((placement, version))
        }
        Err(ObjectStorageError::ObjectNotFound { .. }) => Ok((Placement::default(), None)),
        Err(e) => Err(crate::Error::Internal(format!("reading {KEY}: {e}"))),
    }
}

/// Change the placement: read it, apply `change`, write it back on the condition that nobody
/// wrote it since, and retry on a conflict. `change` may refuse (an `Err`), which writes
/// nothing; it may run more than once, so it must not have effects of its own.
pub async fn update<T>(
    cluster: &ClusterStore,
    mut change: impl FnMut(&mut Placement) -> crate::Result<T>,
) -> crate::Result<T> {
    for attempt in 0..16u32 {
        let (mut placement, version) = load(cluster).await?;
        let result = change(&mut placement)?;
        let bytes = bytes::Bytes::from(
            serde_json::to_vec_pretty(&placement)
                .map_err(|e| crate::Error::Serialization(e.to_string()))?,
        );
        let condition = match version {
            Some(v) => PutCondition::IfMatch(v),
            None => PutCondition::IfAbsent,
        };
        match cluster
            .store()
            .put_object_if(KEY, bytes, Some("application/json"), condition)
            .await
        {
            Ok(_) => return Ok(result),
            Err(ObjectStorageError::PreconditionFailed { .. }) => {
                // Someone else's change landed first: read theirs and apply ours to it.
                let backoff = 10u64 * (1 << attempt.min(6));
                tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
            }
            Err(e) => return Err(crate::Error::Internal(format!("writing {KEY}: {e}"))),
        }
    }
    Err(crate::Error::ServiceUnavailable {
        message: format!("{KEY} kept changing under this router; try again"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::object::filesystem::FilesystemStorage;
    use std::sync::Arc;

    fn cluster(dir: &std::path::Path) -> ClusterStore {
        ClusterStore::with_store(Box::new(FilesystemStorage::new(dir, "bucket").unwrap()))
    }

    #[tokio::test]
    async fn concurrent_updates_all_land() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(cluster(dir.path()));
        let mut tasks = Vec::new();
        for i in 0..12 {
            let store = Arc::clone(&store);
            tasks.push(tokio::spawn(async move {
                update(&store, |p| {
                    p.tables.insert(
                        format!("t{i}"),
                        Placed {
                            shard: format!("s{}", i % 3),
                            state: State::Ready,
                            since: super::super::now(),
                        },
                    );
                    Ok(())
                })
                .await
            }));
        }
        for t in tasks {
            t.await.unwrap().unwrap();
        }
        let (placement, _) = load(&store).await.unwrap();
        assert_eq!(placement.tables.len(), 12, "a concurrent update was lost");
        let load = placement.load_by_shard(["s0", "s1", "s2", "s3"]);
        assert_eq!(load["s0"], 4);
        assert_eq!(load["s3"], 0);
    }

    #[tokio::test]
    async fn a_refused_change_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let store = cluster(dir.path());
        let err = update(&store, |_| -> crate::Result<()> {
            Err(crate::Error::InvalidInput("no".into()))
        })
        .await;
        assert!(err.is_err());
        assert_eq!(load(&store).await.unwrap().1, None);
    }
}
