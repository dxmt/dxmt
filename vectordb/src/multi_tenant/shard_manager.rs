use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    pub shard_id: Uuid,
    pub tablename: String,
    pub key_range: KeyRange,
    pub replicas: Vec<ShardReplica>,
    pub status: ShardStatus,
    pub size_bytes: u64,
    pub record_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRange {
    pub start: Vec<u8>,
    pub end: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardReplica {
    pub nodeid: Uuid,
    pub role: ReplicaRole,
    pub sync_status: SyncStatus,
    pub lag_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicaRole {
    Primary,
    Secondary,
    ReadOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncStatus {
    InSync,
    Syncing,
    Behind,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShardStatus {
    Active,
    Splitting,
    Merging,
    Moving,
    Inactive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMap {
    pub version: u64,
    pub mappings: HashMap<String, Vec<ShardMapping>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMapping {
    pub shard_id: Uuid,
    pub key_range: KeyRange,
    pub primary_node: Uuid,
    pub replica_nodes: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceOperation {
    pub operation_type: RebalanceType,
    pub shard_id: Uuid,
    pub source_node: Uuid,
    pub target_node: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RebalanceType {
    Move,
    Copy,
    Delete,
}

pub struct ShardManager {
    shards: Arc<RwLock<HashMap<Uuid, Shard>>>,
    shard_map: Arc<RwLock<ShardMap>>,
}

impl ShardManager {
    pub fn new() -> Self {
        Self {
            shards: Arc::new(RwLock::new(HashMap::new())),
            shard_map: Arc::new(RwLock::new(ShardMap {
                version: 0,
                mappings: HashMap::new(),
            })),
        }
    }

    pub async fn create_shard(&self, shard: Shard) -> crate::Result<Uuid> {
        let mut shards = self.shards.write().await;
        let shard_id = shard.shard_id;
        shards.insert(shard_id, shard);
        Ok(shard_id)
    }

    pub async fn get_shard_for_key(&self, tablename: &str, key: &[u8]) -> crate::Result<Uuid> {
        let shard_map = self.shard_map.read().await;
        if let Some(mappings) = shard_map.mappings.get(tablename) {
            for mapping in mappings {
                if key >= mapping.key_range.start.as_slice()
                    && key < mapping.key_range.end.as_slice()
                {
                    return Ok(mapping.shard_id);
                }
            }
        }
        Err(crate::error::Error::NotFound(format!(
            "No shard for key in table '{}'",
            tablename
        )))
    }

    pub async fn rebalance_shards(&self) -> crate::Result<Vec<RebalanceOperation>> {
        // Stub: no rebalancing logic yet
        Ok(vec![])
    }
}

impl Default for ShardManager {
    fn default() -> Self {
        Self::new()
    }
}
