//! Shard routing and rebalancing for distributed storage
//!
//! Implements consistent hashing with virtual nodes for even distribution.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{Error, Result};

pub type ShardId = u32;
pub type NodeId = String;
pub type HashRing = BTreeMap<u64, NodeId>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    pub shard_count: u32,
    pub virtual_nodes_per_shard: u32,
    pub replication_factor: usize,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            shard_count: 256,
            virtual_nodes_per_shard: 100,
            replication_factor: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShardInfo {
    pub id: ShardId,
    pub primary_node: NodeId,
    pub replica_nodes: Vec<NodeId>,
    pub status: ShardStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShardStatus {
    Active,
    Migrating,
    Inactive,
}

pub struct ShardRouter {
    config: ShardConfig,
    hash_ring: Arc<RwLock<HashRing>>,
    shard_map: Arc<RwLock<HashMap<ShardId, ShardInfo>>>,
    node_weights: Arc<RwLock<HashMap<NodeId, f64>>>,
}

impl ShardRouter {
    pub fn new(shard_count: u32) -> Self {
        let config = ShardConfig {
            shard_count,
            ..Default::default()
        };

        Self {
            config,
            hash_ring: Arc::new(RwLock::new(BTreeMap::new())),
            shard_map: Arc::new(RwLock::new(HashMap::new())),
            node_weights: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a node to the cluster
    pub async fn add_node(&self, node_id: NodeId, weight: f64) -> Result<()> {
        // Scope the write locks so they are released before rebalance_shards acquires read locks.
        {
            let mut ring = self.hash_ring.write().await;
            let mut weights = self.node_weights.write().await;

            weights.insert(node_id.clone(), weight);

            let virtual_nodes = (self.config.virtual_nodes_per_shard as f64 * weight) as u32;
            for i in 0..virtual_nodes {
                let virtual_node_key = format!("{}:{}", node_id, i);
                let hash = self.hash_key(&virtual_node_key);
                ring.insert(hash, node_id.clone());
            }
        } // ring and weights write guards dropped here

        self.rebalance_shards().await?;
        Ok(())
    }

    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: &NodeId) -> Result<()> {
        {
            let mut ring = self.hash_ring.write().await;
            let mut weights = self.node_weights.write().await;

            ring.retain(|_, id| id != node_id);
            weights.remove(node_id);
        } // write guards dropped here

        self.rebalance_shards().await?;
        Ok(())
    }

    /// Get the shard ID for a given key
    pub fn get_shard_id(&self, key: &str) -> ShardId {
        let hash = self.hash_key(key);
        (hash % self.config.shard_count as u64) as ShardId
    }

    /// Get the nodes responsible for a shard
    pub async fn get_shard_nodes(&self, shard_id: ShardId) -> Result<Vec<NodeId>> {
        let shard_map = self.shard_map.read().await;

        if let Some(shard_info) = shard_map.get(&shard_id) {
            let mut nodes = vec![shard_info.primary_node.clone()];
            nodes.extend(shard_info.replica_nodes.clone());
            Ok(nodes)
        } else {
            Err(Error::InvalidInput(format!("Shard {} not found", shard_id)))
        }
    }

    /// Get the primary node for a shard
    pub async fn get_primary_node(&self, shard_id: ShardId) -> Result<NodeId> {
        let shard_map = self.shard_map.read().await;

        shard_map
            .get(&shard_id)
            .map(|info| info.primary_node.clone())
            .ok_or_else(|| Error::InvalidInput(format!("Shard {} not found", shard_id)))
    }

    /// Get replica nodes for a shard
    pub async fn get_replica_nodes(&self, shard_id: ShardId) -> Result<Vec<NodeId>> {
        let shard_map = self.shard_map.read().await;

        shard_map
            .get(&shard_id)
            .map(|info| info.replica_nodes.clone())
            .ok_or_else(|| Error::InvalidInput(format!("Shard {} not found", shard_id)))
    }

    /// Route a key to the appropriate nodes
    pub async fn route_key(&self, key: &str) -> Result<Vec<NodeId>> {
        let shard_id = self.get_shard_id(key);
        self.get_shard_nodes(shard_id).await
    }

    /// Get cluster statistics
    pub async fn get_cluster_stats(&self) -> ClusterStats {
        let shard_map = self.shard_map.read().await;
        let weights = self.node_weights.read().await;

        let mut node_shard_counts = HashMap::new();
        let mut active_shards = 0;
        let mut migrating_shards = 0;

        for shard_info in shard_map.values() {
            match shard_info.status {
                ShardStatus::Active => active_shards += 1,
                ShardStatus::Migrating => migrating_shards += 1,
                ShardStatus::Inactive => {}
            }

            *node_shard_counts
                .entry(shard_info.primary_node.clone())
                .or_insert(0) += 1;
            for replica in &shard_info.replica_nodes {
                *node_shard_counts.entry(replica.clone()).or_insert(0) += 1;
            }
        }

        ClusterStats {
            total_nodes: weights.len(),
            total_shards: self.config.shard_count,
            active_shards,
            migrating_shards,
            node_shard_distribution: node_shard_counts,
            replication_factor: self.config.replication_factor,
        }
    }

    async fn rebalance_shards(&self) -> Result<()> {
        let ring = self.hash_ring.read().await;
        let mut shard_map = self.shard_map.write().await;

        if ring.is_empty() {
            return Ok(());
        }

        // Assign each shard to nodes based on consistent hashing
        for shard_id in 0..self.config.shard_count {
            let shard_key = format!("shard:{}", shard_id);
            let shard_hash = self.hash_key(&shard_key);

            // Find the primary node (first node >= shard_hash)
            let primary_node = ring
                .range(shard_hash..)
                .next()
                .or_else(|| ring.iter().next())
                .map(|(_, node)| node.clone())
                .unwrap();

            // Find replica nodes (next N-1 nodes in the ring)
            let mut replica_nodes = Vec::new();
            let mut current_hash = shard_hash;

            for _ in 0..self.config.replication_factor.saturating_sub(1) {
                // Find next node after current_hash
                let next_node = ring
                    .range((current_hash + 1)..)
                    .next()
                    .or_else(|| ring.iter().next())
                    .map(|(hash, node)| (*hash, node.clone()));

                if let Some((next_hash, next_node)) = next_node {
                    if next_node != primary_node && !replica_nodes.contains(&next_node) {
                        replica_nodes.push(next_node);
                        current_hash = next_hash;
                    } else {
                        // Skip this node and try the next one
                        current_hash = next_hash;
                    }
                }
            }

            let shard_info = ShardInfo {
                id: shard_id,
                primary_node,
                replica_nodes,
                status: ShardStatus::Active,
            };

            shard_map.insert(shard_id, shard_info);
        }

        Ok(())
    }

    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let result = hasher.finalize();

        // Convert first 8 bytes to u64
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&result[0..8]);
        u64::from_be_bytes(bytes)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    pub total_nodes: usize,
    pub total_shards: u32,
    pub active_shards: u32,
    pub migrating_shards: u32,
    pub node_shard_distribution: HashMap<NodeId, u32>,
    pub replication_factor: usize,
}

// Cluster state management
pub struct ClusterState {
    pub discovery: Arc<crate::cluster::SimpleDiscovery>,
    pub raft: Arc<crate::consensus::RaftNode>,
    pub atomic_storage: Arc<crate::storage::atomic_ops::AtomicStorageManager>,
    pub shard_router: Arc<ShardRouter>,
}

impl ClusterState {
    pub async fn is_leader(&self) -> bool {
        self.raft.is_leader().await
    }

    pub async fn get_leader_id(&self) -> Option<NodeId> {
        self.raft.get_leader_id().await
    }

    pub async fn route_write(&self, key: &str) -> Result<Vec<NodeId>> {
        if !self.is_leader().await {
            return Err(Error::NotLeader);
        }

        self.shard_router.route_key(key).await
    }

    pub async fn route_read(&self, key: &str) -> Result<Vec<NodeId>> {
        // Reads can go to any replica
        self.shard_router.route_key(key).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_shard_router_basic() {
        let router = ShardRouter::new(16);

        // Add some nodes
        router.add_node("node1".to_string(), 1.0).await.unwrap();
        router.add_node("node2".to_string(), 1.0).await.unwrap();
        router.add_node("node3".to_string(), 1.0).await.unwrap();

        // Test key routing
        let nodes = router.route_key("test_key").await.unwrap();
        assert!(!nodes.is_empty());

        // Test shard assignment
        let shard_id = router.get_shard_id("test_key");
        assert!(shard_id < 16);

        let primary = router.get_primary_node(shard_id).await.unwrap();
        assert!(["node1", "node2", "node3"].contains(&primary.as_str()));
    }

    #[tokio::test]
    async fn test_consistent_hashing() {
        let router = ShardRouter::new(256);

        router.add_node("node1".to_string(), 1.0).await.unwrap();
        router.add_node("node2".to_string(), 1.0).await.unwrap();

        // Same key should always route to same shard
        let shard1 = router.get_shard_id("consistent_key");
        let shard2 = router.get_shard_id("consistent_key");
        assert_eq!(shard1, shard2);

        // Different keys should distribute across shards
        let mut shard_distribution = HashMap::new();
        for i in 0..1000 {
            let key = format!("key_{}", i);
            let shard = router.get_shard_id(&key);
            *shard_distribution.entry(shard).or_insert(0) += 1;
        }

        // Should have reasonable distribution (not all keys in one shard)
        assert!(shard_distribution.len() > 10);
    }

    #[tokio::test]
    async fn test_node_removal() {
        let router = ShardRouter::new(16);

        router.add_node("node1".to_string(), 1.0).await.unwrap();
        router.add_node("node2".to_string(), 1.0).await.unwrap();
        router.add_node("node3".to_string(), 1.0).await.unwrap();

        let stats_before = router.get_cluster_stats().await;
        assert_eq!(stats_before.total_nodes, 3);

        // Remove a node
        router.remove_node(&"node2".to_string()).await.unwrap();

        let stats_after = router.get_cluster_stats().await;
        assert_eq!(stats_after.total_nodes, 2);

        // Verify shards are still assigned
        for shard_id in 0..16 {
            let nodes = router.get_shard_nodes(shard_id).await.unwrap();
            assert!(!nodes.is_empty());
            assert!(!nodes.contains(&"node2".to_string()));
        }
    }
}
