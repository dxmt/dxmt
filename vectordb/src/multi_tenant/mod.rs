pub mod consistency;
pub mod query_router;
pub mod shard_manager;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::types::vector::DenseVector;
use crate::Result;

/// Per-tenant usage limits enforced at the API layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageLimits {
    /// Maximum number of collections (tables) this tenant may create.
    pub max_collections: usize,
    /// Maximum records per collection.
    pub max_records_per_collection: usize,
    /// Maximum total vectors across all collections.
    pub max_total_vectors: u64,
}

impl Default for UsageLimits {
    fn default() -> Self {
        Self {
            max_collections: 100,
            max_records_per_collection: 1_000_000,
            max_total_vectors: 10_000_000,
        }
    }
}

/// Tenant quota state — current usage compared against limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantQuota {
    pub tenant_id: String,
    pub limits: UsageLimits,
    pub current_collections: usize,
    pub current_total_vectors: u64,
}

impl TenantQuota {
    pub fn new(tenant_id: impl Into<String>, limits: UsageLimits) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            limits,
            current_collections: 0,
            current_total_vectors: 0,
        }
    }

    /// Returns an error string if creating a new collection would exceed the limit.
    pub fn check_collection_limit(&self) -> Result<()> {
        if self.current_collections >= self.limits.max_collections {
            return Err(crate::error::Error::InvalidInput(format!(
                "Tenant '{}' collection limit reached ({}/{})",
                self.tenant_id, self.current_collections, self.limits.max_collections
            )));
        }
        Ok(())
    }

    /// Returns an error string if inserting `count` records would exceed per-collection limit.
    pub fn check_records_limit(&self, collection_count: usize, count: usize) -> Result<()> {
        if collection_count + count > self.limits.max_records_per_collection {
            return Err(crate::error::Error::InvalidInput(format!(
                "Tenant '{}' per-collection record limit would be exceeded ({} + {} > {})",
                self.tenant_id, collection_count, count, self.limits.max_records_per_collection
            )));
        }
        Ok(())
    }
}

/// Federation node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationNode {
    pub id: Uuid,
    pub name: String,
    pub endpoint: String,
    pub region: String,
    pub capacity: NodeCapacity,
    pub status: NodeStatus,
    pub payload: HashMap<String, String>,
}

/// Node capacity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapacity {
    pub max_vectors: u64,
    pub current_vectors: u64,
    pub max_storage_gb: f64,
    pub current_storage_gb: f64,
    pub cpu_cores: u32,
    pub memorygb: f64,
}

/// Node status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    Active,
    Degraded,
    Maintenance,
    Offline,
}

/// Federation query request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationQuery {
    pub tablename: String,
    pub vector: DenseVector,
    pub k: usize,
    pub filter: Option<String>,
    pub consistency_level: ConsistencyLevel,
    pub timeout_ms: u64,
}

/// Consistency levels for federated queries
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsistencyLevel {
    /// Return as soon as one node responds
    One,
    /// Return when majority of nodes respond
    Quorum,
    /// Return when all nodes respond
    All,
    /// Return when at least N nodes respond
    AtLeast(usize),
}

/// Federation coordinator trait
#[async_trait]
pub trait FederationCoordinator: Send + Sync {
    async fn register_node(&self, node: FederationNode) -> crate::Result<()>;
    async fn unregister_node(&self, nodeid: &Uuid) -> crate::Result<()>;
    async fn list_nodes(&self) -> crate::Result<Vec<FederationNode>>;
    async fn get_node(&self, nodeid: &Uuid) -> crate::Result<Option<FederationNode>>;
    async fn update_node_status(&self, nodeid: &Uuid, status: NodeStatus) -> crate::Result<()>;
    async fn federated_search(
        &self,
        query: FederationQuery,
    ) -> crate::Result<Vec<crate::types::record::SearchResult>>;
    async fn federated_insert(
        &self,
        tablename: &str,
        records: Vec<crate::types::record::Record>,
        replication_factor: usize,
    ) -> crate::Result<()>;
    async fn health_check(&self) -> crate::Result<FederationHealth>;
}

/// Federation health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationHealth {
    pub healthy_nodes: usize,
    pub total_nodes: usize,
    pub total_vectors: u64,
    pub total_storage_gb: f64,
    pub replication_factor: f64,
    pub node_statuses: HashMap<Uuid, NodeStatus>,
}

/// Federation error types
#[derive(Debug, thiserror::Error)]
pub enum FederationError {
    #[error("Node not found: {0}")]
    NodeNotFound(Uuid),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Consistency requirement not met: {0}")]
    ConsistencyError(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Replication error: {0}")]
    ReplicationError(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Cross-region replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    pub replication_factor: usize,
    pub write_consistency: ConsistencyLevel,
    pub read_consistency: ConsistencyLevel,
    pub async_replication: bool,
    pub conflict_resolution: ConflictResolution,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictResolution {
    LastWriteWins,
    VectorClock,
    Custom(String),
}

/// Data placement strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementStrategy {
    RoundRobin,
    ConsistentHashing,
    GeoAware {
        preferred_regions: Vec<String>,
        fallback_regions: Vec<String>,
    },
    CapacityBased {
        threshold: f64,
    },
    Custom(String),
}

/// Federation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStats {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub cross_region_traffic_gb: f64,
    pub replication_lag_ms: f64,
}

/// Multi-region table configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRegionTable {
    pub table: crate::types::table::Table,
    pub regions: Vec<String>,
    pub primary_region: String,
    pub replication_config: ReplicationConfig,
    pub placement_strategy: PlacementStrategy,
    pub geo_partitioning: Option<GeoPartitioning>,
}

/// Geographic partitioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoPartitioning {
    pub partition_key: String,
    pub region_mapping: HashMap<String, Vec<f32>>,
    pub default_region: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federation_node_creation() {
        let node = FederationNode {
            id: Uuid::new_v4(),
            name: "node-1".to_string(),
            endpoint: "http://node1.example.com".to_string(),
            region: "us-east-1".to_string(),
            capacity: NodeCapacity {
                max_vectors: 1_000_000,
                current_vectors: 500_000,
                max_storage_gb: 100.0,
                current_storage_gb: 50.0,
                cpu_cores: 16,
                memorygb: 64.0,
            },
            status: NodeStatus::Active,
            payload: HashMap::new(),
        };

        assert_eq!(node.status, NodeStatus::Active);
        assert_eq!(node.capacity.max_vectors, 1_000_000);
    }

    #[test]
    fn test_consistency_levels_serde() {
        let levels = vec![
            ConsistencyLevel::One,
            ConsistencyLevel::Quorum,
            ConsistencyLevel::All,
            ConsistencyLevel::AtLeast(3),
        ];

        for level in levels {
            let json = serde_json::to_string(&level).unwrap();
            let deserialized: ConsistencyLevel = serde_json::from_str(&json).unwrap();
            match (level, deserialized) {
                (ConsistencyLevel::AtLeast(n1), ConsistencyLevel::AtLeast(n2)) => {
                    assert_eq!(n1, n2);
                }
                _ => {
                    assert_eq!(
                        std::mem::discriminant(&level),
                        std::mem::discriminant(&deserialized)
                    );
                }
            }
        }
    }

    #[test]
    fn test_usage_limits_defaults() {
        let limits = UsageLimits::default();
        assert_eq!(limits.max_collections, 100);
        assert_eq!(limits.max_records_per_collection, 1_000_000);
    }

    #[test]
    fn test_tenant_quota_collection_limit() {
        let limits = UsageLimits {
            max_collections: 2,
            max_records_per_collection: 1000,
            max_total_vectors: 10_000,
        };
        let mut quota = TenantQuota::new("tenant-1", limits);

        // Under limit — ok
        assert!(quota.check_collection_limit().is_ok());

        quota.current_collections = 2;
        assert!(quota.check_collection_limit().is_err());
    }

    #[test]
    fn test_tenant_quota_records_limit() {
        let limits = UsageLimits {
            max_collections: 10,
            max_records_per_collection: 100,
            max_total_vectors: 10_000,
        };
        let quota = TenantQuota::new("tenant-1", limits);

        // 90 existing + 5 new = 95 ≤ 100 → ok
        assert!(quota.check_records_limit(90, 5).is_ok());
        // 90 existing + 20 new = 110 > 100 → err
        assert!(quota.check_records_limit(90, 20).is_err());
    }
}
