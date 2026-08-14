pub mod shard_router;
pub mod simple_discovery;

use serde::{Deserialize, Serialize};

pub use simple_discovery::{SimpleDiscovery, ClusterConfig, NodeAnnouncement};

/// Node ID type
pub type NodeId = String;

/// Node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: NodeId,
    pub address: String,
    pub status: NodeStatus,
}

/// Node status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    Active,
    Inactive,
    Unreachable,
}