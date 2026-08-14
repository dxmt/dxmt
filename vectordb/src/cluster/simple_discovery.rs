use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub node_id: Uuid,
    pub listen_addr: String,
    pub seeds: Vec<String>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            node_id: Uuid::new_v4(),
            listen_addr: "127.0.0.1:8090".to_string(),
            seeds: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAnnouncement {
    pub nodeid: Uuid,
    pub addr: String,
    pub metadata: HashMap<String, String>,
}

pub struct SimpleDiscovery {
    #[allow(dead_code)]
    config: ClusterConfig,
    nodes: Arc<RwLock<HashMap<String, NodeAnnouncement>>>,
}

impl SimpleDiscovery {
    pub fn new(config: ClusterConfig) -> Self {
        Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn handle_announcement(&self, announcement: NodeAnnouncement) -> Result<(), String> {
        let mut nodes = self.nodes.write().await;
        nodes.insert(announcement.nodeid.to_string(), announcement);
        Ok(())
    }

    pub async fn update_heartbeat(&self, node_id: &str) -> Result<(), String> {
        let nodes = self.nodes.read().await;
        if nodes.contains_key(node_id) {
            Ok(())
        } else {
            Err(format!("Node {} not found", node_id))
        }
    }

    pub async fn get_active_nodes(&self) -> Vec<NodeAnnouncement> {
        let nodes = self.nodes.read().await;
        nodes.values().cloned().collect()
    }
}