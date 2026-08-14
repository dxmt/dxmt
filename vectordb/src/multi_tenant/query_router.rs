use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTable {
    pub version: u64,
    pub routes: HashMap<String, Route>,
    pub default_route: Option<Route>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub pattern: String,
    pub targets: Vec<NodeInfo>,
    pub strategy: RoutingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub nodeid: Uuid,
    pub address: String,
    pub weight: f32,
    pub healthy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    RoundRobin,
    Random,
    LeastLoaded,
    ConsistentHash,
    Geographic,
    Custom(String),
}

pub struct QueryRouter {
    routing_table: Arc<RwLock<RoutingTable>>,
}

impl QueryRouter {
    pub fn new() -> Self {
        Self {
            routing_table: Arc::new(RwLock::new(RoutingTable {
                version: 0,
                routes: HashMap::new(),
                default_route: None,
            })),
        }
    }

    pub async fn route_query(&self, table_name: &str) -> crate::Result<Vec<NodeInfo>> {
        let table = self.routing_table.read().await;
        for (pattern, route) in &table.routes {
            if Self::matches_pattern(table_name, pattern) {
                return Ok(route.targets.clone());
            }
        }
        if let Some(default) = &table.default_route {
            return Ok(default.targets.clone());
        }
        Err(crate::error::Error::NotFound(format!(
            "No route for table '{}'",
            table_name
        )))
    }

    pub async fn update_routing_table(&self, new_table: RoutingTable) -> crate::Result<()> {
        let mut table = self.routing_table.write().await;
        *table = new_table;
        Ok(())
    }

    fn matches_pattern(table_name: &str, pattern: &str) -> bool {
        pattern == "*" || table_name == pattern
    }
}

impl Default for QueryRouter {
    fn default() -> Self {
        Self::new()
    }
}
