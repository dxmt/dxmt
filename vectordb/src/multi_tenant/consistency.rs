use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorClock {
    pub nodeid: Uuid,
    pub clocks: HashMap<Uuid, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictResolutionStrategy {
    LastWriteWins,
    FirstWriteWins,
    HighestValueWins,
    MergeVectors,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationLog {
    pub entries: Vec<ReplicationEntry>,
    pub last_applied_index: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationEntry {
    pub index: u64,
    pub timestamp: DateTime<Utc>,
    pub operation: ReplicationOperation,
    pub nodeid: Uuid,
    pub vector_clock: VectorClock,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationOperation {
    Insert { table: String, key: Vec<u8>, value: Vec<u8> },
    Update { table: String, key: Vec<u8>, value: Vec<u8>, previous_version: Option<Vec<u8>> },
    Delete { table: String, key: Vec<u8> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClockComparison {
    Before,
    After,
    Equal,
    Concurrent,
}

#[allow(dead_code)]
pub struct ConsistencyManager {
    vector_clocks: Arc<RwLock<HashMap<Uuid, VectorClock>>>,
    replication_log: Arc<RwLock<ReplicationLog>>,
    strategy: ConflictResolutionStrategy,
}

impl ConsistencyManager {
    pub fn new(strategy: ConflictResolutionStrategy) -> Self {
        Self {
            vector_clocks: Arc::new(RwLock::new(HashMap::new())),
            replication_log: Arc::new(RwLock::new(ReplicationLog {
                entries: Vec::new(),
                last_applied_index: 0,
            })),
            strategy,
        }
    }

    pub async fn update_vector_clock(&self, nodeid: Uuid) -> VectorClock {
        let mut clocks = self.vector_clocks.write().await;
        let clock = clocks.entry(nodeid).or_insert_with(|| VectorClock {
            nodeid,
            clocks: HashMap::new(),
        });
        *clock.clocks.entry(nodeid).or_insert(0) += 1;
        clock.clone()
    }

    pub fn merge_vector_clocks(&self, clock1: &VectorClock, clock2: &VectorClock) -> VectorClock {
        let mut merged = VectorClock {
            nodeid: clock1.nodeid,
            clocks: clock1.clocks.clone(),
        };
        for (node, &time) in &clock2.clocks {
            merged
                .clocks
                .entry(*node)
                .and_modify(|e| *e = (*e).max(time))
                .or_insert(time);
        }
        merged
    }

    pub fn compare_vector_clocks(&self, clock1: &VectorClock, clock2: &VectorClock) -> ClockComparison {
        let mut c1_newer = false;
        let mut c2_newer = false;

        for (node, &t1) in &clock1.clocks {
            let t2 = clock2.clocks.get(node).copied().unwrap_or(0);
            if t1 > t2 {
                c1_newer = true;
            } else if t1 < t2 {
                c2_newer = true;
            }
        }
        for (node, &t2) in &clock2.clocks {
            if !clock1.clocks.contains_key(node) && t2 > 0 {
                c2_newer = true;
            }
        }

        match (c1_newer, c2_newer) {
            (true, false) => ClockComparison::After,
            (false, true) => ClockComparison::Before,
            (false, false) => ClockComparison::Equal,
            (true, true) => ClockComparison::Concurrent,
        }
    }

    pub async fn append_to_replication_log(&self, operation: ReplicationOperation, nodeid: Uuid) {
        let vector_clock = self.update_vector_clock(nodeid).await;
        let mut log = self.replication_log.write().await;
        let index = log.entries.len() as u64 + 1;
        log.entries.push(ReplicationEntry {
            index,
            timestamp: Utc::now(),
            operation,
            nodeid,
            vector_clock,
        });
    }

    pub async fn get_replication_entries_since(&self, index: u64) -> Vec<ReplicationEntry> {
        let log = self.replication_log.read().await;
        log.entries.iter().filter(|e| e.index > index).cloned().collect()
    }
}

impl Default for ConsistencyManager {
    fn default() -> Self {
        Self::new(ConflictResolutionStrategy::LastWriteWins)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_comparison() {
        let mgr = ConsistencyManager::default();
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();

        let mut c1 = VectorClock { nodeid: node_a, clocks: HashMap::new() };
        let mut c2 = VectorClock { nodeid: node_b, clocks: HashMap::new() };

        c1.clocks.insert(node_a, 5);
        c2.clocks.insert(node_a, 3);
        assert_eq!(mgr.compare_vector_clocks(&c1, &c2), ClockComparison::After);

        c2.clocks.insert(node_b, 10);
        assert_eq!(mgr.compare_vector_clocks(&c1, &c2), ClockComparison::Concurrent);
    }

    #[test]
    fn test_merge_vector_clocks() {
        let mgr = ConsistencyManager::default();
        let n1 = Uuid::new_v4();
        let n2 = Uuid::new_v4();

        let mut c1 = VectorClock { nodeid: n1, clocks: HashMap::new() };
        let mut c2 = VectorClock { nodeid: n2, clocks: HashMap::new() };
        c1.clocks.insert(n1, 5);
        c1.clocks.insert(n2, 2);
        c2.clocks.insert(n1, 3);
        c2.clocks.insert(n2, 7);

        let merged = mgr.merge_vector_clocks(&c1, &c2);
        assert_eq!(*merged.clocks.get(&n1).unwrap(), 5);
        assert_eq!(*merged.clocks.get(&n2).unwrap(), 7);
    }
}
