use heed::{Database, Env, EnvOpenOptions, RoTxn, RwTxn};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use crate::types::{RecordId, DistanceMetric};

// Export Storage trait implementation
pub mod storage_impl;
pub use storage_impl::LmdbStorageImpl;

const DB_VECTORS: &str = "vectors";
const DB_METADATA: &str = "metadata";
const DB_PAYLOADS: &str = "payloads";
const DB_GRAPH_NODES: &str = "graph_nodes";
const DB_GRAPH_EDGES: &str = "graph_edges";
const DB_INDICES: &str = "indices";

#[derive(Clone)]
pub struct LmdbStorage {
    env: Arc<Env>,
    vectors_db: Database<heed::types::Bytes, heed::types::Bytes>,
    metadata_db: Database<heed::types::Bytes, heed::types::Bytes>,
    payloads_db: Database<heed::types::Bytes, heed::types::Bytes>,
    graph_nodes_db: Database<heed::types::Bytes, heed::types::Bytes>,
    graph_edges_db: Database<heed::types::Bytes, heed::types::Bytes>,
    #[allow(dead_code)]
    indices_db: Database<heed::types::Bytes, heed::types::Bytes>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: RecordId,
    pub vector: Vec<f32>,
    pub payload: Option<serde_json::Value>,
}

// Internal struct for storing in vectors_db (without payload)
#[derive(Debug, Serialize, Deserialize)]
struct VectorData {
    id: RecordId,
    vector: Vec<f32>,
}

// Internal structs for graph data (without serde_json::Value properties)
#[derive(Debug, Serialize, Deserialize)]
struct GraphNodeData {
    id: Uuid,
    labels: Vec<String>,
    vector_id: Option<RecordId>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GraphEdgeData {
    id: Uuid,
    from_node: Uuid,
    to_node: Uuid,
    label: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: Uuid,
    pub labels: Vec<String>,
    pub properties: serde_json::Value,
    pub vector_id: Option<RecordId>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: Uuid,
    pub from_node: Uuid,
    pub to_node: Uuid,
    pub label: String,
    pub properties: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TableMetadata {
    pub name: String,
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
    pub total_records: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl LmdbStorage {
    pub fn new(path: impl AsRef<Path>, map_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&path)?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(map_size)
                .max_dbs(10)
                .open(path)?
        };

        let mut wtxn = env.write_txn()?;

        let vectors_db: Database<heed::types::Bytes, heed::types::Bytes> = env.database_options()
            .name(DB_VECTORS)
            .types::<heed::types::Bytes, heed::types::Bytes>()
            .create(&mut wtxn)?;

        let metadata_db: Database<heed::types::Bytes, heed::types::Bytes> = env.database_options()
            .name(DB_METADATA)
            .types::<heed::types::Bytes, heed::types::Bytes>()
            .create(&mut wtxn)?;

        let payloads_db: Database<heed::types::Bytes, heed::types::Bytes> = env.database_options()
            .name(DB_PAYLOADS)
            .types::<heed::types::Bytes, heed::types::Bytes>()
            .create(&mut wtxn)?;

        let graph_nodes_db: Database<heed::types::Bytes, heed::types::Bytes> = env.database_options()
            .name(DB_GRAPH_NODES)
            .types::<heed::types::Bytes, heed::types::Bytes>()
            .create(&mut wtxn)?;

        let graph_edges_db: Database<heed::types::Bytes, heed::types::Bytes> = env.database_options()
            .name(DB_GRAPH_EDGES)
            .types::<heed::types::Bytes, heed::types::Bytes>()
            .create(&mut wtxn)?;

        let indices_db: Database<heed::types::Bytes, heed::types::Bytes> = env.database_options()
            .name(DB_INDICES)
            .types::<heed::types::Bytes, heed::types::Bytes>()
            .create(&mut wtxn)?;

        wtxn.commit()?;

        Ok(Self {
            env: Arc::new(env),
            vectors_db,
            metadata_db,
            payloads_db,
            graph_nodes_db,
            graph_edges_db,
            indices_db,
        })
    }

    pub fn insert_vector(&self, record: &VectorRecord) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut wtxn = self.env.write_txn()?;

        let key = record.id.as_bytes();

        // Store vector data without payload (bincode doesn't handle serde_json::Value well)
        let vector_data = VectorData {
            id: record.id.clone(),
            vector: record.vector.clone(),
        };
        let value = bincode::serialize(&vector_data)?;

        self.vectors_db.put(&mut wtxn, key, &value)?;

        if let Some(ref payload) = record.payload {
            let payload_value = serde_json::to_vec(payload)?;
            self.payloads_db.put(&mut wtxn, key, &payload_value)?;
        }

        wtxn.commit()?;
        Ok(())
    }

    pub fn get_vector(&self, id: &RecordId) -> Result<Option<VectorRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let rtxn = self.env.read_txn()?;
        let key = id.as_bytes();

        if let Some(value) = self.vectors_db.get(&rtxn, key)? {
            // Deserialize vector data (without payload)
            let vector_data: VectorData = bincode::deserialize(value)?;

            // Retrieve payload separately from payloads_db
            let payload = if let Some(payload_bytes) = self.payloads_db.get(&rtxn, key)? {
                Some(serde_json::from_slice(payload_bytes)?)
            } else {
                None
            };

            // Reconstruct full VectorRecord
            let record = VectorRecord {
                id: vector_data.id,
                vector: vector_data.vector,
                payload,
            };

            Ok(Some(record))
        } else {
            Ok(None)
        }
    }

    pub fn delete_vector(&self, id: &RecordId) -> Result<bool, Box<dyn std::error::Error>> {
        let mut wtxn = self.env.write_txn()?;
        let key = id.as_bytes();

        let deleted = self.vectors_db.delete(&mut wtxn, key)?;
        self.payloads_db.delete(&mut wtxn, key)?;

        wtxn.commit()?;
        Ok(deleted)
    }

    pub fn insert_graph_node(&self, node: &GraphNode) -> Result<(), Box<dyn std::error::Error>> {
        let mut wtxn = self.env.write_txn()?;

        let key = node.id.as_bytes();

        // Store node data without properties (bincode doesn't handle serde_json::Value well)
        let node_data = GraphNodeData {
            id: node.id,
            labels: node.labels.clone(),
            vector_id: node.vector_id.clone(),
        };
        let value = bincode::serialize(&node_data)?;

        self.graph_nodes_db.put(&mut wtxn, key, &value)?;

        // Store properties separately if present
        let properties_key = format!("props_{}", node.id).into_bytes();
        let properties_value = serde_json::to_vec(&node.properties)?;
        self.metadata_db.put(&mut wtxn, &properties_key, &properties_value)?;

        wtxn.commit()?;
        Ok(())
    }

    pub fn get_graph_node(&self, id: &Uuid) -> Result<Option<GraphNode>, Box<dyn std::error::Error>> {
        let rtxn = self.env.read_txn()?;
        let key = id.as_bytes();

        if let Some(value) = self.graph_nodes_db.get(&rtxn, key)? {
            // Deserialize node data (without properties)
            let node_data: GraphNodeData = bincode::deserialize(value)?;

            // Retrieve properties separately
            let properties_key = format!("props_{}", id).into_bytes();
            let properties = if let Some(props_bytes) = self.metadata_db.get(&rtxn, &properties_key)? {
                serde_json::from_slice(props_bytes)?
            } else {
                serde_json::json!({})
            };

            // Reconstruct full GraphNode
            let node = GraphNode {
                id: node_data.id,
                labels: node_data.labels,
                properties,
                vector_id: node_data.vector_id,
            };

            Ok(Some(node))
        } else {
            Ok(None)
        }
    }

    pub fn insert_graph_edge(&self, edge: &GraphEdge) -> Result<(), Box<dyn std::error::Error>> {
        let mut wtxn = self.env.write_txn()?;

        let key = edge.id.as_bytes();

        // Store edge data without properties (bincode doesn't handle serde_json::Value well)
        let edge_data = GraphEdgeData {
            id: edge.id,
            from_node: edge.from_node,
            to_node: edge.to_node,
            label: edge.label.clone(),
        };
        let value = bincode::serialize(&edge_data)?;

        self.graph_edges_db.put(&mut wtxn, key, &value)?;

        // Store properties separately
        let properties_key = format!("edge_props_{}", edge.id).into_bytes();
        let properties_value = serde_json::to_vec(&edge.properties)?;
        self.metadata_db.put(&mut wtxn, &properties_key, &properties_value)?;

        wtxn.commit()?;
        Ok(())
    }

    pub fn get_graph_edges_from(&self, node_id: &Uuid) -> Result<Vec<GraphEdge>, Box<dyn std::error::Error>> {
        let rtxn = self.env.read_txn()?;
        let mut edges = Vec::new();

        for result in self.graph_edges_db.iter(&rtxn)? {
            let (_, value) = result?;
            // Deserialize edge data (without properties)
            let edge_data: GraphEdgeData = bincode::deserialize(value)?;
            if edge_data.from_node == *node_id {
                // Retrieve properties separately
                let properties_key = format!("edge_props_{}", edge_data.id).into_bytes();
                let properties = if let Some(props_bytes) = self.metadata_db.get(&rtxn, &properties_key)? {
                    serde_json::from_slice(props_bytes)?
                } else {
                    serde_json::json!({})
                };

                // Reconstruct full GraphEdge
                edges.push(GraphEdge {
                    id: edge_data.id,
                    from_node: edge_data.from_node,
                    to_node: edge_data.to_node,
                    label: edge_data.label,
                    properties,
                });
            }
        }

        Ok(edges)
    }

    pub fn get_graph_edges_to(&self, node_id: &Uuid) -> Result<Vec<GraphEdge>, Box<dyn std::error::Error>> {
        let rtxn = self.env.read_txn()?;
        let mut edges = Vec::new();

        for result in self.graph_edges_db.iter(&rtxn)? {
            let (_, value) = result?;
            // Deserialize edge data (without properties)
            let edge_data: GraphEdgeData = bincode::deserialize(value)?;
            if edge_data.to_node == *node_id {
                // Retrieve properties separately
                let properties_key = format!("edge_props_{}", edge_data.id).into_bytes();
                let properties = if let Some(props_bytes) = self.metadata_db.get(&rtxn, &properties_key)? {
                    serde_json::from_slice(props_bytes)?
                } else {
                    serde_json::json!({})
                };

                // Reconstruct full GraphEdge
                edges.push(GraphEdge {
                    id: edge_data.id,
                    from_node: edge_data.from_node,
                    to_node: edge_data.to_node,
                    label: edge_data.label,
                    properties,
                });
            }
        }

        Ok(edges)
    }

    /// Store full Table as JSON under key `"__fullcfg__{name}"` for complete schema recovery.
    pub fn store_full_table(&self, table: &crate::types::table::Table) -> Result<(), Box<dyn std::error::Error>> {
        let mut wtxn = self.env.write_txn()?;
        let key = format!("__fullcfg__{}", table.name);
        let value = serde_json::to_vec(table)?;
        self.metadata_db.put(&mut wtxn, key.as_bytes(), &value)?;
        wtxn.commit()?;
        Ok(())
    }

    /// Load full Table from JSON; returns None if not present (legacy table).
    pub fn load_full_table(&self, table_name: &str) -> Result<Option<crate::types::table::Table>, Box<dyn std::error::Error>> {
        let rtxn = self.env.read_txn()?;
        let key = format!("__fullcfg__{}", table_name);
        if let Some(value) = self.metadata_db.get(&rtxn, key.as_bytes())? {
            let table: crate::types::table::Table = serde_json::from_slice(value)?;
            Ok(Some(table))
        } else {
            Ok(None)
        }
    }

    pub fn update_table_metadata(&self, table_name: &str, metadata: &TableMetadata) -> Result<(), Box<dyn std::error::Error>> {
        let mut wtxn = self.env.write_txn()?;

        let key = table_name.as_bytes();
        let value = bincode::serialize(metadata)?;

        self.metadata_db.put(&mut wtxn, key, &value)?;

        wtxn.commit()?;
        Ok(())
    }

    pub fn get_table_metadata(&self, table_name: &str) -> Result<Option<TableMetadata>, Box<dyn std::error::Error>> {
        let rtxn = self.env.read_txn()?;
        let key = table_name.as_bytes();

        if let Some(value) = self.metadata_db.get(&rtxn, key)? {
            let metadata: TableMetadata = bincode::deserialize(value)?;
            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }

    pub fn begin_read_txn(&self) -> Result<RoTxn<'_>, Box<dyn std::error::Error>> {
        Ok(self.env.read_txn()?)
    }

    pub fn begin_write_txn(&self) -> Result<RwTxn<'_>, Box<dyn std::error::Error>> {
        Ok(self.env.write_txn()?)
    }

    pub fn stat(&self) -> Result<heed::EnvInfo, Box<dyn std::error::Error>> {
        Ok(self.env.info())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_lmdb_storage_basic_operations() {
        let dir = tempdir().unwrap();
        let storage = LmdbStorage::new(dir.path(), 10 * 1024 * 1024).unwrap();

        let record = VectorRecord {
            id: uuid::Uuid::new_v4().to_string(),
            vector: vec![1.0, 2.0, 3.0],
            payload: Some(serde_json::json!({
                "name": "test"
            })),
        };

        storage.insert_vector(&record).unwrap();

        let retrieved = storage.get_vector(&record.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().vector, record.vector);

        let deleted = storage.delete_vector(&record.id).unwrap();
        assert!(deleted);

        let retrieved = storage.get_vector(&record.id).unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_graph_operations() {
        let dir = tempdir().unwrap();
        let storage = LmdbStorage::new(dir.path(), 10 * 1024 * 1024).unwrap();

        let node1 = GraphNode {
            id: Uuid::new_v4(),
            labels: vec!["Person".to_string()],
            properties: serde_json::json!({"name": "Alice"}),
            vector_id: None,
        };

        let node2 = GraphNode {
            id: Uuid::new_v4(),
            labels: vec!["Person".to_string()],
            properties: serde_json::json!({"name": "Bob"}),
            vector_id: None,
        };

        storage.insert_graph_node(&node1).unwrap();
        storage.insert_graph_node(&node2).unwrap();

        let edge = GraphEdge {
            id: Uuid::new_v4(),
            from_node: node1.id,
            to_node: node2.id,
            label: "KNOWS".to_string(),
            properties: serde_json::json!({"since": 2020}),
        };

        storage.insert_graph_edge(&edge).unwrap();

        let edges_from = storage.get_graph_edges_from(&node1.id).unwrap();
        assert_eq!(edges_from.len(), 1);
        assert_eq!(edges_from[0].label, "KNOWS");

        let edges_to = storage.get_graph_edges_to(&node2.id).unwrap();
        assert_eq!(edges_to.len(), 1);
    }
}