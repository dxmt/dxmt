//! LlamaIndex integration for DXMT VectorDB

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use uuid::Uuid;

use crate::types::{Record, Payload};
use crate::types::record::SearchQuery;
use crate::storage::Storage;
use crate::embeddings::{EmbeddingFunction, AutoEmbedding, AutoEmbeddingConfig};

/// LlamaIndex Node structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub embedding: Option<Vec<f32>>,
    pub relationships: HashMap<String, RelatedNode>,
}

impl Node {
    pub fn new(text: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            text,
            metadata: HashMap::new(),
            embedding: None,
            relationships: HashMap::new(),
        }
    }
    
    pub fn with_id(mut self, id: String) -> Self {
        self.id = id;
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
    
    pub fn add_relationship(mut self, rel_type: String, node_id: String) -> Self {
        self.relationships.insert(rel_type, RelatedNode {
            node_id,
            metadata: HashMap::new(),
        });
        self
    }
}

/// Related node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedNode {
    pub node_id: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Storage context for LlamaIndex
#[async_trait]
pub trait StorageContext: Send + Sync {
    /// Add nodes to storage
    async fn add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<String>, StorageError>;
    
    /// Get node by ID
    async fn get_node(&self, node_id: &str) -> Result<Option<Node>, StorageError>;
    
    /// Update node
    async fn update_node(&self, node: Node) -> Result<(), StorageError>;
    
    /// Delete nodes
    async fn delete_nodes(&self, node_ids: Vec<String>) -> Result<(), StorageError>;
    
    /// Get all nodes
    async fn get_all_nodes(&self) -> Result<Vec<Node>, StorageError>;
}

/// Storage errors
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Storage error: {0}")]
    Internal(String),
    
    #[error("Node not found: {0}")]
    NotFound(String),
    
    #[error("Invalid node: {0}")]
    InvalidNode(String),
}

/// DXMT Storage Context implementation
pub struct DXMTStorageContext {
    storage: Arc<dyn Storage + Send + Sync>,
    table_name: String,
    nodes: Arc<RwLock<HashMap<String, Node>>>,
}

impl DXMTStorageContext {
    pub async fn new(
        storage: Arc<dyn Storage + Send + Sync>,
        table_name: String,
    ) -> Result<Self, StorageError> {
        Ok(Self {
            storage,
            table_name,
            nodes: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    fn node_to_record(&self, node: &Node) -> Record {
        let mut payload_map = serde_json::Map::new();
        payload_map.insert("text".to_string(), serde_json::Value::String(node.text.clone()));
        payload_map.insert("node_id".to_string(), serde_json::Value::String(node.id.clone()));
        
        // Add metadata
        for (key, value) in &node.metadata {
            payload_map.insert(format!("meta_{}", key), value.clone());
        }
        
        // Add relationships
        if !node.relationships.is_empty() {
            let relationships_json = serde_json::to_value(&node.relationships).unwrap();
            payload_map.insert("relationships".to_string(), relationships_json);
        }
        
        let payload = serde_json::Value::Object(payload_map);
        
        Record {
            id: node.id.clone(),
            vector: node.embedding.clone().unwrap_or_else(|| vec![0.0; 768]),
            sparse_vector: None,
            payload,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 0,
            status: crate::types::record::RecordStatus::Active,
            named_vectors: None,
        }
    }
    
    fn record_to_node(&self, record: &Record) -> Node {
        let mut node = Node::new(String::new());
        node.id = record.id.to_string();
        node.embedding = Some(record.vector.clone());
        
        if let Some(obj) = record.payload.as_object() {
            let payload = &serde_json::Value::Object(obj.clone());
            if let Some(text) = payload.get("text").and_then(|v| v.as_str()) {
                node.text = text.to_string();
            }
            
            // Extract metadata
            if let Some(obj) = payload.as_object() {
                for (key, value) in obj.iter() {
                    if key.starts_with("meta_") {
                        let meta_key = key.strip_prefix("meta_").unwrap();
                        node.metadata.insert(meta_key.to_string(), value.clone());
                    }
                }
            }
            
            // Extract relationships
            if let Some(relationships) = payload.get("relationships") {
                if let Ok(rels) = serde_json::from_value::<HashMap<String, RelatedNode>>(relationships.clone()) {
                    node.relationships = rels;
                }
            }
        }
        
        node
    }
}

#[async_trait]
impl StorageContext for DXMTStorageContext {
    async fn add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<String>, StorageError> {
        let table = self.storage.get_table(&self.table_name).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        let mut ids = Vec::new();
        let mut nodes_map = self.nodes.write().await;
        
        for node in nodes {
            let record = self.node_to_record(&node);
            let request = crate::types::record::CreateRecordRequest {
                id: record.id.clone(),
                vector: record.vector,
                sparse_vector: record.sparse_vector,
                payload: Some(record.payload),
                named_vectors: None,
            };
            self.storage.insert_record(&self.table_name, request).await
                .map_err(|e| StorageError::Internal(e.to_string()))?;
                
            ids.push(node.id.clone());
            nodes_map.insert(node.id.clone(), node);
        }
        
        Ok(ids)
    }
    
    async fn get_node(&self, node_id: &str) -> Result<Option<Node>, StorageError> {
        // Check cache first
        let nodes = self.nodes.read().await;
        if let Some(node) = nodes.get(node_id) {
            return Ok(Some(node.clone()));
        }
        drop(nodes);
        
        // Fetch from storage
        let table = self.storage.get_table(&self.table_name).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        let record_id = node_id.to_string();
        match self.storage.get_record(&self.table_name, &record_id).await {
            Ok(record) => {
                let node = self.record_to_node(&record);
                let mut nodes = self.nodes.write().await;
                nodes.insert(node_id.to_string(), node.clone());
                Ok(Some(node))
            },
            Err(_) => Ok(None),
        }
    }
    
    async fn update_node(&self, node: Node) -> Result<(), StorageError> {
        let table = self.storage.get_table(&self.table_name).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        let record = self.node_to_record(&node);
        let update_request = crate::types::record::CreateRecordRequest {
            id: record.id.clone(),
            vector: record.vector,
            sparse_vector: record.sparse_vector,
            payload: Some(record.payload),
            named_vectors: None,
        };
        self.storage.update_record(&self.table_name, &node.id, update_request).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        let mut nodes = self.nodes.write().await;
        nodes.insert(node.id.clone(), node);
        
        Ok(())
    }
    
    async fn delete_nodes(&self, node_ids: Vec<String>) -> Result<(), StorageError> {
        let table = self.storage.get_table(&self.table_name).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        let mut nodes = self.nodes.write().await;
        
        for node_id in node_ids {
            let record_id = node_id.clone();
            self.storage.delete_record(&self.table_name, &record_id).await
                .map_err(|e| StorageError::Internal(e.to_string()))?;
            nodes.remove(&node_id);
        }
        
        Ok(())
    }
    
    async fn get_all_nodes(&self) -> Result<Vec<Node>, StorageError> {
        let table = self.storage.get_table(&self.table_name).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        // Use search with a dummy query to get all records
        let dummy_query = SearchQuery {
            vector: vec![0.0; table.dimension],  // Dummy vector
            top_k: 1000,  // Get many results
            filter: None,
            with_payload: true,
            with_vector: true,
            ..Default::default()
        };
        
        let search_results = self.storage.search(&self.table_name, dummy_query).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        let nodes: Vec<Node> = search_results.hits.into_iter()
            .map(|hit| self.record_to_node(&Record {
                id: hit.id,
                vector: hit.vector.unwrap_or_default(),
                sparse_vector: None,
                payload: hit.payload.unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new())),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
                status: crate::types::record::RecordStatus::Active,
                named_vectors: None,
            }))
            .collect();
            
        // Update cache
        let mut nodes_map = self.nodes.write().await;
        for node in &nodes {
            nodes_map.insert(node.id.clone(), node.clone());
        }
        
        Ok(nodes)
    }
}

/// LlamaIndex VectorStoreIndex
pub struct VectorStoreIndex {
    storage_context: Arc<dyn StorageContext>,
    embedding_function: Arc<dyn EmbeddingFunction>,
    // search_engine removed - using storage directly
    storage: Arc<dyn Storage + Send + Sync>,
    table_name: String,
}

impl VectorStoreIndex {
    pub async fn new(
        storage: Arc<dyn Storage + Send + Sync>,
        table_name: String,
        embedding_function: Option<Arc<dyn EmbeddingFunction>>,
    ) -> Result<Self, StorageError> {
        let storage_context = Arc::new(
            DXMTStorageContext::new(Arc::clone(&storage), table_name.clone()).await?
        ) as Arc<dyn StorageContext>;
        
        let embedding_function = match embedding_function {
            Some(ef) => ef,
            None => {
                let config = AutoEmbeddingConfig::default();
                let auto_embedding = AutoEmbedding::new(config).await
                    .map_err(|e| StorageError::Internal(e.to_string()))?;
                Arc::new(auto_embedding) as Arc<dyn EmbeddingFunction>
            }
        };
        
        // search_engine initialization removed
        
        Ok(Self {
            storage_context,
            embedding_function,
            storage,
            table_name,
        })
    }
    
    pub async fn from_documents(&mut self, texts: Vec<String>) -> Result<(), StorageError> {
        let mut nodes = Vec::new();
        
        // Generate embeddings for each text
        for text in texts {
            let embedding = self.embedding_function.embed(&text).await
                .map_err(|e| StorageError::Internal(e.to_string()))?;
            let node = Node::new(text).with_embedding(embedding);
            nodes.push(node);
        }
        
        self.storage_context.add_nodes(nodes).await?;
        Ok(())
    }
    
    pub async fn insert(&mut self, text: String) -> Result<String, StorageError> {
        let embedding = self.embedding_function.embed(&text).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        let node = Node::new(text).with_embedding(embedding);
        let ids = self.storage_context.add_nodes(vec![node]).await?;
        
        ids.into_iter().next()
            .ok_or_else(|| StorageError::Internal("No ID returned".to_string()))
    }
    
    pub async fn delete(&mut self, node_id: String) -> Result<(), StorageError> {
        self.storage_context.delete_nodes(vec![node_id]).await
    }
    
    pub async fn update(&mut self, node: Node) -> Result<(), StorageError> {
        // Generate embedding if not present
        let node = if node.embedding.is_none() {
            let embedding = self.embedding_function.embed(&node.text).await
                .map_err(|e| StorageError::Internal(e.to_string()))?;
            node.with_embedding(embedding)
        } else {
            node
        };
        
        self.storage_context.update_node(node).await
    }
}

/// Query engine for LlamaIndex
pub struct QueryEngine {
    index: Arc<VectorStoreIndex>,
    top_k: usize,
    response_mode: ResponseMode,
}

#[derive(Debug, Clone)]
pub enum ResponseMode {
    Compact,
    Refine,
    TreeSummarize,
    Simple,
}

impl QueryEngine {
    pub fn new(index: Arc<VectorStoreIndex>) -> Self {
        Self {
            index,
            top_k: 5,
            response_mode: ResponseMode::Compact,
        }
    }
    
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }
    
    pub fn with_response_mode(mut self, mode: ResponseMode) -> Self {
        self.response_mode = mode;
        self
    }
    
    pub async fn query(&self, query: &str) -> Result<QueryResponse, StorageError> {
        // Generate query embedding
        let query_embedding = self.index.embedding_function.embed(query).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        // Search for similar nodes
        let table = self.index.storage.get_table(&self.index.table_name).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        let search_query = SearchQuery {
            vector: query_embedding,
            top_k: self.top_k,
            filter: None,
            with_payload: true,
            with_vector: false,
            ..Default::default()
        };
        
        let search_results = self.index.storage.search(&self.index.table_name, search_query).await
            .map_err(|e| StorageError::Internal(e.to_string()))?;
            
        // Extract nodes
        let mut source_nodes = Vec::new();
        for hit in search_results.hits {
            if let Some(payload) = &hit.payload {
                if let Some(text) = payload.get("text").and_then(|v| v.as_str()) {
                    source_nodes.push(SourceNode {
                        node_id: hit.id.to_string(),
                        text: text.to_string(),
                        score: hit.score,
                        metadata: payload.clone(),
                    });
                }
            }
        }
        
        // Generate response based on mode
        let response = match self.response_mode {
            ResponseMode::Compact => self.generate_compact_response(query, &source_nodes),
            ResponseMode::Refine => self.generate_refine_response(query, &source_nodes),
            ResponseMode::TreeSummarize => self.generate_tree_response(query, &source_nodes),
            ResponseMode::Simple => self.generate_simple_response(query, &source_nodes),
        };
        
        Ok(QueryResponse {
            response,
            source_nodes,
        })
    }
    
    fn generate_compact_response(&self, query: &str, nodes: &[SourceNode]) -> String {
        // Combine all node texts
        let context = nodes.iter()
            .map(|n| n.text.clone())
            .collect::<Vec<_>>()
            .join("\n\n");
            
        format!("Based on the context:\n{}\n\nAnswer to '{}': The relevant information has been retrieved.", context, query)
    }
    
    fn generate_refine_response(&self, query: &str, nodes: &[SourceNode]) -> String {
        // Iteratively refine response
        let mut response = String::new();
        for (i, node) in nodes.iter().enumerate() {
            if i == 0 {
                response = format!("Initial context: {}", node.text);
            } else {
                response = format!("{}\n\nAdditional context: {}", response, node.text);
            }
        }
        format!("{}\n\nFinal answer to '{}': The information has been compiled.", response, query)
    }
    
    fn generate_tree_response(&self, query: &str, nodes: &[SourceNode]) -> String {
        // Tree-based summarization (simplified)
        let summaries: Vec<String> = nodes.chunks(2)
            .map(|chunk| {
                chunk.iter()
                    .map(|n| n.text.clone())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
            
        format!("Tree summary for '{}': {}", query, summaries.join(" | "))
    }
    
    fn generate_simple_response(&self, query: &str, nodes: &[SourceNode]) -> String {
        // Simple concatenation
        let texts: Vec<String> = nodes.iter().map(|n| n.text.clone()).collect();
        format!("Results for '{}': {}", query, texts.join(" "))
    }
}

/// Query response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub response: String,
    pub source_nodes: Vec<SourceNode>,
}

/// Source node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceNode {
    pub node_id: String,
    pub text: String,
    pub score: f32,
    pub metadata: Payload,
}

/// Index builder for creating indices
pub struct IndexBuilder {
    storage: Arc<dyn Storage + Send + Sync>,
    embedding_function: Option<Arc<dyn EmbeddingFunction>>,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl IndexBuilder {
    pub fn new(storage: Arc<dyn Storage + Send + Sync>) -> Self {
        Self {
            storage,
            embedding_function: None,
            chunk_size: 1024,
            chunk_overlap: 200,
        }
    }
    
    pub fn with_embedding_function(mut self, ef: Arc<dyn EmbeddingFunction>) -> Self {
        self.embedding_function = Some(ef);
        self
    }
    
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }
    
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }
    
    pub async fn build_from_documents(
        &self,
        documents: Vec<String>,
        table_name: String,
    ) -> Result<VectorStoreIndex, StorageError> {
        let mut index = VectorStoreIndex::new(
            Arc::clone(&self.storage),
            table_name,
            self.embedding_function.clone(),
        ).await?;
        
        // Chunk documents
        let mut all_chunks = Vec::new();
        for doc in documents {
            let chunks = self.chunk_text(doc);
            all_chunks.extend(chunks);
        }
        
        // Add to index
        index.from_documents(all_chunks).await?;
        
        Ok(index)
    }
    
    fn chunk_text(&self, text: String) -> Vec<String> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let total = chars.len();
        
        let mut start = 0;
        while start < total {
            let end = (start + self.chunk_size).min(total);
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);
            
            if start + self.chunk_size >= total {
                break;
            }
            start += self.chunk_size - self.chunk_overlap;
        }
        
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_creation() {
        let node = Node::new("test text".to_string())
            .with_metadata("key".to_string(), serde_json::Value::String("value".to_string()))
            .add_relationship("parent".to_string(), "parent_id".to_string());
            
        assert_eq!(node.text, "test text");
        assert!(node.metadata.contains_key("key"));
        assert!(node.relationships.contains_key("parent"));
    }
    
    #[test]
    fn test_chunk_text() {
        let builder = IndexBuilder::new(Arc::new(crate::storage::memory::MemoryStorage::new()))
            .with_chunk_size(10)
            .with_chunk_overlap(2);
            
        let text = "This is a test document with multiple words".to_string();
        let chunks = builder.chunk_text(text);
        
        assert!(chunks.len() > 1);
        assert!(chunks[0].len() <= 10);
    }
}