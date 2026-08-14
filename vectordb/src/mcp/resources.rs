//! MCP resource management for exposing vector database entities

use super::*;
use crate::storage::traits::_Storage as Storage;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Resource manager for MCP protocol
#[allow(dead_code)]
pub struct ResourceManager<S: Storage> {
    storage: Arc<S>,
    resources: Arc<RwLock<Vec<Resource>>>,
    subscriptions: Arc<RwLock<HashMap<String, Vec<Uuid>>>>,
}

impl<S: Storage> ResourceManager<S> {
    pub fn new(storage: Arc<S>) -> Self {
        Self {
            storage,
            resources: Arc::new(RwLock::new(Vec::new())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// List all available resources
    pub async fn list_resources(&self) -> Result<Vec<Resource>, Box<dyn std::error::Error>> {
        let mut resources = Vec::new();
        
        // Add table resources
        let tables = self.storage.list_tables().await?;
        for table in tables {
            resources.push(Resource {
                uri: format!("vectordb://table/{}", table.name),
                name: table.name.clone(),
                description: Some(format!("Vector table: {}", table.name)),
                mime_type: Some("application/json".to_string()),
            });
            
            // Add sub-resources for table
            resources.push(Resource {
                uri: format!("vectordb://table/{}/schema", table),
                name: format!("{} Schema", table),
                description: Some(format!("Schema definition for table {}", table)),
                mime_type: Some("application/json".to_string()),
            });
            
            resources.push(Resource {
                uri: format!("vectordb://table/{}/stats", table),
                name: format!("{} Statistics", table),
                description: Some(format!("Statistics for table {}", table)),
                mime_type: Some("application/json".to_string()),
            });
            
            resources.push(Resource {
                uri: format!("vectordb://table/{}/sample", table),
                name: format!("{} Sample", table),
                description: Some(format!("Sample vectors from table {}", table)),
                mime_type: Some("application/json".to_string()),
            });
        }
        
        // Add system resources
        resources.push(Resource {
            uri: "vectordb://system/info".to_string(),
            name: "System Information".to_string(),
            description: Some("VectorDB system information and capabilities".to_string()),
            mime_type: Some("application/json".to_string()),
        });
        
        resources.push(Resource {
            uri: "vectordb://system/metrics".to_string(),
            name: "System Metrics".to_string(),
            description: Some("Performance metrics and statistics".to_string()),
            mime_type: Some("application/json".to_string()),
        });
        
        resources.push(Resource {
            uri: "vectordb://system/models".to_string(),
            name: "Available Models".to_string(),
            description: Some("List of available embedding models".to_string()),
            mime_type: Some("application/json".to_string()),
        });
        
        // Add documentation resources
        resources.push(Resource {
            uri: "vectordb://docs/quickstart".to_string(),
            name: "Quick Start Guide".to_string(),
            description: Some("Getting started with DXMT VectorDB".to_string()),
            mime_type: Some("text/markdown".to_string()),
        });
        
        resources.push(Resource {
            uri: "vectordb://docs/api".to_string(),
            name: "API Documentation".to_string(),
            description: Some("Complete API reference".to_string()),
            mime_type: Some("text/markdown".to_string()),
        });
        
        resources.push(Resource {
            uri: "vectordb://docs/examples".to_string(),
            name: "Example Queries".to_string(),
            description: Some("Common query patterns and examples".to_string()),
            mime_type: Some("application/json".to_string()),
        });
        
        Ok(resources)
    }
    
    /// Read a specific resource
    pub async fn read_resource(
        &self,
        uri: &str,
    ) -> Result<ResourceContent, Box<dyn std::error::Error>> {
        if !uri.starts_with("vectordb://") {
            return Err("Invalid resource URI".into());
        }
        
        let path = uri.strip_prefix("vectordb://").unwrap();
        let parts: Vec<&str> = path.split('/').collect();
        
        match parts[0] {
            "table" => self.read_table_resource(&parts[1..]).await,
            "system" => self.read_system_resource(&parts[1..]).await,
            "docs" => self.read_docs_resource(&parts[1..]).await,
            _ => Err(format!("Unknown resource type: {}", parts[0]).into()),
        }
    }
    
    /// Read table-related resources
    async fn read_table_resource(
        &self,
        parts: &[&str],
    ) -> Result<ResourceContent, Box<dyn std::error::Error>> {
        if parts.is_empty() {
            return Err("Missing table name".into());
        }
        
        let table_name = parts[0];
        
        let content = if parts.len() == 1 {
            // Return table info
            let info = self.storage.get_table_info(table_name).await?;
            serde_json::to_string_pretty(&info)?
        } else {
            match parts[1] {
                "schema" => {
                    // Return table schema
                    let schema = self.storage.get_table_schema(table_name).await?;
                    serde_json::to_string_pretty(&schema)?
                }
                "stats" => {
                    // Return table statistics
                    let stats = self.storage.get_table_stats(table_name).await?;
                    serde_json::to_string_pretty(&stats)?
                }
                "sample" => {
                    // Return sample vectors
                    let samples = self.storage.get_sample_vectors(table_name, 10).await?;
                    serde_json::to_string_pretty(&samples)?
                }
                _ => return Err(format!("Unknown table resource: {}", parts[1]).into()),
            }
        };
        
        Ok(ResourceContent::Text {
            uri: format!("vectordb://table/{}", parts.join("/")),
            mime_type: Some("application/json".to_string()),
            text: content,
        })
    }
    
    /// Read system resources
    async fn read_system_resource(
        &self,
        parts: &[&str],
    ) -> Result<ResourceContent, Box<dyn std::error::Error>> {
        if parts.is_empty() {
            return Err("Missing system resource name".into());
        }
        
        let content = match parts[0] {
            "info" => {
                serde_json::to_string_pretty(&serde_json::json!({
                    "name": "DXMT VectorDB",
                    "version": env!("CARGO_PKG_VERSION"),
                    "features": {
                        "mcp_protocol": true,
                        "hybrid_search": true,
                        "gpu_acceleration": cfg!(feature = "gpu"),
                        "simd_optimization": cfg!(feature = "simd"),
                        "distributed": true,
                        "multi_tenancy": true
                    },
                    "storage_backends": ["memory", "ssd", "s3", "gcs", "azure"],
                    "index_algorithms": ["hnsw", "ivf", "ivf_pq", "diskann", "scann", "brute_force"],
                    "distance_metrics": ["cosine", "euclidean", "dot_product", "manhattan"],
                    "embedding_models": [
                        "all-minilm-l6-v2",
                        "all-mpnet-base-v2",
                        "bge-small",
                        "bge-large"
                    ]
                }))?
            }
            "metrics" => {
                let metrics = self.storage.get_metrics().await?;
                serde_json::to_string_pretty(&metrics)?
            }
            "models" => {
                serde_json::to_string_pretty(&serde_json::json!([
                    {
                        "name": "all-minilm-l6-v2",
                        "dimension": 384,
                        "description": "Fast, lightweight model for general text",
                        "use_cases": ["semantic search", "clustering", "classification"]
                    },
                    {
                        "name": "all-mpnet-base-v2",
                        "dimension": 768,
                        "description": "High-quality embeddings for various tasks",
                        "use_cases": ["semantic search", "similarity", "paraphrase detection"]
                    },
                    {
                        "name": "bge-small",
                        "dimension": 384,
                        "description": "Bilingual model (English and Chinese)",
                        "use_cases": ["cross-lingual search", "multilingual applications"]
                    },
                    {
                        "name": "bge-large",
                        "dimension": 1024,
                        "description": "Large bilingual model for best quality",
                        "use_cases": ["high-precision search", "complex similarity tasks"]
                    }
                ]))?
            }
            _ => return Err(format!("Unknown system resource: {}", parts[0]).into()),
        };
        
        Ok(ResourceContent::Text {
            uri: format!("vectordb://system/{}", parts[0]),
            mime_type: Some("application/json".to_string()),
            text: content,
        })
    }
    
    /// Read documentation resources
    async fn read_docs_resource(
        &self,
        parts: &[&str],
    ) -> Result<ResourceContent, Box<dyn std::error::Error>> {
        if parts.is_empty() {
            return Err("Missing documentation resource name".into());
        }
        
        let content = match parts[0] {
            "quickstart" => {
                r#"# DXMT VectorDB Quick Start Guide

## Installation

```bash
# Using Docker
docker run -p 9999:9999 dxmt/vectordb:latest

# From source
cargo install dxmt-vectordb
vectordb serve
```

## Basic Usage

### 1. Create a Table

```json
{
  "name": "my_vectors",
  "dimension": 384,
  "metric": "cosine",
  "index_type": "hnsw"
}
```

### 2. Insert Vectors

```json
{
  "table": "my_vectors",
  "vectors": [
    {
      "id": "vec1",
      "vector": [0.1, 0.2, ...],
      "metadata": {"text": "Hello world"}
    }
  ]
}
```

### 3. Search Similar Vectors

```json
{
  "table": "my_vectors",
  "vector": [0.15, 0.25, ...],
  "limit": 10
}
```

## MCP Integration

The MCP protocol allows AI assistants to interact with the database:

1. Connect via WebSocket: `ws://localhost:9999/mcp`
2. Initialize the session
3. Use tools like `vector_search`, `insert_vector`, etc.

## Next Steps

- Explore advanced search features
- Set up distributed deployment
- Configure custom embedding models
- Enable GPU acceleration
"#.to_string()
            }
            "api" => {
                r#"# API Reference

## REST Endpoints

### Tables

- `POST /tables` - Create a new table
- `GET /tables` - List all tables
- `GET /tables/{name}` - Get table information
- `DELETE /tables/{name}` - Delete a table

### Vectors

- `POST /tables/{name}/vectors` - Insert vectors
- `GET /tables/{name}/vectors/{id}` - Get vector by ID
- `PUT /tables/{name}/vectors/{id}` - Update vector
- `DELETE /tables/{name}/vectors/{id}` - Delete vector

### Search

- `POST /tables/{name}/search` - Vector similarity search
- `POST /tables/{name}/hybrid-search` - Hybrid search
- `POST /tables/{name}/knn` - K-nearest neighbors
- `POST /tables/{name}/range-search` - Range search

## GraphQL API

Connect to `/graphql` endpoint for GraphQL queries.

## MCP Protocol

WebSocket endpoint: `/mcp`

Supported methods:
- `initialize`
- `tools/list`
- `tools/call`
- `resources/list`
- `resources/read`
- `prompts/list`
- `prompts/get`
"#.to_string()
            }
            "examples" => {
                serde_json::to_string_pretty(&serde_json::json!({
                    "similarity_search": {
                        "description": "Find similar documents",
                        "query": {
                            "table": "documents",
                            "vector": "[0.1, 0.2, ...]",
                            "limit": 10,
                            "filter": {"category": "tech"}
                        }
                    },
                    "hybrid_search": {
                        "description": "Combine vector and text search",
                        "query": {
                            "table": "articles",
                            "vector": "[0.1, 0.2, ...]",
                            "text": "machine learning",
                            "vector_weight": 0.7,
                            "limit": 20
                        }
                    },
                    "recommendation": {
                        "description": "Get recommendations based on examples",
                        "query": {
                            "table": "products",
                            "positive_ids": ["prod1", "prod2"],
                            "negative_ids": ["prod3"],
                            "limit": 5
                        }
                    },
                    "clustering": {
                        "description": "Cluster vectors",
                        "query": {
                            "table": "embeddings",
                            "algorithm": "kmeans",
                            "n_clusters": 5
                        }
                    }
                }))?
            }
            _ => return Err(format!("Unknown documentation resource: {}", parts[0]).into()),
        };
        
        let mime_type = if parts[0] == "examples" {
            "application/json"
        } else {
            "text/markdown"
        };
        
        Ok(ResourceContent::Text {
            uri: format!("vectordb://docs/{}", parts[0]),
            mime_type: Some(mime_type.to_string()),
            text: content,
        })
    }
    
    /// Subscribe to resource updates
    pub async fn subscribe(
        &self,
        uri: &str,
        session_id: Uuid,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut subscriptions = self.subscriptions.write().await;
        subscriptions
            .entry(uri.to_string())
            .or_insert_with(Vec::new)
            .push(session_id);
        Ok(())
    }
    
    /// Unsubscribe from resource updates
    pub async fn unsubscribe(
        &self,
        uri: &str,
        session_id: Uuid,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut subscriptions = self.subscriptions.write().await;
        if let Some(subs) = subscriptions.get_mut(uri) {
            subs.retain(|&id| id != session_id);
            if subs.is_empty() {
                subscriptions.remove(uri);
            }
        }
        Ok(())
    }
    
    /// Notify subscribers of resource changes
    pub async fn notify_change(&self, uri: &str) {
        let subscriptions = self.subscriptions.read().await;
        if let Some(subscribers) = subscriptions.get(uri) {
            for &session_id in subscribers {
                // Send notification to session
                // This would integrate with the WebSocket handler
                tracing::debug!("Notifying session {} of change to {}", session_id, uri);
            }
        }
    }
}

/// Resource change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceChangeEvent {
    pub uri: String,
    pub change_type: ResourceChangeType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResourceChangeType {
    Created,
    Updated,
    Deleted,
}