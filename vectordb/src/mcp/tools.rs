//! MCP tool definitions and implementations

use super::*;
use serde_json::json;

/// Tool registry for managing available tools
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut registry = Self { tools: Vec::new() };
        registry.register_default_tools();
        registry
    }
    
    /// Register all default tools
    fn register_default_tools(&mut self) {
        // Vector operations
        self.register_vector_tools();
        
        // Table management
        self.register_table_tools();
        
        // Advanced search
        self.register_search_tools();
        
        // Analytics
        self.register_analytics_tools();
        
        // Import/Export
        self.register_data_tools();
    }
    
    /// Register vector operation tools
    fn register_vector_tools(&mut self) {
        // Batch insert tool
        self.tools.push(Tool {
            name: "batch_insert".to_string(),
            description: Some("Insert multiple vectors in a single operation".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Target table name"
                    },
                    "vectors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "vector": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                },
                                "metadata": {"type": "object"}
                            },
                            "required": ["id", "vector"]
                        },
                        "description": "Array of vectors to insert"
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size for processing",
                        "default": 100
                    }
                },
                "required": ["table", "vectors"]
            }),
        });
        
        // Update vector tool
        self.tools.push(Tool {
            name: "update_vector".to_string(),
            description: Some("Update an existing vector".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "id": {"type": "string"},
                    "vector": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "metadata": {"type": "object"},
                    "merge_metadata": {
                        "type": "boolean",
                        "description": "Merge with existing metadata",
                        "default": true
                    }
                },
                "required": ["table", "id"]
            }),
        });
        
        // Get vector tool
        self.tools.push(Tool {
            name: "get_vector".to_string(),
            description: Some("Retrieve a specific vector by ID".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "id": {"type": "string"},
                    "include_vector": {
                        "type": "boolean",
                        "default": true
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "default": true
                    }
                },
                "required": ["table", "id"]
            }),
        });
    }
    
    /// Register table management tools
    fn register_table_tools(&mut self) {
        // Clone table tool
        self.tools.push(Tool {
            name: "clone_table".to_string(),
            description: Some("Create a copy of an existing table".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source_table": {"type": "string"},
                    "target_table": {"type": "string"},
                    "include_data": {
                        "type": "boolean",
                        "default": true
                    },
                    "include_index": {
                        "type": "boolean",
                        "default": true
                    }
                },
                "required": ["source_table", "target_table"]
            }),
        });
        
        // Optimize table tool
        self.tools.push(Tool {
            name: "optimize_table".to_string(),
            description: Some("Optimize table for better performance".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "rebuild_index": {
                        "type": "boolean",
                        "default": false
                    },
                    "compact_storage": {
                        "type": "boolean",
                        "default": true
                    },
                    "analyze_distribution": {
                        "type": "boolean",
                        "default": true
                    }
                },
                "required": ["table"]
            }),
        });
        
        // Table statistics tool
        self.tools.push(Tool {
            name: "get_table_stats".to_string(),
            description: Some("Get detailed statistics about a table".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "include_distribution": {
                        "type": "boolean",
                        "default": false
                    },
                    "include_index_stats": {
                        "type": "boolean",
                        "default": true
                    }
                },
                "required": ["table"]
            }),
        });
    }
    
    /// Register advanced search tools
    fn register_search_tools(&mut self) {
        // KNN search tool
        self.tools.push(Tool {
            name: "knn_search".to_string(),
            description: Some("K-nearest neighbors search".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "vector": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of neighbors",
                        "minimum": 1,
                        "maximum": 1000
                    },
                    "filter": {"type": "object"},
                    "ef": {
                        "type": "integer",
                        "description": "HNSW search parameter",
                        "default": 64
                    }
                },
                "required": ["table", "vector", "k"]
            }),
        });
        
        // Range search tool
        self.tools.push(Tool {
            name: "range_search".to_string(),
            description: Some("Find vectors within a distance range".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "vector": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "radius": {
                        "type": "number",
                        "description": "Maximum distance",
                        "minimum": 0
                    },
                    "filter": {"type": "object"},
                    "limit": {
                        "type": "integer",
                        "default": 100
                    }
                },
                "required": ["table", "vector", "radius"]
            }),
        });
        
        // Multi-vector search tool
        self.tools.push(Tool {
            name: "multi_vector_search".to_string(),
            description: Some("Search with multiple query vectors".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "vectors": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"}
                        },
                        "description": "Multiple query vectors"
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": ["mean", "max", "min", "weighted"],
                        "default": "mean"
                    },
                    "weights": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Weights for weighted aggregation"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10
                    }
                },
                "required": ["table", "vectors"]
            }),
        });
        
        // Recommendation tool
        self.tools.push(Tool {
            name: "recommend".to_string(),
            description: Some("Get recommendations based on positive and negative examples".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "positive_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of positive examples"
                    },
                    "negative_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of negative examples"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10
                    },
                    "filter": {"type": "object"}
                },
                "required": ["table", "positive_ids"]
            }),
        });
    }
    
    /// Register analytics tools
    fn register_analytics_tools(&mut self) {
        // Clustering tool
        self.tools.push(Tool {
            name: "cluster_vectors".to_string(),
            description: Some("Perform clustering on vectors".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "algorithm": {
                        "type": "string",
                        "enum": ["kmeans", "dbscan", "hierarchical"],
                        "default": "kmeans"
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of clusters (for kmeans)",
                        "minimum": 2
                    },
                    "eps": {
                        "type": "number",
                        "description": "DBSCAN epsilon parameter"
                    },
                    "min_samples": {
                        "type": "integer",
                        "description": "DBSCAN min_samples parameter"
                    },
                    "filter": {"type": "object"}
                },
                "required": ["table", "algorithm"]
            }),
        });
        
        // Diversity analysis tool
        self.tools.push(Tool {
            name: "analyze_diversity".to_string(),
            description: Some("Analyze diversity of vectors in a table".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "sample_size": {
                        "type": "integer",
                        "description": "Number of vectors to sample",
                        "default": 1000
                    },
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["entropy", "coverage", "uniformity", "pairwise_distance"]
                        },
                        "default": ["entropy", "coverage"]
                    }
                },
                "required": ["table"]
            }),
        });
        
        // Anomaly detection tool
        self.tools.push(Tool {
            name: "detect_anomalies".to_string(),
            description: Some("Detect anomalous vectors".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["isolation_forest", "lof", "statistical"],
                        "default": "isolation_forest"
                    },
                    "contamination": {
                        "type": "number",
                        "description": "Expected proportion of anomalies",
                        "minimum": 0,
                        "maximum": 0.5,
                        "default": 0.1
                    },
                    "filter": {"type": "object"}
                },
                "required": ["table"]
            }),
        });
    }
    
    /// Register data import/export tools
    fn register_data_tools(&mut self) {
        // Import from file tool
        self.tools.push(Tool {
            name: "import_data".to_string(),
            description: Some("Import vectors from file".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "file_path": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["json", "csv", "parquet", "numpy"],
                        "default": "json"
                    },
                    "id_column": {"type": "string"},
                    "vector_column": {"type": "string"},
                    "metadata_columns": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "batch_size": {
                        "type": "integer",
                        "default": 1000
                    }
                },
                "required": ["table", "file_path"]
            }),
        });
        
        // Export to file tool
        self.tools.push(Tool {
            name: "export_data".to_string(),
            description: Some("Export vectors to file".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "file_path": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["json", "csv", "parquet", "numpy"],
                        "default": "json"
                    },
                    "filter": {"type": "object"},
                    "include_vectors": {
                        "type": "boolean",
                        "default": true
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "default": true
                    },
                    "limit": {"type": "integer"}
                },
                "required": ["table", "file_path"]
            }),
        });
        
        // Generate embeddings tool
        self.tools.push(Tool {
            name: "generate_embeddings".to_string(),
            description: Some("Generate embeddings for text using built-in models".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Text to embed"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["all-minilm-l6-v2", "all-mpnet-base-v2", "bge-small", "bge-large"],
                        "default": "all-minilm-l6-v2"
                    },
                    "normalize": {
                        "type": "boolean",
                        "default": true
                    }
                },
                "required": ["texts"]
            }),
        });
    }
    
    /// Get all registered tools
    pub fn get_tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }
    
    /// Find tool by name
    pub fn find_tool(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }
    
    /// Register custom tool
    pub fn register_tool(&mut self, tool: Tool) {
        self.tools.push(tool);
    }
}

/// Extended tool capabilities
pub struct ExtendedToolCapabilities {
    pub batch_operations: bool,
    pub streaming: bool,
    pub cancellation: bool,
    pub progress_reporting: bool,
}

impl Default for ExtendedToolCapabilities {
    fn default() -> Self {
        Self {
            batch_operations: true,
            streaming: true,
            cancellation: true,
            progress_reporting: true,
        }
    }
}