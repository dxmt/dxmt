//! MCP server implementation for handling AI assistant connections

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::*;
use crate::storage::traits::_Storage as Storage;
use crate::types::{
    record::{Filter, SearchQuery},
    vector::DenseVector,
};
use axum::{
    extract::{ws::WebSocket, State, WebSocketUpgrade},
    response::Response,
};
use futures::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

/// Type alias for Send+Sync error types
type McpServerError = Box<dyn std::error::Error + Send + Sync>;

/// MCP server state
pub struct McpServer<S: Storage> {
    storage: Arc<S>,
    sessions: Arc<RwLock<HashMap<Uuid, SessionState>>>,
    tools: Arc<RwLock<Vec<Tool>>>,
    resources: Arc<RwLock<Vec<Resource>>>,
    prompts: Arc<RwLock<Vec<Prompt>>>,
}

/// Session state for connected clients
#[derive(Debug, Clone)]
struct SessionState {
    id: Uuid,
    client_info: Option<ClientInfo>,
    capabilities: Option<ClientCapabilities>,
    initialized: bool,
}

impl<S: Storage + 'static> McpServer<S> {
    pub fn new(storage: Arc<S>) -> Self {
        let server = Self {
            storage,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            tools: Arc::new(RwLock::new(Vec::new())),
            resources: Arc::new(RwLock::new(Vec::new())),
            prompts: Arc::new(RwLock::new(Vec::new())),
        };

        // The catalogue is filled by `initialize`, not here. This constructor used to
        // `tokio::spawn` a task that pushed a single tool, which raced every caller: a
        // tools/list arriving first saw an empty or half-filled list, and the other seven tools
        // defined in `initialize_tools` were never registered at all because nothing called it.
        server
    }

    /// Populate the tool and prompt catalogues. Idempotent.
    ///
    /// Must run before the first `tools/list`, so both transports call it up front rather than
    /// leaving it to a background task.
    pub async fn initialize(&self) {
        self.tools.write().await.clear();
        self.prompts.write().await.clear();
        self.initialize_tools().await;
        self.initialize_prompts().await;
        info!(
            "MCP catalogue ready: {} tools, {} prompts",
            self.tools.read().await.len(),
            self.prompts.read().await.len()
        );
    }

    /// Initialize built-in tools
    async fn initialize_tools(&self) {
        let mut tools = self.tools.write().await;

        // Vector search tool
        tools.push(Tool {
            name: "vector_search".to_string(),
            description: Some("Search for similar vectors in a table".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Name of the table to search"
                    },
                    "vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Query vector"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    },
                    "filter": {
                        "type": "object",
                        "description": "Optional metadata filters"
                    }
                },
                "required": ["table", "vector"]
            }),
        });

        // Insert vector tool
        tools.push(Tool {
            name: "insert_vector".to_string(),
            description: Some("Insert a vector with metadata into a table".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Name of the table"
                    },
                    "id": {
                        "type": "string",
                        "description": "Unique identifier for the vector"
                    },
                    "vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Vector data"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Associated metadata"
                    }
                },
                "required": ["table", "id", "vector"]
            }),
        });

        // Create table tool
        tools.push(Tool {
            name: "create_table".to_string(),
            description: Some("Create a new vector table".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Table name"
                    },
                    "dimension": {
                        "type": "integer",
                        "description": "Vector dimension"
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["cosine", "euclidean", "dot_product"],
                        "description": "Distance metric",
                        "default": "cosine"
                    },
                    "index_type": {
                        "type": "string",
                        "enum": ["hnsw", "ivf", "brute_force"],
                        "description": "Index algorithm",
                        "default": "hnsw"
                    }
                },
                "required": ["name", "dimension"]
            }),
        });

        // List tables tool
        tools.push(Tool {
            name: "list_tables".to_string(),
            description: Some("List all available tables".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        });

        // Get table info tool
        tools.push(Tool {
            name: "get_table_info".to_string(),
            description: Some("Get detailed information about a table".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Table name"
                    }
                },
                "required": ["name"]
            }),
        });

        // Delete vectors tool
        tools.push(Tool {
            name: "delete_vectors".to_string(),
            description: Some("Delete vectors by ID or filter".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name"
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Vector IDs to delete"
                    },
                    "filter": {
                        "type": "object",
                        "description": "Filter for batch deletion"
                    }
                },
                "required": ["table"],
                "oneOf": [
                    {"required": ["ids"]},
                    {"required": ["filter"]}
                ]
            }),
        });

        // Hybrid search tool
        tools.push(Tool {
            name: "hybrid_search".to_string(),
            description: Some("Perform hybrid vector + text search".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name"
                    },
                    "vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Query vector"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text query"
                    },
                    "vector_weight": {
                        "type": "number",
                        "description": "Weight for vector similarity (0-1)",
                        "default": 0.7
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 10
                    }
                },
                "required": ["table"]
            }),
        });
    }

    /// Initialize built-in prompts
    async fn initialize_prompts(&self) {
        let mut prompts = self.prompts.write().await;

        // RAG prompt
        prompts.push(Prompt {
            name: "rag_query".to_string(),
            description: Some("Retrieval-augmented generation query".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "question".to_string(),
                    description: Some("User's question".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "context_limit".to_string(),
                    description: Some("Maximum number of context documents".to_string()),
                    required: Some(false),
                },
            ]),
        });

        // Similarity analysis prompt
        prompts.push(Prompt {
            name: "similarity_analysis".to_string(),
            description: Some("Analyze similarity between items".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "items".to_string(),
                    description: Some("Items to compare".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "metric".to_string(),
                    description: Some("Similarity metric to use".to_string()),
                    required: Some(false),
                },
            ]),
        });

        // Data ingestion prompt
        prompts.push(Prompt {
            name: "data_ingestion".to_string(),
            description: Some("Guide for ingesting data into vector database".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "data_source".to_string(),
                    description: Some("Type of data source".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "format".to_string(),
                    description: Some("Data format".to_string()),
                    required: Some(false),
                },
            ]),
        });
    }

    /// Handle MCP WebSocket connection
    /// Serve MCP over stdin/stdout, one JSON-RPC message per line.
    ///
    /// This is the transport MCP clients actually use — Claude Desktop, Cline and the
    /// inspector all launch the server as a subprocess and speak newline-delimited JSON over
    /// its pipes. The WebSocket path below existed but nothing could start it, because the CLI
    /// had no subcommand for MCP at all.
    ///
    /// **Nothing else may write to stdout while this runs.** The caller is responsible for
    /// sending logs to stderr; a stray log line lands in the middle of the JSON-RPC stream and
    /// the client drops the connection with a parse error that points nowhere useful.
    pub async fn serve_stdio(&self) -> Result<(), McpServerError> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        self.initialize().await;

        let session_id = Uuid::new_v4();
        self.sessions.write().await.insert(
            session_id,
            SessionState {
                id: session_id,
                client_info: None,
                capabilities: None,
                initialized: false,
            },
        );

        info!("MCP stdio session {} started", session_id);

        let mut lines = BufReader::new(tokio::io::stdin()).lines();
        let mut stdout = tokio::io::stdout();

        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            let reply = match self.handle_message(session_id, &line).await {
                // Notifications produce no response; writing an empty line would be a protocol
                // violation, so swallow it.
                Ok(response) if response.is_empty() => continue,
                Ok(response) => response,
                Err(e) => {
                    error!("MCP request failed: {}", e);
                    serde_json::to_string(&serde_json::json!({
                        "jsonrpc": "2.0",
                        "error": { "code": -32603, "message": e.to_string() }
                    }))?
                }
            };

            stdout.write_all(reply.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            // Flush per message: the client blocks waiting for this reply, so buffering it
            // until the pipe fills would look exactly like a hung server.
            stdout.flush().await?;
        }

        self.sessions.write().await.remove(&session_id);
        info!("MCP stdio session {} closed", session_id);
        Ok(())
    }

    pub async fn handle_connection(&self, ws: WebSocket) {
        self.initialize().await;

        let session_id = Uuid::new_v4();
        let session = SessionState {
            id: session_id,
            client_info: None,
            capabilities: None,
            initialized: false,
        };

        self.sessions
            .write()
            .await
            .insert(session_id, session.clone());

        let (mut sender, mut receiver) = ws.split();

        info!("New MCP connection established: {}", session_id);

        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(msg) => {
                    if let Ok(text) = msg.to_text() {
                        match self.handle_message(session_id, text).await {
                            Ok(response) => {
                                if let Err(e) = sender.send(response.into()).await {
                                    error!("Failed to send response: {}", e);
                                    break;
                                }
                            }
                            Err(e) => {
                                error!("Error handling message: {}", e);
                                let error_response = serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "error": {
                                        "code": -32603,
                                        "message": e.to_string()
                                    }
                                });
                                if let Ok(text) = serde_json::to_string(&error_response) {
                                    let _ = sender.send(text.into()).await;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
            }
        }

        self.sessions.write().await.remove(&session_id);
        info!("MCP connection closed: {}", session_id);
    }

    /// Handle incoming MCP message
    async fn handle_message(
        &self,
        session_id: Uuid,
        message: &str,
    ) -> Result<String, McpServerError> {
        let json_rpc: JsonRpcMessage = serde_json::from_str(message)?;

        let response = match json_rpc.content {
            JsonRpcContent::Request { method, params } => {
                self.handle_request(session_id, &method, params).await?
            }
            JsonRpcContent::Notification { method, params } => {
                self.handle_notification(session_id, &method, params).await;
                return Ok(String::new()); // No response for notifications
            }
            _ => {
                return Err("Invalid message type".into());
            }
        };

        let response_msg = JsonRpcMessage {
            jsonrpc: "2.0".to_string(),
            id: json_rpc.id,
            content: JsonRpcContent::Response {
                result: Some(serde_json::to_value(response)?),
                error: None,
            },
        };

        Ok(serde_json::to_string(&response_msg)?)
    }

    /// Handle MCP request
    async fn handle_request(
        &self,
        session_id: Uuid,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<McpResponse, McpServerError> {
        match method {
            "initialize" => {
                let params: InitializeParams =
                    serde_json::from_value(params.ok_or("Missing initialize params")?)?;
                Ok(McpResponse::Initialize(
                    self.handle_initialize(session_id, params).await?,
                ))
            }
            "tools/list" => Ok(McpResponse::Tools(self.handle_list_tools().await?)),
            "tools/call" => {
                let params: CallToolParams =
                    serde_json::from_value(params.ok_or("Missing tool call params")?)?;
                Ok(McpResponse::ToolCall(self.handle_call_tool(params).await?))
            }
            "resources/list" => Ok(McpResponse::Resources(self.handle_list_resources().await?)),
            "resources/read" => {
                let params: ReadResourceParams =
                    serde_json::from_value(params.ok_or("Missing resource read params")?)?;
                Ok(McpResponse::Resource(
                    self.handle_read_resource(params).await?,
                ))
            }
            "prompts/list" => Ok(McpResponse::Prompts(self.handle_list_prompts().await?)),
            "prompts/get" => {
                let params: GetPromptParams =
                    serde_json::from_value(params.ok_or("Missing prompt params")?)?;
                Ok(McpResponse::Prompt(self.handle_get_prompt(params).await?))
            }
            _ => Err(format!("Unknown method: {}", method).into()),
        }
    }

    /// Handle notification
    async fn handle_notification(
        &self,
        session_id: Uuid,
        method: &str,
        params: Option<serde_json::Value>,
    ) {
        debug!("Received notification: {} from {}", method, session_id);
        // Handle notifications (logging, events, etc.)
    }

    /// Handle initialize request
    async fn handle_initialize(
        &self,
        session_id: Uuid,
        params: InitializeParams,
    ) -> Result<InitializeResult, McpServerError> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.client_info = Some(params.client_info);
            session.capabilities = Some(params.capabilities);
            session.initialized = true;
        }

        Ok(InitializeResult {
            protocol_version: MCP_VERSION.to_string(),
            capabilities: ServerInfo::default().capabilities,
            server_info: ServerInfo::default(),
        })
    }

    /// Handle list tools request
    async fn handle_list_tools(&self) -> Result<ListToolsResult, McpServerError> {
        let tools = self.tools.read().await;
        Ok(ListToolsResult {
            tools: tools.clone(),
        })
    }

    /// Handle tool call
    async fn handle_call_tool(
        &self,
        params: CallToolParams,
    ) -> Result<CallToolResult, McpServerError> {
        match params.name.as_str() {
            "vector_search" => self.handle_vector_search(params.arguments).await,
            "insert_vector" => self.handle_insert_vector(params.arguments).await,
            "create_table" => self.handle_create_table(params.arguments).await,
            "list_tables" => self.handle_list_tables().await,
            "get_table_info" => self.handle_get_table_info(params.arguments).await,
            "delete_vectors" => self.handle_delete_vectors(params.arguments).await,
            "hybrid_search" => self.handle_hybrid_search(params.arguments).await,
            _ => Err(format!("Unknown tool: {}", params.name).into()),
        }
    }

    /// Handle vector search tool
    async fn handle_vector_search(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let table_name = args["table"].as_str().ok_or("Missing table name")?;
        let vector: Vec<f32> = serde_json::from_value(args["vector"].clone())?;
        let limit = args["limit"].as_u64().unwrap_or(10) as usize;

        // Parse filter if provided
        let filter = if let Some(filter_json) = args.get("filter") {
            Some(
                serde_json::from_value::<Filter>(filter_json.clone())
                    .map_err(|e| format!("Invalid filter format: {}", e))?,
            )
        } else {
            None
        };

        // Perform vector search
        let query = SearchQuery {
            vector: DenseVector::from(vector),
            top_k: limit,
            filter,
            ..Default::default()
        };

        let results = self.storage.search(table_name, query).await?;

        let response_text = serde_json::to_string_pretty(&results)?;

        Ok(CallToolResult {
            content: vec![ToolContent::Text {
                text: response_text,
            }],
            is_error: Some(false),
        })
    }

    /// Handle insert vector tool
    async fn handle_insert_vector(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let table_name = args["table"].as_str().ok_or("Missing table name")?;
        let id = args["id"].as_str().ok_or("Missing vector ID")?;
        let vector: Vec<f32> = serde_json::from_value(args["vector"].clone())?;
        let metadata = args.get("metadata").cloned();

        // Insert vector
        let record = crate::types::record::CreateRecordRequest {
            id: id.to_string(),
            vector,
            sparse_vector: None,
            payload: metadata,
            named_vectors: None,
            text: None,
        };
        self.storage.insert(table_name, record).await?;

        Ok(CallToolResult {
            content: vec![ToolContent::Text {
                text: format!(
                    "Successfully inserted vector '{}' into table '{}'",
                    id, table_name
                ),
            }],
            is_error: Some(false),
        })
    }

    /// Handle create table tool
    async fn handle_create_table(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let name = args["name"].as_str().ok_or("Missing table name")?;
        let dimension = args["dimension"].as_u64().ok_or("Missing dimension")? as usize;
        let metric = args
            .get("metric")
            .and_then(|m| m.as_str())
            .unwrap_or("cosine");
        let index_type = args
            .get("index_type")
            .and_then(|i| i.as_str())
            .unwrap_or("hnsw");

        // The rules `POST /tables` applies. The name becomes a file name in disk mode, and this
        // tool accepted any string.
        crate::utils::validate_table_name(name)?;
        // Create table
        let table_config = crate::types::table::TableConfig {
            name: name.to_string(),
            dimension,
            distance_metric: match metric {
                "euclidean" => crate::types::vector::DistanceMetric::Euclidean,
                "dot_product" => crate::types::vector::DistanceMetric::Dot,
                _ => crate::types::vector::DistanceMetric::Cosine,
            },
            description: None,
            index_type: match index_type {
                "ivf" => Some(crate::types::table::IndexType::Ivf),
                "lsh" => Some(crate::types::table::IndexType::Lsh),
                _ => Some(crate::types::table::IndexType::Hnsw),
            },
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None, // Use global config
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
        };

        self.storage.create_table(table_config).await?;

        Ok(CallToolResult {
            content: vec![ToolContent::Text {
                text: format!(
                    "Successfully created table '{}' with dimension {} and {} metric",
                    name, dimension, metric
                ),
            }],
            is_error: Some(false),
        })
    }

    /// Handle list tables tool
    async fn handle_list_tables(&self) -> Result<CallToolResult, McpServerError> {
        let tables = self.storage.list_tables().await?;
        let response_text = serde_json::to_string_pretty(&tables)?;

        Ok(CallToolResult {
            content: vec![ToolContent::Text {
                text: response_text,
            }],
            is_error: Some(false),
        })
    }

    /// Handle get table info tool
    async fn handle_get_table_info(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let name = args["name"].as_str().ok_or("Missing table name")?;

        let info = self.storage.get_table_info(name).await?;
        let response_text = serde_json::to_string_pretty(&info)?;

        Ok(CallToolResult {
            content: vec![ToolContent::Text {
                text: response_text,
            }],
            is_error: Some(false),
        })
    }

    /// Handle delete vectors tool
    async fn handle_delete_vectors(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let table_name = args["table"].as_str().ok_or("Missing table name")?;

        if let Some(ids) = args.get("ids") {
            let ids: Vec<String> = serde_json::from_value(ids.clone())?;
            let count = ids.len();
            self.storage.delete_by_ids(table_name, ids).await?;
            Ok(CallToolResult {
                content: vec![ToolContent::Text {
                    text: format!(
                        "Successfully deleted {} vectors from table '{}'",
                        count, table_name
                    ),
                }],
                is_error: Some(false),
            })
        } else if let Some(filter) = args.get("filter") {
            self.storage
                .delete_by_filter(table_name, filter.clone())
                .await?;
            Ok(CallToolResult {
                content: vec![ToolContent::Text {
                    text: format!(
                        "Successfully deleted vectors from table '{}' matching filter",
                        table_name
                    ),
                }],
                is_error: Some(false),
            })
        } else {
            Err("Must provide either 'ids' or 'filter'".into())
        }
    }

    /// Handle hybrid search tool
    async fn handle_hybrid_search(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let table_name = args["table"].as_str().ok_or("Missing table name")?;

        let vector = args
            .get("vector")
            .map(|v| serde_json::from_value::<Vec<f32>>(v.clone()))
            .transpose()?;
        let text = args.get("text").and_then(|t| t.as_str());
        let vector_weight = args
            .get("vector_weight")
            .and_then(|w| w.as_f64())
            .unwrap_or(0.7) as f32;
        let limit = args.get("limit").and_then(|l| l.as_u64()).unwrap_or(10) as usize;

        if vector.is_none() && text.is_none() {
            return Err("Must provide either 'vector' or 'text' for hybrid search".into());
        }

        // Parse filter if provided
        let filter = if let Some(filter_json) = args.get("filter") {
            Some(
                serde_json::from_value::<Filter>(filter_json.clone())
                    .map_err(|e| format!("Invalid filter format: {}", e))?,
            )
        } else {
            None
        };

        // For text-only queries, perform a metadata-only scan by using a zero vector
        // with a large top_k and then filter results by payload relevance (best effort).
        // Full BM25 text search requires the full_text_search() method on
        // IntelligentIndexedStorage, which is not accessible through the generic Storage trait.
        let query = if let Some(v) = vector {
            crate::types::record::SearchQuery {
                vector: DenseVector::from(v),
                top_k: limit,
                filter,
                with_payload: true,
                with_vector: false,
                limit,
                ..Default::default()
            }
        } else {
            // Text-only: return an empty result set with a note — the storage trait
            // does not expose full-text search generically.
            let result = serde_json::json!({
                "hits": [],
                "total": 0,
                "note": format!(
                    "Text-only search for '{}' requires full-text search to be enabled on table '{}'. Use the REST API endpoint POST /tables/{}/search/text instead.",
                    text.unwrap_or(""), table_name, table_name
                )
            });
            return Ok(CallToolResult {
                content: vec![ToolContent::Text {
                    text: serde_json::to_string_pretty(&result)?,
                }],
                is_error: Some(false),
            });
        };
        let results = self.storage.hybrid_search(table_name, query).await?;

        let response_text = serde_json::to_string_pretty(&results)?;

        Ok(CallToolResult {
            content: vec![ToolContent::Text {
                text: response_text,
            }],
            is_error: Some(false),
        })
    }

    /// Handle list resources
    async fn handle_list_resources(&self) -> Result<ListResourcesResult, McpServerError> {
        let tables = self.storage.list_tables().await?;

        let resources: Vec<Resource> = tables
            .iter()
            .map(|table| Resource {
                uri: format!("vectordb://table/{}", table.name),
                name: table.name.clone(),
                description: Some(format!("Vector table: {}", table.name)),
                mime_type: Some("application/json".to_string()),
            })
            .collect();

        Ok(ListResourcesResult { resources })
    }

    /// Handle read resource
    async fn handle_read_resource(
        &self,
        params: ReadResourceParams,
    ) -> Result<ReadResourceResult, McpServerError> {
        if !params.uri.starts_with("vectordb://") {
            return Err("Invalid resource URI".into());
        }

        let path = params.uri.strip_prefix("vectordb://").unwrap();
        let parts: Vec<&str> = path.split('/').collect();

        if parts.len() < 2 || parts[0] != "table" {
            return Err("Invalid resource path".into());
        }

        let table_name = parts[1];
        let info = self.storage.get_table_info(table_name).await?;
        let content = serde_json::to_string_pretty(&info)?;

        Ok(ReadResourceResult {
            contents: vec![ResourceContent::Text {
                uri: params.uri,
                mime_type: Some("application/json".to_string()),
                text: content,
            }],
        })
    }

    /// Handle list prompts
    async fn handle_list_prompts(&self) -> Result<ListPromptsResult, McpServerError> {
        let prompts = self.prompts.read().await;
        Ok(ListPromptsResult {
            prompts: prompts.clone(),
        })
    }

    /// Handle get prompt
    async fn handle_get_prompt(
        &self,
        params: GetPromptParams,
    ) -> Result<GetPromptResult, McpServerError> {
        match params.name.as_str() {
            "rag_query" => Ok(self.get_rag_prompt(params.arguments)),
            "similarity_analysis" => Ok(self.get_similarity_prompt(params.arguments)),
            "data_ingestion" => Ok(self.get_ingestion_prompt(params.arguments)),
            _ => Err(format!("Unknown prompt: {}", params.name).into()),
        }
    }

    /// Get RAG prompt
    fn get_rag_prompt(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let question = args
            .as_ref()
            .and_then(|a| a.get("question"))
            .map(|s| s.as_str())
            .unwrap_or("[Question]");
        let context_limit = args
            .as_ref()
            .and_then(|a| a.get("context_limit"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(5);

        GetPromptResult {
            description: Some("Perform retrieval-augmented generation".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are a helpful AI assistant with access to a vector database. \
                         Use the vector search tools to find relevant context, then answer \
                         the user's question based on that context."
                            .to_string(),
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(format!(
                        "Question: {}\n\n\
                         Please search for the {} most relevant documents and use them \
                         to answer this question comprehensively.",
                        question, context_limit
                    )),
                },
            ],
        }
    }

    /// Get similarity analysis prompt
    fn get_similarity_prompt(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let items = args
            .as_ref()
            .and_then(|a| a.get("items"))
            .map(|s| s.as_str())
            .unwrap_or("[Items to analyze]");
        let metric = args
            .as_ref()
            .and_then(|a| a.get("metric"))
            .map(|s| s.as_str())
            .unwrap_or("cosine");

        GetPromptResult {
            description: Some("Analyze similarity between items".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are an AI assistant specialized in similarity analysis. \
                         Use the vector database tools to compute and explain similarities \
                         between different items."
                            .to_string(),
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(format!(
                        "Analyze the similarity between these items using {} metric:\n\n{}\n\n\
                         Please provide detailed similarity scores and explanations.",
                        metric, items
                    )),
                },
            ],
        }
    }

    /// Get data ingestion prompt
    fn get_ingestion_prompt(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let data_source = args
            .as_ref()
            .and_then(|a| a.get("data_source"))
            .map(|s| s.as_str())
            .unwrap_or("[Data source]");
        let format = args
            .as_ref()
            .and_then(|a| a.get("format"))
            .map(|s| s.as_str())
            .unwrap_or("json");

        GetPromptResult {
            description: Some("Guide for data ingestion".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are an AI assistant helping with data ingestion into a vector database. \
                         Guide the user through the process of preparing and importing their data.".to_string()
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(format!(
                        "I need to ingest data from {} in {} format into the vector database. \
                         Please guide me through:\n\
                         1. Data preparation steps\n\
                         2. Choosing the right embedding model\n\
                         3. Table configuration\n\
                         4. Batch import strategy\n\
                         5. Index optimization",
                        data_source, format
                    )),
                },
            ],
        }
    }
}

/// WebSocket handler for Axum
pub async fn mcp_websocket_handler<S: Storage + 'static>(
    ws: WebSocketUpgrade,
    State(server): State<Arc<McpServer<S>>>,
) -> Response {
    ws.on_upgrade(move |socket| {
        Box::pin(async move {
            server.handle_connection(socket).await;
        })
    })
}
