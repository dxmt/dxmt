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

/// The tools a read-only server offers: everything that reads, nothing that writes, creates or
/// deletes. An allow-list, so a tool added later is withheld from a read-only server until it is
/// named here -- a deny-list would expose every new write tool by default.
pub const READ_ONLY_TOOLS: &[&str] = &[
    "vector_search",
    "hybrid_search",
    "list_tables",
    "get_table_info",
    "recall",
    "list_memories",
    "get_memory",
];

/// Whether `DXMT_MCP_READ_ONLY` asks for a read-only MCP server (`1`, `true`, `yes`, `on`).
pub fn read_only_from_env() -> bool {
    crate::config::env::var("DXMT_MCP_READ_ONLY")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// The refusal a read-only server gives a tool outside `READ_ONLY_TOOLS`.
pub fn read_only_refusal(tool: &str) -> String {
    format!(
        "tool '{tool}' is not available: this MCP server is read-only (DXMT_MCP_READ_ONLY); \
         it offers {}",
        READ_ONLY_TOOLS.join(", ")
    )
}

/// MCP server state
pub struct McpServer<S: Storage> {
    storage: Arc<S>,
    /// Offer and run only `READ_ONLY_TOOLS`: for an agent that should consult the store and
    /// memory but never change them.
    read_only: bool,
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
            read_only: read_only_from_env(),
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

    /// Offer and run only the tools that read (`READ_ONLY_TOOLS`), whatever the environment says.
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Populate the tool and prompt catalogues. Idempotent.
    ///
    /// Must run before the first `tools/list`, so both transports call it up front rather than
    /// leaving it to a background task.
    pub async fn initialize(&self) {
        self.tools.write().await.clear();
        self.prompts.write().await.clear();
        self.initialize_tools().await;
        if self.read_only {
            self.tools
                .write()
                .await
                .retain(|tool| READ_ONLY_TOOLS.contains(&tool.name.as_str()));
        }
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
                        "description": "Query vector. Omit it and pass `text` instead when the \
                                        table declares an embedding model."
                    },
                    "text": {
                        "type": "string",
                        "description": "Query text, embedded by the table's own model. Use this \
                                        unless you have a vector; an MCP client has no embedding \
                                        model of its own. `get_table_info` says whether a table \
                                        has one."
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
                "required": ["table"]
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
                        "description": "Vector data. Omit it and pass `text` instead when the \
                                        table declares an embedding model."
                    },
                    "text": {
                        "type": "string",
                        "description": "Text for the table's model to embed. It is also stored \
                                        in the payload, so a later search returns the text this \
                                        record came from."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Associated metadata"
                    }
                },
                "required": ["table", "id"]
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
                        "description": "Vector dimension. Omit it when `embedding` is given and \
                                        the server will ask the model how wide its vectors are, \
                                        which is one fewer number to be wrong about."
                    },
                    "embedding": {
                        "type": "object",
                        "description": "Let the server embed text for this table, so records \
                                        and searches can carry `text` instead of a vector. \
                                        `model` is required. `base_url` points at anything \
                                        speaking the OpenAI /embeddings protocol and defaults \
                                        to OpenAI; `api_key_env` names an environment variable \
                                        of the server (default OPENAI_API_KEY) and never \
                                        carries the key itself.",
                        "properties": {
                            "model": {"type": "string"},
                            "base_url": {"type": "string"},
                            "field": {"type": "string"},
                            "api_key_env": {"type": "string"},
                            "batch_size": {"type": "integer"}
                        },
                        "required": ["model"]
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
                "required": ["name"]
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

        // Documents: what a note-taking agent has is a whole text, often longer than a model
        // reads, and `insert_vector` stores one vector for it.
        tools.push(Tool {
            name: "add_document".to_string(),
            description: Some(
                "Store a long text (a note, an article, a file's contents) as one document: it \
                 is split into overlapping chunks and each is embedded with the table's model. \
                 Adding a document id again replaces it. Search results are chunks whose payload \
                 names the document (`doc_id`). Use the `kb` table unless told otherwise."
                    .to_string(),
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name, with an embedding model (`kb` has one)"},
                    "id": {"type": "string", "description": "Document id, e.g. a file path"},
                    "text": {"type": "string", "description": "The whole document"},
                    "metadata": {"type": "object", "description": "Fields copied onto every chunk, e.g. title, path, tags"},
                    "chunk_size": {"type": "integer", "description": "Characters per chunk, default 500"},
                    "chunk_overlap": {"type": "integer", "description": "Characters repeated between chunks, default 50"}
                },
                "required": ["table", "id", "text"]
            }),
        });
        tools.push(Tool {
            name: "delete_document".to_string(),
            description: Some(
                "Delete every chunk of a document added with add_document".to_string(),
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name"},
                    "id": {"type": "string", "description": "Document id"}
                },
                "required": ["table", "id"]
            }),
        });

        // Memory: what an agent should keep about its user and its work, shared by every agent
        // that talks to this database. See `storage::agent_memory`.
        tools.push(Tool {
            name: "remember".to_string(),
            description: Some(
                "Keep something worth knowing next time: a fact about the user, a preference, \
                 a decision, an event, an instruction. One memory per call, in a sentence that \
                 stands on its own. If it replaces something remembered before (the user moved, \
                 changed their mind), pass the old memory's id in `supersedes`: the old one is \
                 kept as history and no longer recalled. Saying something already remembered \
                 refreshes that memory rather than adding a copy (`duplicate: true`)."
                    .to_string(),
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The memory, e.g. \"Prefers answers in Chinese\""},
                    "kind": {"type": "string", "description": "fact, preference, event, instruction, or note (default)"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "source": {"type": "string", "description": "Which agent or conversation this came from"},
                    "supersedes": {"type": "array", "items": {"type": "string"}, "description": "Ids of memories this replaces"},
                    "id": {"type": "string", "description": "To correct an existing memory in place"},
                    "metadata": {"type": "object", "description": "Anything else to keep with it"}
                },
                "required": ["text"]
            }),
        });
        tools.push(Tool {
            name: "recall".to_string(),
            description: Some(
                "Find what is remembered about a topic, by meaning and by keyword. Use it at the \
                 start of a task and whenever the user refers to something from before. Returns \
                 memories with their ids, kinds, tags, when they were remembered (`at`, Unix \
                 seconds) and a score."
                    .to_string(),
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A question or topic"},
                    "limit": {"type": "integer", "description": "How many, default 25"},
                    "kind": {"type": "string", "description": "Only this kind"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Only memories with any of these tags"},
                    "since": {"description": "Only memories from then on: Unix seconds or RFC 3339"},
                    "include_superseded": {"type": "boolean", "description": "Include replaced memories, for history"}
                },
                "required": ["query"]
            }),
        });
        tools.push(Tool {
            name: "forget".to_string(),
            description: Some(
                "Delete memories for good, by id -- when the user asks you to forget something, \
                 or a memory is plainly wrong. To record that something changed, `remember` the \
                 new state with `supersedes` instead."
                    .to_string(),
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["ids"]
            }),
        });
        tools.push(Tool {
            name: "get_memory".to_string(),
            description: Some(
                "Read one memory by its id -- the id `recall` or `remember` returned -- with its \
                 kind, tags, source, when it was remembered, and what replaced it if anything did."
                    .to_string(),
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"id": {"type": "string", "description": "The memory's id"}},
                "required": ["id"]
            }),
        });
        tools.push(Tool {
            name: "list_memories".to_string(),
            description: Some(
                "List memories, newest first -- to review or audit what is kept.".to_string(),
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "limit": {"type": "integer", "description": "Default 50"},
                    "offset": {"type": "integer"}
                }
            }),
        });

        // Hybrid search tool
        tools.push(Tool {
            name: "hybrid_search".to_string(),
            description: Some(
                "Search a table by meaning and by keywords at once, and fuse the two rankings. \
                 Give `text` (embedded with the table's model and matched by BM25), a `vector`, \
                 or both."
                    .to_string(),
            ),
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
                        "description": "Share of the score from the vector side (0-1); the server's default is 0.6"
                    },
                    "fusion_method": {
                        "type": "string",
                        "enum": ["weighted_sum", "reciprocal_rank_fusion", "dbsf"],
                        "description": "How to fuse the two rankings; the server's default is weighted_sum"
                    },
                    "filter": {
                        "type": "object",
                        "description": "Payload filter, as in vector_search"
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
    /// One JSON-RPC message in, its reply out -- `None` for a notification, which has none.
    /// A failure is an error response carrying the request's id, as JSON-RPC requires; it used
    /// to go out without one, which a client cannot match to the call that failed.
    ///
    /// For transports that are not a session of their own: `POST /mcp` on the HTTP server,
    /// where every request arrives on its own and all of them share this server's one session.
    pub async fn answer(&self, message: &str) -> Option<String> {
        self.initialize().await;
        let session_id = self.shared_session().await;
        match self.handle_message(session_id, message).await {
            Ok(response) if response.is_empty() => None,
            Ok(response) => Some(response),
            Err(e) => {
                error!("MCP request failed: {}", e);
                let id = serde_json::from_str::<serde_json::Value>(message)
                    .ok()
                    .and_then(|v| v.get("id").cloned())
                    .unwrap_or(serde_json::Value::Null);
                Some(
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32603, "message": e.to_string() }
                    })
                    .to_string(),
                )
            }
        }
    }

    /// The session HTTP requests share: made on first use, kept for the life of the server.
    async fn shared_session(&self) -> Uuid {
        // A fixed id: requests over HTTP carry no session, and every one of them is the same
        // client as far as this server's state goes (it keeps none that differs per client).
        let id = Uuid::nil();
        let mut sessions = self.sessions.write().await;
        sessions.entry(id).or_insert_with(|| SessionState {
            id,
            client_info: None,
            capabilities: None,
            initialized: false,
        });
        id
    }

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
                    let id = serde_json::from_str::<serde_json::Value>(&line)
                        .ok()
                        .and_then(|v| v.get("id").cloned())
                        .unwrap_or(serde_json::Value::Null);
                    serde_json::to_string(&serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
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
            // A request and a notification have the same fields, so the untagged enum reads
            // every one as a request; what tells them apart is the id. Without this, the
            // `notifications/initialized` every client sends after `initialize` was answered
            // "Unknown method", on stdout over stdio and as a 200 over HTTP.
            JsonRpcContent::Request { method, params } if json_rpc.id.is_none() => {
                self.handle_notification(session_id, &method, params).await;
                return Ok(String::new());
            }
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
        if self.read_only && !READ_ONLY_TOOLS.contains(&params.name.as_str()) {
            return Err(read_only_refusal(&params.name).into());
        }
        match params.name.as_str() {
            "vector_search" => self.handle_vector_search(params.arguments).await,
            "insert_vector" => self.handle_insert_vector(params.arguments).await,
            "create_table" => self.handle_create_table(params.arguments).await,
            "list_tables" => self.handle_list_tables().await,
            "get_table_info" => self.handle_get_table_info(params.arguments).await,
            "delete_vectors" => self.handle_delete_vectors(params.arguments).await,
            "hybrid_search" => self.handle_hybrid_search(params.arguments).await,
            "add_document" => self.handle_add_document(params.arguments).await,
            "delete_document" => self.handle_delete_document(params.arguments).await,
            "remember" => self.handle_remember(params.arguments).await,
            "recall" => self.handle_recall(params.arguments).await,
            "forget" => self.handle_forget(params.arguments).await,
            "get_memory" => self.handle_get_memory(params.arguments).await,
            "list_memories" => self.handle_list_memories(params.arguments).await,
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
        // Either a vector or text, never both: the storage layer refuses a query carrying two,
        // and quietly preferring one of them is how an agent measures the wrong thing.
        let text = args["text"].as_str().map(str::to_string);
        let vector: Vec<f32> = match args.get("vector") {
            Some(serde_json::Value::Null) | None => Vec::new(),
            Some(value) => serde_json::from_value(value.clone())?,
        };
        if vector.is_empty() && text.is_none() {
            return Err(
                "give `vector`, or `text` on a table that declares an embedding model".into(),
            );
        }
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
            text,
            top_k: limit,
            limit,
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
        let text = args["text"].as_str().map(str::to_string);
        let vector: Vec<f32> = match args.get("vector") {
            Some(serde_json::Value::Null) | None => Vec::new(),
            Some(value) => serde_json::from_value(value.clone())?,
        };
        if vector.is_empty() && text.is_none() {
            return Err(
                "give `vector`, or `text` on a table that declares an embedding model".into(),
            );
        }
        let metadata = args.get("metadata").cloned();

        // Insert vector
        let record = crate::types::record::CreateRecordRequest {
            id: id.to_string(),
            vector,
            sparse_vector: None,
            payload: metadata,
            named_vectors: None,
            text,
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
        let embedding: Option<crate::types::table::EmbeddingConfig> = match args.get("embedding") {
            Some(serde_json::Value::Null) | None => None,
            Some(value) => Some(
                serde_json::from_value(value.clone())
                    .map_err(|e| format!("Invalid embedding config: {e}"))?,
            ),
        };
        // Zero asks the model for its width, which `create_table` only accepts with a model.
        let dimension = match args["dimension"].as_u64() {
            Some(d) => d as usize,
            None if embedding.is_some() => 0,
            None => return Err("Missing dimension, and no `embedding` to take one from".into()),
        };
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
        // A table that stores text gets keyword search over it, as the knowledge-base table
        // does: without it `hybrid_search` on a table this tool made had no text half, and the
        // fusion was the vector ranking alone.
        let text_fields = embedding
            .as_ref()
            .and_then(|e| crate::storage::knowledge_base::text_search_fields(e.field()));
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
            enable_full_text_search: text_fields.is_some(),
            text_fields,
            memory_optimization: None, // Use global config
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding,
            rerank: None,
        };

        let created = self.storage.create_table(table_config).await?;

        Ok(CallToolResult {
            content: vec![ToolContent::Text {
                // `created.dimension`, not the argument: when the model was asked, the number
                // worth telling the caller is the one the table actually has.
                text: format!(
                    "Successfully created table '{}' with dimension {} and {} metric",
                    name, created.dimension, metric
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

    async fn handle_add_document(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let table = args["table"].as_str().ok_or("Missing table name")?;
        let id = args["id"].as_str().ok_or("Missing document id")?;
        let mut body = serde_json::Map::new();
        for key in ["text", "metadata", "chunk_size", "chunk_overlap"] {
            if let Some(value) = args.get(key) {
                body.insert(key.to_string(), value.clone());
            }
        }
        let request: crate::storage::documents::PutDocumentRequest =
            serde_json::from_value(serde_json::Value::Object(body))
                .map_err(|e| format!("Invalid add_document arguments: {e}"))?;
        let written = self.storage.put_document(table, id, request).await?;
        Ok(CallToolResult {
            content: vec![ToolContent::Text {
                text: serde_json::to_string_pretty(&written)?,
            }],
            is_error: Some(false),
        })
    }

    async fn handle_delete_document(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let table = args["table"].as_str().ok_or("Missing table name")?;
        let id = args["id"].as_str().ok_or("Missing document id")?;
        let deleted = self.storage.delete_document(table, id).await?;
        Ok(CallToolResult {
            content: vec![ToolContent::Text {
                text: serde_json::json!({ "deleted_chunks": deleted }).to_string(),
            }],
            is_error: Some(false),
        })
    }

    fn json_result(value: &serde_json::Value) -> Result<CallToolResult, McpServerError> {
        Ok(CallToolResult {
            content: vec![ToolContent::Text {
                text: serde_json::to_string_pretty(value)?,
            }],
            is_error: Some(false),
        })
    }

    async fn handle_remember(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let request: crate::storage::agent_memory::RememberRequest =
            serde_json::from_value(args.ok_or("Missing arguments")?)
                .map_err(|e| format!("Invalid remember arguments: {e}"))?;
        Self::json_result(&self.storage.remember(request).await?)
    }

    async fn handle_recall(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let request: crate::storage::agent_memory::RecallRequest =
            serde_json::from_value(args.ok_or("Missing arguments")?)
                .map_err(|e| format!("Invalid recall arguments: {e}"))?;
        let memories = self.storage.recall(request).await?;
        Self::json_result(&serde_json::json!({ "memories": memories }))
    }

    async fn handle_get_memory(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let id = args["id"]
            .as_str()
            .ok_or("get_memory needs `id`, a string")?
            .to_string();
        let memory = self.storage.get_memory(&id).await?;
        Self::json_result(&memory)
    }

    async fn handle_forget(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let ids: Vec<String> = serde_json::from_value(args["ids"].clone())
            .map_err(|e| format!("`ids` must be a list of memory ids: {e}"))?;
        let forgotten = self.storage.forget(ids).await?;
        Self::json_result(&serde_json::json!({ "forgotten": forgotten }))
    }

    async fn handle_list_memories(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.unwrap_or_default();
        let limit = args["limit"].as_u64().unwrap_or(50) as usize;
        let offset = args["offset"].as_u64().unwrap_or(0) as usize;
        let (memories, total) = self
            .storage
            .list_memories(args["kind"].as_str(), limit, offset)
            .await?;
        Self::json_result(&serde_json::json!({ "memories": memories, "total": total }))
    }

    /// Handle hybrid search tool
    async fn handle_hybrid_search(
        &self,
        args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpServerError> {
        let args = args.ok_or("Missing arguments")?;
        let table_name = args["table"].as_str().ok_or("Missing table name")?;

        if args.get("vector").is_none() && args.get("text").is_none() {
            return Err("Must provide either 'vector' or 'text' for hybrid search".into());
        }
        // The REST body, built from the arguments it shares with this tool, so the defaults
        // (fusion, weight) are the server's and an argument of the wrong type is refused the
        // way the endpoint refuses it. Text-only was answered with an empty page and a note
        // until 2026-10-14, and a vector with a plain vector search: the tool searched
        // nothing by keyword at all. MCP callers are language models, and text is what they
        // have.
        let mut body = serde_json::Map::new();
        for key in [
            "vector",
            "text",
            "vector_weight",
            "fusion_method",
            "limit",
            "filter",
        ] {
            if let Some(value) = args.get(key) {
                body.insert(key.to_string(), value.clone());
            }
        }
        body.entry("limit").or_insert(serde_json::json!(10));
        let request: crate::types::record::HybridSearchRequest =
            serde_json::from_value(serde_json::Value::Object(body))
                .map_err(|e| format!("Invalid hybrid search arguments: {e}"))?;
        let results = self.storage.hybrid_search(table_name, request).await?;

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
