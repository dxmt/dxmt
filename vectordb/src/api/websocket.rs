use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};
use uuid::Uuid;

use crate::api::handlers::AppState;

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsMessage {
    // Client -> Server
    Subscribe {
        table: String,
        filters: Option<Vec<String>>,
    },
    Unsubscribe {
        table: String,
    },
    Insert {
        table: String,
        records: Vec<InsertRecord>,
    },
    Search {
        table: String,
        vector: Vec<f32>,
        k: usize,
    },

    // Server -> Client
    Inserted {
        table: String,
        record_id: String,
    },
    SearchResult {
        table: String,
        results: Vec<SearchHit>,
    },
    Error {
        message: String,
    },
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertRecord {
    pub id: Option<String>,
    pub vector: Vec<f32>,
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub payload: Option<serde_json::Value>,
}

/// WebSocket connection handler
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle WebSocket connection
async fn handle_socket(socket: WebSocket, state: AppState) {
    let (sender, mut receiver) = socket.split();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<WsMessage>(100);

    // Clone sender for the outbound task
    let sender_clone = Arc::new(tokio::sync::Mutex::new(sender));
    let sender_for_task = sender_clone.clone();

    // Spawn task to handle outbound messages
    let tx_task = tokio::spawn(async move {
        let sender = sender_for_task;
        while let Some(msg) = rx.recv().await {
            if let Ok(text) = serde_json::to_string(&msg) {
                let mut guard = sender.lock().await;
                if guard.send(Message::Text(text)).await.is_err() {
                    break;
                }
            }
        }
    });

    // Handle inbound messages
    let rx_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                        if let Err(e) = handle_message(ws_msg, &state, &tx).await {
                            let _ = tx
                                .send(WsMessage::Error {
                                    message: e.to_string(),
                                })
                                .await;
                        }
                    }
                }
                Ok(Message::Ping(data)) => {
                    let mut guard = sender_clone.lock().await;
                    if let Err(e) = guard.send(Message::Pong(data)).await {
                        error!("Failed to send pong: {}", e);
                        break;
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket connection closed");
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = tx_task => {},
        _ = rx_task => {},
    }
}

/// Handle individual WebSocket messages
async fn handle_message(
    msg: WsMessage,
    state: &AppState,
    tx: &tokio::sync::mpsc::Sender<WsMessage>,
) -> crate::Result<()> {
    match msg {
        WsMessage::Subscribe {
            table,
            filters: _filters,
        } => {
            // For now, just acknowledge subscription
            info!("Client subscribed to table: {}", table);
            Ok(())
        }
        WsMessage::Unsubscribe { table } => {
            info!("Client unsubscribed from table: {}", table);
            Ok(())
        }
        WsMessage::Insert { table, records } => handle_insert(table, records, state, tx).await,
        WsMessage::Search { table, vector, k } => handle_search(table, vector, k, state, tx).await,
        _ => Ok(()),
    }
}

/// Handle insert message
async fn handle_insert(
    table_name: String,
    records: Vec<InsertRecord>,
    state: &AppState,
    tx: &tokio::sync::mpsc::Sender<WsMessage>,
) -> crate::Result<()> {
    let table = state.storage.get_table(&table_name).await?;

    let requests: Vec<_> = records
        .into_iter()
        .map(|r| crate::types::record::CreateRecordRequest {
            id: r.id.unwrap_or_else(|| Uuid::new_v4().to_string()),
            vector: r.vector,
            sparse_vector: None,
            payload: r.payload,
            named_vectors: None,
            text: None,
        })
        .collect();

    // Use insert_records method that exists
    let inserted_records = state.storage.insert_records(&table.name, requests).await?;

    // Send confirmation for each inserted record
    for record in inserted_records {
        let _ = tx
            .send(WsMessage::Inserted {
                table: table_name.clone(),
                record_id: record.id,
            })
            .await;
    }

    Ok(())
}

/// Handle search message
async fn handle_search(
    table_name: String,
    vector: Vec<f32>,
    _k: usize,
    state: &AppState,
    tx: &tokio::sync::mpsc::Sender<WsMessage>,
) -> crate::Result<()> {
    let table = state.storage.get_table(&table_name).await?;

    if vector.len() != table.dimension {
        return Err(crate::Error::Internal(
            "Vector dimension mismatch".to_string(),
        ));
    }

    // For now, return empty results as the search method doesn't exist in storage
    // This would need to be implemented based on actual storage interface
    let results = vec![];

    let _ = tx
        .send(WsMessage::SearchResult {
            table: table_name,
            results,
        })
        .await;

    Ok(())
}

/// WebSocket connection manager
pub struct WebSocketManager {
    connections: Arc<tokio::sync::RwLock<Vec<tokio::sync::mpsc::Sender<WsMessage>>>>,
}

impl Default for WebSocketManager {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSocketManager {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(tokio::sync::RwLock::new(Vec::new())),
        }
    }

    /// Broadcast message to all connections
    pub async fn broadcast(&self, message: WsMessage) {
        let connections = self.connections.read().await;
        for tx in connections.iter() {
            let _ = tx.send(message.clone()).await;
        }
    }

    /// Add new connection
    pub async fn add_connection(&self, tx: tokio::sync::mpsc::Sender<WsMessage>) {
        let mut connections = self.connections.write().await;
        connections.push(tx);
    }

    /// Remove connection (called when connection closes)
    pub async fn remove_connection(&self, _connection_id: String) {
        // Implementation would track connections by ID and remove specific ones
        // For now, this is a placeholder
    }
}

/// Create WebSocket router
pub fn create_websocket_router() -> axum::Router<AppState> {
    axum::Router::new().route("/ws", axum::routing::get(websocket_handler))
}

/// Notification system for real-time updates
pub async fn notify_record_inserted(_table_name: &str, _record_id: &str, _interval_ms: u64) {
    // Placeholder for notification system
    // This would integrate with the WebSocket manager to broadcast updates
}
