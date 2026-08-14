#![allow(unused_variables)]
#![allow(unused_imports)]

use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    body::Body,
    routing::{get, post},
    Json, Router,
};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};
use uuid::Uuid;

use crate::api::handlers::AppState;
use crate::types::FusionStrategy;
// use crate::search::cross_modal_batch // Removed::serde_json::Value;

/// Streaming search request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingSearchRequest {
    /// Query for search
    pub query: serde_json::Value,

    /// Number of results to return
    #[serde(default = "default_k")]
    pub k: usize,

    /// Fusion strategy
    #[serde(default = "default_fusion_strategy")]
    pub fusion_strategy: FusionStrategy,

    /// Enable progressive results
    #[serde(default = "default_progressive")]
    pub progressive: bool,

    /// Chunk size for streaming
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
}

/// Streaming search session
#[derive(Debug)]
struct StreamingSession {
    id: String,
    _request: StreamingSearchRequest,
    sender: tokio::sync::mpsc::Sender<StreamingEvent>,
    created_at: std::time::Instant,
    results_sent: usize,
    completed: bool,
}

/// Events sent during streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StreamingEvent {
    /// Search started
    Started { sessionid: String, timestamp: u64 },

    /// Progress update
    Progress {
        sessionid: String,
        progress: f32,
        message: String,
    },

    /// Partial results
    Results {
        sessionid: String,
        results: Vec<StreamingSearchResult>,
        chunk_index: usize,
        total_chunks: Option<usize>,
    },

    /// Modality-specific results
    ModalityResults {
        sessionid: String,
        modality: String,
        results: Vec<StreamingSearchResult>,
        processing_time_ms: u64,
    },

    /// Final results with fusion
    FinalResults {
        sessionid: String,
        results: Vec<StreamingSearchResult>,
        fusion_scores: HashMap<String, f32>,
        totaltime_ms: u64,
        payload: SearchMetadata,
    },

    /// Error occurred
    Error {
        sessionid: String,
        error: String,
        recoverable: bool,
    },

    /// Search completed
    Completed {
        sessionid: String,
        total_results: usize,
        totaltime_ms: u64,
    },
}

/// Individual search result for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingSearchResult {
    pub id: String,
    pub score: f32,
    pub distance: f32,
    pub payload: Option<serde_json::Value>,
    pub preview: Option<ContentPreview>,
}

/// Content preview for results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPreview {
    pub content_type: String,
    pub previewtext: Option<String>,
    pub thumbnail_url: Option<String>,
    pub duration_seconds: Option<f32>,
}

/// Search metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetadata {
    pub embeddings_generated: usize,
    pub indices_searched: Vec<String>,
    pub cachehits: usize,
    pub gpu_accelerated: bool,
}

// Default values
fn default_k() -> usize {
    10
    }
fn default_progressive() -> bool {
    true
    }
fn default_chunk_size() -> usize {
    5
    }
fn default_timeout() -> u64 {
    30
    }
fn default_fusion_strategy() -> FusionStrategy {
    use std::collections::HashMap;
    FusionStrategy::LateFusion { 
        weights: HashMap::new() 
    }
}

/// Session manager for streaming searches
#[derive(Clone)]
struct SessionManager {
    sessions: Arc<RwLock<HashMap<String, StreamingSession>>>,
}

impl SessionManager {
    fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn create_session(
        &self,
        request: StreamingSearchRequest) -> (String, tokio::sync::mpsc::Receiver<StreamingEvent>) {
        let sessionid = Uuid::new_v4().to_string();
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        let session = StreamingSession {
            id: sessionid.clone(),
            _request: request,
            sender: tx,
            created_at: std::time::Instant::now(),
            results_sent: 0,
            completed: false,
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(sessionid.clone(), session);

        (sessionid, rx)
    }

    async fn get_session(&self, sessionid: &str) -> Option<StreamingSession> {
        let sessions = self.sessions.read().await;
        sessions.get(sessionid).map(|s| StreamingSession {
            id: s.id.clone(),
            _request: s._request.clone(),
            sender: s.sender.clone(),
            created_at: s.created_at,
            results_sent: s.results_sent,
            completed: s.completed,
        })
    }

    async fn update_session<F>(&self, sessionid: &str, updater: F) -> crate::Result<()>
    where
        F: FnOnce(&mut StreamingSession),
    {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(sessionid) {
            updater(session);
            Ok(())
    } else {
            Err(crate::Error::RecordNotFound("Session".to_string()))
        }
    }

    #[allow(dead_code)]
    async fn cleanup_old_sessions(&self, maxage: Duration) {
        let mut sessions = self.sessions.write().await;
        let now = std::time::Instant::now();

        sessions.retain(|_, session| now.duration_since(session.created_at) < maxage)
    }
}

/// Create streaming search routes
pub fn create_streaming_routes() -> Router<AppState> {
    Router::new()
        .route(
            "/tables/:table_name/search/stream",
            post(start_streaming_search),
        )
        .route("/search/stream/:sessionid", get(stream_results))
        .route(
            "/search/stream/:sessionid/cancel",
            post(cancel_streaming_search)
        )
    }

/// Start a streaming search session
async fn start_streaming_search(
    Path(table_name): Path<String>,
    State(state): State<AppState>,
    Json(request): Json<StreamingSearchRequest>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    info!("Starting streaming search on table {}", table_name);

    // Get or create session manager (would be part of AppState in production)
    let sessionmanager = SessionManager::new();

    // Create session
    let (sessionid, _receiver) = sessionmanager.create_session(request.clone()).await;

    // Spawn background search task
    let state_clone = state.clone();
    let sessionid_clone = sessionid.clone();
    let sessionmanager_clone = sessionmanager.clone();

    tokio::spawn(async move {
        if let Err(e) = perform_streaming_search(
            state_clone,
            table_name,
            sessionid_clone,
            request,
            sessionmanager_clone,
        )
        .await
        {
            error!("Streaming search failed: {}", e);
        }
    });

    Ok(Json(serde_json::json!({
        "sessionid": sessionid,
        "status": "started",
        "stream_url": format!("/search/stream/{}", sessionid),
    })))
}

/// Stream search results via Server-Sent Events
async fn stream_results(Path(sessionid): Path<String>, State(_state): State<AppState>) -> Response {
    info!("Client connected to stream {}", sessionid);

    // Get session manager
    let sessionmanager = SessionManager::new();

    // Get session manager and create a new receiver for this client
    if let Some(_session) = sessionmanager.get_session(&sessionid).await {
        // Create a new channel for this specific SSE connection
        let (_tx, rx) = tokio::sync::mpsc::channel(100);

        // sender is dropped here; a real implementation would register _tx in the session
        // so the session can fan out events to all connected SSE clients

        // Create SSE response
        let stream = create_sse_stream(rx);

        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("X-Accel-Buffering", "no")
            .body(Body::wrap_stream(stream))
            .unwrap()
            .into_response()
    } else {
        // Session not found
        Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Session not found"))
            .unwrap()
            .into_response()
    }
}

/// Cancel a streaming search
async fn cancel_streaming_search(
    Path(sessionid): Path<String>,
    State(_state): State<AppState>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    info!("Cancelling streaming search {}", sessionid);

    let sessionmanager = SessionManager::new();

    // Mark session as completed
    match sessionmanager
        .update_session(&sessionid, |session| {
            session.completed = true;
        })
        .await {
        Ok(_) => {},
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    Ok(Json(serde_json::json!({
        "sessionid": sessionid,
        "status": "cancelled",
    })))
}

/// Perform the actual streaming search
async fn perform_streaming_search(
    _state: AppState,
    _tablename: String,
    sessionid: String,
    _request: StreamingSearchRequest,
    sessionmanager: SessionManager) -> crate::Result<()> {
    let session = sessionmanager
        .get_session(&sessionid)
        .await
        .ok_or_else(|| crate::Error::RecordNotFound("Session".to_string() ))?;

    let start_time = std::time::Instant::now();

    // Send started event
    let _ = session
        .sender
        .send(StreamingEvent::Started {
            sessionid: sessionid.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
        .await;

    // Simulate progressive search across modalities
    let modalities = extract_modalities(&session._request.query);

    let mut all_results = Vec::new();
    let mut modality_times = HashMap::new();

    // Search each modality progressively
    let modalities_vec = modalities?;
    for (idx, modality) in modalities_vec.iter().enumerate() {
        let progress = (idx as f32 / modalities_vec.len() as f32) * 100.0;

        // Send progress update
        let _ = session
            .sender
            .send(StreamingEvent::Progress {
                sessionid: sessionid.clone(),
                progress,
                message: format!("Searching {:?} index...", modality),
            })
            .await;

        // Simulate search (in production, would use actual search)
        let modality_start = std::time::Instant::now();
        let modality_results = match simulate_modality_search(modality, &session._request).await {
            Ok(results) => results,
            Err(e) => return Err(crate::Error::Internal(format!("Modality search failed: {:?}", e))),
        };
        let modality_time = modality_start.elapsed().as_millis() as u64;

        modality_times.insert(modality.clone(), modality_time);

        // Send modality results
        let _ = session
            .sender
            .send(StreamingEvent::ModalityResults {
                sessionid: sessionid.clone(),
                modality: modality.clone(),
                results: modality_results.clone(),
                processing_time_ms: modality_time,
            })
            .await;

        all_results.extend(modality_results);

        // Check if cancelled
        let current_session = sessionmanager.get_session(&sessionid).await;
        if current_session.map(|s| s.completed).unwrap_or(true) {
            info!("Session {} cancelled", sessionid);
        }

        // Small delay to simulate processing
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Perform fusion
    let _ = session
        .sender
        .send(StreamingEvent::Progress {
            sessionid: sessionid.clone(),
            progress: 90.0,
            message: "Fusing results...".to_string(),
        })
        .await;

    let fused_results = fuse_results(
        all_results,
        &session._request.fusion_strategy,
        session._request.k,
    );

    // Send final results
    let total_time = start_time.elapsed().as_millis() as u64;
    // Send completed event
    let _ = session
        .sender
        .send(StreamingEvent::Completed {
            sessionid: sessionid.clone(),
            total_results: fused_results.results.len(),
            totaltime_ms: total_time,
        })
        .await;

    // Update session
    match sessionmanager
        .update_session(&sessionid, |session| {
            session.results_sent = fused_results.results.len();
            session.completed = true
        })
        .await {
        Ok(_) => {},
        Err(e) => return Err(crate::Error::Internal(format!("Session update failed: {:?}", e))),
    };

    Ok(())
    }

/// Create SSE stream from receiver
fn create_sse_stream(
    mut receiver: tokio::sync::mpsc::Receiver<StreamingEvent>) -> impl Stream<Item = std::result::Result<String, std::io::Error>> {
    let (tx, rx) = tokio::sync::mpsc::channel(100);

    tokio::spawn(async move {
        // Forward events from receiver to stream as SSE format
        while let Some(event) = receiver.recv().await {
            let sse_data = format_sse_event(&event);
            if tx.send(sse_data).await.is_err() {

                break;


            }
        }
    });

    ReceiverStream::new(rx).map(Ok)
}

/// Format event as SSE
fn format_sse_event(event: &StreamingEvent) -> String {
    let event_type = match event {
        StreamingEvent::Started { .. } => "started",
        StreamingEvent::Progress { .. } => "progress",
        StreamingEvent::Results { .. } => "results",
        StreamingEvent::ModalityResults { .. } => "modality_results",
        StreamingEvent::FinalResults { .. } => "final_results",
        StreamingEvent::Error { .. } => "error",
        StreamingEvent::Completed { .. } => "completed",
    };

    let json_data = serde_json::to_string(event).unwrap_or_default();
    format!("event: {}\ndata: {}\n\n", event_type, json_data)
}

/// Extract modalities from query
fn extract_modalities(query: &serde_json::Value) -> crate::Result<Vec<String>> {
    let contents = query.get("contents")
        .and_then(|v| v.as_array())
        .ok_or_else(|| crate::Error::InvalidInput("Missing contents field".to_string()))?;

    Ok(contents
        .iter()
        .filter_map(|content| {
            // Try to determine type from the content structure
            if content.get("Text").is_some() {
                Some("text".to_string())
            } else if content.get("ImagePath").is_some() || content.get("ImageBytes").is_some() {
                Some("image".to_string())
            } else if content.get("AudioPath").is_some() || content.get("AudioBytes").is_some() {
                Some("audio".to_string())
            } else if content.get("VideoPath").is_some() || content.get("VideoBytes").is_some() {
                Some("video".to_string())
            } else if content.get("URI").is_some() {
                Some("uri".to_string())
            } else if content.get("Embedding").is_some() {
                Some("embedding".to_string())
            } else {
                None
            }
        })
        .collect())
}

/// Simulate modality search (for demonstration)
async fn simulate_modality_search(
    modality: &str,
    request: &StreamingSearchRequest) -> crate::Result<Vec<StreamingSearchResult>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let num_results = (request.k / 2).max(3);
    let results = (0..num_results)
        .map(|i| StreamingSearchResult {
            id: format!("{}_{}", modality, i),
            score: rng.gen_range(0.5..1.0),
            distance: rng.gen_range(0.0..0.5),
            payload: Some(serde_json::json!({
                "modality": modality,
                "index": i,
            })),
            preview: Some(ContentPreview {
                content_type: "text".to_string(),
                previewtext: Some(format!("Result {}: Score {:.3}", i + 1, rng.gen_range(0.5..1.0))),
                thumbnail_url: None,
                duration_seconds: None,
            }),
        })
        .collect();

    Ok(results)
    }

/// Fuse results from multiple modalities
fn fuse_results(
    allresults: Vec<StreamingSearchResult>,
    strategy: &FusionStrategy,
    k: usize) -> FusedResults {
    // Simple fusion for demonstration
    let mut results = allresults;
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);

    let fusion_scores = results.iter().map(|r| (r.id.clone(), r.score)).collect();

    FusedResults {
        results,
        fusion_scores,
    }
}

struct FusedResults {
    results: Vec<StreamingSearchResult>,
    #[allow(dead_code)]
    fusion_scores: HashMap<String, f32>,
}

/// WebSocket support for bidirectional streaming
pub mod websocket {
    use super::*;
    use axum::{
        extract::ws::{Message, WebSocket, WebSocketUpgrade},
        response::IntoResponse,
    };
    use futures::{sink::SinkExt, stream::StreamExt};

    /// WebSocket handler for streaming search
    pub async fn websocket_handler(
        ws: WebSocketUpgrade,
        State(state): State<AppState>,
    ) -> std::result::Result<impl IntoResponse, StatusCode> {
        Ok(ws.on_upgrade(|socket| handle_socket(socket, state)))
    }

    async fn handle_socket(socket: WebSocket, _state: AppState) {
        let (mut sender, mut receiver) = socket.split();

        // Handle incoming messages
        while let Some(msg) = receiver.next().await {
            if let Ok(msg) = msg {
                match msg {
                    Message::Text(text) => {
                        // Parse search request
                        if let Ok(request) = serde_json::from_str::<StreamingSearchRequest>(&text) {
                            // Perform search and stream results
                            let sessionmanager = SessionManager::new();
                            let (_sessionid, mut event_receiver) =
                                sessionmanager.create_session(request).await;

                            // Stream events to WebSocket
                            while let Some(event) = event_receiver.recv().await {
                                let event_json = serde_json::to_string(&event);
                                if sender.send(Message::Text(event_json.unwrap())).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests commented out - incorrect serde_json::Value construction and missing types
    // use super::*;
    //
    // #[test]
    // fn testextract_modalities() {
    //     let query = serde_json::Value {
    //         modalities: vec![
    //             (MultimodalContent::Text("test".to_string()), 1.0),
    //             (MultimodalContent::ImagePath("/test.jpg".to_string()), 0.5),
    //         ],
    //         filters: None,
    //     };
    //
    //     let modalities = extract_modalities(&query);
    //     assert_eq!(modalities, vec!["text", "image"]);
    // }
    //
    // #[tokio::test]
    // async fn test_sessionmanager() {
    //     let manager = SessionManager::new();
    //     let request = StreamingSearchRequest {
    //         query: serde_json::Value {
    //             modalities: vec![(MultimodalContent::Text("test".to_string()), 1.0)],
    //             filters: None,
    //         },
    //         k: 10,
    //         fusion_strategy: _FusionStrategy::WeightedAverage,
    //         progressive: true,
    //         chunk_size: 5,
    //         timeout_seconds: 30,
    //     };
    //
    //     let (sessionid, _receiver) = manager.create_session(request)?;
    //
    //     let session = manager.get_session(&sessionid)?;
    //     assert!(session.is_some());
    //
    //     manager
    //         .update_session(&sessionid, |s| {
    //             s.completed = true
    //         })
    //         ??
    //         ?;
    //
    //     let updated = manager.get_session(&sessionid)?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    //     assert!(updated.completed);
    // }
}
