use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

use crate::api::handlers::AppState;

/// Request body for batch cross-modal search
#[derive(Debug, Deserialize)]
pub struct BatchCrossModalSearchRequest {
    pub queries: Vec<SingleQueryRequest>,
    pub config: Option<BatchConfigRequest>,
    pub fusion_strategy: FusionStrategyRequest,
    pub filters: Option<HashMap<String, serde_json::Value>>,
    pub asynchronous: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct SingleQueryRequest {
    pub id: String,
    pub modalities: Vec<ModalityRequest>,
    pub filters: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
pub struct ModalityRequest {
    pub content: ContentRequest,
    pub weight: f64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ContentRequest {
    Text { value: String },
    ImagePath { path: String },
    ImageBytes { data: Vec<u8> },
    AudioPath { path: String },
    AudioBytes { data: Vec<u8> },
}

#[derive(Debug, Deserialize)]
pub struct BatchConfigRequest {
    pub max_batch_size: Option<usize>,
    pub parallel_processing: Option<bool>,
    pub timeout_ms: Option<u64>,
    pub adaptive_sizing: Option<bool>,
    pub prefetch_embeddings: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum FusionStrategyRequest {
    WeightedAverage,
    RankFusion,
    MaxScore,
    LateFusion { threshold: f32 },
}

/// Response body for batch cross-modal search
#[derive(Debug, Serialize)]
pub struct BatchCrossModalSearchResponse {
    pub results: Vec<SingleQueryResponse>,
    pub statistics: BatchStatistics,
    pub job_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SingleQueryResponse {
    pub query_id: String,
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct BatchStatistics {
    pub total_queries: usize,
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub processing_time_ms: u64,
    pub average_query_time_ms: f64,
}

/// Response for batch job status
#[derive(Debug, Serialize)]
pub struct BatchJobStatusResponse {
    pub job_id: String,
    pub status: String,
    pub progress: f64,
    pub results_available: bool,
}

/// Handle batch cross-modal search
pub async fn batch_cross_modal_search(
    State(_state): State<Arc<AppState>>,
    Path(table_name): Path<String>,
    Json(request): Json<BatchCrossModalSearchRequest>,
) -> Result<Json<BatchCrossModalSearchResponse>, (StatusCode, Json<serde_json::Value>)> {
    info!(
        "Batch cross-modal search on table '{}' with {} queries",
        table_name,
        request.queries.len()
    );

    // This is a stub implementation after cleanup
    // The actual cross-modal batch search functionality was removed
    // Return a simple response indicating the feature is not available

    let response = BatchCrossModalSearchResponse {
        results: request
            .queries
            .into_iter()
            .map(|q| SingleQueryResponse {
                query_id: q.id,
                results: vec![],
            })
            .collect(),
        statistics: BatchStatistics {
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            processing_time_ms: 0,
            average_query_time_ms: 0.0,
        },
        job_id: None,
    };

    Ok(Json(response))
}

/// Handle batch job status check
pub async fn batch_job_status(
    State(_state): State<Arc<AppState>>,
    Path((_table_name, job_id)): Path<(String, String)>,
) -> Result<Json<BatchJobStatusResponse>, (StatusCode, Json<serde_json::Value>)> {
    info!("Checking status for batch job '{}'", job_id);

    // Stub implementation
    let response = BatchJobStatusResponse {
        job_id,
        status: "not_found".to_string(),
        progress: 0.0,
        results_available: false,
    };

    Ok(Json(response))
}

/// Handle batch job results retrieval
pub async fn batch_job_results(
    State(_state): State<Arc<AppState>>,
    Path((_table_name, job_id)): Path<(String, String)>,
) -> Result<Json<BatchCrossModalSearchResponse>, (StatusCode, Json<serde_json::Value>)> {
    info!("Retrieving results for batch job '{}'", job_id);

    // Stub implementation
    Err((
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": "Batch job not found"
        })),
    ))
}

/// Create router for batch cross-modal search
pub fn create_batch_router() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/tables/:table_name/batch/cross-modal",
            post(batch_cross_modal_search),
        )
        .route(
            "/tables/:table_name/batch/jobs/:job_id/status",
            axum::routing::get(batch_job_status),
        )
        .route(
            "/tables/:table_name/batch/jobs/:job_id/results",
            axum::routing::get(batch_job_results),
        )
}
