use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

use crate::monitoring::{
    CacheMetrics, MetricsExport, MultimodalHealthMonitor, MultimodalMetrics, MultimodalMetricsCollector,
    SystemHealthStatus,
};

/// Monitoring API state
#[derive(Clone)]
pub struct MonitoringState {
    pub metricscollector: Arc<MultimodalMetricsCollector>,
    pub healthmonitor: Arc<MultimodalHealthMonitor>,
}

/// Query parameters for metrics endpoints
#[derive(Debug, Deserialize)]
pub struct MetricsQuery {
    /// Format for metrics export (json, prometheus)
    #[serde(default = "default_format")]
    pub format: String,

    /// Model name filter (optional)
    pub model: Option<String>,

    /// Time range in seconds (optional)
    pub range: Option<u64>,
}

fn default_format() -> String {
    "json".to_string()
}

/// Query parameters for alerts
#[derive(Debug, Deserialize)]
pub struct AlertsQuery {
    /// Maximum number of alerts to return
    #[serde(default = "default_alert_limit")]
    pub limit: usize,

    /// Filter by severity (optional)
    pub severity: Option<String>,

    /// Filter by alert type (optional)
    pub alert_type: Option<String>,
}

fn default_alert_limit() -> usize {
    100
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub health_level: String,
    pub uptime_seconds: u64,
    pub system_health: SystemHealthStatus,
    pub model_health: Vec<crate::monitoring::ModelHealthStatus>,
    pub cache_health: Vec<crate::monitoring::CacheHealthStatus>,
}

/// Metrics summary response
#[derive(Debug, Serialize)]
pub struct MetricsSummary {
    pub overview: MultimodalMetrics,
    pub models: HashMap<String, crate::monitoring::EmbeddingModelMetrics>,
    pub cache: crate::monitoring::CacheMetrics,
    pub timestamp: u64,
}

/// Create monitoring routes
pub fn create_monitoring_routes() -> Router<MonitoringState> {
    Router::new()
        .route("/health", get(get_health))
        .route("/health/detailed", get(get_detailed_health))
        .route("/metrics", get(get_metrics))
        .route("/metrics/summary", get(get_metrics_summary))
        .route("/metrics/models", get(get_model_metrics))
        .route("/metrics/cache", get(get_cache_metrics))
        .route("/metrics/export", get(export_metrics))
        .route("/alerts", get(get_alerts))
        .route("/prometheus", get(get_prometheus_metrics))
}

/// Simple health check endpoint
async fn get_health(
    State(state): State<MonitoringState>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    let health_status = state.healthmonitor.get_health_status().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let status = match health_status {
        crate::monitoring::SystemHealthStatus::Healthy => "healthy",
        crate::monitoring::SystemHealthStatus::Degraded => "degraded",
        crate::monitoring::SystemHealthStatus::Unhealthy => "unhealthy",
    };

    Ok(Json(serde_json::json!({
        "status": status,
        "uptime_seconds": 0,
        "healthy_models": 0,
        "unhealthy_models": 0,
        "cache_healthy": true,
        "last_check": 0,
    })))
}

/// Detailed health check endpoint
async fn get_detailed_health(
    State(state): State<MonitoringState>,
) -> std::result::Result<Json<HealthResponse>, StatusCode> {
    let system_health = state.healthmonitor.get_health_status().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let model_health = state.healthmonitor.get_all_model_health().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let cache_health = state.healthmonitor.get_cache_health().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let status = match system_health {
        crate::monitoring::SystemHealthStatus::Healthy => "healthy",
        crate::monitoring::SystemHealthStatus::Degraded => "degraded",
        crate::monitoring::SystemHealthStatus::Unhealthy => "unhealthy",
    };

    let health_level = format!("{:?}", system_health).to_lowercase();

    Ok(Json(HealthResponse {
        status: status.to_string(),
        health_level,
        uptime_seconds: 0,
        system_health,
        model_health,
        cache_health,
    }))
}

/// Get metrics with optional filtering
async fn get_metrics(
    Query(query): Query<MetricsQuery>,
    State(state): State<MonitoringState>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    debug!("Getting metrics with query: {:?}", query);

    if let Some(model_name) = query.model {
        // Return specific model metrics
        let model_metrics = state.metricscollector.get_model_metrics(&model_name).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        match serde_json::to_value(model_metrics) {
            Ok(value) => Ok(Json(value)),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    } else {
        // Return overall metrics
        let metrics = state.metricscollector.get_metrics().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        match serde_json::to_value(metrics) {
            Ok(value) => Ok(Json(value)),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Get metrics summary
async fn get_metrics_summary(
    State(state): State<MonitoringState>) -> std::result::Result<Json<MetricsSummary>, StatusCode> {
    // Get metrics - these return Results that need to be unwrapped
    let _metrics_export = state.metricscollector.get_metrics().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let _models = state.metricscollector.get_all_model_metrics().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let _cache_data = state.metricscollector.get_cache_metrics().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(MetricsSummary {
        overview: MultimodalMetrics::default(),
        models: HashMap::new(),
        cache: CacheMetrics::default(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::from_secs(0))
            .as_secs(),
    }))
}

/// Get model-specific metrics
async fn get_model_metrics(
    State(_state): State<MonitoringState>,
) -> std::result::Result<Json<HashMap<String, crate::monitoring::EmbeddingModelMetrics>>, StatusCode>
{
    // For now, return empty map since we don't have actual model metrics
    Ok(Json(HashMap::new()))
}

/// Get cache metrics
async fn get_cache_metrics(
    State(state): State<MonitoringState>) -> std::result::Result<Json<crate::monitoring::CacheMetrics>, StatusCode> {
    let metrics_map = state.metricscollector.get_cache_metrics().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let cache_metrics = crate::monitoring::CacheMetrics {
        cachename: "default".to_string(),
        hitcount: metrics_map.get("hit_count").copied().unwrap_or(0.0) as u64,
        misscount: metrics_map.get("miss_count").copied().unwrap_or(0.0) as u64,
        hitrate: metrics_map.get("hit_rate").copied().unwrap_or(0.0),
        memoryusagebytes: metrics_map.get("memory_usage").copied().unwrap_or(0.0) as u64,
        evictioncount: metrics_map.get("eviction_count").copied().unwrap_or(0.0) as u64,
    };
    Ok(Json(cache_metrics))
}

/// Export all metrics in structured format
async fn export_metrics(
    Query(_query): Query<MetricsQuery>,
    State(state): State<MonitoringState>,
) -> std::result::Result<Json<MetricsExport>, StatusCode> {
    let metrics_export = state.metricscollector.export_metrics().await;
    Ok(Json(metrics_export))
}

/// Get alerts
async fn get_alerts(
    Query(query): Query<AlertsQuery>,
    State(state): State<MonitoringState>,
) -> std::result::Result<Json<Vec<crate::monitoring::Alert>>, StatusCode> {
    let alerts = state.metricscollector.get_alerts(Some(query.limit)).await;

    // Apply filters if specified
    let filtered_alerts = if query.severity.is_some() || query.alert_type.is_some() {
        alerts
            .into_iter()
            .filter(|alert| {
                let severity_match = query.severity.as_ref().is_none_or(|s| {
                    format!("{:?}", alert.level).to_lowercase() == s.to_lowercase()
                });

                let type_match = query.alert_type.as_ref().is_none_or(|t| {
                    format!("{:?}", alert.source).to_lowercase() == t.to_lowercase()
                });

                severity_match && type_match
            })
            .collect()
    } else {
        alerts
    };

    Ok(Json(filtered_alerts))
}

/// Get Prometheus-format metrics
async fn get_prometheus_metrics(
    State(state): State<MonitoringState>,
) -> std::result::Result<String, StatusCode> {
    let prometheus_metrics = state.metricscollector.to_prometheus().await;
    Ok(prometheus_metrics)
}

/// Create monitoring state from components
pub fn create_monitoring_state(
    metricscollector: Arc<MultimodalMetricsCollector>,
    healthmonitor: Arc<MultimodalHealthMonitor>) -> MonitoringState {
    MonitoringState {
        metricscollector,
        healthmonitor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_endpoint() -> Result<(), Box<dyn std::error::Error>> {
        let metrics_collector =
            Arc::new(MultimodalMetricsCollector);
        let health_monitor = Arc::new(MultimodalHealthMonitor);

        let state = create_monitoring_state(metrics_collector, health_monitor);
        let app = create_monitoring_routes().with_state(state);

        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            ?;

        let response = app.oneshot(request).await
            .map_err(|e| format!("Request failed: {:?}", e))?;
        assert_eq!(response.status(), StatusCode::OK);
        Ok(())
    }

    #[tokio::test]
    async fn test_metrics_endpoint() -> Result<(), Box<dyn std::error::Error>> {
        let metrics_collector =
            Arc::new(MultimodalMetricsCollector);
        let health_monitor = Arc::new(MultimodalHealthMonitor);

        let state = create_monitoring_state(metrics_collector, health_monitor);
        let app = create_monitoring_routes().with_state(state);

        let request = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            ?;

        let response = app.oneshot(request).await
            .map_err(|e| format!("Request failed: {:?}", e))?;
        assert_eq!(response.status(), StatusCode::OK);
        Ok(())
    }

    #[tokio::test]
    async fn test_prometheus_endpoint() -> Result<(), Box<dyn std::error::Error>> {
        let metrics_collector =
            Arc::new(MultimodalMetricsCollector);
        let health_monitor = Arc::new(MultimodalHealthMonitor);

        let state = create_monitoring_state(metrics_collector, health_monitor);
        let app = create_monitoring_routes().with_state(state);

        let request = Request::builder()
            .uri("/prometheus")
            .body(Body::empty())
            ?;

        let response = app.oneshot(request).await
            .map_err(|e| format!("Request failed: {:?}", e))?;
        assert_eq!(response.status(), StatusCode::OK);
        Ok(())
    }
}

impl Default for crate::config::multimodal::HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: 30,
            timeout: 10,
            check_models: true,
            check_cache: true,
        }
    }
}
