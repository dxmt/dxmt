use axum::{
    extract::{Json, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

use crate::api::response::{ApiError, ApiResponse};
use crate::cluster::simple_discovery::{NodeAnnouncement, SimpleDiscovery};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub from: uuid::Uuid,
    pub timestamp: std::time::SystemTime,
}

/// 处理节点宣告
pub async fn handle_announce(
    State(discovery): State<Arc<SimpleDiscovery>>,
    Json(announcement): Json<NodeAnnouncement>,
) -> std::result::Result<ApiResponse, StatusCode> {
    info!("Received announcement from node {}", announcement.nodeid);

    match discovery.handle_announcement(announcement).await {
        Ok(_) => Ok(ApiResponse::success(
            "Node announcement received".to_string(),
        )),
        Err(e) => Ok(ApiResponse::error(ApiError::new(
            "CLUSTER_ERROR",
            &format!("Failed to process announcement: {}", e),
        ))),
    }
}

/// 处理心跳
pub async fn handle_heartbeat(
    State(discovery): State<Arc<SimpleDiscovery>>,
    Json(heartbeat): Json<HeartbeatRequest>,
) -> std::result::Result<ApiResponse, StatusCode> {
    match discovery
        .update_heartbeat(&heartbeat.from.to_string())
        .await
    {
        Ok(_) => Ok(ApiResponse::success("Heartbeat received".to_string())),
        Err(e) => Ok(ApiResponse::error(ApiError::new(
            "CLUSTER_ERROR",
            &format!("Failed to process heartbeat: {}", e),
        ))),
    }
}

/// 获取集群状态
pub async fn get_cluster_status(
    State(discovery): State<Arc<SimpleDiscovery>>,
) -> std::result::Result<ApiResponse, StatusCode> {
    let nodes = discovery.get_active_nodes().await;

    let status = serde_json::json!({
        "total_nodes": nodes.len(),
        "active_nodes": nodes,
    });

    Ok(ApiResponse::success(status))
}
