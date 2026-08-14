use axum::Router;
#[cfg(feature = "api-swagger")]
use utoipa::OpenApi;
#[cfg(feature = "api-swagger")]
use utoipa::openapi::{ContactBuilder, InfoBuilder, LicenseBuilder};

// Import all request/response types
use crate::api::handlers::{
    BatchDeleteRequest, BatchInsertRequest, CreateTableRequest, PaginationQuery,
    UpdateRecordRequest,
};
use crate::types::{
    record::{CreateRecordRequest, Record, SearchQuery, SearchResult},
    table::{IndexType, Table, TableConfig, TableInfoResponse},
    vector::DistanceMetric,
};

/// DXMT VectorDB 完整API文档
#[derive(OpenApi)]
#[openapi(
    paths(
        // ========== System APIs ==========
        crate::api::handlers::get_status,
        crate::api::handlers::health_check,

        // ========== Table Management APIs ==========
        crate::api::handlers::create_table,
        crate::api::handlers::get_table,
        crate::api::handlers::list_tables,
        crate::api::handlers::delete_table,

        // ========== Record Operations APIs ==========
        crate::api::handlers::insert_records,
        crate::api::handlers::bulk_insert_records,
        crate::api::handlers::get_record,
        crate::api::handlers::update_record,
        crate::api::handlers::delete_records,

        // ========== Search APIs ==========
        crate::api::handlers::search_vectors,
        crate::api::handlers::hybrid_search,
        crate::api::handlers::batch_search_vectors,

        // ========== Backup APIs ==========
        crate::api::handlers::create_full_backup,
        crate::api::handlers::create_incremental_backup,
        crate::api::handlers::create_manual_backup,
        crate::api::handlers::get_backup_status,
        crate::api::handlers::list_backups,
        crate::api::handlers::restore_backup,
        crate::api::handlers::delete_backup,

        // ========== Multimodal APIs ==========
        crate::api::handlers::generate_embedding,
        crate::api::handlers::generate_multimodal_embedding,
        crate::api::handlers::multimodal_search,
        crate::api::handlers::validate_multimodal_content,

        // ========== Monitoring APIs ==========
        crate::api::monitoring::get_health,
        crate::api::monitoring::get_detailed_health,
        crate::api::monitoring::get_metrics,
        crate::api::monitoring::get_metrics_summary,
        crate::api::monitoring::get_model_metrics,
        crate::api::monitoring::get_cache_metrics,
        crate::api::monitoring::export_metrics,
        crate::api::monitoring::get_alerts,
        crate::api::monitoring::get_prometheus_metrics,

        // ========== Streaming APIs ==========
        crate::api::streaming_search::streaming_search_vectors,
        crate::api::streaming_search::stream_results,
        crate::api::streaming_search::stop_streaming_search,

        // ========== Cross-Modal Batch APIs ==========
        crate::api::cross_modal_batch::batch_search,
        crate::api::cross_modal_batch::hybrid_batch_search,

        // ========== Simple APIs (Zero-Config) ==========
        // Removed - now using simple_v2 which provides Collections API

        // ========== Cluster APIs (if enabled) ==========
        #[cfg(feature = "cluster")]
        crate::api::server_with_cluster::handle_announce,
        #[cfg(feature = "cluster")]
        crate::api::server_with_cluster::handle_heartbeat,
        #[cfg(feature = "cluster")]
        crate::api::server_with_cluster::get_cluster_status,
    ),
    components(
        schemas(
            // Core Types
            Table,
            TableConfig,
            TableInfoResponse,
            CreateTableRequest,
            IndexType,
            DistanceMetric,

            // Record Types
            Record,
            CreateRecordRequest,
            BatchInsertRequest,
            UpdateRecordRequest,
            BatchDeleteRequest,

            // Search Types
            SearchQuery,
            SearchResult,
            HybridSearchRequest,
            BatchSearchRequest,
            StreamingSearchRequest,

            // Backup Types
            BackupRequest,
            BackupStatus,
            BackupInfo,
            RestoreRequest,

            // Multimodal Types
            EmbeddingRequest,
            MultimodalEmbeddingRequest,
            MultimodalSearchRequest,
            MultimodalContent,

            // Monitoring Types
            HealthStatus,
            DetailedHealthStatus,
            MetricsResponse,
            MetricsSummary,
            ModelMetrics,
            CacheMetrics,
            AlertInfo,
            PrometheusMetrics,

            // Simple API Types
            CreateCollectionRequest,
            InsertPointsRequest,
            Point,
            SearchRequest,
            SearchPointsResult,
            ScoredPoint,

            // Response Types
            crate::api::response::ApiResponse,
            crate::api::response::ApiError,
            SimpleResponse,

            // Pagination
            PaginationQuery,
        )
    ),
    tags(
        (name = "System", description = "系统管理和健康检查"),
        (name = "Tables", description = "表（集合）管理操作"),
        (name = "Records", description = "向量记录的增删改查"),
        (name = "Search", description = "向量搜索和混合搜索"),
        (name = "Backup", description = "备份和恢复管理"),
        (name = "Multimodal", description = "多模态嵌入和搜索"),
        (name = "Monitoring", description = "监控指标和性能分析"),
        (name = "Streaming", description = "流式搜索接口"),
        (name = "Batch", description = "批量操作接口"),
        (name = "Simple", description = "简化API（零配置）"),
        (name = "Cluster", description = "集群管理（可选）"),
        (name = "GraphQL", description = "GraphQL接口"),
        (name = "WebSocket", description = "WebSocket实时接口")
    ),
    info(
        title = "DXMT VectorDB API",
        description = r#"
# DXMT VectorDB - 高性能向量数据库

## 概述
DXMT VectorDB 是一个轻量级、高性能的向量数据库，专为单机部署优化，同时支持分布式扩展。

## 特性
- 🚀 **零配置启动** - 自动检测资源，无需配置文件
- ⚡ **极致性能** - 900K写入/秒，1.8M读取/秒
- 🎯 **简单易用** - 仅5个核心API即可完成所有操作
- 💾 **混合存储** - LMDB读优化 + RocksDB写优化
- 🔍 **混合搜索** - 支持向量搜索 + BM25文本搜索
- 🖼️ **多模态** - 支持文本、图像、音频嵌入
- 📊 **实时监控** - 内置性能指标和健康检查
- 🔄 **流式处理** - 支持流式搜索和批量操作

## 快速开始

### 极简模式（推荐）
使用 `/collections` 系列API，仅需5个接口：
- `POST /collections` - 创建集合
- `POST /collections/{name}/points` - 插入向量
- `POST /collections/{name}/search` - 搜索向量
- `GET /collections/{name}/points/{id}` - 获取向量
- `DELETE /collections/{name}/points/{id}` - 删除向量

### 标准模式
使用 `/api/v1/tables` 系列API，提供完整功能。

## 认证
当前版本不需要认证。生产环境建议使用反向代理添加认证层。

## 限流
- 默认请求体限制：10MB
- 批量操作限制：10000条/批次
- 并发连接限制：10000

## 响应格式
所有API返回统一的JSON格式：
```json
{
    "success": true,
    "data": {...},
    "message": "操作成功",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

错误响应：
```json
{
    "success": false,
    "error": {
        "code": "ERROR_CODE",
        "message": "错误描述",
        "details": {...}
    },
    "timestamp": "2024-01-01T00:00:00Z"
}
```
        "#,
        version = "0.2.0",
        contact(
            name = "DXMT Team",
            url = "https://github.com/dxmt/vectordb",
            email = "support@dxmt.io"
        ),
        license(
            name = "Apache-2.0",
            url = "https://www.apache.org/licenses/LICENSE-2.0"
        )
    ),
    servers(
        (url = "/", description = "当前服务器"),
        (url = "http://localhost:9999", description = "本地开发（默认）"),
        (url = "https://api.dxmt.io", description = "生产环境API")
    ),
    external_docs(
        url = "https://docs.dxmt.io",
        description = "完整文档"
    )
)]
pub struct CompleteApiDoc;

/// API路径组织结构
pub struct ApiPaths;

impl ApiPaths {
    /// 系统管理路径
    pub const SYSTEM: &'static str = "/api/v1";

    /// 表管理路径
    pub const TABLES: &'static str = "/api/v1/tables";

    /// 简化API路径（零配置）
    pub const SIMPLE: &'static str = "/collections";

    /// 监控路径
    pub const MONITORING: &'static str = "/metrics";

    /// GraphQL路径
    pub const GRAPHQL: &'static str = "/graphql";

    /// WebSocket路径
    pub const WEBSOCKET: &'static str = "/ws";
}

/// 创建完整的Swagger UI路由 - 仅在 /api/docs 提供
pub fn create_complete_swagger_ui<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        // 唯一的文档入口
        .route("/api/docs", axum::routing::get(swagger_ui_handler))
        .route("/api/docs/*path", axum::routing::get(swagger_ui_handler))
        .route("/api/docs/openapi.json", axum::routing::get(openapi_json_handler))
}


/// Swagger UI HTML处理
async fn swagger_ui_handler() -> axum::response::Html<String> {
    let html = r#"<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DXMT VectorDB API Documentation</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.10.3/swagger-ui.css">
    <style>
        body { margin: 0; padding: 0; }
        .swagger-ui .topbar { display: none; }
        .swagger-ui .info { margin-bottom: 20px; }
        .swagger-ui .info .title { font-size: 36px; }
        .swagger-ui .scheme-container { background: #f7f7f7; padding: 15px; border-radius: 4px; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.10.3/swagger-ui-bundle.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.10.3/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: "/api/docs/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                docExpansion: 'none',
                filter: true,
                showExtensions: true,
                showCommonExtensions: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                persistAuthorization: true,
                withCredentials: true,
                requestInterceptor: (request) => {
                    // 添加自定义请求头
                    request.headers['X-Client'] = 'swagger-ui';
                    return request;
                },
                responseInterceptor: (response) => {
                    // 自定义响应处理
                    return response;
                }
            });
        };
    </script>
</body>
</html>"#;

    axum::response::Html(html.to_string())
}

/// OpenAPI JSON规范处理
async fn openapi_json_handler() -> axum::Json<utoipa::openapi::OpenApi> {
    axum::Json(CompleteApiDoc::openapi())
}

// ========== 请求/响应类型定义 ==========

use serde::{Deserialize, Serialize};
#[cfg(feature = "api-swagger")]
use utoipa::ToSchema;

/// 混合搜索请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchRequest {
    /// 向量查询
    pub vector: Option<Vec<f32>>,
    /// 文本查询（BM25）
    pub text: Option<String>,
    /// 返回数量
    pub limit: usize,
    /// 融合方法
    pub fusion_method: Option<String>,
}

/// 批量搜索请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSearchRequest {
    /// 批量查询
    pub queries: Vec<SearchQuery>,
    /// 批大小
    pub batch_size: Option<usize>,
}

/// 流式搜索请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingSearchRequest {
    /// 查询向量
    pub vector: Vec<f32>,
    /// 流式返回数量
    pub stream_size: usize,
}

/// 备份请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRequest {
    /// 备份类型
    pub backup_type: String,
    /// 备份路径
    pub path: Option<String>,
}

/// 备份状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupStatus {
    /// 备份ID
    pub id: String,
    /// 状态
    pub status: String,
    /// 进度
    pub progress: f32,
}

/// 备份信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
    /// 备份ID
    pub id: String,
    /// 创建时间
    pub created_at: String,
    /// 大小
    pub size_bytes: u64,
}

/// 恢复请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreRequest {
    /// 备份ID
    pub backup_id: String,
    /// 目标表
    pub target_table: Option<String>,
}

/// 嵌入请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// 文本内容
    pub text: String,
    /// 模型名称
    pub model: Option<String>,
}

/// 多模态嵌入请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalEmbeddingRequest {
    /// 文本
    pub text: Option<String>,
    /// 图像（base64）
    pub image: Option<String>,
    /// 音频（base64）
    pub audio: Option<String>,
}

/// 多模态搜索请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalSearchRequest {
    /// 多模态查询
    pub query: MultimodalEmbeddingRequest,
    /// 返回数量
    pub limit: usize,
}

/// 多模态内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalContent {
    /// 内容类型
    pub content_type: String,
    /// 内容数据
    pub data: serde_json::Value,
}

/// 健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// 状态
    pub status: String,
    /// 版本
    pub version: String,
}

/// 详细健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedHealthStatus {
    /// 基本状态
    pub status: HealthStatus,
    /// 组件状态
    pub components: serde_json::Value,
}

/// 指标响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResponse {
    /// 指标数据
    pub metrics: serde_json::Value,
}

/// 指标摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    /// QPS
    pub qps: f64,
    /// 延迟
    pub latency_ms: f64,
    /// 内存使用
    pub memory_mb: u64,
}

/// 模型指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// 模型名称
    pub model: String,
    /// 推理次数
    pub inference_count: u64,
    /// 平均延迟
    pub avg_latency_ms: f64,
}

/// 缓存指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// 命中率
    pub hit_rate: f64,
    /// 缓存大小
    pub size_mb: u64,
    /// 条目数
    pub entries: u64,
}

/// 告警信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertInfo {
    /// 告警级别
    pub level: String,
    /// 告警消息
    pub message: String,
    /// 时间戳
    pub timestamp: String,
}

/// Prometheus指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusMetrics {
    /// Prometheus格式的指标文本
    pub metrics: String,
}

/// 简化API类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    /// 集合名称
    pub name: String,
    /// 向量维度
    pub dimension: usize,
    /// 距离度量
    pub metric: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertPointsRequest {
    /// 向量点列表
    pub points: Vec<Point>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    /// ID
    pub id: Option<String>,
    /// 向量
    pub vector: Vec<f32>,
    /// 元数据
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// 查询向量
    pub vector: Vec<f32>,
    /// 返回数量
    pub limit: Option<usize>,
    /// 过滤条件
    pub filter: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPointsResult {
    /// 结果列表
    pub results: Vec<ScoredPoint>,
    /// 耗时毫秒
    pub took_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredPoint {
    /// ID
    pub id: String,
    /// 分数
    pub score: f32,
    /// 向量
    pub vector: Option<Vec<f32>>,
    /// 元数据
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleResponse {
    /// 是否成功
    pub success: bool,
    /// 消息
    pub message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openapi_generation() {
        let openapi = CompleteApiDoc::openapi();
        assert_eq!(openapi.info.title, "DXMT VectorDB API");
        assert_eq!(openapi.info.version, "0.2.0");
    }

    #[test]
    fn test_api_paths() {
        assert_eq!(ApiPaths::SYSTEM, "/api/v1");
        assert_eq!(ApiPaths::SIMPLE, "/collections");
        assert_eq!(ApiPaths::GRAPHQL, "/graphql");
    }
}