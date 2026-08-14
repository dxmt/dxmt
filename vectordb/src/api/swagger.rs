use axum::Router;
#[cfg(feature = "api-swagger")]
use utoipa::OpenApi;

use crate::api::handlers::{
    BatchDeleteRequest, BatchInsertRequest, CreateTableRequest, PaginationQuery,
    UpdateRecordRequest,
};
use crate::types::{
    record::{CreateRecordRequest, Record, SearchQuery, SearchResult},
    table::{IndexType, Table, TableConfig},
    vector::DistanceMetric,
};

/// DXMT VectorDB API 文档
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::handlers::get_status,
        crate::api::handlers::create_table,
        crate::api::handlers::get_table,
        crate::api::handlers::list_tables,
        crate::api::handlers::delete_table,
        crate::api::handlers::insert_records,
        crate::api::handlers::get_record,
        crate::api::handlers::update_record,
        crate::api::handlers::delete_records,
        crate::api::handlers::search_vectors),
    components(
        schemas(
            Table, TableConfig, CreateRecordRequest, SearchQuery, SearchResult, Record,
            DistanceMetric, IndexType, CreateTableRequest, BatchInsertRequest,
            UpdateRecordRequest, BatchDeleteRequest, PaginationQuery,
            crate::api::response::ApiResponse,
            crate::api::response::ApiError,
            crate::types::table::TableInfoResponse)
    ),
    tags(
        (name = "system", description = "系统管理API"),
        (name = "tables", description = "表管理API"),
        (name = "records", description = "记录操作API"),
        (name = "search", description = "向量搜索API")
    ),
    info(
        title = "DXMT VectorDB API",
        description = "轻量级、水平可扩展的向量数据库API",
        version = "1.0.0",
        contact(
            name = "DXMT Team",
            url = "https://github.com/dxmt/vectordb"
        ),
        license(
            name = "Apache-2.0",
            url = "https://www.apache.org/licenses/LICENSE-2.0"
        )
    ),
    servers(
        (url = "http://localhost:9999", description = "本地开发服务器"),
        (url = "https://api.dxmt.tech", description = "生产环境API")
    )
)]
pub struct ApiDoc;

/// 创建Swagger UI路由 - 仅在 /api/docs 提供
pub fn create_swagger_ui<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/api/docs", axum::routing::get(swagger_ui_handler))
        .route("/api/docs/*path", axum::routing::get(swagger_ui_handler))
        .route("/api/docs/openapi.json", axum::routing::get(openapi_handler))
}

async fn swagger_ui_handler() -> axum::response::Html<String> {
    let html = r#"<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DXMT VectorDB API Documentation</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.10.3/swagger-ui.css">
    <style>
        body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .swagger-ui .topbar { display: none; }
        .swagger-ui .info { margin-bottom: 20px; }
        .swagger-ui .info .title { font-size: 36px; color: #333; }
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
                persistAuthorization: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            });
        };
    </script>
</body>
</html>"#;
    axum::response::Html(html.to_string())
}

async fn openapi_handler() -> axum::Json<utoipa::openapi::OpenApi> {
    axum::Json(ApiDoc::openapi())
}
