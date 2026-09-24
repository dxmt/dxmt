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
    // Every route the router serves, grouped the way the tags below group them. It listed ten
    // of them before, so nine handlers that carried an annotation never reached the page and
    // the whole of index, maintenance and query was missing -- a reader of the UI would have
    // concluded the server could not snapshot, back up, rebuild an index or run DXQL.
    paths(
        // health
        crate::api::handlers::health_check,
        crate::api::handlers::liveness,
        crate::api::handlers::readiness,
        crate::api::handlers::health_metrics,
        crate::api::handlers::get_status,
        // tables
        crate::api::handlers::create_table,
        crate::api::handlers::list_tables,
        crate::api::handlers::get_table,
        crate::api::handlers::alter_table,
        crate::api::handlers::delete_table,
        crate::api::handlers::get_table_stats,
        crate::api::handlers::branch_table,
        // records
        crate::api::handlers::insert_records,
        crate::api::handlers::insert_records_optimized,
        crate::api::handlers::list_records,
        crate::api::handlers::count_records,
        crate::api::handlers::get_record,
        crate::api::handlers::update_record,
        crate::api::handlers::delete_records,
        // search
        crate::api::handlers::search_vectors,
        crate::api::handlers::search_groups,
        crate::api::handlers::sparse_search,
        crate::api::handlers::full_text_search,
        crate::api::handlers::hybrid_search_combined,
        crate::api::handlers::hybrid_search,
        // index
        crate::api::handlers::rebuild_index,
        crate::api::handlers::drop_index,
        crate::api::handlers::payload_index_report,
        crate::api::handlers::index_payload_field,
        crate::api::handlers::drop_payload_index_field,
        // query
        crate::api::handlers::execute_dxql,
        // maintenance
        crate::api::handlers::create_snapshot,
        crate::api::handlers::create_backup,
        crate::api::handlers::export_table,
        crate::api::handlers::import_table),
    components(
        // Every type a request body names has to be listed here, or utoipa emits a `$ref` to a
        // schema it never defines and the page renders that operation with **no fields at all**.
        // Seven of the thirteen request bodies were in that state -- `/records/list`,
        // `/search/text`, `/search/hybrid`, `/records/bulk`, `/backup`, `/hybrid_search` and the
        // table PATCH -- so the spec described them as taking an empty object. It is also what
        // `tests/api/test_request_fields.py` reads to decide which fields it must cover, so an
        // unregistered schema is a silent hole in that test as well as in the docs.
        schemas(
            Table, TableConfig, CreateRecordRequest, SearchQuery, SearchResult, Record,
            DistanceMetric, IndexType, CreateTableRequest, BatchInsertRequest,
            UpdateRecordRequest, BatchDeleteRequest, PaginationQuery,
            crate::api::handlers::ListRecordsRequest,
            crate::api::handlers::CountRecordsRequest,
            crate::api::handlers::SearchGroupsRequest,
            crate::types::record::SparseSearchQuery,
            crate::types::record::SearchGroup,
            crate::types::record::SearchGroupsResult,
            crate::api::handlers::OptimizedBatchInsertRequest,
            crate::api::handlers::CreateBackupRequest,
            crate::api::handlers::ExportTableRequest,
            crate::api::handlers::ImportTableRequest,
            crate::api::handlers::KnnGraphRequest,
            crate::api::handlers::PayloadUpdateMode,
            crate::types::record::FullTextSearchRequest,
            crate::types::record::HybridSearchRequest,
            crate::types::record::HybridSearchQuery,
            crate::types::record::FusionMethod,
            crate::types::record::SearchMode,
            crate::types::record::Filter,
            crate::types::record::Condition,
            crate::types::record::FilterOperation,
            crate::types::sparse_vector::SparseVector,
            crate::types::table::TableAlterRequest,
            crate::types::table::HnswParams,
            crate::types::table::QuantizationParams,
            crate::types::table::PayloadIndexPolicy,
            crate::types::table::TextFieldConfig,
            crate::api::response::ApiResponse,
            crate::api::response::ApiError,
            crate::types::table::TableInfoResponse)
    ),
    tags(
        (name = "health", description =
            "Probes. /live says the process is up; /ready says a write can be made durable and \
             answers 503 when it cannot. /health and /status are diagnostic documents and are \
             not a substitute for either. Each probe answers at more than one path, and the \
             generated page can only show one of them: /live is also /health/live, /ready is \
             also /health/ready, and all four are mounted under /api/v1 as well, for clients \
             that prefix everything. Same handler, same answer."),
        (name = "tables", description =
            "Create, describe, alter, branch and drop. A table's dimension is fixed when it is \
             created."),
        (name = "records", description =
            "Insert (which is an upsert), read, update and delete. An update with no vector \
             keeps the stored one, along with the sparse and named vectors."),
        (name = "search", description =
            "Nearest neighbours, full-text, and the two hybrid routes. /search/hybrid fuses \
             vector and text; /hybrid_search is dense-only and refuses sparse fusion by name \
             rather than discarding half the query."),
        (name = "index", description =
            "Rebuild or drop a table's index. Dropping one leaves searches answering by brute \
             force -- exactly, and slowly."),
        (name = "query", description = "DXQL, the SQL-shaped query language. See POST /query."),
        (name = "maintenance", description =
            "Snapshot and backup. Both are refused while local durability is degraded, so an \
             unacknowledged write cannot become a new durable frontier.")
    ),
    info(
        title = "DXMT VectorDB API",
        description = "The HTTP surface of DXMT VectorDB. Alpha: see the repository's README \
                       for what is supported and what is experimental.",
        version = "0.2.0",
        contact(
            name = "DXMT Team",
            url = "https://github.com/dxmt/dxmt"
        ),
        license(
            name = "Apache-2.0",
            url = "https://www.apache.org/licenses/LICENSE-2.0"
        )
    ),
    servers(
        (url = "http://localhost:9999", description = "A server started with `vectordb serve`")
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
        .route(
            "/api/docs/openapi.json",
            axum::routing::get(openapi_handler),
        )
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
