use crate::api::response::ApiResponse;
use axum::{
    extract::{Path, Query, State},
    http::{
        header::{HeaderMap, HeaderValue},
        StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::storage::traits::_Storage as _;
use crate::{
    api::response::ApiError,
    api::server_health::ServerHealth,
    storage::{BackupManager, IntelligentIndexedStorage, RestoreOptions},
    types::{
        record::CreateRecordRequest,
        record::HybridSearchQuery,
        record::SearchQuery,
        record::{FullTextSearchRequest, HybridSearchRequest},
        table::IndexType,
        table::{TableAlterRequest, TableConfig},
        vector::DistanceMetric,
    },
    utils::{validate_record_id, validate_table_name},
};

/// Helper function to convert Error into appropriate HTTP response
/// Uses the error's status_code() method to return correct HTTP status codes
fn error_response(err: crate::Error) -> Response {
    let status =
        StatusCode::from_u16(err.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, ApiResponse::error(err.into())).into_response()
}

/// 应用状态
#[derive(Clone)]
pub struct AppState {
    pub storage: Arc<IntelligentIndexedStorage>,
    pub backup_manager: Option<Arc<BackupManager>>,
    pub server_health: Arc<ServerHealth>,
    // Extensions will be handled through request extractors instead
}

/// 分页查询参数
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct PaginationQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// 创建表的请求
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct CreateTableRequest {
    pub name: String,
    pub dimension: usize,
    pub distance_metric: Option<String>,
    pub description: Option<String>,
    pub index_type: Option<String>,
    pub text_fields: Option<Vec<crate::types::table::TextFieldConfig>>,
    #[serde(default)]
    pub enable_full_text_search: Option<bool>,
    /// 是否启用内存优化（可选，None = 使用全局配置）
    #[serde(default)]
    pub memory_optimization: Option<bool>,
}

/// 批量插入记录的请求
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct BatchInsertRequest {
    pub records: Vec<CreateRecordRequest>,
}

/// 优化的批量插入请求（支持不同的插入模式）
#[derive(Debug, Deserialize)]
pub struct OptimizedBatchInsertRequest {
    pub records: Vec<CreateRecordRequest>,
    #[serde(default = "default_insert_mode")]
    pub mode: BulkInsertMode,
}

/// 批量插入模式
#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum BulkInsertMode {
    /// 标准批量插入
    Standard,
    /// 无锁高性能插入
    LockFree,
    /// 流式处理插入
    Streaming,
    /// 内存池优化插入
    MemoryPool,
    /// 延迟索引构建
    DeferredIndex,
    /// 批量优化插入
    BulkOptimized,
    /// Ultra-fast columnar insert
    UltraFast,
    /// VM resource-optimized insert
    VmOptimized,
    /// NVMe SSD optimized insert
    NvmeOptimized,
    /// Ultimate performance mode (all optimizations)
    Ultimate,
}

fn default_insert_mode() -> BulkInsertMode {
    BulkInsertMode::Standard
}

// 添加验证常量
const MAXBATCH_SIZE: usize = 10000; // 增加到10000以支持更大批量
const MAX_TABLE_NAME_LENGTH: usize = 64;
const MAX_RECORD_ID_LENGTH: usize = 128;
const MAX_DIMENSION: usize = 4096;
const MIN_DIMENSION: usize = 1;

/// Comprehensive batch validation following Weaviate/Qdrant patterns
fn validate_batch_insert(
    request: &BatchInsertRequest,
    table: &crate::types::table::Table,
) -> Result<(), String> {
    use crate::types::vector::VectorValidator;
    use std::collections::HashSet;

    // 1. Batch size limits (max 10,000)
    if request.records.is_empty() {
        return Err("Batch cannot be empty".to_string());
    }

    if request.records.len() > MAXBATCH_SIZE {
        return Err(format!(
            "Batch size {} exceeds maximum of {}",
            request.records.len(),
            MAXBATCH_SIZE
        ));
    }

    // 2. ID uniqueness within batch
    let mut seen_ids = HashSet::new();
    for (idx, record) in request.records.iter().enumerate() {
        if !seen_ids.insert(&record.id) {
            return Err(format!(
                "Duplicate ID '{}' found at position {}",
                record.id, idx
            ));
        }
    }

    // 3. Validate each record (ID, vector, sparse_vector)
    for (idx, record) in request.records.iter().enumerate() {
        // ID validation
        if record.id.is_empty() {
            return Err(format!("Record at position {} has empty ID", idx));
        }

        if record.id.len() > MAX_RECORD_ID_LENGTH {
            return Err(format!(
                "Record ID at position {} exceeds maximum length of {} characters",
                idx, MAX_RECORD_ID_LENGTH
            ));
        }

        if let Err(e) = validate_record_id(&record.id) {
            return Err(format!("Invalid record ID at position {}: {}", idx, e));
        }

        // Dense vector validation
        if let Err(e) = VectorValidator::validate(&record.vector, table.dimension) {
            return Err(format!(
                "Invalid vector at position {} (ID: '{}'): {}",
                idx, record.id, e
            ));
        }

        // Sparse vector validation (if present)
        if let Some(ref sparse_vec) = record.sparse_vector {
            if let Err(e) = sparse_vec.validate() {
                return Err(format!(
                    "Invalid sparse vector at position {} (ID: '{}'): {}",
                    idx, record.id, e
                ));
            }
        }
    }

    Ok(())
}

/// Payload update mode: replace or merge
#[derive(Debug, Default, Deserialize, Serialize, Clone, Copy)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum PayloadUpdateMode {
    /// Replace entire payload (default for backward compatibility)
    #[default]
    Replace,
    /// Merge with existing payload
    Merge,
}

/// 更新记录的请求
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct UpdateRecordRequest {
    pub vector: Option<Vec<f32>>,
    pub payload: Option<serde_json::Value>,
    /// Payload update mode: "replace" (default) or "merge"
    #[serde(default)]
    pub merge_mode: PayloadUpdateMode,
    /// Whether to create record if it doesn't exist (default: false)
    #[serde(default)]
    pub upsert: bool,
}

/// 批量删除记录的请求
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct BatchDeleteRequest {
    pub ids: Vec<String>,
}

/// 按元数据列出记录的请求
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct ListRecordsRequest {
    /// Optional filter to apply to record payloads
    pub filter: Option<crate::types::Filter>,
    /// Maximum number of records to return (default: 100, max: 1000)
    pub limit: Option<usize>,
    /// Number of records to skip for pagination (default: 0)
    pub offset: Option<usize>,
}

// ============ 表管理API ============

/// 创建表
#[utoipa::path(
    post,
    path = "/api/v1/tables",
    request_body = CreateTableRequest,
    responses(
        (status = 200, description = "创建表成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 409, description = "表已存在", body = ApiResponse)
    ),
    tag = "tables"
)]
#[axum::debug_handler]
pub async fn create_table(
    State(state): State<AppState>,
    Json(request): Json<CreateTableRequest>,
) -> impl IntoResponse {
    // 增强输入验证
    if request.name.len() > MAX_TABLE_NAME_LENGTH {
        let error = ApiError::new(
            "INVALID_TABLE_NAME",
            &format!(
                "Table name exceeds maximum length of {}",
                MAX_TABLE_NAME_LENGTH
            ),
        );
        return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
    }

    if request.dimension > MAX_DIMENSION || request.dimension < MIN_DIMENSION {
        let error = ApiError::new(
            "INVALID_INPUT",
            &format!(
                "Dimension must be between {} and {}",
                MIN_DIMENSION, MAX_DIMENSION
            ),
        );
        return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
    }

    // 验证表名
    if let Err(e) = validate_table_name(&request.name) {
        let error = ApiError::new("INVALID_TABLE_NAME", &e.to_string());
        return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
    }

    // 解析距离度量
    let distance_metric = match request
        .distance_metric
        .as_deref()
        .unwrap_or("cosine")
        .parse::<DistanceMetric>()
    {
        Ok(metric) => metric,
        Err(e) => {
            let error = ApiError::new(
                "INVALID_DISTANCE_METRIC",
                &format!("Invalid distance metric: {}", e),
            );
            return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
        }
    };

    // 解析索引类型
    let index_type = match request
        .index_type
        .as_deref()
        .map(|s| s.parse::<IndexType>())
        .transpose()
    {
        Ok(index_type) => index_type,
        Err(e) => {
            let error = ApiError::new("INVALID_INDEX_TYPE", &format!("Invalid index type: {}", e));
            return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
        }
    };

    let config = TableConfig {
        name: request.name,
        dimension: request.dimension,
        distance_metric,
        description: request.description,
        index_type,
        text_fields: request.text_fields,
        enable_full_text_search: request.enable_full_text_search.unwrap_or(false),
        memory_optimization: request.memory_optimization,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    };

    match state.storage.create_table(config).await {
        Ok(table) => {
            let value = serde_json::to_value(table).unwrap_or(serde_json::Value::Null);
            ApiResponse::success(value).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 获取表信息
#[utoipa::path(
    get,
    path = "/api/v1/tables/{table_name}",
    params(
        ("table_name" = String, Path, description = "表名")
    ),
    responses(
        (status = 200, description = "获取表信息成功", body = TableWithCount),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "tables"
)]
#[axum::debug_handler]
pub async fn get_table(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
) -> impl IntoResponse {
    match state.storage.get_table(&table_name).await {
        Ok(table) => {
            let resp = crate::types::table::TableWithCount {
                record_count: table.record_count as usize,
                table,
            };
            let value = serde_json::to_value(resp).unwrap_or(serde_json::Value::Null);
            ApiResponse::success(value).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 列出所有表
#[utoipa::path(
    get,
    path = "/api/v1/tables",
    params(
        ("limit" = Option<usize>, Query, description = "返回记录数量限制"),
        ("offset" = Option<usize>, Query, description = "偏移量")
    ),
    responses(
        (status = 200, description = "获取表列表成功", body = ApiResponse)
    ),
    tag = "tables"
)]
#[axum::debug_handler]
pub async fn list_tables(
    State(state): State<AppState>,
    Query(pagination): Query<PaginationQuery>,
) -> impl IntoResponse {
    let tables = match state.storage.list_tables().await {
        Ok(tables) => tables,
        Err(e) => return error_response(e),
    };

    // 应用分页
    let offset = pagination.offset.unwrap_or(0);
    let limit = pagination.limit.unwrap_or(50);

    let total = tables.len();
    let paginatedtables: Vec<_> = tables.into_iter().skip(offset).take(limit).collect();

    let response = serde_json::json!({
        "tables": paginatedtables,
        "total": total,
        "offset": offset,
        "limit": limit
    });

    ApiResponse::success(response).into_response()
}

/// 删除表
#[utoipa::path(
    delete,
    path = "/api/v1/tables/{table_name}",
    params(
        ("table_name" = String, Path, description = "表名")
    ),
    responses(
        (status = 200, description = "删除表成功", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "tables"
)]
#[axum::debug_handler]
pub async fn delete_table(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
) -> impl IntoResponse {
    match state.storage.delete_table(&table_name).await {
        Ok(_) => {
            let response = serde_json::json!({
                "message": format!("Table '{}' deleted successfully", table_name)
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// Alter table schema — safe mutations only (description, memory_optimization,
/// enable_full_text_search toggle, add/drop payload field declarations).
///
/// PATCH /api/v1/tables/:table_name
pub async fn alter_table(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(req): Json<TableAlterRequest>,
) -> impl IntoResponse {
    match state.storage.alter_table(&table_name, req).await {
        Ok(table) => {
            ApiResponse::success(serde_json::to_value(table).unwrap_or_default()).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// Drop the vector index of a table without deleting records.
///
/// `DELETE /api/v1/tables/:table_name/index`
///
/// Frees ANN index memory (HNSW graph, PQ codebooks, etc.) and replaces it
/// with a lightweight brute-force index. Records are preserved. Useful when
/// you want to reclaim memory and don't need ANN search performance temporarily.
pub async fn drop_index(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
) -> impl IntoResponse {
    match state.storage.drop_index(&table_name).await {
        Ok(_) => {
            let response = serde_json::json!({
                "message": format!(
                    "Vector index for table '{}' has been dropped. Records are preserved; \
                     searches will use brute-force until the index is rebuilt.",
                    table_name
                )
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// Rebuild the vector index from persisted records.
///
/// POST /tables/:table_name/index/rebuild
///
/// Adaptively re-selects the best algorithm for the current dataset size, swaps in a
/// fresh empty index, then re-inserts all records. Use this after `drop_index` or on
/// cold start to restore ANN performance.
pub async fn rebuild_index(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
) -> impl IntoResponse {
    match state.storage.rebuild_index(&table_name).await {
        Ok(_) => {
            let response = serde_json::json!({
                "message": format!(
                    "Vector index for table '{}' has been rebuilt from {} persisted records.",
                    table_name, ""
                ),
                "table": table_name,
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// Write a snapshot of every table so the next restart skips replaying the whole WAL.
///
/// Explicit rather than automatic: snapshotting walks every table and serializes every index,
/// so when it runs is an operational decision. It also runs after `index/rebuild`, which is the
/// other moment the index is known to be both current and expensive to reproduce.
pub async fn create_snapshot(State(state): State<AppState>) -> impl IntoResponse {
    match state.storage.snapshot().await {
        Ok(tables) => ApiResponse::success(serde_json::json!({
            "message": format!("Snapshot written for {} table(s).", tables),
            "tables": tables,
        }))
        .into_response(),
        Err(e) => error_response(e),
    }
}

/// Branch a table — create an independent copy of source table's schema and records.
///
/// POST /tables/:table_name/branch
/// Body: { "branch_name": "my-branch" }
pub async fn branch_table(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let branch_name = match body.get("branch_name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                ApiResponse::error(ApiError::new("BAD_REQUEST", "branch_name is required")),
            )
                .into_response();
        }
    };

    match state.storage.branch_table(&table_name, &branch_name).await {
        Ok(branch) => {
            let response = serde_json::json!({
                "source": table_name,
                "branch": branch_name,
                "table": branch,
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

// ============ 记录操作API ============

/// 批量插入记录
#[utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/records",
    params(
        ("table_name" = String, Path, description = "表名")
    ),
    request_body = BatchInsertRequest,
    responses(
        (status = 200, description = "插入记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "records"
)]
#[axum::debug_handler]
pub async fn insert_records(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(request): Json<BatchInsertRequest>,
) -> impl IntoResponse {
    // Check server health and apply backpressure if needed
    let (should_accept, health_state) = state.server_health.should_accept_insert().await;

    if !should_accept {
        // Server is overloaded, return 429 with Retry-After header
        let mut headers = HeaderMap::new();
        let retry_after = health_state.retry_after_seconds();
        if let Ok(value) = HeaderValue::from_str(&retry_after.to_string()) {
            headers.insert("Retry-After", value);
        }

        let error_response = serde_json::json!({
            "error": "Server is currently overloaded",
            "details": health_state.description(),
            "retry_after_seconds": retry_after,
            "suggestion": format!("Please retry your request after {} seconds. The server is under heavy load due to index building or resource constraints.", retry_after)
        });

        return (StatusCode::TOO_MANY_REQUESTS, headers, Json(error_response)).into_response();
    }

    // Get table for validation
    let table = match state.storage.get_table(&table_name).await {
        Ok(table) => table,
        Err(e) => {
            let error_response = serde_json::json!({
                "error": "Table not found",
                "details": format!("{:?}", e)
            });
            return (StatusCode::NOT_FOUND, Json(error_response)).into_response();
        }
    };

    // Comprehensive batch validation
    if let Err(e) = validate_batch_insert(&request, &table) {
        let error_response = serde_json::json!({
            "error": "Validation failed",
            "details": e
        });
        return (StatusCode::BAD_REQUEST, Json(error_response)).into_response();
    }

    // Start tracking this operation
    let _guard = state.server_health.start_insert();

    match state
        .storage
        .insert_records(&table_name, request.records)
        .await
    {
        Ok(inserted_records) => {
            let ids: Vec<&str> = inserted_records.iter().map(|r| r.id.as_str()).collect();
            let response = serde_json::json!({
                "inserted_ids": ids,
                "count": ids.len()
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => {
            // Record the error for health tracking
            state.server_health.record_error();

            // Return detailed error with suggestions
            let error_response = serde_json::json!({
                "error": "Failed to insert records",
                "details": format!("{:?}", e),
                "suggestion": "If this error persists, the server may be under heavy load. Consider retrying with exponential backoff or reducing batch size."
            });

            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response()
        }
    }
}

/// 获取单个记录
#[utoipa::path(
    get,
    path = "/api/v1/tables/{table_name}/records/{record_id}",
    params(
        ("table_name" = String, Path, description = "表名"),
        ("record_id" = String, Path, description = "记录ID")
    ),
    responses(
        (status = 200, description = "获取记录成功", body = ApiResponse),
        (status = 404, description = "记录不存在", body = ApiResponse)
    ),
    tag = "records"
)]
#[axum::debug_handler]
pub async fn get_record(
    State(state): State<AppState>,
    Path((table_name, record_id)): Path<(String, String)>,
) -> impl IntoResponse {
    match state.storage.get_record(&table_name, &record_id).await {
        Ok(record) => {
            let value = serde_json::to_value(record).unwrap_or(serde_json::Value::Null);
            ApiResponse::success(value).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 更新记录
#[utoipa::path(
    put,
    path = "/api/v1/tables/{table_name}/records/{record_id}",
    params(
        ("table_name" = String, Path, description = "表名"),
        ("record_id" = String, Path, description = "记录ID")
    ),
    request_body = UpdateRecordRequest,
    responses(
        (status = 200, description = "更新记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "记录不存在", body = ApiResponse)
    ),
    tag = "records"
)]
#[axum::debug_handler]
pub async fn update_record(
    State(state): State<AppState>,
    Path((table_name, record_id)): Path<(String, String)>,
    Json(request): Json<UpdateRecordRequest>,
) -> impl IntoResponse {
    // 1. Validate vector dimension if provided
    if let Some(ref vector) = request.vector {
        // Get table to check dimension
        match state.storage.get_table(&table_name).await {
            Ok(table) => {
                if vector.len() != table.dimension {
                    let details = serde_json::json!({
                        "expected_dimension": table.dimension,
                        "actual_dimension": vector.len()
                    });
                    let error = ApiError::with_details(
                        "DIMENSION_MISMATCH",
                        &format!(
                            "Vector dimension mismatch: expected {}, got {}",
                            table.dimension,
                            vector.len()
                        ),
                        details,
                    );
                    return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
                }
            }
            Err(e) => {
                let error = ApiError::new("TABLE_NOT_FOUND", &format!("Table not found: {}", e));
                return (StatusCode::NOT_FOUND, ApiResponse::error(error)).into_response();
            }
        }
    }

    // 2. Check if record exists (if upsert=false)
    let existing_record = state.storage.get_record(&table_name, &record_id).await;

    if !request.upsert && existing_record.is_err() {
        let details = serde_json::json!({
            "record_id": record_id,
            "hint": "Set upsert=true to create if not exists"
        });
        let error = ApiError::with_details(
            "RECORD_NOT_FOUND",
            &format!("Record '{}' does not exist", record_id),
            details,
        );
        return (StatusCode::NOT_FOUND, ApiResponse::error(error)).into_response();
    }

    // 3. Prepare payload (merge if needed) and resolve final vector
    let existing_ok = existing_record.ok();
    let final_vector = match request.vector {
        Some(v) => v,
        None => existing_ok
            .as_ref()
            .map(|r: &crate::types::record::Record| r.vector.clone())
            .unwrap_or_default(),
    };

    let final_payload = match request.merge_mode {
        PayloadUpdateMode::Merge => {
            // If merge mode, read existing payload and merge
            if let Some(existing) = &existing_ok {
                let mut merged = existing.payload.clone();
                if let Some(new_payload) = request.payload {
                    // Merge new payload into existing
                    if let (Some(merged_obj), Some(new_obj)) =
                        (merged.as_object_mut(), new_payload.as_object())
                    {
                        for (key, value) in new_obj {
                            merged_obj.insert(key.clone(), value.clone());
                        }
                    } else {
                        // If not both objects, replace
                        merged = new_payload;
                    }
                }
                Some(merged)
            } else {
                // No existing record, use new payload as-is
                request.payload
            }
        }
        PayloadUpdateMode::Replace => {
            // Replace mode: use new payload as-is
            request.payload
        }
    };

    // 4. Prepare final update request
    let update_request = CreateRecordRequest {
        id: record_id.clone(),
        vector: final_vector,
        sparse_vector: None,
        payload: final_payload,
        named_vectors: None,
    };

    // 5. Perform update
    match state
        .storage
        .update_record(&table_name, &record_id, update_request)
        .await
    {
        Ok(updated_record) => {
            let response = serde_json::json!({
                "message": format!("Record '{}' updated successfully", record_id),
                "record": {
                    "id": updated_record.id,
                    "updated_at": updated_record.updated_at,
                    "version": updated_record.version
                }
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => {
            let error = ApiError::new("UPDATE_FAILED", &format!("Failed to update record: {}", e));
            (StatusCode::INTERNAL_SERVER_ERROR, ApiResponse::error(error)).into_response()
        }
    }
}

/// 批量删除记录
#[utoipa::path(
    delete,
    path = "/api/v1/tables/{table_name}/records",
    params(
        ("table_name" = String, Path, description = "表名")
    ),
    request_body = BatchDeleteRequest,
    responses(
        (status = 200, description = "删除记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "records"
)]
#[axum::debug_handler]
pub async fn delete_records(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(request): Json<BatchDeleteRequest>,
) -> impl IntoResponse {
    let requested_count = request.ids.len();
    match state.storage.delete_records(&table_name, request.ids).await {
        Ok(()) => {
            let response = serde_json::json!({
                "deleted_count": requested_count,
                "requested_count": requested_count
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

// ============ 搜索API ============

/// 向量搜索
#[utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/search",
    params(
        ("table_name" = String, Path, description = "表名")
    ),
    request_body = SearchQuery,
    responses(
        (status = 200, description = "搜索成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "search"
)]
#[axum::debug_handler]
pub async fn search_vectors(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(query): Json<SearchQuery>,
) -> impl IntoResponse {
    if query.vector.is_empty() {
        let error = ApiError::new("INVALID_VECTOR", "Query vector must not be empty");
        return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
    }
    if query.top_k == 0 {
        let error = ApiError::new("INVALID_INPUT", "top_k must be greater than 0");
        return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
    }
    match state.storage.search(&table_name, query).await {
        Ok(result) => {
            let value = serde_json::to_value(result).unwrap_or(serde_json::Value::Null);
            ApiResponse::success(value).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 按元数据过滤列出记录（无需向量查询）
#[utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/records/list",
    params(
        ("table_name" = String, Path, description = "表名")
    ),
    request_body = ListRecordsRequest,
    responses(
        (status = 200, description = "列出记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "records"
)]
#[axum::debug_handler]
pub async fn list_records(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(request): Json<ListRecordsRequest>,
) -> impl IntoResponse {
    let limit = request.limit.unwrap_or(100).min(1000); // Max 1000 records per request
    let offset = request.offset.unwrap_or(0);

    match state
        .storage
        .list_records(&table_name, request.filter.as_ref(), limit, offset)
        .await
    {
        Ok((records, total)) => {
            let response = serde_json::json!({
                "records": records,
                "total": total,
                "limit": limit,
                "offset": offset,
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 混合搜索（向量 + 文本 + 稀疏向量）
#[utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/hybrid_search",
    params(
        ("table_name" = String, Path, description = "表名")
    ),
    request_body = HybridSearchQuery,
    responses(
        (status = 200, description = "搜索成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "search"
)]
#[axum::debug_handler]
pub async fn hybrid_search(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(query): Json<HybridSearchQuery>,
) -> impl IntoResponse {
    // For now, convert to regular search if only dense vector is provided
    if query.dense_vector.is_some() && query.sparse_vector.is_none() && query.text_query.is_none() {
        // Safe to unwrap here because of the is_some() check above
        let dense_vector = match query.dense_vector {
            Some(v) => v,
            None => {
                return ApiResponse::error(crate::api::response::ApiError::new(
                    "BAD_REQUEST",
                    "Dense vector is required for this search type",
                ))
                .into_response();
            }
        };

        let search_query = SearchQuery {
            vector: dense_vector,
            top_k: query.top_k,
            filter: query.filter,
            with_payload: query.with_payload,
            with_vector: query.with_vector,
            limit: query.top_k,
            include_metadata: query.with_payload,
            ..Default::default()
        };

        match state.storage.search(&table_name, search_query).await {
            Ok(result) => {
                let value = serde_json::to_value(result).unwrap_or(serde_json::Value::Null);
                ApiResponse::success(value).into_response()
            }
            Err(e) => error_response(e),
        }
    } else {
        // Anything beyond a lone dense vector cannot be served here. This branch used to run
        // the dense search, drop the sparse and text components on the floor, and return the
        // result as if the requested fusion had happened — the caller's alpha/beta/gamma
        // weights changed nothing. Refuse, and name the endpoints that do work.
        ApiResponse::error(crate::api::response::ApiError::new(
            "NOT_IMPLEMENTED",
            "This endpoint only serves a lone dense_vector; it fuses nothing. For BM25 use \
             POST /tables/{table}/search/text, for vector+text fusion use \
             POST /tables/{table}/search/hybrid. Sparse-vector fusion is not implemented.",
        ))
        .into_response()
    }
}

// ============ 系统API ============

/// 获取系统状态
#[utoipa::path(
    get,
    path = "/status",
    responses(
        (status = 200, description = "系统健康状态", body = ApiResponse)
    ),
    tag = "system"
)]
#[axum::debug_handler]
pub async fn get_status(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state.storage.get_stats().await;

    let status = serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "storage": stats
    });

    ApiResponse::success(status).into_response()
}

/// 健康检查端点（用于 Consul）
#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "服务健康"),
        (status = 503, description = "服务不健康")
    ),
    tag = "system"
)]
/// Server health check endpoint
///
/// Returns the current health state of the server including operation metrics.
/// Use this endpoint to monitor server load and determine if backpressure is active.
pub async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let health_state = state.server_health.current_state().await;
    let metrics = state.server_health.get_metrics();

    // 检查存储系统是否正常
    let storage_healthy = !state.storage.get_stats().await.is_empty();

    let response = serde_json::json!({
        "status": format!("{:?}", health_state),
        "service": "vectordb",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "storage_healthy": storage_healthy,
        "description": health_state.description(),
        "retry_after_seconds": health_state.retry_after_seconds(),
        "metrics": {
            "ongoing_inserts": metrics.ongoing_inserts,
            "ongoing_searches": metrics.ongoing_searches,
            "ongoing_deletes": metrics.ongoing_deletes,
            "ongoing_table_ops": metrics.ongoing_table_ops,
            "total_operations": metrics.total_operations,
            "recent_errors": metrics.recent_errors,
            "index_building": metrics.index_building,
        },
        "accepting_requests": {
            "inserts": health_state != crate::api::server_health::HealthState::Critical,
            "searches": true,
        }
    });

    ApiResponse::success(response).into_response()
}

// ============ 备份恢复API ============

/// 创建备份请求
#[derive(Debug, Deserialize)]
pub struct CreateBackupRequest {
    pub backup_type: Option<String>, // "full", "incremental", "manual"
    pub tables: Option<Vec<String>>, // 指定要备份的表，为空则备份所有表
    pub description: Option<String>,
}

/// 恢复备份请求
#[derive(Debug, Deserialize)]
pub struct RestoreBackupRequest {
    pub backup_id: String,
    pub target_tables: Option<Vec<String>>,
    pub point_in_time: Option<String>, // ISO 8601 格式
    pub verify_integrity: Option<bool>,
    pub parallel_threads: Option<usize>,
    pub overwrite_existing: Option<bool>,
}

/// 创建全量备份
#[utoipa::path(
    post,
    path = "/api/v1/backup/full",
    request_body = CreateBackupRequest,
    responses(
        (status = 200, description = "备份创建成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 500, description = "备份创建失败", body = ApiResponse)
    ),
    tag = "backup"
)]
#[axum::debug_handler]
pub async fn create_full_backup(
    State(state): State<AppState>,
    Json(_request): Json<CreateBackupRequest>,
) -> impl IntoResponse {
    let backup_manager = match &state.backup_manager {
        Some(manager) => manager,
        None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    match backup_manager.create_full_backup().await {
        Ok(task_id) => {
            let response = serde_json::json!({
                "task_id": task_id,
                "message": "Full backup task created successfully"
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 创建增量备份
#[utoipa::path(
    post,
    path = "/api/v1/backup/incremental",
    request_body = CreateBackupRequest,
    responses(
        (status = 200, description = "增量备份创建成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 500, description = "备份创建失败", body = ApiResponse)
    ),
    tag = "backup"
)]
#[axum::debug_handler]
pub async fn create_incremental_backup(
    State(state): State<AppState>,
    Json(_request): Json<CreateBackupRequest>,
) -> impl IntoResponse {
    let backup_manager = match &state.backup_manager {
        Some(manager) => manager,
        None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    match backup_manager.create_incremental_backup().await {
        Ok(task_id) => {
            let response = serde_json::json!({
                "task_id": task_id,
                "message": "Incremental backup task created successfully"
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 创建手动备份
#[utoipa::path(
    post,
    path = "/api/v1/backup/manual",
    request_body = CreateBackupRequest,
    responses(
        (status = 200, description = "手动备份创建成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 500, description = "备份创建失败", body = ApiResponse)
    ),
    tag = "backup"
)]
#[axum::debug_handler]
pub async fn create_manual_backup(
    State(state): State<AppState>,
    Json(request): Json<CreateBackupRequest>,
) -> impl IntoResponse {
    let backup_manager = match &state.backup_manager {
        Some(manager) => manager,
        None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    match backup_manager.create_manual_backup(request.tables).await {
        Ok(task_id) => {
            let response = serde_json::json!({
                "task_id": task_id,
                "message": "Manual backup task created successfully"
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 获取备份状态
#[utoipa::path(
    get,
    path = "/api/v1/backup/status",
    responses(
        (status = 200, description = "备份状态", body = ApiResponse),
        (status = 500, description = "获取状态失败", body = ApiResponse)
    ),
    tag = "backup"
)]
#[axum::debug_handler]
pub async fn get_backup_status(State(state): State<AppState>) -> impl IntoResponse {
    let backup_manager = match &state.backup_manager {
        Some(manager) => manager,
        None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    let backup_state = backup_manager.get_backup_state();
    let value = serde_json::to_value(backup_state).unwrap_or(serde_json::Value::Null);
    ApiResponse::success(value).into_response()
}

/// 列出所有备份
#[utoipa::path(
    get,
    path = "/api/v1/backup/list",
    responses(
        (status = 200, description = "备份列表", body = ApiResponse),
        (status = 500, description = "获取备份列表失败", body = ApiResponse)
    ),
    tag = "backup"
)]
#[axum::debug_handler]
pub async fn list_backups(State(state): State<AppState>) -> impl IntoResponse {
    let backup_manager = match &state.backup_manager {
        Some(manager) => manager,
        None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    let backups = backup_manager.list_backups();
    let value = serde_json::to_value(backups).unwrap_or(serde_json::Value::Null);
    ApiResponse::success(value).into_response()
}

/// 恢复备份
#[utoipa::path(
    post,
    path = "/api/v1/backup/restore",
    request_body = RestoreBackupRequest,
    responses(
        (status = 200, description = "恢复成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "备份不存在", body = ApiResponse),
        (status = 500, description = "恢复失败", body = ApiResponse)
    ),
    tag = "backup"
)]
#[axum::debug_handler]
pub async fn restore_backup(
    State(state): State<AppState>,
    Json(request): Json<RestoreBackupRequest>,
) -> impl IntoResponse {
    let backup_manager = match &state.backup_manager {
        Some(manager) => manager,
        None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    // 解析备份ID
    let backup_id = match uuid::Uuid::parse_str(&request.backup_id) {
        Ok(id) => id,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    // 解析时间点（如果提供）
    let point_in_time = if let Some(time_str) = request.point_in_time {
        match chrono::DateTime::parse_from_rfc3339(&time_str) {
            Ok(dt) => Some(dt.with_timezone(&chrono::Utc)),
            Err(_) => return StatusCode::BAD_REQUEST.into_response(),
        }
    } else {
        None
    };

    // 构建恢复选项
    let options = RestoreOptions {
        target_tables: request.target_tables,
        point_in_time,
        verify_integrity: request.verify_integrity.unwrap_or(true),
        parallel_threads: request.parallel_threads.unwrap_or(4),
        overwrite_existing: request.overwrite_existing.unwrap_or(false),
    };

    match backup_manager.restore_backup(backup_id, options).await {
        Ok(_) => {
            let response = serde_json::json!({
                "message": "Backup restored successfully"
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 删除备份
#[utoipa::path(
    delete,
    path = "/api/v1/backup/{backup_id}",
    params(
        ("backup_id" = String, Path, description = "备份ID")
    ),
    responses(
        (status = 200, description = "备份删除成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "备份不存在", body = ApiResponse),
        (status = 500, description = "删除失败", body = ApiResponse)
    ),
    tag = "backup"
)]
#[axum::debug_handler]
pub async fn delete_backup(
    State(state): State<AppState>,
    Path(backup_id): Path<String>,
) -> impl IntoResponse {
    let backup_manager = match &state.backup_manager {
        Some(manager) => manager,
        None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    // 解析备份ID
    let backup_id = match uuid::Uuid::parse_str(&backup_id) {
        Ok(id) => id,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    match backup_manager.delete_backup(backup_id).await {
        Ok(_) => {
            let response = serde_json::json!({
                "message": "Backup deleted successfully"
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

// ============ 优化的批量插入API ============

/// 优化的批量插入记录
#[utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/records/bulk",
    params(
        ("table_name" = String, Path, description = "表名")
    ),
    request_body = OptimizedBatchInsertRequest,
    responses(
        (status = 200, description = "插入记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "records"
)]
#[axum::debug_handler]
pub async fn insert_records_optimized(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(request): Json<OptimizedBatchInsertRequest>,
) -> impl IntoResponse {
    // 批量大小验证
    let max_size = match request.mode {
        BulkInsertMode::LockFree | BulkInsertMode::Streaming => 100000, // 10万条
        BulkInsertMode::BulkOptimized | BulkInsertMode::DeferredIndex => 50000, // 5万条
        _ => MAXBATCH_SIZE,
    };

    if request.records.len() > max_size {
        let error = ApiError::new(
            "INVALID_INPUT",
            &format!(
                "Batch size {} exceeds maximum {} for mode {:?}",
                request.records.len(),
                max_size,
                request.mode
            ),
        );
        return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
    }

    if request.records.is_empty() {
        let error = ApiError::new("INVALID_INPUT", "records array must not be empty");
        return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
    }

    // 验证记录ID和向量
    for record in &request.records {
        if record.id.len() > MAX_RECORD_ID_LENGTH {
            let error = ApiError::new(
                "INVALID_INPUT",
                &format!(
                    "Record ID '{}' exceeds maximum length {}",
                    record.id, MAX_RECORD_ID_LENGTH
                ),
            );
            return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
        }

        if let Err(e) = validate_record_id(&record.id) {
            let error = ApiError::new("INVALID_INPUT", &e.to_string());
            return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
        }

        if record.vector.is_empty() {
            let error = ApiError::new(
                "INVALID_VECTOR",
                &format!("Record '{}' has an empty vector", record.id),
            );
            return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
        }
    }

    // 根据模式选择不同的插入方法
    let result = match request.mode {
        BulkInsertMode::Standard => {
            match state
                .storage
                .insert_records(&table_name, request.records)
                .await
            {
                Ok(records) => Ok(records.into_iter().map(|r| r.id).collect()),
                Err(e) => Err(e),
            }
        }
        BulkInsertMode::LockFree => {
            match state
                .storage
                .insert_records_lockfree(&table_name, request.records)
                .await
            {
                Ok(records) => Ok(records),
                Err(e) => Err(e),
            }
        }
        BulkInsertMode::Streaming => {
            match state
                .storage
                .insert_records_streaming(&table_name, request.records)
                .await
            {
                Ok(records) => Ok(records),
                Err(e) => Err(e),
            }
        }
        BulkInsertMode::MemoryPool => {
            match state
                .storage
                .insert_records_with_memory_pool(&table_name, request.records)
                .await
            {
                Ok(records) => Ok(records),
                Err(e) => Err(e),
            }
        }
        BulkInsertMode::DeferredIndex => {
            match state
                .storage
                .insert_records_deferred_index(&table_name, request.records)
                .await
            {
                Ok(records) => Ok(records),
                Err(e) => Err(e),
            }
        }
        BulkInsertMode::BulkOptimized => {
            match state
                .storage
                .insert_records(&table_name, request.records)
                .await
            {
                Ok(records) => Ok(records.into_iter().map(|r| r.id).collect()),
                Err(e) => Err(e),
            }
        }
        BulkInsertMode::UltraFast => {
            match state
                .storage
                .insert_records_ultra_fast(&table_name, request.records)
                .await
            {
                Ok(records) => Ok(records),
                Err(e) => Err(e),
            }
        }
        BulkInsertMode::VmOptimized => {
            match state
                .storage
                .insert_records_vm_optimized(&table_name, request.records)
                .await
            {
                Ok(records) => Ok(records),
                Err(e) => Err(e),
            }
        }
        BulkInsertMode::NvmeOptimized => {
            match state
                .storage
                .insert_records_nvme_optimized(&table_name, request.records)
                .await
            {
                Ok(records) => Ok(records),
                Err(e) => Err(e),
            }
        }
        BulkInsertMode::Ultimate => {
            match state
                .storage
                .insert_records_ultimate(&table_name, request.records)
                .await
            {
                Ok(records) => Ok(records),
                Err(e) => Err(e),
            }
        }
    };

    match result {
        Ok(inserted_ids) => {
            let response = serde_json::json!({
                "inserted_ids": inserted_ids,
                "count": inserted_ids.len(),
                "mode": request.mode
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 全文搜索
#[utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/search/text",
    params(
        ("table_name" = String, Path, description = "表名")
    ),
    request_body = FullTextSearchRequest,
    responses(
        (status = 200, description = "搜索成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse),
        (status = 501, description = "功能未启用", body = ApiResponse)
    ),
    tag = "search"
)]
#[axum::debug_handler]
pub async fn full_text_search(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(request): Json<FullTextSearchRequest>,
) -> impl IntoResponse {
    #[cfg(not(feature = "text-bm25"))]
    {
        return ApiResponse::error(ApiError::new(
            "FEATURE_DISABLED",
            "Full-text search feature is not enabled. Compile with --features text-bm25",
        ))
        .into_response();
    }

    #[cfg(feature = "text-bm25")]
    {
        // Validate table name
        if let Err(e) = validate_table_name(&table_name) {
            return ApiResponse::error(ApiError::new("INVALID_TABLE_NAME", &e.to_string()))
                .into_response();
        }

        // Get table to check if full-text search is enabled
        let table = match state.storage.get_table(&table_name).await {
            Ok(t) => t,
            Err(e) => {
                return ApiResponse::error(ApiError::new(
                    "TABLE_NOT_FOUND",
                    &format!("Table '{}' not found: {}", table_name, e),
                ))
                .into_response();
            }
        };

        if !table.enable_full_text_search {
            return ApiResponse::error(ApiError::new(
                "FULL_TEXT_NOT_ENABLED",
                "Full-text search is not enabled for this table",
            ))
            .into_response();
        }

        // Perform the full-text search
        match state.storage.full_text_search(&table_name, request).await {
            Ok(result) => ApiResponse::success(
                serde_json::to_value(result).unwrap_or(serde_json::Value::Null),
            )
            .into_response(),
            Err(e) => ApiResponse::error(ApiError::new(
                "SEARCH_FAILED",
                &format!("Full-text search failed: {}", e),
            ))
            .into_response(),
        }
    }
}

/// 混合搜索（向量 + 文本组合）
#[utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/search/hybrid",
    params(
        ("table_name" = String, Path, description = "表名")
    ),
    request_body = HybridSearchRequest,
    responses(
        (status = 200, description = "搜索成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "search"
)]
#[axum::debug_handler]
pub async fn hybrid_search_combined(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(request): Json<HybridSearchRequest>,
) -> impl IntoResponse {
    // Validate inputs
    if request.vector.is_none() && request.text.is_none() {
        return ApiResponse::error(ApiError::new(
            "BAD_REQUEST",
            "At least one of 'vector' or 'text' must be provided",
        ))
        .into_response();
    }

    // Validate table name
    if let Err(e) = validate_table_name(&table_name) {
        return ApiResponse::error(ApiError::new("INVALID_TABLE_NAME", &e.to_string()))
            .into_response();
    }

    // Perform hybrid search using the storage layer
    match state.storage.hybrid_search(&table_name, request).await {
        Ok(result) => {
            ApiResponse::success(serde_json::to_value(result).unwrap_or(serde_json::Value::Null))
                .into_response()
        }
        Err(e) => ApiResponse::error(ApiError::new(
            "SEARCH_FAILED",
            &format!("Hybrid search failed: {}", e),
        ))
        .into_response(),
    }
}

/// Execute a DXQL query
///
/// POST /api/v1/query
/// Body: { "query": "SELECT * FROM my_table LIMIT 10" }
pub async fn execute_dxql(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let sql = match body.get("query").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "missing 'query' field"})),
            )
                .into_response();
        }
    };

    // Parse → compile → execute
    use crate::dxql::{compiler::Compiler, executor_impl::StorageAwareExecutor, parser::Parser};
    let stmt = match Parser::new().parse(&sql) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("parse error: {}", e)})),
            )
                .into_response();
        }
    };
    let plan = match Compiler::new().compile(stmt) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("compile error: {}", e)})),
            )
                .into_response();
        }
    };
    let executor = StorageAwareExecutor::new(
        state.storage.clone() as Arc<dyn crate::storage::traits::_Storage>
    );
    match executor.execute(plan).await {
        Ok(result) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "data": {
                    "records": result.records,
                    "metadata": {
                        "total_results": result.metadata.total_results,
                        "execution_time_ms": result.metadata.execution_time_ms,
                        "plan_type": result.metadata.plan_type,
                    }
                }
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": format!("{}", e)})),
        )
            .into_response(),
    }
}

/// Detailed server health metrics endpoint
///
/// Returns comprehensive health metrics for monitoring and debugging.
#[utoipa::path(
    get,
    path = "/api/v1/health/metrics",
    responses(
        (status = 200, description = "Detailed health metrics", body = serde_json::Value),
    ),
    tag = "health"
)]
pub async fn health_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.server_health.get_metrics();

    let response = serde_json::json!({
        "ongoing_operations": {
            "inserts": metrics.ongoing_inserts,
            "searches": metrics.ongoing_searches,
            "deletes": metrics.ongoing_deletes,
            "table_operations": metrics.ongoing_table_ops,
            "total": metrics.ongoing_inserts + metrics.ongoing_searches
                    + metrics.ongoing_deletes + metrics.ongoing_table_ops,
        },
        "completed_operations": metrics.total_operations,
        "errors": {
            "recent_count": metrics.recent_errors,
        },
        "index_status": {
            "building": metrics.index_building,
        }
    });

    ApiResponse::success(response).into_response()
}
