use crate::api::extract::{ApiJson, ApiQuery};
use crate::api::response::ApiResponse;
use axum::{
    extract::{Path, State},
    http::{
        header::{HeaderMap, HeaderValue},
        StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    api::response::ApiError,
    api::server_health::ServerHealth,
    storage::IntelligentIndexedStorage,
    types::{
        record::CreateRecordRequest,
        record::HybridSearchQuery,
        record::SearchQuery,
        record::SparseSearchQuery,
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
    pub server_health: Arc<ServerHealth>,
    // Extensions will be handled through request extractors instead
}

/// 分页查询参数
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct PaginationQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// 创建表的请求
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct CreateTableRequest {
    pub name: String,
    /// Vector width, fixed for the life of the table.
    ///
    /// Optional only when `embedding` names a model: the server then asks the model for one
    /// vector and takes its width. A caller who has to look up that `text-embedding-3-small`
    /// is 1,536 and `bge-m3` is 1,024, and gets it wrong, finds out at the first insert —
    /// which is why the model is asked instead of guessed.
    #[serde(default)]
    pub dimension: Option<usize>,
    pub distance_metric: Option<String>,
    pub description: Option<String>,
    pub index_type: Option<String>,
    pub text_fields: Option<Vec<crate::types::table::TextFieldConfig>>,
    #[serde(default)]
    pub enable_full_text_search: Option<bool>,
    /// 是否启用内存优化（可选，None = 使用全局配置）
    #[serde(default)]
    pub memory_optimization: Option<bool>,
    /// Additional vector spaces, each with its own dimension and metric, searchable by name
    /// through `vector_name` on `POST /search`.
    ///
    /// This was missing from this struct while `TableConfig` had it, and the body is
    /// deserialised without `deny_unknown_fields` at the time — so a request declaring named
    /// vector spaces was accepted, the field was dropped, no named index was built, and a later
    /// search naming one of those spaces silently scored against the *default* vector instead.
    /// A wrong answer rather than an error, over a feature the engine has had all along. That
    /// default is what `ApiJson` and the attribute above this struct exist to take away.
    #[serde(default)]
    pub named_vectors: Option<Vec<crate::types::table::VectorSpaceConfig>>,
    /// Declared payload fields. Dropped here for the same reason and with the same shape of
    /// consequence: the table reported no schema, and `PATCH /tables/{t}` was the only way to
    /// get one.
    #[serde(default)]
    pub payload_schema: Option<Vec<crate::types::table::FieldDef>>,
    /// Graph parameters for this table, overriding the server defaults: `m`,
    /// `ef_construction`, `ef_search`. They decide the graph and cannot be recovered
    /// afterwards, and they were a global environment variable until now — the wrong grain for
    /// a server holding a 5,000-row lookup table beside a 10M-row corpus.
    #[serde(default)]
    pub hnsw: Option<crate::types::table::HnswParams>,
    /// Quantization parameters for this table: `num_subvectors`, `num_centroids`,
    /// `quantize_at_rows`. Setting `quantize_at_rows` above the table's eventual size is how to
    /// say "keep this one exact".
    #[serde(default)]
    pub quantization: Option<crate::types::table::QuantizationParams>,
    /// Which payload fields to index for filtering: `only` limits it to those, `except`
    /// removes them. Unset indexes every field, which is what makes a filter fast here without
    /// declaring anything — and what a field nobody filters on pays for. A filter on a field
    /// this excludes is refused rather than answered empty.
    #[serde(default)]
    pub payload_index: Option<crate::types::table::PayloadIndexPolicy>,
    /// The model this table embeds text with: `model`, and optionally `base_url`, `field`,
    /// `api_key_env`, `batch_size`. With it, records and searches may carry `text` instead of
    /// a `vector`. Without it, they may not, and the server takes vectors the caller computed.
    ///
    /// The API key is never in this body — `api_key_env` names an environment variable of the
    /// server process.
    #[serde(default)]
    pub embedding: Option<crate::types::table::EmbeddingConfig>,
    /// A cross-encoder a search may rerank its page with: `model`, `base_url`, and optionally
    /// `format` (`cohere`, the default, or `tei`), `api_key_env`, `field`, `top_n`. A search
    /// opts in with `rerank: {}`; without this, that search is refused. The key is read from
    /// the server's environment, never from this body.
    #[serde(default)]
    pub rerank: Option<crate::types::table::RerankConfig>,
}

/// 批量插入记录的请求
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct BatchInsertRequest {
    pub records: Vec<CreateRecordRequest>,
    /// What a record that fails validation does to the rest of the batch.
    ///
    /// `fail` (the default) writes nothing and answers 400 with every failing row listed
    /// under `error.details.failed`. `skip` writes the rows that pass and lists the rest
    /// under `data.failed` on a 200. The default stays all-or-nothing because a caller who
    /// did not ask for partial success must not get one; `skip` is for the bulk load, where
    /// one malformed row in a million should cost one row and not the batch.
    #[serde(default)]
    pub on_error: OnError,
}

/// See [`BatchInsertRequest::on_error`].
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum OnError {
    #[default]
    Fail,
    Skip,
}

/// One record a batch validation refused: its position in `records`, its id, and why.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct RowError {
    pub index: usize,
    pub id: String,
    pub error: String,
}

/// How many failing rows a refusal lists. The count is always exact; the list is a sample
/// large enough to fix a file by and small enough not to be the file.
const MAX_REPORTED_ROW_ERRORS: usize = 100;

/// 优化的批量插入请求（支持不同的插入模式）
#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
/// The body of `POST /tables/{t}/records/bulk`.
///
/// It used to carry a `mode` with ten values -- `lock_free`, `streaming`, `memory_pool`,
/// `deferred_index`, `bulk_optimized`, `ultra_fast`, `vm_optimized`, `nvme_optimized`,
/// `ultimate` and `standard`. Every one of them reached `insert_records_batch`, whose body is
/// `// For now, just use regular insert` followed by a call to `insert_records`: **one code
/// path behind ten names**, each promising an optimisation that does not exist. The only real
/// difference was the size cap the mode selected.
///
/// The field is gone rather than kept as a set of aliases, because a knob that does nothing is
/// the thing this API had too many of. A body that still sends `mode` is now refused naming it
/// (`deny_unknown_fields`): it was accepted and ignored for a while, and that is precisely how
/// every SDK went on sending it through the release that deleted it.
#[serde(deny_unknown_fields)]
pub struct OptimizedBatchInsertRequest {
    pub records: Vec<CreateRecordRequest>,
}

// 添加验证常量
const MAXBATCH_SIZE: usize = 10000; // 增加到10000以支持更大批量
const MAX_TABLE_NAME_LENGTH: usize = 64;
const MAX_RECORD_ID_LENGTH: usize = 128;
const MAX_DIMENSION: usize = 4096;
const MIN_DIMENSION: usize = 1;

/// Batch validation: the batch-level checks fail outright (`Err`); every per-row defect is
/// collected and returned (`Ok(errors)`), so the caller can list them all or skip them.
///
/// It used to return at the first bad row, which made a bulk load a round trip per defect:
/// a million-row file with fifty bad rows was fifty refusals, each naming one.
fn validate_batch_insert(
    request: &BatchInsertRequest,
    table: &crate::types::table::Table,
) -> Result<Vec<RowError>, String> {
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

    let mut errors = Vec::new();
    let mut seen_ids = HashSet::new();
    for (idx, record) in request.records.iter().enumerate() {
        let mut fail = |error: String| {
            errors.push(RowError {
                index: idx,
                id: record.id.clone(),
                error,
            })
        };

        // 2. ID: non-empty, bounded, well-formed, unique within the batch. The second copy
        // of a duplicated id is the one refused, so the first keeps its place.
        if record.id.is_empty() {
            fail("empty ID".to_string());
            continue;
        }
        if record.id.len() > MAX_RECORD_ID_LENGTH {
            fail(format!(
                "ID exceeds maximum length of {} characters",
                MAX_RECORD_ID_LENGTH
            ));
            continue;
        }
        if let Err(e) = validate_record_id(&record.id) {
            fail(format!("invalid ID: {}", e));
            continue;
        }
        if !seen_ids.insert(&record.id) {
            fail(format!("duplicate ID '{}' within the batch", record.id));
            continue;
        }

        // 3. Dense vector validation.
        //
        // An empty vector is left to the storage layer on a table with an embedding model:
        // that is where the text becomes the vector, and where the record that has neither is
        // named. Validating width here would reject every text record with "expected 1536,
        // got 0" -- an error about a vector the caller never claimed to send.
        // Also deferred when the record carries `text` on a table with *no* model: the storage
        // layer's refusal names the missing model, where "Vector cannot be empty" describes a
        // field the caller never claimed to send and does not say what to do about it.
        let server_may_embed =
            record.vector.is_empty() && (table.embedding.is_some() || record.text.is_some());
        if !server_may_embed {
            if let Err(e) = VectorValidator::validate_for_metric(
                &record.vector,
                table.dimension,
                table.distance_metric,
            ) {
                fail(format!("invalid vector: {}", e));
                continue;
            }
        }

        // 4. Sparse vector validation (if present)
        if let Some(ref sparse_vec) = record.sparse_vector {
            if let Err(e) = sparse_vec.validate() {
                fail(format!("invalid sparse vector: {}", e));
                continue;
            }
        }
    }

    Ok(errors)
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
#[serde(deny_unknown_fields)]
pub struct UpdateRecordRequest {
    /// Optional dense vector replacement. When omitted, the current dense, sparse and named
    /// vectors are preserved; this makes payload-only updates safe for hybrid search.
    pub vector: Option<Vec<f32>>,
    pub payload: Option<serde_json::Value>,
    /// Payload update mode: "replace" (default) or "merge"
    #[serde(default)]
    pub merge_mode: PayloadUpdateMode,
    /// Whether to create record if it doesn't exist (default: false)
    #[serde(default)]
    pub upsert: bool,
    /// New text for the table's model to embed, replacing this record's vector.
    ///
    /// Exclusive with `vector`. Re-embedding a record after its text changed is otherwise a
    /// delete and an insert, and the record loses its `created_at` and its version history.
    #[serde(default)]
    pub text: Option<String>,
}

/// 批量删除记录的请求
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct BatchDeleteRequest {
    /// The records to delete. Exactly one of `ids` and `filter` is given.
    #[serde(default)]
    pub ids: Vec<String>,
    /// Delete everything this filter matches, instead of naming ids.
    ///
    /// Without it, "delete every record belonging to tenant X" could not be expressed: the
    /// caller had to page the table, collect ids and send them back, a round trip per page and
    /// at the mercy of anything written meanwhile. The filter is the same one `search` and
    /// `records/list` take, and is read the same way.
    #[serde(default)]
    pub filter: Option<crate::types::Filter>,
}

/// 按元数据列出记录的请求
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct ListRecordsRequest {
    /// Optional filter to apply to record payloads
    pub filter: Option<crate::types::Filter>,
    /// Maximum number of records to return (default: 100, max: 1000)
    pub limit: Option<usize>,
    /// Number of records to skip for pagination (default: 0). Prefer `cursor`: an offset shifts
    /// when records are inserted while you page, and the next page repeats or skips some.
    pub offset: Option<usize>,
    /// Where to continue from: the `next_cursor` of the previous page. Every record present for
    /// the whole scroll is returned exactly once, however the table is written to meanwhile.
    /// Opaque; cannot be combined with `offset`.
    pub cursor: Option<String>,
    /// Include each record's full vector. Default **false**.
    ///
    /// Off by default because a page of 100 rows at 768 dimensions is over a megabyte of JSON
    /// floats, and almost nothing listing records wants them.
    ///
    /// This field exists because `vector` used to mean two different things here. The handler
    /// returned whatever the record's own field held, which is the vector for an in-memory
    /// table and an **empty array** for one with its vectors in a row file — the default once a
    /// WAL directory exists. So the same request answered differently depending on a server
    /// setting the caller cannot see, and `[]` was indistinguishable from "this record has no
    /// vector". `GET /records/{id}` and `POST /search` with `with_vector` never had that
    /// problem; they read through the row file. Now neither does this.
    pub with_vector: Option<bool>,
}

// ============ 表管理API ============

/// 创建表
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables",
    request_body(content = CreateTableRequest, example = json!({
        "name": "demo",
        "dimension": 4,
        "distance_metric": "cosine",
        "description": "A table to try the routes on",
        "enable_full_text_search": true
    })),
    responses(
        (status = 200, description = "创建表成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 409, description = "表已存在", body = ApiResponse),
        // Creating a table is a write, so it is refused while durability is degraded.
        // Undocumented until the OpenAPI validation saw the server answer it.
        (status = 503, description = "Durability degraded; the server is read-only",
         body = ApiResponse)
    ),
    tag = "tables"
))]
#[axum::debug_handler]
pub async fn create_table(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<CreateTableRequest>,
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

    match request.dimension {
        Some(dimension) if !(MIN_DIMENSION..=MAX_DIMENSION).contains(&dimension) => {
            let error = ApiError::new(
                "INVALID_INPUT",
                &format!(
                    "Dimension must be between {} and {}",
                    MIN_DIMENSION, MAX_DIMENSION
                ),
            );
            return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
        }
        None if request.embedding.is_none() => {
            let error = ApiError::new(
                "INVALID_INPUT",
                "dimension is required, unless `embedding` names a model to take it from",
            );
            return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
        }
        _ => {}
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
        // Zero is the sentinel `create_table` reads as "ask the model", checked above to be
        // reachable only with an `embedding`. `TableConfig::dimension` stays a `usize` because
        // making it an `Option` would touch every one of its sixty construction sites to say
        // something only this one path can mean.
        dimension: request.dimension.unwrap_or(0),
        distance_metric,
        description: request.description,
        index_type,
        text_fields: request.text_fields,
        enable_full_text_search: request.enable_full_text_search.unwrap_or(false),
        memory_optimization: request.memory_optimization,
        payload_schema: request.payload_schema,
        // Server-managed: it counts the migrations `PATCH /tables/{t}` has applied, so a caller
        // choosing its own starting number would be describing a history that did not happen.
        // Deliberately not on `CreateTableRequest`.
        schema_version: 1,
        named_vectors: request.named_vectors,
        hnsw: request.hnsw,
        quantization: request.quantization,
        payload_index: request.payload_index,
        embedding: request.embedding,
        rerank: request.rerank,
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
#[cfg_attr(feature = "api-swagger", utoipa::path(
    get,
    path = "/api/v1/tables/{table_name}",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    responses(
        (status = 200, description = "获取表信息成功", body = TableWithCount),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "tables"
))]
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
#[cfg_attr(feature = "api-swagger", utoipa::path(
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
))]
#[axum::debug_handler]
pub async fn list_tables(
    State(state): State<AppState>,
    ApiQuery(pagination): ApiQuery<PaginationQuery>,
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
#[cfg_attr(feature = "api-swagger", utoipa::path(
    delete,
    path = "/api/v1/tables/{table_name}",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    responses(
        (status = 200, description = "删除表成功", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "tables"
))]
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

/// Change a table's schema in place. The dimension is not changeable: it is fixed when the table is
/// created and every vector in it is that wide.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    patch,
    path = "/api/v1/tables/{table_name}",
    request_body(content = TableAlterRequest, example = json!({"description": "Renamed by PATCH"})),
    params(("table_name" = String, Path, description = "Table name", example = "demo")),
    responses(
        (status = 200, description = "Table altered", body = ApiResponse),
        (status = 400, description = "The change is not one that can be made", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "tables"
))]
/// Alter table schema — safe mutations only (description, memory_optimization,
/// enable_full_text_search toggle, add/drop payload field declarations).
///
/// PATCH /api/v1/tables/:table_name
pub async fn alter_table(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(req): ApiJson<TableAlterRequest>,
) -> impl IntoResponse {
    match state.storage.alter_table(&table_name, req).await {
        Ok(table) => {
            ApiResponse::success(serde_json::to_value(table).unwrap_or_default()).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// What the payload (filter) index holds, per field, and what it costs.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    get,
    path = "/api/v1/tables/{table_name}/payload_index",
    params(("table_name" = String, Path, description = "Table name", example = "demo")),
    responses(
        (status = 200, description = "Per-field payload index report", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "index"
))]
/// `GET /api/v1/tables/:table_name/payload_index`
///
/// Every payload field is indexed by default, so this is the answer to "what is that costing
/// me": distinct values, postings, and an estimate of the bytes, worst field first. It does not
/// build the index to describe it -- on a table that has not filtered since it was restored,
/// `built` is false and there is nothing to report yet.
pub async fn payload_index_report(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
) -> impl IntoResponse {
    match state.storage.payload_index_report(&table_name).await {
        Ok(report) => ApiResponse::success(report).into_response(),
        Err(e) => error_response(e),
    }
}

/// Index a payload field for filtering, or stop indexing it.
async fn set_payload_index_field(
    state: AppState,
    table_name: String,
    field: String,
    indexed: bool,
) -> axum::response::Response {
    let table = match state.storage.get_table(&table_name).await {
        Ok(table) => table,
        Err(e) => return error_response(e),
    };
    let mut policy = table.payload_index.clone().unwrap_or_default();
    if indexed {
        policy.include(&field);
    } else {
        policy.exclude(&field);
    }
    let req = crate::types::table::TableAlterRequest {
        payload_index: Some(policy),
        ..Default::default()
    };
    match state.storage.alter_table(&table_name, req).await {
        Ok(table) => {
            let message = if indexed {
                format!(
                    "Payload field '{}' is indexed on table '{}'. The index is rebuilt from the \
                     records on the next query that filters.",
                    field, table_name
                )
            } else {
                format!(
                    "Payload field '{}' is no longer indexed on table '{}'. A filter on it is \
                     now refused rather than answered, and the index is rebuilt without it on \
                     the next query that filters.",
                    field, table_name
                )
            };
            ApiResponse::success(serde_json::json!({
                "message": message,
                "payload_index": table.payload_index,
            }))
            .into_response()
        }
        Err(e) => error_response(e),
    }
}

/// Index a payload field for filtering.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    put,
    path = "/api/v1/tables/{table_name}/payload_index/{field}",
    params(
        ("table_name" = String, Path, description = "Table name", example = "demo"),
        ("field" = String, Path, description = "Payload field", example = "category")
    ),
    responses(
        (status = 200, description = "Field is indexed", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "index"
))]
/// `PUT /api/v1/tables/:table_name/payload_index/:field`
///
/// Idempotent, and a no-op on a table that indexes everything -- which is the default, so
/// client code written against a database where indexes must be declared works unchanged.
pub async fn index_payload_field(
    State(state): State<AppState>,
    Path((table_name, field)): Path<(String, String)>,
) -> impl IntoResponse {
    set_payload_index_field(state, table_name, field, true).await
}

/// Stop indexing a payload field.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    delete,
    path = "/api/v1/tables/{table_name}/payload_index/{field}",
    params(
        ("table_name" = String, Path, description = "Table name", example = "demo"),
        ("field" = String, Path, description = "Payload field", example = "notes")
    ),
    responses(
        (status = 200, description = "Field is no longer indexed", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "index"
))]
/// `DELETE /api/v1/tables/:table_name/payload_index/:field`
///
/// Frees what that field's postings cost, at the price of filtering on it: a filter naming an
/// unindexed field is refused, because the only other honest answer this engine could give is
/// an empty page, and an empty page looks like a result.
pub async fn drop_payload_index_field(
    State(state): State<AppState>,
    Path((table_name, field)): Path<(String, String)>,
) -> impl IntoResponse {
    set_payload_index_field(state, table_name, field, false).await
}

/// Drop the table's index. Searches still answer -- by brute force, exactly and slowly -- until an
/// index is rebuilt.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    delete,
    path = "/api/v1/tables/{table_name}/index",
    params(("table_name" = String, Path, description = "Table name", example = "demo")),
    responses(
        (status = 200, description = "Index dropped", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "index"
))]
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

/// Rebuild the index now rather than waiting for the periodic re-evaluation. The replacement is
/// built before it is swapped in, so queries keep being served by the old index throughout.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/index/rebuild",
    params(("table_name" = String, Path, description = "Table name", example = "demo")),
    responses(
        (status = 200, description = "Index rebuilt", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "index"
))]
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
                // This used to read "rebuilt from {} persisted records" with an empty string in
                // the placeholder; `rebuild_index` returns no count to put there.
                "message": format!(
                    "Vector index for table '{}' has been rebuilt from its persisted records.",
                    table_name
                ),
                "table": table_name,
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// Give every graph node that nothing points at an in-edge again, now rather than at the
/// next sweep. A stranded node is in the table and in the index and no query returns it at
/// any budget; `GET /tables/{t}/stats` reports under `reachability` how many the last pass
/// found. See `HnswIndex::repair_reachability`.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/index/repair",
    params(("table_name" = String, Path, description = "Table name", example = "demo")),
    responses(
        (status = 200, description = "How many nodes had no in-edge and how many were given one", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "index"
))]
pub async fn repair_reachability(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
) -> impl IntoResponse {
    match state.storage.repair_reachability(&table_name).await {
        Ok((stranded, repaired)) => ApiResponse::success(serde_json::json!({
            "table": table_name,
            "stranded": stranded,
            "repaired": repaired,
        }))
        .into_response(),
        Err(e) => error_response(e),
    }
}

/// Measure this table's recall@10 against its own exact scan and set the `normal` search
/// budget to the smallest that meets the table's recall target. The sweep does this on its
/// own once a table is large enough and again whenever it doubles; this runs it now, at any
/// size, and returns what it found. See `RecallCalibration`.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/recall/calibrate",
    params(("table_name" = String, Path, description = "Table name", example = "demo")),
    responses(
        (status = 200, description = "The calibration, or null when the table has nothing to calibrate (calibration off, a brute-force index, or fewer rows than a sample needs)", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "index"
))]
pub async fn calibrate_recall(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
) -> impl IntoResponse {
    match state.storage.calibrate_recall(&table_name, true).await {
        Ok(cal) => ApiResponse::success(serde_json::json!({
            "table": table_name,
            "calibration": cal,
        }))
        .into_response(),
        Err(e) => error_response(e),
    }
}

/// Write every table to disk and compact the write-ahead log up to that point. A snapshot is the
/// only copy of what it covers once the log is trimmed, so back up the snapshot directory like the
/// data it is. Refused while local durability is degraded, so an unacknowledged write cannot become
/// a new durable frontier.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/snapshot",
    responses(
        (status = 200, description = "Snapshot written", body = ApiResponse),
        (status = 503, description = "Durability degraded; the server is read-only", body = ApiResponse)
    ),
    tag = "maintenance"
))]
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

/// Size and memory statistics for one table.
///
/// Added because dxmt-js-sdk's `getTableStats()` has always called `GET /tables/{name}/stats`,
/// a route that did not exist — the method threw for every user of the JS client.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    get,
    path = "/api/v1/tables/{table_name}/stats",
    params(("table_name" = String, Path, description = "表名", example = "demo")),
    responses(
        (status = 200, description = "表统计", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "tables"
))]
pub async fn get_table_stats(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
) -> impl IntoResponse {
    match state.storage.get_table_stats(&table_name).await {
        Ok(stats) => ApiResponse::success(stats).into_response(),
        Err(e) => error_response(e),
    }
}

/// Copy a table under a new name, records and index included, so an experiment can run against real
/// data without touching it.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/branch",
    request_body(
        content = serde_json::Value,
        description = "The name for the copy, under `branch_name`",
        example = json!({"branch_name": "demo_experiment"})
    ),
    params(("table_name" = String, Path, description = "Table name", example = "demo")),
    responses(
        (status = 200, description = "Branch created", body = ApiResponse),
        (status = 400, description = "branch_name missing", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse),
        // The taken-name case is a 409, not the 400 the description above claimed.
        (status = 409, description = "branch_name is already a table or an alias",
         body = ApiResponse)
    ),
    tag = "tables"
))]
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

    // A branch is a table, and gets the rules `POST /tables` has always applied. Without this,
    // `"bad name!"` and `""` became tables that survived a restart, and a 300-character name was a
    // 500 in disk mode ("File name too long") and a table in memory mode.
    if let Err(e) = validate_table_name(&branch_name) {
        let error = ApiError::new("INVALID_TABLE_NAME", &e.to_string());
        return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
    }
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
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/records",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    request_body(content = BatchInsertRequest, example = json!({
        "records": [
            {"id": "doc-1", "vector": [1.0, 0.0, 0.0, 0.0],
             "payload": {"topic": "rust", "rank": 1, "content": "rust borrows instead of collecting"}},
            {"id": "doc-2", "vector": [0.0, 1.0, 0.0, 0.0],
             "payload": {"topic": "go", "rank": 2, "content": "go collects instead of borrowing"}},
            {"id": "doc-3", "vector": [0.9, 0.1, 0.0, 0.0],
             "payload": {"topic": "rust", "rank": 3, "content": "rust traits are not interfaces"}}
        ]
    })),
    responses(
        (status = 200, description = "插入记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse),
        // Both of these were reachable and undocumented, which the OpenAPI validation in
        // `tests/api/helpers/openapi.py` found by seeing the server answer a status the
        // document did not list. An insert is the one call that gets refused for reasons
        // that are about the server rather than the request.
        (status = 429, description = "Overloaded; see Retry-After", body = ApiResponse),
        (status = 503, description = "Durability degraded, or below the free-memory floor; \
                                      the server refuses writes until it recovers",
         body = ApiResponse)
    ),
    tag = "records"
))]
#[axum::debug_handler]
pub async fn insert_records(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<BatchInsertRequest>,
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

        // The envelope, like every other refusal. This was a hand-built JSON object with
        // `error` as a *string*, so a client reading `body.error.code` -- which is what the
        // document promises and what every other endpoint returns -- got a string where it
        // expected an object, on the one response it is most important to handle. The
        // OpenAPI validation in `tests/api/helpers/openapi.py` is what found it.
        let body = ApiResponse::error(ApiError {
            code: "SERVER_OVERLOADED".to_string(),
            message: "Server is currently overloaded".to_string(),
            details: Some(serde_json::json!({
                "state": health_state.description(),
                "retry_after_seconds": retry_after,
            })),
        });

        return (StatusCode::TOO_MANY_REQUESTS, headers, Json(body)).into_response();
    }

    // Get table for validation
    let table = match state.storage.get_table(&table_name).await {
        Ok(table) => table,
        Err(e) => return error_response(e),
    };

    // Validation. A batch-level fault (empty, too large) is one message; per-row faults are
    // a list, and what happens next depends on `on_error`.
    let row_errors = match validate_batch_insert(&request, &table) {
        Ok(errors) => errors,
        Err(e) => {
            let body = ApiResponse::error(ApiError {
                code: "VALIDATION_FAILED".to_string(),
                message: e,
                details: None,
            });
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };
    let total = request.records.len();
    let failed_count = row_errors.len();
    let records = if failed_count == 0 {
        request.records
    } else if request.on_error == OnError::Skip && failed_count < total {
        let bad: std::collections::HashSet<usize> = row_errors.iter().map(|e| e.index).collect();
        request
            .records
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !bad.contains(i))
            .map(|(_, r)| r)
            .collect()
    } else {
        let first = &row_errors[0];
        let body = ApiResponse::error(ApiError {
            code: "VALIDATION_FAILED".to_string(),
            message: format!(
                "{} of {} records are invalid; nothing was written. First: record {} (ID '{}'): {}",
                failed_count, total, first.index, first.id, first.error
            ),
            details: Some(serde_json::json!({
                "failed": &row_errors[..row_errors.len().min(MAX_REPORTED_ROW_ERRORS)],
                "failed_count": failed_count,
                "hint": "send on_error: \"skip\" to write the valid rows and have the rest listed",
            })),
        });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    };

    // Start tracking this operation
    let _guard = state.server_health.start_insert();

    match state.storage.insert_records(&table_name, records).await {
        Ok(inserted_records) => {
            let ids: Vec<&str> = inserted_records.iter().map(|r| r.id.as_str()).collect();
            let response = serde_json::json!({
                "inserted_ids": ids,
                "count": ids.len(),
                "failed": &row_errors[..row_errors.len().min(MAX_REPORTED_ROW_ERRORS)],
                "failed_count": failed_count,
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => {
            // Record the error for health tracking
            state.server_health.record_error();
            error_response(e)
        }
    }
}

/// 获取单个记录
#[cfg_attr(feature = "api-swagger", utoipa::path(
    get,
    path = "/api/v1/tables/{table_name}/records/{record_id}",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo"),
        ("record_id" = String, Path, description = "记录ID", example = "doc-1")
    ),
    responses(
        (status = 200, description = "获取记录成功", body = ApiResponse),
        (status = 404, description = "记录不存在", body = ApiResponse)
    ),
    tag = "records"
))]
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
#[cfg_attr(feature = "api-swagger", utoipa::path(
    put,
    path = "/api/v1/tables/{table_name}/records/{record_id}",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo"),
        ("record_id" = String, Path, description = "记录ID", example = "doc-1")
    ),
    request_body(content = UpdateRecordRequest, example = json!({
        "payload": {"topic": "systems"},
        "merge_mode": "merge"
    })),
    responses(
        (status = 200, description = "更新记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "记录不存在", body = ApiResponse)
    ),
    tag = "records"
))]
#[axum::debug_handler]
pub async fn update_record(
    State(state): State<AppState>,
    Path((table_name, record_id)): Path<(String, String)>,
    ApiJson(request): ApiJson<UpdateRecordRequest>,
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

    // 3. Apply it. What an update keeps and what it replaces is decided in one place,
    //    `IntelligentIndexedStorage::apply_update`, which gRPC calls too. This handler used to
    //    carry its own sixty-line copy of that decision, and both copies had the same bug: a
    //    replace-mode update without a payload -- `{"vector": [...]}`, re-embedding a record --
    //    replaced the payload with nothing, and the record lost all of its metadata with a 200.
    let merge = matches!(request.merge_mode, PayloadUpdateMode::Merge);
    match state
        .storage
        .apply_update(
            &table_name,
            &record_id,
            crate::storage::RecordUpdate {
                vector: request.vector,
                payload: request.payload,
                text: request.text,
                merge,
                upsert: request.upsert,
            },
        )
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
        Err(e) => error_response(e),
    }
}

/// 批量删除记录
#[cfg_attr(feature = "api-swagger", utoipa::path(
    delete,
    path = "/api/v1/tables/{table_name}/records",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    request_body(content = BatchDeleteRequest, example = json!({"ids": ["doc-2", "doc-3"]})),
    responses(
        (status = 200, description = "删除记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "records"
))]
#[axum::debug_handler]
pub async fn delete_records(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<BatchDeleteRequest>,
) -> impl IntoResponse {
    // One selector or the other. Both would have to mean intersection or union, and neither
    // reading is obvious enough to guess at; neither means "delete the whole table", which is
    // not something to do by omission.
    match (request.ids.is_empty(), &request.filter) {
        (true, None) => {
            return ApiResponse::error(ApiError::new(
                "INVALID_INPUT",
                "give either `ids` or `filter`: an empty request would otherwise read as \
                 \"delete everything\"",
            ))
            .into_response();
        }
        (false, Some(_)) => {
            return ApiResponse::error(ApiError::new(
                "INVALID_INPUT",
                "give either `ids` or `filter`, not both",
            ))
            .into_response();
        }
        _ => {}
    }

    if let Some(filter) = request.filter {
        return match state
            .storage
            .delete_records_matching(&table_name, &filter)
            .await
        {
            Ok(deleted) => ApiResponse::success(serde_json::json!({
                "deleted_count": deleted,
                // There is no "requested" count for a filter: what it matched is what it was.
                "requested_count": deleted,
            }))
            .into_response(),
            Err(e) => error_response(e),
        };
    }

    let requested_count = request.ids.len();
    match state.storage.delete_records(&table_name, request.ids).await {
        // `deleted` is what was actually there, not what was asked about. These were the same
        // number until the storage layer started counting: deleting three ids that do not
        // exist answered `deleted_count: 3`.
        Ok(deleted) => {
            let response = serde_json::json!({
                "deleted_count": deleted,
                "requested_count": requested_count
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

// ============ 搜索API ============

/// How many records a filter matches.
///
/// `GET /tables/{t}/stats` carries a record count for the whole table and nothing for a
/// condition, so "how many of these are there" meant paging the matches and counting them
/// client-side — every byte of every match on the wire to produce one integer. The count is
/// exact: there is no sampled mode, and a filter here is read exactly as `search` and
/// `records/list` read it.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct CountRecordsRequest {
    /// Count only what this matches. Omitted, it counts the table.
    #[serde(default)]
    pub filter: Option<crate::types::Filter>,
}

#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/records/count",
    params(("table_name" = String, Path, description = "表名", example = "demo")),
    request_body(content = CountRecordsRequest, example = json!({
        "filter": {"must": [{"key": "topic", "operation": {"type": "Equal", "value": "rust"}}]}
    })),
    responses(
        (status = 200, description = "匹配的记录数", body = ApiResponse),
        (status = 400, description = "过滤条件不合法", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse),
    ),
    tag = "records"
))]
pub async fn count_records(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<CountRecordsRequest>,
) -> impl IntoResponse {
    match state
        .storage
        .count_records(&table_name, request.filter.as_ref())
        .await
    {
        Ok(count) => ApiResponse::success(serde_json::json!({ "count": count })).into_response(),
        Err(e) => error_response(e),
    }
}

/// The body of `POST /tables/{t}/facets`.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct FacetRequest {
    /// The payload field to count values of. A dotted path works, as in a filter.
    pub key: String,
    /// Count only within what this matches. Omitted, it counts the table.
    #[serde(default)]
    pub filter: Option<crate::types::Filter>,
    /// How many of the most common values to return. Default 20.
    ///
    /// Capped, because a field holding a distinct value per row has as many buckets as the
    /// table has rows and nobody wants them: a facet is for the shape of a field, and a field
    /// with no shape is one the caller should have discovered from `distinct_values` instead.
    #[serde(default)]
    pub limit: Option<usize>,
}

const MAX_FACET_LIMIT: usize = 1_000;
const DEFAULT_FACET_LIMIT: usize = 20;

#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/facets",
    params(("table_name" = String, Path, description = "表名", example = "demo")),
    request_body(content = FacetRequest, example = json!({"key": "topic", "limit": 20})),
    responses(
        (status = 200, description = "Value counts, most common first", body = ApiResponse),
        (status = 400, description = "The field is not indexed, or the filter is not valid", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse),
    ),
    tag = "records"
))]
/// How many records hold each value of a payload field.
///
/// The answer carries `covered` and `matched` as well as the counts: a facet built from the
/// payload index describes what the index holds, and a record it does not hold — one whose
/// value there is an object, say — is missing from every bucket rather than being in a bucket
/// of its own. `matched - covered` is that number, and a caller drawing a chart from `hits`
/// without it is drawing one that omits part of the corpus without saying so.
pub async fn facets(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<FacetRequest>,
) -> impl IntoResponse {
    if request.key.is_empty() {
        return error_response(crate::Error::InvalidInput(
            "`key` names the payload field to count values of, and is empty".to_string(),
        ));
    }
    let limit = request
        .limit
        .unwrap_or(DEFAULT_FACET_LIMIT)
        .clamp(1, MAX_FACET_LIMIT);
    match state
        .storage
        .facet(&table_name, &request.key, request.filter.as_ref(), limit)
        .await
    {
        Ok(result) => {
            let value = serde_json::to_value(result).unwrap_or(serde_json::Value::Null);
            ApiResponse::success(value).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// Search by sparse vector alone.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/search/sparse",
    params(("table_name" = String, Path, description = "Table name", example = "demo")),
    request_body(content = SparseSearchQuery, example = json!({
        "sparse_vector": {"indices": [3, 17, 42], "values": [0.7, 0.2, 0.9], "dimension": 1000},
        "limit": 5
    })),
    responses(
        (status = 200, description = "Hits, best first", body = ApiResponse),
        (status = 400, description = "Bad request", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "search"
))]
/// `POST /api/v1/tables/:table_name/search/sparse`
///
/// Sparse vectors could be written and indexed here long before they could be searched: the
/// only path into the sparse index was `/search/hybrid`, which refuses a body with no dense
/// vector and no text, and needs the `text-bm25` build. A table of SPLADE vectors was stored,
/// indexed and unreachable. This is the default build, no feature required.
#[axum::debug_handler]
pub async fn sparse_search(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<SparseSearchQuery>,
) -> impl IntoResponse {
    match state.storage.sparse_search(&table_name, &request).await {
        Ok(result) => {
            let value = serde_json::to_value(result).unwrap_or(serde_json::Value::Null);
            ApiResponse::success(value).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// `POST /tables/{t}/search/groups` — a search whose answer is groups, not hits.
///
/// The schema is written out rather than derived. utoipa 4.2 does not understand
/// `#[serde(flatten)]`: the derive produced an object with **no properties** and, because of
/// `deny_unknown_fields`, `additionalProperties: false` — a document saying this endpoint
/// accepts nothing at all, which is what a generated client would then refuse to send. The
/// OpenAPI validation in `tests/api/helpers/openapi.py` is what found it. `allOf` is the
/// shape a flattened struct has: everything `SearchQuery` takes, plus the three below.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchGroupsRequest {
    /// Everything a plain search takes. `top_k` and `limit` are ignored here -- how many hits
    /// are scored is decided by `groups` and `group_size` and reported back as
    /// `candidates_fetched` -- and naming them would suggest otherwise.
    #[serde(flatten)]
    pub query: SearchQuery,
    /// The payload field whose value defines a group. A record with no value there is in no
    /// group, and the answer counts those rather than dropping them quietly.
    pub group_by: String,
    /// How many groups to return.
    #[serde(default = "default_groups")]
    pub groups: usize,
    /// How many hits to keep in each group, best first.
    #[serde(default = "default_group_size")]
    pub group_size: usize,
}

#[cfg(feature = "api-swagger")]
impl<'s> utoipa::ToSchema<'s> for SearchGroupsRequest {
    /// `SearchQuery`'s own schema with three properties added, not an `allOf` over it.
    ///
    /// `allOf` is the obvious spelling for a flattened struct and it cannot work here:
    /// `SearchQuery` is `deny_unknown_fields`, so its schema carries
    /// `additionalProperties: false`, and every branch of an `allOf` must accept the *whole*
    /// object. A body with `group_by` fails the `SearchQuery` branch. That was the second
    /// wrong document for this endpoint in one sitting -- the derived one accepted nothing at
    /// all, and the `allOf` one rejected the three fields the endpoint is named for.
    ///
    /// Taking `SearchQuery::schema()` and extending it keeps the thirteen inherited fields
    /// from being copied here, where they would drift the first time one changed.
    fn schema() -> (
        &'s str,
        utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
    ) {
        use utoipa::openapi::schema::{ObjectBuilder, Schema, SchemaType};
        use utoipa::openapi::RefOr;

        let (_, base) = <SearchQuery as utoipa::ToSchema>::schema();
        let mut object = match base {
            RefOr::T(Schema::Object(o)) => o,
            // `SearchQuery` is a plain struct, so this is unreachable; an empty object is the
            // safe answer rather than a panic in a documentation path.
            _ => ObjectBuilder::new().build(),
        };
        object.properties.insert(
            "group_by".to_string(),
            ObjectBuilder::new()
                .schema_type(SchemaType::String)
                .description(Some("The payload field whose value defines a group."))
                .into(),
        );
        object.required.push("group_by".to_string());
        object.properties.insert(
            "groups".to_string(),
            ObjectBuilder::new()
                .schema_type(SchemaType::Integer)
                .description(Some("How many groups to return. Default 10."))
                .into(),
        );
        object.properties.insert(
            "group_size".to_string(),
            ObjectBuilder::new()
                .schema_type(SchemaType::Integer)
                .description(Some("How many hits to keep in each group. Default 3."))
                .into(),
        );
        ("SearchGroupsRequest", RefOr::T(Schema::Object(object)))
    }
}

fn default_groups() -> usize {
    10
}

fn default_group_size() -> usize {
    3
}

/// Nearest neighbours grouped by a payload field.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/search/groups",
    params(("table_name" = String, Path, description = "Table name", example = "demo")),
    request_body(content = SearchGroupsRequest, example = json!({
        "vector": [0.12, 0.98, 0.31, 0.04],
        "group_by": "topic",
        "groups": 3,
        "group_size": 1
    })),
    responses(
        (status = 200, description = "Groups, best group first", body = ApiResponse),
        (status = 400, description = "Bad request", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "search"
))]
/// `POST /api/v1/tables/:table_name/search/groups`
///
/// One document chunked into twenty pieces takes a whole top-10 by itself. Grouping by the
/// document's id with `group_size: 1` gives ten documents instead. The groups are filled from a
/// window of nearest neighbours that widens until enough groups appear or a cap is reached, and
/// `candidates_fetched` reports how wide it got -- a short answer means the data ran out or the
/// cap did, and the number says which.
#[axum::debug_handler]
pub async fn search_groups(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<SearchGroupsRequest>,
) -> impl IntoResponse {
    if request.query.vector.is_empty() {
        let error = ApiError::new("INVALID_VECTOR", "Query vector must not be empty");
        return (StatusCode::BAD_REQUEST, ApiResponse::error(error)).into_response();
    }
    match state
        .storage
        .search_groups(
            &table_name,
            request.query,
            &request.group_by,
            request.groups,
            request.group_size,
        )
        .await
    {
        Ok(result) => {
            let value = serde_json::to_value(result).unwrap_or(serde_json::Value::Null);
            ApiResponse::success(value).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// Nearest neighbours to a query vector.
///
/// `search_mode` trades recall against latency: `fast` (ef = top_k), `normal` (the default),
/// `accurate` (4× ef), `auto` (scaled by filter selectivity), and `exact`. The first four walk
/// the index and answer approximately — `accurate` measures 0.98–0.99 recall depending on the
/// corpus, not 1.0. **`exact` scores every live row**, which is the only way to find out what
/// recall your index is actually giving you on your own data: run the same query twice and
/// compare. It costs a full pass over the table's vectors and is refused above
/// `DXMT_EXACT_SCAN_MAX` rows (default 1,000,000); a filter narrows what it has to read.
///
/// `POST /api/v1/tables/{table_name}/batch_search` is an alias for this route and answers
/// identically; it does not batch anything.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/search",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    request_body(content = SearchQuery, example = json!({
        "vector": [1.0, 0.0, 0.0, 0.0],
        "top_k": 3,
        "limit": 3,
        "with_payload": true,
        "filter": {
            "must": [{"key": "topic", "operation": {"type": "Equal", "value": "rust"}}]
        }
    })),
    responses(
        (status = 200, description = "搜索成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "search"
))]
#[axum::debug_handler]
pub async fn search_vectors(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(query): ApiJson<SearchQuery>,
) -> impl IntoResponse {
    if query.vector.is_empty() && query.text.is_none() {
        let error = ApiError::new(
            "INVALID_VECTOR",
            "a search needs a `vector`, or a `text` on a table that declares an embedding model",
        );
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
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/records/list",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    request_body(content = ListRecordsRequest, example = json!({
        "limit": 10, "offset": 0, "with_vector": false
    })),
    responses(
        (status = 200, description = "列出记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "records"
))]
#[axum::debug_handler]
pub async fn list_records(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<ListRecordsRequest>,
) -> impl IntoResponse {
    let limit = request.limit.unwrap_or(100).min(1000); // Max 1000 records per request
    let offset = request.offset.unwrap_or(0);
    let with_vector = request.with_vector.unwrap_or(false);

    // An offset past zero is the old way of paging and is answered the old way, without a
    // cursor; everything else — the first page included — pages by cursor, so the first page
    // hands back the cursor that continues it.
    if offset > 0 {
        if request.cursor.is_some() {
            return error_response(crate::Error::InvalidInput(
                "pass either `cursor` or `offset`, not both".to_string(),
            ));
        }
        return match state
            .storage
            .list_records(
                &table_name,
                request.filter.as_ref(),
                limit,
                offset,
                with_vector,
            )
            .await
        {
            Ok((records, total)) => ApiResponse::success(serde_json::json!({
                "records": records,
                "total": total,
                "limit": limit,
                "offset": offset,
                "next_cursor": null,
            }))
            .into_response(),
            Err(e) => error_response(e),
        };
    }

    let cursor = match request
        .cursor
        .as_deref()
        .map(crate::storage::payload_index::ListCursor::decode)
        .transpose()
    {
        Ok(cursor) => cursor,
        Err(message) => return error_response(crate::Error::InvalidInput(message)),
    };
    match state
        .storage
        .list_records_after(
            &table_name,
            request.filter.as_ref(),
            cursor.as_ref(),
            limit,
            with_vector,
        )
        .await
    {
        Ok((records, total, next)) => ApiResponse::success(serde_json::json!({
            "records": records,
            "total": total,
            "limit": limit,
            "offset": 0,
            "next_cursor": next.map(|c| c.encode()),
        }))
        .into_response(),
        Err(e) => error_response(e),
    }
}

/// Dense-vector search under a hybrid-shaped body.
///
/// Despite the name it fuses nothing: a body carrying `sparse_vector` or `text_query` is
/// answered 501 naming the endpoint that does fuse them (`POST /tables/{name}/search/hybrid`,
/// feature `text-bm25`). The example is therefore a lone `dense_vector`, which is the only
/// request this route serves.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/hybrid_search",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    request_body(content = HybridSearchQuery, example = json!({
        "dense_vector": [1.0, 0.0, 0.0, 0.0],
        "top_k": 3
    })),
    responses(
        (status = 200, description = "搜索成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse),
        // This endpoint serves a lone `dense_vector` and refuses anything with a sparse or
        // text component, naming the endpoints that do work. That refusal is its most common
        // answer and was not in the document at all.
        (status = 501, description = "A sparse or text component; use /search/hybrid",
         body = ApiResponse)
    ),
    tag = "search"
))]
#[axum::debug_handler]
pub async fn hybrid_search(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(query): ApiJson<HybridSearchQuery>,
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
#[cfg_attr(feature = "api-swagger", utoipa::path(
    get,
    path = "/status",
    responses(
        (status = 200, description = "系统健康状态", body = ApiResponse)
    ),
    tag = "health"
))]
#[axum::debug_handler]
pub async fn get_status(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state.storage.get_stats().await;
    let storage = state.storage.health_snapshot().await;

    let status = serde_json::json!({
        "status": if storage.ready { "ready" } else { "not_ready" },
        "version": env!("CARGO_PKG_VERSION"),
        "storage": stats,
        "persistence": storage,
    });

    ApiResponse::success(status).into_response()
}

/// 健康检查端点（用于 Consul）
#[cfg_attr(feature = "api-swagger", utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "服务健康"),
        (status = 503, description = "服务不健康")
    ),
    tag = "health"
))]
/// Server health check endpoint
///
/// Returns the current health state of the server including operation metrics.
/// Use this endpoint to monitor server load and determine if backpressure is active.
pub async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let health_state = state.server_health.current_state().await;
    let metrics = state.server_health.get_metrics();
    let storage = state.storage.health_snapshot().await;

    let response = serde_json::json!({
        "status": format!("{:?}", health_state),
        "service": "vectordb",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        // An empty database is healthy; readiness is based on the durability boundary instead
        // of using table count as a proxy for storage health.
        "storage_healthy": storage.ready,
        "ready": storage.ready,
        "description": health_state.description(),
        "retry_after_seconds": health_state.retry_after_seconds(),
        "storage": storage,
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

/// Liveness only: the process is up and the HTTP runtime is answering. It says nothing about
/// whether a write can be made durable -- use /ready for that.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    get,
    path = "/live",
    responses((status = 200, description = "The process is alive")),
    tag = "health"
))]
/// Process liveness probe. This endpoint intentionally does not touch storage or wait for
/// recovery; a process that is alive but not ready must be restarted only when the orchestrator
/// policy says so, not because a cold snapshot takes time to load.
pub async fn liveness() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "alive",
            "service": "vectordb",
            "timestamp": chrono::Utc::now().to_rfc3339(),
        })),
    )
}

/// Readiness, and the probe a deployment should use. Answers 503 once a write-ahead log write has
/// failed and the process has gone read-only, where /health still answers 200 with a diagnostic
/// document.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    get,
    path = "/ready",
    responses(
        (status = 200, description = "Ready to serve writes"),
        (status = 503, description = "Local durability is degraded")
    ),
    tag = "health"
))]
/// Readiness probe for load balancers and orchestrators.
///
/// Readiness is separate from liveness: a read-only process after a WAL failure is still alive,
/// but must not receive traffic that expects successful mutations.
pub async fn readiness(State(state): State<AppState>) -> impl IntoResponse {
    let storage = state.storage.health_snapshot().await;
    let status = if storage.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(serde_json::json!({
            "status": if storage.ready { "ready" } else { "not_ready" },
            "service": "vectordb",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "reason": storage.durability_error.clone(),
            "storage": storage,
        })),
    )
}

// ============ 备份API ============

/// `POST /tables/{t}/export` — write one table to a file on the server.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct ExportTableRequest {
    /// File the table is written to, inside `DXMT_BACKUP_ROOT`. Its parent must exist.
    pub path: String,
}

/// `POST /tables/import` — read a table back out of such a file.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct ImportTableRequest {
    /// The file to read, inside `DXMT_BACKUP_ROOT`.
    pub path: String,
    /// Import it under this name instead of the one in the file. This is how a table is copied
    /// beside itself -- to try a different index on the same data, say.
    #[serde(default)]
    pub name: Option<String>,
    /// Replace a table of that name if one exists. Without it, an import onto an existing name
    /// is refused rather than merged: a merge and a replacement are different operations and
    /// neither is the obvious reading of "import".
    #[serde(default)]
    pub overwrite: bool,
}

/// `PUT /aliases/{alias}` — point an alias at a table.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct SetAliasRequest {
    /// The table the alias resolves to. May itself be an alias, which is followed once so no
    /// reader ever has to follow a chain.
    pub table: String,
}

/// Every alias and the table it points at.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    get,
    path = "/api/v1/aliases",
    responses((status = 200, description = "Aliases, by name", body = ApiResponse)),
    tag = "tables"
))]
/// `GET /api/v1/aliases`
#[axum::debug_handler]
pub async fn list_aliases(State(state): State<AppState>) -> impl IntoResponse {
    let aliases = state.storage.list_aliases();
    ApiResponse::success(serde_json::json!({
        "aliases": aliases,
        "count": aliases.len(),
    }))
    .into_response()
}

/// Point an alias at a table, creating it or moving it.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    put,
    path = "/api/v1/aliases/{alias}",
    params(("alias" = String, Path, description = "Alias name", example = "docs")),
    request_body(content = SetAliasRequest, example = json!({"table": "demo"})),
    responses(
        (status = 200, description = "The alias points at the table", body = ApiResponse),
        (status = 400, description = "A table already has that name", body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "tables"
))]
/// `PUT /api/v1/aliases/:alias`
///
/// What this is for is a swap. Rebuilding a table with different index parameters means
/// building a second one and moving traffic across, and without an alias that move is a client
/// deploy. Moving an alias is one map write, so a reader arriving mid-swap sees one table or
/// the other, never neither.
#[axum::debug_handler]
pub async fn set_alias(
    State(state): State<AppState>,
    Path(alias): Path<String>,
    ApiJson(request): ApiJson<SetAliasRequest>,
) -> impl IntoResponse {
    match state.storage.set_alias(&alias, &request.table).await {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "alias": alias,
            "table": request.table,
        }))
        .into_response(),
        Err(e) => error_response(e),
    }
}

/// Remove an alias. The table it pointed at is untouched.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    delete,
    path = "/api/v1/aliases/{alias}",
    params(("alias" = String, Path, description = "Alias name", example = "docs")),
    responses(
        (status = 200, description = "Removed", body = ApiResponse),
        (status = 404, description = "No such alias", body = ApiResponse)
    ),
    tag = "tables"
))]
/// `DELETE /api/v1/aliases/:alias`
#[axum::debug_handler]
pub async fn delete_alias(
    State(state): State<AppState>,
    Path(alias): Path<String>,
) -> impl IntoResponse {
    match state.storage.delete_alias(&alias).await {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "alias": alias,
            "message": format!("Alias '{alias}' removed; the table it pointed at is unchanged."),
        }))
        .into_response(),
        Err(e) => error_response(e),
    }
}

/// Write one table -- configuration and every record, vectors included -- to a file.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/export",
    params(("table_name" = String, Path, description = "Table name", example = "demo")),
    request_body(content = ExportTableRequest, example = json!({"path": "demo.dxmt"})),
    responses(
        (status = 200, description = "Exported", body = ApiResponse),
        (status = 400, description = "Path outside DXMT_BACKUP_ROOT, or no durable storage",
         body = ApiResponse),
        (status = 404, description = "No such table", body = ApiResponse)
    ),
    tag = "maintenance"
))]
/// `POST /api/v1/tables/:table_name/export`
///
/// The unit `POST /backup` is not: a backup is the whole instance and restoring it means
/// stopping the server and replacing its directory. This moves one table. The file is the
/// snapshot format, vectors included; the graph is not in it and the import rebuilds it.
#[axum::debug_handler]
pub async fn export_table(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<ExportTableRequest>,
) -> impl IntoResponse {
    match state
        .storage
        .export_table(&table_name, std::path::Path::new(&request.path))
        .await
    {
        Ok(report) => ApiResponse::success(report).into_response(),
        Err(e) => error_response(e),
    }
}

/// Read a table back out of a file written by the export above.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/import",
    request_body(content = ImportTableRequest, example = json!({"path": "demo.dxmt"})),
    responses(
        (status = 200, description = "Imported", body = ApiResponse),
        (status = 400, description = "Unreadable file, a name already in use, or a path outside \
                                      DXMT_BACKUP_ROOT", body = ApiResponse)
    ),
    tag = "maintenance"
))]
/// `POST /api/v1/tables/import`
///
/// Records keep their ids, payloads, versions and `created_at` -- they are applied the way WAL
/// replay applies them, not re-inserted through the API, so an import does not restamp every
/// row. Each is logged, so the table survives a restart without waiting for a snapshot.
#[axum::debug_handler]
pub async fn import_table(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ImportTableRequest>,
) -> impl IntoResponse {
    match state
        .storage
        .import_table(
            std::path::Path::new(&request.path),
            request.name.as_deref(),
            request.overwrite,
        )
        .await
    {
        Ok(report) => ApiResponse::success(report).into_response(),
        Err(e) => error_response(e),
    }
}

#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct CreateBackupRequest {
    /// Directory the backup is written to. Must be writable by the server process.
    pub path: String,
}

/// Write a self-contained backup to a directory on the server.
///
/// Replaces seven endpoints (`/backup/full`, `/incremental`, `/manual`, `/status`, `/list`,
/// `/restore`, `DELETE /backup/{id}`) that returned 500 unconditionally: `backup_manager` was
/// hardcoded to `None`, and the BackupManager behind them collected objects under a `tables/`
/// prefix that nothing has ever written to — wiring it up would have produced empty backups,
/// which is worse than an error.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/backup",
    request_body(content = CreateBackupRequest, example = json!({"path": "/tmp/dxmt-backup"})),
    responses(
        (status = 200, description = "备份完成", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse)
    ),
    tag = "maintenance"
))]
pub async fn create_backup(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<CreateBackupRequest>,
) -> impl IntoResponse {
    match state
        .storage
        .backup_to(std::path::Path::new(&request.path))
        .await
    {
        Ok(manifest) => ApiResponse::success(manifest).into_response(),
        Err(e) => error_response(e),
    }
}

// ============ 优化的批量插入API ============

/// 优化的批量插入记录
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/records/bulk",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    request_body(content = OptimizedBatchInsertRequest, example = json!({
        "records": [
            {"id": "doc-4", "vector": [0.0, 0.0, 1.0, 0.0],
             "payload": {"topic": "python", "content": "python collects and does not borrow"}}
        ]
    })),
    responses(
        (status = 200, description = "插入记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "records"
))]
#[axum::debug_handler]
pub async fn insert_records_optimized(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<OptimizedBatchInsertRequest>,
) -> impl IntoResponse {
    // The cap used to depend on the `mode` the caller named -- 100,000 for two of them, 50,000
    // for two more -- which made the limit a property of a word rather than of the work. One
    // cap now, and it is the higher one: the path is the same path either way.
    const BULK_MAX: usize = 100_000;
    if request.records.len() > BULK_MAX {
        let error = ApiError::new(
            "INVALID_INPUT",
            &format!(
                "Batch size {} exceeds the maximum of {BULK_MAX} for this endpoint",
                request.records.len(),
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
    // One insert. `insert_records_batch` and its eight differently-named siblings all reduce
    // to this, and calling it directly is the honest spelling of what was already happening.
    let result = state
        .storage
        .insert_records(&table_name, request.records)
        .await
        .map(|records| records.into_iter().map(|r| r.id).collect::<Vec<_>>());

    match result {
        Ok(inserted_ids) => {
            let response = serde_json::json!({
                "inserted_ids": inserted_ids,
                "count": inserted_ids.len()
            });
            ApiResponse::success(response).into_response()
        }
        Err(e) => error_response(e),
    }
}

/// 请求一张 kNN 图
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct KnnGraphRequest {
    /// How many records to sample as nodes. Default 1,000, capped by `DXMT_KNN_GRAPH_MAX_NODES`.
    pub sample: Option<usize>,
    /// Neighbours per node. Default 15.
    pub neighbours: Option<usize>,
    /// Restrict the sample to records the filter accepts.
    pub filter: Option<crate::types::Filter>,
    /// Shifts which records the even spacing lands on, so a run is reproducible. Default 0.
    pub seed: Option<u64>,
    /// Include each node's payload, for labelling and colouring. Default **true**.
    ///
    /// On by default here, unlike `records/list`, because the two requests are for different
    /// things: listing records is browsing and usually wants neither vectors nor extra weight,
    /// while a graph is fetched in order to draw something, and a node with no label cannot be
    /// drawn usefully. Payloads are also negligible beside what this endpoint saves.
    pub with_payload: Option<bool>,
}

/// The nearest-neighbour structure of a sample, without its vectors.
///
/// A client that wants to lay out or summarise a table needs to know which records are near
/// which. Answering that from a browser means downloading the vectors: at 768 dimensions a
/// thousand records is already thirteen megabytes of JSON floats, and nothing here is
/// compressed — so the sample a projection can use is bounded by the wire rather than by
/// anything about the data.
///
/// This returns the answer instead of the inputs. A thousand nodes with fifteen neighbours each
/// is fifteen thousand edges, tens of times smaller than the vectors they were computed from.
///
/// Neighbours are computed **within the sample** and **exactly**. Within, because a layout is
/// drawn from the points on screen and an edge to a point that is not drawn describes nothing.
/// Exactly, because this is the reference a layout gets judged against, and an approximate
/// graph would turn "how much structure survived the projection" into a measurement of two
/// approximations at once.
///
/// O(n² · d), so it refuses above the node cap rather than running for a minute — a caller
/// cannot tell a slow answer from a hung one.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/graph",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    request_body(content = KnnGraphRequest, example = json!({
        "sample": 500, "neighbours": 10
    })),
    responses(
        (status = 200, description = "图构建成功", body = ApiResponse),
        (status = 400, description = "采样数超过上限", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse)
    ),
    tag = "search"
))]
#[axum::debug_handler]
pub async fn knn_graph(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<KnnGraphRequest>,
) -> impl IntoResponse {
    let sample = request.sample.unwrap_or(1_000);
    let neighbours = request.neighbours.unwrap_or(15);

    match state
        .storage
        .knn_graph(
            &table_name,
            sample,
            neighbours,
            request.filter.as_ref(),
            request.seed.unwrap_or(0),
            request.with_payload.unwrap_or(true),
        )
        .await
    {
        Ok(graph) => {
            ApiResponse::success(serde_json::to_value(graph).unwrap_or(serde_json::Value::Null))
                .into_response()
        }
        Err(e) => error_response(e),
    }
}

/// BM25 search over the payload's text, for a table created with `enable_full_text_search`.
///
/// Needs the `text-bm25` feature; a build without it answers `FEATURE_DISABLED`. Without a
/// declared `text_fields`, the indexed field is `content` — which is why the insert examples
/// put one there.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/search/text",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    request_body(content = FullTextSearchRequest, example = json!({"query": "rust", "limit": 3})),
    responses(
        (status = 200, description = "搜索成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse),
        (status = 501, description = "功能未启用", body = ApiResponse)
    ),
    tag = "search"
))]
#[axum::debug_handler]
pub async fn full_text_search(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<FullTextSearchRequest>,
) -> impl IntoResponse {
    #[cfg(not(feature = "text-bm25"))]
    {
        let _ = (&state, &table_name, &request);
        ApiResponse::error(ApiError::new(
            "FEATURE_DISABLED",
            "Full-text search feature is not enabled. Compile with --features text-bm25",
        ))
        .into_response()
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
            Err(e) => error_response(e),
        }
    }
}

/// 混合搜索（向量 + 文本组合）
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/search/hybrid",
    params(
        ("table_name" = String, Path, description = "表名", example = "demo")
    ),
    // `top_k` and `alpha` used to be in this example and are not fields of
    // `HybridSearchRequest`. They were accepted and ignored, and left anyone tuning `alpha`
    // watching a number that changed nothing; the body is now refused naming the field. The
    // real knobs are `limit`, `fusion_method` and — for `weighted_sum` only — `vector_weight`.
    request_body(content = HybridSearchRequest, example = json!({
        "vector": [1.0, 0.0, 0.0, 0.0],
        "text": "rust",
        "limit": 3,
        "fusion_method": "reciprocal_rank_fusion"
    })),
    responses(
        (status = 200, description = "搜索成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse),
        (status = 501, description = "本次构建没有全文索引，`text` 无法作答", body = ApiResponse)
    ),
    tag = "search"
))]
#[axum::debug_handler]
pub async fn hybrid_search_combined(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    ApiJson(request): ApiJson<HybridSearchRequest>,
) -> impl IntoResponse {
    // A `text` component on a build without `text-bm25` used to be *dropped*, in
    // `IntelligentIndexedStorage::hybrid_search`, where the engine is `None` when the feature is
    // off -- and the dense half came back with a 200 on it, indistinguishable from a fusion that
    // happened. That is the same defect `/hybrid_search` was fixed for ("it used to run the dense
    // search, drop the other components, and return the result as if the fusion had happened"),
    // and the two endpoints were answering it opposite ways.
    //
    // A body carrying only a vector still works: there is nothing to drop, so nothing to hide.
    #[cfg(not(feature = "text-bm25"))]
    {
        if request.text.is_some() {
            return ApiResponse::error(ApiError::new(
                "FEATURE_DISABLED",
                "This build has no full-text index, so the text half of a hybrid query cannot \
                 be answered. Compile with --features text-bm25, or send the vector alone.",
            ))
            .into_response();
        }
    }

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

    // The same rule as the feature check above, one level down: a table that declared no
    // text fields has no text index, and a hybrid query with `text` against it used to run
    // the dense half and return it shaped like a fusion. `/search/text` refuses such a
    // table with FULL_TEXT_NOT_ENABLED; so does this.
    if request.text.is_some() {
        match state.storage.get_table(&table_name).await {
            Ok(table) if !table.enable_full_text_search => {
                return ApiResponse::error(ApiError::new(
                    "FULL_TEXT_NOT_ENABLED",
                    "Full-text search is not enabled for this table, so the text half of a \
                     hybrid query cannot be answered. Declare `text_fields` at creation, or \
                     send the vector alone.",
                ))
                .into_response();
            }
            Ok(_) => {}
            Err(e) => {
                return ApiResponse::error(ApiError::new(
                    "TABLE_NOT_FOUND",
                    &format!("Table '{}' not found: {}", table_name, e),
                ))
                .into_response();
            }
        }
    }

    // Perform hybrid search using the storage layer
    match state.storage.hybrid_search(&table_name, request).await {
        Ok(result) => {
            ApiResponse::success(serde_json::to_value(result).unwrap_or(serde_json::Value::Null))
                .into_response()
        }
        Err(e) => error_response(e),
    }
}

/// Run a DXQL statement. SELECT, INSERT, UPDATE, DELETE, CREATE TABLE and DROP are executed; vector
/// search, text search and graph traversal inside a WHERE clause are refused by name, with the
/// route to use instead, rather than being accepted and quietly doing something else.
#[cfg_attr(feature = "api-swagger", utoipa::path(
    post,
    path = "/api/v1/query",
    request_body(
        content = serde_json::Value,
        description = "A DXQL statement under `query`",
        example = json!({"query": "SELECT * FROM demo WHERE rank > 1 LIMIT 10"})
    ),
    responses(
        (status = 200, description = "Statement executed", body = ApiResponse),
        (status = 400, description = "The statement does not parse or compile", body = ApiResponse),
        (status = 404, description = "The statement names a table that does not exist", body = ApiResponse),
        (status = 501, description = "Valid DXQL this server does not execute", body = ApiResponse),
        (status = 422, description = "Understood, and storage refused it", body = ApiResponse)
    ),
    tag = "query"
))]
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

    // Parse, compile, execute. Each failure gets the status that says whose problem it is: 400
    // the statement is wrong, 404 its table does not exist, 501 it is valid DXQL this server
    // does not execute, 422 it was understood and storage refused it.
    use crate::dxql::{
        compiler::Compiler, executor_impl::StorageAwareExecutor, parser::Parser, DxqlError,
    };
    fn failure(e: DxqlError) -> axum::response::Response {
        let status = match &e {
            DxqlError::ParseError(_)
            | DxqlError::CompilationError(_)
            | DxqlError::TypeError(_)
            | DxqlError::UnknownField(_)
            | DxqlError::InvalidOperation(_) => StatusCode::BAD_REQUEST,
            DxqlError::UnknownTable(_) => StatusCode::NOT_FOUND,
            DxqlError::Unsupported(_) => StatusCode::NOT_IMPLEMENTED,
            DxqlError::OptimizationError(_) | DxqlError::ExecutionError(_) => {
                StatusCode::UNPROCESSABLE_ENTITY
            }
        };
        (status, Json(serde_json::json!({ "error": e.to_string() }))).into_response()
    }

    let plan = match Parser::new()
        .parse(&sql)
        .and_then(|statement| Compiler::new().compile(statement))
    {
        Ok(plan) => plan,
        Err(e) => return failure(e),
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
                        "affected_rows": result.metadata.affected_rows,
                        "execution_time_ms": result.metadata.execution_time_ms,
                        "plan_type": result.metadata.plan_type,
                    }
                }
            })),
        )
            .into_response(),
        Err(e) => failure(e),
    }
}

/// Detailed server health metrics endpoint
///
/// Returns comprehensive health metrics for monitoring and debugging.
/// The same numbers as `/health/metrics`, in Prometheus text exposition format.
///
/// A JSON health document is not something a monitoring system can scrape: Prometheus, and
/// everything that speaks its format, wants `# HELP` / `# TYPE` and one sample per line. Until
/// this existed, running DXMT under the monitoring every other database assumes meant writing
/// an exporter first -- and the `metrics` cargo feature that suggested otherwise pulled in the
/// `prometheus` crate and was used by no code at all.
///
/// Rendered by hand rather than through that crate, so it is in the default build: needing a
/// special build to be observable is the wrong way round.
///
/// Timestamps are emitted as Unix seconds, which is the convention for a `_timestamp_seconds`
/// gauge; an absent one is omitted rather than sent as zero, because zero is 1970 and a
/// dashboard cannot tell that from "never happened" if it is sent a number.
pub async fn prometheus_metrics(State(state): State<AppState>) -> impl IntoResponse {
    use std::fmt::Write as _;

    let metrics = state.server_health.get_metrics();
    let storage = state.storage.health_snapshot().await;
    let mut out = String::with_capacity(2048);

    fn gauge(out: &mut String, name: &str, help: &str, value: String) {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} gauge");
        let _ = writeln!(out, "{name} {value}");
    }

    gauge(
        &mut out,
        "dxmt_up",
        "1 when the storage layer is ready to serve, 0 while it is not.",
        u8::from(storage.ready).to_string(),
    );
    gauge(
        &mut out,
        "dxmt_durability_degraded",
        "1 after a local WAL sync failure, when the process is read-only until restart.",
        u8::from(storage.durability_degraded).to_string(),
    );
    gauge(
        &mut out,
        "dxmt_tables",
        "Tables that exist.",
        storage.table_count.to_string(),
    );
    gauge(
        &mut out,
        "dxmt_tables_resident",
        "Tables held in memory; the rest are rehydrated on the next query that needs them.",
        storage.resident_table_count.to_string(),
    );
    gauge(
        &mut out,
        "dxmt_records",
        "Records across every table.",
        storage.total_records.to_string(),
    );
    gauge(
        &mut out,
        "dxmt_index_building",
        "1 while an index build or rebuild is running.",
        // A bool renders as "true"/"false", which is not a sample value Prometheus can parse.
        u8::from(metrics.index_building).to_string(),
    );
    gauge(
        &mut out,
        "dxmt_recent_errors",
        "Errors in the recent window the backpressure logic watches.",
        metrics.recent_errors.to_string(),
    );

    let _ = writeln!(
        out,
        "# HELP dxmt_ongoing_operations Requests in flight, by kind."
    );
    let _ = writeln!(out, "# TYPE dxmt_ongoing_operations gauge");
    for (kind, value) in [
        ("insert", metrics.ongoing_inserts),
        ("search", metrics.ongoing_searches),
        ("delete", metrics.ongoing_deletes),
        ("table", metrics.ongoing_table_ops),
    ] {
        let _ = writeln!(out, "dxmt_ongoing_operations{{kind=\"{kind}\"}} {value}");
    }

    // Which planner branch answered each filtered search. The four differ by orders of
    // magnitude -- a filter matching an eighth to a half of a million-row table costs
    // 160-800 ms where an unfiltered search is ~2 ms -- so a workload sitting on
    // `post_filter` is the explanation for a latency complaint, not a detail.
    let _ = writeln!(
        out,
        "# HELP dxmt_filter_path_total Filtered searches answered, by planner branch."
    );
    let _ = writeln!(out, "# TYPE dxmt_filter_path_total counter");
    for (branch, value) in [
        ("exact_scan", storage.filter_paths.exact_scan),
        ("code_scan", storage.filter_paths.code_scan),
        ("in_graph", storage.filter_paths.in_graph),
        ("pre_filter", storage.filter_paths.pre_filter),
        ("post_filter", storage.filter_paths.post_filter),
    ] {
        let _ = writeln!(out, "dxmt_filter_path_total{{branch=\"{branch}\"}} {value}");
    }
    // And what each branch cost: the count says where a workload is, this says what it
    // pays there. `eval` is the part every branch shares -- estimating the filter and
    // materialising its matching ids -- reported on its own so plumbing is told apart
    // from search.
    let _ = writeln!(
        out,
        "# HELP dxmt_filter_path_seconds_total Seconds spent answering filtered searches, by planner branch."
    );
    let _ = writeln!(out, "# TYPE dxmt_filter_path_seconds_total counter");
    for (branch, ns) in [
        ("exact_scan", storage.filter_paths.exact_scan_ns),
        ("code_scan", storage.filter_paths.code_scan_ns),
        ("in_graph", storage.filter_paths.in_graph_ns),
        ("pre_filter", storage.filter_paths.pre_filter_ns),
        ("post_filter", storage.filter_paths.post_filter_ns),
        ("eval", storage.filter_paths.eval_ns),
    ] {
        let _ = writeln!(
            out,
            "dxmt_filter_path_seconds_total{{branch=\"{branch}\"}} {:.6}",
            ns as f64 / 1e9
        );
    }

    let _ = writeln!(
        out,
        "# HELP dxmt_operations_total Operations completed since process start."
    );
    let _ = writeln!(out, "# TYPE dxmt_operations_total counter");
    let _ = writeln!(out, "dxmt_operations_total {}", metrics.total_operations);

    let _ = writeln!(
        out,
        "# HELP dxmt_snapshot_failures_total Snapshot attempts that failed since process start."
    );
    let _ = writeln!(out, "# TYPE dxmt_snapshot_failures_total counter");
    let _ = writeln!(
        out,
        "dxmt_snapshot_failures_total {}",
        storage.snapshot_failures
    );

    // The durability frontier: how much a restart would replay, and how far behind the last
    // snapshot is. `wal_next_sequence - snapshot_wal_next_sequence` is the replay depth.
    if let Some(sequence) = storage.wal_next_sequence {
        gauge(
            &mut out,
            "dxmt_wal_next_sequence",
            "The sequence number the next WAL entry will take.",
            sequence.to_string(),
        );
    }
    if let Some(sequence) = storage.snapshot_wal_next_sequence {
        gauge(
            &mut out,
            "dxmt_snapshot_wal_next_sequence",
            "The WAL sequence the newest installed snapshot covers up to.",
            sequence.to_string(),
        );
    }
    for (name, help, value) in [
        (
            "dxmt_last_wal_sync_timestamp_seconds",
            "Unix time of the most recent successful WAL fsync.",
            storage.last_wal_sync_at.as_deref(),
        ),
        (
            "dxmt_last_snapshot_timestamp_seconds",
            "Unix time of the most recently installed snapshot.",
            storage.last_snapshot_at.as_deref(),
        ),
    ] {
        if let Some(seconds) = value
            .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
            .map(|at| at.timestamp())
        {
            gauge(&mut out, name, help, seconds.to_string());
        }
    }

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        out,
    )
}

#[cfg_attr(feature = "api-swagger", utoipa::path(
    get,
    path = "/api/v1/health/metrics",
    responses(
        (status = 200, description = "Detailed health metrics", body = serde_json::Value),
    ),
    tag = "health"
))]
pub async fn health_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.server_health.get_metrics();
    let storage = state.storage.health_snapshot().await;

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
        },
        "storage": storage,
    });

    ApiResponse::success(response).into_response()
}
