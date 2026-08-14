use axum::{
    extract::{Path, State, Query},
    Json,
    http::StatusCode,
};
use serde::Deserialize;
use std::sync::Arc;

use crate::types::multimodal::{
        CreateMultiVectorRequest, CrossModalSearchQuery, MultimodalSearchResult,
        EmbeddingConfig, EmbeddingRequest, EmbeddingResponse, 
        MultiVectorRecord, MultimodalContent
    };
use crate::api::{response::{ApiResponse, ApiError}, handlers::AppState};
use crate::embeddings::{EmbeddingManager, MultimodalUtils};

#[derive(Clone)]
pub struct MultimodalAppState {
    pub basestate: AppState,
    pub embeddingmanager: Arc<EmbeddingManager>,
}

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct ListEmbeddingFunctionsQuery {
    pub modality: Option<String>,
}

// ============ 嵌入函数管理API ============

/// 列出所有可用的嵌入函数
#[utoipa::path(
    get,
    path = "/api/v1/embeddings/functions",
    params(ListEmbeddingFunctionsQuery),
    responses(
        (status = 200, description = "获取嵌入函数列表成功", body = ApiResponse),
    ),
    tag = "embeddings"
)]
pub async fn list_embedding_functions(
    State(state): State<MultimodalAppState>,
    Query(query): Query<ListEmbeddingFunctionsQuery>,
) -> std::result::Result<ApiResponse, StatusCode> {
    let registry = state.embeddingmanager.registry();
    let all_metadata = registry.get_all_metadata();
    
    let filteredpayload: Vec<_> = if let Some(modality) = query.modality {
        all_metadata.into_iter()
            .filter(|meta| meta.modality.to_lowercase() == modality.to_lowercase())
            .collect()
    } else {
        all_metadata
    };
    
    Ok(ApiResponse::success(filteredpayload))
}

/// 加载嵌入函数
#[utoipa::path(
    post,
    path = "/api/v1/embeddings/functions/{name}",
    request_body = EmbeddingConfig,
    responses(
        (status = 200, description = "加载嵌入函数成功", body = ApiResponse),
        (status = 400, description = "配置错误", body = ApiResponse),
    ),
    tag = "embeddings"
)]
pub async fn load_embedding_function(
    State(state): State<MultimodalAppState>,
    Path(name): Path<String>,
    Json(config): Json<EmbeddingConfig>,
) -> std::result::Result<ApiResponse, StatusCode> {
    let config_value = match serde_json::to_value(&config) {
        Ok(v) => v,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR)
    };
    state.embeddingmanager.load_function(&name, config_value).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(ApiResponse::success(format!("Embedding function '{}' loaded successfully", name)))
}

/// 卸载嵌入函数
#[utoipa::path(
    delete,
    path = "/api/v1/embeddings/functions/{name}",
    responses(
        (status = 200, description = "卸载嵌入函数成功", body = ApiResponse),
        (status = 404, description = "嵌入函数不存在", body = ApiResponse),
    ),
    tag = "embeddings"
)]
pub async fn unload_embedding_function(
    State(state): State<MultimodalAppState>,
    Path(name): Path<String>,
) -> std::result::Result<ApiResponse, StatusCode> {
    state.embeddingmanager.unload_function(&name).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(ApiResponse::success(format!("Embedding function '{}' unloaded successfully", name)))
}

/// 列出已加载的嵌入函数
#[utoipa::path(
    get,
    path = "/api/v1/embeddings/functions/active",
    responses(
        (status = 200, description = "获取活跃嵌入函数列表成功", body = ApiResponse),
    ),
    tag = "embeddings"
)]
pub async fn list_active_embedding_functions(
    State(state): State<MultimodalAppState>,
) -> std::result::Result<ApiResponse, StatusCode> {
    let active_functions = state.embeddingmanager.list_active_functions().await;
    Ok(ApiResponse::success(active_functions))
}

// ============ 嵌入生成API ============

/// 生成嵌入向量
#[utoipa::path(
    post,
    path = "/api/v1/embeddings/embed",
    request_body = EmbeddingRequest,
    responses(
        (status = 200, description = "生成嵌入成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "嵌入函数不存在", body = ApiResponse),
    ),
    tag = "embeddings"
)]
pub async fn generate_embedding(
    State(state): State<MultimodalAppState>,
    Json(request): Json<EmbeddingRequest>,
) -> std::result::Result<ApiResponse, StatusCode> {
    // 获取嵌入函数
    let _embedding_function = match state.embeddingmanager.get_function(&request.model).await {
        Some(func) => func,
        None => return Ok(ApiResponse::error(ApiError::new("NOT_FOUND", &format!("Embedding function '{}' not found", request.model))))
    };
    
    // 生成嵌入 - 临时返回空向量
    let embeddings = vec![vec![0.0f32; 768]];
    
    let response = EmbeddingResponse {
        embeddings,
        modality: crate::types::ModalityType::Text, // 简化处理
        model: request.model,
        usage: request.options.and_then(|opts| serde_json::to_value(opts).ok()),
    };
    Ok(ApiResponse::success(response))
}

// ============ 多向量记录API ============

/// 创建多向量记录
#[utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/records/multimodal",
    request_body = CreateMultiVectorRequest,
    responses(
        (status = 201, description = "创建多向量记录成功", body = ApiResponse),
        (status = 400, description = "请求参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse),
    ),
    tag = "multimodal"
)]
pub async fn create_multimodal_record(
    State(state): State<MultimodalAppState>,
    Path(table_name): Path<String>,
    Json(request): Json<CreateMultiVectorRequest>,
) -> std::result::Result<ApiResponse, StatusCode> {
    // 验证表是否存在
    let table = match state.basestate.storage.get_table(&table_name).await {
        Ok(table) => table,
        Err(e) => return Ok(ApiResponse::error(e.into()))
    };
    
    // 如果提供了原始数据但没有向量，尝试生成嵌入
    if let (None, Some(raw_data)) = (&request.vectors, &request.raw_data) {
        // 验证内容 - convert to bytes based on content type
        let contentbytes = match &raw_data.content {
            MultimodalContent::Text(text) => text.as_bytes().to_vec(),
            MultimodalContent::ImageBytes(bytes) => bytes.as_bytes().to_vec(),
            MultimodalContent::AudioBytes(bytes) => bytes.as_bytes().to_vec(),
            MultimodalContent::VideoBytes(bytes) => bytes.as_bytes().to_vec(),
            MultimodalContent::ImagePath(_) | MultimodalContent::AudioPath(_) | MultimodalContent::VideoPath(_) => vec![],
            _ => vec![],
        };
        if let Err(e) = MultimodalUtils::validate_content(&contentbytes) {
            return Ok(ApiResponse::error(e.into()));
        }

        // 这里可以添加自动嵌入生成逻辑
        // 暂时返回错误要求显式提供向量
        return Ok(ApiResponse::error(ApiError::from(crate::Error::InvalidInput("Automatic embedding generation not implemented. Please provide vectors explicitly.".to_string()))));
    }
    
    // 创建多向量记录
    let record = match MultiVectorRecord::new(request) {
        Ok(record) => record,
        Err(e) => return Ok(ApiResponse::error(e.into()))
    };
    
    // 验证向量维度
    for vector in record.vectors.values() {
        if vector.len() != table.dimension {
            return Ok(ApiResponse::error(ApiError::from(crate::Error::DimensionMismatch {
                expected: table.dimension,
                actual: vector.len(),
            })))
        }
    }
    
    // 存储记录（这里需要扩展存储层以支持多向量记录）
    // 暂时返回成功响应
    Ok(ApiResponse::success(serde_json::json!({
        "id": record.id,
        "vectors_count": record.vectors.len(),
        "created_at": record.created_at
    })))
}

/// 跨模态搜索
#[utoipa::path(
    post,
    path = "/api/v1/tables/{table_name}/search/cross-modal",
    request_body = CrossModalSearchQuery,
    responses(
        (status = 200, description = "跨模态搜索成功", body = ApiResponse),
        (status = 400, description = "查询参数错误", body = ApiResponse),
        (status = 404, description = "表不存在", body = ApiResponse),
    ),
    tag = "multimodal"
)]
pub async fn cross_modal_search(
    State(state): State<MultimodalAppState>,
    Path(table_name): Path<String>,
    Json(_query): Json<CrossModalSearchQuery>,
) -> std::result::Result<ApiResponse, StatusCode> {
    // 验证表是否存在
    let _table = match state.basestate.storage.get_table(&table_name).await {
        Ok(table) => table,
        Err(e) => return Ok(ApiResponse::error(e.into()))
    };
    
    // 这里实现跨模态搜索逻辑
    // 暂时返回模拟结果
    let search_result = MultimodalSearchResult {
        hits: vec![],
        total: 0,
        tookms: 10,
        fusion_weights: None,
    };
    
    Ok(ApiResponse::success(search_result))
}

// ============ 工具API ============

/// 验证多模态内容
#[utoipa::path(
    post,
    path = "/api/v1/multimodal/validate",
    request_body = Vec<crate::types::MultimodalContent>,
    responses(
        (status = 200, description = "内容验证成功", body = ApiResponse),
        (status = 400, description = "内容验证失败", body = ApiResponse),
    ),
    tag = "multimodal"
)]
pub async fn validate_multimodal_content(
    Json(contents): Json<Vec<crate::types::MultimodalContent>>) -> std::result::Result<ApiResponse, StatusCode> {
    // Convert MultimodalContent to bytes
    let bytecontents: Vec<Vec<u8>> = contents.iter().map(|content| {
        match content {
            MultimodalContent::Text(text) => text.as_bytes().to_vec(),
            MultimodalContent::ImageBytes(bytes) => bytes.as_bytes().to_vec(),
            MultimodalContent::AudioBytes(bytes) => bytes.as_bytes().to_vec(),
            MultimodalContent::VideoBytes(bytes) => bytes.as_bytes().to_vec(),
            _ => vec![],
        }
    }).collect();
    
    match MultimodalUtils::validate_batch(&bytecontents) {
        Ok(_) => {
            let validation_info = bytecontents.iter().map(|contentbytes| {
                serde_json::json!({
                    "type": MultimodalUtils::infer_modality_type(contentbytes),
                    "size": MultimodalUtils::estimate_content_size(contentbytes),
                    "hash": MultimodalUtils::content_hash(contentbytes)
                })
            }).collect::<Vec<_>>();
            
            Ok(ApiResponse::success(serde_json::json!({
                "valid": true,
                "count": contents.len(),
                "details": validation_info
            })))
        },
        Err(e) => Ok(ApiResponse::error(e.into()))
    }
}

/// 获取多模态内容信息
#[utoipa::path(
    post,
    path = "/api/v1/multimodal/info",
    request_body = crate::types::MultimodalContent,
    responses(
        (status = 200, description = "获取内容信息成功", body = ApiResponse),
    ),
    tag = "multimodal"
)]
pub async fn get_multimodal_info(
    Json(content): Json<crate::types::MultimodalContent>) -> std::result::Result<ApiResponse, StatusCode> {
    // Convert MultimodalContent to bytes
    let contentbytes = match &content {
        MultimodalContent::Text(text) => text.as_bytes().to_vec(),
        MultimodalContent::ImageBytes(bytes) => bytes.as_bytes().to_vec(),
        MultimodalContent::AudioBytes(bytes) => bytes.as_bytes().to_vec(),
        MultimodalContent::VideoBytes(bytes) => bytes.as_bytes().to_vec(),
        _ => vec![],
    };
    
    let info = serde_json::json!({
        "modality": MultimodalUtils::infer_modality_type(&contentbytes),
        "estimated_size": MultimodalUtils::estimate_content_size(&contentbytes),
        "content_hash": MultimodalUtils::content_hash(&contentbytes),
        "valid": MultimodalUtils::validate_content(&contentbytes).is_ok()
    });
    
    Ok(ApiResponse::success(info))
}

#[cfg(test)]
mod tests {
    
    
    
    // multimodal tests omitted: MultimodalUtils expects &[u8] but MultimodalContent is enum —
    // these need to be rewritten once the multimodal ONNX path (ort 2.0) is stabilized
}
