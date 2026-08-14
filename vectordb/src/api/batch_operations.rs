use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::instrument;

use crate::{
    api::{handlers::AppState, response::ApiResponse},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchQuery},
        vector::RecordId,
    },
};

/// 批量操作配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// 并发度
    pub concurrency: usize,
    /// 批次大小
    pub batchsize: usize,
    /// 失败时是否继续
    pub continue_on_error: bool,
    /// 是否返回详细结果
    pub detailed_results: bool,
    /// 超时时间（秒）
    pub timeout_seconds: Option<u64>,
    /// 是否使用事务
    pub use_transaction: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            concurrency: 4,
            batchsize: 1000,
            continue_on_error: true,
            detailed_results: false,
            timeout_seconds: Some(300),
            use_transaction: false,
        }
    }
}

/// 批量插入请求
#[derive(Debug, Deserialize, Serialize)]
pub struct BatchInsertRequest {
    pub records: Vec<CreateRecordRequest>,
    pub config: Option<BatchConfig>,
}

/// 批量更新请求
#[derive(Debug, Deserialize, Serialize)]
pub struct BatchUpdateRequest {
    pub updates: Vec<UpdateItem>,
    pub config: Option<BatchConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UpdateItem {
    pub id: RecordId,
    pub vector: Option<Vec<f32>>,
    pub payload: Option<serde_json::Value>,
}

/// 批量删除请求
#[derive(Debug, Deserialize, Serialize)]
pub struct BatchDeleteRequest {
    pub ids: Vec<RecordId>,
    pub config: Option<BatchConfig>,
}

/// 批量搜索请求
#[derive(Debug, Deserialize, Serialize)]
pub struct BatchSearchRequest {
    pub queries: Vec<SearchQuery>,
    pub config: Option<BatchConfig>,
}

/// 批量操作结果
#[derive(Debug, Serialize)]
pub struct BatchOperationResult {
    pub successcount: usize,
    pub failure_count: usize,
    pub totalcount: usize,
    pub duration_ms: u128,
    pub errors: Vec<BatchError>,
    pub results: Option<Vec<IndividualResult>>,
}

#[derive(Debug, Serialize)]
pub struct BatchError {
    pub index: usize,
    pub id: Option<RecordId>,
    pub error: String,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum IndividualResult {
    Success { id: RecordId },
    Failure { id: Option<RecordId>, error: String },
    SearchResult { hits: Vec<SearchHit> },
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub id: RecordId,
    pub score: f32,
    pub payload: Option<serde_json::Value>,
}

/// 批量插入处理器
pub struct BatchProcessor {
    storage: Arc<IntelligentIndexedStorage>,
    semaphore: Arc<Semaphore>,
}

impl BatchProcessor {
    pub fn new(storage: Arc<IntelligentIndexedStorage>, maxconcurrency: usize) -> Self {
        Self {
            storage,
            semaphore: Arc::new(Semaphore::new(maxconcurrency)),
        }
    }

    /// 批量插入记录
    #[instrument(skip(self, tablename, request))]
    pub async fn batch_insert(
        &self,
        tablename: String,
        request: BatchInsertRequest,
    ) -> crate::Result<BatchOperationResult> {
        let config = request.config.unwrap_or_default();
        let start_time = std::time::Instant::now();

        let mut success_count = 0;
        let mut failure_count = 0;
        let mut errors = Vec::new();
        let mut results = if config.detailed_results {
            Some(Vec::with_capacity(request.records.len()))
        } else {
            None
        };

        // 将记录分批处理
        let chunks: Vec<_> = request
            .records
            .chunks(config.batchsize)
            .map(|chunk| chunk.to_vec())
            .collect();

        // 并发处理每个批次
        let mut futures = Vec::new();
        for (batch_idx, chunk) in chunks.into_iter().enumerate() {
            let storage = self.storage.clone();
            let table = tablename.clone();
            let semaphore = self.semaphore.clone();

            futures.push(async move {
                let _permit = semaphore
                    .acquire()
                    .await
                    .map_err(|e| crate::Error::Internal(e.to_string()))?;
                let mut batch_results = Vec::new();

                // Use batch insert for each chunk
                match storage.insert_records(&table, chunk.clone()).await {
                    Ok(records) => {
                        for (idx, record) in records.into_iter().enumerate() {
                            let global_idx = batch_idx * config.batchsize + idx;
                            batch_results.push((global_idx, Ok(record.id)))
                        }
                    }
                    Err(e) => {
                        if !config.continue_on_error {
                            return Err(e);
                        }
                        // If batch fails, add errors for all records in chunk
                        for (idx, record) in chunk.into_iter().enumerate() {
                            let global_idx = batch_idx * config.batchsize + idx;
                            let err = crate::Error::Internal(format!("Batch insert failed: {}", e));
                            batch_results.push((global_idx, Err((record.id, err))));
                        }
                    }
                }

                Ok(batch_results)
            });
        }

        // 收集所有结果
        let batch_results = futures::future::join_all(futures).await;

        for batch_result in batch_results {
            match batch_result {
                Ok(items) => {
                    for (idx, result) in items {
                        match result {
                            Ok(id) => {
                                success_count += 1;
                                if let Some(ref mut res) = results {
                                    res.push(IndividualResult::Success { id });
                                }
                            }
                            Err((id, e)) => {
                                failure_count += 1;
                                errors.push(BatchError {
                                    index: idx,
                                    id: Some(id.clone()),
                                    error: e.to_string(),
                                });
                                if let Some(ref mut res) = results {
                                    res.push(IndividualResult::Failure {
                                        id: Some(id),
                                        error: e.to_string(),
                                    })
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        Ok(BatchOperationResult {
            successcount: success_count,
            failure_count,
            totalcount: request.records.len(),
            duration_ms: start_time.elapsed().as_millis(),
            errors,
            results,
        })
    }

    /// 批量更新记录
    #[instrument(skip(self, tablename, request))]
    pub async fn batch_update(
        &self,
        tablename: String,
        request: BatchUpdateRequest,
    ) -> crate::Result<BatchOperationResult> {
        let config = request.config.unwrap_or_default();
        let start_time = std::time::Instant::now();

        let mut success_count = 0;
        let mut failure_count = 0;
        let mut errors = Vec::new();
        let mut results = if config.detailed_results {
            Some(Vec::with_capacity(request.updates.len()))
        } else {
            None
        };

        // 并发处理更新
        let mut updatefutures = Vec::new();
        for (idx, update) in request.updates.into_iter().enumerate() {
            let storage = self.storage.clone();
            let table = tablename.clone();
            let semaphore = self.semaphore.clone();

            updatefutures.push(async move {
                let _permit = match semaphore.acquire().await {
                    Ok(p) => p,
                    Err(e) => return (idx, Err((None, crate::Error::Internal(e.to_string())))),
                };

                let update_request = CreateRecordRequest {
                    id: update.id.clone(),
                    vector: update.vector.unwrap_or_default(),
                    sparse_vector: None,
                    payload: update.payload,
                    named_vectors: None,
                };

                match storage
                    .update_record(&table, &update.id, update_request)
                    .await
                {
                    Ok(_) => (idx, Ok(update.id)),
                    Err(e) => (idx, Err((Some(update.id), e))),
                }
            });
        }

        let update_results = futures::future::join_all(updatefutures).await;
        let total_count = update_results.len();

        for (idx, result) in update_results {
            match result {
                Ok(id) => {
                    success_count += 1;
                    if let Some(ref mut res) = results {
                        res.push(IndividualResult::Success { id });
                    }
                }
                Err((id, e)) => {
                    failure_count += 1;
                    errors.push(BatchError {
                        index: idx,
                        id: id.clone(),
                        error: e.to_string(),
                    });
                    if let Some(ref mut res) = results {
                        res.push(IndividualResult::Failure {
                            id,
                            error: e.to_string(),
                        })
                    }
                }
            }
        }

        Ok(BatchOperationResult {
            successcount: success_count,
            failure_count,
            totalcount: total_count,
            duration_ms: start_time.elapsed().as_millis(),
            errors,
            results,
        })
    }

    /// 批量删除记录
    #[instrument(skip(self, tablename, request))]
    pub async fn batch_delete(
        &self,
        tablename: String,
        request: BatchDeleteRequest,
    ) -> crate::Result<BatchOperationResult> {
        let _config = request.config.unwrap_or_default();
        let start_time = std::time::Instant::now();

        // 使用存储层的批量删除方法
        match self
            .storage
            .delete_records(&tablename, request.ids.clone())
            .await
        {
            Ok(_) => {
                // Assume all were deleted successfully
                Ok(BatchOperationResult {
                    successcount: request.ids.len(),
                    failure_count: 0,
                    totalcount: request.ids.len(),
                    duration_ms: start_time.elapsed().as_millis(),
                    errors: Vec::new(),
                    results: None,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// 批量搜索
    #[instrument(skip(self, tablename, request))]
    pub async fn batch_search(
        &self,
        tablename: String,
        request: BatchSearchRequest,
    ) -> crate::Result<BatchOperationResult> {
        let _config = request.config.unwrap_or_default();
        let start_time = std::time::Instant::now();

        let mut success_count = 0;
        let mut failure_count = 0;
        let mut errors = Vec::new();
        let mut results = Some(Vec::with_capacity(request.queries.len()));

        // 并发执行搜索
        let mut searchfutures = Vec::new();
        for (idx, query) in request.queries.into_iter().enumerate() {
            let storage = self.storage.clone();
            let table = tablename.clone();
            let semaphore = self.semaphore.clone();

            searchfutures.push(async move {
                let _permit = match semaphore.acquire().await {
                    Ok(p) => p,
                    Err(e) => return (idx, Err(crate::Error::Internal(e.to_string()))),
                };

                match storage.search(&table, query).await {
                    Ok(search_result) => {
                        let hits: Vec<SearchHit> = search_result
                            .hits
                            .into_iter()
                            .map(|hit| SearchHit {
                                id: hit.id,
                                score: hit.score,
                                payload: hit.payload,
                            })
                            .collect();
                        (idx, Ok(hits))
                    }
                    Err(e) => (idx, Err(e)),
                }
            });
        }

        let search_results = futures::future::join_all(searchfutures).await;
        let total_count = search_results.len();

        for (idx, result) in search_results {
            match result {
                Ok(hits) => {
                    success_count += 1;
                    if let Some(ref mut res) = results {
                        res.push(IndividualResult::SearchResult { hits });
                    }
                }
                Err(e) => {
                    failure_count += 1;
                    errors.push(BatchError {
                        index: idx,
                        id: None,
                        error: e.to_string(),
                    });
                    if let Some(ref mut res) = results {
                        res.push(IndividualResult::Failure {
                            id: None,
                            error: e.to_string(),
                        })
                    }
                }
            }
        }

        Ok(BatchOperationResult {
            successcount: success_count,
            failure_count,
            totalcount: total_count,
            duration_ms: start_time.elapsed().as_millis(),
            errors,
            results,
        })
    }
}

// ============ HTTP Handlers ============

/// 批量插入记录
pub async fn batch_insert_records(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(request): Json<BatchInsertRequest>,
) -> std::result::Result<ApiResponse, StatusCode> {
    let processor = BatchProcessor::new(
        state.storage.clone(),
        request.config.as_ref().map(|c| c.concurrency).unwrap_or(4),
    );

    match processor.batch_insert(table_name, request).await {
        Ok(result) => Ok(ApiResponse::success(result)),
        Err(e) => Ok(ApiResponse::error(e.into())),
    }
}

/// 批量更新记录
pub async fn batch_update_records(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(request): Json<BatchUpdateRequest>,
) -> std::result::Result<ApiResponse, StatusCode> {
    let processor = BatchProcessor::new(
        state.storage.clone(),
        request.config.as_ref().map(|c| c.concurrency).unwrap_or(4),
    );

    match processor.batch_update(table_name, request).await {
        Ok(result) => Ok(ApiResponse::success(result)),
        Err(e) => Ok(ApiResponse::error(e.into())),
    }
}

/// 批量删除记录
pub async fn batch_delete_records(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(request): Json<BatchDeleteRequest>,
) -> std::result::Result<ApiResponse, StatusCode> {
    let processor = BatchProcessor::new(
        state.storage.clone(),
        request.config.as_ref().map(|c| c.concurrency).unwrap_or(4),
    );

    match processor.batch_delete(table_name, request).await {
        Ok(result) => Ok(ApiResponse::success(result)),
        Err(e) => Ok(ApiResponse::error(e.into())),
    }
}

/// 批量搜索
pub async fn batch_search_vectors(
    State(state): State<AppState>,
    Path(table_name): Path<String>,
    Json(request): Json<BatchSearchRequest>,
) -> std::result::Result<ApiResponse, StatusCode> {
    let processor = BatchProcessor::new(
        state.storage.clone(),
        request.config.as_ref().map(|c| c.concurrency).unwrap_or(4),
    );

    match processor.batch_search(table_name, request).await {
        Ok(result) => Ok(ApiResponse::success(result)),
        Err(e) => Ok(ApiResponse::error(e.into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.concurrency, 4);
        assert_eq!(config.batchsize, 1000);
        assert!(config.continue_on_error);
    }
}
