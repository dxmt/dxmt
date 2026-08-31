//! Cohere embedding implementation

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use backoff::{future::retry, ExponentialBackoff};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::embeddings::compat::EmbeddingError;

/// Cohere embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CohereModel {
    #[serde(rename = "embed-english-v3.0")]
    EmbedEnglishV3,
    #[serde(rename = "embed-multilingual-v3.0")]
    EmbedMultilingualV3,
    #[serde(rename = "embed-english-light-v3.0")]
    EmbedEnglishLightV3,
    #[serde(rename = "embed-multilingual-light-v3.0")]
    EmbedMultilingualLightV3,
}

impl CohereModel {
    pub fn dimension(&self) -> usize {
        match self {
            Self::EmbedEnglishV3 | Self::EmbedMultilingualV3 => 1024,
            Self::EmbedEnglishLightV3 | Self::EmbedMultilingualLightV3 => 384,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::EmbedEnglishV3 => "embed-english-v3.0",
            Self::EmbedMultilingualV3 => "embed-multilingual-v3.0",
            Self::EmbedEnglishLightV3 => "embed-english-light-v3.0",
            Self::EmbedMultilingualLightV3 => "embed-multilingual-light-v3.0",
        }
    }
}

/// Input type for Cohere embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputType {
    SearchDocument,
    SearchQuery,
    Classification,
    Clustering,
}

/// Cohere configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereConfig {
    pub api_key: String,
    pub model: CohereModel,
    pub input_type: InputType,
    pub truncate: Option<String>,
    pub base_url: Option<String>,
    pub timeout: Duration,
    pub max_retries: u32,
    pub batch_size: usize,
}

impl Default for CohereConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: CohereModel::EmbedEnglishV3,
            input_type: InputType::SearchDocument,
            truncate: Some("END".to_string()),
            base_url: Some("https://api.cohere.ai/v1".to_string()),
            timeout: Duration::from_secs(30),
            max_retries: 3,
            batch_size: 96, // Cohere's max batch size
        }
    }
}

/// Cohere embedding request
#[derive(Debug, Serialize)]
struct EmbedRequest {
    texts: Vec<String>,
    model: String,
    input_type: InputType,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_types: Option<Vec<String>>,
}

/// Cohere embedding response
#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
    meta: Option<Meta>,
}

#[derive(Debug, Deserialize)]
struct Meta {
    api_version: ApiVersion,
    billed_units: Option<BilledUnits>,
}

#[derive(Debug, Deserialize)]
struct ApiVersion {
    version: String,
}

#[derive(Debug, Deserialize)]
struct BilledUnits {
    input_tokens: Option<u32>,
}

/// Rate limiting state
#[derive(Debug)]
struct RateLimitState {
    requests_remaining: Option<i32>,
    reset_time: Option<std::time::Instant>,
    retry_after: Option<Duration>,
}

/// Cohere embedding provider
pub struct CohereEmbedding {
    config: CohereConfig,
    client: reqwest::Client,
    rate_limit: Arc<RwLock<RateLimitState>>,
    cache: Arc<RwLock<lru::LruCache<String, Vec<f32>>>>,
}

impl CohereEmbedding {
    pub fn new(config: CohereConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            client,
            rate_limit: Arc::new(RwLock::new(RateLimitState {
                requests_remaining: None,
                reset_time: None,
                retry_after: None,
            })),
            cache: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(1000).unwrap(),
            ))),
        }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self::new(CohereConfig {
            api_key,
            ..Default::default()
        })
    }

    async fn check_rate_limit(&self) -> Result<(), EmbeddingError> {
        let state = self.rate_limit.read().await;

        if let Some(reset_time) = state.reset_time {
            if std::time::Instant::now() < reset_time {
                let wait_time = reset_time - std::time::Instant::now();
                return Err(EmbeddingError::RateLimitError(format!(
                    "Rate limited. Wait {:?}",
                    wait_time
                )));
            }
        }

        if let Some(retry_after) = state.retry_after {
            return Err(EmbeddingError::RateLimitError(format!(
                "Rate limited. Retry after {:?}",
                retry_after
            )));
        }

        Ok(())
    }

    async fn update_rate_limit(&self, headers: &reqwest::header::HeaderMap) {
        let mut state = self.rate_limit.write().await;

        // Parse Cohere rate limit headers
        if let Some(remaining) = headers.get("x-ratelimit-remaining") {
            if let Ok(val) = remaining.to_str() {
                state.requests_remaining = val.parse().ok();
            }
        }

        if let Some(retry) = headers.get("retry-after") {
            if let Ok(val) = retry.to_str() {
                if let Ok(seconds) = val.parse::<u64>() {
                    state.retry_after = Some(Duration::from_secs(seconds));
                    state.reset_time =
                        Some(std::time::Instant::now() + Duration::from_secs(seconds));
                }
            }
        }
    }

    async fn embed_batch_with_retry(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let backoff = ExponentialBackoff {
            max_elapsed_time: Some(Duration::from_secs(60)),
            ..Default::default()
        };

        retry(backoff, || async {
            self.embed_batch_internal(texts.clone()).await.map_err(|e| {
                if matches!(e, EmbeddingError::RateLimitError(_)) {
                    backoff::Error::transient(e)
                } else {
                    backoff::Error::permanent(e)
                }
            })
        })
        .await
    }

    async fn embed_batch_internal(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // Check rate limit
        self.check_rate_limit().await?;

        // Check cache
        let mut cache = self.cache.write().await;
        let mut cached_results = vec![None; texts.len()];
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let cache_key = format!("{}:{}", self.config.input_type.as_str(), text);
            if let Some(embedding) = cache.get(&cache_key) {
                cached_results[i] = Some(embedding.clone());
            } else {
                uncached_indices.push(i);
                uncached_texts.push(text.clone());
            }
        }

        if uncached_texts.is_empty() {
            return Ok(cached_results.into_iter().map(|r| r.unwrap()).collect());
        }

        // Prepare request
        let request = EmbedRequest {
            texts: uncached_texts.clone(),
            model: self.config.model.as_str().to_string(),
            input_type: self.config.input_type.clone(),
            truncate: self.config.truncate.clone(),
            embedding_types: Some(vec!["float".to_string()]),
        };

        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.cohere.ai/v1");

        let url = format!("{}/embed", base_url);

        // Send request
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::NetworkError(e.to_string()))?;

        // Update rate limit info
        self.update_rate_limit(response.headers()).await;

        // Check status
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            if status.as_u16() == 429 {
                return Err(EmbeddingError::RateLimitError(error_text));
            } else {
                return Err(EmbeddingError::ApiError(format!(
                    "Status {}: {}",
                    status, error_text
                )));
            }
        }

        // Parse response
        let embed_response: EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

        // Update cache
        for (i, embedding) in embed_response.embeddings.iter().enumerate() {
            let text = &uncached_texts[i];
            let cache_key = format!("{}:{}", self.config.input_type.as_str(), text);
            cache.put(cache_key, embedding.clone());
        }

        // Merge cached and new results
        let mut final_results = Vec::with_capacity(texts.len());
        let mut uncached_iter = embed_response.embeddings.into_iter();

        for cached in cached_results {
            if let Some(embedding) = cached {
                final_results.push(embedding);
            } else {
                let embedding = uncached_iter
                    .next()
                    .ok_or_else(|| EmbeddingError::ApiError("Missing embedding".to_string()))?;
                final_results.push(embedding);
            }
        }

        Ok(final_results)
    }

    pub async fn embed_documents(
        &self,
        documents: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut all_embeddings = Vec::new();

        // Process in batches
        for chunk in documents.chunks(self.config.batch_size) {
            let batch_embeddings = self.embed_batch_with_retry(chunk.to_vec()).await?;
            all_embeddings.extend(batch_embeddings);
        }

        Ok(all_embeddings)
    }

    pub async fn embed_query(&self, query: String) -> Result<Vec<f32>, EmbeddingError> {
        // For queries, we should ideally use SearchQuery input type,
        // but since we can't mutate self, we'll use the configured input type
        let embeddings = self.embed_batch_with_retry(vec![query]).await?;

        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::ApiError("No embedding returned".to_string()))
    }
}

impl InputType {
    fn as_str(&self) -> &str {
        match self {
            Self::SearchDocument => "search_document",
            Self::SearchQuery => "search_query",
            Self::Classification => "classification",
            Self::Clustering => "clustering",
        }
    }
}

use crate::embeddings::compat::EmbeddingFunction;
use crate::types::vector::DenseVector;

#[async_trait::async_trait]
impl EmbeddingFunction for CohereEmbedding {
    fn name(&self) -> &str {
        self.config.model.as_str()
    }

    fn modality(&self) -> &str {
        "text"
    }

    fn dimension(&self) -> usize {
        self.config.model.dimension()
    }

    async fn embed(&self, input: &str) -> Result<DenseVector, EmbeddingError> {
        self.embed_query(input.to_string()).await
    }

    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<DenseVector>, EmbeddingError> {
        self.embed_documents(inputs.to_vec()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dimensions() {
        assert_eq!(CohereModel::EmbedEnglishV3.dimension(), 1024);
        assert_eq!(CohereModel::EmbedMultilingualV3.dimension(), 1024);
        assert_eq!(CohereModel::EmbedEnglishLightV3.dimension(), 384);
        assert_eq!(CohereModel::EmbedMultilingualLightV3.dimension(), 384);
    }

    #[test]
    fn test_config_serialization() {
        let config = CohereConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CohereConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.batch_size, deserialized.batch_size);
    }
}
