//! OpenAI embedding implementation with production features

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use backoff::{future::retry, ExponentialBackoff};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::embeddings::compat::EmbeddingError;

/// OpenAI embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpenAIModel {
    #[serde(rename = "text-embedding-3-small")]
    TextEmbedding3Small,
    #[serde(rename = "text-embedding-3-large")]
    TextEmbedding3Large,
    #[serde(rename = "text-embedding-ada-002")]
    TextEmbeddingAda002,
}

impl OpenAIModel {
    pub fn dimension(&self) -> usize {
        match self {
            Self::TextEmbedding3Small => 1536,
            Self::TextEmbedding3Large => 3072,
            Self::TextEmbeddingAda002 => 1536,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::TextEmbedding3Small => "text-embedding-3-small",
            Self::TextEmbedding3Large => "text-embedding-3-large",
            Self::TextEmbeddingAda002 => "text-embedding-ada-002",
        }
    }

    pub fn max_tokens(&self) -> usize {
        match self {
            Self::TextEmbedding3Small => 8191,
            Self::TextEmbedding3Large => 8191,
            Self::TextEmbeddingAda002 => 8191,
        }
    }
}

/// OpenAI embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub model: OpenAIModel,
    pub organization_id: Option<String>,
    pub base_url: Option<String>,
    pub timeout: Duration,
    pub max_retries: u32,
    pub batch_size: usize,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: OpenAIModel::TextEmbedding3Small,
            organization_id: None,
            base_url: Some("https://api.openai.com/v1".to_string()),
            timeout: Duration::from_secs(30),
            max_retries: 3,
            batch_size: 100,
        }
    }
}

/// OpenAI embedding request
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

/// OpenAI embedding response
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    model: String,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

/// Rate limiting state
#[derive(Debug)]
struct RateLimitState {
    requests_remaining: Option<i32>,
    tokens_remaining: Option<i32>,
    reset_time: Option<std::time::Instant>,
}

/// OpenAI embedding provider with production features
pub struct OpenAIEmbedding {
    config: OpenAIConfig,
    client: reqwest::Client,
    rate_limit: Arc<RwLock<RateLimitState>>,
    cache: Arc<RwLock<lru::LruCache<String, Vec<f32>>>>,
}

impl OpenAIEmbedding {
    pub fn new(config: OpenAIConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            client,
            rate_limit: Arc::new(RwLock::new(RateLimitState {
                requests_remaining: None,
                tokens_remaining: None,
                reset_time: None,
            })),
            cache: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(1000).unwrap(),
            ))),
        }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self::new(OpenAIConfig {
            api_key,
            ..Default::default()
        })
    }

    async fn check_rate_limit(&self) -> Result<(), EmbeddingError> {
        let state = self.rate_limit.read().await;

        // Check if we're rate limited
        if let Some(reset_time) = state.reset_time {
            if std::time::Instant::now() < reset_time {
                let wait_time = reset_time - std::time::Instant::now();
                return Err(EmbeddingError::RateLimitError(format!(
                    "Rate limited. Wait {:?}",
                    wait_time
                )));
            }
        }

        // Check remaining requests
        if let Some(remaining) = state.requests_remaining {
            if remaining <= 0 {
                return Err(EmbeddingError::RateLimitError(
                    "No requests remaining".to_string(),
                ));
            }
        }

        Ok(())
    }

    async fn update_rate_limit(&self, headers: &reqwest::header::HeaderMap) {
        let mut state = self.rate_limit.write().await;

        // Parse rate limit headers
        if let Some(remaining) = headers.get("x-ratelimit-remaining-requests") {
            if let Ok(val) = remaining.to_str() {
                state.requests_remaining = val.parse().ok();
            }
        }

        if let Some(remaining) = headers.get("x-ratelimit-remaining-tokens") {
            if let Ok(val) = remaining.to_str() {
                state.tokens_remaining = val.parse().ok();
            }
        }

        if let Some(reset) = headers.get("x-ratelimit-reset-requests") {
            if let Ok(val) = reset.to_str() {
                if let Ok(timestamp) = val.parse::<u64>() {
                    let duration = Duration::from_secs(timestamp);
                    state.reset_time = Some(std::time::Instant::now() + duration);
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
            if let Some(embedding) = cache.get(text) {
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
        let request = EmbeddingRequest {
            input: uncached_texts.clone(),
            model: self.config.model.as_str().to_string(),
            encoding_format: Some("float".to_string()),
            dimensions: None,
            user: None,
        };

        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");

        let url = format!("{}/embeddings", base_url);

        // Build request
        let mut req = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");

        if let Some(org_id) = &self.config.organization_id {
            req = req.header("OpenAI-Organization", org_id);
        }

        // Send request
        let response = req
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
        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

        // Sort embeddings by index
        let mut embeddings = embedding_response.data;
        embeddings.sort_by_key(|e| e.index);

        // Update cache and build results
        for (i, embedding_data) in embeddings.iter().enumerate() {
            let text = &uncached_texts[i];
            cache.put(text.clone(), embedding_data.embedding.clone());
        }

        // Merge cached and new results
        let mut final_results = Vec::with_capacity(texts.len());
        let mut uncached_iter = embeddings.into_iter();

        for cached in cached_results {
            if let Some(embedding) = cached {
                final_results.push(embedding);
            } else {
                let embedding_data = uncached_iter
                    .next()
                    .ok_or_else(|| EmbeddingError::ApiError("Missing embedding".to_string()))?;
                final_results.push(embedding_data.embedding);
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
        let embeddings = self.embed_batch_with_retry(vec![query]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::ApiError("No embedding returned".to_string()))
    }
}

use crate::embeddings::compat::EmbeddingFunction;
use crate::types::vector::DenseVector;

#[async_trait::async_trait]
impl EmbeddingFunction for OpenAIEmbedding {
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
        assert_eq!(OpenAIModel::TextEmbedding3Small.dimension(), 1536);
        assert_eq!(OpenAIModel::TextEmbedding3Large.dimension(), 3072);
        assert_eq!(OpenAIModel::TextEmbeddingAda002.dimension(), 1536);
    }

    #[test]
    fn test_config_serialization() {
        let config = OpenAIConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OpenAIConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.batch_size, deserialized.batch_size);
    }
}
