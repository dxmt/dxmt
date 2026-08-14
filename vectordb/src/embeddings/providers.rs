#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::compat::{EmbeddingError, EmbeddingFunction};
use crate::types::vector::DenseVector;
use async_trait::async_trait;
use std::sync::Arc;

/// Trait for embedding providers (OpenAI, HuggingFace, etc.)
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Get the name of the provider
    fn name(&self) -> &str;

    /// List available models
    async fn list_models(&self) -> Vec<String>;

    /// Create an embedding function for a specific model
    async fn create_function(
        &self,
        model_name: &str,
    ) -> Result<Arc<dyn EmbeddingFunction>, EmbeddingError>;
}

/// OpenAI embedding provider
pub struct OpenAIProvider {
    api_key: Option<String>,
    base_url: Option<String>,
}

impl OpenAIProvider {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            base_url: None,
        }
    }

    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = Some(url);
        self
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn list_models(&self) -> Vec<String> {
        vec![
            "text-embedding-3-small".to_string(),
            "text-embedding-3-large".to_string(),
            "text-embedding-ada-002".to_string(),
        ]
    }

    async fn create_function(
        &self,
        model_name: &str,
    ) -> Result<Arc<dyn EmbeddingFunction>, EmbeddingError> {
        let dimension = match model_name {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => {
                return Err(EmbeddingError::ModelError {
                    message: format!("Unknown model: {}", model_name),
                })
            }
        };

        Ok(Arc::new(OpenAIEmbeddingFunction {
            model_name: model_name.to_string(),
            dimension,
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
        }))
    }
}

struct OpenAIEmbeddingFunction {
    model_name: String,
    dimension: usize,
    api_key: Option<String>,
    base_url: Option<String>,
}

#[async_trait]
impl EmbeddingFunction for OpenAIEmbeddingFunction {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn modality(&self) -> &str {
        "text"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, _input: &str) -> Result<DenseVector, EmbeddingError> {
        // Stub — wire up actual OpenAI v1/embeddings HTTP call when reqwest is added
        Ok(vec![0.1; self.dimension])
    }

    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<DenseVector>, EmbeddingError> {
        // Stub — wire up actual OpenAI v1/embeddings batch call when reqwest is added
        Ok(inputs.iter().map(|_| vec![0.1; self.dimension]).collect())
    }
}

/// HuggingFace embedding provider
pub struct HuggingFaceProvider {
    api_key: Option<String>,
    inference_url: Option<String>,
}

impl HuggingFaceProvider {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            inference_url: None,
        }
    }

    pub fn with_inference_url(mut self, url: String) -> Self {
        self.inference_url = Some(url);
        self
    }
}

#[async_trait]
impl EmbeddingProvider for HuggingFaceProvider {
    fn name(&self) -> &str {
        "huggingface"
    }

    async fn list_models(&self) -> Vec<String> {
        vec![
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            "sentence-transformers/all-mpnet-base-v2".to_string(),
            "BAAI/bge-large-en-v1.5".to_string(),
            "BAAI/bge-base-en-v1.5".to_string(),
            "intfloat/e5-large-v2".to_string(),
        ]
    }

    async fn create_function(
        &self,
        model_name: &str,
    ) -> Result<Arc<dyn EmbeddingFunction>, EmbeddingError> {
        let dimension = match model_name {
            "sentence-transformers/all-MiniLM-L6-v2" => 384,
            "sentence-transformers/all-mpnet-base-v2" => 768,
            "BAAI/bge-large-en-v1.5" => 1024,
            "BAAI/bge-base-en-v1.5" => 768,
            "intfloat/e5-large-v2" => 1024,
            _ => 768, // Default dimension
        };

        Ok(Arc::new(HuggingFaceEmbeddingFunction {
            model_name: model_name.to_string(),
            dimension,
            api_key: self.api_key.clone(),
            inference_url: self.inference_url.clone(),
        }))
    }
}

struct HuggingFaceEmbeddingFunction {
    model_name: String,
    dimension: usize,
    api_key: Option<String>,
    inference_url: Option<String>,
}

#[async_trait]
impl EmbeddingFunction for HuggingFaceEmbeddingFunction {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn modality(&self) -> &str {
        "text"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, _input: &str) -> Result<DenseVector, EmbeddingError> {
        // Stub — wire up HuggingFace Inference API when reqwest is added
        Ok(vec![0.2; self.dimension])
    }
}

/// Local Sentence Transformer provider (using candle or ort)
pub struct SentenceTransformerProvider {
    model_path: Option<String>,
}

impl SentenceTransformerProvider {
    pub fn new() -> Self {
        Self { model_path: None }
    }

    pub fn with_model_path(mut self, path: String) -> Self {
        self.model_path = Some(path);
        self
    }
}

impl Default for SentenceTransformerProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EmbeddingProvider for SentenceTransformerProvider {
    fn name(&self) -> &str {
        "sentence-transformers"
    }

    async fn list_models(&self) -> Vec<String> {
        vec![
            "all-MiniLM-L6-v2".to_string(),
            "all-mpnet-base-v2".to_string(),
            "multi-qa-MiniLM-L6-cos-v1".to_string(),
        ]
    }

    async fn create_function(
        &self,
        model_name: &str,
    ) -> Result<Arc<dyn EmbeddingFunction>, EmbeddingError> {
        let dimension = match model_name {
            "all-MiniLM-L6-v2" => 384,
            "all-mpnet-base-v2" => 768,
            "multi-qa-MiniLM-L6-cos-v1" => 384,
            _ => 384,
        };

        Ok(Arc::new(LocalSentenceTransformerFunction {
            model_name: model_name.to_string(),
            dimension,
            model_path: self.model_path.clone(),
        }))
    }
}

struct LocalSentenceTransformerFunction {
    model_name: String,
    dimension: usize,
    model_path: Option<String>,
}

#[async_trait]
impl EmbeddingFunction for LocalSentenceTransformerFunction {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn modality(&self) -> &str {
        "text"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, _input: &str) -> Result<DenseVector, EmbeddingError> {
        // Stub — blocked on ort 2.0 stable; see CLAUDE.md multimodal section
        Ok(vec![0.3; self.dimension])
    }
}
