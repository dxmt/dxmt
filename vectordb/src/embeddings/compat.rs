use async_trait::async_trait;
use crate::types::vector::DenseVector;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub enum EmbeddingError {
    UnsupportedModality { modality: String, function_name: String },
    ModelError { message: String },
    InternalError(String),
    ModelLoadError(String),
    InferenceError(String),
    InvalidInput(String),
    RateLimitError(String),
    ApiError(String),
    ModelNotFound(String),
    NetworkError(String),
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingError::UnsupportedModality { modality, function_name } => {
                write!(f, "Unsupported modality {} for function {}", modality, function_name)
            }
            EmbeddingError::ModelError { message } => write!(f, "Model error: {}", message),
            EmbeddingError::InternalError(msg) => write!(f, "Internal error: {}", msg),
            EmbeddingError::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
            EmbeddingError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            EmbeddingError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            EmbeddingError::RateLimitError(msg) => write!(f, "Rate limit error: {}", msg),
            EmbeddingError::ApiError(msg) => write!(f, "API error: {}", msg),
            EmbeddingError::ModelNotFound(msg) => write!(f, "Model not found: {}", msg),
            EmbeddingError::NetworkError(msg) => write!(f, "Network error: {}", msg),
        }
    }
}

impl std::error::Error for EmbeddingError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    pub embeddings: Vec<DenseVector>,
    pub model: String,
    pub usage: Option<HashMap<String, serde_json::Value>>,
}

/// Trait for embedding functions
#[async_trait]
pub trait EmbeddingFunction: Send + Sync {
    /// Get the name of the embedding function
    fn name(&self) -> &str;
    
    /// Get the modality this function supports
    fn modality(&self) -> &str;
    
    /// Get the dimension of embeddings produced
    fn dimension(&self) -> usize;
    
    /// Embed a single input
    async fn embed(&self, input: &str) -> Result<DenseVector, EmbeddingError>;
    
    /// Embed a batch of inputs
    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<DenseVector>, EmbeddingError> {
        let mut results = Vec::new();
        for input in inputs {
            results.push(self.embed(input).await?);
        }
        Ok(results)
    }
}

/// Simple text embedding function implementation
pub struct TextEmbeddingFunction {
    name: String,
    dimension: usize,
}

impl TextEmbeddingFunction {
    pub fn new(name: String, dimension: usize) -> Self {
        Self { name, dimension }
    }
}

#[async_trait]
impl EmbeddingFunction for TextEmbeddingFunction {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn modality(&self) -> &str {
        "text"
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    async fn embed(&self, _input: &str) -> Result<DenseVector, EmbeddingError> {
        // Return dummy embedding for now
        Ok(vec![0.0; self.dimension])
    }
}