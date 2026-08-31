//! Simplified Sentence Transformers implementation (ONNX dependencies commented out for compilation)

use async_trait::async_trait;
use std::path::PathBuf;

use crate::embeddings::{EmbeddingError, EmbeddingFunction};
use crate::types::DenseVector;

/// Configuration for Sentence Transformers
#[derive(Debug, Clone)]
pub struct SentenceTransformersConfig {
    pub model_name: String,
    pub model_path: Option<PathBuf>,
    pub dimension: usize,
    pub max_length: usize,
    pub normalize: bool,
}

impl Default for SentenceTransformersConfig {
    fn default() -> Self {
        Self {
            model_name: "all-MiniLM-L6-v2".to_string(),
            model_path: None,
            dimension: 384,
            max_length: 512,
            normalize: true,
        }
    }
}

/// Sentence Transformers embedding implementation
pub struct SentenceTransformersEmbedding {
    config: SentenceTransformersConfig,
    // backend: Arc<RwLock<Option<OnnxBackend>>>,
}

impl SentenceTransformersEmbedding {
    pub fn new(config: SentenceTransformersConfig) -> Self {
        Self {
            config,
            // backend: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn load_model(&self) -> Result<(), EmbeddingError> {
        // Temporarily return success for compilation
        Ok(())
    }
}

#[async_trait]
impl EmbeddingFunction for SentenceTransformersEmbedding {
    fn name(&self) -> &str {
        "sentence-transformers"
    }

    fn modality(&self) -> &str {
        "text"
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    async fn embed(&self, _input: &str) -> Result<DenseVector, EmbeddingError> {
        // Simplified implementation for compilation - returns dummy vector
        Ok(vec![0.1; self.config.dimension])
    }

    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<DenseVector>, EmbeddingError> {
        // Simplified implementation for compilation - returns dummy vectors
        let dummy_vectors: Vec<DenseVector> = inputs
            .iter()
            .map(|_| vec![0.1; self.config.dimension])
            .collect();
        Ok(dummy_vectors)
    }
}
