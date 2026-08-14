use std::sync::Arc;
use async_trait::async_trait;
use crate::types::{
    multimodal::{ModalityType, MultimodalContent},
    DenseVector,
};
use crate::embeddings::EmbeddingFunction;

/// Trait for processing multimodal content
#[async_trait]
pub trait MultimodalProcessor: Send + Sync {
    /// Get the modality this processor handles
    fn modality(&self) -> ModalityType;
    
    /// Process content and generate embedding
    async fn process(&self, content: &MultimodalContent) -> crate::Result<DenseVector>;
    
    /// Validate content format
    fn validate(&self, content: &MultimodalContent) -> crate::Result<()>;
    
    /// Get embedding dimension
    fn dimension(&self) -> usize;
}

/// Text processor using embedding functions
pub struct TextProcessor {
    embedding_function: Arc<dyn EmbeddingFunction>,
}

impl TextProcessor {
    pub fn new(embedding_function: Arc<dyn EmbeddingFunction>) -> Self {
        Self { embedding_function }
    }
}

#[async_trait]
impl MultimodalProcessor for TextProcessor {
    fn modality(&self) -> ModalityType {
        ModalityType::Text
    }
    
    async fn process(&self, content: &MultimodalContent) -> crate::Result<DenseVector> {
        match content {
            MultimodalContent::Text(text) => {
                self.embedding_function.embed(text).await
                    .map_err(|e| crate::Error::Internal(format!("Embedding failed: {:?}", e)))
            }
            _ => Err(crate::Error::InvalidInput("Expected text content".to_string()))
        }
    }
    
    fn validate(&self, content: &MultimodalContent) -> crate::Result<()> {
        match content {
            MultimodalContent::Text(text) if !text.is_empty() => Ok(()),
            MultimodalContent::Text(_) => Err(crate::Error::InvalidInput("Empty text".to_string())),
            _ => Err(crate::Error::InvalidInput("Expected text content".to_string()))
        }
    }
    
    fn dimension(&self) -> usize {
        self.embedding_function.dimension()
    }
}

/// Image processor (placeholder - would use vision models)
pub struct ImageProcessor {
    dimension: usize,
}

impl ImageProcessor {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl MultimodalProcessor for ImageProcessor {
    fn modality(&self) -> ModalityType {
        ModalityType::Image
    }
    
    async fn process(&self, content: &MultimodalContent) -> crate::Result<DenseVector> {
        match content {
            MultimodalContent::ImagePath(_path) => {
                // TODO: Load image and process with vision model
                Ok(vec![0.0; self.dimension])
            }
            MultimodalContent::ImageBytes(_data) => {
                // TODO: Decode base64 and process with vision model
                Ok(vec![0.0; self.dimension])
            }
            MultimodalContent::Embedding(vec) => Ok(vec.clone()),
            _ => Err(crate::Error::InvalidInput("Expected image content".to_string()))
        }
    }
    
    fn validate(&self, content: &MultimodalContent) -> crate::Result<()> {
        match content {
            MultimodalContent::ImagePath(_) | 
            MultimodalContent::ImageBytes(_) | 
            MultimodalContent::Embedding(_) => Ok(()),
            _ => Err(crate::Error::InvalidInput("Expected image content".to_string()))
        }
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Audio processor (placeholder - would use audio models)
pub struct AudioProcessor {
    dimension: usize,
}

impl AudioProcessor {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl MultimodalProcessor for AudioProcessor {
    fn modality(&self) -> ModalityType {
        ModalityType::Audio
    }
    
    async fn process(&self, content: &MultimodalContent) -> crate::Result<DenseVector> {
        match content {
            MultimodalContent::AudioPath(_path) => {
                // TODO: Load audio and process with audio model
                Ok(vec![0.0; self.dimension])
            }
            MultimodalContent::AudioBytes(_data) => {
                // TODO: Decode and process with audio model
                Ok(vec![0.0; self.dimension])
            }
            MultimodalContent::Embedding(vec) => Ok(vec.clone()),
            _ => Err(crate::Error::InvalidInput("Expected audio content".to_string()))
        }
    }
    
    fn validate(&self, content: &MultimodalContent) -> crate::Result<()> {
        match content {
            MultimodalContent::AudioPath(_) | 
            MultimodalContent::AudioBytes(_) | 
            MultimodalContent::Embedding(_) => Ok(()),
            _ => Err(crate::Error::InvalidInput("Expected audio content".to_string()))
        }
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
}