//! Sentence Transformers embedding implementation for local models

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::path::PathBuf;

use crate::embeddings::compat::{EmbeddingFunction, EmbeddingError, EmbeddingResult};
use crate::types::DenseVector;

/// Popular sentence transformer models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SentenceTransformerModel {
    #[serde(rename = "all-MiniLM-L6-v2")]
    AllMiniLML6V2,
    #[serde(rename = "all-MiniLM-L12-v2")]
    AllMiniLML12V2,
    #[serde(rename = "all-mpnet-base-v2")]
    AllMpnetBaseV2,
    #[serde(rename = "multi-qa-MiniLM-L6-cos-v1")]
    MultiQaMiniLML6CosV1,
    #[serde(rename = "paraphrase-multilingual-MiniLM-L12-v2")]
    ParaphraseMultilingualMiniLML12V2,
    #[serde(rename = "custom")]
    Custom(String),
}

impl SentenceTransformerModel {
    pub fn model_name(&self) -> String {
        match self {
            Self::AllMiniLML6V2 => "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            Self::AllMiniLML12V2 => "sentence-transformers/all-MiniLM-L12-v2".to_string(),
            Self::AllMpnetBaseV2 => "sentence-transformers/all-mpnet-base-v2".to_string(),
            Self::MultiQaMiniLML6CosV1 => "sentence-transformers/multi-qa-MiniLM-L6-cos-v1".to_string(),
            Self::ParaphraseMultilingualMiniLML12V2 => "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string(),
            Self::Custom(name) => name.clone(),
        }
    }
    
    pub fn dimension(&self) -> usize {
        match self {
            Self::AllMiniLML6V2 => 384,
            Self::AllMiniLML12V2 => 384,
            Self::AllMpnetBaseV2 => 768,
            Self::MultiQaMiniLML6CosV1 => 384,
            Self::ParaphraseMultilingualMiniLML12V2 => 384,
            Self::Custom(_) => 768, // Default, should be determined at runtime
        }
    }
    
    pub fn max_sequence_length(&self) -> usize {
        match self {
            Self::AllMiniLML6V2 => 256,
            Self::AllMiniLML12V2 => 256,
            Self::AllMpnetBaseV2 => 384,
            Self::MultiQaMiniLML6CosV1 => 512,
            Self::ParaphraseMultilingualMiniLML12V2 => 128,
            Self::Custom(_) => 512, // Default
        }
    }
}

/// Configuration for sentence transformers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceTransformerConfig {
    pub model: SentenceTransformerModel,
    pub cache_dir: Option<PathBuf>,
    pub device: Device,
    pub normalize_embeddings: bool,
    pub batch_size: usize,
    pub show_progress: bool,
    pub pooling_strategy: PoolingStrategy,
}

impl Default for SentenceTransformerConfig {
    fn default() -> Self {
        Self {
            model: SentenceTransformerModel::AllMiniLML6V2,
            cache_dir: None,
            device: Device::Auto,
            normalize_embeddings: true,
            batch_size: 32,
            show_progress: false,
            pooling_strategy: PoolingStrategy::Mean,
        }
    }
}

/// Device selection for model inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Mps,
    Auto,
}

/// Pooling strategy for token embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingStrategy {
    Mean,
    Max,
    MeanSqrt,
    WeightedMean,
    Cls,
}

/// ONNX Runtime backend for efficient inference
pub struct OnnxBackend {
    session: Arc<RwLock<Option<ort::Session>>>,
    tokenizer: Arc<RwLock<Option<tokenizers::Tokenizer>>>,
    dimension: usize,
    max_length: usize,
}

impl OnnxBackend {
    pub async fn new(model_path: PathBuf, tokenizer_path: PathBuf) -> Result<Self, EmbeddingError> {
        // Load ONNX model
        let session = ort::Session::builder()
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to set optimization: {}", e)))?
            .with_model_from_file(&model_path)
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load model: {}", e)))?;
            
        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load tokenizer: {}", e)))?;
            
        // Get model dimensions from output shape
        let outputs = session.outputs();
        let dimension = if let Some(output) = outputs.first() {
            output.dimensions()
                .last()
                .copied()
                .and_then(|d| d.try_into().ok())
                .unwrap_or(768)
        } else {
            768
        };
        
        Ok(Self {
            session: Arc::new(RwLock::new(Some(session))),
            tokenizer: Arc::new(RwLock::new(Some(tokenizer))),
            dimension,
            max_length: 512,
        })
    }
    
    pub async fn encode(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let session_guard = self.session.read().await;
        let session = session_guard.as_ref()
            .ok_or_else(|| EmbeddingError::ModelLoadError("Session not initialized".to_string()))?;
            
        let tokenizer_guard = self.tokenizer.read().await;
        let tokenizer = tokenizer_guard.as_ref()
            .ok_or_else(|| EmbeddingError::ModelLoadError("Tokenizer not initialized".to_string()))?;
            
        let mut all_embeddings = Vec::new();
        
        for text in texts {
            // Tokenize
            let encoding = tokenizer.encode(text, true)
                .map_err(|e| EmbeddingError::InferenceError(format!("Tokenization failed: {}", e)))?;
                
            let input_ids = encoding.get_ids().to_vec();
            let attention_mask = encoding.get_attention_mask().to_vec();
            
            // Prepare inputs
            let input_ids_array = ndarray::Array2::from_shape_vec(
                (1, input_ids.len()),
                input_ids.iter().map(|&id| id as i64).collect()
            ).map_err(|e| EmbeddingError::InferenceError(format!("Array creation failed: {}", e)))?;
            
            let attention_mask_array = ndarray::Array2::from_shape_vec(
                (1, attention_mask.len()),
                attention_mask.iter().map(|&m| m as i64).collect()
            ).map_err(|e| EmbeddingError::InferenceError(format!("Array creation failed: {}", e)))?;
            
            // Run inference
            let inputs = ort::inputs![
                "input_ids" => input_ids_array.view(),
                "attention_mask" => attention_mask_array.view()
            ].map_err(|e| EmbeddingError::InferenceError(format!("Input preparation failed: {}", e)))?;
            
            let outputs = session.run(inputs)
                .map_err(|e| EmbeddingError::InferenceError(format!("Inference failed: {}", e)))?;
                
            // Extract embeddings (pooling)
            let output = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EmbeddingError::InferenceError(format!("Output extraction failed: {}", e)))?;
                
            let output_view = output.view();
            let shape = output_view.shape();
            
            // Mean pooling
            let mut embedding = vec![0.0f32; shape[2]];
            for i in 0..shape[1] {
                for j in 0..shape[2] {
                    embedding[j] += output_view[[0, i, j]];
                }
            }
            
            // Average
            for value in &mut embedding {
                *value /= shape[1] as f32;
            }
            
            // Normalize if needed
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for value in &mut embedding {
                    *value /= norm;
                }
            }
            
            all_embeddings.push(embedding);
        }
        
        Ok(all_embeddings)
    }
}

/// Sentence Transformer embedding provider
pub struct SentenceTransformerEmbedding {
    config: SentenceTransformerConfig,
    backend: Arc<RwLock<Option<OnnxBackend>>>,
    cache: Arc<RwLock<lru::LruCache<String, Vec<f32>>>>,
    dimension: usize,
}

impl SentenceTransformerEmbedding {
    pub async fn new(config: SentenceTransformerConfig) -> Result<Self, EmbeddingError> {
        let dimension = config.model.dimension();
        
        Ok(Self {
            config,
            backend: Arc::new(RwLock::new(None)),
            cache: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(1000).unwrap()
            ))),
            dimension,
        })
    }
    
    pub async fn load_model(&self) -> Result<(), EmbeddingError> {
        // In production, this would download and cache the model
        // For now, we'll simulate loading
        let model_name = self.config.model.model_name();
        
        // Check if model is cached
        let cache_dir = self.config.cache_dir.clone()
            .unwrap_or_else(|| PathBuf::from(".cache/models"));
            
        let model_dir = cache_dir.join(model_name.replace('/', "_"));
        
        if !model_dir.exists() {
            // Download model (simulated)
            return Err(EmbeddingError::ModelLoadError(
                format!("Model {} not found. Please download it first.", model_name)
            ));
        }
        
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");
        
        let backend = OnnxBackend::new(model_path, tokenizer_path).await?;
        
        let mut backend_guard = self.backend.write().await;
        *backend_guard = Some(backend);
        
        Ok(())
    }
    
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
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
        
        // Get backend
        let backend_guard = self.backend.read().await;
        let backend = backend_guard.as_ref()
            .ok_or_else(|| EmbeddingError::ModelLoadError("Model not loaded".to_string()))?;
            
        // Encode uncached texts
        let embeddings = backend.encode(uncached_texts.clone()).await?;
        
        // Update cache
        for (text, embedding) in uncached_texts.iter().zip(embeddings.iter()) {
            cache.put(text.clone(), embedding.clone());
        }
        
        // Merge results
        let mut final_results = Vec::with_capacity(texts.len());
        let mut uncached_iter = embeddings.into_iter();
        
        for cached in cached_results {
            if let Some(embedding) = cached {
                final_results.push(embedding);
            } else {
                let embedding = uncached_iter.next()
                    .ok_or_else(|| EmbeddingError::InferenceError("Missing embedding".to_string()))?;
                final_results.push(embedding);
            }
        }
        
        Ok(final_results)
    }
    
    pub async fn embed_documents(&self, documents: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut all_embeddings = Vec::new();
        
        // Process in batches
        for chunk in documents.chunks(self.config.batch_size) {
            let batch_embeddings = self.embed_batch(chunk.to_vec()).await?;
            all_embeddings.extend(batch_embeddings);
        }
        
        Ok(all_embeddings)
    }
    
    pub async fn embed_query(&self, query: String) -> Result<Vec<f32>, EmbeddingError> {
        let embeddings = self.embed_batch(vec![query]).await?;
        embeddings.into_iter().next()
            .ok_or_else(|| EmbeddingError::InferenceError("No embedding returned".to_string()))
    }
}

#[async_trait]
impl EmbeddingFunction for SentenceTransformerEmbedding {
    async fn embed(&self, texts: Vec<String>) -> EmbeddingResult<Vec<DenseVector>> {
        let embeddings = self.embed_documents(texts).await?;
        Ok(embeddings)
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn name(&self) -> String {
        format!("sentence-transformers-{}", 
            match &self.config.model {
                SentenceTransformerModel::Custom(name) => name.clone(),
                model => model.model_name(),
            }
        )
    }
}

/// Utility to download and cache models
pub struct ModelDownloader {
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new(cache_dir: Option<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.unwrap_or_else(|| PathBuf::from(".cache/models")),
        }
    }
    
    pub async fn download_model(&self, model_name: &str) -> Result<PathBuf, EmbeddingError> {
        let model_dir = self.cache_dir.join(model_name.replace('/', "_"));
        
        if model_dir.exists() {
            return Ok(model_dir);
        }
        
        // Create directory
        std::fs::create_dir_all(&model_dir)
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to create directory: {}", e)))?;
            
        // In production, download from HuggingFace Hub
        // For now, return error indicating manual download needed
        Err(EmbeddingError::ModelLoadError(
            format!("Please manually download model {} to {:?}", model_name, model_dir)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_dimensions() {
        assert_eq!(SentenceTransformerModel::AllMiniLML6V2.dimension(), 384);
        assert_eq!(SentenceTransformerModel::AllMpnetBaseV2.dimension(), 768);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = SentenceTransformerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SentenceTransformerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.batch_size, deserialized.batch_size);
    }
}