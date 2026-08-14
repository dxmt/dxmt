//! Automatic embedding system with intelligent provider selection

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

use crate::embeddings::compat::{EmbeddingFunction, EmbeddingError};
use crate::embeddings::openai::{OpenAIEmbedding, OpenAIConfig, OpenAIModel};
use crate::embeddings::cohere::{CohereEmbedding, CohereConfig, CohereModel};
// Commented out until sentence_transformers module is available
// use crate::embeddings::sentence_transformers::{
//     SentenceTransformerEmbedding, SentenceTransformerConfig, SentenceTransformerModel
// };
use crate::types::DenseVector;

/// Auto-embedding strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingStrategy {
    /// Use the fastest available provider
    Fastest,
    /// Use the most cost-effective provider
    CostOptimized,
    /// Use the highest quality provider
    QualityFirst,
    /// Use local models when possible, fallback to API
    LocalFirst,
    /// Use a specific provider
    Specific(String),
    /// Custom strategy with priorities
    Custom {
        priorities: Vec<String>,
        fallback: bool,
    },
}

/// Provider capabilities
#[derive(Debug, Clone)]
struct ProviderCapabilities {
    name: String,
    dimension: usize,
    max_batch_size: usize,
    cost_per_1k_tokens: f32,
    latency_ms: u32,
    quality_score: f32,
    is_local: bool,
    requires_api_key: bool,
}

/// Auto-embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoEmbeddingConfig {
    pub strategy: EmbeddingStrategy,
    pub dimension: Option<usize>,
    pub cache_embeddings: bool,
    pub max_cache_size: usize,
    pub enable_fallback: bool,
    pub providers: HashMap<String, ProviderConfig>,
}

impl Default for AutoEmbeddingConfig {
    fn default() -> Self {
        let mut providers = HashMap::new();
        
        // Default provider configurations
        providers.insert("openai".to_string(), ProviderConfig::OpenAI {
            api_key: None,
            model: OpenAIModel::TextEmbedding3Small,
        });
        
        providers.insert("cohere".to_string(), ProviderConfig::Cohere {
            api_key: None,
            model: CohereModel::EmbedEnglishV3,
        });
        
        // Commented out until sentence_transformers module is available
        // providers.insert("sentence-transformers".to_string(), ProviderConfig::SentenceTransformers {
        //     model: SentenceTransformerModel::AllMiniLML6V2,
        // });
        
        Self {
            strategy: EmbeddingStrategy::LocalFirst,
            dimension: None,
            cache_embeddings: true,
            max_cache_size: 10000,
            enable_fallback: true,
            providers,
        }
    }
}

/// Provider-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderConfig {
    OpenAI {
        api_key: Option<String>,
        model: OpenAIModel,
    },
    Cohere {
        api_key: Option<String>,
        model: CohereModel,
    },
    // Commented out until sentence_transformers module is available
    // SentenceTransformers {
    //     model: SentenceTransformerModel,
    // },
    HuggingFace {
        api_key: Option<String>,
        model: String,
    },
}

/// Embedding provider trait
#[async_trait]
trait EmbeddingProvider: Send + Sync {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    fn capabilities(&self) -> ProviderCapabilities;
    fn is_available(&self) -> bool;
}

/// OpenAI provider wrapper
struct OpenAIProvider {
    embedding: OpenAIEmbedding,
    capabilities: ProviderCapabilities,
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.embedding.embed_documents(texts).await
    }
    
    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities.clone()
    }
    
    fn is_available(&self) -> bool {
        // Assume available if the struct was created
        true
    }
}

/// Cohere provider wrapper
struct CohereProvider {
    embedding: CohereEmbedding,
    capabilities: ProviderCapabilities,
}

#[async_trait]
impl EmbeddingProvider for CohereProvider {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.embedding.embed_documents(texts).await
    }
    
    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities.clone()
    }
    
    fn is_available(&self) -> bool {
        // Assume available if the struct was created
        true
    }
}

// Commented out until sentence_transformers module is available
// /// Sentence Transformers provider wrapper
// struct SentenceTransformersProvider {
//     embedding: SentenceTransformerEmbedding,
//     capabilities: ProviderCapabilities,
// }
// 
// #[async_trait]
// impl EmbeddingProvider for SentenceTransformersProvider {
//     async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
//         self.embedding.embed_documents(texts).await
//     }
//     
//     fn capabilities(&self) -> ProviderCapabilities {
//         self.capabilities.clone()
//     }
//     
//     fn is_available(&self) -> bool {
//         true // Local models are always available
//     }
// }

/// Automatic embedding system
pub struct AutoEmbedding {
    config: AutoEmbeddingConfig,
    providers: Arc<RwLock<HashMap<String, Arc<dyn EmbeddingProvider>>>>,
    cache: Arc<RwLock<lru::LruCache<String, Vec<f32>>>>,
    metrics: Arc<RwLock<EmbeddingMetrics>>,
}

/// Embedding metrics for monitoring
#[derive(Debug, Default)]
pub(crate) struct EmbeddingMetrics {
    total_requests: u64,
    cache_hits: u64,
    cache_misses: u64,
    provider_usage: HashMap<String, u64>,
    total_tokens: u64,
    total_cost: f32,
}

impl AutoEmbedding {
    pub async fn new(config: AutoEmbeddingConfig) -> Result<Self, EmbeddingError> {
        let mut providers: HashMap<String, Arc<dyn EmbeddingProvider>> = HashMap::new();
        
        // Initialize providers based on configuration
        for (name, provider_config) in &config.providers {
            let provider: Arc<dyn EmbeddingProvider> = match provider_config {
                ProviderConfig::OpenAI { api_key, model } => {
                    let resolved_key = api_key.clone()
                        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                        .unwrap_or_default();
                    let openai_config = OpenAIConfig {
                        model: model.clone(),
                        api_key: resolved_key,
                        ..Default::default()
                    };
                    let embedding = OpenAIEmbedding::new(openai_config);
                    let capabilities = ProviderCapabilities {
                        name: name.clone(),
                        dimension: model.dimension(),
                        max_batch_size: 100,
                        cost_per_1k_tokens: 0.00013, // $0.13 per 1M tokens
                        latency_ms: 500,
                        quality_score: 0.95,
                        is_local: false,
                        requires_api_key: true,
                    };
                    
                    Arc::new(OpenAIProvider { embedding, capabilities })
                },
                
                ProviderConfig::Cohere { api_key, model } => {
                    let resolved_key = api_key.clone()
                        .or_else(|| std::env::var("COHERE_API_KEY").ok())
                        .unwrap_or_default();
                    let cohere_config = CohereConfig {
                        model: model.clone(),
                        api_key: resolved_key,
                        ..Default::default()
                    };
                    let embedding = CohereEmbedding::new(cohere_config);
                    let capabilities = ProviderCapabilities {
                        name: name.clone(),
                        dimension: model.dimension(),
                        max_batch_size: 96,
                        cost_per_1k_tokens: 0.00010, // $0.10 per 1M tokens
                        latency_ms: 400,
                        quality_score: 0.92,
                        is_local: false,
                        requires_api_key: true,
                    };
                    
                    Arc::new(CohereProvider { embedding, capabilities })
                },
                
                // Commented out until sentence_transformers module is available
                // ProviderConfig::SentenceTransformers { model } => {
                //     let mut st_config = SentenceTransformerConfig::default();
                //     st_config.model = model.clone();
                //     
                //     let embedding = SentenceTransformerEmbedding::new(st_config).await?;
                //     let capabilities = ProviderCapabilities {
                //         name: name.clone(),
                //         dimension: model.dimension(),
                //         max_batch_size: 32,
                //         cost_per_1k_tokens: 0.0, // Free (local)
                //         latency_ms: 50,
                //         quality_score: 0.85,
                //         is_local: true,
                //         requires_api_key: false,
                //     };
                //     
                //     Arc::new(SentenceTransformersProvider { embedding, capabilities })
                // },
                
                _ => continue, // Skip unsupported providers
            };
            
            providers.insert(name.clone(), provider);
        }
        
        let cache_size = std::num::NonZeroUsize::new(config.max_cache_size)
            .unwrap_or(std::num::NonZeroUsize::new(10000).unwrap());
            
        Ok(Self {
            config,
            providers: Arc::new(RwLock::new(providers)),
            cache: Arc::new(RwLock::new(lru::LruCache::new(cache_size))),
            metrics: Arc::new(RwLock::new(EmbeddingMetrics::default())),
        })
    }
    
    async fn select_provider(&self) -> Result<Arc<dyn EmbeddingProvider>, EmbeddingError> {
        let providers = self.providers.read().await;
        
        let available_providers: Vec<_> = providers.values()
            .filter(|p| p.is_available())
            .collect();
            
        if available_providers.is_empty() {
            return Err(EmbeddingError::ModelNotFound("No available providers".to_string()));
        }
        
        let selected = match &self.config.strategy {
            EmbeddingStrategy::Fastest => {
                available_providers.into_iter()
                    .min_by_key(|p| p.capabilities().latency_ms)
                    .unwrap()
            },
            
            EmbeddingStrategy::CostOptimized => {
                available_providers.into_iter()
                    .min_by(|a, b| {
                        a.capabilities().cost_per_1k_tokens
                            .partial_cmp(&b.capabilities().cost_per_1k_tokens)
                            .unwrap()
                    })
                    .unwrap()
            },
            
            EmbeddingStrategy::QualityFirst => {
                available_providers.into_iter()
                    .max_by(|a, b| {
                        a.capabilities().quality_score
                            .partial_cmp(&b.capabilities().quality_score)
                            .unwrap()
                    })
                    .unwrap()
            },
            
            EmbeddingStrategy::LocalFirst => {
                available_providers.iter()
                    .find(|p| p.capabilities().is_local)
                    .or_else(|| available_providers.first())
                    .unwrap()
            },
            
            EmbeddingStrategy::Specific(name) => {
                providers.get(name)
                    .filter(|p| p.is_available())
                    .ok_or_else(|| EmbeddingError::ModelNotFound(
                        format!("Provider {} not available", name)
                    ))?
            },
            
            EmbeddingStrategy::Custom { priorities, .. } => {
                priorities.iter()
                    .find_map(|name| providers.get(name).filter(|p| p.is_available()))
                    .or_else(|| available_providers.first().copied())
                    .ok_or_else(|| EmbeddingError::ModelNotFound(
                        "No providers match custom strategy".to_string()
                    ))?
            },
        };
        
        Ok(Arc::clone(selected))
    }
    
    pub async fn embed_with_fallback(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // Try primary provider
        match self.embed_batch_internal(texts.clone()).await {
            Ok(embeddings) => Ok(embeddings),
            Err(e) if self.config.enable_fallback => {
                // Try fallback providers
                let providers = self.providers.read().await;
                for provider in providers.values() {
                    if provider.is_available() {
                        if let Ok(embeddings) = provider.embed_batch(texts.clone()).await {
                            return Ok(embeddings);
                        }
                    }
                }
                Err(e)
            },
            Err(e) => Err(e),
        }
    }
    
    async fn embed_batch_internal(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += texts.len() as u64;
        
        // Check cache
        let mut cache = self.cache.write().await;
        let mut cached_results = vec![None; texts.len()];
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();
        
        for (i, text) in texts.iter().enumerate() {
            if let Some(embedding) = cache.get(text) {
                cached_results[i] = Some(embedding.clone());
                metrics.cache_hits += 1;
            } else {
                uncached_texts.push(text.clone());
                uncached_indices.push(i);
                metrics.cache_misses += 1;
            }
        }
        
        if uncached_texts.is_empty() {
            return Ok(cached_results.into_iter().map(|r| r.unwrap()).collect());
        }
        
        // Select provider
        let provider = self.select_provider().await?;
        let provider_name = provider.capabilities().name.clone();
        
        // Update metrics
        *metrics.provider_usage.entry(provider_name.clone()).or_insert(0) += uncached_texts.len() as u64;
        
        // Estimate tokens (rough approximation)
        let estimated_tokens: usize = uncached_texts.iter()
            .map(|t| t.split_whitespace().count() * 4 / 3)
            .sum();
        metrics.total_tokens += estimated_tokens as u64;
        
        // Calculate cost
        let cost = (estimated_tokens as f32 / 1000.0) * provider.capabilities().cost_per_1k_tokens;
        metrics.total_cost += cost;
        
        // Get embeddings
        let embeddings = provider.embed_batch(uncached_texts.clone()).await?;
        
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
                final_results.push(uncached_iter.next().unwrap());
            }
        }
        
        Ok(final_results)
    }
    
    pub(crate) async fn get_metrics(&self) -> EmbeddingMetrics {
        self.metrics.read().await.clone()
    }
    
    pub async fn clear_cache(&self) {
        self.cache.write().await.clear();
    }
}

#[async_trait]
impl EmbeddingFunction for AutoEmbedding {
    fn name(&self) -> &str {
        "auto-embedding"
    }
    
    fn modality(&self) -> &str {
        "text"
    }
    
    fn dimension(&self) -> usize {
        self.config.dimension.unwrap_or(768)
    }
    
    async fn embed(&self, input: &str) -> Result<DenseVector, EmbeddingError> {
        let results = self.embed_with_fallback(vec![input.to_string()]).await?;
        results.into_iter().next()
            .ok_or_else(|| EmbeddingError::InternalError("No embedding returned".to_string()))
    }
    
    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<DenseVector>, EmbeddingError> {
        self.embed_with_fallback(inputs.to_vec()).await
    }
}

impl Clone for EmbeddingMetrics {
    fn clone(&self) -> Self {
        Self {
            total_requests: self.total_requests,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            provider_usage: self.provider_usage.clone(),
            total_tokens: self.total_tokens,
            total_cost: self.total_cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_auto_embedding_creation() {
        let config = AutoEmbeddingConfig::default();
        let auto_embedding = AutoEmbedding::new(config).await;
        assert!(auto_embedding.is_ok());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = AutoEmbeddingConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AutoEmbeddingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.max_cache_size, deserialized.max_cache_size);
    }
}