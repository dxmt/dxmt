//! Embeddings module with support for multiple providers

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub mod client;
pub mod cohere;
pub mod compat;
pub mod openai;
pub mod providers;
// pub mod sentence_transformers; // Temporarily disabled for compilation
pub mod auto_embedding;
pub mod sentence_transformers_simplified;

// Re-export the trait and implementations
pub use cohere::{CohereConfig, CohereEmbedding, CohereModel};
pub use compat::{EmbeddingError, EmbeddingFunction, EmbeddingResult, TextEmbeddingFunction};
pub use openai::{OpenAIConfig, OpenAIEmbedding, OpenAIModel};
pub use providers::{
    EmbeddingProvider, HuggingFaceProvider, OpenAIProvider, SentenceTransformerProvider,
};
// pub use sentence_transformers::{SentenceTransformerEmbedding, SentenceTransformerConfig, SentenceTransformerModel};
pub use auto_embedding::{AutoEmbedding, AutoEmbeddingConfig, EmbeddingStrategy};
pub use sentence_transformers_simplified::{
    SentenceTransformersConfig, SentenceTransformersEmbedding,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingFunctionMetadata {
    pub name: String,
    pub description: String,
    pub modality: String,
}

pub struct EmbeddingFunctionRegistry {
    functions: HashMap<String, EmbeddingFunctionMetadata>,
}

impl Default for EmbeddingFunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingFunctionRegistry {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    pub fn list_functions(&self) -> Vec<EmbeddingFunctionMetadata> {
        self.functions.values().cloned().collect()
    }

    pub fn get_all_metadata(&self) -> Vec<EmbeddingFunctionMetadata> {
        self.functions.values().cloned().collect()
    }
}

/// Registry for managing embedding functions
pub struct EmbeddingRegistry {
    functions: Arc<RwLock<HashMap<String, Arc<dyn EmbeddingFunction>>>>,
    providers: Arc<RwLock<HashMap<String, Arc<dyn EmbeddingProvider>>>>,
}

impl EmbeddingRegistry {
    pub fn new() -> Self {
        Self {
            functions: Arc::new(RwLock::new(HashMap::new())),
            providers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new embedding function
    pub async fn register_function(&self, name: String, function: Arc<dyn EmbeddingFunction>) {
        let mut functions = self.functions.write().await;
        functions.insert(name, function);
    }

    /// Register a new embedding provider
    pub async fn register_provider(&self, name: String, provider: Arc<dyn EmbeddingProvider>) {
        let mut providers = self.providers.write().await;
        providers.insert(name, provider);
    }

    /// Get an embedding function by name
    pub async fn get_function(&self, name: &str) -> Option<Arc<dyn EmbeddingFunction>> {
        let functions = self.functions.read().await;
        functions.get(name).cloned()
    }

    /// Get an embedding provider by name
    pub async fn get_provider(&self, name: &str) -> Option<Arc<dyn EmbeddingProvider>> {
        let providers = self.providers.read().await;
        providers.get(name).cloned()
    }

    /// List all available embedding functions
    pub async fn list_functions(&self) -> Vec<String> {
        let functions = self.functions.read().await;
        functions.keys().cloned().collect()
    }

    /// List all available providers
    pub async fn list_providers(&self) -> Vec<String> {
        let providers = self.providers.read().await;
        providers.keys().cloned().collect()
    }

    /// Create and register a function from a provider
    pub async fn create_function_from_provider(
        &self,
        function_name: String,
        provider_name: &str,
        model_name: &str,
    ) -> Result<(), EmbeddingError> {
        let provider = self.get_provider(provider_name).await.ok_or_else(|| {
            EmbeddingError::InternalError(format!("Provider {} not found", provider_name))
        })?;

        let function = provider.create_function(model_name).await?;
        self.register_function(function_name, function).await;
        Ok(())
    }
}

impl Default for EmbeddingRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Main embedding manager that integrates with the API
pub struct EmbeddingManager {
    registry: Arc<RwLock<EmbeddingFunctionRegistry>>,
    functions: Arc<RwLock<HashMap<String, Arc<dyn EmbeddingFunction>>>>,
}

impl EmbeddingManager {
    pub fn new() -> Self {
        let mut registry = EmbeddingFunctionRegistry::new();

        // Register built-in functions
        registry.functions.insert(
            "openai/text-embedding-3-small".to_string(),
            EmbeddingFunctionMetadata {
                name: "openai/text-embedding-3-small".to_string(),
                description: "OpenAI's small text embedding model".to_string(),
                modality: "text".to_string(),
            },
        );

        registry.functions.insert(
            "huggingface/all-MiniLM-L6-v2".to_string(),
            EmbeddingFunctionMetadata {
                name: "huggingface/all-MiniLM-L6-v2".to_string(),
                description: "Sentence transformer MiniLM model".to_string(),
                modality: "text".to_string(),
            },
        );

        Self {
            registry: Arc::new(RwLock::new(registry)),
            functions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn registry(&self) -> EmbeddingFunctionRegistry {
        // Return a clone for now (in real implementation, would return a reference)
        EmbeddingFunctionRegistry::new()
    }

    pub async fn load_function(&self, name: &str, config: serde_json::Value) -> crate::Result<()> {
        // Parse provider from name (e.g., "openai/text-embedding-3-small")
        let parts: Vec<&str> = name.split('/').collect();
        if parts.len() != 2 {
            return Err(crate::Error::InvalidInput(format!(
                "Invalid function name: {}",
                name
            )));
        }

        let provider_name = parts[0];
        let model_name = parts[1];

        // Create the appropriate provider and function
        let function: Arc<dyn EmbeddingFunction> = match provider_name {
            "openai" => {
                let api_key = config
                    .get("api_key")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let provider = OpenAIProvider::new(api_key);
                provider.create_function(model_name).await.map_err(|e| {
                    crate::Error::Internal(format!("Failed to create function: {:?}", e))
                })?
            }
            "huggingface" => {
                let api_key = config
                    .get("api_key")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let provider = HuggingFaceProvider::new(api_key);
                provider.create_function(model_name).await.map_err(|e| {
                    crate::Error::Internal(format!("Failed to create function: {:?}", e))
                })?
            }
            _ => {
                return Err(crate::Error::InvalidInput(format!(
                    "Unknown provider: {}",
                    provider_name
                )));
            }
        };

        // Store the function
        let mut functions = self.functions.write().await;
        functions.insert(name.to_string(), function);

        Ok(())
    }

    pub async fn unload_function(&self, name: &str) -> crate::Result<()> {
        let mut functions = self.functions.write().await;
        functions
            .remove(name)
            .ok_or_else(|| crate::Error::NotFound(format!("Function {} not found", name)))?;
        Ok(())
    }

    pub async fn list_active_functions(&self) -> Vec<String> {
        let functions = self.functions.read().await;
        functions.keys().cloned().collect()
    }

    pub async fn get_function(&self, name: &str) -> Option<Arc<dyn EmbeddingFunction>> {
        let functions = self.functions.read().await;
        functions.get(name).cloned()
    }

    pub async fn get_function_metadata(&self, name: &str) -> Option<EmbeddingFunctionMetadata> {
        let registry = self.registry.read().await;
        registry.functions.get(name).cloned()
    }
}

impl Default for EmbeddingManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for multimodal content processing
pub struct MultimodalUtils;

impl MultimodalUtils {
    pub fn validate_content(content: &[u8]) -> crate::Result<()> {
        if content.is_empty() {
            return Err(crate::Error::InvalidInput("Empty content".to_string()));
        }

        let modality = Self::infer_modality_type(content);
        match modality {
            "image" => Self::validate_image_content(content),
            "audio" => Self::validate_audio_content(content),
            "text" => Self::validate_text_content(content),
            _ => Ok(()), // Allow unknown types
        }
    }

    pub fn validate_batch(contents: &[Vec<u8>]) -> crate::Result<()> {
        if contents.is_empty() {
            return Err(crate::Error::InvalidInput("Empty batch".to_string()));
        }

        for (index, content) in contents.iter().enumerate() {
            Self::validate_content(content).map_err(|e| {
                crate::Error::InvalidInput(format!("Validation failed at index {}: {}", index, e))
            })?;
        }
        Ok(())
    }

    pub fn infer_modality_type(content: &[u8]) -> &'static str {
        if content.len() < 4 {
            return "unknown";
        }

        match &content[..4] {
            [0xFF, 0xD8, 0xFF, _] => "image",    // JPEG
            [0x89, 0x50, 0x4E, 0x47] => "image", // PNG
            [0x47, 0x49, 0x46, 0x38] => "image", // GIF
            [0x42, 0x4D, _, _] => "image",       // BMP
            [0x52, 0x49, 0x46, 0x46] => {
                // RIFF container - could be WAV, AVI, WEBP
                if content.len() >= 12 && &content[8..12] == b"WAVE" {
                    "audio" // WAV
                } else if content.len() >= 12 && &content[8..12] == b"WEBP" {
                    "image" // WEBP
                } else {
                    "audio" // Default to audio for RIFF
                }
            }
            [0x49, 0x44, 0x33, _] => "audio",    // MP3 with ID3
            [0xFF, 0xFB, _, _] => "audio",       // MP3
            [0xFF, 0xF3, _, _] => "audio",       // MP3
            [0xFF, 0xF2, _, _] => "audio",       // MP3
            [0x66, 0x4C, 0x61, 0x43] => "audio", // FLAC
            [0x4F, 0x67, 0x67, 0x53] => "audio", // OGG
            _ => {
                // Check if it's valid UTF-8 text
                if std::str::from_utf8(content).is_ok() {
                    "text"
                } else {
                    "binary"
                }
            }
        }
    }

    pub fn estimate_content_size(content: &[u8]) -> usize {
        content.len()
    }

    pub fn content_hash(content: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn validate_image_content(content: &[u8]) -> crate::Result<()> {
        if content.len() < 10 {
            return Err(crate::Error::InvalidInput(
                "Image content too small".to_string(),
            ));
        }

        match &content[..4] {
            [0xFF, 0xD8, 0xFF, _] => {
                // JPEG validation
                if !content.ends_with(&[0xFF, 0xD9]) {
                    return Err(crate::Error::InvalidInput("Invalid JPEG: missing end marker".to_string()));
                }
            },
            [0x89, 0x50, 0x4E, 0x47] => {
                // PNG validation
                if content.len() < 8 || content[4..8] != [0x0D, 0x0A, 0x1A, 0x0A] {
                    return Err(crate::Error::InvalidInput("Invalid PNG: incorrect signature".to_string()));
                }
            },
            [0x47, 0x49, 0x46, 0x38]
                // GIF validation
                if (content.len() < 6 || (content[4] != b'7' && content[4] != b'9') || content[5] != b'a') => {
                    return Err(crate::Error::InvalidInput("Invalid GIF: incorrect version".to_string()));
                },
            _ => {} // Other formats pass through
        }
        Ok(())
    }

    fn validate_audio_content(content: &[u8]) -> crate::Result<()> {
        if content.len() < 12 {
            return Err(crate::Error::InvalidInput(
                "Audio content too small".to_string(),
            ));
        }

        match &content[..4] {
            [0x52, 0x49, 0x46, 0x46] => {
                // WAV/RIFF validation
                if &content[8..12] != b"WAVE" {
                    return Err(crate::Error::InvalidInput("Invalid WAV: missing WAVE marker".to_string()));
                }
            },
            [0x49, 0x44, 0x33, _] => {
                // MP3 with ID3 tag
                if content.len() < 10 {
                    return Err(crate::Error::InvalidInput("Invalid MP3: ID3 tag too short".to_string()));
                }
            },
            [0x66, 0x4C, 0x61, 0x43]
                // FLAC validation - should have streaminfo block
                if content.len() < 8 => {
                    return Err(crate::Error::InvalidInput("Invalid FLAC: too short".to_string()));
                },
            _ => {} // Other formats pass through
        }
        Ok(())
    }

    fn validate_text_content(content: &[u8]) -> crate::Result<()> {
        match std::str::from_utf8(content) {
            Ok(_) => Ok(()),
            Err(e) => Err(crate::Error::InvalidInput(format!(
                "Invalid UTF-8 text: {}",
                e
            ))),
        }
    }
}
