pub mod audio_processor;
pub mod image_processor;
pub mod manager;
pub mod processor;
pub mod clip;
pub mod whisper;
pub mod fusion;
pub mod unified;

// PyTorch backend (optional, requires 'python' feature)
#[cfg(feature = "python")]
pub mod clip_pytorch;

// Re-export core types
pub use processor::{
    MultimodalProcessor,
    TextProcessor,
    ImageProcessor,
    AudioProcessor,
};

pub use image_processor::{
    CLIPImageProcessor,
    ResNetImageProcessor,
};

pub use audio_processor::{
    WhisperAudioProcessor,
    MusicEmbeddingProcessor,
};

pub use manager::{
    MultimodalManager,
    MultimodalConfig,
};

// Re-export new implementations
pub use clip::{CLIPEmbedding, CLIPModel, CLIPConfig};
pub use whisper::{WhisperProcessor, WhisperModel, WhisperConfig};
pub use fusion::{FusionStrategy, ModalityWeight, FusionEngine};
pub use unified::{UnifiedMultimodalAPI, MultimodalQuery, MultimodalResult};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Core multimodal types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Code,
}

impl Modality {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Code => "code",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalContent {
    pub modality: Modality,
    pub data: Vec<u8>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MultimodalContent {
    pub fn text(text: String) -> Self {
        Self {
            modality: Modality::Text,
            data: text.into_bytes(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn image(data: Vec<u8>) -> Self {
        Self {
            modality: Modality::Image,
            data,
            metadata: HashMap::new(),
        }
    }
    
    pub fn audio(data: Vec<u8>) -> Self {
        Self {
            modality: Modality::Audio,
            data,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

#[async_trait]
pub trait MultimodalEmbedding: Send + Sync {
    async fn embed(&self, content: MultimodalContent) -> Result<Vec<f32>, crate::embeddings::EmbeddingError>;
    fn supported_modalities(&self) -> Vec<Modality>;
    fn dimension(&self) -> usize;
    fn name(&self) -> String;
}