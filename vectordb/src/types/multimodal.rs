use crate::types::{DenseVector, Payload, RecordId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Error;
use crate::types::sparse_vector::SparseVector;

/// 模态类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModalityType {
    /// 文本模态
    Text,
    /// 图像模态
    Image,
    /// 音频模态
    Audio,
    /// 视频模态
    Video,
    /// 文档模态
    Document,
}

impl std::fmt::Display for ModalityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModalityType::Text => write!(f, "text"),
            ModalityType::Image => write!(f, "image"),
            ModalityType::Audio => write!(f, "audio"),
            ModalityType::Video => write!(f, "video"),
            ModalityType::Document => write!(f, "document"),
        }
    }
}

impl std::str::FromStr for ModalityType {
    type Err = Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.to_lowercase().as_str() {
            "text" => Ok(ModalityType::Text),
            "image" => Ok(ModalityType::Image),
            "audio" => Ok(ModalityType::Audio),
            "video" => Ok(ModalityType::Video),
            "document" => Ok(ModalityType::Document),
            _ => Err(crate::Error::InvalidInput(format!(
                "Invalid modality type: {}",
                s
            ))),
        }
    }
}

/// 多模态内容
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum MultimodalContent {
    /// 文本内容
    Text(String),
    /// 图像文件路径
    ImagePath(String),
    /// 图像字节数据 (base64编码)
    ImageBytes(String),
    /// 音频文件路径
    AudioPath(String),
    /// 音频字节数据 (base64编码)
    AudioBytes(String),
    /// 视频文件路径
    VideoPath(String),
    /// 视频字节数据 (base64编码)
    VideoBytes(String),
    /// 外部URI引用
    URI(String),
    /// 嵌入向量（预计算）
    Embedding(DenseVector),
}

/// 多模态数据包装器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalData {
    pub modality: ModalityType,
    pub content: MultimodalContent,
    pub payload: Option<serde_json::Value>,
}

/// 多向量记录 - 支持多模态嵌入
#[derive(Debug, Clone)]
pub struct MultiVectorRecord {
    pub id: RecordId,
    /// 多模态向量映射 (模态名 -> 向量)
    pub vectors: HashMap<String, DenseVector>,
    /// 多模态稀疏向量映射
    pub sparse_vectors: Option<HashMap<String, SparseVector>>,
    /// 原始多模态数据
    pub raw_data: Option<MultimodalData>,
    /// 元数据负载
    pub payload: Payload,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 版本号
    pub version: u64,
    /// 记录状态
    pub status: crate::types::RecordStatus,
}

/// 创建多向量记录的请求
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateMultiVectorRequest {
    pub id: RecordId,
    /// 直接提供的向量映射
    pub vectors: Option<HashMap<String, DenseVector>>,
    /// 稀疏向量映射
    pub sparse_vectors: Option<HashMap<String, SparseVector>>,
    /// 需要生成嵌入的原始数据
    pub raw_data: Option<MultimodalData>,
    /// 元数据负载
    pub payload: Option<Payload>,
}

/// 跨模态搜索查询
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossModalSearchQuery {
    /// 查询输入（可以是任意模态）
    pub query: MultimodalInput,
    /// 目标模态类型（返回结果的模态）
    pub target_modalities: Vec<ModalityType>,
    /// 融合策略
    pub fusion_strategy: FusionStrategy,
    /// 返回结果数量
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// 过滤条件
    pub filter: Option<crate::types::Filter>,
    /// 是否返回负载
    #[serde(default = "default_true")]
    pub with_payload: bool,
    /// 是否返回向量
    #[serde(default)]
    pub with_vector: bool,
}

/// 多模态输入
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum MultimodalInput {
    /// 单一文本查询
    Text(String),
    /// 多个文本查询
    Texts(Vec<String>),
    /// 图像查询
    Image(MultimodalContent),
    /// 多个图像查询
    Images(Vec<MultimodalContent>),
    /// 音频查询
    Audio(MultimodalContent),
    /// 混合模态查询
    Mixed(Vec<(ModalityType, MultimodalContent)>),
    /// 直接向量查询
    Vector {
        modality: ModalityType,
        vector: DenseVector,
    },
}

/// 融合策略
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "config")]
pub enum FusionStrategy {
    /// 早期融合：在向量级别进行加权组合
    EarlyFusion { weights: HashMap<String, f32> },
    /// 晚期融合：在结果级别进行加权组合
    LateFusion { weights: HashMap<String, f32> },
    /// 自适应融合：根据查询自动调整权重
    Adaptive {
        learning_rate: f32,
        baseline_weights: Option<HashMap<String, f32>>,
    },
    /// 注意力融合：使用注意力机制
    Attention { temperature: f32 },
}

/// 多模态搜索结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalSearchResult {
    pub hits: Vec<MultimodalSearchHit>,
    pub total: usize,
    pub tookms: u64,
    pub fusion_weights: Option<HashMap<String, f32>>,
}

/// 多模态搜索命中结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalSearchHit {
    pub id: RecordId,
    pub score: f32,
    pub modality_scores: HashMap<String, f32>,
    pub payload: Option<Payload>,
    pub vectors: Option<HashMap<String, DenseVector>>,
    pub raw_data: Option<MultimodalData>,
}

/// 嵌入函数配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    pub function_name: String,
    pub model_name: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub supported_modalities: Vec<ModalityType>,
    pub dimension: usize,
}

/// 嵌入请求
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: MultimodalInput,
    pub options: Option<HashMap<String, serde_json::Value>>,
}

/// 嵌入响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<DenseVector>,
    pub modality: ModalityType,
    pub model: String,
    pub usage: Option<serde_json::Value>,
}

/// 嵌入结果 - 用于嵌入函数返回的结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    pub embeddings: Vec<DenseVector>,
    pub model: String,
    pub usage: Option<HashMap<String, serde_json::Value>>,
}

fn default_top_k() -> usize {
    10
}

fn default_true() -> bool {
    true
}

impl MultiVectorRecord {
    /// 创建新的多向量记录
    pub fn new(request: CreateMultiVectorRequest) -> crate::Result<Self> {
        let now = chrono::Utc::now();
        let vectors = request.vectors.unwrap_or_default();

        // 验证向量维度一致性
        if !vectors.is_empty() {
            let dimensions: Vec<usize> = vectors.values().map(|v| v.len()).collect();
            let first_dim = dimensions[0];
            if !dimensions.iter().all(|&d| d == first_dim) {
                return Err(crate::Error::InvalidInput(
                    "All vectors must have the same dimension".to_string(),
                ));
            }
        }

        Ok(MultiVectorRecord {
            id: request.id,
            vectors,
            sparse_vectors: request.sparse_vectors,
            raw_data: request.raw_data,
            payload: request
                .payload
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
            created_at: now,
            updated_at: now,
            version: 1,
            status: crate::types::RecordStatus::Active,
        })
    }

    /// 添加或更新向量
    pub fn add_vector(&mut self, modality_name: String, vector: DenseVector) -> crate::Result<()> {
        // 检查维度一致性
        if !self.vectors.is_empty() {
            let existing_dim = self
                .vectors
                .values()
                .next()
                .ok_or_else(|| crate::Error::Internal("No vectors found".to_string()))?
                .len();
            if vector.len() != existing_dim {
                return Err(crate::Error::DimensionMismatch {
                    expected: existing_dim,
                    actual: vector.len(),
                });
            }
        }

        self.vectors.insert(modality_name, vector);
        self.updated_at = chrono::Utc::now();
        self.version += 1;
        Ok(())
    }

    /// 获取指定模态的向量
    pub fn get_vector(&self, modality_name: &str) -> Option<&DenseVector> {
        self.vectors.get(modality_name)
    }

    /// 获取所有支持的模态
    pub fn supported_modalities(&self) -> Vec<String> {
        self.vectors.keys().cloned().collect()
    }

    /// 检查是否支持指定模态
    pub fn supports_modality(&self, modality_name: &str) -> bool {
        self.vectors.contains_key(modality_name)
    }
}

impl Default for FusionStrategy {
    fn default() -> Self {
        FusionStrategy::EarlyFusion {
            weights: HashMap::new(),
        }
    }
}

impl Default for CrossModalSearchQuery {
    fn default() -> Self {
        Self {
            query: MultimodalInput::Text("".to_string()),
            target_modalities: vec![ModalityType::Text],
            fusion_strategy: FusionStrategy::default(),
            top_k: 10,
            filter: None,
            with_payload: true,
            with_vector: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_type_conversion() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!("text".parse::<ModalityType>()?, ModalityType::Text);
        assert_eq!("image".parse::<ModalityType>()?, ModalityType::Image);
        assert_eq!(ModalityType::Audio.to_string(), "audio");
        Ok(())
    }

    #[test]
    fn test_multi_vector_record_creation() -> Result<(), Box<dyn std::error::Error>> {
        let mut vectors = HashMap::new();
        vectors.insert("text".to_string(), vec![0.1, 0.2, 0.3]);
        vectors.insert("image".to_string(), vec![0.4, 0.5, 0.6]);

        let request = CreateMultiVectorRequest {
            id: "test_record".to_string(),
            vectors: Some(vectors),
            sparse_vectors: None,
            raw_data: None,
            payload: Some(serde_json::json!({"category": "multimodal"})),
        };

        let record = MultiVectorRecord::new(request)?;
        assert_eq!(record.id, "test_record");
        assert_eq!(record.vectors.len(), 2);
        assert!(record.supports_modality("text"));
        assert!(record.supports_modality("image"));
        Ok(())
    }

    #[test]
    fn test_dimension_mismatch_validation() {
        let mut vectors = HashMap::new();
        vectors.insert("text".to_string(), vec![0.1, 0.2]);
        vectors.insert("image".to_string(), vec![0.4, 0.5, 0.6]); // 不同维度

        let request = CreateMultiVectorRequest {
            id: "test_record".to_string(),
            vectors: Some(vectors),
            sparse_vectors: None,
            raw_data: None,
            payload: None,
        };

        assert!(MultiVectorRecord::new(request).is_err());
    }

    #[test]
    fn test_multimodal_content_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let content = MultimodalContent::Text("hello world".to_string());
        let serialized = serde_json::to_string(&content)?;
        let deserialized: MultimodalContent = serde_json::from_str(&serialized)?;

        if let MultimodalContent::Text(text) = deserialized {
            assert_eq!(text, "hello world");
        } else {
            panic!("Unexpected content type");
        }
        Ok(())
    }
}
