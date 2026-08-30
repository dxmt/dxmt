// 量化模块 - 支持多种向量量化技术

#[cfg(feature = "experimental-indexes")]
pub mod binary;
#[cfg(feature = "experimental-indexes")]
pub mod enhanced_pq;
#[cfg(feature = "experimental-indexes")]
pub mod hybrid;
#[cfg(feature = "experimental-indexes")]
pub mod product;
pub mod product_quantization;
#[cfg(feature = "experimental-indexes")]
pub mod rabitq; // New: RaBitQ binary quantization
#[cfg(feature = "experimental-indexes")]
pub mod scalar;
#[cfg(feature = "experimental-indexes")]
pub mod storage;
#[cfg(feature = "experimental-indexes")]
pub mod utils;

use crate::types::{vector::DenseVector, DistanceMetric};
use serde::{Deserialize, Serialize};

/// 量化类型枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationType {
    /// 标量量化 (Scalar Quantization)
    Scalar(ScalarQuantizationConfig),
    /// 乘积量化 (Product Quantization)
    Product(ProductQuantizationConfig),
    /// 二进制量化 (Binary Quantization)
    Binary(BinaryQuantizationConfig),
    /// 混合量化 (Hybrid Quantization)
    Hybrid(HybridQuantizationConfig),
}

/// 标量量化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizationConfig {
    /// 量化位数 (通常为8位)
    pub bits: u8,
    /// 分位数范围 [0.5, 1.0]，用于控制量化范围
    pub quantile: Option<f32>,
    /// 是否始终存储在内存中
    pub always_ram: Option<bool>,
}

impl Default for ScalarQuantizationConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            quantile: None,
            always_ram: None,
        }
    }
}

/// 乘积量化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizationConfig {
    /// 压缩比例
    pub compression_ratio: CompressionRatio,
    /// 子向量数量
    pub num_sub_vectors: Option<usize>,
    /// 每个子向量的位数
    pub num_bits: Option<u8>,
    /// 是否始终存储在内存中
    pub always_ram: Option<bool>,
}

impl Default for ProductQuantizationConfig {
    fn default() -> Self {
        Self {
            compression_ratio: CompressionRatio::X8,
            num_sub_vectors: None,
            num_bits: Some(8),
            always_ram: None,
        }
    }
}

/// 二进制量化配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BinaryQuantizationConfig {
    /// 是否始终存储在内存中
    pub always_ram: Option<bool>,
}

/// 混合量化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridQuantizationConfig {
    /// 基础量化类型
    pub base_quantization: Box<QuantizationType>,
    /// 索引类型
    pub index_type: HybridIndexType,
}

/// 混合索引类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HybridIndexType {
    /// IVF + 量化
    IvfFlat,
    /// IVF + HNSW + 量化
    IvfHnsw,
    /// HNSW + 量化
    Hnsw,
}

/// 压缩比例
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompressionRatio {
    X4,  // 4倍压缩
    X8,  // 8倍压缩
    X16, // 16倍压缩
    X32, // 32倍压缩
    X64, // 64倍压缩
}

impl CompressionRatio {
    pub fn ratio(&self) -> usize {
        match self {
            CompressionRatio::X4 => 4,
            CompressionRatio::X8 => 8,
            CompressionRatio::X16 => 16,
            CompressionRatio::X32 => 32,
            CompressionRatio::X64 => 64,
        }
    }
}

/// 量化向量
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantizedVector {
    /// 原始向量ID
    pub id: String,
    /// 量化后的数据
    pub data: QuantizedData,
    /// 量化元数据
    pub payload: QuantizationMetadata,
}

/// 量化数据
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum QuantizedData {
    /// 标量量化数据 (u8)
    Scalar(Vec<u8>),
    /// 乘积量化数据
    Product(Vec<u8>),
    /// 二进制量化数据
    Binary(Vec<u8>),
}

/// 量化元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    /// 原始维度
    pub original_dim: usize,
    /// 量化后维度
    pub quantized_dim: usize,
    /// 距离度量
    pub distance_metric: DistanceMetric,
    /// 量化参数
    pub params: QuantizationParams,
}

/// 量化参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationParams {
    /// 标量量化参数
    Scalar {
        alpha: f32,
        offset: f32,
        multiplier: f32,
    },
    /// 乘积量化参数
    Product {
        centroids: Vec<Vec<f32>>,
        vector_division: Vec<(usize, usize)>,
        num_sub_vectors: usize,
        num_bits: u8,
    },
    /// 二进制量化参数
    Binary { threshold: f32 },
}

/// 量化索引接口
pub trait QuantizedIndex {
    /// 插入量化向量
    fn insert(&mut self, vector: QuantizedVector) -> crate::Result<()>;

    /// 搜索最近邻
    fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(String, f32)>>;

    /// 获取索引统计信息
    fn get_stats(&self) -> QuantizationStats;

    /// 序列化索引
    fn serialize(&self) -> crate::Result<Vec<u8>>;

    /// 反序列化索引
    fn deserialize(&mut self, data: &[u8]) -> crate::Result<()>;
}

/// 量化统计信息
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantizationStats {
    /// 向量总数
    pub total_vectors: usize,
    /// 内存使用量 (字节)
    pub memory_usagebytes: usize,
    /// 压缩比例
    pub compression_ratio: f32,
    /// 量化类型
    pub quantization_type: String,
    /// 平均查询延迟 (微秒)
    pub avg_query_latency_us: Option<u64>,
}

/// 量化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// 量化类型
    pub quantization_type: QuantizationType,
    /// 是否启用量化
    pub enabled: bool,
    /// 训练样本数量
    pub training_samples: Option<usize>,
    /// 最大内存使用量 (字节)
    pub max_memorybytes: Option<usize>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::Scalar(ScalarQuantizationConfig::default()),
            enabled: true,
            training_samples: None,
            max_memorybytes: None,
        }
    }
}

// 重新导出子模块
#[cfg(feature = "experimental-indexes")]
pub use binary::{BinaryQuantizedIndex, BinaryQuantizer};
#[cfg(feature = "experimental-indexes")]
pub use hybrid::{HybridQuantizedIndex, HybridQuantizer};
#[cfg(feature = "experimental-indexes")]
pub use product::{ProductQuantizedIndex, ProductQuantizer};
#[cfg(feature = "experimental-indexes")]
pub use rabitq::{Codebook, RaBitQConfig, RaBitQIndex, RaBitQQuantizer, RaBitQVector};
#[cfg(feature = "experimental-indexes")]
pub use scalar::{ScalarQuantizedIndex, ScalarQuantizer};
#[cfg(feature = "experimental-indexes")]
pub use storage::{QuantizedStorage, QuantizedStorageBuilder, QuantizedStorageConfig};
#[cfg(feature = "experimental-indexes")]
pub use utils::{QuantizationError, QuantizationUtils};
