// 二进制量化实现
// 参考 Qdrant 的 Binary Quantization

use super::{
    BinaryQuantizationConfig, QuantizationMetadata, QuantizationParams, QuantizationStats,
    QuantizedIndex,
};
use crate::search::quantization::{QuantizedData, QuantizedVector};
use crate::types::{vector::DenseVector, DistanceMetric};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// 二进制量化器
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BinaryQuantizer {
    config: BinaryQuantizationConfig,
    distance_metric: DistanceMetric,
    threshold: f32,
    is_trained: bool,
}

/// 二进制量化索引
pub struct BinaryQuantizedIndex {
    quantizer: BinaryQuantizer,
    vectors: HashMap<String, QuantizedVector>,
    stats: QuantizationStats,
    query_latency_us: Arc<AtomicU64>,
}

/// 二进制量化查询
#[derive(Debug, Clone)]
pub struct BinaryQuery {
    pub encoded_query: Vec<u8>,
}

impl BinaryQuantizer {
    /// 创建新的二进制量化器
    pub fn new(config: BinaryQuantizationConfig, distance_metric: DistanceMetric) -> Self {
        Self {
            config,
            distance_metric,
            threshold: 0.0,
            is_trained: false,
        }
    }

    /// 训练量化器
    pub fn train(&mut self, training_vectors: &[DenseVector]) -> crate::Result<()> {
        if training_vectors.is_empty() {
            return Err(crate::Error::InvalidInput(
                "Training vectors cannot be empty".to_string(),
            ));
        }

        let dimension = training_vectors[0].len();

        // 验证所有向量维度一致
        for vector in training_vectors {
            if vector.len() != dimension {
                return Err(crate::Error::DimensionMismatch {
                    expected: dimension,
                    actual: vector.len(),
                });
            }
        }

        // 计算阈值（所有向量的平均值）
        let total_sum: f32 = training_vectors.iter().flat_map(|v| v.iter()).sum();

        let total_elements = training_vectors.len() * dimension;
        self.threshold = total_sum / total_elements as f32;

        self.is_trained = true;
        Ok(())
    }

    /// 量化向量
    pub fn quantize(&self, vector: &DenseVector) -> crate::Result<QuantizedVector> {
        if !self.is_trained {
            return Err(crate::Error::InvalidInput(
                "Quantizer must be trained before quantization".to_string(),
            ));
        }

        let mut encoded_data = Vec::new();

        // 将向量转换为二进制表示
        for &value in vector {
            let bit = if value >= self.threshold { 1 } else { 0 };
            encoded_data.push(bit);
        }

        // 将位打包成字节
        let packed_data = self.pack_bits_to_bytes(&encoded_data);

        let metadata = QuantizationMetadata {
            original_dim: vector.len(),
            quantized_dim: packed_data.len(),
            distance_metric: self.distance_metric,
            params: QuantizationParams::Binary {
                threshold: self.threshold,
            },
        };

        Ok(QuantizedVector {
            id: String::new(),
            data: QuantizedData::Binary(packed_data),
            payload: metadata,
        })
    }

    /// 编码查询向量
    pub fn encode_query(&self, query: &DenseVector) -> crate::Result<BinaryQuery> {
        if !self.is_trained {
            return Err(crate::Error::InvalidInput(
                "Quantizer must be trained before encoding".to_string(),
            ));
        }

        let mut encoded_bits = Vec::new();

        for &value in query {
            let bit = if value >= self.threshold { 1 } else { 0 };
            encoded_bits.push(bit);
        }

        let packed_data = self.pack_bits_to_bytes(&encoded_bits);

        Ok(BinaryQuery {
            encoded_query: packed_data,
        })
    }

    /// 计算两个量化向量之间的距离
    pub fn distance(&self, query: &BinaryQuery, vector: &QuantizedVector) -> crate::Result<f32> {
        if let QuantizedData::Binary(ref encoded_vector) = vector.data {
            let mut total_distance = 0.0;

            for (i, &bit) in query.encoded_query.iter().enumerate() {
                let actual_bit = encoded_vector[i];
                let distance = if bit == actual_bit { 0.0 } else { 1.0 };
                total_distance += distance;
            }

            Ok(total_distance)
        } else {
            Err(crate::Error::InvalidInput(
                "Vector is not binary quantized".to_string(),
            ))
        }
    }

    // 私有辅助方法

    fn pack_bits_to_bytes(&self, bits: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut byte = 0;
        let mut bit_count = 0;

        for &bit in bits {
            byte |= bit << bit_count;
            bit_count += 1;

            if bit_count == 8 {
                bytes.push(byte);
                byte = 0;
                bit_count = 0
            }
        }

        if bit_count > 0 {
            bytes.push(byte);
        }
        bytes
    }
}

impl BinaryQuantizedIndex {
    /// 创建新的二进制量化索引
    pub fn new(config: BinaryQuantizationConfig, distance_metric: DistanceMetric) -> Self {
        let quantizer = BinaryQuantizer::new(config, distance_metric);

        Self {
            quantizer,
            vectors: HashMap::new(),
            stats: QuantizationStats {
                total_vectors: 0,
                memory_usagebytes: 0,
                compression_ratio: 1.0,
                quantization_type: "binary".to_string(),
                avg_query_latency_us: None,
            },
            query_latency_us: Arc::new(AtomicU64::new(0)),
        }
    }

    /// 训练索引
    pub fn train(&mut self, training_vectors: &[DenseVector]) -> crate::Result<()> {
        self.quantizer.train(training_vectors)
    }
}

impl QuantizedIndex for BinaryQuantizedIndex {
    fn insert(&mut self, vector: QuantizedVector) -> crate::Result<()> {
        if let QuantizedData::Binary(_) = vector.data {
            let memory_usage = std::mem::size_of_val(&vector);
            self.stats.memory_usagebytes += memory_usage;
            self.stats.total_vectors += 1;

            self.vectors.insert(vector.id.clone(), vector);
            Ok(())
        } else {
            Err(crate::Error::InvalidInput(
                "Vector must be binary quantized".to_string(),
            ))
        }
    }

    fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(String, f32)>> {
        let start_time = std::time::Instant::now();

        let encoded_query = self.quantizer.encode_query(query)?;
        let mut candidates = Vec::new();

        for (id, vector) in &self.vectors {
            let distance = self.quantizer.distance(&encoded_query, vector)?;
            candidates.push((id.clone(), distance))
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        self.query_latency_us
            .store(start_time.elapsed().as_micros() as u64, Ordering::Relaxed);

        Ok(candidates)
    }

    fn get_stats(&self) -> QuantizationStats {
        let mut s = self.stats.clone();
        s.avg_query_latency_us = Some(self.query_latency_us.load(Ordering::Relaxed));
        s
    }

    fn serialize(&self) -> crate::Result<Vec<u8>> {
        bincode::serialize(&(&self.quantizer, &self.vectors, &self.stats))
            .map_err(|e| crate::Error::Serialization(e.to_string()))
    }

    fn deserialize(&mut self, data: &[u8]) -> crate::Result<()> {
        let (quantizer, vectors, stats): (
            BinaryQuantizer,
            HashMap<String, QuantizedVector>,
            QuantizationStats,
        ) = bincode::deserialize(data).map_err(|e| crate::Error::Serialization(e.to_string()))?;

        self.quantizer = quantizer;
        self.vectors = vectors;
        self.stats = stats;
        self.query_latency_us = Arc::new(AtomicU64::new(0));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_quantization() -> Result<(), Box<dyn std::error::Error>> {
        let config = BinaryQuantizationConfig::default();
        let mut quantizer = BinaryQuantizer::new(config, DistanceMetric::Euclidean);

        let training_vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];

        quantizer.train(&training_vectors)?;

        let test_vector = vec![2.0, 3.0, 4.0, 5.0];
        let quantized = quantizer.quantize(&test_vector)?;

        // Check the quantized data length
        match quantized.data {
            crate::search::quantization::QuantizedData::Binary(ref data) => {
                assert_eq!(data.len(), 1); // 4 bits packed into 1 byte
            }
            _ => panic!("Expected Binary quantized data"),
        }
        Ok(())
    }
}
