// 标量量化实现

use super::{
    QuantizationMetadata, QuantizationParams, QuantizationStats, QuantizedData, QuantizedIndex,
    QuantizedVector, ScalarQuantizationConfig,
};
use crate::types::{vector::DenseVector, DistanceMetric};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// 标量量化器
#[derive(Serialize, Deserialize)]
pub struct ScalarQuantizer {
    config: ScalarQuantizationConfig,
    distance_metric: DistanceMetric,
    alpha: f32,
    offset: f32,
    multiplier: f32,
    is_trained: bool,
}

/// 标量量化索引
pub struct ScalarQuantizedIndex {
    quantizer: ScalarQuantizer,
    vectors: HashMap<String, QuantizedVector>,
    stats: QuantizationStats,
    query_latency_us: Arc<AtomicU64>,
}

/// 标量量化查询
#[derive(Debug, Clone)]
pub struct ScalarQuery {
    pub encoded_query: Vec<u8>,
    pub offset: f32,
}

impl ScalarQuantizer {
    /// 创建新的标量量化器
    pub fn new(config: ScalarQuantizationConfig, distance_metric: DistanceMetric) -> Self {
        Self {
            config,
            distance_metric,
            alpha: 0.0,
            offset: 0.0,
            multiplier: 0.0,
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

        // 计算量化参数
        let (alpha, offset) = self.calculate_quantization_params(training_vectors)?;

        // 根据分位数调整参数
        if let Some(quantile) = self.config.quantile {
            let (min, max) = self.find_quantile_range(training_vectors, quantile)?;
            let (alpha_adj, offset_adj) = self.alpha_offset_from_min_max(min, max);
            self.alpha = alpha_adj;
            self.offset = offset_adj;
        } else {
            self.alpha = alpha;
            self.offset = offset;
        }

        // 计算乘数
        self.multiplier = match self.distance_metric {
            DistanceMetric::Cosine | DistanceMetric::Euclidean => -2.0 * self.alpha * self.alpha,
            DistanceMetric::Dot => self.alpha * self.alpha,
            DistanceMetric::Manhattan => self.alpha,
        };

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

        let mut encoded_data = Vec::with_capacity(vector.len());

        // 量化每个维度
        for &value in vector {
            let encoded = self.f32_to_u8(value);
            encoded_data.push(encoded);
        }

        let metadata = QuantizationMetadata {
            original_dim: vector.len(),
            quantized_dim: encoded_data.len(),
            distance_metric: self.distance_metric,
            params: QuantizationParams::Scalar {
                alpha: self.alpha,
                offset: self.offset,
                multiplier: self.multiplier,
            },
        };

        Ok(QuantizedVector {
            id: String::new(), // 将在插入时设置
            data: QuantizedData::Scalar(encoded_data),
            payload: metadata,
        })
    }

    /// 编码查询向量
    pub fn encode_query(&self, query: &DenseVector) -> crate::Result<ScalarQuery> {
        if !self.is_trained {
            return Err(crate::Error::InvalidInput(
                "Quantizer must be trained before encoding".to_string(),
            ));
        }

        let mut encoded_query = Vec::with_capacity(query.len());

        for &value in query {
            let encoded = self.f32_to_u8(value);
            encoded_query.push(encoded);
        }

        let offset = self.calculate_query_offset(query);

        Ok(ScalarQuery {
            encoded_query,
            offset,
        })
    }

    /// 计算两个量化向量之间的距离
    pub fn distance(&self, query: &ScalarQuery, vector: &QuantizedVector) -> crate::Result<f32> {
        if let QuantizedData::Scalar(ref encoded_vector) = vector.data {
            let score = self.score_vectors(&query.encoded_query, encoded_vector);
            Ok(self.multiplier * score + query.offset)
        } else {
            Err(crate::Error::InvalidInput(
                "Vector is not scalar quantized".to_string(),
            ))
        }
    }

    // 私有辅助方法

    fn calculate_quantization_params(&self, vectors: &[DenseVector]) -> crate::Result<(f32, f32)> {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        // 找到全局最小值和最大值
        for vector in vectors {
            for &value in vector {
                if value.is_finite() {
                    min_val = min_val.min(value);
                    max_val = max_val.max(value);
                }
            }
        }

        if min_val == f32::INFINITY || max_val == f32::NEG_INFINITY {
            return Err(crate::Error::InvalidInput(
                "No valid values found in training vectors".to_string(),
            ));
        }

        let range = max_val - min_val;
        let alpha = (1 << self.config.bits) as f32 / range;
        let offset = min_val;

        Ok((alpha, offset))
    }

    fn find_quantile_range(
        &self,
        vectors: &[DenseVector],
        quantile: f32,
    ) -> crate::Result<(f32, f32)> {
        if !(0.5..=1.0).contains(&quantile) {
            return Err(crate::Error::InvalidInput(
                "Quantile must be between 0.5 and 1.0".to_string(),
            ));
        }

        // 收集所有值
        let mut all_values: Vec<f32> = vectors
            .iter()
            .flat_map(|v| v.iter().copied())
            .filter(|&v| v.is_finite())
            .collect();

        if all_values.is_empty() {
            return Err(crate::Error::InvalidInput(
                "No valid values found".to_string(),
            ));
        }

        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx = ((1.0 - quantile) * all_values.len() as f32 / 2.0) as usize;
        let upper_idx = all_values.len() - lower_idx;

        let min_val = all_values[lower_idx];
        let max_val = all_values[upper_idx - 1];

        Ok((min_val, max_val))
    }

    fn alpha_offset_from_min_max(&self, min: f32, max: f32) -> (f32, f32) {
        let range = max - min;
        let alpha = (1 << self.config.bits) as f32 / range;
        (alpha, min)
    }

    fn f32_to_u8(&self, value: f32) -> u8 {
        let normalized = (value - self.offset) * self.alpha;
        let clamped = normalized.clamp(0.0, 255.0);
        clamped as u8
    }

    fn calculate_query_offset(&self, query: &DenseVector) -> f32 {
        match self.distance_metric {
            DistanceMetric::Cosine | DistanceMetric::Euclidean => {
                let norm_squared: f32 = query.iter().map(|x| x * x).sum();
                -norm_squared * self.alpha * self.alpha
            }
            DistanceMetric::Dot => 0.0,
            DistanceMetric::Manhattan => 0.0,
        }
    }

    fn score_vectors(&self, query: &[u8], vector: &[u8]) -> f32 {
        if query.len() != vector.len() {
            return 0.0;
        }

        match self.distance_metric {
            DistanceMetric::Cosine | DistanceMetric::Euclidean | DistanceMetric::Dot => {
                // 点积计算
                query
                    .iter()
                    .zip(vector.iter())
                    .map(|(&q, &v)| (q as f32) * (v as f32))
                    .sum()
            }
            DistanceMetric::Manhattan => {
                // 曼哈顿距离
                query
                    .iter()
                    .zip(vector.iter())
                    .map(|(&q, &v)| (q as f32 - v as f32).abs())
                    .sum()
            }
        }
    }
}

impl ScalarQuantizedIndex {
    /// 创建新的标量量化索引
    pub fn new(config: ScalarQuantizationConfig, distance_metric: DistanceMetric) -> Self {
        let quantizer = ScalarQuantizer::new(config, distance_metric);

        Self {
            quantizer,
            vectors: HashMap::new(),
            stats: QuantizationStats {
                total_vectors: 0,
                memory_usagebytes: 0,
                compression_ratio: 4.0,
                quantization_type: "scalar".to_string(),
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

impl QuantizedIndex for ScalarQuantizedIndex {
    fn insert(&mut self, vector: QuantizedVector) -> crate::Result<()> {
        // 确保向量是标量量化的
        if let QuantizedData::Scalar(_) = vector.data {
            // 计算内存使用量
            let memory_usage = std::mem::size_of_val(&vector);
            self.stats.memory_usagebytes += memory_usage;
            self.stats.total_vectors += 1;

            // 插入向量
            self.vectors.insert(vector.id.clone(), vector);
            Ok(())
        } else {
            Err(crate::Error::InvalidInput(
                "Vector must be scalar quantized".to_string(),
            ))
        }
    }

    fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(String, f32)>> {
        let start_time = std::time::Instant::now();

        let encoded_query = self.quantizer.encode_query(query)?;
        let mut candidates = Vec::new();

        // 计算所有向量的距离
        for (id, vector) in &self.vectors {
            let distance = self.quantizer.distance(&encoded_query, vector)?;
            candidates.push((id.clone(), distance))
        }

        // 排序并取前k个
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
            ScalarQuantizer,
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
    fn test_scalar_quantization() -> Result<(), Box<dyn std::error::Error>> {
        let config = ScalarQuantizationConfig::default();
        let mut quantizer = ScalarQuantizer::new(config, DistanceMetric::Euclidean);

        // 训练数据
        let training_vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        quantizer.train(&training_vectors)?;

        // 测试量化
        let test_vector = vec![2.0, 3.0, 4.0];
        let quantized = quantizer.quantize(&test_vector)?;

        // Check the quantized data length
        match quantized.data {
            crate::search::quantization::QuantizedData::Scalar(ref data) => {
                assert_eq!(data.len(), 3);
            }
            _ => panic!("Expected Scalar quantized data"),
        }
        Ok(())
    }

    #[test]
    fn test_scalar_index_search() -> Result<(), Box<dyn std::error::Error>> {
        let config = ScalarQuantizationConfig::default();
        let mut index = ScalarQuantizedIndex::new(config, DistanceMetric::Euclidean);

        // 训练
        let training_vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        index.train(&training_vectors)?;

        // 插入向量
        for (i, vector) in training_vectors.iter().enumerate() {
            let mut quantized = index.quantizer.quantize(vector)?;
            quantized.id = format!("vec_{}", i);
            index.insert(quantized)?;
        }

        // 搜索
        let query = vec![1.0, 0.1, 0.0];
        let results = index.search(&query, 2)?;

        assert!(!results.is_empty());
        assert_eq!(results[0].0, "vec_0"); // 应该找到最接近的向量
        Ok(())
    }
}
