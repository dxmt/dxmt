// 乘积量化实现

use super::{
    CompressionRatio, ProductQuantizationConfig, QuantizationMetadata, QuantizationParams,
    QuantizationStats, QuantizedData, QuantizedIndex, QuantizedVector,
};
use crate::types::{vector::DenseVector, DistanceMetric};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// 乘积量化器
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ProductQuantizer {
    config: ProductQuantizationConfig,
    distance_metric: DistanceMetric,
    centroids: Vec<Vec<f32>>,
    vector_division: Vec<(usize, usize)>,
    num_sub_vectors: usize,
    num_bits: u8,
    is_trained: bool,
}

/// 乘积量化索引
pub struct ProductQuantizedIndex {
    quantizer: ProductQuantizer,
    vectors: HashMap<String, QuantizedVector>,
    stats: QuantizationStats,
    query_latency_us: Arc<AtomicU64>,
}

/// 乘积量化查询
#[derive(Debug, Clone)]
pub struct ProductQuery {
    pub lookup_table: Vec<f32>,
    pub num_sub_vectors: usize,
}

impl ProductQuantizer {
    /// 创建新的乘积量化器
    pub fn new(config: ProductQuantizationConfig, distance_metric: DistanceMetric) -> Self {
        let num_sub_vectors = config.num_sub_vectors.unwrap_or(16);
        let num_bits = config.num_bits.unwrap_or(8);

        Self {
            config,
            distance_metric,
            centroids: Vec::new(),
            vector_division: Vec::new(),
            num_sub_vectors,
            num_bits,
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

        // 计算子向量数量（优先使用配置中显式指定的值）
        self.num_sub_vectors = self
            .config
            .num_sub_vectors
            .unwrap_or_else(|| self.calculate_optimal_sub_vectors(dimension));

        // 分割向量
        self.vector_division = self.divide_vector(dimension);

        // 训练每个子空间的聚类中心
        self.centroids = self.train_centroids(training_vectors)?;

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

        let mut encoded_data = Vec::with_capacity(self.num_sub_vectors);

        // 对每个子向量进行量化
        for (sub_vector_idx, &(start, end)) in self.vector_division.iter().enumerate() {
            let sub_vector = &vector[start..end];
            let centroid_idx = self.find_nearest_centroid(sub_vector, sub_vector_idx)?;
            encoded_data.push(centroid_idx as u8);
        }

        let metadata = QuantizationMetadata {
            original_dim: vector.len(),
            quantized_dim: encoded_data.len(),
            distance_metric: self.distance_metric,
            params: QuantizationParams::Product {
                centroids: self.centroids.clone(),
                vector_division: self.vector_division.clone(),
                num_sub_vectors: self.num_sub_vectors,
                num_bits: self.num_bits,
            },
        };

        Ok(QuantizedVector {
            id: String::new(),
            data: QuantizedData::Product(encoded_data),
            payload: metadata,
        })
    }

    /// 编码查询向量
    pub fn encode_query(&self, query: &DenseVector) -> crate::Result<ProductQuery> {
        if !self.is_trained {
            return Err(crate::Error::InvalidInput(
                "Quantizer must be trained before encoding".to_string(),
            ));
        }

        let mut lookup_table = Vec::new();

        // 为每个子向量计算查找表
        for (sub_vector_idx, &(start, end)) in self.vector_division.iter().enumerate() {
            let sub_query = &query[start..end];
            let centroids_start = sub_vector_idx * (1 << self.num_bits);

            for centroid_idx in 0..(1 << self.num_bits) {
                let centroid = &self.centroids[centroids_start + centroid_idx];
                let distance = self.calculate_sub_distance(sub_query, centroid);
                lookup_table.push(distance);
            }
        }

        Ok(ProductQuery {
            lookup_table,
            num_sub_vectors: self.num_sub_vectors,
        })
    }

    /// 计算两个量化向量之间的距离
    pub fn distance(&self, query: &ProductQuery, vector: &QuantizedVector) -> crate::Result<f32> {
        if let QuantizedData::Product(ref encoded_vector) = vector.data {
            let mut total_distance = 0.0;

            for (sub_vector_idx, &centroid_idx) in encoded_vector.iter().enumerate() {
                let lookup_idx = sub_vector_idx * (1 << self.num_bits) + centroid_idx as usize;
                if lookup_idx < query.lookup_table.len() {
                    total_distance += query.lookup_table[lookup_idx]
                }
            }

            Ok(total_distance)
        } else {
            Err(crate::Error::InvalidInput(
                "Vector is not product quantized".to_string(),
            ))
        }
    }

    // 私有辅助方法

    fn calculate_optimal_sub_vectors(&self, dimension: usize) -> usize {
        match self.config.compression_ratio {
            CompressionRatio::X4 => dimension / 4,
            CompressionRatio::X8 => dimension / 8,
            CompressionRatio::X16 => dimension / 16,
            CompressionRatio::X32 => dimension / 32,
            CompressionRatio::X64 => dimension / 64,
        }
        .max(1)
    }

    fn divide_vector(&self, dimension: usize) -> Vec<(usize, usize)> {
        let sub_vector_size = dimension / self.num_sub_vectors;
        let remainder = dimension % self.num_sub_vectors;

        let mut divisions = Vec::new();
        let mut current_pos = 0;

        for i in 0..self.num_sub_vectors {
            let size = sub_vector_size + if i < remainder { 1 } else { 0 };
            divisions.push((current_pos, current_pos + size));
            current_pos += size;
        }

        divisions
    }

    fn train_centroids(&self, training_vectors: &[DenseVector]) -> crate::Result<Vec<Vec<f32>>> {
        let mut all_centroids = Vec::new();

        // 为每个子空间训练聚类中心
        for &(start, end) in self.vector_division.iter() {
            let sub_vectors: Vec<Vec<f32>> = training_vectors
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            let centroids = self.kmeans_clustering(&sub_vectors, 1 << self.num_bits)?;
            all_centroids.extend(centroids);
        }

        Ok(all_centroids)
    }

    fn kmeans_clustering(&self, vectors: &[Vec<f32>], k: usize) -> crate::Result<Vec<Vec<f32>>> {
        if vectors.is_empty() || k == 0 {
            return Err(crate::Error::InvalidInput(
                "Invalid clustering parameters".to_string(),
            ));
        }

        let dimension = vectors[0].len();
        let mut centroids = self.initialize_centroids(vectors, k)?;
        let max_iterations = 100;

        for _ in 0..max_iterations {
            let mut assignments = vec![0; vectors.len()];
            let mut cluster_sizes = vec![0; k];
            let mut cluster_sums = vec![vec![0.0; dimension]; k];

            // 分配阶段
            for (vector_idx, vector) in vectors.iter().enumerate() {
                let mut min_distance = f32::INFINITY;
                let mut best_centroid = 0;

                for (centroid_idx, centroid) in centroids.iter().enumerate() {
                    let distance = self.calculate_sub_distance(vector, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        best_centroid = centroid_idx;
                    }
                }

                assignments[vector_idx] = best_centroid;
                cluster_sizes[best_centroid] += 1;

                for (dim, &value) in vector.iter().enumerate() {
                    cluster_sums[best_centroid][dim] += value;
                }
            }

            // 更新阶段
            let mut converged = true;
            for (centroid_idx, centroid) in centroids.iter_mut().enumerate() {
                if cluster_sizes[centroid_idx] > 0 {
                    let new_centroid: Vec<f32> = cluster_sums[centroid_idx]
                        .iter()
                        .map(|&sum| sum / cluster_sizes[centroid_idx] as f32)
                        .collect();

                    // 检查是否收敛
                    let distance = self.calculate_sub_distance(centroid, &new_centroid);
                    if distance > 1e-6 {
                        converged = false;
                    }

                    *centroid = new_centroid;
                }
            }

            if converged {
                break;
            }
        }

        Ok(centroids)
    }

    fn initialize_centroids(&self, vectors: &[Vec<f32>], k: usize) -> crate::Result<Vec<Vec<f32>>> {
        if vectors.len() < k {
            return Err(crate::Error::InvalidInput(
                "Not enough vectors for clustering".to_string(),
            ));
        }

        let mut centroids = Vec::new();
        let mut rng = rand::thread_rng();

        // K-means++ 初始化
        use rand::Rng;
        let first_idx = rng.gen_range(0..vectors.len());
        centroids.push(vectors[first_idx].clone());

        for _ in 1..k {
            let mut distances = Vec::new();
            let mut total_distance = 0.0;

            for vector in vectors {
                let min_distance = centroids
                    .iter()
                    .map(|centroid| self.calculate_sub_distance(vector, centroid))
                    .fold(f32::INFINITY, f32::min);

                distances.push(min_distance);
                total_distance += min_distance
            }

            // 按距离加权随机选择
            let random_value = rng.gen::<f32>() * total_distance;
            let mut cumulative = 0.0;

            for (idx, &distance) in distances.iter().enumerate() {
                cumulative += distance;
                if cumulative >= random_value {
                    centroids.push(vectors[idx].clone());
                    break;
                }
            }
        }

        Ok(centroids)
    }

    fn find_nearest_centroid(
        &self,
        sub_vector: &[f32],
        sub_vector_idx: usize,
    ) -> crate::Result<usize> {
        let centroids_start = sub_vector_idx * (1 << self.num_bits);
        let centroids_end = centroids_start + (1 << self.num_bits);

        let mut min_distance = f32::INFINITY;
        let mut best_centroid = 0;

        for (centroid_idx, centroid) in self.centroids[centroids_start..centroids_end]
            .iter()
            .enumerate()
        {
            let distance = self.calculate_sub_distance(sub_vector, centroid);
            if distance < min_distance {
                min_distance = distance;
                best_centroid = centroids_start + centroid_idx
            }
        }

        Ok(best_centroid)
    }

    fn calculate_sub_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.distance_metric {
            DistanceMetric::Cosine | DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::Dot => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
            DistanceMetric::Manhattan => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .sum::<f32>(),
        }
    }
}

impl ProductQuantizedIndex {
    /// 创建新的乘积量化索引
    pub fn new(config: ProductQuantizationConfig, distance_metric: DistanceMetric) -> Self {
        let quantizer = ProductQuantizer::new(config, distance_metric);

        Self {
            quantizer,
            vectors: HashMap::new(),
            stats: QuantizationStats {
                total_vectors: 0,
                memory_usagebytes: 0,
                compression_ratio: 8.0,
                quantization_type: "product".to_string(),
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

impl QuantizedIndex for ProductQuantizedIndex {
    fn insert(&mut self, vector: QuantizedVector) -> crate::Result<()> {
        if let QuantizedData::Product(_) = vector.data {
            let memory_usage = std::mem::size_of_val(&vector);
            self.stats.memory_usagebytes += memory_usage;
            self.stats.total_vectors += 1;

            self.vectors.insert(vector.id.clone(), vector);
            Ok(())
        } else {
            Err(crate::Error::InvalidInput(
                "Vector must be product quantized".to_string(),
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
            ProductQuantizer,
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
    fn test_product_quantization() -> Result<(), Box<dyn std::error::Error>> {
        let config = ProductQuantizationConfig {
            compression_ratio: CompressionRatio::X4,
            num_sub_vectors: Some(2),
            num_bits: Some(2), // 4 centroids per subspace, only needs 4 training vectors
            always_ram: None,
        };
        let mut quantizer = ProductQuantizer::new(config, DistanceMetric::Euclidean);

        // Need at least 2^num_bits = 4 training vectors for k-means
        let training_vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![2.0, 4.0, 6.0, 8.0],
        ];

        quantizer.train(&training_vectors)?;

        let test_vector = vec![2.0, 3.0, 4.0, 5.0];
        let quantized = quantizer.quantize(&test_vector)?;

        if let QuantizedData::Product(ref data) = quantized.data {
            assert_eq!(data.len(), 2); // 4维分成2个子向量
        } else {
            panic!("Expected product quantized data");
        }
        Ok(())
    }
}
