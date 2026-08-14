// IVF (Inverted File) 索引实现
// 支持大规模向量数据的高效索引和搜索

use crate::search::brute_force::calculate_distance;
use crate::search::index_traits::{VectorIndex, SerializableIndex, IndexStats};
use async_trait::async_trait;

use crate::types::{vector::{RecordId, DenseVector}, DistanceMetric};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// IVF索引配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfConfig {
    /// 聚类中心数量
    pub nlist: usize,
    /// 搜索时探测的聚类数量
    pub nprobe: usize,
    /// K-means最大迭代次数
    pub max_iterations: usize,
    /// 收敛阈值
    pub tolerance: f32,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            nlist: 100,         // 100个聚类中心
            nprobe: 10,         // 搜索10个聚类
            max_iterations: 20, // 最大20次迭代
            tolerance: 1e-4,    // 收敛阈值
        }
    }
}

/// 聚类中心信息
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Centroid {
    /// 中心向量
    vector: DenseVector,
    /// 分配到该聚类的向量列表
    vectors: Vec<(RecordId, DenseVector)>,
}

/// IVF索引结构
#[derive(Debug, Serialize, Deserialize)]
pub struct IvfIndex {
    /// 配置参数
    config: IvfConfig,
    /// 距离度量
    distance_metric: DistanceMetric,
    /// 聚类中心
    centroids: Vec<Centroid>,
    /// 向量维度
    dimension: Option<usize>,
    /// 是否已训练
    is_trained: bool,
}

impl IvfIndex {
    /// 创建新的IVF索引
    pub fn new(distancemetric: DistanceMetric, config: Option<IvfConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
            distance_metric: distancemetric,
            centroids: Vec::new(),
            dimension: None,
            is_trained: false,
        }
    }

    /// 训练聚类中心（K-means算法）
    pub fn train(&mut self, training_vectors: &[DenseVector]) -> crate::Result<()> {
        if training_vectors.is_empty() {
            return Err(crate::Error::InvalidInput("Training vectors cannot be empty".to_string() ));
        }

        let dimension = training_vectors[0].len();
        self.dimension = Some(dimension);

        // 验证所有向量维度一致
        for vector in training_vectors {
            if vector.len() != dimension {
                return Err(crate::Error::DimensionMismatch {
                    expected: dimension,
                    actual: vector.len(),
                })
            }
        }

        // 初始化聚类中心（K-means++算法）
        let mut centroids = self.initialize_centroids_plus_plus(training_vectors)?;

        // K-means迭代
        for iteration in 0..self.config.max_iterations {
            let old_centroids = centroids.clone();

            // 分配阶段：将每个向量分配到最近的聚类中心
            let assignments = self.assign_vectors_to_centroids(training_vectors, &centroids)?;

            // 更新阶段：重新计算聚类中心
            centroids = self.update_centroids(&assignments, dimension)?;

            // 检查收敛
            let convergence = self.calculate_convergence(&old_centroids, &centroids)?;
            if convergence < self.config.tolerance {
                println!("IVF training converged after {} iterations", iteration + 1);
                break;
            }
        }

        // 保存聚类中心
        self.centroids = centroids
            .into_iter()
            .map(|c| Centroid {
                vector: c,
                vectors: Vec::new(),
            })
            .collect();

        self.is_trained = true;
        Ok(())
    }

    /// K-means++初始化算法
    fn initialize_centroids_plus_plus(&self, vectors: &[DenseVector]) -> crate::Result<Vec<DenseVector>> {
        let mut centroids = Vec::new();
        let mut rng = rand::thread_rng();

        // 随机选择第一个中心点
        use rand::Rng;
        let first_idx = rng.gen_range(0..vectors.len());
        centroids.push(vectors[first_idx].clone());

        // 依次选择剩余的中心点
        for _ in 1..self.config.nlist {
            let distances: Vec<f32> = vectors
                .par_iter()
                .map(|v| {
                    // 计算到最近聚类中心的距离
                    centroids
                        .iter()
                        .map(|c| {
                            calculate_distance(&self.distance_metric, v, c)
                        })
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();

            // 按距离平方进行加权随机选择
            let weights: Vec<f32> = distances.iter().map(|d| d * d).collect();
            let total_weight: f32 = weights.iter().sum();

            if total_weight == 0.0 {
                break; // 避免无限循环
            }

            let mut cumulative = 0.0;
            let random_value = rng.gen::<f32>() * total_weight;

            for (i, &weight) in weights.iter().enumerate() {
                cumulative += weight;
                if cumulative >= random_value {

                    centroids.push(vectors[i].clone());
                    break;

                }
            }
        }

        Ok(centroids)
    }

    /// 将向量分配到最近的聚类中心
    fn assign_vectors_to_centroids(
        &self,
        vectors: &[DenseVector],
        centroids: &[DenseVector]) -> crate::Result<Vec<Vec<DenseVector>>> {
        let mut assignments = vec![Vec::new(); centroids.len()];

        for vector in vectors {
            let mut best_centroid_idx = 0;
            let mut best_distance = f32::INFINITY;

            for (i, centroid) in centroids.iter().enumerate() {
                let distance = calculate_distance(&self.distance_metric, vector, centroid);
                if distance < best_distance {
                    best_distance = distance;
                    best_centroid_idx = i
                }
            }

            assignments[best_centroid_idx].push(vector.clone());
        }

        Ok(assignments)
    }

    /// 更新聚类中心
    fn update_centroids(
        &self,
        assignments: &[Vec<DenseVector>],
        dimension: usize) -> crate::Result<Vec<DenseVector>> {
        let mut new_centroids = Vec::new();

        for cluster_vectors in assignments {
            if cluster_vectors.is_empty() {

                // 如果聚类为空，保持原来的中心点
                new_centroids.push(vec![0.0; dimension]);
                continue;

            }
            // 计算平均值作为新的聚类中心
            let mut centroid = vec![0.0; dimension];
            for vector in cluster_vectors {
                for (i, &value) in vector.iter().enumerate() {
                    centroid[i] += value
                }
            }

            let count = cluster_vectors.len() as f32;
            for value in centroid.iter_mut() {
                *value /= count
            }

            new_centroids.push(centroid);
        }

        Ok(new_centroids)
    }

    /// 计算收敛性
    fn calculate_convergence(
        &self,
        old_centroids: &[DenseVector],
        new_centroids: &[DenseVector]) -> crate::Result<f32> {
        let mut total_movement = 0.0;

        for (old, new) in old_centroids.iter().zip(new_centroids.iter()) {
            let distance = calculate_distance(&DistanceMetric::Euclidean, old, new);
            total_movement += distance
        }

        Ok(total_movement / old_centroids.len() as f32)
    }

    /// 找到最近的聚类中心
    fn find_nearest_centroids(&self, query: &DenseVector, nprobe: usize) -> crate::Result<Vec<usize>> {
        let mut distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let distance = calculate_distance(&self.distance_metric, &centroid.vector, query);
                (i, distance)
            })
            .collect();

        // 按距离排序
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // 返回前nprobe个聚类的索引
        Ok(distances
            .into_iter()
            .take(nprobe)
            .map(|(idx, _)| idx)
            .collect())
    }
}

impl SerializableIndex for IvfIndex {
    fn save(&self, path: &std::path::Path) -> crate::Result<()> {
        crate::search::index_persistence::write_index(path, "ivf", self)
    }

    fn load(path: &std::path::Path) -> crate::Result<Self> {
        crate::search::index_persistence::read_index(path, "ivf")
    }

    fn stats(&self) -> IndexStats {
        let total_vectors: usize = self.centroids.iter().map(|c| c.vectors.len()).sum();
        let dimension = self.dimension.unwrap_or(0);

        // IVF memory: vectors in centroids + centroid vectors
        let vectors_memory = total_vectors * dimension * std::mem::size_of::<f32>();
        let centroids_memory = self.centroids.len() * dimension * std::mem::size_of::<f32>();

        IndexStats {
            total_vectors: self.len(),
            index_size_bytes: vectors_memory + centroids_memory,
            build_time_ms: 0, // Not tracked in current implementation
        }
    }
}

#[async_trait]
impl VectorIndex for IvfIndex {
    async fn add(&mut self, id: String, vector: Vec<f32>) -> crate::Result<()> {
        if !self.is_trained {
            return Err(crate::Error::InvalidInput("IVF index must be trained before insertion".to_string() ));
        }

        // 验证向量维度
        if let Some(expected_dim) = self.dimension {
            if vector.len() != expected_dim {
                return Err(crate::Error::DimensionMismatch {
                    expected: expected_dim,
                    actual: vector.len(),
                })
            }
        }

        // 找到最近的聚类中心
        let nearest_centroids = self.find_nearest_centroids(&vector, 1)?;
        if let Some(&nearest_idx) = nearest_centroids.first() {
            self.centroids[nearest_idx].vectors.push((id, vector))
        }

        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> crate::Result<Vec<(String, f32)>> {
        if !self.is_trained {
            return Err(crate::Error::InvalidInput("IVF index must be trained before searching".to_string() ));
        }

        // 找到最近的nprobe个聚类中心
        let query_vec = query.to_vec();
        let nearest_centroids = self.find_nearest_centroids(&query_vec, self.config.nprobe)?;

        // 在选中的聚类中搜索
        let mut all_candidates = Vec::new();

        for centroid_idx in nearest_centroids {
            if centroid_idx < self.centroids.len() {
                for (id, vector) in &self.centroids[centroid_idx].vectors {
                    let distance = calculate_distance(&self.distance_metric, vector, &query_vec);
                    all_candidates.push((id.clone(), distance))
                }
            }
        }

        // 按距离排序并返回前k个结果
        all_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_candidates.truncate(k);

        Ok(all_candidates)
    }

    async fn delete(&mut self, id: &str) -> crate::Result<()> {
        for centroid in &mut self.centroids {
            if let Some(pos) = centroid
                .vectors
                .iter()
                .position(|(record_id, _)| record_id == id)
            {
                centroid.vectors.remove(pos);
                return Ok(());
            }
        }

        Err(crate::Error::NotFound(format!("Vector with id {} not found", id)))
    }

    fn len(&self) -> usize {
        self.centroids.iter().map(|c| c.vectors.len()).sum()
    }

    fn is_empty(&self) -> bool {
        self.centroids.iter().all(|c| c.vectors.is_empty())
    }
}

#[cfg(test)]
mod test {}
