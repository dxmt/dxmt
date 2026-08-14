use ndarray::Array2;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use parking_lot::RwLock;
use rand::prelude::*;
use crate::types::vector::{RecordId, DenseVector, DistanceMetric};
use std::sync::atomic::{AtomicUsize, Ordering};

/// IVF-PQ配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPqConfig {
    /// IVF聚类中心数
    pub num_centroids: usize,
    /// 子量化器数量
    pub num_subquantizers: usize,
    /// 每个子量化器的比特数
    pub bits_per_subquantizer: u8,
    /// 搜索时探测的聚类中心数
    pub nprobe: usize,
    /// 是否使用预计算的距离表
    pub use_precomputed_tables: bool,
    /// 训练迭代次数
    pub train_iterations: usize,
    /// 是否使用对称距离计算
    pub symmetric_distance: bool,
    /// 是否启用并行处理
    pub parallel: bool,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            num_centroids: 256,
            num_subquantizers: 8,
            bits_per_subquantizer: 8,
            nprobe: 8,
            use_precomputed_tables: true,
            train_iterations: 25,
            symmetric_distance: false,
            parallel: true,
        }
    }
}

/// 倒排列表
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InvertedList {
    ids: Vec<RecordId>,
    codes: Vec<Vec<u8>>,  // PQ codes
}

impl InvertedList {
    fn new() -> Self {
        Self {
            ids: Vec::new(),
            codes: Vec::new(),
        }
    }
}

/// IVF-PQ索引实现
#[derive(Debug)]
pub struct IvfPqIndex {
    config: IvfPqConfig,
    dimension: usize,
    distance_metric: DistanceMetric,
    
    // IVF components
    centroids: Vec<DenseVector>,
    inverted_lists: Vec<RwLock<InvertedList>>,
    
    // PQ components
    subquantizer_dim: usize,
    codebooks: Vec<Array2<f32>>,  // M codebooks, each K x D/M
    
    // Statistics
    total_vectors: AtomicUsize,
    is_trained: bool,
}

impl IvfPqIndex {
    /// 创建新的IVF-PQ索引
    pub fn new(
        config: IvfPqConfig,
        dimension: usize,
        distance_metric: DistanceMetric) -> Self {
        assert!(
            dimension.is_multiple_of(config.num_subquantizers),
            "Dimension must be divisible by num_subquantizers"
        );
        
        let subquantizer_dim = dimension / config.num_subquantizers;
        let num_codes = 1 << config.bits_per_subquantizer;
        
        Self {
            config: config.clone(),
            dimension,
            distance_metric,
            centroids: vec![vec![0.0; dimension]; config.num_centroids],
            inverted_lists: (0..config.num_centroids)
                .map(|_| RwLock::new(InvertedList::new()))
                .collect(),
            subquantizer_dim,
            codebooks: (0..config.num_subquantizers)
                .map(|_| Array2::zeros((num_codes, subquantizer_dim)))
                .collect(),
            total_vectors: AtomicUsize::new(0),
            is_trained: false,
        }
    }
    
    /// 训练索引
    pub fn train(&mut self, vectors: &[DenseVector]) -> crate::Result<()> {
        if vectors.is_empty() {
            return Err(crate::error::Error::InvalidInput("Cannot train with empty vector set".to_string()));
        }
        
        println!("Training IVF-PQ index with {} vectors", vectors.len());
        
        // 1. Train IVF centroids using k-means
        self.train_centroids(vectors)?;
        
        // 2. Assign vectors to centroids
        let assignments = self.assign_to_centroids(vectors);
        
        // 3. Train PQ codebooks for each subquantizer
        self.train_pq_codebooks(vectors, &assignments)?;
        
        self.is_trained = true;
        Ok(())
    }
    
    /// 训练聚类中心
    fn train_centroids(&mut self, vectors: &[DenseVector]) -> crate::Result<()> {
        let mut rng = thread_rng();
        
        // Initialize centroids randomly
        for i in 0..self.config.num_centroids {
            let idx = rng.gen_range(0..vectors.len());
            self.centroids[i] = vectors[idx].clone()
        }
        
        // Lloyd's k-means algorithm
        for iteration in 0..self.config.train_iterations {
            // Assign vectors to nearest centroid
            let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); self.config.num_centroids];
            
            if self.config.parallel {

            
                let assignments: Vec<_> = vectors
                    .par_iter()
                    .enumerate()
                    .map(|(idx, vector)| (idx, self.find_nearest_centroid(vector)))
                    .collect();
                
                for (idx, centroid_idx) in assignments {
                    clusters[centroid_idx].push(idx);

            }
            } else {
                for (idx, vector) in vectors.iter().enumerate() {
                    let nearest = self.find_nearest_centroid(vector);
                    clusters[nearest].push(idx);
                }
            }
            
            // Update centroids
            for (c_idx, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let mut new_centroid = vec![0.0; self.dimension];
                    for &v_idx in cluster {
                        for i in 0..self.dimension {
                            new_centroid[i] += vectors[v_idx][i]
                        }
                    }
                    let scale = 1.0 / cluster.len() as f32;
                    for v in &mut new_centroid {
                        *v *= scale;
                    }
                    self.centroids[c_idx] = new_centroid;
                }
            }
            
            if iteration % 5 == 0 {
                println!("K-means iteration {}/{}", iteration + 1, self.config.train_iterations);
            }
        }
        
        Ok(())
    }
    
    /// 将向量分配到最近的聚类中心
    fn assign_to_centroids(&self, vectors: &[DenseVector]) -> Vec<usize> {
        if self.config.parallel {
            vectors
                .par_iter()
                .map(|v| self.find_nearest_centroid(v))
                .collect()
        } else {
            vectors
                .iter()
                .map(|v| self.find_nearest_centroid(v))
                .collect()
        }
    }
    
    /// 找到最近的聚类中心
    fn find_nearest_centroid(&self, vector: &DenseVector) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                self.distance_metric.calculate(vector, c).ok().map(|dist| (i, dist))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0))
            .0
    }
    
    /// 训练PQ码本
    fn train_pq_codebooks(
        &mut self,
        vectors: &[DenseVector],
        assignments: &[usize]) -> crate::Result<()> {
        // Group vectors by centroid assignment and compute residuals
        let mut residuals: Vec<DenseVector> = Vec::with_capacity(vectors.len());
        
        for (vector, &assignment) in vectors.iter().zip(assignments) {
            let centroid = &self.centroids[assignment];
            let residual: Vec<f32> = vector.iter()
                .zip(centroid.iter())
                .map(|(v, c)| v - c)
                .collect();
            residuals.push(residual);
        }
        
        // Train PQ for each subquantizer
        for m in 0..self.config.num_subquantizers {
            let start_dim = m * self.subquantizer_dim;
            let end_dim = start_dim + self.subquantizer_dim;
            
            // Collect all subvectors for this subquantizer
            let subvectors: Vec<Vec<f32>> = residuals
                .iter()
                .map(|v| v.as_slice()[start_dim..end_dim].to_vec())
                .collect();
            
            // Train codebook using k-means
            let codebook = self.train_single_codebook(
                &subvectors,
                1 << self.config.bits_per_subquantizer,
            )?;
            
            // Store codebook
            for (code_idx, codeword) in codebook.iter().enumerate() {
                for (dim_idx, &val) in codeword.iter().enumerate() {
                    self.codebooks[m][[code_idx, dim_idx]] = val
                }
            }
            
            println!(
                "Trained codebook for subquantizer {}/{}",
                m + 1,
                self.config.num_subquantizers
            );
        }
        
        Ok(())
    }
    
    /// 训练单个码本
    fn train_single_codebook(
        &self,
        subvectors: &[Vec<f32>],
        num_codes: usize) -> crate::Result<Vec<Vec<f32>>> {
        let mut rng = thread_rng();
        let dim = subvectors[0].len();
        
        // Initialize codewords randomly
        let mut codewords = vec![vec![0.0; dim]; num_codes];
        for codeword in &mut codewords {
            let idx = rng.gen_range(0..subvectors.len());
            *codeword = subvectors[idx].clone();
        }
        
        // Mini k-means
        for _ in 0..10 {
            // Assign subvectors to nearest codeword
            let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); num_codes];
            for (idx, subvector) in subvectors.iter().enumerate() {
                let mut min_dist = f32::INFINITY;
                let mut best_code = 0;
                
                for (code, codeword) in codewords.iter().enumerate() {
                    let dist = self.l2_distance(subvector, codeword);
                    if dist < min_dist {
                        min_dist = dist;
                        best_code = code
                    }
                }
                clusters[best_code].push(idx);
            }
            
            // Update codewords
            for (code, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let mut new_codeword = vec![0.0; dim];
                    for &idx in cluster {
                        for (i, &val) in subvectors[idx].iter().enumerate() {
                            new_codeword[i] += val
                        }
                    }
                    for val in new_codeword.iter_mut() {
                        *val /= cluster.len() as f32
                    }
                    codewords[code] = new_codeword;
                }
            }
        }
        
        Ok(codewords)
    }
    
    /// 添加向量到索引
    pub fn add(&self, vectors: &[(RecordId, DenseVector)]) -> crate::Result<()> {
        if !self.is_trained {
            return Err(crate::error::Error::InvalidInput("Index must be trained before adding vectors".to_string()));
        }
        
        for (id, vector) in vectors {
            // 1. Find nearest centroid
            let centroid_idx = self.find_nearest_centroid(vector);
            
            // 2. Compute residual
            let centroid = &self.centroids[centroid_idx];
            let residual: Vec<f32> = vector.iter()
                .zip(centroid.iter())
                .map(|(v, c)| v - c)
                .collect();
            
            // 3. Encode using PQ
            let pq_code = self.encode_pq(&residual)?;
            
            // 4. Add to inverted list
            let mut list = self.inverted_lists[centroid_idx].write();
            list.ids.push(id.clone());
            list.codes.push(pq_code);
        }
        
        self.total_vectors.fetch_add(vectors.len(), Ordering::Relaxed);
        Ok(())
    }
    
    /// PQ编码
    fn encode_pq(&self, residual: &DenseVector) -> crate::Result<Vec<u8>> {
        let mut codes = Vec::with_capacity(self.config.num_subquantizers);
        
        for m in 0..self.config.num_subquantizers {
            let start = m * self.subquantizer_dim;
            let end = start + self.subquantizer_dim;
            let subvector = &residual.as_slice()[start..end];
            
            // Find nearest codeword
            let mut min_dist = f32::INFINITY;
            let mut best_code = 0u8;
            
            for (code, codeword_row) in self.codebooks[m].outer_iter().enumerate() {
                let codeword = codeword_row.as_slice().ok_or_else(|| crate::Error::Internal("Invalid codeword".to_string()))?;
                let dist = self.l2_distance(subvector, codeword);
                if dist < min_dist {
                    min_dist = dist;
                    best_code = code as u8
                }
            }
            
            codes.push(best_code);
        }
        
        Ok(codes)
    }
    
    /// 搜索K最近邻
    pub fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        if !self.is_trained {
            return Err(crate::error::Error::InvalidInput("Index must be trained before searching".to_string()));
        }
        
        // 1. Find nprobe nearest centroids
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .filter_map(|(i, c)| self.distance_metric.calculate(query, c).ok().map(|dist| (i, dist)))
            .collect();
        centroid_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let probe_centroids: Vec<usize> = centroid_distances
            .iter()
            .take(self.config.nprobe)
            .map(|(i, _)| *i)
            .collect();
        
        // 2. Precompute distance tables if enabled
        let distance_tables = if self.config.use_precomputed_tables {
            Some(self.precompute_distance_tables(query, &probe_centroids)?)
        } else {
            None
        };
        
        // 3. Search in selected inverted lists
        let mut candidates = Vec::new();
        
        for &centroid_idx in &probe_centroids {
            let list = self.inverted_lists[centroid_idx].read();
            let centroid = &self.centroids[centroid_idx];
            
            for (id, codes) in list.ids.iter().zip(&list.codes) {
                let distance = if let Some(ref tables) = distance_tables {
                    self.asymmetric_distance_precomputed(codes, &tables[&centroid_idx])?
                } else {
                    self.asymmetric_distance(query, codes, centroid)?
                };
                
                candidates.push((id.clone(), distance));
            }
        }
        
        // 4. Sort and return top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        
        Ok(candidates)
    }
    
    /// 预计算距离表
    fn precompute_distance_tables(
        &self,
        query: &DenseVector,
        probe_centroids: &[usize]) -> crate::Result<HashMap<usize, Vec<Vec<f32>>>> {
        let mut tables = HashMap::new();
        
        for &centroid_idx in probe_centroids {
            let centroid = &self.centroids[centroid_idx];
            let mut centroid_tables = Vec::new();
            
            for m in 0..self.config.num_subquantizers {
                let start = m * self.subquantizer_dim;
                let end = start + self.subquantizer_dim;
                
                let query_sub = &query.as_slice()[start..end];
                let centroid_sub = &centroid.as_slice()[start..end];
                
                // Compute distances to all codewords
                let mut distances = vec![0.0; 1 << self.config.bits_per_subquantizer];
                for (code, codeword_row) in self.codebooks[m].outer_iter().enumerate() {
                    let codeword = codeword_row.as_slice().ok_or_else(|| crate::Error::Internal("Invalid codeword".to_string()))?;
                    let residual_dist = self.l2_distance_residual(
                        query_sub,
                        centroid_sub,
                        codeword,
                    );
                    distances[code] = residual_dist;
                }
                
                centroid_tables.push(distances);
            }
            
            tables.insert(centroid_idx, centroid_tables);
        }
        
        Ok(tables)
    }
    
    /// 使用预计算表的非对称距离
    fn asymmetric_distance_precomputed(
        &self,
        codes: &[u8],
        distance_table: &[Vec<f32>]) -> crate::Result<f32> {
        Ok(codes
            .iter()
            .enumerate()
            .map(|(m, &code)| distance_table[m][code as usize])
            .sum())
    }
    
    /// 非对称距离计算
    fn asymmetric_distance(
        &self,
        query: &DenseVector,
        codes: &[u8],
        centroid: &DenseVector) -> crate::Result<f32> {
        let mut distance = 0.0;
        
        for (m, &code) in codes.iter().enumerate() {
            let start = m * self.subquantizer_dim;
            let end = start + self.subquantizer_dim;
            
            let query_sub = &query.as_slice()[start..end];
            let centroid_sub = &centroid.as_slice()[start..end];
            let codeword_row = self.codebooks[m].row(code as usize);
            let codeword = codeword_row
                .as_slice()
                .ok_or_else(|| crate::Error::Internal("Invalid codeword".to_string()))?;
            
            distance += self.l2_distance_residual(query_sub, centroid_sub, codeword)
        }
        
        Ok(distance)
    }
    
    /// L2距离计算
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    /// 残差L2距离
    fn l2_distance_residual(&self, query: &[f32], centroid: &[f32], codeword: &[f32]) -> f32 {
        query
            .iter()
            .zip(centroid)
            .zip(codeword)
            .map(|((q, c), w)| (q - c - w).powi(2))
            .sum::<f32>()
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> IvfPqStats {
        let vectors_per_centroid: Vec<usize> = self
            .inverted_lists
            .iter()
            .map(|list| list.read().ids.len())
            .collect();
        
        let max_list_size = vectors_per_centroid.iter().max().copied().unwrap_or(0);
        let min_list_size = vectors_per_centroid.iter().min().copied().unwrap_or(0);
        let avg_list_size = if !vectors_per_centroid.is_empty() {
            vectors_per_centroid.iter().sum::<usize>() as f32 / vectors_per_centroid.len() as f32
        } else {
            0.0
        };
        
        IvfPqStats {
            total_vectors: self.total_vectors.load(Ordering::Relaxed),
            num_centroids: self.config.num_centroids,
            num_subquantizers: self.config.num_subquantizers,
            bits_per_subquantizer: self.config.bits_per_subquantizer,
            compression_ratio: self.calculate_compression_ratio(),
            max_list_size,
            min_list_size,
            avg_list_size,
            is_trained: self.is_trained,
        }
    }
    
    /// 计算压缩比
    fn calculate_compression_ratio(&self) -> f32 {
        let original_bits = self.dimension * 32; // f32
        let compressed_bits = self.config.num_subquantizers * self.config.bits_per_subquantizer as usize;
        original_bits as f32 / compressed_bits as f32
    }
}

/// IVF-PQ统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPqStats {
    pub total_vectors: usize,
    pub num_centroids: usize,
    pub num_subquantizers: usize,
    pub bits_per_subquantizer: u8,
    pub compression_ratio: f32,
    pub max_list_size: usize,
    pub min_list_size: usize,
    pub avg_list_size: f32,
    pub is_trained: bool,
}

// Note: IvfPqIndex does not implement VectorIndex trait because it requires training
// and has different semantics (quantized vectors, no direct updates/deletes)

#[cfg(test)]
mod tests {
    use super::*;
use crate::RecordId;

    #[test]
    fn test_ivf_pq_basic() -> Result<(), Box<dyn std::error::Error>> {
        let config = IvfPqConfig {
            num_centroids: 4,
            num_subquantizers: 2,
            bits_per_subquantizer: 4,
            train_iterations: 5,
            ..Default::default()
        };

        let mut index = IvfPqIndex::new(config, 8, DistanceMetric::Euclidean);

        // Generate training data
        let mut training_data = Vec::new();
        for i in 0..100 {
            let vector = DenseVector::from(vec![i as f32; 8]);
            training_data.push(vector);
        }

        // Train the index
        index.train(&training_data)?;

        // Add vectors
        let mut vectors = Vec::new();
        for i in 0..50 {
            let id = RecordId::from(format!("vec_{}", i));
            let vector = DenseVector::from(vec![i as f32 * 2.0; 8]);
            vectors.push((id, vector));
        }
        index.add(&vectors)?;

        // Search
        let query = DenseVector::from(vec![25.0; 8]);
        let results = index.search(&query, 5)?;

        assert_eq!(results.len(), 5);

        // Check stats
        let stats = index.get_stats();
        assert_eq!(stats.total_vectors, 50);
        assert!(stats.is_trained);
        Ok(())
    }
}
