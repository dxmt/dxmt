use crate::precomputed_query::PrecomputedQuery;
use crate::search::index_traits::VectorIndex;
use crate::search::simd_distance::SimdDistance;
use crate::types::vector::{DenseVector, DistanceMetric};
use crate::types::RecordId;
use async_trait::async_trait;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

#[cfg(feature = "gpu")]
use crate::acceleration::GpuDistanceCalculator;

/// BruteForce 索引 - 使用 Vec 而非 HashMap，提升缓存局部性
/// 适用于小数据集 (<10K vectors)，线性扫描比哈希查找更快
///
/// When a GPU is available and the dataset exceeds the batch threshold,
/// distance computations are offloaded to the GPU automatically.
#[derive(Debug, Serialize, Deserialize)]
pub struct BruteForceIndex {
    /// Vec 存储：更好的缓存局部性，无哈希开销
    vectors: Vec<(RecordId, DenseVector)>,
    distance_metric: DistanceMetric,

    /// Deferred indexing: mark if index needs rebuilding (thread-safe for parallel search)
    #[serde(skip)]
    dirty: AtomicBool,

    /// Optional GPU distance calculator -- injected at index creation time.
    /// When present and the dataset is large enough, search uses GPU.
    #[serde(skip)]
    #[cfg(feature = "gpu")]
    gpu_calculator: Option<Arc<GpuDistanceCalculator>>,
}

impl Clone for BruteForceIndex {
    fn clone(&self) -> Self {
        Self {
            vectors: self.vectors.clone(),
            distance_metric: self.distance_metric,
            dirty: AtomicBool::new(self.dirty.load(AtomicOrdering::Relaxed)),
            #[cfg(feature = "gpu")]
            gpu_calculator: self.gpu_calculator.clone(),
        }
    }
}

impl BruteForceIndex {
    pub fn new(distance_metric: DistanceMetric) -> Self {
        Self {
            vectors: Vec::new(),
            distance_metric,
            dirty: AtomicBool::new(false),
            #[cfg(feature = "gpu")]
            gpu_calculator: None,
        }
    }

    /// Inject a GPU distance calculator for accelerated search.
    ///
    /// When set, `search()` will automatically use GPU for datasets
    /// exceeding the calculator's batch threshold.
    #[cfg(feature = "gpu")]
    pub fn set_gpu_calculator(&mut self, calc: Arc<GpuDistanceCalculator>) {
        self.gpu_calculator = Some(calc);
    }

    /// 插入向量 - 延迟索引构建，只标记为 dirty
    pub fn insert(&mut self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        // 检查是否已存在（线性搜索，但对小数据集很快）
        if let Some(pos) = self
            .vectors
            .iter()
            .position(|(existing_id, _)| existing_id == &id)
        {
            // 更新现有向量
            self.vectors[pos].1 = vector;
        } else {
            // 添加新向量（O(1) amortized）
            self.vectors.push((id, vector));
        }

        // Mark as dirty for deferred indexing (thread-safe atomic operation)
        self.dirty.store(true, AtomicOrdering::Relaxed);

        Ok(())
    }

    /// Build or rebuild index if dirty (called before search)
    fn ensure_index_built(&self) {
        if self.dirty.load(AtomicOrdering::Relaxed) {
            // For BruteForce, there's no actual index structure to build
            // Just mark as clean
            self.dirty.store(false, AtomicOrdering::Relaxed);
        }
    }

    /// 搜索 - Vec 的顺序遍历比 HashMap 更快（缓存友好）
    /// 对于大数据集 (>1000 vectors)，自动使用并行搜索
    pub fn search_sync(
        &self,
        query: &DenseVector,
        k: usize,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        // Ensure index is built (deferred indexing optimization)
        self.ensure_index_built();

        // Use parallel search for datasets > 1000 vectors (parallelization overhead threshold)
        if self.vectors.len() > 1000 {
            return self.search_parallel(query, k);
        }

        // Use SIMD-optimized distance calculation if available
        let use_simd = self.distance_metric.supports_simd();

        // For small k values or small datasets, use heap-based selection
        // For large k values close to dataset size, use full sort
        if k >= self.vectors.len() {
            // Return all results, sorted
            // Vec 的迭代器遍历利用了缓存预取，比 HashMap 快
            let mut results: Vec<(RecordId, f32)> = self
                .vectors
                .iter()
                .map(|(id, vector)| {
                    let distance = if use_simd {
                        self.distance_metric
                            .calculate_simd(query, vector)
                            .unwrap_or(f32::MAX)
                    } else {
                        self.distance_metric
                            .calculate(query, vector)
                            .unwrap_or(f32::MAX)
                    };
                    (id.clone(), distance)
                })
                .collect();

            results.sort_by(|a, b| {
                match a.1.partial_cmp(&b.1) {
                    Some(ord) => ord,
                    None => std::cmp::Ordering::Equal, // NaN values are treated as equal
                }
            });
            Ok(results)
        } else {
            // Use max-heap to efficiently maintain top-k results
            // The heap stores (negative distance, id) to simulate min-heap behavior
            #[derive(Debug)]
            struct HeapItem {
                distance: f32,
                id: RecordId,
            }

            impl PartialEq for HeapItem {
                fn eq(&self, other: &Self) -> bool {
                    self.distance == other.distance
                }
            }

            impl Eq for HeapItem {}

            #[allow(clippy::non_canonical_partial_ord_impl)]
            impl PartialOrd for HeapItem {
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    // Normal order for max-heap: pop the LARGEST distance (worst match)
                    // so we retain the k SMALLEST distances (nearest neighbors)
                    self.distance.partial_cmp(&other.distance)
                }
            }

            impl Ord for HeapItem {
                fn cmp(&self, other: &Self) -> Ordering {
                    self.partial_cmp(other).unwrap_or(Ordering::Equal)
                }
            }

            let mut heap = BinaryHeap::with_capacity(k + 1);

            for (id, vector) in self.vectors.iter() {
                let distance = if use_simd {
                    self.distance_metric
                        .calculate_simd(query, vector)
                        .unwrap_or(f32::MAX)
                } else {
                    self.distance_metric
                        .calculate(query, vector)
                        .unwrap_or(f32::MAX)
                };

                heap.push(HeapItem {
                    distance,
                    id: id.clone(),
                });

                // Keep only top-k elements
                if heap.len() > k {
                    heap.pop();
                }
            }

            // Convert heap to sorted vector
            let mut results: Vec<(RecordId, f32)> = heap
                .into_iter()
                .map(|item| (item.id, item.distance))
                .collect();

            results.sort_by(|a, b| {
                match a.1.partial_cmp(&b.1) {
                    Some(ord) => ord,
                    None => std::cmp::Ordering::Equal, // NaN values are treated as equal
                }
            });
            Ok(results)
        }
    }

    /// 并行搜索 - 使用 Rayon 多线程加速距离计算
    /// 适用于大数据集 (>1000 vectors)，可实现近线性加速
    fn search_parallel(
        &self,
        query: &DenseVector,
        k: usize,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        let use_simd = self.distance_metric.supports_simd();
        let metric = &self.distance_metric;

        // Parallel distance calculation using Rayon
        // Each thread processes a chunk of vectors independently
        let mut results: Vec<(RecordId, f32)> = self
            .vectors
            .par_iter() // Parallel iterator - distributes work across CPU cores
            .map(|(id, vector)| {
                let distance = if use_simd {
                    metric.calculate_simd(query, vector).unwrap_or(f32::MAX)
                } else {
                    metric.calculate(query, vector).unwrap_or(f32::MAX)
                };
                (id.clone(), distance)
            })
            .collect();

        // Efficient top-k selection using partial sort
        // Only sorts the first k elements, rest remain unsorted
        if k < results.len() {
            results
                .select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            results.truncate(k);
        }

        // Final sort of top-k results
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// GPU-accelerated search: compute all distances on GPU in one batch,
    /// then select top-k on CPU.
    #[cfg(feature = "gpu")]
    fn search_gpu(
        &self,
        query: &DenseVector,
        k: usize,
        gpu: &GpuDistanceCalculator,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        // Extract just the vectors for GPU batch computation
        let db_vectors: Vec<DenseVector> = self.vectors.iter().map(|(_, v)| v.clone()).collect();

        // Compute all distances on GPU in one batch
        let distances = gpu.single_query_distances(&self.distance_metric, query, &db_vectors)?;

        // Pair distances with IDs and find top-k
        let mut results: Vec<(RecordId, f32)> = self
            .vectors
            .iter()
            .zip(distances.iter())
            .map(|((id, _), &dist)| (id.clone(), dist))
            .collect();

        // Efficient top-k selection
        if k < results.len() {
            results
                .select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            results.truncate(k);
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// 使用预计算查询进行搜索 - 避免重复计算查询向量属性 (+10-15% 性能)
    ///
    /// 适用于需要多次搜索同一查询向量的场景
    /// - Cosine 距离: 避免重复计算查询向量范数
    /// - Euclidean 距离: 避免重复计算平方范数
    pub fn search_with_precomputed(
        &self,
        query: &PrecomputedQuery,
        k: usize,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        self.ensure_index_built();

        // Use parallel search for large datasets
        if self.vectors.len() > 1000 {
            return self.search_parallel_precomputed(query, k);
        }

        // Serial search for small datasets
        if k >= self.vectors.len() {
            let mut results: Vec<(RecordId, f32)> = self
                .vectors
                .iter()
                .map(|(id, vector)| {
                    let distance = query.distance_to(vector).unwrap_or(f32::MAX);
                    (id.clone(), distance)
                })
                .collect();

            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            Ok(results)
        } else {
            // Heap-based selection for small k
            #[derive(Debug)]
            struct HeapItem {
                distance: f32,
                id: RecordId,
            }

            impl PartialEq for HeapItem {
                fn eq(&self, other: &Self) -> bool {
                    self.distance == other.distance
                }
            }

            impl Eq for HeapItem {}

            #[allow(clippy::non_canonical_partial_ord_impl)]
            impl PartialOrd for HeapItem {
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    self.distance.partial_cmp(&other.distance)
                }
            }

            impl Ord for HeapItem {
                fn cmp(&self, other: &Self) -> Ordering {
                    self.partial_cmp(other).unwrap_or(Ordering::Equal)
                }
            }

            let mut heap = BinaryHeap::with_capacity(k + 1);

            for (id, vector) in self.vectors.iter() {
                let distance = query.distance_to(vector).unwrap_or(f32::MAX);

                heap.push(HeapItem {
                    distance,
                    id: id.clone(),
                });

                if heap.len() > k {
                    heap.pop();
                }
            }

            let mut results: Vec<(RecordId, f32)> = heap
                .into_iter()
                .map(|item| (item.id, item.distance))
                .collect();

            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            Ok(results)
        }
    }

    /// 并行搜索（使用预计算查询）
    fn search_parallel_precomputed(
        &self,
        query: &PrecomputedQuery,
        k: usize,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        let mut results: Vec<(RecordId, f32)> = self
            .vectors
            .par_iter()
            .map(|(id, vector)| {
                let distance = query.distance_to(vector).unwrap_or(f32::MAX);
                (id.clone(), distance)
            })
            .collect();

        if k < results.len() {
            results
                .select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            results.truncate(k);
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// 删除向量 - Vec 的 swap_remove 是 O(1)
    pub fn remove(&mut self, id: &RecordId) -> crate::Result<()> {
        if let Some(pos) = self
            .vectors
            .iter()
            .position(|(existing_id, _)| existing_id == id)
        {
            self.vectors.swap_remove(pos); // O(1), 不保持顺序但更快
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.vectors.clear();
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

#[async_trait]
impl VectorIndex for BruteForceIndex {
    async fn add(&mut self, id: String, vector: Vec<f32>) -> crate::Result<()> {
        self.insert(id, vector)
    }

    async fn search(&self, query: &[f32], k: usize) -> crate::Result<Vec<(String, f32)>> {
        // GPU-accelerated path: if GPU calculator is present and workload is large enough
        #[cfg(feature = "gpu")]
        {
            if let Some(ref gpu) = self.gpu_calculator {
                let dim = query.len();
                if gpu.should_use_gpu(self.vectors.len(), dim) {
                    return self.search_gpu(&query.to_vec(), k, gpu);
                }
            }
        }

        // Fallback: CPU + SIMD + Rayon
        self.search_sync(&query.to_vec(), k)
    }

    async fn delete(&mut self, id: &str) -> crate::Result<()> {
        self.remove(&id.to_string())
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

pub fn calculate_distance(metric: &DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    metric.calculate(a, b).unwrap_or(f32::INFINITY)
}
