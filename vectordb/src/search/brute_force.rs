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

/// Nearest first; among equal distances, the smaller id first.
///
/// The storage layer orders results the same way (`25e5515`), but it can only reorder what an
/// index hands back, so the order has to hold here too, before anything is cut to `k`.
/// Selecting the top `k` by distance alone -- a heap or `select_nth_unstable` -- kept an
/// arbitrary member of a tie at the cutoff, chosen by insertion order: after
/// `DELETE /tables/{name}/index` the same query returned `r8` from one table and `r7` from the
/// next. A NaN distance sorts after every number.
fn nearest_first(a_id: &str, a: f32, b_id: &str, b: f32) -> Ordering {
    match a.partial_cmp(&b) {
        Some(Ordering::Equal) => a_id.cmp(b_id),
        Some(ordering) => ordering,
        None => a.is_nan().cmp(&b.is_nan()).then_with(|| a_id.cmp(b_id)),
    }
}

fn by_nearest(a: &(RecordId, f32), b: &(RecordId, f32)) -> Ordering {
    nearest_first(&a.0, a.1, &b.0, b.1)
}

/// A candidate in a bounded max-heap. The top is the one to evict: the farthest, and among
/// equally far ones the largest id.
struct Candidate(RecordId, f32);

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        nearest_first(&self.0, self.1, &other.0, other.1)
    }
}

/// The `k` nearest of `results`, nearest first.
fn top_k_sorted(mut results: Vec<(RecordId, f32)>, k: usize) -> Vec<(RecordId, f32)> {
    if k < results.len() {
        results.select_nth_unstable_by(k, by_nearest);
        results.truncate(k);
    }
    results.sort_by(by_nearest);
    results
}

/// The `k` nearest of a stream of candidates, nearest first, holding at most `k + 1`.
fn top_k_streaming(
    candidates: impl Iterator<Item = (RecordId, f32)>,
    k: usize,
) -> Vec<(RecordId, f32)> {
    if k == 0 {
        return Vec::new();
    }
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (id, distance) in candidates {
        heap.push(Candidate(id, distance));
        if heap.len() > k {
            heap.pop();
        }
    }
    heap.into_sorted_vec()
        .into_iter()
        .map(|Candidate(id, distance)| (id, distance))
        .collect()
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

    fn distance(&self, query: &DenseVector, vector: &DenseVector, use_simd: bool) -> f32 {
        if use_simd {
            self.distance_metric
                .calculate_simd(query, vector)
                .unwrap_or(f32::MAX)
        } else {
            self.distance_metric
                .calculate(query, vector)
                .unwrap_or(f32::MAX)
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

        let use_simd = self.distance_metric.supports_simd();
        let distances = self
            .vectors
            .iter()
            .map(|(id, vector)| (id.clone(), self.distance(query, vector, use_simd)));

        // Every result, sorted, when `k` covers the table; a bounded heap otherwise.
        if k >= self.vectors.len() {
            let mut results: Vec<(RecordId, f32)> = distances.collect();
            results.sort_by(by_nearest);
            Ok(results)
        } else {
            Ok(top_k_streaming(distances, k))
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
        let results: Vec<(RecordId, f32)> = self
            .vectors
            .par_iter()
            .map(|(id, vector)| (id.clone(), self.distance(query, vector, use_simd)))
            .collect();
        Ok(top_k_sorted(results, k))
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

        let results: Vec<(RecordId, f32)> = self
            .vectors
            .iter()
            .zip(distances.iter())
            .map(|((id, _), &dist)| (id.clone(), dist))
            .collect();
        Ok(top_k_sorted(results, k))
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

        let distances = self
            .vectors
            .iter()
            .map(|(id, vector)| (id.clone(), query.distance_to(vector).unwrap_or(f32::MAX)));
        if k >= self.vectors.len() {
            let mut results: Vec<(RecordId, f32)> = distances.collect();
            results.sort_by(by_nearest);
            Ok(results)
        } else {
            Ok(top_k_streaming(distances, k))
        }
    }

    /// 并行搜索（使用预计算查询）
    fn search_parallel_precomputed(
        &self,
        query: &PrecomputedQuery,
        k: usize,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        let results: Vec<(RecordId, f32)> = self
            .vectors
            .par_iter()
            .map(|(id, vector)| (id.clone(), query.distance_to(vector).unwrap_or(f32::MAX)))
            .collect();
        Ok(top_k_sorted(results, k))
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

    /// Bytes held: the id and vector of every entry, and the vector of entries itself.
    pub fn memory_bytes(&self) -> usize {
        self.vectors.capacity() * std::mem::size_of::<(RecordId, DenseVector)>()
            + self
                .vectors
                .iter()
                .map(|(id, vector)| id.capacity() + vector.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>()
    }
}

#[async_trait]
impl VectorIndex for BruteForceIndex {
    async fn memory_bytes(&self) -> usize {
        BruteForceIndex::memory_bytes(self)
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(results: &[(RecordId, f32)]) -> Vec<&str> {
        results.iter().map(|(id, _)| id.as_str()).collect()
    }

    /// One record at the query and every id in `tied` at the same distance from it, inserted
    /// in the order given.
    fn index_with_ties<I: IntoIterator<Item = String>>(tied: I) -> BruteForceIndex {
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
        index.insert("exact".to_string(), vec![1.0, 0.0]).unwrap();
        for id in tied {
            index.insert(id, vec![0.0, 1.0]).unwrap();
        }
        index
    }

    #[test]
    fn a_tie_at_the_cutoff_keeps_the_smallest_ids_whatever_the_insertion_order() {
        let query = vec![1.0, 0.0];
        for order in [
            ["t1", "t2", "t3", "t4"],
            ["t4", "t3", "t2", "t1"],
            ["t3", "t1", "t4", "t2"],
        ] {
            let index = index_with_ties(order.iter().map(|s| s.to_string()));
            // The heap path (k below the table size) and the full sort (k covering it).
            assert_eq!(
                ids(&index.search_sync(&query, 3).unwrap()),
                ["exact", "t1", "t2"],
                "inserted as {order:?}"
            );
            assert_eq!(
                ids(&index.search_sync(&query, 10).unwrap()),
                ["exact", "t1", "t2", "t3", "t4"],
                "inserted as {order:?}"
            );
        }
    }

    #[test]
    fn the_parallel_path_breaks_ties_the_same_way() {
        // Past 1,000 vectors the search goes through `search_parallel`, whose
        // `select_nth_unstable` used to keep an arbitrary member of the tie.
        let tied = (0..1_200).rev().map(|i| format!("t{i:04}"));
        let index = index_with_ties(tied);
        let query = vec![1.0, 0.0];
        assert_eq!(
            ids(&index.search_sync(&query, 4).unwrap()),
            ["exact", "t0000", "t0001", "t0002"]
        );
    }

    #[test]
    fn a_nan_distance_sorts_after_every_number() {
        let mut results = vec![
            ("b".to_string(), f32::NAN),
            ("a".to_string(), 1.0),
            ("c".to_string(), 0.5),
            ("d".to_string(), f32::NAN),
        ];
        results.sort_by(by_nearest);
        assert_eq!(ids(&results), ["c", "a", "b", "d"]);
        assert_eq!(ids(&top_k_sorted(results.clone(), 2)), ["c", "a"]);
        assert_eq!(
            ids(&top_k_streaming(results.into_iter(), 3)),
            ["c", "a", "b"]
        );
    }

    #[test]
    fn memory_grows_with_what_is_stored() {
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
        let empty = index.memory_bytes();
        for i in 0..100 {
            index.insert(format!("id{i}"), vec![0.0; 64]).unwrap();
        }
        assert!(
            index.memory_bytes() >= empty + 100 * 64 * std::mem::size_of::<f32>(),
            "100 vectors of 64 floats are at least 25,600 bytes"
        );
    }
}
