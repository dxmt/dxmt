//! Memory optimization utilities for DXMT VectorDB

use parking_lot::RwLock;
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::mem;
use std::ptr;
use std::sync::Arc;

use super::CacheSizes;
use crate::embeddings::EmbeddingError;

/// Memory optimizer for cache-efficient data structures
#[allow(dead_code)]
pub struct MemoryOptimizer {
    cache_sizes: CacheSizes,
    memory_pool: Arc<RwLock<MemoryPool>>,
    prefetcher: Arc<Prefetcher>,
}

impl MemoryOptimizer {
    pub fn new(cache_sizes: CacheSizes) -> Self {
        Self {
            cache_sizes,
            memory_pool: Arc::new(RwLock::new(MemoryPool::new())),
            prefetcher: Arc::new(Prefetcher::new()),
        }
    }

    pub fn allocate_aligned<T>(&self, count: usize) -> Result<CacheLineAligned<T>, EmbeddingError>
    where
        T: Default + Clone,
    {
        CacheLineAligned::new(count)
    }

    pub fn optimize_layout<T>(&self, data: Vec<T>) -> OptimizedBuffer<T> {
        OptimizedBuffer::from_vec(data, self.cache_sizes.l1_kb * 1024)
    }

    pub fn prefetch_data<T>(&self, data: &[T], stride: usize) {
        self.prefetcher.prefetch_array(data, stride);
    }

    pub fn get_optimal_batch_size(&self, element_size: usize) -> usize {
        // Optimize for L2 cache
        let l2_bytes = self.cache_sizes.l2_kb * 1024;
        (l2_bytes / element_size / 2).max(1) // Use half of L2 for safety
    }

    pub fn get_cache_sizes(&self) -> &CacheSizes {
        &self.cache_sizes
    }
}

/// Cache-line aligned data structure
#[repr(align(64))] // 64-byte cache line alignment
pub struct CacheLineAligned<T> {
    data: Vec<T>,
    ptr: *mut T,
    layout: Layout,
}

impl<T> CacheLineAligned<T>
where
    T: Default + Clone,
{
    pub fn new(count: usize) -> Result<Self, EmbeddingError> {
        let alignment = 64; // Cache line size
        let size = count * mem::size_of::<T>();

        // Round up to multiple of alignment
        let aligned_size = (size + alignment - 1) & !(alignment - 1);

        let layout = Layout::from_size_align(aligned_size, alignment)
            .map_err(|e| EmbeddingError::InternalError(e.to_string()))?;

        unsafe {
            let ptr = alloc(layout) as *mut T;
            if ptr.is_null() {
                return Err(EmbeddingError::InternalError(
                    "Failed to allocate aligned memory".to_string(),
                ));
            }

            // Initialize memory
            for i in 0..count {
                ptr::write(ptr.add(i), T::default());
            }

            let data = vec![T::default(); count];

            Ok(Self { data, ptr, layout })
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T> Drop for CacheLineAligned<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

unsafe impl<T: Send> Send for CacheLineAligned<T> {}
unsafe impl<T: Sync> Sync for CacheLineAligned<T> {}

/// Memory pool for efficient allocation
pub struct MemoryPool {
    pools: HashMap<usize, Vec<Vec<u8>>>,
    total_allocated: usize,
    max_size: usize,
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            total_allocated: 0,
            max_size: 1024 * 1024 * 1024, // 1GB max
        }
    }

    pub fn allocate(&mut self, size: usize) -> Option<Vec<u8>> {
        // Round up to power of 2
        let aligned_size = size.next_power_of_two();

        // Check if we have a free buffer
        if let Some(pool) = self.pools.get_mut(&aligned_size) {
            if let Some(mut buffer) = pool.pop() {
                buffer.resize(size, 0);
                return Some(buffer);
            }
        }

        // Allocate new buffer if under limit
        if self.total_allocated + aligned_size <= self.max_size {
            self.total_allocated += aligned_size;
            Some(vec![0u8; aligned_size])
        } else {
            None
        }
    }

    pub fn deallocate(&mut self, mut buffer: Vec<u8>) {
        buffer.clear();
        let size = buffer.capacity();

        self.pools.entry(size).or_default().push(buffer);
    }

    pub fn clear(&mut self) {
        self.pools.clear();
        self.total_allocated = 0;
    }

    pub fn stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            total_allocated: self.total_allocated,
            num_pools: self.pools.len(),
            buffers_cached: self.pools.values().map(|p| p.len()).sum(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_allocated: usize,
    pub num_pools: usize,
    pub buffers_cached: usize,
}

/// Optimized buffer with cache-friendly layout
pub struct OptimizedBuffer<T> {
    data: Vec<T>,
    chunk_size: usize,
}

impl<T> OptimizedBuffer<T> {
    pub fn from_vec(data: Vec<T>, cache_size: usize) -> Self {
        let element_size = mem::size_of::<T>();
        let chunk_size = (cache_size / element_size).max(1);

        Self { data, chunk_size }
    }

    pub fn chunks(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks(self.chunk_size)
    }

    pub fn chunks_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.data.chunks_mut(self.chunk_size)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Prefetcher for improving cache hit rates
pub struct Prefetcher;

impl Default for Prefetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Prefetcher {
    pub fn new() -> Self {
        Self
    }

    pub fn prefetch_array<T>(&self, data: &[T], stride: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            for i in (0..data.len()).step_by(stride) {
                unsafe {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch(
                        &data[i] as *const T as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
        }
    }

    pub fn prefetch_random<T>(&self, data: &[T], indices: &[usize]) {
        #[cfg(target_arch = "x86_64")]
        {
            for &idx in indices {
                if idx < data.len() {
                    unsafe {
                        use std::arch::x86_64::_mm_prefetch;
                        _mm_prefetch(
                            &data[idx] as *const T as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }
        }
    }
}

/// NUMA-aware memory allocation
pub struct NumaAllocator {
    node_count: usize,
    preferred_node: usize,
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl NumaAllocator {
    pub fn new() -> Self {
        Self {
            node_count: Self::detect_numa_nodes(),
            preferred_node: 0,
        }
    }

    pub fn set_preferred_node(&mut self, node: usize) {
        if node < self.node_count {
            self.preferred_node = node;
        }
    }

    pub fn allocate_on_node<T>(&self, count: usize, _node: usize) -> Result<Vec<T>, EmbeddingError>
    where
        T: Default + Clone,
    {
        // In production, use libnuma or similar
        // For now, just allocate normally
        Ok(vec![T::default(); count])
    }

    fn detect_numa_nodes() -> usize {
        // Simplified NUMA detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(nodes) = std::fs::read_dir("/sys/devices/system/node/") {
                nodes
                    .filter(|entry| {
                        entry
                            .as_ref()
                            .ok()
                            .and_then(|e| e.file_name().to_str().map(|s| s.starts_with("node")))
                            .unwrap_or(false)
                    })
                    .count()
            } else {
                1
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            1
        }
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }
}

/// Column-store layout for better cache utilization
pub struct ColumnStore<T> {
    columns: Vec<Vec<T>>,
    row_count: usize,
}

impl<T: Clone> ColumnStore<T> {
    pub fn new(data: Vec<Vec<T>>) -> Self {
        if data.is_empty() {
            return Self {
                columns: Vec::new(),
                row_count: 0,
            };
        }

        let row_count = data.len();
        let col_count = data[0].len();

        let mut columns = vec![Vec::with_capacity(row_count); col_count];

        for row in data {
            for (col_idx, value) in row.into_iter().enumerate() {
                columns[col_idx].push(value);
            }
        }

        Self { columns, row_count }
    }

    pub fn get_column(&self, index: usize) -> Option<&[T]> {
        self.columns.get(index).map(|c| c.as_slice())
    }

    pub fn get_row(&self, index: usize) -> Option<Vec<T>> {
        if index >= self.row_count {
            return None;
        }

        Some(self.columns.iter().map(|col| col[index].clone()).collect())
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }
}

/// Memory access pattern optimizer
pub struct AccessPatternOptimizer {
    access_history: Vec<usize>,
    pattern_cache: HashMap<Vec<usize>, Vec<usize>>,
}

impl Default for AccessPatternOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AccessPatternOptimizer {
    pub fn new() -> Self {
        Self {
            access_history: Vec::new(),
            pattern_cache: HashMap::new(),
        }
    }

    pub fn record_access(&mut self, index: usize) {
        self.access_history.push(index);

        // Keep last 1000 accesses
        if self.access_history.len() > 1000 {
            self.access_history.remove(0);
        }
    }

    pub fn predict_next_accesses(&self, window_size: usize) -> Vec<usize> {
        if self.access_history.len() < window_size {
            return Vec::new();
        }

        let recent = &self.access_history[self.access_history.len() - window_size..];

        // Look for pattern in cache
        if let Some(prediction) = self.pattern_cache.get(recent) {
            return prediction.clone();
        }

        // Simple prediction: return most frequently accessed after this pattern
        let mut frequency = HashMap::new();

        for window in self.access_history.windows(window_size + 1) {
            if &window[..window_size] == recent {
                *frequency.entry(window[window_size]).or_insert(0) += 1;
            }
        }

        let mut predictions: Vec<_> = frequency.into_iter().collect();
        predictions.sort_by_key(|(_, count)| -*count);

        predictions
            .into_iter()
            .take(10)
            .map(|(idx, _)| idx)
            .collect()
    }

    pub fn optimize_layout<T: Clone>(&self, data: &[T]) -> Vec<T> {
        // Reorder data based on access patterns
        let mut optimized = Vec::with_capacity(data.len());
        let mut visited = vec![false; data.len()];

        // Add frequently accessed items first
        let mut access_counts = HashMap::new();
        for &idx in &self.access_history {
            if idx < data.len() {
                *access_counts.entry(idx).or_insert(0) += 1;
            }
        }

        let mut sorted_indices: Vec<_> = access_counts.into_iter().collect();
        sorted_indices.sort_by_key(|(_, count)| -*count);

        for (idx, _) in sorted_indices {
            if !visited[idx] {
                optimized.push(data[idx].clone());
                visited[idx] = true;
            }
        }

        // Add remaining items
        for (idx, item) in data.iter().enumerate() {
            if !visited[idx] {
                optimized.push(item.clone());
            }
        }

        optimized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_line_aligned() {
        let aligned: CacheLineAligned<f32> = CacheLineAligned::new(100).unwrap();
        assert_eq!(aligned.len(), 100);

        // Check alignment
        let ptr = aligned.ptr as usize;
        assert_eq!(ptr % 64, 0);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new();

        let buffer1 = pool.allocate(1024).unwrap();
        assert_eq!(buffer1.len(), 1024);

        pool.deallocate(buffer1);

        let buffer2 = pool.allocate(1024).unwrap();
        assert_eq!(buffer2.len(), 1024);

        let stats = pool.stats();
        assert_eq!(stats.buffers_cached, 0);
    }

    #[test]
    fn test_column_store() {
        let data = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];

        let store = ColumnStore::new(data);

        assert_eq!(store.column_count(), 3);
        assert_eq!(store.row_count(), 3);

        assert_eq!(store.get_column(0), Some(&[1, 4, 7][..]));
        assert_eq!(store.get_row(1), Some(vec![4, 5, 6]));
    }
}
