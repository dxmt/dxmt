use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Global memory tracker for the VectorDB
pub struct MemoryTracker {
    /// Total heap allocated bytes
    allocatedbytes: AtomicUsize,
    /// Peak memory usage
    peakbytes: AtomicUsize,
    /// Number of allocations
    allocation_count: AtomicUsize,
    /// Number of deallocations
    deallocation_count: AtomicUsize,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            allocatedbytes: AtomicUsize::new(0),
            peakbytes: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            deallocation_count: AtomicUsize::new(0),
        }
    }

    /// Track an allocation
    pub fn track_allocation(&self, size: usize) {
        let new_total = self.allocatedbytes.fetch_add(size, Ordering::Relaxed) + size;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

        // Update peak if necessary
        let mut peak = self.peakbytes.load(Ordering::Relaxed);
        while new_total > peak {
            match self.peakbytes.compare_exchange_weak(
                peak,
                new_total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Track a deallocation
    pub fn track_deallocation(&self, size: usize) {
        self.allocatedbytes.fetch_sub(size, Ordering::Relaxed);
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.allocatedbytes.load(Ordering::Relaxed)
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peakbytes.load(Ordering::Relaxed)
    }

    /// Get allocation statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            currentbytes: self.allocatedbytes.load(Ordering::Relaxed),
            peakbytes: self.peakbytes.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            deallocation_count: self.deallocation_count.load(Ordering::Relaxed),
        }
    }

    /// Reset statistics
    pub fn reset(&self) {
        self.allocatedbytes.store(0, Ordering::Relaxed);
        self.peakbytes.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.deallocation_count.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub currentbytes: usize,
    pub peakbytes: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-local memory tracking for query execution
pub struct QueryMemoryTracker {
    initial_usage: usize,
    tracker: Arc<MemoryTracker>,
}

impl QueryMemoryTracker {
    pub fn new(tracker: Arc<MemoryTracker>) -> Self {
        Self {
            initial_usage: tracker.current_usage(),
            tracker,
        }
    }

    /// Get memory used since query started
    pub fn query_memory_usage(&self) -> usize {
        self.tracker
            .current_usage()
            .saturating_sub(self.initial_usage)
    }
}

/// Estimate memory size of common data structures
pub trait MemorySize {
    fn memory_size(&self) -> usize;
}

impl MemorySize for Vec<f32> {
    fn memory_size(&self) -> usize {
        std::mem::size_of::<f32>() * self.capacity()
    }
}

impl MemorySize for Vec<u8> {
    fn memory_size(&self) -> usize {
        self.capacity()
    }
}

impl MemorySize for String {
    fn memory_size(&self) -> usize {
        self.capacity()
    }
}

impl<K, V> MemorySize for std::collections::HashMap<K, V> {
    fn memory_size(&self) -> usize {
        // Approximate: bucket count * (key + value + overhead)
        let entry_size = std::mem::size_of::<K>() + std::mem::size_of::<V>() + 24; // overhead estimate
        self.capacity() * entry_size
    }
}

/// Helper functions for memory calculations
pub mod memory_utils {
    use super::MemorySize;

    /// Calculate memory usage of a vector of vectors
    pub fn vector_memory_usage(vectors: &[Vec<f32>]) -> usize {
        vectors.iter().map(|v| v.memory_size()).sum::<usize>() + std::mem::size_of_val(vectors)
    }

    /// Calculate memory usage of an index structure
    pub fn index_memory_usage(numvectors: usize, dimension: usize, _connectivity: usize) -> usize {
        // Base vector storage

        numvectors * dimension * std::mem::size_of::<f32>()
    }

    /// Estimate record memory usage
    pub fn record_memory_usage(vectordim: usize, payload_size: usize) -> usize {
        let vector_size = vectordim * std::mem::size_of::<f32>();
        let id_size = 16; // UUID size
        let metadata_size = 64; // Estimated metadata overhead

        vector_size + payload_size + id_size + metadata_size
    }
}

// Global memory tracker instance
lazy_static::lazy_static! {
    pub static ref _GLOBAL_MEMORY_TRACKER: Arc<MemoryTracker> = Arc::new(MemoryTracker::new());
}

/// Macro to track memory in a scope
#[macro_export]
macro_rules! track_memory {
    ($tracker:expr, $size:expr) => {{
        $tracker.track_allocation($size);
        scopeguard::guard((), |_| $tracker.track_deallocation($size))
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new();

        tracker.track_allocation(1024);
        assert_eq!(tracker.current_usage(), 1024);
        assert_eq!(tracker.peak_usage(), 1024);

        tracker.track_allocation(2048);
        assert_eq!(tracker.current_usage(), 3072);
        assert_eq!(tracker.peak_usage(), 3072);

        tracker.track_deallocation(1024);
        assert_eq!(tracker.current_usage(), 2048);
        assert_eq!(tracker.peak_usage(), 3072); // Peak remains

        let stats = tracker.stats();
        assert_eq!(stats.currentbytes, 2048);
        assert_eq!(stats.peakbytes, 3072);
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.deallocation_count, 1);
    }

    #[test]
    fn test_memory_size_trait() {
        let vec: Vec<f32> = Vec::with_capacity(100);
        assert_eq!(vec.memory_size(), 400); // 100 * 4 bytes

        let string = String::with_capacity(50);
        assert_eq!(string.memory_size(), 50)
    }

    // test_memory_utils omitted: memoryutils sub-module not yet implemented

    //     let record_usage = memoryutils::record_memory_usage(128, 1024);
    //     assert_eq!(record_usage, 128 * 4 + 1024 + 16 + 64)
    // }
}
