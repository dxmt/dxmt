//! Memory pool for vector allocation optimization
//!
//! Provides pre-allocated vector pools to avoid runtime allocation overhead
//! and reduce memory fragmentation.

use std::sync::{Arc, Mutex};

/// Vector memory pool for reusing allocated vectors
pub struct VectorPool {
    /// Pool of available vectors
    pool: Arc<Mutex<Vec<Vec<f32>>>>,
    /// Vector dimension
    dimension: usize,
    /// Maximum pool size
    max_size: usize,
}

impl VectorPool {
    /// Create a new vector pool
    pub fn new(dimension: usize, max_size: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(Vec::with_capacity(max_size))),
            dimension,
            max_size,
        }
    }

    /// Allocate a vector from the pool (or create new if pool is empty)
    pub fn alloc(&self) -> Vec<f32> {
        let mut pool = self.pool.lock().unwrap();
        pool.pop().unwrap_or_else(|| vec![0.0; self.dimension])
    }

    /// Return a vector to the pool for reuse
    pub fn free(&self, mut vec: Vec<f32>) {
        // Clear the vector but keep capacity
        vec.clear();
        vec.resize(self.dimension, 0.0);

        let mut pool = self.pool.lock().unwrap();
        if pool.len() < self.max_size {
            pool.push(vec);
        }
        // If pool is full, just drop the vector
    }

    /// Pre-fill the pool with vectors
    pub fn prefill(&self, count: usize) {
        let mut pool = self.pool.lock().unwrap();
        let to_add = count.min(self.max_size - pool.len());
        for _ in 0..to_add {
            pool.push(vec![0.0; self.dimension]);
        }
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.pool.lock().unwrap().len()
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.pool.lock().unwrap().clear();
    }
}

impl Clone for VectorPool {
    fn clone(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
            dimension: self.dimension,
            max_size: self.max_size,
        }
    }
}

/// RAII guard for automatic vector return to pool
pub struct PooledVector {
    vector: Option<Vec<f32>>,
    pool: VectorPool,
}

impl PooledVector {
    /// Create a new pooled vector
    pub fn new(pool: VectorPool) -> Self {
        let vector = pool.alloc();
        Self {
            vector: Some(vector),
            pool,
        }
    }

    /// Get a reference to the vector
    pub fn vector(&self) -> &Vec<f32> {
        self.vector.as_ref().unwrap()
    }

    /// Get a mutable reference to the vector
    pub fn vector_mut(&mut self) -> &mut Vec<f32> {
        self.vector.as_mut().unwrap()
    }

    /// Take ownership of the vector (will not be returned to pool)
    pub fn take(mut self) -> Vec<f32> {
        self.vector.take().unwrap()
    }
}

impl Drop for PooledVector {
    fn drop(&mut self) {
        if let Some(vec) = self.vector.take() {
            self.pool.free(vec);
        }
    }
}

impl std::ops::Deref for PooledVector {
    type Target = Vec<f32>;

    fn deref(&self) -> &Self::Target {
        self.vector.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for PooledVector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vector.as_mut().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_pool_basic() {
        let pool = VectorPool::new(128, 10);

        // Allocate a vector
        let vec = pool.alloc();
        assert_eq!(vec.len(), 128);

        // Return to pool
        pool.free(vec);
        assert_eq!(pool.size(), 1);

        // Allocate again (should reuse)
        let vec2 = pool.alloc();
        assert_eq!(vec2.len(), 128);
        assert_eq!(pool.size(), 0);
    }

    #[test]
    fn test_vector_pool_max_size() {
        let pool = VectorPool::new(64, 3);

        // Fill pool
        pool.prefill(5); // Should only add 3
        assert_eq!(pool.size(), 3);

        // Try to add more
        pool.free(vec![0.0; 64]);
        assert_eq!(pool.size(), 3); // Still 3 (max size)
    }

    #[test]
    fn test_pooled_vector_raii() {
        let pool = VectorPool::new(256, 10);
        assert_eq!(pool.size(), 0);

        {
            let _vec = PooledVector::new(pool.clone());
            assert_eq!(pool.size(), 0); // Allocated
        }

        assert_eq!(pool.size(), 1); // Returned on drop
    }
}
