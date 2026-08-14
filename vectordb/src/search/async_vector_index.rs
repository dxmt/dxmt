use crate::search::VectorIndex;
use crate::types::vector::{RecordId, DenseVector};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Async wrapper for VectorIndex trait
///
/// This wrapper provides async methods and thread-safe access to any VectorIndex implementation
#[async_trait]
pub trait AsyncVectorIndex: Send + Sync {
    /// Add a vector to the index
    async fn add(&self, id: RecordId, vector: DenseVector) -> crate::Result<()>;

    /// Search for nearest neighbors
    async fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>>;

    /// Update a vector in the index
    async fn update(&self, id: RecordId, vector: DenseVector) -> crate::Result<()>;

    /// Remove a vector from the index
    async fn remove(&self, id: &RecordId) -> crate::Result<bool>;

    /// Get the number of vectors in the index
    async fn len(&self) -> usize;

    /// Clear all vectors from the index
    async fn clear(&self) -> crate::Result<()>;
}

/// Thread-safe async wrapper for any VectorIndex implementation
pub struct AsyncVectorIndexWrapper<T: Send + Sync + 'static> {
    inner: Arc<RwLock<T>>,
}

impl<T: Send + Sync + 'static> AsyncVectorIndexWrapper<T> {
    /// Create a new async wrapper
    pub fn new(index: T) -> Self {
        Self {
            inner: Arc::new(RwLock::new(index)),
        }
    }
}

#[async_trait]
impl<T: VectorIndex + Send + Sync> AsyncVectorIndex for AsyncVectorIndexWrapper<T> {
    async fn add(&self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        let mut index = self.inner.write().await;
        index.add(id, vector).await
    }

    async fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        let index = self.inner.read().await;
        let results = index.search(query, k).await?;
        // Convert String to RecordId (they should be the same type)
        Ok(results)
    }

    async fn update(&self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        let mut index = self.inner.write().await;
        // Remove old vector if it exists
        let _ = index.delete(&id).await;
        // Insert new vector
        index.add(id, vector).await
    }

    async fn remove(&self, id: &RecordId) -> crate::Result<bool> {
        let mut index = self.inner.write().await;
        match index.delete(id).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    async fn len(&self) -> usize {
        let index = self.inner.read().await;
        index.len()
    }

    async fn clear(&self) -> crate::Result<()> {
        // Since clear is not part of VectorIndex trait, we can't clear
        // Just return an error for now
        Err(crate::Error::NotImplemented("Clear not supported".to_string()))
    }
}

/// Shared async vector index that can be cloned
#[derive(Clone)]
pub struct SharedAsyncVectorIndex {
    inner: Arc<dyn AsyncVectorIndex>,
}

impl SharedAsyncVectorIndex {
    /// Create a new shared async vector index
    pub fn new<T: AsyncVectorIndex + 'static>(index: T) -> Self {
        Self {
            inner: Arc::new(index),
        }
    }

    /// Create from a sync VectorIndex implementation
    pub fn from_sync<T: VectorIndex + Send + Sync + 'static>(index: T) -> Self {
        Self::new(AsyncVectorIndexWrapper::new(index))
    }
}

#[async_trait]
impl AsyncVectorIndex for SharedAsyncVectorIndex {
    async fn add(&self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        self.inner.add(id, vector).await
    }

    async fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        self.inner.search(query, k).await
    }

    async fn update(&self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        self.inner.update(id, vector).await
    }

    async fn remove(&self, id: &RecordId) -> crate::Result<bool> {
        self.inner.remove(id).await
    }

    async fn len(&self) -> usize {
        self.inner.len().await
    }

    async fn clear(&self) -> crate::Result<()> {
        self.inner.clear().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::BruteForceIndex;
use crate::RecordId;

    #[tokio::test]
    async fn test_async_wrapper() {
        let index = BruteForceIndex::new(3);
        let async_index = AsyncVectorIndexWrapper::new(index);

        // Test add
        async_index
            .add("vec1".to_string(), vec![1.0, 2.0, 3.0])
            .await?;
        async_index
            .add("vec2".to_string(), vec![4.0, 5.0, 6.0])
            .await?;

        // Test search
        let results = async_index.search(&vec![1.0, 2.0, 3.0], 2).await?;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "vec1");

        // Test update
        async_index
            .update("vec1".to_string(), vec![7.0, 8.0, 9.0])
            .await?;
        let results = async_index.search(&vec![7.0, 8.0, 9.0], 1).await?;
        assert_eq!(results[0].0, "vec1");

        // Test remove
        let removed = async_index.remove(&"vec2".to_string()).await?;
        assert!(removed);
        assert_eq!(async_index.len().await, 1);

        // Test clear
        async_index.clear().await?;
        assert_eq!(async_index.len().await, 0);
    }

    #[tokio::test]
    async fn test_shared_async_index() {
        let index = BruteForceIndex::new(2);
        let shared_index = SharedAsyncVectorIndex::from_sync(index);

        // Clone the index
        let index_clone = shared_index.clone();

        // Add from original
        shared_index
            .add("doc1".to_string(), vec![1.0, 2.0])
            .await?;

        // Search from clone
        let results = index_clone.search(&vec![1.0, 2.0], 1).await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc1")
    }
}
