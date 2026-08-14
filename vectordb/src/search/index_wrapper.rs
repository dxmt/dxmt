//! Index wrapper providing interior mutability for Arc<dyn VectorIndex>
//!
//! This module solves the Arc<dyn VectorIndex> design conflict by providing
//! a wrapper that uses RwLock to enable mutations through immutable references.

use crate::types::vector::{RecordId, DenseVector};
use std::sync::Arc;
use tokio::sync::RwLock;

// Use index_traits::VectorIndex which HnswIndex and BruteForceIndex implement
use super::index_traits::{VectorIndex, IndexStats};
use crate::Result;

#[cfg(test)]
use async_trait::async_trait;

/// Wrapper around a boxed VectorIndex with RwLock for interior mutability
///
/// This enables Arc<IndexWrapper> to provide both read (search) and write
/// (insert, delete) operations without requiring mutable references.
pub struct IndexWrapper {
    /// The actual index, wrapped in RwLock for interior mutability
    index: Arc<RwLock<Box<dyn VectorIndex>>>,
}

impl IndexWrapper {
    /// Create a new index wrapper from a boxed trait object
    pub fn new(index: Box<dyn VectorIndex>) -> Self {
        Self {
            index: Arc::new(RwLock::new(index)),
        }
    }

    /// Create a new index wrapper from an Arc<RwLock<>>
    pub fn from_arc(index: Arc<RwLock<Box<dyn VectorIndex>>>) -> Self {
        Self { index }
    }

    /// Clone the underlying Arc for sharing between async tasks
    pub fn clone_arc(&self) -> Arc<RwLock<Box<dyn VectorIndex>>> {
        Arc::clone(&self.index)
    }

    /// Insert a vector - requires write lock
    pub async fn insert(&self, id: RecordId, vector: DenseVector) -> Result<()> {
        let mut index = self.index.write().await;
        // Convert RecordId to String for trait compatibility
        index.add(id.to_string(), vector).await
    }

    /// Search for nearest neighbors - requires read lock
    pub async fn search(&self, query: &DenseVector, k: usize) -> Result<Vec<(RecordId, f32)>> {
        let index = self.index.read().await;
        // Call async search and convert String back to RecordId
        let results = index.search(query, k).await?;
        Ok(results.into_iter()
            .map(|(id_str, score)| (RecordId::from(id_str), score))
            .collect())
    }

    /// Search with an explicit ef_search hint (HNSW respects this; other indexes fall back to search).
    pub async fn search_with_ef_hint(&self, query: &DenseVector, k: usize, ef: usize) -> Result<Vec<(RecordId, f32)>> {
        let index = self.index.read().await;
        let results = index.search_with_ef_hint(query, k, ef).await?;
        Ok(results.into_iter()
            .map(|(id_str, score)| (RecordId::from(id_str), score))
            .collect())
    }

    /// Filter-in-graph search. HNSW uses Acorn-style in-graph filtering;
    /// other indexes fall back to amplified post-filter.
    pub async fn search_with_filter(
        &self,
        query: &DenseVector,
        k: usize,
        ef: usize,
        filter_ids: &std::collections::HashSet<String>,
    ) -> Result<Vec<(RecordId, f32)>> {
        let index = self.index.read().await;
        let results = index.search_with_filter(query, k, ef, filter_ids).await?;
        Ok(results.into_iter()
            .map(|(id_str, score)| (RecordId::from(id_str), score))
            .collect())
    }

    /// Remove a vector - requires write lock
    pub async fn remove(&self, id: &RecordId) -> Result<bool> {
        let mut index = self.index.write().await;
        index.delete(&id.to_string()).await?;
        Ok(true)
    }

    /// Get index length - requires read lock
    pub async fn len(&self) -> usize {
        let index = self.index.read().await;
        index.len()
    }

    /// Check if index is empty - requires read lock
    pub async fn is_empty(&self) -> bool {
        let index = self.index.read().await;
        index.is_empty()
    }

    /// Clear the index - requires write lock
    pub async fn clear(&self) -> Result<()> {
        // Note: VectorIndex trait doesn't have a clear() method
        // This would require iterating through all IDs and deleting them
        // or recreating the index. For now, return an error.
        Err(crate::Error::InvalidInput(
            "Index clear operation not yet implemented".to_string()
        ))
    }

    /// Get index statistics - requires read lock
    pub async fn get_stats(&self) -> IndexStats {
        let index = self.index.read().await;
        // Create default stats - SerializableIndex trait methods not directly accessible
        IndexStats {
            total_vectors: index.len(),
            index_size_bytes: 0,
            build_time_ms: 0,
        }
    }
}

impl Clone for IndexWrapper {
    fn clone(&self) -> Self {
        Self {
            index: Arc::clone(&self.index),
        }
    }
}

/// Collection of indexes with Arc<RwLock> interior mutability
#[derive(Clone)]
pub struct IndexCollection {
    /// Map of index name to wrapper
    indexes: Arc<RwLock<std::collections::HashMap<String, IndexWrapper>>>,
}

impl IndexCollection {
    /// Create a new index collection
    pub fn new() -> Self {
        Self {
            indexes: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Add an index to the collection
    pub async fn add_index(&self, name: String, index: IndexWrapper) {
        let mut indexes = self.indexes.write().await;
        indexes.insert(name, index);
    }

    /// Get an index by name
    pub async fn get_index(&self, name: &str) -> Option<IndexWrapper> {
        let indexes = self.indexes.read().await;
        indexes.get(name).cloned()
    }

    /// Remove an index from the collection
    pub async fn remove_index(&self, name: &str) -> Option<IndexWrapper> {
        let mut indexes = self.indexes.write().await;
        indexes.remove(name)
    }

    /// List all index names
    pub async fn list_index_names(&self) -> Vec<String> {
        let indexes = self.indexes.read().await;
        indexes.keys().cloned().collect()
    }

    /// Get count of indexes
    pub async fn index_count(&self) -> usize {
        let indexes = self.indexes.read().await;
        indexes.len()
    }
}

impl Default for IndexCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock VectorIndex for testing
    #[derive(Debug)]
    struct MockIndex {
        vectors: std::collections::HashMap<String, Vec<f32>>,
    }

    impl MockIndex {
        fn new() -> Self {
            Self {
                vectors: std::collections::HashMap::new(),
            }
        }
    }

    #[async_trait]
    impl VectorIndex for MockIndex {
        async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
            self.vectors.insert(id, vector);
            Ok(())
        }

        async fn search(&self, _query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
            // Return mock results
            Ok(self
                .vectors
                .keys()
                .take(k)
                .map(|id| (id.clone(), 0.9))
                .collect())
        }

        async fn delete(&mut self, id: &str) -> Result<()> {
            self.vectors.remove(id);
            Ok(())
        }

        fn len(&self) -> usize {
            self.vectors.len()
        }

        fn is_empty(&self) -> bool {
            self.vectors.is_empty()
        }
    }

    #[tokio::test]
    async fn test_index_wrapper_insert_and_search() {
        let mock_index = Box::new(MockIndex::new());
        let wrapper = IndexWrapper::new(mock_index);

        // Insert
        wrapper
            .insert(
                "doc1".to_string(),
                vec![0.1, 0.2, 0.3],
            )
            .await
            .unwrap();

        // Check length
        assert_eq!(wrapper.len().await, 1);

        // Search
        let results = wrapper
            .search(&vec![0.1, 0.2, 0.3], 10)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_index_wrapper_concurrent_access() {
        let mock_index = Box::new(MockIndex::new());
        let wrapper = IndexWrapper::new(mock_index);

        let wrapper1 = wrapper.clone();
        let wrapper2 = wrapper.clone();

        // Concurrent insert and search
        let insert_handle = tokio::spawn(async move {
            wrapper1
                .insert("doc1".to_string(), vec![0.1; 128])
                .await
        });

        let search_handle = tokio::spawn(async move {
            wrapper2
                .search(&vec![0.1; 128], 10)
                .await
        });

        let _ = insert_handle.await;
        let _ = search_handle.await;

        assert_eq!(wrapper.len().await, 1);
    }

    #[tokio::test]
    async fn test_index_collection() {
        let collection = IndexCollection::new();

        let mock_index = Box::new(MockIndex::new());
        let wrapper = IndexWrapper::new(mock_index);

        collection.add_index("index1".to_string(), wrapper).await;

        let retrieved = collection.get_index("index1").await;
        assert!(retrieved.is_some());

        assert_eq!(collection.index_count().await, 1);

        let names = collection.list_index_names().await;
        assert_eq!(names.len(), 1);
    }
}
