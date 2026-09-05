//! Index wrapper providing interior mutability for Arc<dyn VectorIndex>
//!
//! This module solves the Arc<dyn VectorIndex> design conflict by providing
//! a wrapper that uses RwLock to enable mutations through immutable references.

use crate::types::vector::{DenseVector, RecordId};
use std::sync::Arc;
use tokio::sync::RwLock;

// Use index_traits::VectorIndex which HnswIndex and BruteForceIndex implement
use super::index_traits::{IndexStats, VectorIndex};
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
    /// Held across a phased insert, and by anything that swaps the index out.
    ///
    /// A phased insert releases the `RwLock` between its phases so searches can run, which
    /// opens a window `replace` could step into: the plans were computed against one graph
    /// and would be applied to another, and the node indices in them would mean something
    /// else. Nothing else contends for this -- inserts are already serialised by the storage
    /// layer's single insert task, and a rebuild is rare.
    structural: Arc<tokio::sync::Mutex<()>>,
}

impl IndexWrapper {
    /// Create a new index wrapper from a boxed trait object
    pub fn new(index: Box<dyn VectorIndex>) -> Self {
        Self {
            index: Arc::new(RwLock::new(index)),
            structural: Arc::new(tokio::sync::Mutex::new(())),
        }
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

    /// Insert several vectors under one write lock.
    ///
    /// Calling `insert` in a loop took the lock once per record — a batch of two
    /// thousand acquired and released it two thousand times — and gave the index no
    /// chance to do the batch any faster than one at a time.
    pub async fn insert_batch(&self, items: Vec<(RecordId, DenseVector)>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let items: Vec<(String, Vec<f32>)> = items
            .into_iter()
            .map(|(id, v)| (id.to_string(), v))
            .collect();

        // The write lock is taken for the appending and nothing else.
        //
        // Searches take this lock for reading, so holding it for a whole batch parks every
        // reader for as long as the batch takes: at a million nodes, 4 ms of planning plus
        // 0.25 ms of appending plus 28 ms of reverse-edge pruning. Only the appending needs
        // exclusive access. Measured before this change, on Cohere 1M at 2,000 rows/s with
        // concurrent search: p95 4.6 ms, p99 91 ms, against a p99 of 4.7 ms once the writes
        // stopped (design doc section 33).
        //
        // Chunking the batch was tried first and made it worse -- p99 down by a third, p95 up
        // five to nine times -- because tokio's `RwLock` is write-preferring, so acquiring it
        // more often blocks readers more often rather than less (section 33.1). Holding it
        // less is the fix; holding it more briefly is not.
        let _structural = self.structural.lock().await;

        let prepared = {
            let mut index = self.index.write().await;
            if index.as_hnsw().is_none() {
                return index.add_batch(items).await;
            }
            index
                .as_hnsw_mut()
                .expect("checked on the line above, under the same lock")
                .prepare_batch(items)?
        };
        // `None` means the batch was small enough that the serial path took it already.
        let Some(prepared) = prepared else {
            return Ok(());
        };

        // A. Planning: shared, and every core.
        let plans = {
            let index = self.index.read().await;
            index
                .as_hnsw()
                .expect("the structural lock holds the index type still")
                .plan_batch(&prepared)
        };

        // B. Appending: exclusive, and the only phase that has to be.
        let applied = {
            let mut index = self.index.write().await;
            index
                .as_hnsw_mut()
                .expect("the structural lock holds the index type still")
                .apply_batch(prepared, plans)
        };

        // C. Reverse edges and pruning: shared, and the longest of the three.
        {
            let index = self.index.read().await;
            index
                .as_hnsw()
                .expect("the structural lock holds the index type still")
                .link_batch(&applied);
        }
        Ok(())
    }

    /// Search for nearest neighbors - requires read lock
    pub async fn search(&self, query: &DenseVector, k: usize) -> Result<Vec<(RecordId, f32)>> {
        let index = self.index.read().await;
        // Call async search and convert String back to RecordId
        let results = index.search(query, k).await?;
        Ok(results
            .into_iter()
            .map(|(id_str, score)| (RecordId::from(id_str), score))
            .collect())
    }

    /// Search with an explicit ef_search hint (HNSW respects this; other indexes fall back to search).
    pub async fn search_with_ef_hint(
        &self,
        query: &DenseVector,
        k: usize,
        ef: usize,
    ) -> Result<Vec<(RecordId, f32)>> {
        let index = self.index.read().await;
        let results = index.search_with_ef_hint(query, k, ef).await?;
        Ok(results
            .into_iter()
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
        Ok(results
            .into_iter()
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

    /// Build the whole index from `vectors` in one pass.
    pub async fn bulk_build(&self, vectors: Vec<(RecordId, DenseVector)>) -> Result<()> {
        let mut index = self.index.write().await;
        index
            .bulk_build(
                vectors
                    .into_iter()
                    .map(|(id, v)| (id.to_string(), v))
                    .collect(),
            )
            .await
    }

    /// Run the index's incremental upkeep; false means it has none.
    ///
    /// This takes the index write lock, so searches wait — but only for as long as the new data
    /// takes to absorb, not for a pass over the whole index.
    pub async fn maintain(&self) -> Result<bool> {
        let mut index = self.index.write().await;
        index.maintain().await
    }

    /// Object-storage keys this index still needs, if it keeps any.
    pub async fn live_segment_keys(&self) -> Option<Vec<String>> {
        let index = self.index.read().await;
        index.live_segment_keys()
    }

    /// Write the index to `path`; false means this index type has no on-disk form.
    pub async fn snapshot_to(&self, path: &std::path::Path) -> Result<bool> {
        let index = self.index.read().await;
        // Serialising a 1M-node graph takes tens of seconds of CPU. Done inline it parks the
        // worker thread this future is on, and the runtime's other tasks with it: a 1 s timer
        // measured 53.6 s late during one snapshot (design doc section 24.3.3).
        // `block_in_place` hands the thread over and lets the runtime start a replacement; it
        // only exists on the multi-thread runtime, so the current-thread runtime that
        // `#[tokio::test]` builds keeps the inline call.
        let multi_thread = tokio::runtime::Handle::try_current()
            .map(|h| h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread)
            .unwrap_or(false);
        if multi_thread {
            tokio::task::block_in_place(|| index.snapshot_to(path))
        } else {
            index.snapshot_to(path)
        }
    }

    /// Replace the wrapped index in place, keeping the same `Arc` so existing clones of this
    /// wrapper — the table holds one, and searches hold more — observe the new index.
    pub async fn replace(&self, new_index: Box<dyn VectorIndex>) {
        // Ahead of the write lock: a phased insert releases that lock between its phases, and
        // swapping the index out in one of those gaps would apply plans built against the old
        // graph to a new one.
        let _structural = self.structural.lock().await;
        let mut index = self.index.write().await;
        *index = new_index;
    }

    /// Clear the index - requires write lock
    pub async fn clear(&self) -> Result<()> {
        // Note: VectorIndex trait doesn't have a clear() method
        // This would require iterating through all IDs and deleting them
        // or recreating the index. For now, return an error.
        Err(crate::Error::InvalidInput(
            "Index clear operation not yet implemented".to_string(),
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
            structural: Arc::clone(&self.structural),
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

    /// A phased insert must build the same graph the one-shot path did.
    ///
    /// `insert_batch` now releases the index lock between planning, appending and linking so
    /// searches can run in the gaps (design doc section 33.1). The phases were already there
    /// inside `HnswIndex`; what is new is that a lock is dropped between them, and what that
    /// must not change is the result. The graph has to be over `PARALLEL_PLAN_MIN_NODES` for
    /// the phased path to be taken at all, which is why this builds ten thousand nodes first.
    #[tokio::test]
    async fn a_phased_batch_insert_finds_what_a_serial_one_finds() {
        use crate::search::hnsw::HnswIndex;
        use crate::types::vector::DistanceMetric;

        const SEED_NODES: usize = 10_200;
        const BATCH: usize = 400;
        const DIM: usize = 16;

        // Deterministic and all distinct: vectors from `i % n` would make the nearest
        // neighbour a tie among identical rows and the comparison meaningless.
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        };
        let vectors: Vec<Vec<f32>> = (0..SEED_NODES + BATCH)
            .map(|_| (0..DIM).map(|_| next()).collect())
            .collect();

        let build = |phased: bool| {
            let vectors = vectors.clone();
            async move {
                let mut index = HnswIndex::new(DistanceMetric::Cosine, None);
                for (i, v) in vectors.iter().take(SEED_NODES).enumerate() {
                    index.insert(i.to_string(), v.clone()).unwrap();
                }
                let wrapper = IndexWrapper::new(Box::new(index));
                let batch: Vec<(RecordId, DenseVector)> = (SEED_NODES..SEED_NODES + BATCH)
                    .map(|i| (RecordId::from(i.to_string()), vectors[i].clone()))
                    .collect();
                if phased {
                    wrapper.insert_batch(batch).await.unwrap();
                } else {
                    for (id, v) in batch {
                        wrapper.insert(id, v).await.unwrap();
                    }
                }
                wrapper
            }
        };

        let phased = build(true).await;
        let serial = build(false).await;
        assert_eq!(phased.len().await, serial.len().await);

        // The freshly inserted vectors must be found, and found in the same place.
        let mut agreed = 0;
        for (i, v) in vectors.iter().enumerate().skip(SEED_NODES).take(20) {
            let want = i.to_string();
            let hits = phased.search(v, 5).await.unwrap();
            assert_eq!(
                hits.first().map(|(id, _)| id.as_str()),
                Some(want.as_str()),
                "a vector inserted through the phased path is not its own nearest neighbour"
            );
            let other = serial.search(v, 5).await.unwrap();
            if hits.first().map(|h| &h.0) == other.first().map(|h| &h.0) {
                agreed += 1;
            }
        }
        assert_eq!(agreed, 20, "phased and serial disagreed on a top hit");
    }
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
            .insert("doc1".to_string(), vec![0.1, 0.2, 0.3])
            .await
            .unwrap();

        // Check length
        assert_eq!(wrapper.len().await, 1);

        // Search
        let results = wrapper.search(&vec![0.1, 0.2, 0.3], 10).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_index_wrapper_concurrent_access() {
        let mock_index = Box::new(MockIndex::new());
        let wrapper = IndexWrapper::new(mock_index);

        let wrapper1 = wrapper.clone();
        let wrapper2 = wrapper.clone();

        // Concurrent insert and search
        let insert_handle =
            tokio::spawn(async move { wrapper1.insert("doc1".to_string(), vec![0.1; 128]).await });

        let search_handle = tokio::spawn(async move { wrapper2.search(&vec![0.1; 128], 10).await });

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
