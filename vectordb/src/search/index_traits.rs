use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[async_trait]
pub trait VectorIndex: Send + Sync {
    async fn add(&mut self, id: String, vector: Vec<f32>) -> crate::Result<()>;
    async fn search(&self, query: &[f32], k: usize) -> crate::Result<Vec<(String, f32)>>;
    /// Search with an explicit ef_search hint (only meaningful for graph-based indexes like HNSW).
    /// Default implementation ignores the hint and calls `search`.
    async fn search_with_ef_hint(
        &self,
        query: &[f32],
        k: usize,
        _ef: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search(query, k).await
    }
    /// Filter-in-graph search: only return results whose IDs are in `filter_ids`.
    /// Default implementation: search with ef amplification, then post-filter.
    /// HNSW overrides this with true Acorn-style in-graph filtering.
    async fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_ids: &HashSet<String>,
    ) -> crate::Result<Vec<(String, f32)>> {
        // Fallback: amplify k and post-filter
        let amplified_k = (k * 20).max(ef);
        let results = self.search_with_ef_hint(query, amplified_k, ef).await?;
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|(id, _)| filter_ids.contains(id))
            .take(k)
            .collect();
        Ok(filtered)
    }
    async fn delete(&mut self, id: &str) -> crate::Result<()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;

    /// Write this index to `path`, returning false if the implementation has no on-disk form.
    ///
    /// `SerializableIndex::load` is a static constructor, so it cannot live on this trait — the
    /// caller has to know which concrete type to build. Saving has no such problem, and routing
    /// it through here spares the storage layer a downcast on `Box<dyn VectorIndex>`. Returning
    /// false rather than erroring keeps "this index cannot be snapshotted" (BruteForce has
    /// nothing worth storing) distinct from "snapshotting it failed".
    fn snapshot_to(&self, _path: &std::path::Path) -> crate::Result<bool> {
        Ok(false)
    }
}

pub trait SerializableIndex {
    fn save(&self, path: &std::path::Path) -> crate::Result<()>;
    fn load(path: &std::path::Path) -> crate::Result<Self>
    where
        Self: Sized;
    fn stats(&self) -> IndexStats;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_vectors: usize,
    pub index_size_bytes: usize,
    pub build_time_ms: u64,
}
