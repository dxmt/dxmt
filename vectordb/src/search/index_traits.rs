use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[async_trait]
pub trait VectorIndex: Send + Sync {
    async fn add(&mut self, id: String, vector: Vec<f32>) -> crate::Result<()>;
    /// Add several vectors at once.
    ///
    /// The default is the loop every caller used to write by hand. Graph indexes
    /// override it: an insert is mostly a read-only search, so a batch can do that half
    /// on every core, which is the difference between building an index on one core and
    /// on all of them.
    async fn add_batch(&mut self, items: Vec<(String, Vec<f32>)>) -> crate::Result<()> {
        for (id, vector) in items {
            self.add(id, vector).await?;
        }
        Ok(())
    }
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
    /// The HNSW graph behind this index, when it is one.
    ///
    /// `IndexWrapper` uses these to hold its write lock for only the part of a batch insert
    /// that needs exclusive access. Searches take the same lock for reading, so holding it
    /// for a whole batch parks every reader for the length of the batch -- 32 ms at a million
    /// nodes, of which 0.25 ms is the appending that actually needs `&mut` (design doc
    /// section 33.1). `None` means the caller falls back to `add_batch`.
    fn as_hnsw(&self) -> Option<&crate::search::hnsw::HnswIndex> {
        None
    }

    /// How many candidates a top-`k` should fetch for the storage layer to rerank against
    /// the stored vectors, or `None` when the index's own scores are exact. An index that
    /// traverses on compressed vectors returns `Some`; its scores rank well enough to find
    /// the neighbourhood and not well enough to order it.
    fn rerank_candidates(&self, _k: usize) -> Option<usize> {
        None
    }

    /// Whether a rebuild would produce a materially different index even though this one
    /// already holds every record -- an HNSW-PQ graph that has not quantized yet, say. The
    /// storage layer's rebuild skips "equivalent" indexes unless this says otherwise.
    fn wants_rebuild(&self) -> bool {
        false
    }

    /// Called by the index wrapper after a batch it inserted through the phased HNSW path
    /// (`as_hnsw_mut`), which bypasses `add_batch`. A wrapper index that has bookkeeping to
    /// do per batch -- HNSW-PQ quantizing itself once it is large enough -- does it here.
    fn after_batch(&mut self) -> crate::Result<()> {
        Ok(())
    }

    fn as_hnsw_mut(&mut self) -> Option<&mut crate::search::hnsw::HnswIndex> {
        None
    }

    async fn delete(&mut self, id: &str) -> crate::Result<()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;

    /// Build the index from a complete vector set in one pass.
    ///
    /// The default inserts one at a time, which is right for graph indexes. Partitioned indexes
    /// override it because they need every vector at once to cluster — feeding them one by one
    /// would leave everything in an overflow buffer and build no partitions at all.
    async fn bulk_build(&mut self, vectors: Vec<(String, Vec<f32>)>) -> crate::Result<()> {
        for (id, vector) in vectors {
            self.add(id, vector).await?;
        }
        Ok(())
    }

    /// Perform incremental upkeep, returning false if the implementation has none to do.
    ///
    /// Partitioned indexes use this to fold buffered writes into their segments and to split or
    /// merge those that have drifted out of range — work proportional to what has changed
    /// rather than to the whole index, which is the difference between maintenance and a
    /// rebuild.
    async fn maintain(&mut self) -> crate::Result<bool> {
        Ok(false)
    }

    /// Object-storage keys this index still needs, or None if it keeps nothing there.
    ///
    /// Used to collect segments that maintenance superseded. They cannot be deleted when they
    /// are superseded, because a snapshot taken earlier still names them — dropping them then
    /// makes that snapshot unable to recover the records they hold.
    fn live_segment_keys(&self) -> Option<Vec<String>> {
        None
    }

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
