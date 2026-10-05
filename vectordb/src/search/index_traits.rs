use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// What an index hands out to be trained outside its lock: a sample of its vectors, the
/// quantizer to fit to it, and -- when the index wants the fast-scan sidecar -- the
/// dimension and metric to fit 16-centroid codebooks for as well.
pub struct QuantizationJob {
    pub sample: Vec<crate::types::vector::DenseVector>,
    pub pq: crate::search::quantization::product_quantization::ProductQuantizer,
    pub fast_scan: Option<(usize, crate::types::vector::DistanceMetric)>,
}

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

    /// Score `ids` against the query without walking the graph, best `k` first, with the
    /// index's own (possibly approximate) distance. `None` when the index has no cheaper way
    /// than a walk to score an arbitrary set -- an exact HNSW would have to read every
    /// vector, which the caller can do itself. A quantized graph scores a code in a few
    /// hundred nanoseconds from memory, so a 1%-selectivity filter over 1M rows is 10,000
    /// scores and one rerank instead of a 30,000-wide walk (docs/architecture/disk-index.md §4, phase 3).
    fn scan_ids(&self, _query: &[f32], _ids: &[&str], _k: usize) -> Option<Vec<(String, f32)>> {
        None
    }

    /// Resolve matching record ids to this index's own node positions, once per query, so
    /// the scan and the filtered walk below test a node by position rather than by hashing
    /// its id. `None` when the index has no positions to speak of (a brute-force scan); the
    /// caller then falls back to the id-set paths above.
    fn node_filter(
        &self,
        _ids: &mut dyn Iterator<Item = &str>,
    ) -> Option<crate::search::node_filter::NodeFilter> {
        None
    }

    /// [`Self::scan_ids`] over a node filter.
    fn scan_nodes(
        &self,
        _query: &[f32],
        _filter: &crate::search::node_filter::NodeFilter,
        _k: usize,
    ) -> Option<Vec<(String, f32)>> {
        None
    }

    /// [`Self::search_with_filter`] with the filter answered per node position by `gate`.
    /// Only called on an index whose positions the caller has resolved (`node_filter` or
    /// `node_id` answered); the default ignores the gate.
    async fn search_with_gate(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        _gate: &dyn crate::search::node_filter::NodeGate,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_with_ef_hint(query, k, ef).await
    }

    /// The record id at node position `idx`, live or tombstoned, for an index that has
    /// positions (`slot_count` counts them); `None` past the end, or always for an index
    /// without positions. What lets a caller keep its own map from positions to records.
    fn node_id(&self, _idx: usize) -> Option<&str> {
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

    /// Give nodes nothing points at an in-edge again; `(stranded, repaired)`. Only a graph
    /// index has anything to repair. See `HnswIndex::repair_reachability`.
    fn repair_reachability(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Whether this index holds a live entry for `id`; `None` when the index cannot say
    /// cheaply, in which case a caller reconciling it against the records leaves it alone.
    fn contains_id(&self, _id: &str) -> Option<bool> {
        None
    }

    /// Slots the index holds, tombstoned ones included, where [`Self::len`] counts only the
    /// live ones.
    ///
    /// An update is a delete and an insert: the graph cannot move a node, because where a node
    /// sits is decided by the vector it had. So an upsert tombstones the old slot and appends a
    /// new one, and a table of constant size under steady upserts grows a slot per write with
    /// nothing reclaiming them. Measured in a 17-hour soak at 200 rows/s: 11.1 million slots
    /// for 400,000 records, a 5.3 GB index where the live rows need about 200 MB, snapshots
    /// grown from 1.2 s to 18.9 s, and 196 s to recover from `kill -9`. The re-evaluation sweep
    /// compares these two numbers and rebuilds when the gap is large enough to be worth it.
    fn slot_count(&self) -> usize {
        self.len()
    }

    /// Whether this index scores from compressed codes rather than from the vectors.
    ///
    /// The difference is visible to a user as recall and as memory: a quantized graph is a
    /// tenth the size and answers approximately, with an exact rerank on top. Which regime a
    /// table is in was not reported anywhere, so "why did recall change at some size" had no
    /// answer short of reading the logs. An index with no compressed form answers `false`
    /// forever, which is the truth for it.
    fn is_quantized(&self) -> bool {
        false
    }

    /// Whether [`Self::quantization_job`] would return work, without paying for the sample.
    ///
    /// `quantization_job` copies up to 65,536 vectors out of the graph -- 200 MB at 768
    /// dimensions -- so the caller needs a way to ask the cheap question on the common path
    /// where the answer is no. Kept separate for that reason alone; an index that answers
    /// `true` here must return `Some` there, barring a race.
    fn quantization_pending(&self) -> bool {
        false
    }

    /// The work this index wants done before it can quantize itself: a training sample and an
    /// untrained quantizer configured for it, or `None` if it does not want quantizing now.
    ///
    /// Both are returned by value so the caller can train with no lock held. That is the whole
    /// point of splitting this out of [`Self::after_batch`]: fitting 192 subspaces to 65,536
    /// vectors takes about a minute, and doing it inside the index's write lock froze every
    /// read and every write on the table for that minute. Measured on
    /// `StreamingPerformanceCase`, which inserts and searches at the same time: search p99
    /// 47.7 s and 200+ insertions backed up, against a table that answers in milliseconds
    /// either side of the stall.
    fn quantization_job(&self) -> Option<QuantizationJob> {
        None
    }

    /// Adopt a quantizer trained by [`Self::quantization_job`]'s output, and the fast-scan
    /// codebooks trained beside it when the job asked for them.
    ///
    /// Must be a no-op if the index quantized itself in the meantime: the sample was taken
    /// under a read lock that was then released, so two callers can arrive here with two
    /// separately trained quantizers and only the first may be installed.
    fn install_quantizer(
        &mut self,
        _pq: crate::search::quantization::product_quantization::ProductQuantizer,
        _fast: Option<crate::search::quantization::fast_scan::FastScan>,
    ) -> crate::Result<()> {
        Ok(())
    }

    /// Whether `scan_nodes` has the block-scan codes (`quantization::fast_scan`), which
    /// make a large match set a flat cost rather than one per matched code.
    fn has_fast_scan(&self) -> bool {
        false
    }

    fn as_hnsw_mut(&mut self) -> Option<&mut crate::search::hnsw::HnswIndex> {
        None
    }

    /// Bytes this index holds in memory -- vectors, links, codes, codebooks -- counted from its
    /// containers. Required rather than defaulted: the default would be 0, which is what
    /// `GET /tables/{name}/stats` reported as `index_size_bytes` for every index until this
    /// existed.
    async fn memory_bytes(&self) -> usize;
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
