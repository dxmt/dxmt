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
    /// True while a quantizer is training in a background task, so the next batch does not
    /// start a second one. Training is minutes; batches are milliseconds apart.
    quantizing: Arc<std::sync::atomic::AtomicBool>,
}

impl IndexWrapper {
    /// Create a new index wrapper from a boxed trait object
    pub fn new(index: Box<dyn VectorIndex>) -> Self {
        Self {
            index: Arc::new(RwLock::new(index)),
            structural: Arc::new(tokio::sync::Mutex::new(())),
            quantizing: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Clone the underlying Arc for sharing between async tasks
    pub fn clone_arc(&self) -> Arc<RwLock<Box<dyn VectorIndex>>> {
        Arc::clone(&self.index)
    }

    /// Insert a vector - requires write lock
    pub async fn insert(&self, id: RecordId, vector: DenseVector) -> Result<()> {
        {
            let mut index = self.index.write().await;
            // Convert RecordId to String for trait compatibility
            index.add(id.to_string(), vector).await?;
        }
        // A table fed one row at a time crosses the quantization threshold too, and the row
        // that crosses it must not be the one that pays a minute of exclusive lock.
        self.quantize_if_wanted().await
    }

    /// Rows per lock acquisition on the serial fallback. Small enough that a reader waits
    /// milliseconds, large enough that the acquisitions are a rounding error: 5,000 rows is
    /// eight of them, against 5,000 if each record took the lock.
    const SERIAL_BATCH_CHUNK: usize = 128;

    /// Whether the graph will plan this batch in parallel, asked under a read lock.
    async fn plans_batch_in_parallel(&self, batch: usize) -> bool {
        self.index
            .read()
            .await
            .as_hnsw()
            .is_some_and(|hnsw| hnsw.plans_batches_in_parallel(batch))
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

        // A graph too young to plan a batch in parallel inserts it one row at a time, and does
        // that under whatever lock it is holding. Handing it the whole batch under one write
        // lock is what made a 5,000-row insert into a small table block every search on that
        // table for the full 8.7 s it took -- measured over HTTP, with searches on other tables
        // and `GET /tables` unaffected throughout, so this is that table's index lock and not
        // the map's.
        //
        // Chunked, the same insert leaves a gap between chunks for a reader to take the lock,
        // and the worst search drops to 302 ms. This is not the per-record locking this method
        // replaced: that took the lock two thousand times for a two-thousand-row batch, where
        // this takes it forty.
        if !self.plans_batch_in_parallel(items.len()).await {
            for chunk in items.chunks(Self::SERIAL_BATCH_CHUNK) {
                self.insert_chunk(chunk.to_vec()).await?;
                // Dropping the lock is not enough on its own. tokio's `RwLock` is
                // write-preferring, so this task re-queues for the next chunk before a waiting
                // reader is served, and starves it for the rest of the batch: chunking without
                // this yield took the worst search from 8.0 s only to 3.8 s. Yielding lets a
                // reader that is already waiting take the lock before the next chunk asks.
                tokio::task::yield_now().await;
            }
        } else {
            self.insert_chunk(items).await?;
        }

        // The phased path talks to the inner graph directly, so an index wrapping it never saw
        // the batch; this is its chance. HNSW-PQ quantizes itself here at its row threshold --
        // the 1M acceptance run streamed a million rows through this path as an exact graph and
        // only quantized in the rebuild afterwards.
        self.index.write().await.after_batch()?;
        self.quantize_if_wanted().await
    }

    /// Prepare, plan, append and link one batch, releasing the index between the phases.
    ///
    /// Whether the graph plans in parallel is *its* decision, made inside `prepare_batch`, and
    /// this handles both answers. An earlier version of the chunked loop above called
    /// `prepare_batch` directly and assumed the answer was always "inserted serially, nothing
    /// to apply" -- true only while the graph is below the planner's threshold. A chunk
    /// arriving after it crossed came back with a plan instead, the plan was dropped, and those
    /// rows were in the record store and absent from the graph with nothing said.
    /// `streaming_quantize_test` caught it by passing 64x faster than it had any right to.
    async fn insert_chunk(&self, items: Vec<(String, Vec<f32>)>) -> Result<()> {
        let prepared = {
            let mut index = self.index.write().await;
            let Some(hnsw) = index.as_hnsw_mut() else {
                return index.add_batch(items).await;
            };
            hnsw.prepare_batch(items)?
        };
        // `None`: the graph took the rows one at a time under the lock above.
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
        let index = self.index.read().await;
        index
            .as_hnsw()
            .expect("the structural lock holds the index type still")
            .link_batch(&applied);
        Ok(())
    }

    /// Start quantizing the index, in the background, if it has grown into wanting it.
    ///
    /// Returns as soon as the job is handed off. That is the whole point: this is called at
    /// the end of every batch insert, on the storage layer's single insertion task, and
    /// training 192 subspaces on 65,536 vectors takes minutes. Awaiting it here stops that one
    /// task, so **every** table's inserts stop with it -- the lock is not the only thing that
    /// serialises writes.
    ///
    /// Measured in a soak at 200 rows/s on eight cores: training took 256 s, held no index
    /// lock the whole time (searches were never slowed), and still cost eight batches -- 800
    /// rows -- which sat in the queue past the client's 60 s timeout and were answered by
    /// `Failed to send insert response: insertion result was dropped` when their turn finally
    /// came. Releasing the lock in `54777e8` fixed what searches saw; this is the other half.
    ///
    /// Three properties the background task has to keep:
    ///
    /// * **One at a time.** `quantization_pending` stays true for the whole of training, so
    ///   without the flag every batch would start another job.
    /// * **Off the async pool.** `train` is CPU-bound and synchronous; on a tokio worker it
    ///   blocks that worker for the duration. It goes to `spawn_blocking`.
    /// * **Installed like a structural change.** The install takes the same `structural` lock
    ///   a phased insert and a `replace` take, so it cannot land between an insert's planning
    ///   and its application.
    ///
    /// Rows inserted while the training ran are in the graph by then and get encoded with the
    /// rest, so nothing has to be replayed.
    pub async fn quantize_if_wanted(&self) -> Result<()> {
        use std::sync::atomic::Ordering;

        // The cheap question first: `quantization_job` copies the sample out, which is 200 MB
        // at 768 dimensions, and the answer is no on all but one batch in a table's life.
        if !self.index.read().await.quantization_pending() {
            return Ok(());
        }
        if self.quantizing.swap(true, Ordering::SeqCst) {
            return Ok(()); // one is already training
        }

        let index = Arc::clone(&self.index);
        let structural = Arc::clone(&self.structural);
        let flag = Arc::clone(&self.quantizing);
        tokio::spawn(async move {
            let result = Self::quantize_now(&index, &structural).await;
            flag.store(false, Ordering::SeqCst);
            if let Err(e) = result {
                tracing::warn!("HNSW-PQ: quantization failed, graph stays exact: {}", e);
            }
        });
        Ok(())
    }

    /// Sample, train, install. Split out so the spawned task and the tests share it.
    async fn quantize_now(
        index: &Arc<RwLock<Box<dyn VectorIndex>>>,
        structural: &Arc<tokio::sync::Mutex<()>>,
    ) -> Result<()> {
        let Some((sample, mut pq)) = index.read().await.quantization_job() else {
            return Ok(());
        };
        let started = std::time::Instant::now();
        // No lock held, not on a runtime worker, and -- the part that took two attempts -- not
        // on the rayon pool the insert path needs.
        //
        // `ProductQuantizer::train` is `(0..num_subvectors).into_par_iter()`, which rayon
        // splits across the workers of the *global* pool and then runs without a yield point:
        // 192 subspaces over eight threads is 24 k-means runs per worker, back to back, and a
        // job submitted from outside is only picked up when a worker finishes its whole chunk.
        // `HnswIndex::plan_batch` submits exactly such a job above `PARALLEL_PLAN_MIN_NODES`,
        // so every insert queued behind the training even though the training held no lock and
        // no runtime worker. Measured in a soak on eight cores: acknowledged writes stopped
        // dead for the 256 s the training ran, and the WAL sequence did not move.
        //
        // Half the cores, so serving keeps the rest. Training takes correspondingly longer;
        // it happens once in a table's life, and the alternative is a table that accepts
        // nothing while it runs.
        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(1);
        let pq = tokio::task::spawn_blocking(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("pq-train-{i}"))
                .build()
                .map_err(|e| crate::Error::Internal(format!("PQ training pool: {e}")))?;
            pool.install(|| pq.train(&sample, 25))?;
            Ok::<_, crate::Error>(pq)
        })
        .await
        .map_err(|e| crate::Error::Internal(format!("PQ training task failed: {e}")))??;
        let trained = started.elapsed();
        let installing = std::time::Instant::now();
        let _structural = structural.lock().await;
        index.write().await.install_quantizer(pq)?;
        tracing::info!(
            "HNSW-PQ: quantizer trained in {:.1}s in the background, installed in {:.1}s",
            trained.as_secs_f64(),
            installing.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Whether a background quantization is in flight. For tests and diagnostics.
    /// Whether the index scores from codes rather than from vectors right now.
    ///
    /// Reported in table stats: an HNSW-PQ index is exact until it has trained, and which side
    /// of that a table is on decides both its recall and its memory.
    pub async fn is_quantized(&self) -> bool {
        self.index.read().await.is_quantized()
    }

    pub fn is_quantizing(&self) -> bool {
        self.quantizing.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Search for nearest neighbors - requires read lock
    pub async fn search(&self, query: &DenseVector, k: usize) -> Result<Vec<(RecordId, f32)>> {
        let index = self.index.read().await;
        Self::top_k(&**index, query, k, None, None).await
    }

    /// Search with an explicit ef_search hint (HNSW respects this; other indexes fall back to search).
    pub async fn search_with_ef_hint(
        &self,
        query: &DenseVector,
        k: usize,
        ef: usize,
    ) -> Result<Vec<(RecordId, f32)>> {
        let index = self.index.read().await;
        Self::top_k(&**index, query, k, Some(ef), None).await
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
        Self::top_k(&**index, query, k, Some(ef), Some(filter_ids)).await
    }

    /// The index's best `k`, with a tie at the `k`-th distance broken by id.
    ///
    /// An index cuts to `k` itself. When several records sit at exactly the `k`-th distance it
    /// keeps whichever it met first -- walk order for a graph, insertion order for a scan -- and
    /// nothing that orders the results afterwards can bring back the ones it dropped. The storage
    /// layer's tie-break by id (`25e5515`) was therefore decided by build order: over HNSW-PQ a
    /// query whose tied neighbours were `t00`..`t19` got `["exact", "t11", "t12"]`, and the HTTP
    /// suite saw r1's fifth neighbour change after a rebuild.
    ///
    /// So every search asks for one more than it needs. When that one is farther, the cutoff is
    /// not tied and the answer stands -- nearly always, and for a graph at no extra cost, since
    /// the walk keeps `max(ef, k)` candidates either way. When it is not, the question is asked
    /// again with a wider `k` until the cutoff settles or the index has nothing more to give.
    async fn top_k(
        index: &dyn VectorIndex,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        filter_ids: Option<&std::collections::HashSet<String>>,
    ) -> Result<Vec<(RecordId, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let rows = index.len();
        let cap = |want: usize| if rows > 0 { want.min(rows) } else { want };
        let mut want = cap(k.saturating_add(1));
        loop {
            let mut results = match (ef, filter_ids) {
                (_, Some(ids)) => {
                    index
                        .search_with_filter(query, want, ef.unwrap_or(want), ids)
                        .await?
                }
                (Some(ef), None) => index.search_with_ef_hint(query, want, ef).await?,
                (None, None) => index.search(query, want).await?,
            };
            results.sort_by(Self::nearest_first);
            let settled = results.len() <= k
                || results.len() < want
                || want >= rows
                || results[k].1 > results[k - 1].1;
            if settled {
                results.truncate(k);
                return Ok(results);
            }
            want = cap(want.saturating_mul(4));
        }
    }

    /// Nearest first; among equal distances the smaller id; a NaN distance after every number.
    fn nearest_first(a: &(RecordId, f32), b: &(RecordId, f32)) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match a.1.partial_cmp(&b.1) {
            Some(Ordering::Equal) => a.0.cmp(&b.0),
            Some(ordering) => ordering,
            None => a.1.is_nan().cmp(&b.1.is_nan()).then_with(|| a.0.cmp(&b.0)),
        }
    }

    /// See `VectorIndex::rerank_candidates`.
    pub async fn rerank_candidates(&self, k: usize) -> Option<usize> {
        self.index.read().await.rerank_candidates(k)
    }

    /// See `VectorIndex::scan_ids`.
    ///
    /// Borrowed, not owned: a selective filter hands this every matching id, and at 100,000
    /// matches copying them into a `Vec<String>` first was 100,000 heap allocations a query.
    pub async fn scan_ids(
        &self,
        query: &DenseVector,
        ids: &[&str],
        k: usize,
    ) -> Option<Vec<(RecordId, f32)>> {
        self.index.read().await.scan_ids(query, ids, k)
    }

    /// See `VectorIndex::wants_rebuild`.
    pub async fn wants_rebuild(&self) -> bool {
        self.index.read().await.wants_rebuild()
    }

    /// Remove a vector - requires write lock
    pub async fn remove(&self, id: &RecordId) -> Result<bool> {
        let mut index = self.index.write().await;
        index.delete(&id.to_string()).await?;
        Ok(true)
    }

    /// Live entries and the slots holding them: `(live, slots)`. Equal for an index that
    /// reclaims what it deletes, and far apart for a graph under steady upserts, where every
    /// update leaves a tombstone behind. See `VectorIndex::slot_count`.
    pub async fn live_and_slots(&self) -> (usize, usize) {
        let index = self.index.read().await;
        (index.len(), index.slot_count())
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
        IndexStats {
            total_vectors: index.len(),
            // A literal 0 for every index until `VectorIndex::memory_bytes` existed, so table
            // stats reported an index of no size and left it out of `memory_usage_bytes`.
            index_size_bytes: index.memory_bytes().await,
            build_time_ms: 0,
        }
    }
}

impl Clone for IndexWrapper {
    fn clone(&self) -> Self {
        Self {
            index: Arc::clone(&self.index),
            structural: Arc::clone(&self.structural),
            // Shared, not reset: a clone of the wrapper is another handle on the same index,
            // and two handles must not each start a training job for it.
            quantizing: Arc::clone(&self.quantizing),
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

    /// A batch into a young graph must not lock every reader out for its whole duration.
    ///
    /// Below `PARALLEL_PLAN_MIN_NODES` a batch is inserted one row at a time, and handing the
    /// whole batch to that path under one write lock is what made a 5,000-row insert into a
    /// small table block searches on it for the full 8.0 s the insert took -- measured over
    /// HTTP on 2026-09-25, with `GET /tables` and searches on other tables unaffected, so it
    /// was this lock and not the table map's.
    ///
    /// What is asserted is a *ratio*, not a latency: the longest gap between two completed
    /// searches has to be a fraction of how long the insert took. A millisecond threshold would
    /// measure the machine; a ratio measures the locking. The first version of this test
    /// asserted only that some search completed, and passed with the whole batch under one
    /// lock -- the reader gets its turn while the writer *waits* for the lock, which proves
    /// nothing about what happens once the writer has it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reads_get_through_while_a_batch_goes_into_a_young_graph() {
        use crate::search::hnsw::HnswIndex;
        use crate::types::vector::DistanceMetric;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};
        use std::time::Instant;

        const DIM: usize = 64;
        // Small enough to stay on the serial path (the threshold is 10,000 nodes), large
        // enough that the insert takes long enough for the gaps to mean something.
        const SEED: usize = 500;
        const BATCH: usize = 2_000;

        let mut state = 0x5DEE_CE66_D123_4567_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        };
        let vector =
            |n: &mut dyn FnMut() -> f32| -> DenseVector { (0..DIM).map(|_| n()).collect() };

        let mut index = HnswIndex::new(DistanceMetric::Cosine, None);
        for i in 0..SEED {
            index.insert(i.to_string(), vector(&mut next)).unwrap();
        }
        let wrapper = Arc::new(IndexWrapper::new(Box::new(index)));
        let batch: Vec<(RecordId, DenseVector)> = (SEED..SEED + BATCH)
            .map(|i| (RecordId::from(i.to_string()), vector(&mut next)))
            .collect();

        let marks: Arc<Mutex<Vec<Instant>>> = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let reader = {
            let wrapper = Arc::clone(&wrapper);
            let marks = Arc::clone(&marks);
            let stop = Arc::clone(&stop);
            let probe: DenseVector = vec![0.1; DIM];
            tokio::spawn(async move {
                while !stop.load(Ordering::SeqCst) {
                    if wrapper.search(&probe, 5).await.is_ok() {
                        marks.lock().expect("not poisoned").push(Instant::now());
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            })
        };
        // Let the reader establish itself, so the first gap is not the test's own start-up.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let started = Instant::now();
        wrapper.insert_batch(batch).await.unwrap();
        let insert_took = started.elapsed();
        stop.store(true, Ordering::SeqCst);
        reader.await.unwrap();

        let marks = marks.lock().expect("not poisoned").clone();
        let during: Vec<Instant> = marks.into_iter().filter(|m| *m >= started).collect();
        assert!(
            !during.is_empty(),
            "no search completed at all while {BATCH} rows went in"
        );
        let mut longest = during[0].duration_since(started);
        for pair in during.windows(2) {
            longest = longest.max(pair[1].duration_since(pair[0]));
        }
        assert!(
            longest.as_secs_f64() < insert_took.as_secs_f64() * 0.5,
            "searches stopped for {:.2?} of a {:.2?} insert -- the write lock is being held \
             across the batch rather than per chunk",
            longest,
            insert_took
        );
    }

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

        async fn memory_bytes(&self) -> usize {
            self.vectors
                .iter()
                .map(|(id, v)| id.len() + v.len() * std::mem::size_of::<f32>())
                .sum()
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
    async fn a_tie_at_the_cutoff_is_broken_by_id_whatever_order_the_index_returns() {
        // MockIndex scores every record 0.9 and hands them back in HashMap order, cut to `k`: a
        // tie across the whole table, the worst case of an index that keeps whichever tied
        // records it meets first.
        let wrapper = IndexWrapper::new(Box::new(MockIndex::new()));
        for i in (0..16).rev() {
            wrapper
                .insert(format!("t{i:02}"), vec![0.0; 2])
                .await
                .unwrap();
        }
        let query = vec![0.0; 2];
        let expected = ["t00", "t01", "t02"];
        let ids = |results: Vec<(RecordId, f32)>| {
            results.into_iter().map(|(id, _)| id).collect::<Vec<_>>()
        };

        assert_eq!(
            ids(wrapper.search(&query, 3).await.unwrap()),
            expected,
            "search"
        );
        assert_eq!(
            ids(wrapper.search_with_ef_hint(&query, 3, 64).await.unwrap()),
            expected,
            "search_with_ef_hint"
        );
        let every_id: std::collections::HashSet<String> =
            (0..16).map(|i| format!("t{i:02}")).collect();
        assert_eq!(
            ids(wrapper
                .search_with_filter(&query, 3, 64, &every_id)
                .await
                .unwrap()),
            expected,
            "search_with_filter"
        );
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

#[cfg(test)]
mod after_batch_tests {
    use super::*;
    use crate::search::hnsw_pq::{HnswPqConfig, HnswPqIndex};
    use crate::types::DistanceMetric;

    /// The wrapper's phased insert reaches the inner graph through `as_hnsw_mut`; the
    /// index around it must still get to quantize itself at its threshold.
    #[tokio::test]
    async fn a_phased_batch_insert_lets_hnsw_pq_quantize_itself() {
        let config = HnswPqConfig {
            num_subvectors: 16,
            num_centroids: 32,
            quantize_at_rows: 1500,
            ..HnswPqConfig::default()
        };
        let index = HnswPqIndex::new(64, DistanceMetric::Cosine, Some(config)).unwrap();
        let wrapper = IndexWrapper::new(Box::new(index));

        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        };
        let batch = |from: usize,
                     to: usize,
                     next: &mut dyn FnMut() -> f32|
         -> Vec<(RecordId, DenseVector)> {
            (from..to)
                .map(|i| (i.to_string(), (0..64).map(|_| next()).collect()))
                .collect()
        };
        wrapper
            .insert_batch(batch(0, 1000, &mut next))
            .await
            .unwrap();
        assert!(
            wrapper.wants_rebuild().await,
            "exact and past the training minimum"
        );
        wrapper
            .insert_batch(batch(1000, 2000, &mut next))
            .await
            .unwrap();
        // The crossing insert hands the training to a background task and returns, so the
        // index is not quantized the instant it comes back -- that is the point of the
        // handoff, and asserting otherwise is asserting that writers wait for k-means. What
        // must still hold is that the quantizing happens.
        let waited = std::time::Instant::now();
        while wrapper.is_quantizing() {
            assert!(
                waited.elapsed() < std::time::Duration::from_secs(60),
                "the background quantization never finished"
            );
            tokio::task::yield_now().await;
        }
        assert!(
            !wrapper.wants_rebuild().await,
            "crossing the threshold through the phased path must quantize"
        );
        assert_eq!(wrapper.rerank_candidates(10).await, Some(100));
        assert_eq!(wrapper.len().await, 2000);
    }
}
