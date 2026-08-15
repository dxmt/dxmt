//! An IVF index whose partitions live in object storage and load on demand.
//!
//! Table-level eviction bounds memory by the working set of *tables*. It does nothing for a
//! single table larger than memory — that table is either fully resident or fully gone. This
//! index is the answer to that case: the centroids stay in memory (nlist × dim floats, a few
//! hundred kilobytes for typical settings) and each partition is an immutable segment fetched
//! only when a query probes it.
//!
//! Why IVF rather than a partially-resident HNSW: graph traversal jumps to arbitrary neighbours,
//! so serving it from object storage means a network round trip per hop. A centroid layout
//! decides which partitions matter *before* reading any vectors, which is why Turbopuffer and
//! DiskANN-style systems are built this way.
//!
//! Writes land in an in-memory overflow buffer that every search also scans, so a partitioned
//! table stays writable. [`PartitionedIvfIndex::rebuild`] folds the buffer back into partitions.

use crate::search::index_traits::{IndexStats, SerializableIndex, VectorIndex};
use crate::search::simd_distance::SimdDistance;
use crate::storage::object::traits::ObjectStorage;
use crate::types::vector::{DenseVector, DistanceMetric, RecordId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionedIvfConfig {
    /// Number of centroids, and therefore of partitions.
    pub nlist: usize,
    /// Partitions probed per query.
    pub nprobe: usize,
    /// How many partitions may stay resident at once. This is the memory bound.
    pub max_resident_partitions: usize,
    /// k-means iterations when building.
    pub max_iterations: usize,
    /// A partition longer than this splits in two.
    pub split_threshold: usize,
    /// A partition shorter than this merges into its nearest neighbour.
    pub merge_threshold: usize,
}

impl Default for PartitionedIvfConfig {
    fn default() -> Self {
        Self {
            nlist: 64,
            nprobe: 8,
            // Enough to hold a full probe set plus a little reuse across queries. Smaller than
            // nlist on purpose: if every partition can be resident there is no tiering.
            max_resident_partitions: 16,
            max_iterations: 20,
            // SPANN's defaults, as shipped in chroma: a posting list splits at 100 and merges
            // below 50. The gap matters — split and merge thresholds that meet would make a
            // partition oscillate between the two on every maintenance pass.
            split_threshold: 100,
            merge_threshold: 50,
        }
    }
}

/// 一次 [`PartitionedIvfIndex::maintain`] 做了什么。
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MaintenanceReport {
    /// 从溢出缓冲折进分区的向量数。
    pub absorbed: usize,
    /// 分裂的分区数。
    pub split: usize,
    /// 合并掉的分区数。
    pub merged: usize,
    /// 实际重写的分区数——增量维护的意义就在这个数远小于分区总数。
    pub rewritten: usize,
    /// 顺带清掉的墓碑记录数。
    pub purged: usize,
}

/// One partition's vectors, as stored in object storage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartitionData {
    vectors: Vec<(RecordId, DenseVector)>,
}

/// What the index keeps about a partition regardless of residency.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionMeta {
    /// Stable identity, never reused. Segment keys are built from this rather than from the
    /// position in `partitions`, because merging removes an entry and would otherwise renumber
    /// every segment after it — turning a local edit into a full rewrite.
    id: u64,
    centroid: DenseVector,
    /// Vector count, so `len()` and planning work without loading anything.
    count: usize,
}

/// The parts worth writing to disk. Resident partition data is deliberately excluded — it is a
/// cache of object storage, not state.
#[derive(Debug, Serialize, Deserialize)]
struct PartitionedIvfState {
    config: PartitionedIvfConfig,
    distance_metric: DistanceMetric,
    dimension: Option<usize>,
    partitions: Vec<PartitionMeta>,
    /// Key prefix under which this index's segments live.
    prefix: String,
    /// Vectors added since the last rebuild.
    overflow: Vec<(RecordId, DenseVector)>,
    /// Tombstones, applied to whatever a partition yields on load.
    ///
    /// A set, not a list: this is consulted once per candidate on every query, so a linear
    /// membership test makes each query O(candidates x tombstones). Measured at 10,000
    /// tombstones, ten queries took 19.7s. weaviate keeps its tombstones in a hash map for the
    /// same reason and has since gone further, caching a snapshot for the read path.
    deleted: std::collections::HashSet<RecordId>,
    /// Next partition id to hand out.
    next_partition_id: u64,
}

pub struct PartitionedIvfIndex {
    state: PartitionedIvfState,
    /// Where partitions are fetched from. `None` means everything must already be resident,
    /// which is the case in tests that never offload.
    store: Option<Arc<dyn ObjectStorage + Send + Sync>>,
    /// Loaded partitions, keyed by id. Bounded by `max_resident_partitions`.
    ///
    /// The value is an `Arc` so a cache hit hands back a refcount rather than a copy of the
    /// partition. Cloning it per probe copies nprobe x partition_size x dimension floats on
    /// every query — at nprobe 8, 100-vector partitions and 1536 dimensions that is ~5 MB per
    /// query, for data that is already in memory.
    resident: Arc<RwLock<HashMap<u64, Arc<PartitionData>>>>,
    /// Last-use tick per resident partition, for eviction.
    ///
    /// A counter rather than an ordered list: keeping a `Vec` in LRU order costs a linear scan
    /// and a memmove on *every* access, while a tick is an O(1) write and the scan for a victim
    /// happens only when something has to be evicted.
    last_used: Arc<RwLock<HashMap<u64, u64>>>,
    /// Monotonic source for `last_used`.
    clock: Arc<std::sync::atomic::AtomicU64>,
}

impl PartitionedIvfIndex {
    pub fn new(
        distance_metric: DistanceMetric,
        prefix: impl Into<String>,
        config: Option<PartitionedIvfConfig>,
    ) -> Self {
        Self {
            state: PartitionedIvfState {
                config: config.unwrap_or_default(),
                distance_metric,
                dimension: None,
                partitions: Vec::new(),
                prefix: prefix.into(),
                overflow: Vec::new(),
                deleted: std::collections::HashSet::new(),
                next_partition_id: 0,
            },
            store: None,
            resident: Arc::new(RwLock::new(HashMap::new())),
            last_used: Arc::new(RwLock::new(HashMap::new())),
            clock: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Attach the object store partitions are fetched from and written to.
    pub fn with_store(mut self, store: Arc<dyn ObjectStorage + Send + Sync>) -> Self {
        self.store = Some(store);
        self
    }

    /// How many partitions are currently in memory.
    ///
    /// Async rather than a `try_read` that falls back to zero: tests assert this is zero right
    /// after a rebuild, and a lock-contention fallback would make that assertion pass for the
    /// wrong reason.
    pub async fn resident_partition_count(&self) -> usize {
        self.resident.read().await.len()
    }

    pub fn partition_count(&self) -> usize {
        self.state.partitions.len()
    }

    fn partition_key(&self, id: u64) -> String {
        format!("{}/partition-{:06}.seg", self.state.prefix, id)
    }

    /// Rebuild every partition from `vectors`, writing each to object storage.
    ///
    /// This is the compaction step. Doing it on the write path instead would mean rewriting a
    /// whole segment per insert, which is why writes go to the overflow buffer and land here in
    /// bulk.
    pub async fn rebuild(&mut self, vectors: Vec<(RecordId, DenseVector)>) -> crate::Result<()> {
        if vectors.is_empty() {
            self.state.partitions.clear();
            self.state.overflow.clear();
            self.state.deleted.clear();
            self.resident.write().await.clear();
            self.last_used.write().await.clear();
            return Ok(());
        }

        let dim = vectors[0].1.len();
        self.state.dimension = Some(dim);

        // Never ask for more clusters than there are vectors; k-means would emit empty centroids
        // and every query would probe partitions that cannot contain anything.
        let nlist = self.state.config.nlist.min(vectors.len()).max(1);
        let centroids = kmeans(
            &vectors,
            nlist,
            self.state.config.max_iterations,
            self.state.distance_metric,
        );

        let mut buckets: Vec<PartitionData> = vec![PartitionData::default(); centroids.len()];
        for (id, vector) in vectors {
            let i = nearest_centroid(&centroids, &vector, self.state.distance_metric);
            buckets[i].vectors.push((id, vector));
        }

        self.state.partitions = centroids
            .iter()
            .zip(buckets.iter())
            .map(|(c, b)| {
                let id = self.state.next_partition_id;
                self.state.next_partition_id += 1;
                PartitionMeta {
                    id,
                    centroid: c.clone(),
                    count: b.vectors.len(),
                }
            })
            .collect();

        if let Some(ref store) = self.store {
            for (meta, bucket) in self.state.partitions.iter().zip(buckets.iter()) {
                let bytes = bincode::serialize(bucket)
                    .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                store
                    .put_object(
                        &self.partition_key(meta.id),
                        bytes::Bytes::from(bytes),
                        Some("application/octet-stream"),
                    )
                    .await
                    .map_err(|e| {
                        crate::Error::Internal(format!("writing partition {}: {}", meta.id, e))
                    })?;
            }
            // Everything is durable in object storage now, so drop the local copies and let
            // queries pull back only what they probe. Keeping them would defeat the purpose.
            self.resident.write().await.clear();
            self.last_used.write().await.clear();
        } else {
            // No store: the partitions have nowhere else to live, so they stay resident.
            let mut resident = self.resident.write().await;
            let mut last_used = self.last_used.write().await;
            resident.clear();
            last_used.clear();
            for (i, (meta, bucket)) in self.state.partitions.iter().zip(buckets).enumerate() {
                resident.insert(meta.id, Arc::new(bucket));
                last_used.insert(meta.id, i as u64);
            }
        }

        self.state.overflow.clear();
        self.state.deleted.clear();
        Ok(())
    }

    /// 把溢出缓冲折进分区，并把过大的分裂、过小的合并。
    ///
    /// 这是 [`Self::rebuild`] 的增量替代品，也是它存在的理由：全量重建要跑一遍 k-means，
    /// 代价 O(n · nlist · iterations)。千万行的表——正是这个索引为之设计的规模——每次维护都
    /// 重跑一遍是不可行的。这里只碰被写入影响到的那几个分区，代价与**新增量**成正比，
    /// 而不是与总量成正比。
    ///
    /// 阈值取自 SPANN（chroma 的实现用 split=100 / merge=50）：一个列表长过 `split_threshold`
    /// 就一分为二，短于 `merge_threshold` 就并进最近的邻居。
    pub async fn maintain(&mut self) -> crate::Result<MaintenanceReport> {
        let mut report = MaintenanceReport::default();

        if self.state.partitions.is_empty() {
            // Nothing to maintain — the caller wants rebuild(), not this.
            return Ok(report);
        }

        // 1. Assign every buffered vector to its nearest centroid.
        let overflow = std::mem::take(&mut self.state.overflow);
        report.absorbed = overflow.len();

        let mut touched: HashMap<u64, PartitionData> = HashMap::new();
        for (id, vector) in overflow {
            let pid = self.nearest_partition(&vector);
            let data = match touched.entry(pid) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    // maintain() rewrites partitions, so it needs its own copy.
                    e.insert((*self.load_partition(pid).await?).clone())
                }
            };
            data.vectors.retain(|(existing, _)| existing != &id);
            data.vectors.push((id, vector));
        }

        // 2. Apply tombstones to the partitions we are already rewriting. The rest keep theirs
        //    until something else brings them here — a tombstone costs a comparison at search
        //    time, which is cheaper than fetching a segment to remove one row from it.
        if !self.state.deleted.is_empty() {
            let deleted = &self.state.deleted;
            for data in touched.values_mut() {
                let before = data.vectors.len();
                data.vectors.retain(|(id, _)| !deleted.contains(id));
                report.purged += before - data.vectors.len();
            }
        }

        // 3. Split what has grown too large.
        let split_threshold = self.state.config.split_threshold;
        let mut split_from: Vec<u64> = touched
            .iter()
            .filter(|(_, d)| d.vectors.len() > split_threshold)
            .map(|(pid, _)| *pid)
            .collect();
        split_from.sort_unstable();

        for pid in split_from {
            let Some(data) = touched.remove(&pid) else {
                continue;
            };
            let metric = self.state.distance_metric;
            let centroids = kmeans(&data.vectors, 2, self.state.config.max_iterations, metric);

            let mut halves = vec![PartitionData::default(), PartitionData::default()];
            for (id, vector) in data.vectors {
                let side = nearest_centroid(&centroids, &vector, metric);
                halves[side].vectors.push((id, vector));
            }

            // A split that puts everything on one side has not split anything; keeping the
            // partition whole is better than writing an empty segment and probing it forever.
            if halves.iter().any(|h| h.vectors.is_empty()) {
                let merged = PartitionData {
                    vectors: halves.into_iter().flat_map(|h| h.vectors).collect(),
                };
                touched.insert(pid, merged);
                continue;
            }

            self.remove_partition(pid, &mut touched).await;
            for (centroid, half) in centroids.into_iter().zip(halves) {
                let new_id = self.state.next_partition_id;
                self.state.next_partition_id += 1;
                self.state.partitions.push(PartitionMeta {
                    id: new_id,
                    centroid,
                    count: half.vectors.len(),
                });
                touched.insert(new_id, half);
            }
            report.split += 1;
        }

        // 4. Merge what has shrunk too far. Only partitions we already hold, for the same
        //    reason as the tombstones above.
        let merge_threshold = self.state.config.merge_threshold;
        let mut merge_from: Vec<u64> = touched
            .iter()
            .filter(|(_, d)| d.vectors.len() < merge_threshold)
            .map(|(pid, _)| *pid)
            .collect();
        merge_from.sort_unstable();

        for pid in merge_from {
            if self.state.partitions.len() <= 1 {
                break;
            }
            let Some(data) = touched.remove(&pid) else {
                continue;
            };
            let Some(target) = self.nearest_other_partition(pid) else {
                touched.insert(pid, data);
                continue;
            };

            let into = match touched.entry(target) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert((*self.load_partition(target).await?).clone())
                }
            };
            into.vectors.extend(data.vectors);
            self.remove_partition(pid, &mut touched).await;
            report.merged += 1;
        }

        // 5. Persist only what changed, and refresh the counts the planner reads.
        for (pid, data) in &touched {
            if let Some(meta) = self.state.partitions.iter_mut().find(|m| m.id == *pid) {
                meta.count = data.vectors.len();
            }
            self.write_partition(*pid, data).await?;
        }
        report.rewritten = touched.len();

        // Tombstones that survive here still apply to partitions we did not touch.
        Ok(report)
    }

    /// 最近质心所属分区的 id。
    fn nearest_partition(&self, vector: &[f32]) -> u64 {
        let metric = self.state.distance_metric;
        self.state
            .partitions
            .iter()
            .map(|p| (p.id, distance(metric, vector, &p.centroid)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
            .unwrap_or(0)
    }

    /// 除 `pid` 之外质心最近的分区。
    fn nearest_other_partition(&self, pid: u64) -> Option<u64> {
        let metric = self.state.distance_metric;
        let me = self.state.partitions.iter().find(|p| p.id == pid)?;
        self.state
            .partitions
            .iter()
            .filter(|p| p.id != pid)
            .map(|p| (p.id, distance(metric, &me.centroid, &p.centroid)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    /// 从元数据、常驻缓存和对象存储里彻底移除一个分区。
    async fn remove_partition(&mut self, pid: u64, touched: &mut HashMap<u64, PartitionData>) {
        self.state.partitions.retain(|p| p.id != pid);
        touched.remove(&pid);
        self.resident.write().await.remove(&pid);
        self.last_used.write().await.remove(&pid);

        if let Some(ref store) = self.store {
            // A leaked segment costs storage, not correctness — nothing references it once the
            // metadata is gone.
            let _ = store.delete_object(&self.partition_key(pid)).await;
        }
    }

    /// 写一个分区，并把它放进常驻缓存。
    async fn write_partition(&self, pid: u64, data: &PartitionData) -> crate::Result<()> {
        if let Some(ref store) = self.store {
            let bytes =
                bincode::serialize(data).map_err(|e| crate::Error::Serialization(e.to_string()))?;
            store
                .put_object(
                    &self.partition_key(pid),
                    bytes::Bytes::from(bytes),
                    Some("application/octet-stream"),
                )
                .await
                .map_err(|e| crate::Error::Internal(format!("writing partition {}: {}", pid, e)))?;
        }
        self.admit(pid, Arc::new(data.clone())).await;
        Ok(())
    }

    /// Make partition `id` resident, fetching it if necessary, and return its vectors.
    async fn load_partition(&self, id: u64) -> crate::Result<Arc<PartitionData>> {
        let hit = self.resident.read().await.get(&id).cloned();
        if let Some(data) = hit {
            self.touch(id).await;
            return Ok(data);
        }

        let Some(ref store) = self.store else {
            // Without a store an absent partition is simply empty; this is the in-memory mode.
            return Ok(Arc::new(PartitionData::default()));
        };

        let bytes = store
            .get_object(&self.partition_key(id))
            .await
            .map_err(|e| crate::Error::Internal(format!("fetching partition {}: {}", id, e)))?;
        let data: PartitionData =
            bincode::deserialize(&bytes).map_err(|e| crate::Error::Serialization(e.to_string()))?;

        debug!("Loaded partition {} ({} vectors)", id, data.vectors.len());
        let data = Arc::new(data);
        self.admit(id, data.clone()).await;
        Ok(data)
    }

    async fn touch(&self, id: u64) {
        let tick = self
            .clock
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.last_used.write().await.insert(id, tick);
    }

    /// Insert a partition, evicting the least recently used one if we are at the cap.
    async fn admit(&self, id: u64, data: Arc<PartitionData>) {
        let mut resident = self.resident.write().await;
        let mut last_used = self.last_used.write().await;

        while resident.len() >= self.state.config.max_resident_partitions {
            let Some(victim) = last_used
                .iter()
                .filter(|(pid, _)| resident.contains_key(*pid))
                .min_by_key(|(_, tick)| **tick)
                .map(|(pid, _)| *pid)
            else {
                break;
            };
            resident.remove(&victim);
            last_used.remove(&victim);
        }

        let tick = self
            .clock
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        resident.insert(id, data);
        last_used.insert(id, tick);
    }
}

#[async_trait]
impl VectorIndex for PartitionedIvfIndex {
    async fn add(&mut self, id: String, vector: Vec<f32>) -> crate::Result<()> {
        if self.state.dimension.is_none() {
            self.state.dimension = Some(vector.len());
        }
        // Writes buffer rather than rewriting a segment each time. rebuild() folds them in.
        self.state.overflow.retain(|(existing, _)| existing != &id);
        self.state.deleted.remove(&id);
        self.state.overflow.push((id, vector));
        Ok(())
    }

    async fn bulk_build(&mut self, vectors: Vec<(String, Vec<f32>)>) -> crate::Result<()> {
        self.rebuild(vectors).await
    }

    async fn maintain(&mut self) -> crate::Result<bool> {
        let report = PartitionedIvfIndex::maintain(self).await?;
        Ok(report != MaintenanceReport::default())
    }

    async fn search(&self, query: &[f32], k: usize) -> crate::Result<Vec<(String, f32)>> {
        let metric = self.state.distance_metric;
        let mut hits: Vec<(String, f32)> = Vec::new();

        // Rank partitions by centroid distance and probe only the closest. This is the whole
        // point: the decision about what to read is made from centroids alone.
        let mut ranked: Vec<(u64, f32)> = self
            .state
            .partitions
            .iter()
            .map(|p| (p.id, distance(metric, query, &p.centroid)))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (pid, _) in ranked.into_iter().take(self.state.config.nprobe) {
            match self.load_partition(pid).await {
                Ok(data) => {
                    for (id, vector) in &data.vectors {
                        if self.state.deleted.contains(id) {
                            continue;
                        }
                        hits.push((id.clone(), distance(metric, query, vector)));
                    }
                }
                // A partition that cannot be read costs recall, not correctness. Losing the
                // whole query because one segment is unavailable would be worse.
                Err(e) => warn!(
                    "Partition {} unavailable, results may be incomplete: {}",
                    pid, e
                ),
            }
        }

        // Anything written since the last rebuild is only in the buffer.
        for (id, vector) in &self.state.overflow {
            hits.push((id.clone(), distance(metric, query, vector)));
        }

        hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        hits.dedup_by(|a, b| a.0 == b.0);
        hits.truncate(k);
        Ok(hits)
    }

    async fn delete(&mut self, id: &str) -> crate::Result<()> {
        self.state.overflow.retain(|(existing, _)| existing != id);
        // Tombstone rather than rewrite: the vector may sit in a segment that is not even
        // resident. rebuild() drops it for good.
        self.state.deleted.insert(id.to_string());
        Ok(())
    }

    fn len(&self) -> usize {
        let stored: usize = self.state.partitions.iter().map(|p| p.count).sum();
        (stored + self.state.overflow.len()).saturating_sub(self.state.deleted.len())
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn snapshot_to(&self, path: &std::path::Path) -> crate::Result<bool> {
        SerializableIndex::save(self, path)?;
        Ok(true)
    }
}

impl SerializableIndex for PartitionedIvfIndex {
    fn save(&self, path: &std::path::Path) -> crate::Result<()> {
        // Only the state: partition data belongs to object storage, and writing it here would
        // duplicate the thing this index exists to avoid holding.
        crate::search::index_persistence::write_index(path, "partitioned-ivf", &self.state)
    }

    fn load(path: &std::path::Path) -> crate::Result<Self> {
        let state: PartitionedIvfState =
            crate::search::index_persistence::read_index(path, "partitioned-ivf")?;
        Ok(Self {
            state,
            // The caller must re-attach a store with `with_store`; without one the index answers
            // from its overflow buffer alone.
            store: None,
            resident: Arc::new(RwLock::new(HashMap::new())),
            last_used: Arc::new(RwLock::new(HashMap::new())),
            clock: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    fn stats(&self) -> IndexStats {
        let dim = self.state.dimension.unwrap_or(0);
        let centroid_bytes = self.state.partitions.len() * dim * std::mem::size_of::<f32>();
        let overflow_bytes = self.state.overflow.len() * dim * std::mem::size_of::<f32>();
        IndexStats {
            total_vectors: self.len(),
            // Resident partitions are a cache; what this index costs when idle is its centroids.
            index_size_bytes: centroid_bytes + overflow_bytes,
            build_time_ms: 0,
        }
    }
}

fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    metric
        .calculate_simd(a, b)
        .unwrap_or_else(|_| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum())
}

fn nearest_centroid(centroids: &[DenseVector], v: &[f32], metric: DistanceMetric) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, distance(metric, v, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Plain Lloyd's algorithm, seeded by evenly spaced samples.
///
/// Deterministic on purpose: a rebuild that reshuffled partitions differently every time would
/// make the object-storage segments churn for no benefit.
fn kmeans(
    vectors: &[(RecordId, DenseVector)],
    k: usize,
    iterations: usize,
    metric: DistanceMetric,
) -> Vec<DenseVector> {
    let dim = vectors[0].1.len();
    let stride = (vectors.len() / k).max(1);
    let mut centroids: Vec<DenseVector> = (0..k)
        .map(|i| vectors[(i * stride) % vectors.len()].1.clone())
        .collect();

    for _ in 0..iterations {
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (_, v) in vectors {
            let i = nearest_centroid(&centroids, v, metric);
            for (d, x) in v.iter().enumerate() {
                sums[i][d] += x;
            }
            counts[i] += 1;
        }

        let mut moved = false;
        for i in 0..k {
            if counts[i] == 0 {
                continue;
            }
            let next: DenseVector = sums[i].iter().map(|s| s / counts[i] as f32).collect();
            if next != centroids[i] {
                moved = true;
                centroids[i] = next;
            }
        }
        if !moved {
            break;
        }
    }

    centroids
}
