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
        }
    }
}

/// One partition's vectors, as stored in object storage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartitionData {
    vectors: Vec<(RecordId, DenseVector)>,
}

/// What the index keeps about a partition regardless of residency.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionMeta {
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
    deleted: Vec<RecordId>,
}

pub struct PartitionedIvfIndex {
    state: PartitionedIvfState,
    /// Where partitions are fetched from. `None` means everything must already be resident,
    /// which is the case in tests that never offload.
    store: Option<Arc<dyn ObjectStorage + Send + Sync>>,
    /// Loaded partitions, keyed by index. Bounded by `max_resident_partitions`.
    resident: Arc<RwLock<HashMap<usize, PartitionData>>>,
    /// Least-recently-used ordering over `resident`, most recent last.
    lru: Arc<RwLock<Vec<usize>>>,
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
                deleted: Vec::new(),
            },
            store: None,
            resident: Arc::new(RwLock::new(HashMap::new())),
            lru: Arc::new(RwLock::new(Vec::new())),
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

    fn partition_key(&self, i: usize) -> String {
        format!("{}/partition-{:06}.seg", self.state.prefix, i)
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
            self.lru.write().await.clear();
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
            .map(|(c, b)| PartitionMeta {
                centroid: c.clone(),
                count: b.vectors.len(),
            })
            .collect();

        if let Some(ref store) = self.store {
            for (i, bucket) in buckets.iter().enumerate() {
                let bytes = bincode::serialize(bucket)
                    .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                store
                    .put_object(
                        &self.partition_key(i),
                        bytes::Bytes::from(bytes),
                        Some("application/octet-stream"),
                    )
                    .await
                    .map_err(|e| {
                        crate::Error::Internal(format!("writing partition {}: {}", i, e))
                    })?;
            }
            // Everything is durable in object storage now, so drop the local copies and let
            // queries pull back only what they probe. Keeping them would defeat the purpose.
            self.resident.write().await.clear();
            self.lru.write().await.clear();
        } else {
            // No store: the partitions have nowhere else to live, so they stay resident.
            let mut resident = self.resident.write().await;
            let mut lru = self.lru.write().await;
            resident.clear();
            lru.clear();
            for (i, bucket) in buckets.into_iter().enumerate() {
                resident.insert(i, bucket);
                lru.push(i);
            }
        }

        self.state.overflow.clear();
        self.state.deleted.clear();
        Ok(())
    }

    /// Make partition `i` resident, fetching it if necessary, and return its vectors.
    async fn load_partition(&self, i: usize) -> crate::Result<PartitionData> {
        if let Some(data) = self.resident.read().await.get(&i) {
            self.touch(i).await;
            return Ok(data.clone());
        }

        let Some(ref store) = self.store else {
            // Without a store an absent partition is simply empty; this is the in-memory mode.
            return Ok(PartitionData::default());
        };

        let bytes = store
            .get_object(&self.partition_key(i))
            .await
            .map_err(|e| crate::Error::Internal(format!("fetching partition {}: {}", i, e)))?;
        let data: PartitionData =
            bincode::deserialize(&bytes).map_err(|e| crate::Error::Serialization(e.to_string()))?;

        debug!("Loaded partition {} ({} vectors)", i, data.vectors.len());
        self.admit(i, data.clone()).await;
        Ok(data)
    }

    async fn touch(&self, i: usize) {
        let mut lru = self.lru.write().await;
        if let Some(pos) = lru.iter().position(|&x| x == i) {
            lru.remove(pos);
        }
        lru.push(i);
    }

    /// Insert a partition, evicting the least recently used one if we are at the cap.
    async fn admit(&self, i: usize, data: PartitionData) {
        let mut resident = self.resident.write().await;
        let mut lru = self.lru.write().await;

        while resident.len() >= self.state.config.max_resident_partitions && !lru.is_empty() {
            let victim = lru.remove(0);
            resident.remove(&victim);
        }

        resident.insert(i, data);
        lru.retain(|&x| x != i);
        lru.push(i);
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
        self.state.deleted.retain(|d| d != &id);
        self.state.overflow.push((id, vector));
        Ok(())
    }

    async fn bulk_build(&mut self, vectors: Vec<(String, Vec<f32>)>) -> crate::Result<()> {
        self.rebuild(vectors).await
    }

    async fn search(&self, query: &[f32], k: usize) -> crate::Result<Vec<(String, f32)>> {
        let metric = self.state.distance_metric;
        let mut hits: Vec<(String, f32)> = Vec::new();

        // Rank partitions by centroid distance and probe only the closest. This is the whole
        // point: the decision about what to read is made from centroids alone.
        let mut ranked: Vec<(usize, f32)> = self
            .state
            .partitions
            .iter()
            .enumerate()
            .map(|(i, p)| (i, distance(metric, query, &p.centroid)))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, _) in ranked.into_iter().take(self.state.config.nprobe) {
            match self.load_partition(i).await {
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
                    i, e
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
        let owned = id.to_string();
        if !self.state.deleted.contains(&owned) {
            self.state.deleted.push(owned);
        }
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
            lru: Arc::new(RwLock::new(Vec::new())),
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
