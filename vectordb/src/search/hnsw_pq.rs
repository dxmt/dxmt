//! HNSW over PQ codes: the graph holds no vectors, only codes.
//!
//! This is the resident half of the disk index (docs/architecture/disk-index.md §1): the graph and
//! a 192-byte code per vector stay in memory, the full vectors live in the record store, and a
//! search returns the walk's best candidates ranked by ADC for the storage layer to rerank
//! against the originals. Measured on 1M and 10M Cohere 768-d, a walk on 192-byte codes with
//! the best 500 reranked matched an exact walk at the same `ef` to within 0.1 points of
//! recall@100 (sections 2.7, 2.10); 96-byte codes needed twice the `ef` and four times the
//! rerank for the same (section 2.7).
//!
//! The previous version of this file kept a full-vector `HnswIndex` *and* a PQ code per
//! record *and*, with reranking on, a third copy of every vector -- strictly more memory than
//! the plain graph it was meant to replace -- and searched the graph on the full vectors before
//! reranking with the codes, which is the two roles the wrong way round.

use crate::search::hnsw::{HnswConfig, HnswIndex};
use crate::search::quantization::product_quantization::ProductQuantizer;
use crate::types::{DenseVector, DistanceMetric, RecordId};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct HnswPqIndex {
    /// Exact until the quantizer is trained, on codes from then on.
    hnsw: HnswIndex,
    config: HnswPqConfig,
    distance_metric: DistanceMetric,
    dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswPqConfig {
    pub hnsw_config: HnswConfig,
    /// Bytes per code. 768 dimensions in 192 subspaces is the measured default (4 dimensions
    /// per subspace); 96 loses too much at recall@100 (section 2.7).
    pub num_subvectors: usize,
    /// Centroids per subspace; 256 keeps a code to one byte per subspace.
    pub num_centroids: usize,
    /// Candidates handed back for reranking, as a multiple of `k`, bounded below by
    /// `rerank_min`. 5x: recall@10 needed 100 candidates and recall@100 needed 500 on both
    /// 1M and 10M (sections 2.8-2.10).
    pub rerank_factor: usize,
    pub rerank_min: usize,
    /// Rows at which a streaming build trains and quantizes on its own, without waiting for
    /// a rebuild. High enough that the codebooks see a representative sample -- the first
    /// batches of a load are often one cluster -- and the one-off training cost (about two
    /// minutes for 192 subspaces on 32 cores) lands once.
    pub quantize_at_rows: usize,
}

impl HnswPqConfig {
    /// How many PQ subspaces a table of this width gets when nobody says otherwise.
    ///
    /// It was `dimension / 4` everywhere -- four dimensions per subspace, the layout
    /// measured at 768 (192 bytes a code, recall@100 held at 1M and 10M). Applied to a
    /// 25-dimension corpus that is five subspaces of five dimensions: a 100-byte vector in a
    /// 5-byte code, and the walk on those codes cannot tell neighbours apart. Measured on
    /// 2026-10-02, 1.18M rows, recall@10 at `normal` against the published ground truth:
    ///
    /// | corpus     | dims/subspace 4 (old) | 2      | 1      | no PQ  |
    /// |------------|-----------------------|--------|--------|--------|
    /// | glove-25   | 0.7595 (5 x 5)        |        | 1.0000 | 1.0000 |
    /// | glove-100  | 0.8950                | 0.9235 | 0.9275 |        |
    /// | glove-200  | 0.8485                | 0.8775 |        |        |
    /// | gist-960   | 0.9780                |        |        |        |
    ///
    /// At 25 dimensions the code was the whole problem; one dimension per subspace -- a
    /// 4x code rather than 20x -- is exact at `normal`, for 0.7 ms. At 100 and 200 the finer
    /// code buys three points and the rest of the gap is the traversal budget, which is a
    /// different knob (`ef_search`, the rerank pool) and a different change. So: one
    /// dimension per subspace below 64, two below 256, four from there, rounded to the
    /// nearest divisor of the width that does not make a subspace coarser than that.
    pub fn default_subvectors(dimension: usize) -> usize {
        if dimension == 0 {
            return 1;
        }
        let dims_per_subspace = match dimension {
            0..=63 => 1,
            64..=255 => 2,
            _ => 4,
        };
        let wanted = dimension.div_ceil(dims_per_subspace);
        // The smallest divisor of `dimension` that is at least `wanted`: at most
        // `dims_per_subspace` dimensions in any subspace, never more.
        (wanted..=dimension)
            .find(|m| dimension.is_multiple_of(*m))
            .unwrap_or(dimension)
    }

    pub const DEFAULT_QUANTIZE_AT_ROWS: usize = 65_536;
    pub const DEFAULT_RERANK_FACTOR: usize = 5;
    pub const DEFAULT_RERANK_MIN: usize = 100;

    /// `DXMT_RERANK_FACTOR` and `DXMT_RERANK_MIN` override how many candidates a search reranks.
    ///
    /// This is the one knob that trades recall against single-query latency, and it was a
    /// constant, so measuring the trade meant recompiling. At a matched recall it is possible
    /// to refine an order of magnitude fewer candidates than the default does here, which
    /// makes the size of this rerank the first thing to measure on a latency question rather
    /// than the last. Out of range or unparsable values are reported and ignored.
    ///
    /// (That comparison came from a head-to-head run against another engine. The numbers are
    /// in the internal measurement notes, not here: see the note on `HnswIndex::bulk_build`.)
    fn env_usize(var: &str, default: usize, min: usize) -> usize {
        Self::parse_override(
            var,
            crate::config::env::var(var).ok().as_deref(),
            default,
            min,
        )
    }

    /// The parsing half, separated from the environment so it can be tested without
    /// mutating process-wide state: a unit test that sets a variable races every other test
    /// that reads it, and `serial_test` only orders tests that opt in.
    fn parse_override(var: &str, raw: Option<&str>, default: usize, min: usize) -> usize {
        match raw {
            None => default,
            Some(v) => match v.parse::<usize>() {
                Ok(n) if n >= min => n,
                _ => {
                    tracing::warn!("{var}={v} is not a number >= {min}; using {default}");
                    default
                }
            },
        }
    }
}

impl Default for HnswPqConfig {
    fn default() -> Self {
        Self {
            hnsw_config: HnswConfig::default(),
            num_subvectors: 192,
            num_centroids: 256,
            rerank_factor: Self::env_usize("DXMT_RERANK_FACTOR", Self::DEFAULT_RERANK_FACTOR, 1),
            rerank_min: Self::env_usize("DXMT_RERANK_MIN", Self::DEFAULT_RERANK_MIN, 1),
            quantize_at_rows: Self::DEFAULT_QUANTIZE_AT_ROWS,
        }
    }
}

impl HnswPqIndex {
    /// Fewest training vectors per centroid. k-means over 256 centroids fitted to 300 points
    /// describes those points, not the distribution; at 16 apiece the default 256-centroid
    /// codebook wants 4,096 vectors.
    const MIN_VECTORS_PER_CENTROID: usize = 16;

    /// Most vectors to train on. Training is k-means over every subspace and its cost is
    /// linear in this; past a few tens of thousands the codebooks stop moving. 65,536 trained
    /// 192 subspaces in 113 s on 32 cores.
    const MAX_TRAINING_VECTORS: usize = 65_536;

    pub fn new(
        dimension: usize,
        distance_metric: DistanceMetric,
        config: Option<HnswPqConfig>,
    ) -> crate::Result<Self> {
        let mut config = config.unwrap_or_default();
        if !dimension.is_multiple_of(config.num_subvectors) {
            // A count that does not divide the width cannot be honoured; the width's own
            // default is the nearest thing to what was asked that can.
            config.num_subvectors = HnswPqConfig::default_subvectors(dimension);
        }
        Ok(Self {
            hnsw: HnswIndex::new(distance_metric, Some(config.hnsw_config.clone())),
            config,
            distance_metric,
            dimension,
        })
    }

    fn training_minimum(&self) -> usize {
        self.config
            .num_centroids
            .saturating_mul(Self::MIN_VECTORS_PER_CENTROID)
    }

    /// Quantize from the vectors the exact graph still holds, once there are enough of them.
    fn quantize_from_graph(&mut self) -> crate::Result<()> {
        let n = crate::search::index_traits::VectorIndex::len(&self.hnsw);
        if self.hnsw.is_quantized() || n < self.training_minimum() {
            return Ok(());
        }
        let sample = self.hnsw.sample_vectors(Self::MAX_TRAINING_VECTORS);
        tracing::info!(
            "HNSW-PQ: quantizing a {}-row graph from a {}-vector sample",
            n,
            sample.len()
        );
        // Graph vectors are already metric-prepared; `train` prepares again, which is
        // idempotent for unit vectors.
        self.train(&sample)
    }

    /// Train the codebooks on `training_data` and switch the graph to codes.
    ///
    /// The quantizer works in the graph's own space: unit-length vectors for cosine, where
    /// the squared Euclidean distance the ADC table sums ranks exactly as the cosine distance.
    pub fn train(&mut self, training_data: &[DenseVector]) -> crate::Result<()> {
        if self.hnsw.is_quantized() {
            return Ok(());
        }
        let prepared: Vec<DenseVector> = training_data
            .iter()
            .map(|v| self.prepare(v.clone()))
            .collect();
        let mut pq = ProductQuantizer::new(
            self.dimension,
            self.config.num_subvectors,
            self.config.num_centroids,
            DistanceMetric::Euclidean,
        )?;
        pq.train(&prepared, 25)?;
        self.hnsw.quantize(pq)
    }

    fn prepare(&self, mut v: DenseVector) -> DenseVector {
        if matches!(self.distance_metric, DistanceMetric::Cosine) {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
        }
        v
    }

    pub fn is_quantized(&self) -> bool {
        self.hnsw.is_quantized()
    }

    pub fn insert(&mut self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        self.hnsw.insert(id, vector)?;
        self.maybe_quantize()
    }

    /// Whether a streaming build has reached the row count at which it quantizes itself.
    ///
    /// Below `quantize_at_rows` it waits for a rebuild (`wants_rebuild`) or stays exact.
    fn wants_quantizing(&self) -> bool {
        !self.hnsw.is_quantized()
            && crate::search::index_traits::VectorIndex::len(&self.hnsw)
                >= self.config.quantize_at_rows
            && crate::search::index_traits::VectorIndex::len(&self.hnsw) >= self.training_minimum()
    }

    /// Quantize in one go, holding whatever lock the caller holds.
    ///
    /// The embedded path, where there is no wrapper to split the work across locks and no
    /// concurrent reader to starve. `IndexWrapper` uses `quantization_job` and
    /// `install_quantizer` instead, because on a server this takes about a minute and nothing
    /// else may touch the table while it runs.
    fn maybe_quantize(&mut self) -> crate::Result<()> {
        if self.wants_quantizing() {
            self.quantize_from_graph()?;
        }
        Ok(())
    }

    /// The walk's best `k` by ADC (exact distances while untrained). Approximate once
    /// quantized: callers rerank through [`Self::rerank_candidates`].
    pub fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        self.hnsw.search(query, k)
    }

    /// How many candidates to fetch and rerank exactly for a top-`k`, or `None` while the
    /// graph still scores exactly.
    pub fn rerank_candidates(&self, k: usize) -> Option<usize> {
        self.hnsw
            .is_quantized()
            .then(|| (k * self.config.rerank_factor).max(self.config.rerank_min))
    }

    pub fn memory_stats(&self) -> MemoryStats {
        let num_vectors = crate::search::index_traits::VectorIndex::len(&self.hnsw);
        let resident = self.hnsw.resident_bytes_without_vectors();
        let vectors = if self.hnsw.is_quantized() {
            0
        } else {
            num_vectors * self.dimension * std::mem::size_of::<f32>()
        };
        MemoryStats {
            total_bytes: resident + vectors,
            graph_and_codes_bytes: resident,
            vectors_bytes: vectors,
            num_vectors,
            compression_ratio: (num_vectors * self.dimension * 4) as f32
                / (resident + vectors).max(1) as f32,
        }
    }

    pub fn len(&self) -> usize {
        crate::search::index_traits::VectorIndex::len(&self.hnsw)
    }

    pub fn is_empty(&self) -> bool {
        crate::search::index_traits::VectorIndex::is_empty(&self.hnsw)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_bytes: usize,
    /// Links, codes, codebooks and the centroid table -- everything resident once quantized.
    pub graph_and_codes_bytes: usize,
    /// Full vectors still on the nodes; zero once quantized.
    pub vectors_bytes: usize,
    pub num_vectors: usize,
    pub compression_ratio: f32,
}

impl MemoryStats {
    pub fn total_mb(&self) -> f32 {
        self.total_bytes as f32 / 1024.0 / 1024.0
    }
}

#[async_trait::async_trait]
impl crate::search::index_traits::VectorIndex for HnswPqIndex {
    async fn memory_bytes(&self) -> usize {
        crate::search::index_traits::VectorIndex::memory_bytes(&self.hnsw).await
    }

    async fn add(&mut self, id: String, vector: Vec<f32>) -> crate::Result<()> {
        self.insert(id, vector)
    }

    async fn add_batch(&mut self, items: Vec<(String, Vec<f32>)>) -> crate::Result<()> {
        self.hnsw.insert_batch(items)?;
        self.maybe_quantize()
    }

    /// Train on a sample, quantize, then build on every core.
    ///
    /// Training needs enough vectors to place `num_centroids` centroids per subspace; below
    /// that, k-means would fit codebooks to noise, so the graph stays exact and reports no
    /// rerank need. The sample is a fixed stride rather than a random draw so two builds of
    /// the same data produce the same codebooks.
    async fn bulk_build(&mut self, vectors: Vec<(String, Vec<f32>)>) -> crate::Result<()> {
        let wanted = self.training_minimum();
        if !self.hnsw.is_quantized() && vectors.len() >= wanted {
            let stride = (vectors.len() / Self::MAX_TRAINING_VECTORS).max(1);
            let sample: Vec<DenseVector> = vectors
                .iter()
                .step_by(stride)
                .take(Self::MAX_TRAINING_VECTORS)
                .map(|(_, v)| v.clone())
                .collect();
            self.train(&sample)?;
        } else if !self.hnsw.is_quantized() {
            tracing::info!(
                "HNSW-PQ: {} vectors is below the {} needed to train {} centroids per \
                 subspace; building an exact graph instead",
                vectors.len(),
                wanted,
                self.config.num_centroids
            );
        }
        crate::search::index_traits::VectorIndex::bulk_build(&mut self.hnsw, vectors).await
    }

    async fn search(&self, query: &[f32], k: usize) -> crate::Result<Vec<(String, f32)>> {
        self.search(&query.to_vec(), k)
    }

    async fn search_with_ef_hint(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.hnsw.search_with_ef(&query.to_vec(), k, ef)
    }

    async fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_ids: &std::collections::HashSet<String>,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.hnsw
            .search_with_filter(&query.to_vec(), k, ef, filter_ids)
    }

    fn rerank_candidates(&self, k: usize) -> Option<usize> {
        HnswPqIndex::rerank_candidates(self, k)
    }

    /// ADC over the codes of `ids`, once the graph is quantized; before that the graph
    /// scores exactly, which is still one distance per id from memory.
    fn scan_ids(&self, query: &[f32], ids: &[&str], k: usize) -> Option<Vec<(String, f32)>> {
        Some(self.hnsw.scan_ids(&query.to_vec(), ids, k))
    }

    /// Deliberately not `maybe_quantize`. `IndexWrapper` calls `quantization_job` and
    /// `install_quantizer` around this, so the training happens with no lock held; leaving
    /// the inline path here as well would train twice, the second time for nothing.
    fn after_batch(&mut self) -> crate::Result<()> {
        Ok(())
    }

    fn slot_count(&self) -> usize {
        crate::search::index_traits::VectorIndex::slot_count(&self.hnsw)
    }

    fn is_quantized(&self) -> bool {
        // The inherent method of the same name; qualified so this impl cannot call itself.
        HnswPqIndex::is_quantized(self)
    }

    fn quantization_pending(&self) -> bool {
        // The inherent method of the same intent; named apart so this impl cannot call itself.
        HnswPqIndex::wants_quantizing(self)
    }

    fn quantization_job(
        &self,
    ) -> Option<(
        Vec<DenseVector>,
        crate::search::quantization::product_quantization::ProductQuantizer,
    )> {
        if !self.wants_quantizing() {
            return None;
        }
        let sample = self.hnsw.sample_vectors(Self::MAX_TRAINING_VECTORS);
        if sample.len() < self.training_minimum() {
            return None;
        }
        // Graph vectors are already metric-prepared, and `prepare` is idempotent for unit
        // vectors, so the sample goes to the caller as it is.
        let pq = crate::search::quantization::product_quantization::ProductQuantizer::new(
            self.dimension,
            self.config.num_subvectors,
            self.config.num_centroids,
            DistanceMetric::Euclidean,
        )
        .ok()?;
        tracing::info!(
            "HNSW-PQ: {}-row graph wants quantizing; training {} subspaces on {} vectors              outside the index lock",
            crate::search::index_traits::VectorIndex::len(&self.hnsw),
            self.config.num_subvectors,
            sample.len()
        );
        Some((sample, pq))
    }

    fn install_quantizer(
        &mut self,
        pq: crate::search::quantization::product_quantization::ProductQuantizer,
    ) -> crate::Result<()> {
        // Someone else got there while this quantizer was training. Theirs is as good as
        // this one -- both were fitted to a stride sample of the same graph -- so drop this.
        if self.hnsw.is_quantized() {
            return Ok(());
        }
        self.hnsw.quantize(pq)
    }

    fn wants_rebuild(&self) -> bool {
        !self.hnsw.is_quantized()
            && crate::search::index_traits::VectorIndex::len(&self.hnsw) >= self.training_minimum()
    }

    fn as_hnsw(&self) -> Option<&HnswIndex> {
        Some(&self.hnsw)
    }

    fn as_hnsw_mut(&mut self) -> Option<&mut HnswIndex> {
        Some(&mut self.hnsw)
    }

    async fn delete(&mut self, id: &str) -> crate::Result<()> {
        self.hnsw.delete(id).await
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn snapshot_to(&self, path: &std::path::Path) -> crate::Result<bool> {
        crate::search::index_traits::SerializableIndex::save(self, path)?;
        Ok(true)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

impl crate::search::index_traits::SerializableIndex for HnswPqIndex {
    fn save(&self, path: &std::path::Path) -> crate::Result<()> {
        // "hnsw-pq-v2": the previous layout carried a code map and optional originals
        // beside the graph; this one is the graph with the codes inside it.
        crate::search::index_persistence::write_index(path, "hnsw-pq-v2", self)
    }

    fn load(path: &std::path::Path) -> crate::Result<Self> {
        let mut index: Self = crate::search::index_persistence::read_index(path, "hnsw-pq-v2")?;
        index.hnsw.after_load();
        Ok(index)
    }

    fn stats(&self) -> crate::search::index_traits::IndexStats {
        let inner = crate::search::index_traits::SerializableIndex::stats(&self.hnsw);
        crate::search::index_traits::IndexStats {
            total_vectors: inner.total_vectors,
            index_size_bytes: inner.index_size_bytes,
            build_time_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::index_traits::VectorIndex;

    /// Distinct unit vectors around shared centres, the structure a quantizer relies on.
    fn corpus(n: usize, dim: usize, seed: u64) -> Vec<(String, Vec<f32>)> {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        };
        let mut cstate = 0xC0FFEE_u64;
        let mut cnext = move || {
            cstate ^= cstate << 13;
            cstate ^= cstate >> 7;
            cstate ^= cstate << 17;
            (cstate >> 40) as f32 / 16_777_216.0 - 0.5
        };
        let centres: Vec<Vec<f32>> = (0..40)
            .map(|_| (0..dim).map(|_| cnext()).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centres[i % centres.len()];
                let v: Vec<f32> = (0..dim).map(|d| c[d] + 0.15 * next()).collect();
                (format!("v{i}"), v)
            })
            .collect()
    }

    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        1.0 - dot / (na * nb)
    }

    fn small_config() -> HnswPqConfig {
        HnswPqConfig {
            num_subvectors: 16,
            num_centroids: 32,
            ..HnswPqConfig::default()
        }
    }

    /// The rerank size is the one knob that trades recall against single-query latency, so
    /// it has to be settable without a rebuild -- and a bad value must not silently become a
    /// different search. The parsing is tested directly rather than through the environment:
    /// a test that sets a process-wide variable races every other test that reads one, which
    /// is how the first version of this broke `a_phased_batch_insert_lets_hnsw_pq_quantize`.
    #[test]
    fn a_rerank_override_is_parsed_or_refused() {
        let p = |raw: Option<&str>| HnswPqConfig::parse_override("DXMT_RERANK_FACTOR", raw, 5, 1);
        assert_eq!(p(None), 5, "unset keeps the default");
        assert_eq!(p(Some("2")), 2);
        assert_eq!(p(Some("20")), 20);
        for bad in ["0", "-1", "many", "", " 2", "2.5"] {
            assert_eq!(p(Some(bad)), 5, "{bad:?} should have been refused");
        }
        assert_eq!(
            HnswPqConfig::parse_override("DXMT_RERANK_MIN", Some("50"), 100, 1),
            50
        );
    }

    /// And the number reaches the search: a config asking for 2x hands back 2k candidates.
    #[tokio::test]
    async fn the_rerank_size_reaches_the_search() {
        let config = HnswPqConfig {
            rerank_factor: 2,
            rerank_min: 50,
            ..small_config()
        };
        let mut index = HnswPqIndex::new(64, DistanceMetric::Cosine, Some(config)).unwrap();
        assert_eq!(
            index.rerank_candidates(100),
            None,
            "an exact graph reranks nothing"
        );
        index
            .bulk_build(corpus(3000, 64, 0x2545_F491_4F6C_DD1D))
            .await
            .unwrap();
        assert!(index.is_quantized());
        assert_eq!(index.rerank_candidates(100), Some(200), "2 x k");
        assert_eq!(index.rerank_candidates(10), Some(50), "the floor");
    }

    #[tokio::test]
    async fn bulk_build_quantizes_and_the_graph_holds_no_vectors() {
        let mut index = HnswPqIndex::new(64, DistanceMetric::Cosine, Some(small_config())).unwrap();
        assert!(!index.is_quantized());
        assert_eq!(
            index.rerank_candidates(10),
            None,
            "an exact graph needs no rerank"
        );

        // 32 centroids x 16 per centroid = 512 vectors to train; 3,000 is plenty.
        let data = corpus(3000, 64, 0x9E3779B97F4A7C15);
        index.bulk_build(data.clone()).await.unwrap();

        assert!(index.is_quantized());
        assert_eq!(index.len(), 3000);
        let stats = index.memory_stats();
        assert_eq!(
            stats.vectors_bytes, 0,
            "the graph must not keep the vectors"
        );
        assert!(
            stats.graph_and_codes_bytes >= 3000 * 16,
            "codes are resident"
        );
        assert_eq!(index.rerank_candidates(10), Some(100));
        assert_eq!(index.rerank_candidates(100), Some(500));

        // The walk's candidates, reranked exactly, put the query's own vector first.
        let mut found = 0;
        for (id, v) in data.iter().take(100) {
            let cands = index.search_with_ef_hint(v, 100, 128).await.unwrap();
            let best = cands
                .iter()
                .min_by(|a, b| {
                    let da = cosine_distance(v, &data[a.0[1..].parse::<usize>().unwrap()].1);
                    let db = cosine_distance(v, &data[b.0[1..].parse::<usize>().unwrap()].1);
                    da.total_cmp(&db)
                })
                .unwrap();
            if &best.0 == id {
                found += 1;
            }
        }
        assert!(
            found >= 98,
            "{found}/100 queries found their own vector after rerank"
        );
    }

    #[tokio::test]
    async fn a_streaming_build_quantizes_itself_at_the_threshold() {
        let config = HnswPqConfig {
            quantize_at_rows: 2000,
            ..small_config()
        };
        let mut index = HnswPqIndex::new(64, DistanceMetric::Cosine, Some(config)).unwrap();
        let data = corpus(3000, 64, 0x9E3779B97F4A7C15);
        index.add_batch(data[..1000].to_vec()).await.unwrap();
        assert!(!index.is_quantized());
        assert!(
            index.wants_rebuild(),
            "past the training minimum but under the threshold: a rebuild would quantize"
        );
        index.add_batch(data[1000..2500].to_vec()).await.unwrap();
        assert!(
            index.is_quantized(),
            "2,500 rows crossed the 2,000-row threshold"
        );
        assert!(!index.wants_rebuild());
        index.add_batch(data[2500..].to_vec()).await.unwrap();
        assert_eq!(index.len(), 3000);
        assert_eq!(index.memory_stats().vectors_bytes, 0);
        // Rows inserted after quantizing were planned with ADC against the codes; nearly
        // all of them are reachable within the rerank window (a single miss is what a
        // 16 x 32-centroid code on this data allows, not a defect).
        let mut found = 0;
        for (id, v) in &data[2500..2600] {
            let cands = index.search_with_ef_hint(v, 100, 128).await.unwrap();
            if cands.iter().any(|(c, _)| c == id) {
                found += 1;
            }
        }
        assert!(found >= 90, "{found}/100 late inserts reachable");
    }

    #[tokio::test]
    async fn a_small_corpus_builds_an_exact_graph() {
        let mut index = HnswPqIndex::new(64, DistanceMetric::Cosine, Some(small_config())).unwrap();
        // Under the 512 that 32 centroids want.
        let data = corpus(200, 64, 0x2545F4914F6CDD1D);
        index.bulk_build(data.clone()).await.unwrap();
        assert!(!index.is_quantized());
        assert_eq!(index.rerank_candidates(10), None);
        let hits = index.search(&data[7].1, 3).unwrap();
        assert_eq!(hits[0].0, "v7");
    }

    #[tokio::test]
    async fn a_quantized_index_round_trips() {
        use crate::search::index_traits::SerializableIndex;
        let mut index = HnswPqIndex::new(64, DistanceMetric::Cosine, Some(small_config())).unwrap();
        let data = corpus(1500, 64, 0x1234_5678_9ABC_DEF1);
        index.bulk_build(data.clone()).await.unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pq.idx");
        index.save(&path).unwrap();
        let back = HnswPqIndex::load(&path).unwrap();
        assert!(back.is_quantized());
        for (_, v) in data.iter().take(20) {
            assert_eq!(
                index.search_with_ef_hint(v, 10, 64).await.unwrap(),
                back.search_with_ef_hint(v, 10, 64).await.unwrap()
            );
        }
    }

    #[test]
    fn subvectors_fall_back_to_a_divisor_of_the_dimension() {
        // The default config says 192, which does not divide 100; the width's own rule does.
        let index = HnswPqIndex::new(100, DistanceMetric::Euclidean, None).unwrap();
        assert_eq!(
            index.config.num_subvectors, 50,
            "100 dims: 50 subspaces of 2"
        );
        let index = HnswPqIndex::new(768, DistanceMetric::Euclidean, None).unwrap();
        assert_eq!(index.config.num_subvectors, 192);
    }

    /// The layout the width gets by default, per the table in `default_subvectors`.
    #[test]
    fn default_subvectors_follow_the_width() {
        let cases = [
            (25, 25), // one dimension per subspace below 64
            (32, 32),
            (50, 50),
            (64, 32), // two per subspace from 64
            (100, 50),
            (128, 64),
            (200, 100),
            (255, 255), // prime: nothing divides it, so one per subspace
            (256, 64),  // four per subspace from 256
            (384, 96),
            (768, 192),
            (960, 240),
            (1024, 256),
            (1536, 384),
            (3072, 768),
        ];
        for (dim, want) in cases {
            assert_eq!(
                HnswPqConfig::default_subvectors(dim),
                want,
                "{dim} dimensions"
            );
            assert!(dim.is_multiple_of(HnswPqConfig::default_subvectors(dim)));
        }
    }
}
