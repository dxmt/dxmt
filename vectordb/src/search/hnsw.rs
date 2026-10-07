use crate::precomputed_query::PrecomputedQuery;
use crate::search::node_filter::{NodeFilter, NodeGate};
use crate::search::quantization::product_quantization::ProductQuantizer;
use crate::search::simd_distance::SimdDistance;
use crate::types::{DenseVector, DistanceMetric, RecordId};
use parking_lot::RwLock;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    pub m: usize,
    pub max_m: usize,
    pub max_m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub ml: f64,
    pub seed: u64,
    // Compatibility fields for legacy code
    pub max_connections: usize,
    pub level_multiplier: f64,
    pub max_level: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            max_m: 16,
            max_m0: 32,
            ef_construction: 200,
            ef_search: 100,
            // 1/ln(m), the scale factor from the HNSW paper: bigger m means fewer layers,
            // because each layer already reaches further. Hard-coding ln(2) instead made the
            // number of layers independent of m.
            ml: 1.0 / (16.0_f64.ln()),
            seed: 42,
            // Compatibility defaults
            max_connections: 16,
            level_multiplier: 1.0 / (16.0_f64.ln()),
            max_level: 16,
        }
    }
}

impl HnswConfig {
    pub fn from_params(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m,
            max_m: m,
            max_m0: m * 2,
            ef_construction,
            ef_search,
            ml: 1.0 / (m.max(2) as f64).ln(),
            seed: 42,
            // Compatibility
            max_connections: m,
            level_multiplier: 1.0 / (m.max(2) as f64).ln(),
            max_level: 16,
        }
    }
}

/// A visited-set for one graph traversal, reused across traversals on the same thread.
///
/// Every layer search needs "have I already looked at this node?", and the obvious answer — a
/// fresh `HashSet` per call — allocates on every layer of every insert and every query, then pays
/// SipHash on each of the thousands of membership tests a single traversal makes.
///
/// Instead: one buffer per thread holding a generation number per node. A node counts as visited
/// when its stamp equals the current generation, so starting a traversal costs an increment
/// rather than a clear, and a test is an array read. A thread-local is used instead of a shared
/// pool because it needs no pool and cannot be contended, which suits us: `search_layer` takes
/// `&self` and would otherwise have to lock something to get a scratch buffer.
mod visited {
    // The set of nodes a search has touched.
    //
    // This used to be one `u32` stamp per node of the graph, kept per thread: 4 bytes a
    // node, which at 10M nodes is 40 MB a thread and, with every worker thread that ever
    // searched holding one, about 2.4 GB of the 7 GB the 10M acceptance run kept resident.
    // At 100M it would be 400 MB a thread. A search visits a few tens of thousands of nodes
    // however large the graph is, so the set is now a small open-addressing hash table with
    // generation stamps -- a few megabytes a thread, independent of the graph's size, and
    // nothing to clear between searches.
    use std::cell::RefCell;

    thread_local! {
        static BUFFER: RefCell<(u32, Vec<u64>)> = const { RefCell::new((0, Vec::new())) };
    }

    /// Slots to start with: enough for the visits of an `ef`-2048 search without growing.
    const INITIAL_SLOTS: usize = 1 << 16;

    /// A slot packs `generation << 32 | node index`; a slot from an older generation is free.
    pub(super) struct Visited {
        generation: u32,
        slots: Vec<u64>,
        len: usize,
    }

    impl Visited {
        /// Take the thread's table and start a new generation. `_graph_size` is what the
        /// stamp array used to be sized by; the table does not depend on it.
        pub(super) fn start(_graph_size: usize) -> Self {
            BUFFER.with(|cell| {
                let (generation, slots) = &mut *cell.borrow_mut();
                let mut slots = std::mem::take(slots);
                if slots.is_empty() {
                    slots = vec![0; INITIAL_SLOTS];
                }
                *generation = generation.wrapping_add(1);
                if *generation == 0 {
                    slots.iter_mut().for_each(|s| *s = 0);
                    *generation = 1;
                }
                Visited {
                    generation: *generation,
                    slots,
                    len: 0,
                }
            })
        }

        #[inline]
        fn hash(idx: usize) -> usize {
            // Fibonacci hashing: node indices are dense, so a multiply spreads them.
            (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize >> 20
        }

        #[inline]
        fn slot_of(&self, idx: usize) -> (usize, bool) {
            let mask = self.slots.len() - 1;
            let tag = (self.generation as u64) << 32 | idx as u64;
            let mut i = Self::hash(idx) & mask;
            loop {
                let s = self.slots[i];
                if s == tag {
                    return (i, true);
                }
                if (s >> 32) as u32 != self.generation {
                    return (i, false);
                }
                i = (i + 1) & mask;
            }
        }

        pub(super) fn contains(&self, idx: usize) -> bool {
            self.slot_of(idx).1
        }

        /// Mark `idx` visited; `true` if it already was.
        pub(super) fn insert(&mut self, idx: usize) -> bool {
            if self.len * 2 >= self.slots.len() {
                self.grow();
            }
            let (i, seen) = self.slot_of(idx);
            if !seen {
                self.slots[i] = (self.generation as u64) << 32 | idx as u64;
                self.len += 1;
            }
            seen
        }

        fn grow(&mut self) {
            let doubled = self.slots.len() * 2;
            let old = std::mem::replace(&mut self.slots, vec![0; doubled]);
            self.len = 0;
            for s in old {
                if (s >> 32) as u32 == self.generation {
                    let idx = (s & 0xffff_ffff) as usize;
                    let (i, _) = self.slot_of(idx);
                    self.slots[i] = s;
                    self.len += 1;
                }
            }
        }
    }

    impl Drop for Visited {
        fn drop(&mut self) {
            BUFFER.with(|cell| {
                cell.borrow_mut().1 = std::mem::take(&mut self.slots);
            });
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_set_forgets_across_generations_and_grows_within_one() {
            let mut v = Visited::start(1_000_000);
            assert!(!v.contains(7));
            assert!(!v.insert(7));
            assert!(v.insert(7), "second insert reports it was seen");
            assert!(v.contains(7));
            // More than the initial slots' half-load, so it grows and rehashes.
            for i in 0..100_000usize {
                v.insert(i * 13);
            }
            for i in 0..100_000usize {
                assert!(v.contains(i * 13), "{i} lost in a grow");
            }
            assert!(!v.contains(5));
            drop(v);
            let v = Visited::start(1_000_000);
            assert!(!v.contains(7), "a new generation starts empty");
            assert!(!v.contains(13 * 500));
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct LegacyNode {
    id: RecordId,
    vector: DenseVector,
    level: usize,
    #[serde(rename = "neighbors")]
    links: Vec<RwLock<Vec<usize>>>,
    deleted: bool,
}

/// Layer-0 edge slots per node. Fixed so a node's layer-0 list is one 132-byte value behind
/// one 8-byte lock, with no heap allocation, no `Vec` header and no capacity slack; serde
/// derives arrays up to this size, which is why it is exactly 32.
pub const MAX_M0_SLOTS: usize = 32;

/// A node's layer-0 edges: `slots[..len]` are the neighbours.
#[derive(Clone, Copy, Serialize, Deserialize)]
struct Links0 {
    len: u8,
    slots: [u32; MAX_M0_SLOTS],
}

impl Links0 {
    const EMPTY: Self = Self {
        len: 0,
        slots: [0; MAX_M0_SLOTS],
    };

    fn as_slice(&self) -> &[u32] {
        &self.slots[..self.len as usize]
    }

    fn set(&mut self, links: &[u32]) {
        let n = links.len().min(MAX_M0_SLOTS);
        self.slots[..n].copy_from_slice(&links[..n]);
        self.len = n as u8;
    }
}

/// One node's edge list at one layer, borrowed mutably under its lock.
enum LinkSlot<'a> {
    L0(&'a mut Links0),
    Upper(&'a mut Vec<u32>),
}

impl LinkSlot<'_> {
    fn as_slice(&self) -> &[u32] {
        match self {
            LinkSlot::L0(l) => l.as_slice(),
            LinkSlot::Upper(v) => v,
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn contains(&self, x: u32) -> bool {
        self.as_slice().contains(&x)
    }

    /// Append if there is room; layer 0 is bounded by its slots, upper layers by nothing.
    fn push(&mut self, x: u32) -> bool {
        match self {
            LinkSlot::L0(l) => {
                if (l.len as usize) < MAX_M0_SLOTS {
                    l.slots[l.len as usize] = x;
                    l.len += 1;
                    true
                } else {
                    false
                }
            }
            LinkSlot::Upper(v) => {
                v.push(x);
                true
            }
        }
    }

    fn set(&mut self, links: &[u32]) {
        match self {
            LinkSlot::L0(l) => l.set(links),
            LinkSlot::Upper(v) => {
                v.clear();
                v.extend_from_slice(links);
            }
        }
    }

    fn retain(&mut self, keep: impl Fn(u32) -> bool) {
        match self {
            LinkSlot::L0(l) => {
                let kept: Vec<u32> = l.as_slice().iter().copied().filter(|&x| keep(x)).collect();
                l.set(&kept);
            }
            LinkSlot::Upper(v) => v.retain(|&x| keep(x)),
        }
    }
}

/// A node's edge lists at layers `1..=level`, one locked list per layer.
type UpperLayers = Box<[RwLock<Vec<u32>>]>;

/// Every node of the graph, as columns rather than as a `Vec` of node structs.
///
/// Per node this is about 140 bytes at layer 0 (an 8-byte lock over 132 bytes of slots), a
/// 16-byte `Option<Box<..>>` that is `None` for the fifteen in sixteen nodes with no upper
/// layer, one byte each of level and tombstone, a 4-byte id offset into a shared byte arena,
/// and the vector itself -- contiguous, so an exact-mode scan walks memory in order, and
/// absent once quantized. The `Vec` header the old node kept for its (by then empty) vector
/// is gone with it.
#[derive(Serialize, Deserialize)]
struct Nodes {
    dim: usize,
    /// `n * dim` floats while the graph is exact; empty once quantized.
    vectors: Vec<f32>,
    level: Vec<u8>,
    deleted: Vec<bool>,
    l0: Vec<RwLock<Links0>>,
    /// Layers `1..=level`, only for nodes with `level >= 1`.
    upper: Vec<Option<UpperLayers>>,
    id_bytes: Vec<u8>,
    /// `len() + 1` entries: node `i`'s id is `id_bytes[id_offsets[i]..id_offsets[i + 1]]`.
    id_offsets: Vec<u32>,
}

impl Nodes {
    fn new() -> Self {
        Self {
            dim: 0,
            vectors: Vec::new(),
            level: Vec::new(),
            deleted: Vec::new(),
            l0: Vec::new(),
            upper: Vec::new(),
            id_bytes: Vec::new(),
            id_offsets: vec![0],
        }
    }

    /// Slots including tombstoned ones; node indices run to this.
    fn len(&self) -> usize {
        self.level.len()
    }

    fn is_empty(&self) -> bool {
        self.level.is_empty()
    }

    fn live_count(&self) -> usize {
        self.deleted.iter().filter(|d| !**d).count()
    }

    /// Append a node. `vector` is `None` once the graph is quantized (the code went to the
    /// quantized store instead); `links[l]` is the edge list at layer `l`.
    fn push(
        &mut self,
        id: &str,
        vector: Option<&[f32]>,
        level: usize,
        links: &[Vec<u32>],
    ) -> usize {
        let idx = self.len();
        if let Some(v) = vector {
            if self.dim == 0 {
                self.dim = v.len();
            }
            self.vectors.extend_from_slice(v);
        }
        self.level.push(level as u8);
        self.deleted.push(false);
        let mut l0 = Links0::EMPTY;
        if let Some(first) = links.first() {
            l0.set(first);
        }
        self.l0.push(RwLock::new(l0));
        self.upper.push(if level == 0 {
            None
        } else {
            Some(
                (1..=level)
                    .map(|l| RwLock::new(links.get(l).cloned().unwrap_or_default()))
                    .collect(),
            )
        });
        self.id_bytes.extend_from_slice(id.as_bytes());
        self.id_offsets.push(self.id_bytes.len() as u32);
        idx
    }

    fn id(&self, idx: usize) -> &str {
        let (a, b) = (
            self.id_offsets[idx] as usize,
            self.id_offsets[idx + 1] as usize,
        );
        // Only ever filled from `&str`, so this is valid UTF-8.
        std::str::from_utf8(&self.id_bytes[a..b]).unwrap_or("")
    }

    /// The stored id as bytes, for comparison. See `utils::id_hash::bytes_eq`: confirming a
    /// hash hit does not need the UTF-8 validation that reading it back as `&str` runs.
    #[inline]
    fn id_raw(&self, idx: usize) -> &[u8] {
        let (a, b) = (
            self.id_offsets[idx] as usize,
            self.id_offsets[idx + 1] as usize,
        );
        &self.id_bytes[a..b]
    }

    fn vector(&self, idx: usize) -> &[f32] {
        &self.vectors[idx * self.dim..(idx + 1) * self.dim]
    }

    #[cfg(test)]
    fn vectors_dropped(&self) -> bool {
        self.vectors.is_empty() && !self.is_empty()
    }

    fn drop_vectors(&mut self) {
        self.vectors = Vec::new();
    }

    fn level(&self, idx: usize) -> usize {
        self.level[idx] as usize
    }

    fn layers(&self, idx: usize) -> usize {
        self.level[idx] as usize + 1
    }

    fn is_deleted(&self, idx: usize) -> bool {
        self.deleted[idx]
    }

    fn mark_deleted(&mut self, idx: usize) {
        self.deleted[idx] = true;
    }

    /// Copy node `idx`'s edges at `layer` into `out` (cleared first). A copy of at most 32
    /// `u32`s per hop, taken under the read lock and released before any distance is
    /// computed, so a search never holds a lock while it scores.
    fn copy_links(&self, idx: usize, layer: usize, out: &mut Vec<u32>) {
        out.clear();
        if layer == 0 {
            out.extend_from_slice(self.l0[idx].read().as_slice());
        } else if let Some(upper) = &self.upper[idx] {
            if let Some(list) = upper.get(layer - 1) {
                out.extend_from_slice(&list.read());
            }
        }
    }

    #[cfg(test)]
    fn edge_count(&self, idx: usize, layer: usize) -> usize {
        if layer == 0 {
            self.l0[idx].read().len as usize
        } else {
            self.upper[idx]
                .as_ref()
                .and_then(|u| u.get(layer - 1))
                .map(|l| l.read().len())
                .unwrap_or(0)
        }
    }

    /// Edit node `idx`'s edges at `layer` under its write lock. Nothing else is locked while
    /// `f` runs, and `f` must not lock another node: that is the whole deadlock argument.
    fn with_links_mut<R>(
        &self,
        idx: usize,
        layer: usize,
        f: impl FnOnce(&mut LinkSlot<'_>) -> R,
    ) -> Option<R> {
        if layer == 0 {
            let mut guard = self.l0[idx].write();
            Some(f(&mut LinkSlot::L0(&mut guard)))
        } else {
            let upper = self.upper[idx].as_ref()?;
            let list = upper.get(layer - 1)?;
            let mut guard = list.write();
            Some(f(&mut LinkSlot::Upper(&mut guard)))
        }
    }

    /// All edges over all layers, for tests.
    #[cfg(test)]
    fn total_edges(&self) -> usize {
        (0..self.len())
            .map(|i| {
                (0..self.layers(i))
                    .map(|l| self.edge_count(i, l))
                    .sum::<usize>()
            })
            .sum()
    }

    /// Bytes this layout holds per node besides the vectors: the columns above plus the id
    /// arena, counted from the containers rather than estimated.
    fn resident_bytes_without_vectors(&self) -> usize {
        use std::mem::size_of;
        self.level.len()
            * (size_of::<u8>()
                + size_of::<bool>()
                + size_of::<RwLock<Links0>>()
                + size_of::<Option<UpperLayers>>())
            + self
                .upper
                .iter()
                .flatten()
                .map(|u| {
                    u.len() * size_of::<RwLock<Vec<u32>>>()
                        + u.iter().map(|l| l.read().capacity() * 4).sum::<usize>()
                })
                .sum::<usize>()
            + self.id_bytes.len()
            + self.id_offsets.len() * 4
    }

    fn from_legacy(nodes: Vec<LegacyNode>) -> Self {
        let mut out = Self::new();
        for node in nodes {
            let links: Vec<Vec<u32>> = node
                .links
                .into_iter()
                .map(|l| l.into_inner().into_iter().map(|x| x as u32).collect())
                .collect();
            let idx = out.push(
                &node.id,
                if node.vector.is_empty() {
                    None
                } else {
                    Some(&node.vector)
                },
                node.level,
                &links,
            );
            if node.deleted {
                out.mark_deleted(idx);
            }
        }
        out
    }
}

use crate::utils::id_hash::{bytes_eq, id_hash, IdHashMap};

// For the candidate priority queue (min-heap, closer is better)
#[derive(Clone)]
struct Candidate {
    idx: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Min-heap: smaller distances have higher priority
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// For the result set (max-heap, farther is better for pruning)
#[derive(Clone)]
struct Result {
    idx: usize,
    distance: f32,
}

impl PartialEq for Result {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Result {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for Result {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Max-heap: larger distances have higher priority
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Result {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Cumulative time in each phase of `insert_batch`, logged at info every `PHASE_LOG_EVERY`
/// batches. Three `Instant::now()` per batch, so it stays on.
///
/// Measured on Cohere 1M x 768, 16 cores: 87% of insert time is in `link_back`'s pruning,
/// 12% in planning, under 1% in the serial append. That breakdown is what to watch when the
/// next insert-path change lands -- a throughput number without it is how a phantom 31%
/// regression from chunked vector storage took seven refuted hypotheses to see through.
static PHASE_NS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
static PHASE_BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const PHASE_LOG_EVERY: u64 = 250;

/// Add one phase's elapsed time to the running totals.
///
/// Recorded inside each phase rather than around the batch, because the phases are called
/// separately by anything that wants to hold a lock for only part of an insert -- timing them
/// from the outside would stop working the moment that happened, and these numbers are what
/// found the p99 tail in the first place.
fn record_phase(phase: usize, started: std::time::Instant) {
    use std::sync::atomic::Ordering::Relaxed;
    PHASE_NS[phase].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
}

/// One line per `PHASE_LOG_EVERY` batches, with the cumulative time in each phase.
fn report_phases(nodes: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let n = PHASE_BATCHES.fetch_add(1, Relaxed) + 1;
    if n.is_multiple_of(PHASE_LOG_EVERY) {
        let ms = |i: usize| PHASE_NS[i].load(Relaxed) as f64 / 1e6;
        tracing::info!(
            "insert_batch phases after {} batches: plan {:.0} ms, append {:.0} ms, \
             link_back {:.0} ms (nodes now {})",
            n,
            ms(0),
            ms(1),
            ms(2),
            nodes
        );
    }
}

/// A batch is planned in parallel only when it is at most this fraction of the graph:
/// `nodes / PARALLEL_PLAN_MAX_SHARE`. The rows of one batch are planned against the graph as
/// it stood before any of them, so they never choose each other as neighbours; while a batch
/// is a small share of the graph that costs nothing measurable, and when it is a large share
/// it costs real recall, because early records are what the rest of the graph hangs off.
///
/// This replaces a fixed floor of 10,000 nodes below which every row went in one at a time
/// on one thread. That floor was 14% of the CPU samples of a 100K build and a larger share
/// of its wall time, since nothing else ran meanwhile. Under the proportional rule the caller
/// chunks a batch to the share the graph allows -- 64 rows at 1,024 nodes, 625 at 10,000 --
/// and only the first `PARALLEL_PLAN_SEED` rows are serial.
///
/// An eighth, from 128 nodes, was tried first and failed two self-retrieval tests on
/// corpora of a thousand-odd rows: a graph that young has no backbone yet, and a chunk of
/// sixteen rows that cannot see each other is a sixteenth of it. A sixteenth from a thousand
/// nodes passes them and costs the 100K build under a second.
const PARALLEL_PLAN_MAX_SHARE: usize = 16;
/// Nodes the graph needs before any batch is planned in parallel.
const PARALLEL_PLAN_SEED: usize = 1_024;
/// `scan_nodes` takes the block scan once the match set is this dense (one position in
/// `FAST_SCAN_FROM_ONE_IN`).
const FAST_SCAN_FROM_ONE_IN: usize = 64;
/// Below this many records the rayon fan-out costs more than the serial loop.
const PARALLEL_PLAN_MIN_BATCH: usize = 16;
/// How many records a bulk build hands to `insert_batch` at a time.
///
/// Sized for parallelism alone: rayon can only use as many cores as there are, so a chunk past
/// a few hundred adds nothing, while every chunk boundary is one write-lock acquisition and one
/// serial append phase.
///
/// It would be natural to also cap this to protect recall -- records inside one chunk plan
/// against the graph as it stood before the chunk, so they cannot link to each other. Measured
/// at 1M x 768 that effect is not there to protect against: coalescing on (batches of ~380)
/// gave recall 0.9592, coalescing off (~100) 0.9584, and disabling batch planning altogether
/// so every record saw every predecessor gave 0.9572 -- a spread inside the noise band, and
/// pointing the wrong way. If a measurement ever shows `optimize` still behind, raising this
/// is safe on the evidence there is.
const BULK_BUILD_CHUNK: usize = 1024;

/// What [`HnswIndex::prepare_batch`] hands to the planning and appending phases.
pub struct PreparedBatch {
    items: Vec<(RecordId, DenseVector, usize)>,
    entry: usize,
    entry_level: usize,
}

/// What [`HnswIndex::apply_batch`] hands to the linking phase: the new nodes and the entry
/// level everything in the batch was planned against.
/// What `plan_batch` decided for a prepared batch: each node's neighbours per layer, and --
/// once the graph is quantized -- each node's code, encoded in the parallel planning phase
/// rather than one at a time under the write lock while appending. Encoding a 768-d vector
/// into 192 subspaces is about 200k multiplies; serially in the append phase that was
/// 0.44 ms per row, 440 s over a million.
pub struct BatchPlan {
    links: Vec<Vec<Vec<usize>>>,
    codes: Vec<Option<Vec<u8>>>,
}

pub struct AppliedBatch {
    nodes: Vec<(usize, usize)>,
    entry_level: usize,
}

#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    #[serde(skip, default = "default_rng")]
    rng: StdRng,
    config: HnswConfig,
    distance_metric: DistanceMetric,
    nodes: Nodes,
    /// Hash of the id to node index. A hash rather than the string: the string lives once,
    /// in the arena inside `nodes`, and this table costs about 24 bytes a node instead of
    /// a second copy of every id plus a `String` header.
    id_to_idx: IdHashMap<u32>,
    /// Ids whose hash collided with a different id's. Consulted after `id_to_idx` disagrees
    /// with the arena; in practice empty.
    id_overflow: HashMap<String, u32>,
    entry_point: Option<usize>,
    /// Present once the graph traverses on PQ codes instead of full vectors; see
    /// [`QuantizedStore`]. Files written without it stay readable as "hnsw-v2".
    quantized: Option<QuantizedStore>,
    /// The 4-bit block-scan codes beside the 8-bit ones, from the same quantization step;
    /// persisted as a sidecar by `HnswPqIndex`, never in this struct's own layout.
    #[serde(skip)]
    fast: Option<crate::search::quantization::fast_scan::FastScan>,
}

/// The graph's vectors as PQ codes, once [`HnswIndex::quantize`] has run.
///
/// After quantizing, `nodes[i].vector` is empty and every distance the graph needs comes from
/// here: query-to-node through an ADC table built once per query, node-to-node through a
/// per-subspace centroid-to-centroid table. That is what lets the full vectors leave the index
/// (and, later, memory): measured on 1M and 10M Cohere 768-d, a graph walked on 192-byte codes
/// with the best 500 reranked on the originals matched an exact walk at the same `ef` to within
/// 0.1 points of recall@100 (docs/architecture/disk-index.md §2).
///
/// `codes` is indexed by node position: node `i` owns bytes `[i*m, (i+1)*m)`. Tombstoned nodes
/// keep their slot, as they keep their `Node`.
#[derive(Serialize, Deserialize)]
pub struct QuantizedStore {
    pq: ProductQuantizer,
    m: usize,
    codes: Vec<u8>,
    /// The codebooks as one array, `flat[(sub * k + c) * d ..][..d]` = centroid `c` of
    /// subspace `sub`: what a code decodes through. Rebuilt from `pq` on load. The
    /// node-to-node distance used to come from a table of every centroid pair per subspace
    /// (`sub * k * k + a * k + b`): 6.5 MB at 25 dimensions and 50 MB at 768, so every one
    /// of its gathers missed L2, and it was 36% of a 25-d build's CPU. Decoding both codes
    /// (gathers into this array, 25 KB at 25-d) and taking a plain squared distance is the
    /// same number from cache-resident data.
    #[serde(skip)]
    flat: Vec<f32>,
    /// `sub * k` per subspace, as `i32` lanes for the gather path.
    #[serde(skip)]
    adc_bases: Vec<i32>,
}

/// Sum of `table[bases[sub] + code[sub]]` over subspaces: the ADC score of one code.
type AdcFn = fn(&[f32], &[u8], &[i32]) -> f32;

thread_local! {
    /// Two decoded vectors, for a node-to-node distance on a quantized graph.
    static DECODED: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Eight independent partial sums. The obvious loop is one dependent chain of float adds --
/// 192 of them at four cycles each, about 250 ns per candidate before a single cache miss --
/// and that chain, not the loads, was what made the walk on codes twice as slow as the
/// exact one (DISK_INDEX.md section 2.7).
#[allow(clippy::needless_range_loop)] // the lane index is the unroll
fn adc_scalar(table: &[f32], code: &[u8], bases: &[i32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (chunks, rest) = code.as_chunks::<8>();
    let (base_chunks, _) = bases.as_chunks::<8>();
    for (c, b) in chunks.iter().zip(base_chunks) {
        for lane in 0..8 {
            acc[lane] += table[(b[lane] + c[lane] as i32) as usize];
        }
    }
    let mut total = acc.iter().sum::<f32>();
    let full = code.len() - rest.len();
    for (i, &c) in rest.iter().enumerate() {
        total += table[(bases[full + i] + c as i32) as usize];
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn adc_avx2(table: &[f32], code: &[u8], bases: &[i32]) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let full = code.len() - code.len() % 8;
    let mut sub = 0;
    while sub < full {
        // Eight code bytes -> eight i32 lanes, plus each lane's subspace base, gathered
        // from the table in one instruction.
        let bytes = _mm_loadl_epi64(code.as_ptr().add(sub) as *const __m128i);
        let idx = _mm256_add_epi32(
            _mm256_cvtepu8_epi32(bytes),
            _mm256_loadu_si256(bases.as_ptr().add(sub) as *const __m256i),
        );
        acc = _mm256_add_ps(acc, _mm256_i32gather_ps::<4>(table.as_ptr(), idx));
        sub += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut total = lanes.iter().sum::<f32>();
    for s in full..code.len() {
        total += table[(bases[s] + code[s] as i32) as usize];
    }
    total
}

/// Resolved once, like `simd_distance::dot_product`: the feature check is off the hot path.
fn adc_fn() -> AdcFn {
    static F: std::sync::OnceLock<AdcFn> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return |t, c, b| unsafe { adc_avx2(t, c, b) };
            }
        }
        adc_scalar
    })
}

impl QuantizedStore {
    fn from_quantizer(pq: ProductQuantizer) -> Self {
        let m = pq.num_subvectors;
        let mut store = Self {
            pq,
            m,
            codes: Vec::new(),
            flat: Vec::new(),
            adc_bases: Vec::new(),
        };
        store.rebuild_tables();
        store
    }

    fn rebuild_tables(&mut self) {
        let k = self.pq.num_centroids;
        let d = self.pq.subvector_dim;
        let mut flat = Vec::with_capacity(self.m * k * d);
        for codebook in &self.pq.codebooks {
            for centroid in codebook {
                flat.extend_from_slice(centroid);
            }
        }
        self.flat = flat;
        self.adc_bases = (0..self.m).map(|sub| (sub * k) as i32).collect();
    }

    /// The vector a node's code stands for, written into `out` (`dim` long).
    #[inline]
    fn decode_into(&self, idx: usize, out: &mut [f32]) {
        let k = self.pq.num_centroids;
        let d = self.pq.subvector_dim;
        let code = self.code(idx);
        // A subspace is one dimension below 64 and two below 256 (`default_subvectors`), and
        // a `copy_from_slice` of four or eight bytes is a call to memcpy per subspace -- at
        // 25 dimensions that was most of the decode. Index directly for the narrow cases.
        match d {
            1 => {
                for (sub, &c) in code.iter().enumerate() {
                    out[sub] = self.flat[sub * k + c as usize];
                }
            }
            2 => {
                for (sub, &c) in code.iter().enumerate() {
                    let at = (sub * k + c as usize) * 2;
                    out[sub * 2] = self.flat[at];
                    out[sub * 2 + 1] = self.flat[at + 1];
                }
            }
            _ => {
                for (sub, &c) in code.iter().enumerate() {
                    let at = (sub * k + c as usize) * d;
                    out[sub * d..(sub + 1) * d].copy_from_slice(&self.flat[at..at + d]);
                }
            }
        }
    }

    #[inline]
    fn dim(&self) -> usize {
        self.m * self.pq.subvector_dim
    }

    #[inline]
    fn code(&self, idx: usize) -> &[u8] {
        &self.codes[idx * self.m..(idx + 1) * self.m]
    }

    /// The code's bytes, for a prefetch hint; empty past the end rather than a panic.
    #[inline]
    fn code_bytes(&self, idx: usize) -> &[u8] {
        let start = idx * self.m;
        self.codes.get(start..start + self.m).unwrap_or(&[])
    }

    fn push(&mut self, vector: &[f32]) -> crate::Result<()> {
        let code = self.encode(vector);
        self.codes.extend_from_slice(&code);
        Ok(())
    }

    /// The code for one (metric-prepared) vector. Infallible: the graph checks dimensions
    /// before a vector reaches it.
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        self.pq
            .encode(&vector.to_vec())
            .expect("vector was dimension-checked before reaching the graph")
            .codes
    }

    /// Squared distance between two nodes: both codes decoded, then one SIMD pass.
    #[inline]
    fn node_distance(&self, a: usize, b: usize) -> f32 {
        let dim = self.dim();
        DECODED.with(|cell| {
            let mut bufs = cell.borrow_mut();
            let (va, vb) = &mut *bufs;
            va.resize(dim, 0.0);
            vb.resize(dim, 0.0);
            self.decode_into(a, va);
            self.decode_into(b, vb);
            crate::search::simd_distance::squared_euclidean(va, vb)
        })
    }

    #[inline]
    fn adc(&self, table: &[f32], idx: usize) -> f32 {
        adc_fn()(table, self.code(idx), &self.adc_bases)
    }

    fn resident_bytes(&self) -> usize {
        self.codes.len() + self.pq.memory_usage() + self.flat.len() * std::mem::size_of::<f32>()
    }
}

/// How a query scores the nodes it visits. Built once per search by [`HnswIndex::scorer`].
///
/// Exact mode compares against the vector stored on the node; quantized mode compares an ADC
/// table (squared distance from each query subvector to each centroid) against the node's
/// code. Both rank the same way as the metric on unit-length inputs; only the exact one returns
/// the metric's own value, which is why quantized results are reranked before they leave.
pub enum QueryScorer<'a> {
    Exact(&'a DenseVector),
    Adc(Vec<f32>),
}

/// The "hnsw-v2" file layout, read-only.
#[derive(Deserialize)]
struct LegacyV2 {
    config: HnswConfig,
    distance_metric: DistanceMetric,
    nodes: Vec<LegacyNode>,
    #[allow(dead_code)]
    id_to_idx: HashMap<RecordId, usize>,
    entry_point: Option<usize>,
}

/// The "hnsw-v3" file layout (v2 plus the quantized store), read-only.
#[derive(Deserialize)]
struct LegacyV3 {
    config: HnswConfig,
    distance_metric: DistanceMetric,
    nodes: Vec<LegacyNode>,
    #[allow(dead_code)]
    id_to_idx: HashMap<RecordId, usize>,
    entry_point: Option<usize>,
    quantized: Option<QuantizedStore>,
}

fn default_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

impl HnswIndex {
    pub fn new(distance_metric: DistanceMetric, config: Option<HnswConfig>) -> Self {
        let mut config = config.unwrap_or_default();
        if config.max_m0 > MAX_M0_SLOTS {
            tracing::warn!(
                "HNSW max_m0 {} exceeds the {} layer-0 slots per node; clamping",
                config.max_m0,
                MAX_M0_SLOTS
            );
            config.max_m0 = MAX_M0_SLOTS;
        }
        Self {
            rng: StdRng::seed_from_u64(config.seed),
            config,
            distance_metric,
            nodes: Nodes::new(),
            id_to_idx: Default::default(),
            id_overflow: HashMap::new(),
            entry_point: None,
            quantized: None,
            fast: None,
        }
    }

    /// Node index of `id`, if present and live.
    fn idx_of(&self, id: &str) -> Option<usize> {
        if let Some(&idx) = self.id_to_idx.get(&id_hash(id)) {
            if bytes_eq(self.nodes.id_raw(idx as usize), id) {
                return Some(idx as usize);
            }
        }
        self.id_overflow.get(id).map(|&i| i as usize)
    }

    fn register_id(&mut self, id: &str, idx: usize) {
        match self.id_to_idx.entry(id_hash(id)) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(idx as u32);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                self.id_overflow.insert(id.to_string(), idx as u32);
            }
        }
    }

    fn unregister_id(&mut self, id: &str) {
        let h = id_hash(id);
        if self
            .id_to_idx
            .get(&h)
            .is_some_and(|&i| bytes_eq(self.nodes.id_raw(i as usize), id))
        {
            self.id_to_idx.remove(&h);
        } else {
            self.id_overflow.remove(id);
        }
    }

    /// Given a legacy node list and an entry point, the rest of the index.
    fn from_legacy(
        config: HnswConfig,
        distance_metric: DistanceMetric,
        nodes: Vec<LegacyNode>,
        entry_point: Option<usize>,
        quantized: Option<QuantizedStore>,
    ) -> Self {
        let nodes = Nodes::from_legacy(nodes);
        let mut index = HnswIndex {
            rng: default_rng(),
            config,
            distance_metric,
            nodes,
            id_to_idx: Default::default(),
            id_overflow: HashMap::new(),
            fast: None,
            entry_point,
            quantized,
        };
        for idx in 0..index.nodes.len() {
            if !index.nodes.is_deleted(idx) {
                let id = index.nodes.id(idx).to_string();
                index.register_id(&id, idx);
            }
        }
        if let Some(store) = &mut index.quantized {
            store.rebuild_tables();
        }
        index
    }

    /// Build the scorer for one (already `prepare`d) query.
    fn scorer<'a>(&self, query: &'a DenseVector) -> QueryScorer<'a> {
        match &self.quantized {
            None => QueryScorer::Exact(query),
            Some(q) => {
                let k = q.pq.num_centroids;
                let dim = q.pq.subvector_dim;
                let mut table = vec![0.0f32; q.m * k];
                for (sub, codebook) in q.pq.codebooks.iter().enumerate() {
                    let qs = &query[sub * dim..(sub + 1) * dim];
                    for (c, centroid) in codebook.iter().enumerate() {
                        table[sub * k + c] = qs
                            .iter()
                            .zip(centroid)
                            .map(|(x, y)| (x - y) * (x - y))
                            .sum();
                    }
                }
                QueryScorer::Adc(table)
            }
        }
    }

    /// Distance from the query to node `idx` under `scorer`.
    /// Ask the cache for node `idx`'s scoring data ahead of its use. On x86-64 a hint per
    /// cache line of the vector, or of the PQ code when the graph is quantized; elsewhere
    /// nothing. A hint for an index past the end is harmless, so the bound is the only check.
    #[inline]
    fn prefetch(&self, idx: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            if idx >= self.nodes.len() {
                return;
            }
            let bytes: &[u8] = match &self.quantized {
                Some(store) => store.code_bytes(idx),
                None => {
                    let v = self.nodes.vector(idx);
                    // SAFETY: a reinterpretation for address arithmetic only; nothing reads it.
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
                }
            };
            let mut offset = 0;
            while offset < bytes.len() {
                // SAFETY: prefetch takes any address and never faults.
                unsafe { _mm_prefetch(bytes.as_ptr().add(offset) as *const i8, _MM_HINT_T0) };
                offset += 64;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = idx;
        }
    }

    #[inline]
    fn score(&self, scorer: &QueryScorer<'_>, idx: usize) -> f32 {
        match scorer {
            QueryScorer::Exact(q) => self.calculate_distance(q, self.nodes.vector(idx)),
            QueryScorer::Adc(table) => self
                .quantized
                .as_ref()
                .expect("an ADC scorer exists only for a quantized graph")
                .adc(table, idx),
        }
    }

    /// Distance between two nodes, for neighbour selection and pruning.
    #[inline]
    fn node_distance(&self, a: usize, b: usize) -> f32 {
        match &self.quantized {
            None => self.calculate_distance(self.nodes.vector(a), self.nodes.vector(b)),
            Some(q) => q.node_distance(a, b),
        }
    }

    /// Run `f` with a distance function from node `a`: on a quantized graph `a` is decoded
    /// once for the whole call rather than once per distance. Every pruning loop is one node
    /// against its links, which is the shape this is for.
    #[inline]
    fn with_distances_from<R>(
        &self,
        a: usize,
        f: impl FnOnce(&mut dyn FnMut(usize) -> f32) -> R,
    ) -> R {
        match &self.quantized {
            None => {
                let va = self.nodes.vector(a);
                let mut dist = |b: usize| self.calculate_distance(va, self.nodes.vector(b));
                f(&mut dist)
            }
            Some(q) => {
                let dim = q.dim();
                let mut va = vec![0.0f32; dim];
                let mut vb = vec![0.0f32; dim];
                q.decode_into(a, &mut va);
                let mut dist = |b: usize| {
                    q.decode_into(b, &mut vb);
                    crate::search::simd_distance::squared_euclidean(&va, &vb)
                };
                f(&mut dist)
            }
        }
    }

    /// Restore what serialisation skips (the centroid table); `load` does this itself, a
    /// wrapper that deserialises this graph inside its own struct has to call it.
    pub fn after_load(&mut self) {
        if let Some(store) = &mut self.quantized {
            store.rebuild_tables();
        }
    }

    /// Up to `want` of the stored vectors at a fixed stride, for training a quantizer on a
    /// graph that still holds them. Empty once quantized.
    pub fn sample_vectors(&self, want: usize) -> Vec<DenseVector> {
        if self.quantized.is_some() || self.nodes.is_empty() {
            return Vec::new();
        }
        let stride = (self.nodes.len() / want.max(1)).max(1);
        (0..self.nodes.len())
            .step_by(stride)
            .filter(|&i| !self.nodes.is_deleted(i))
            .take(want)
            .map(|i| self.nodes.vector(i).to_vec())
            .collect()
    }

    /// Whether the graph traverses on PQ codes.
    pub fn is_quantized(&self) -> bool {
        self.quantized.is_some()
    }

    /// Switch the graph to PQ codes: encode every node, drop the stored vectors.
    ///
    /// `pq` must be trained on this graph's metric-prepared vectors (unit length for cosine).
    /// The graph itself is untouched -- links chosen with exact distances stay -- and every
    /// later insert plans with ADC against the codes. A graph built on approximate distances
    /// from the start measured no worse (DISK_INDEX.md section 2.5).
    pub fn quantize(
        &mut self,
        pq: ProductQuantizer,
        fast: Option<crate::search::quantization::fast_scan::FastScan>,
    ) -> crate::Result<()> {
        if self.quantized.is_some() {
            return Err(crate::Error::InvalidInput(
                "HNSW graph is already quantized".to_string(),
            ));
        }
        let mut store = QuantizedStore::from_quantizer(pq);
        store.codes.reserve(self.nodes.len() * store.m);
        // Encoded on every core: serially, a million 768-d rows took 442 s.
        let codes: Vec<Vec<u8>> = (0..self.nodes.len())
            .into_par_iter()
            .map(|idx| store.encode(self.nodes.vector(idx)))
            .collect();
        for code in codes {
            store.codes.extend_from_slice(&code);
        }
        // The block-scan codes, likewise, before the vectors go.
        if let Some(mut fast) = fast {
            let codes: Vec<Vec<u8>> = (0..self.nodes.len())
                .into_par_iter()
                .map(|idx| fast.encode(self.nodes.vector(idx)))
                .collect::<crate::Result<_>>()?;
            for code in &codes {
                fast.push_code(code);
            }
            self.fast = Some(fast);
        }
        self.nodes.drop_vectors();
        self.quantized = Some(store);
        Ok(())
    }

    /// The block-scan codes, when the graph has them.
    pub fn fast_scan(&self) -> Option<&crate::search::quantization::fast_scan::FastScan> {
        self.fast.as_ref()
    }

    pub fn set_fast_scan(
        &mut self,
        fast: Option<crate::search::quantization::fast_scan::FastScan>,
    ) {
        self.fast = fast;
    }

    pub fn has_fast_scan(&self) -> bool {
        self.fast.is_some()
    }

    /// Bytes the graph holds beyond its vectors: links, plus codes and tables once quantized.
    pub fn resident_bytes_without_vectors(&self) -> usize {
        self.nodes.resident_bytes_without_vectors()
            + self.id_to_idx.capacity() * (std::mem::size_of::<u64>() + std::mem::size_of::<u32>())
            + self
                .quantized
                .as_ref()
                .map(|q| q.resident_bytes())
                .unwrap_or(0)
            + self.fast.as_ref().map(|f| f.code_bytes()).unwrap_or(0)
    }

    /// Put a vector into the form this index scores in.
    ///
    /// For cosine that means unit length. Cosine similarity is `dot(a,b) / (‖a‖·‖b‖)`, and
    /// computing it that way costs three passes over both vectors and two square roots on *every*
    /// distance evaluation — which is the inner loop of both search and construction. Normalising
    /// once on the way in leaves a plain dot product, one pass and no roots — the normalisation
    /// moves out of the inner loop and into a one-time preprocessing step.
    ///
    /// This only touches the copy the index keeps. Records are stored separately by the storage
    /// layer, so what the API hands back is still what the caller sent.
    ///
    /// Every vector entering the index and every query has to come through here, or the two sides
    /// of a comparison are on different scales and the scores are quietly wrong.
    fn prepare(&self, mut v: DenseVector) -> DenseVector {
        if matches!(self.distance_metric, DistanceMetric::Cosine) {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            // A zero vector has no direction; leave it alone rather than producing NaNs. It
            // scores 0 against everything either way.
            if norm > 0.0 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
        }
        v
    }

    fn calculate_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // Cosine inputs are unit length by construction (see `prepare`), so the similarity is the
        // dot product and the distance is one minus it.
        match self.distance_metric {
            DistanceMetric::Cosine => {
                return 1.0 - crate::search::simd_distance::dot_product(a, b);
            }
            // Resolved once, like the dot product: the trait path below re-runs the CPU
            // feature check and two dispatch levels on every call.
            DistanceMetric::Euclidean => {
                return crate::search::simd_distance::euclidean_distance(a, b);
            }
            DistanceMetric::Dot => {
                return -crate::search::simd_distance::dot_product(a, b);
            }
            DistanceMetric::Manhattan => {}
        }
        self.distance_metric
            .calculate_simd(a, b)
            .unwrap_or_else(|_| {
                // Fallback to scalar calculation if SIMD fails
                match self.distance_metric {
                    DistanceMetric::Cosine => {
                        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                        1.0 - (dot / (norm_a * norm_b))
                    }
                    DistanceMetric::Euclidean => a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| (x - y).powi(2))
                        .sum::<f32>()
                        .sqrt(),
                    DistanceMetric::Dot => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
                    _ => 0.0,
                }
            })
    }

    /// Pick up to `m` neighbours out of `candidates`, keeping the graph navigable.
    ///
    /// Algorithm 4 from the HNSW paper. `candidates` carries each node's distance to the point
    /// being linked, and a candidate is kept only if it is closer to that point than to any
    /// neighbour already kept — so the chosen set spreads around it rather than piling into
    /// whichever direction happens to be densest.
    ///
    /// The obvious alternative — take the `m` nearest — is what this replaces. In a cluster a
    /// node's `m` nearest all sit on the same side of it, the graph loses the long edges that let
    /// a search cross between clusters, and a query settles into whichever cluster it landed in.
    /// Measured on 40,000 clustered 128-d vectors, recall@10 was **0.38 and did not move with
    /// `ef` at all** — 0.378 at ef=100 through 0.382 at ef=1000, the signature of true neighbours
    /// the search cannot reach however long it looks. With this heuristic: 0.62 at ef=100 rising
    /// to 0.85 at ef=1000. It costs roughly a quarter of build time.
    ///
    /// Uniformly random vectors cannot show any of that — no clusters, nothing for the heuristic
    /// to fix — and measuring on them alone said it bought nothing. Worth remembering before
    /// benchmarking an ANN change on uniform data again.
    ///
    /// **This returns fewer than `m` when candidates crowd each other, and that is left as
    /// it is.** The paper's `keepPrunedConnections` — filling the remainder from the rejects
    /// — was implemented here, measured on three corpora, and removed again:
    ///
    /// | corpus | unreachable without | with | ingest |
    /// | --- | --- | --- | --- |
    /// | GIST-960, 100K rows | 2.0% | 0.1% | 1.8x slower |
    /// | bge-m3 1024d, 60K real paragraphs | 0.00% | 0.00% | unchanged |
    /// | OpenAI 3-large 3072d, 60K paragraphs | 0.00% | — | — |
    /// | clustered synthetic, 3K x 64 | — | — | self-retrieval 98+ to 82–91 |
    ///
    /// It helps only on GIST, which is a 2008 image descriptor with near-duplicates and
    /// isolated points. **On what a modern embedding model produces — which is what anyone
    /// actually stores — nothing is stranded at all**, so there is nothing for it to fix; and
    /// it measurably degrades a clustered corpus, because the edges it adds are precisely the
    /// ones this heuristic rejected for being redundant.
    ///
    /// Those zeroes were 0.50% until the measurement was checked. Comparing the graph's score
    /// against the exact scan's with a tolerance of `1e-6` is tighter than the floating-point
    /// noise of summing 1024 products, let alone 3072: at 3072 dimensions the median gap is
    /// 2e-6 and the criterion reported **87.9% unreachable on a corpus with recall@10 of
    /// 1.0000**. Two numbers that cannot both be true is what exposed it. Compare the returned
    /// *id*, or use a tolerance above the metric's own noise. Trading a real regression on one workload for a fix on another, while the
    /// representative workload is indifferent, is not a trade worth making.
    ///
    /// A narrower version — rescuing a node at insert time when every neighbour had just
    /// pruned it away — was also tried: GIST went 2.0% to 1.6% only. Stranding does not happen
    /// at insert, when the node still has in-edges. It happens a thousand inserts later as
    /// others crowd it out one list at a time, and nothing revisits it.
    ///
    /// **So a small fraction of points is unreachable by the graph, by design of this
    /// heuristic**: about 0.5% on modern embeddings and 2% on GIST. They are the points
    /// nothing else is near, so they are nobody's nearest neighbour and recall is unaffected
    /// (0.9955 on GIST while 5% was unreachable at a million rows). What does not work is
    /// searching with a vector that is already stored and expecting it back.
    /// `search_mode: "exact"` is the deterministic answer, and `index_record_count` on
    /// `GET /tables/{t}/stats` is how to tell "the graph cannot reach it" from "the graph
    /// never got it".
    ///
    /// `candidates` must be sorted nearest-first, which is what `search_layer` returns.
    fn select_neighbors_heuristic(&self, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(m);
        // On a quantized graph each candidate is decoded once and the kept set is held
        // decoded, so a candidate-against-kept comparison is one SIMD pass and no gather.
        let Some(store) = &self.quantized else {
            for &(idx, dist_to_base) in candidates {
                if selected.len() >= m {
                    break;
                }
                let crowded = selected
                    .iter()
                    .any(|&kept| self.node_distance(idx, kept) < dist_to_base);
                if !crowded {
                    selected.push(idx);
                }
            }
            return selected;
        };
        let dim = store.dim();
        let mut kept: Vec<f32> = Vec::with_capacity(m * dim);
        let mut candidate = vec![0.0f32; dim];
        for &(idx, dist_to_base) in candidates {
            if selected.len() >= m {
                break;
            }
            store.decode_into(idx, &mut candidate);
            let crowded = kept.chunks_exact(dim).any(|k| {
                crate::search::simd_distance::squared_euclidean(&candidate, k) < dist_to_base
            });
            if !crowded {
                selected.push(idx);
                kept.extend_from_slice(&candidate);
            }
        }
        selected
    }

    /// Draw a node's top layer from the exponential distribution HNSW is built on.
    ///
    /// `-ln(u) * ml` for uniform `u`, which is what the paper specifies. `ml` is a *scale
    /// factor*, `1/ln(m)`, not a probability.
    ///
    /// This used to be a coin-flip loop, `while rng.gen::<f64>() < ml`, comparing a uniform in
    /// [0,1) against a factor of about 1.44 — a condition that is simply always true. Every node
    /// landed on layer 16. That is not a hierarchy: all seventeen layers held every node, so each
    /// insert ran seventeen full-width beam searches and each search descended through seventeen
    /// copies of the same flat graph. Build time came out super-linear (2.5-2.75x for 2x rows on
    /// random vectors) rather than the n·log n the structure exists to provide.
    fn select_level(&mut self) -> usize {
        let u: f64 = self.rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let level = (-u.ln() * self.config.ml).round() as usize;
        level.min(self.config.max_level)
    }

    /// Search at a specific layer with proper beam search.
    /// When `filter_idxs` is Some, only nodes present in the set are added to
    /// the candidate result set (Acorn-style filter-in-graph). Non-matching
    /// neighbors are still marked visited to prevent repeated re-exploration.
    fn search_layer(
        &self,
        scorer: &QueryScorer<'_>,
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
        gate: Option<&dyn NodeGate>,
    ) -> Vec<(usize, f32)> {
        self.search_layer_by(|idx| self.score(scorer, idx), entry_points, ef, layer, gate)
    }

    /// `search_layer` with the distance supplied by the caller, per node index.
    ///
    /// The traversal does not care where a distance comes from. Scoring a node from a
    /// compressed copy of its vector -- an ADC table over PQ codes, say -- instead of from the
    /// full vector stored on the node is what lets the full vectors leave memory, and the
    /// route probe measures what that costs in recall before the index is changed to do it.
    fn search_layer_by(
        &self,
        score: impl Fn(usize) -> f32,
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
        gate: Option<&dyn NodeGate>,
    ) -> Vec<(usize, f32)> {
        let mut visited = visited::Visited::start(self.nodes.len());
        let mut candidates = BinaryHeap::new(); // Min-heap for exploration
        let mut w = BinaryHeap::new(); // Max-heap for results
        let mut links: Vec<u32> = Vec::with_capacity(MAX_M0_SLOTS);

        // Initialize with entry points
        for idx in entry_points {
            if idx >= self.nodes.len() || self.nodes.is_deleted(idx) {
                continue; // Skip deleted nodes
            }
            // An entry point that fails the filter is still the way into the graph: it goes on
            // the exploration queue, and only the *result* set is gated on the filter. Gating
            // both is how a filtered search returns nothing — with the queue empty the loop
            // below never runs, so a selective filter came back empty for every query whose
            // entry point happened not to match. At 2% selectivity that was 98% of queries.
            let dist = score(idx);
            candidates.push(Candidate {
                idx,
                distance: dist,
            });
            if gate.is_none_or(|g| g.allows(idx)) {
                w.push(Result {
                    idx,
                    distance: dist,
                });
            }
            visited.insert(idx);
        }

        // Main search loop
        while let Some(current) = candidates.pop() {
            // Check stopping condition: current candidate is farther than worst result
            // and we have enough results
            if w.len() >= ef {
                let worst = w.peek().unwrap();
                if current.distance > worst.distance {
                    break;
                }
            }

            // Explore neighbors
            if layer < self.nodes.layers(current.idx) {
                // The edge list is copied out under its read lock and the lock released
                // before any neighbour is scored, so a search never holds a lock while it
                // computes, and a writer pruning some node never waits on a reader's walk.
                self.nodes.copy_links(current.idx, layer, &mut links);
                // The edge list is known before any neighbour is scored, so the next
                // neighbour's vector can be on its way while this one is being scored. The
                // walk's cost is this load: a 100-dimension vector is seven cache lines at a
                // random address in a flat array larger than L2, and with nothing in flight
                // each visit paid the full miss -- the in-process profile charged 63% of the
                // build's CPU to the distance kernel, and a kernel four times faster at the
                // arithmetic moved the build rate not at all. hnswlib prefetches here for
                // the same reason.
                for (at, &neighbor_idx) in links.iter().enumerate() {
                    if let Some(&next) = links.get(at + 1) {
                        self.prefetch(next as usize);
                    }
                    let neighbor_idx = neighbor_idx as usize;
                    if neighbor_idx >= self.nodes.len()
                        || visited.contains(neighbor_idx)
                        || self.nodes.is_deleted(neighbor_idx)
                    {
                        continue;
                    }
                    // Mark visited regardless — prevents redundant re-visits
                    visited.insert(neighbor_idx);

                    let dist = score(neighbor_idx);

                    // Always add to exploration candidates so graph can bridge through non-matching nodes
                    if dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY) || w.len() < ef
                    {
                        candidates.push(Candidate {
                            idx: neighbor_idx,
                            distance: dist,
                        });
                    }

                    // Only add to result window if this node passes the filter
                    if gate.is_none_or(|g| g.allows(neighbor_idx))
                        && (dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY)
                            || w.len() < ef)
                    {
                        w.push(Result {
                            idx: neighbor_idx,
                            distance: dist,
                        });
                        // Prune results if we have too many
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vector, filtering out deleted nodes
        let mut results: Vec<_> = w
            .into_iter()
            .filter(|r| !self.nodes.is_deleted(r.idx))
            .map(|r| (r.idx, r.distance))
            .collect();
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    /// Parallel search at layer 0 for improved throughput.
    /// Accepts the same `filter_idxs` parameter as `search_layer` for Acorn filtering.
    fn search_layer_parallel(
        &self,
        scorer: &QueryScorer<'_>,
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
        gate: Option<&dyn NodeGate>,
    ) -> Vec<(usize, f32)> {
        let mut visited = visited::Visited::start(self.nodes.len());
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        let mut links: Vec<u32> = Vec::with_capacity(MAX_M0_SLOTS);

        // Initialize with entry points
        for idx in entry_points {
            if idx >= self.nodes.len() || self.nodes.is_deleted(idx) {
                continue;
            }
            // Same rule as `search_layer`: the queue is unfiltered, the result set is not.
            let dist = self.score(scorer, idx);
            candidates.push(Candidate {
                idx,
                distance: dist,
            });
            if gate.is_none_or(|g| g.allows(idx)) {
                w.push(Result {
                    idx,
                    distance: dist,
                });
            }
            visited.insert(idx);
        }

        // Main search loop with parallel distance calculations
        while let Some(current) = candidates.pop() {
            if w.len() >= ef {
                let worst = w.peek().unwrap();
                if current.distance > worst.distance {
                    break;
                }
            }

            if layer < self.nodes.layers(current.idx) {
                self.nodes.copy_links(current.idx, layer, &mut links);
                let candidate_neighbors: Vec<usize> = links
                    .iter()
                    .map(|&n| n as usize)
                    .filter(|&neighbor_idx| {
                        neighbor_idx < self.nodes.len()
                            && !visited.contains(neighbor_idx)
                            && !self.nodes.is_deleted(neighbor_idx)
                    })
                    .collect();

                // Mark all as visited before computing distances (prevents re-entry)
                for &idx in &candidate_neighbors {
                    visited.insert(idx);
                }

                // Now compute distances only for filter-matching nodes
                let neighbor_distances: Vec<_> = if candidate_neighbors.len() > 8 {
                    candidate_neighbors
                        .par_iter()
                        .filter(|&&idx| gate.is_none_or(|g| g.allows(idx)))
                        .map(|&neighbor_idx| {
                            let dist = self.score(scorer, neighbor_idx);
                            (neighbor_idx, dist)
                        })
                        .collect()
                } else {
                    candidate_neighbors
                        .iter()
                        .filter(|&&idx| gate.is_none_or(|g| g.allows(idx)))
                        .map(|&neighbor_idx| {
                            let dist = self.score(scorer, neighbor_idx);
                            (neighbor_idx, dist)
                        })
                        .collect()
                };

                // Process computed distances from filter-passing neighbors
                // (candidate_neighbors were already all marked visited above)
                for (neighbor_idx, dist) in neighbor_distances {
                    // These already passed the filter check in the collection above
                    if dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY) || w.len() < ef
                    {
                        candidates.push(Candidate {
                            idx: neighbor_idx,
                            distance: dist,
                        });
                        w.push(Result {
                            idx: neighbor_idx,
                            distance: dist,
                        });
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }

                // Also add non-filter-matching candidates for graph traversal (bridges)
                if let Some(g) = gate {
                    for &idx in candidate_neighbors.iter().filter(|&&i| !g.allows(i)) {
                        let dist = self.score(scorer, idx);
                        if dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY)
                            || candidates.len() < ef
                        {
                            candidates.push(Candidate {
                                idx,
                                distance: dist,
                            });
                        }
                    }
                }
            }
        }

        let mut results: Vec<_> = w
            .into_iter()
            .filter(|r| !self.nodes.is_deleted(r.idx))
            .map(|r| (r.idx, r.distance))
            .collect();
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    /// Greedy search for upper layers (find single closest point)
    fn search_layer_closest(
        &self,
        scorer: &QueryScorer<'_>,
        entry_points: Vec<usize>,
        layer: usize,
    ) -> Vec<usize> {
        self.search_layer_closest_by(|idx| self.score(scorer, idx), entry_points, layer)
    }

    /// `search_layer_closest` with the distance supplied by the caller, per node index.
    fn search_layer_closest_by(
        &self,
        score: impl Fn(usize) -> f32,
        entry_points: Vec<usize>,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = visited::Visited::start(self.nodes.len());
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        let mut links: Vec<u32> = Vec::with_capacity(MAX_M0_SLOTS);

        // Initialize
        for idx in entry_points {
            if idx >= self.nodes.len() || self.nodes.is_deleted(idx) {
                continue; // Skip deleted nodes
            }
            let dist = score(idx);
            candidates.push(Candidate {
                idx,
                distance: dist,
            });
            w.push(Result {
                idx,
                distance: dist,
            });
            visited.insert(idx);
        }

        // Search
        while let Some(current) = candidates.pop() {
            // Stop if current is worse than best
            if let Some(best) = w.peek() {
                if current.distance > best.distance {
                    break;
                }
            }

            if layer < self.nodes.layers(current.idx) {
                self.nodes.copy_links(current.idx, layer, &mut links);
                for &neighbor_idx in links.iter() {
                    let neighbor_idx = neighbor_idx as usize;
                    if neighbor_idx >= self.nodes.len()
                        || visited.contains(neighbor_idx)
                        || self.nodes.is_deleted(neighbor_idx)
                    {
                        continue; // Skip deleted nodes
                    }
                    visited.insert(neighbor_idx);

                    let dist = score(neighbor_idx);

                    // Update if better
                    if dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY) {
                        candidates.push(Candidate {
                            idx: neighbor_idx,
                            distance: dist,
                        });
                        w.clear();
                        w.push(Result {
                            idx: neighbor_idx,
                            distance: dist,
                        });
                    }
                }
            }
        }

        w.into_iter().map(|r| r.idx).collect()
    }

    pub fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    /// Search with explicit ef_search override. Used by SearchMode to tune recall/speed.
    pub fn search_with_ef(
        &self,
        query: &DenseVector,
        k: usize,
        ef_override: usize,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }
        let query = &self.prepare(query.clone());
        let scorer = self.scorer(query);

        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes.level(entry);
        let mut curr_nearest = vec![entry];

        // Search from top layer to layer 1 (greedy search for closest point)
        for lc in (1..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(&scorer, curr_nearest, lc);
        }

        // Rayon has a non-trivial scheduling cost for the small neighbor lists
        // in a normal HNSW graph (M=16 => at most 32 neighbors at layer 0).
        // The old `>1000 nodes` switch therefore made every 10K query slower
        // than the scalar traversal.  Keep the parallel path for filtered
        // searches, where it can avoid scoring rejected candidates, but use
        // the cache-friendly sequential loop for the common unfiltered case.
        let ef = ef_override.max(k);
        let candidates = self.search_layer(&scorer, curr_nearest, ef, 0, None);

        // Return top k
        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes.id(idx).to_string(), dist))
            .collect())
    }

    /// Acorn-style filter-in-graph search.
    ///
    /// Converts `filter_ids` (record ID strings) to node indices and performs
    /// in-graph filtering: non-matching nodes are never added to the result window.
    /// The graph traversal still uses all nodes as bridges to reach matching ones.
    ///
    /// Falls back to regular search when the index is empty or `filter_ids` is empty.
    pub fn search_with_filter(
        &self,
        query: &DenseVector,
        k: usize,
        ef_override: usize,
        filter_ids: &HashSet<String>,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        if self.entry_point.is_none() || filter_ids.is_empty() {
            return self.search_with_ef(query, k, ef_override);
        }

        // Convert string record IDs to node indices for O(1) filter checks
        let filter_idxs: HashSet<usize> =
            filter_ids.iter().filter_map(|id| self.idx_of(id)).collect();

        if filter_idxs.is_empty() {
            return Ok(Vec::new());
        }
        let query = &self.prepare(query.clone());
        let scorer = self.scorer(query);

        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes.level(entry);
        let mut curr_nearest = vec![entry];

        // Upper layers: greedy traversal without filter (find good entry to layer 0)
        for lc in (1..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(&scorer, curr_nearest, lc);
        }

        // Layer 0: apply Acorn filter
        let ef = ef_override.max(k);
        let gate: &dyn NodeGate = &filter_idxs;
        let candidates = if self.nodes.len() > 1000 {
            self.search_layer_parallel(&scorer, curr_nearest, ef, 0, Some(gate))
        } else {
            self.search_layer(&scorer, curr_nearest, ef, 0, Some(gate))
        };

        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes.id(idx).to_string(), dist))
            .collect())
    }

    /// The node positions of `ids`, as a bitmap: one hash per id, no allocation per id.
    /// Ids the graph does not hold, and deleted nodes, are left out.
    pub fn node_filter<'a>(&self, ids: impl IntoIterator<Item = &'a str>) -> NodeFilter {
        let mut filter = NodeFilter::with_capacity(self.nodes.len());
        for id in ids {
            if let Some(idx) = self.idx_of(id) {
                if !self.nodes.is_deleted(idx) {
                    filter.insert(idx);
                }
            }
        }
        filter
    }

    /// [`Self::scan_ids`] over a node filter: score every set position, best `k` first.
    pub fn scan_nodes(
        &self,
        query: &DenseVector,
        filter: &NodeFilter,
        k: usize,
    ) -> Vec<(RecordId, f32)> {
        let query = self.prepare(query.clone());
        // Dense enough for the block scan: from one position in 64, most blocks hold a
        // match and the per-block kernel beats one gather-ADC per matched code; below it
        // the scattered codes are cheaper one at a time.
        if let Some(fast) = &self.fast {
            if filter.len().saturating_mul(FAST_SCAN_FROM_ONE_IN) >= self.nodes.len() {
                let lut = fast.lut(&query);
                // Twice the pool: the 4-bit codes rank a little less sharply than the
                // 8-bit ones (recall@10 0.9976 against 1.0 on YFCC-10M at one pool), the
                // scan's cost does not depend on `k`, and the rerank reads the extra rows.
                return fast
                    .scan(&lut, Some(filter), k * 2)
                    .into_iter()
                    .filter(|(idx, _)| !self.nodes.is_deleted(*idx))
                    .map(|(idx, dist)| (self.nodes.id(idx).to_string(), dist))
                    .collect();
            }
        }
        let scorer = self.scorer(&query);
        let mut heap: BinaryHeap<Result> = BinaryHeap::with_capacity(k + 1);
        let mut it = filter.iter().peekable();
        while let Some(idx) = it.next() {
            if let Some(&next) = it.peek() {
                self.prefetch(next);
            }
            if idx >= self.nodes.len() || self.nodes.is_deleted(idx) {
                continue;
            }
            let distance = self.score(&scorer, idx);
            if heap.len() < k {
                heap.push(Result { idx, distance });
            } else if heap.peek().is_some_and(|w| distance < w.distance) {
                heap.pop();
                heap.push(Result { idx, distance });
            }
        }
        let mut out: Vec<(RecordId, f32)> = heap
            .into_iter()
            .map(|r| (self.nodes.id(r.idx).to_string(), r.distance))
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// [`Self::search_with_filter`] with the filter answered per node position by `gate`.
    pub fn search_with_gate(
        &self,
        query: &DenseVector,
        k: usize,
        ef_override: usize,
        gate: &dyn NodeGate,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }
        let query = &self.prepare(query.clone());
        let scorer = self.scorer(query);
        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes.level(entry);
        let mut curr_nearest = vec![entry];
        for lc in (1..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(&scorer, curr_nearest, lc);
        }
        let ef = ef_override.max(k);
        // The serial walk, the one every unfiltered search takes. `search_layer_parallel`
        // spawns a rayon batch per visited node and scores every non-matching neighbour as
        // well; on 2M rows with a filter matching 10% it answered in 360 ms where the
        // amplified unfiltered fetch took 8 -- it was never on a path anything measured.
        let candidates = self.search_layer(&scorer, curr_nearest, ef, 0, Some(gate));
        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes.id(idx).to_string(), dist))
            .collect())
    }

    /// Search with the distance to each node supplied by the caller, by node index.
    ///
    /// Returns the `ef` best `(node index, score)` at layer 0, nearest first; map an index back
    /// to its record with [`Self::node_id`]. This is the traversal alone: no reranking, no
    /// filter, no deleted-node tombstones beyond what the layer walk skips itself.
    pub fn search_by_score(&self, score: impl Fn(usize) -> f32, ef: usize) -> Vec<(usize, f32)> {
        let Some(entry) = self.entry_point else {
            return Vec::new();
        };
        let mut curr_nearest = vec![entry];
        for lc in (1..=self.nodes.level(entry)).rev() {
            curr_nearest = self.search_layer_closest_by(&score, curr_nearest, lc);
        }
        self.search_layer_by(&score, curr_nearest, ef, 0, None)
    }

    /// The record stored at a node index, as returned by [`Self::search_by_score`].
    pub fn node_id(&self, idx: usize) -> &str {
        self.nodes.id(idx)
    }

    /// Whether node `idx` is a tombstone.
    pub fn is_deleted(&self, idx: usize) -> bool {
        self.nodes.is_deleted(idx)
    }

    /// Score the nodes of `ids` against `query` -- no walk -- and return the best `k`
    /// nearest first. Ids the graph does not hold are skipped. Exact on an exact graph, ADC
    /// on a quantized one; either way it is one distance per id from memory.
    pub fn scan_ids(&self, query: &DenseVector, ids: &[&str], k: usize) -> Vec<(RecordId, f32)> {
        let query = self.prepare(query.clone());
        let scorer = self.scorer(&query);
        let mut heap: BinaryHeap<Result> = BinaryHeap::with_capacity(k + 1);
        for id in ids {
            let Some(idx) = self.idx_of(id) else {
                continue;
            };
            if self.nodes.is_deleted(idx) {
                continue;
            }
            let distance = self.score(&scorer, idx);
            if heap.len() < k {
                heap.push(Result { idx, distance });
            } else if heap.peek().is_some_and(|w| distance < w.distance) {
                heap.pop();
                heap.push(Result { idx, distance });
            }
        }
        let mut out: Vec<(RecordId, f32)> = heap
            .into_iter()
            .map(|r| (self.nodes.id(r.idx).to_string(), r.distance))
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Search with precomputed query (for cosine similarity optimization)
    pub fn search_with_precomputed(
        &self,
        precomputed: &PrecomputedQuery,
        k: usize,
    ) -> crate::Result<Vec<(RecordId, f32)>> {
        if self.distance_metric != DistanceMetric::Cosine {
            // Fallback to regular search for non-cosine metrics
            return self.search(&precomputed.vector, k);
        }

        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes.level(entry);
        let mut curr_nearest = vec![entry];

        let scorer = self.scorer(&precomputed.vector);
        // Search upper layers
        for lc in (1..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(&scorer, curr_nearest, lc);
        }

        // Search layer 0 with precomputed optimization
        let ef = self.config.ef_search.max(k);
        let candidates = self.search_layer(&scorer, curr_nearest, ef, 0, None);

        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes.id(idx).to_string(), dist))
            .collect())
    }

    /// Insert a batch, planning every record in parallel and writing them in order.
    ///
    /// The expensive half of an insert is `ef_construction` layer expansions per record, and
    /// it only reads the graph; the half that mutates is pointer work. So the whole batch
    /// plans against the graph as it stood when the batch arrived, on every core, and the
    /// plans are then applied one at a time. No node-level locking, no change to how the
    /// graph is stored.
    ///
    /// What this gives up: records in the same batch cannot link to each other, because none
    /// of them is in the graph while the others are planning. Against a graph many times the
    /// batch's size that is a rounding error. Against a small one it is not — early records
    /// are what the rest of the graph hangs off — so below `PARALLEL_PLAN_MIN_NODES` this
    /// inserts one at a time and gives up nothing.
    /// A batch insert, split so a caller can hold exclusive access only where it is needed.
    ///
    /// Searches contend with inserts on whichever lock guards the index, and holding it for a
    /// whole batch parks every reader for the length of the batch: at a million nodes that
    /// measured 4 ms of planning, 0.25 ms of appending and 28 ms of pruning, and a streaming
    /// workload saw a p99 of 90-120 ms against 4.7 ms when idle (design doc section 33).
    /// Only the appending needs `&mut self`.
    ///
    /// `prepare_batch` and `apply_batch` take `&mut self`; `plan_batch` and `link_batch` take
    /// `&self`. Callers that do not care run [`Self::insert_batch`], which is the four in
    /// order. Only one batch may be in flight at a time -- the storage layer's insert
    /// processor is a single task, which is what makes releasing the lock between phases safe.
    pub fn insert_batch(&mut self, items: Vec<(RecordId, DenseVector)>) -> crate::Result<()> {
        // In chunks the graph will plan in parallel at its current size, which grow as the
        // chunks land; the first rows of an empty graph go in one at a time. The wrapper
        // does the same split around its locks; this is the same rule for a caller holding
        // the graph directly.
        let mut rest = items;
        while !rest.is_empty() {
            let limit = self.parallel_batch_limit();
            let take = if limit >= PARALLEL_PLAN_MIN_BATCH {
                limit.min(rest.len())
            } else {
                PARALLEL_PLAN_MIN_BATCH.min(rest.len())
            };
            let chunk: Vec<(RecordId, DenseVector)> = rest.drain(..take).collect();
            let Some(prepared) = self.prepare_batch(chunk)? else {
                continue;
            };
            let plans = self.plan_batch(&prepared);
            let applied = self.apply_batch(prepared, plans);
            self.link_batch(&applied);
        }
        Ok(())
    }

    /// Whether a batch handed to this graph will be planned in parallel or inserted one row at
    /// a time.
    ///
    /// The caller needs to know *before* it takes a lock, because the two cases want different
    /// locking: the parallel path releases the index between its phases, while the serial
    /// fallback runs the whole batch under whatever lock it was given. A 5,000-row batch into a
    /// young graph therefore held the write lock for 8.7 s, and a search on that table waited
    /// all of it -- measured over HTTP on 2026-09-25.
    pub fn plans_batches_in_parallel(&self, batch: usize) -> bool {
        batch >= PARALLEL_PLAN_MIN_BATCH
            && batch <= self.parallel_batch_limit()
            && self.entry_point.is_some()
    }

    /// The largest batch the graph will plan in parallel right now: the share of its nodes
    /// `PARALLEL_PLAN_MAX_SHARE` allows, or zero while it is younger than the seed. A caller
    /// with more rows than this splits them and asks again, since the limit grows with every
    /// chunk it lands.
    pub fn parallel_batch_limit(&self) -> usize {
        let nodes = self.nodes.len();
        if nodes < PARALLEL_PLAN_SEED || self.entry_point.is_none() {
            0
        } else {
            nodes / PARALLEL_PLAN_MAX_SHARE
        }
    }

    /// Draw the levels and reject duplicates. `&mut self` for the RNG; no distance work.
    ///
    /// `Ok(None)` means the batch was small enough, or the graph young enough, that the
    /// serial path already inserted it -- there is nothing left to plan.
    pub fn prepare_batch(
        &mut self,
        items: Vec<(RecordId, DenseVector)>,
    ) -> crate::Result<Option<PreparedBatch>> {
        if !self.plans_batches_in_parallel(items.len()) {
            for (id, vector) in items {
                self.insert(id, vector)?;
            }
            return Ok(None);
        }
        // Levels come from the RNG, which needs `&mut self`. Draw them here so that planning
        // needs nothing but `&self`. Duplicate ids are rejected against the graph and against
        // the rest of the batch -- the serial path catches the second one because the first is
        // already in `id_to_idx`, and nothing in this batch is yet.
        let mut prepared: Vec<(RecordId, DenseVector, usize)> = Vec::with_capacity(items.len());
        let mut seen: HashSet<RecordId> = HashSet::with_capacity(items.len());
        for (id, vector) in &items {
            self.check_dimension(id, vector)?;
        }
        for (id, vector) in items {
            if self.idx_of(&id).is_some() || !seen.insert(id.clone()) {
                return Err(crate::Error::InvalidInput(format!(
                    "ID {} already exists",
                    id
                )));
            }
            let vector = self.prepare(vector);
            let level = self.select_level();
            prepared.push((id, vector, level));
        }
        // One entry point for the whole batch, for the same reason: it is what everything
        // plans against, and the later phases use the same value so a plan and its application
        // never disagree about which layers were searched.
        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes.level(entry);
        Ok(Some(PreparedBatch {
            items: prepared,
            entry,
            entry_level,
        }))
    }

    /// A. Plan. Read-only, `&self`, every core.
    pub fn plan_batch(&self, prepared: &PreparedBatch) -> BatchPlan {
        let started = std::time::Instant::now();
        let (links, codes): (Vec<_>, Vec<_>) = prepared
            .items
            .par_iter()
            .map(|(_, vector, level)| {
                let links =
                    self.plan_neighbors(vector, *level, prepared.entry, prepared.entry_level);
                let code = self.quantized.as_ref().map(|store| store.encode(vector));
                (links, code)
            })
            .unzip();
        record_phase(0, started);
        BatchPlan { links, codes }
    }

    /// B. Append the nodes and their own forward edges. Serial `&mut self`, but pure pushes --
    /// for a 380-record group at 768 dimensions this is about a megabyte of memcpy. The whole
    /// batch is appended before the linking starts, so `nodes` does not grow while that runs
    /// and no index into it can be invalidated.
    pub fn apply_batch(&mut self, prepared: PreparedBatch, plan: BatchPlan) -> AppliedBatch {
        let started = std::time::Instant::now();
        let entry_level = prepared.entry_level;
        let mut applied: Vec<(usize, usize)> = Vec::with_capacity(plan.links.len());
        let mut new_entry: Option<usize> = None;
        for (((id, vector, level), links), code) in
            prepared.items.into_iter().zip(plan.links).zip(plan.codes)
        {
            let idx = self.push_planned_node(id, vector, level, links, code);
            if level > entry_level {
                new_entry = Some(idx);
            }
            applied.push((idx, level));
        }
        if let Some(idx) = new_entry {
            self.entry_point = Some(idx);
        }
        record_phase(1, started);
        AppliedBatch {
            nodes: applied,
            entry_level,
        }
    }

    /// C. Reverse edges and pruning. `&self`, one edge lock at a time, every core. This is the
    /// phase worth parallelising: the pruning inside `link_back` is where the insert path spent
    /// about 78% of its CPU, and it was serial.
    pub fn link_batch(&self, applied: &AppliedBatch) {
        let started = std::time::Instant::now();
        let graph: &Self = self;
        let entry_level = applied.entry_level;
        applied
            .nodes
            .par_iter()
            .for_each(|&(idx, level)| graph.link_back_all(idx, level, entry_level));
        graph.link_siblings(&applied.nodes);
        record_phase(2, started);
        // The batch counter advances with the last phase, so one line covers one batch
        // however the caller split it.
        report_phases(graph.nodes.len());
    }

    /// Link the rows of one batch to their nearest neighbours *within the batch*.
    ///
    /// A batch is planned against the graph as it stood before any of it, so two rows of the
    /// same batch never choose each other, however close they are; the serial path never had
    /// this gap, because each row saw every row before it. On a large graph the gap is a
    /// fraction of a percent of the candidates and invisible. On a young one it is not: with
    /// parallel planning starting at a thousand nodes, a self-retrieval test on a 1,500-row
    /// table failed intermittently -- the stored vector's true neighbours were in its own
    /// chunk and it had no edge to any of them.
    ///
    /// So each new node is offered its `m` nearest siblings at layer 0, in both directions,
    /// through the same `link_back` that adds a reverse edge -- the heuristic keeps the edge
    /// or not, exactly as it would have had the sibling been in the graph already. The
    /// pairwise distances are `n^2` for a batch of `n`: at the 1,000 rows a request carries
    /// that is a million at 100 dimensions, about a millisecond on sixteen cores, against the
    /// hundreds of millions the planning cost.
    fn link_siblings(&self, nodes: &[(usize, usize)]) {
        if nodes.len() < 2 {
            return;
        }
        let m = self.config.m;
        let idxs: Vec<usize> = nodes.iter().map(|&(idx, _)| idx).collect();
        // A sibling is worth offering only if it is nearer than the farthest link the node
        // already chose: anything beyond that would have lost the selection to what is there.
        // On a large graph that is almost no sibling, and the pass costs the pairwise
        // distances and nothing else; the first version offered the `m` nearest siblings
        // unconditionally, and 32,000 prunings a batch took the build from 6,262 to 4,465
        // rows/s. (A node with fewer than `m` links has room, so for it the bar is infinity.)
        let nearest: Vec<Vec<usize>> = idxs
            .par_iter()
            .map(|&i| {
                let mut links: Vec<u32> = Vec::with_capacity(MAX_M0_SLOTS);
                self.nodes.copy_links(i, 0, &mut links);
                let mut d: Vec<(usize, f32)> = self.with_distances_from(i, |dist| {
                    let bar = if links.len() < m {
                        f32::INFINITY
                    } else {
                        links
                            .iter()
                            .map(|&l| dist(l as usize))
                            .fold(0.0f32, f32::max)
                    };
                    idxs.iter()
                        .filter(|&&j| j != i)
                        .map(|&j| (j, dist(j)))
                        .filter(|&(_, d)| d < bar)
                        .collect()
                });
                d.sort_by(|a, b| a.1.total_cmp(&b.1));
                d.truncate(m);
                d.into_iter().map(|(j, _)| j).collect()
            })
            .collect();
        idxs.par_iter()
            .zip(nearest.par_iter())
            .for_each(|(&i, near)| {
                for &j in near {
                    self.link_back(i, 0, j);
                    self.link_back(j, 0, i);
                }
            });
    }

    /// The read-only half of an insert: which neighbours this vector should link to, per layer.
    ///
    /// Identical to what `insert` computes; kept separate so it can run with `&self`.
    fn plan_neighbors(
        &self,
        vector: &DenseVector,
        level: usize,
        entry: usize,
        entry_level: usize,
    ) -> Vec<Vec<usize>> {
        let mut per_layer = vec![Vec::new(); level + 1];
        let mut curr_nearest = vec![entry];
        let scorer = self.scorer(vector);

        for lc in ((level + 1).min(entry_level + 1)..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(&scorer, curr_nearest, lc);
        }

        for lc in (0..=level.min(entry_level)).rev() {
            let candidates = self.search_layer(
                &scorer,
                curr_nearest.clone(),
                self.config.ef_construction,
                lc,
                None,
            );
            let m = if lc == 0 {
                self.config.max_m0
            } else {
                self.config.m
            };
            per_layer[lc] = self.select_neighbors_heuristic(&candidates, m);
            if lc > 0 {
                curr_nearest = candidates.into_iter().map(|(idx, _)| idx).collect();
            }
        }
        per_layer
    }

    /// Append a planned node with its own forward edges, and nothing else.
    ///
    /// This is the only part of applying a plan that needs `&mut self`: it pushes onto
    /// `nodes` and `id_to_idx`. The reverse edges, which is where the work is, go through
    /// `link_back_all` under `&self`. `apply_planned_insert`, which did both for one node at
    /// a time, is what that split replaced -- a batch cannot run the second half on every
    /// core if it is interleaved with the first.
    fn push_planned_node(
        &mut self,
        id: RecordId,
        vector: DenseVector,
        level: usize,
        neighbors_per_layer: Vec<Vec<usize>>,
        code: Option<Vec<u8>>,
    ) -> usize {
        let links: Vec<Vec<u32>> = neighbors_per_layer
            .into_iter()
            .map(|l| l.into_iter().map(|n| n as u32).collect())
            .collect();
        match (code, &mut self.quantized) {
            (Some(code), Some(store)) => {
                store.codes.extend_from_slice(&code);
                if let Some(fast) = &mut self.fast {
                    fast.push(&vector)
                        .expect("vector was dimension-checked before reaching the graph");
                }
                let idx = self.nodes.push(&id, None, level, &links);
                self.register_id(&id, idx);
                idx
            }
            _ => self.push_node(&id, vector, level, &links),
        }
    }

    /// Append a node: its vector goes on the node while the graph is exact, or into the
    /// quantized store as a code once it is not. Called exactly once per node, which is what
    /// keeps `codes` aligned with `nodes`.
    fn push_node(
        &mut self,
        id: &str,
        vector: DenseVector,
        level: usize,
        links: &[Vec<u32>],
    ) -> usize {
        let idx = match &mut self.quantized {
            None => self.nodes.push(id, Some(&vector), level, links),
            Some(store) => {
                store
                    .push(&vector)
                    .expect("vector was dimension-checked before reaching the graph");
                if let Some(fast) = &mut self.fast {
                    fast.push(&vector)
                        .expect("vector was dimension-checked before reaching the graph");
                }
                self.nodes.push(id, None, level, links)
            }
        };
        self.register_id(id, idx);
        idx
    }

    /// Add the reverse edge on every neighbour `idx` chose, pruning any list that overflows.
    ///
    /// `&self`, so a batch can run this for many nodes at once.
    /// Give every node nothing points at an in-edge again, so a search can reach it.
    ///
    /// `select_neighbors_heuristic` keeps fewer than `m` edges when candidates crowd each
    /// other, and a node's in-edges are other nodes' out-edges: a node that loses every
    /// pruning contest on every list it was on -- not at insert, when it still has them, but a
    /// thousand inserts later as its neighbours fill up -- ends with no in-edges at all. It is
    /// in the graph and no query returns it, at any `ef`; 2% of GIST-960 at 100K rows and 5%
    /// at a million, measured. The two insert-time fixes the paper offers were tried and
    /// removed (see the heuristic's doc): one costs 1.8x the build on GIST and degrades a
    /// clustered corpus, the other fixes 0.4 of the 2 points because stranding does not
    /// happen at insert. This is the shape that fits what was measured: a pass over the
    /// layer-0 in-degrees, which is an O(edges) read -- tens of milliseconds at a million
    /// rows -- and a single edge added per stranded node, from the nearest of its own
    /// out-neighbours back to it.
    ///
    /// The added edge is not put through the heuristic, which is what removed it in the
    /// first place: when the neighbour's list is full, the edge replaces the neighbour's
    /// farthest link, provided that link has another in-edge somewhere, so one node is never
    /// freed at the cost of stranding another. Searches run concurrently; each list is
    /// edited under its own lock, one at a time, like `link_back`.
    ///
    /// Returns `(stranded, repaired)`. The difference is nodes whose every out-neighbour's
    /// list was full of singletons, which the next pass may see differently.
    pub fn repair_reachability(&self) -> (usize, usize) {
        let n = self.nodes.len();
        if n < 2 {
            return (0, 0);
        }
        let mut in_degree = vec![0u32; n];
        let mut links: Vec<u32> = Vec::with_capacity(MAX_M0_SLOTS);
        for from in 0..n {
            if self.nodes.is_deleted(from) {
                continue;
            }
            self.nodes.copy_links(from, 0, &mut links);
            for &to in &links {
                if let Some(d) = in_degree.get_mut(to as usize) {
                    *d += 1;
                }
            }
        }
        let stranded: Vec<usize> = (0..n)
            .filter(|&i| {
                in_degree[i] == 0 && !self.nodes.is_deleted(i) && Some(i) != self.entry_point
            })
            .collect();
        let max_m0 = self.config.max_m0;
        let mut repaired = 0usize;
        for &idx in &stranded {
            // Its own out-neighbours, nearest first: the nodes that already consider it
            // close enough to point at, and the ones a walk towards it will pass through.
            self.nodes.copy_links(idx, 0, &mut links);
            let mut candidates: Vec<(usize, f32)> = links
                .iter()
                .map(|&f| f as usize)
                .filter(|&f| f != idx && !self.nodes.is_deleted(f))
                .map(|f| (f, self.node_distance(idx, f)))
                .collect();
            candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
            let mut done = false;
            for (from, _) in candidates {
                // `Some(evicted)` when the edge went in; the evicted link, if any, has one
                // in-edge fewer from here on, and the count has to say so or a later eviction
                // could take its last one.
                let outcome: Option<Option<u32>> = self
                    .nodes
                    .with_links_mut(from, 0, |list| {
                        if list.contains(idx as u32) {
                            return Some(None);
                        }
                        if list.len() < max_m0 && list.push(idx as u32) {
                            return Some(None);
                        }
                        // Full: evict the farthest link that is not somebody's only in-edge.
                        let mut farthest: Option<(usize, f32)> = None;
                        for (pos, &l) in list.as_slice().iter().enumerate() {
                            if in_degree[l as usize] <= 1 {
                                continue;
                            }
                            let d = self.node_distance(from, l as usize);
                            if farthest.is_none_or(|(_, fd)| d > fd) {
                                farthest = Some((pos, d));
                            }
                        }
                        let (pos, _) = farthest?;
                        let mut kept: Vec<u32> = list.as_slice().to_vec();
                        let evicted = kept[pos];
                        kept[pos] = idx as u32;
                        list.set(&kept);
                        Some(Some(evicted))
                    })
                    .flatten();
                if let Some(evicted) = outcome {
                    in_degree[idx] += 1;
                    if let Some(e) = evicted {
                        in_degree[e as usize] = in_degree[e as usize].saturating_sub(1);
                    }
                    done = true;
                    break;
                }
            }
            if done {
                repaired += 1;
            }
        }
        (stranded.len(), repaired)
    }

    fn link_back_all(&self, idx: usize, level: usize, entry_level: usize) {
        // Copy the plan out and drop the guard before taking any write lock, so that no
        // thread ever holds two edge locks at once.
        let mut chosen: Vec<u32> = Vec::with_capacity(MAX_M0_SLOTS);
        for lc in 0..=level.min(entry_level) {
            self.nodes.copy_links(idx, lc, &mut chosen);
            for &neighbor_idx in &chosen {
                self.link_back(neighbor_idx as usize, lc, idx);
            }
        }
    }

    /// Add the reverse edge `neighbor_idx -> new_idx` on layer `lc`, pruning the neighbour's
    /// edge list back to its cap if that push overflowed it.
    ///
    /// This is the expensive half of an insert. The pruning branch recomputes the distance
    /// from the neighbour to every one of its links, and that is where about 78% of insert
    /// CPU went in the profile — the part a single graph-wide write lock kept on one core.
    ///
    /// It takes `&self` and exactly one lock, the neighbour's edge list on this one layer.
    /// `calculate_distance` and `select_neighbors_heuristic` read `nodes[..].vector`, which is
    /// never behind a lock. **Nothing here holds two locks at once, so the parallel phase
    /// cannot deadlock** however the batch interleaves.
    ///
    /// `insert` and `apply_planned_insert` used to carry a verbatim copy of this each.
    fn link_back(&self, neighbor_idx: usize, lc: usize, new_idx: usize) {
        if lc >= self.nodes.layers(neighbor_idx) {
            return;
        }
        let max_neighbors = if lc == 0 {
            self.config.max_m0
        } else {
            self.config.m
        };

        self.nodes.with_links_mut(neighbor_idx, lc, |links| {
            if links.len() < max_neighbors && links.push(new_idx as u32) {
                return;
            }
            // Prune by the same heuristic used to choose links in the first place. Keeping
            // the `max_neighbors` closest instead would undo the diversity on exactly the
            // nodes that matter most -- the ones popular enough to overflow. The new edge
            // takes part in the selection like any other.
            // Every link's data is about to be read, at addresses known now; ask for them
            // all before the first distance rather than paying each miss in turn.
            for &n in links.as_slice() {
                self.prefetch(n as usize);
            }
            let mut neighbors_with_dist: Vec<(usize, f32)> =
                self.with_distances_from(neighbor_idx, |dist| {
                    links
                        .as_slice()
                        .iter()
                        .map(|&n| n as usize)
                        .chain(std::iter::once(new_idx))
                        .map(|n_idx| (n_idx, dist(n_idx)))
                        .collect()
                });
            neighbors_with_dist.sort_by(|a, b| a.1.total_cmp(&b.1));
            let kept: Vec<u32> = self
                .select_neighbors_heuristic(&neighbors_with_dist, max_neighbors)
                .into_iter()
                .map(|n| n as u32)
                .collect();
            links.set(&kept);
        });
    }

    /// Refuse a vector the graph cannot hold.
    ///
    /// `Nodes` stores every vector in one flat array and learns the dimension from the first
    /// one; after that it appends without looking. An empty vector appended nothing but still
    /// took a node, so from that node on every index pointed one vector past its data, and the
    /// first read of the last one panicked. That happened on restart: `update_record` logged a
    /// record whose vector had moved to the row file, replay handed the empty vector here, and
    /// with `panic = "abort"` the server could no longer start after any update followed by a
    /// crash. The log is fixed at the source; this is the boundary that keeps any other path
    /// from doing the same.
    fn check_dimension(&self, id: &str, vector: &[f32]) -> crate::Result<()> {
        let expected = self.nodes.dim;
        if vector.is_empty() || (expected != 0 && vector.len() != expected) {
            return Err(crate::Error::InvalidVector(format!(
                "record '{id}' has {} dimensions and the graph holds {expected}",
                vector.len()
            )));
        }
        Ok(())
    }

    pub fn insert(&mut self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        self.check_dimension(&id, &vector)?;
        let vector = self.prepare(vector);
        if self.idx_of(&id).is_some() {
            return Err(crate::Error::InvalidInput(format!(
                "ID {} already exists",
                id
            )));
        }

        let level = self.select_level();

        if self.nodes.is_empty() || self.entry_point.is_none() {
            let idx = self.push_node(&id, vector, level, &[]);
            self.entry_point = Some(idx);
            return Ok(());
        }
        let mut planned: Vec<Vec<u32>> = vec![Vec::new(); level + 1];

        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes.level(entry);
        let mut curr_nearest = vec![entry];
        let scorer = self.scorer(&vector);

        // Search from top to target layer
        for lc in ((level + 1).min(entry_level + 1)..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(&scorer, curr_nearest, lc);
        }

        // Insert at all levels from level to 0
        for lc in (0..=level.min(entry_level)).rev() {
            let candidates = self.search_layer(
                &scorer,
                curr_nearest.clone(),
                self.config.ef_construction,
                lc,
                None,
            );

            let m = if lc == 0 {
                self.config.max_m0
            } else {
                self.config.m
            };

            planned[lc] = self
                .select_neighbors_heuristic(&candidates, m)
                .into_iter()
                .map(|n| n as u32)
                .collect();

            // Update curr_nearest for next layer
            if lc > 0 {
                curr_nearest = candidates.into_iter().map(|(idx, _)| idx).collect();
            }
        }

        // Add node
        let idx = self.push_node(&id, vector, level, &planned);

        // Add bidirectional links -- the same `link_back` the batch path uses, so the two
        // cannot drift apart again.
        self.link_back_all(idx, level, entry_level);

        // Update entry point if new node has higher layer
        if level > entry_level {
            self.entry_point = Some(idx);
        }

        Ok(())
    }
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("num_nodes", &self.nodes.len())
            .field("config", &self.config)
            .field("entry_point", &self.entry_point)
            .finish()
    }
}

// Adapter to implement index_traits::VectorIndex for HnswIndex
#[async_trait::async_trait]
impl crate::search::index_traits::VectorIndex for HnswIndex {
    /// Links, ids, and codes once quantized (`resident_bytes_without_vectors`), plus the vector
    /// arena, which a quantized graph has emptied.
    async fn memory_bytes(&self) -> usize {
        self.resident_bytes_without_vectors()
            + self.nodes.vectors.capacity() * std::mem::size_of::<f32>()
    }

    async fn add(&mut self, id: String, vector: Vec<f32>) -> crate::Result<()> {
        self.insert(id, vector)
    }

    fn as_hnsw(&self) -> Option<&HnswIndex> {
        Some(self)
    }

    fn repair_reachability(&self) -> (usize, usize) {
        HnswIndex::repair_reachability(self)
    }

    fn contains_id(&self, id: &str) -> Option<bool> {
        Some(
            self.idx_of(id)
                .is_some_and(|idx| !self.nodes.is_deleted(idx)),
        )
    }

    fn as_hnsw_mut(&mut self) -> Option<&mut HnswIndex> {
        Some(self)
    }

    async fn add_batch(&mut self, items: Vec<(String, Vec<f32>)>) -> crate::Result<()> {
        self.insert_batch(items)
    }

    /// Build the whole index from a vector set, on every core.
    ///
    /// Without this override HnswIndex took the trait's default, which inserts one record at a
    /// time. A 1M-vector `rebuild_index` was therefore a million serial single-threaded
    /// `insert` calls that never touched `insert_batch` -- the path made parallel in 2133ed6.
    /// A rebuild was several times slower than the same graph built through the insert path,
    /// which had already been made parallel: the bulk path was the one left behind.
    ///
    /// (A head-to-head measurement against another engine is what first showed the size of
    /// the gap. The numbers are in the internal measurement notes and deliberately not here:
    /// `docs/SUPPORT_MATRIX.md` says a competitive figure is worth stating only beside the
    /// method and the tuning used on both sides, and this comment used to be the one place in
    /// the tree that did not hold to that.)
    ///
    /// Chunked rather than handed over whole: see `BULK_BUILD_CHUNK` for why bigger is not
    /// better.
    async fn bulk_build(&mut self, vectors: Vec<(String, Vec<f32>)>) -> crate::Result<()> {
        let mut rest = vectors.into_iter();
        loop {
            // `by_ref().take(n)` consumes rather than clones -- these are 768-float vectors
            // and there are a million of them.
            let chunk: Vec<(String, Vec<f32>)> = rest.by_ref().take(BULK_BUILD_CHUNK).collect();
            if chunk.is_empty() {
                return Ok(());
            }
            self.insert_batch(chunk)?;
        }
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
        self.search_with_ef(&query.to_vec(), k, ef)
    }

    async fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_ids: &HashSet<String>,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_with_filter(&query.to_vec(), k, ef, filter_ids)
    }

    async fn delete(&mut self, id: &str) -> crate::Result<()> {
        // Find node index
        let idx = self
            .idx_of(id)
            .ok_or_else(|| crate::Error::NotFound(format!("ID {} not found", id)))?;

        self.nodes.mark_deleted(idx);

        self.unregister_id(id);

        // Reconnect neighbors to maintain graph connectivity
        self.reconnect_neighbors(idx)?;

        // Update entry point if it was deleted
        if self.entry_point == Some(idx) {
            self.find_new_entry_point();
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.id_to_idx.len() + self.id_overflow.len()
    }

    /// Node slots, tombstones included. `nodes` never gives a slot back: an update marks the
    /// old one deleted and appends a new one.
    fn slot_count(&self) -> usize {
        self.nodes.len()
    }

    fn node_id(&self, idx: usize) -> Option<&str> {
        (idx < self.nodes.len()).then(|| self.nodes.id(idx))
    }

    async fn search_with_gate(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        gate: &dyn NodeGate,
    ) -> crate::Result<Vec<(String, f32)>> {
        HnswIndex::search_with_gate(self, &query.to_vec(), k, ef, gate)
    }

    fn snapshot_to(&self, path: &std::path::Path) -> crate::Result<bool> {
        crate::search::index_traits::SerializableIndex::save(self, path)?;
        Ok(true)
    }

    fn is_empty(&self) -> bool {
        self.id_to_idx.is_empty() && self.id_overflow.is_empty()
    }
}

// Helper methods for HNSW index (not part of VectorIndex trait)
impl HnswIndex {
    /// Reconnect neighbors when a node is deleted
    fn reconnect_neighbors(&mut self, deleted_idx: usize) -> crate::Result<()> {
        // Clone the data we need before borrowing mutably
        let deleted_level = self.nodes.level(deleted_idx);
        let deleted_neighbors: Vec<Vec<usize>> = (0..self.nodes.layers(deleted_idx))
            .map(|layer| {
                let mut buf = Vec::new();
                self.nodes.copy_links(deleted_idx, layer, &mut buf);
                buf.into_iter().map(|n| n as usize).collect()
            })
            .collect();

        // For each layer where the deleted node had connections
        for (layer, neighbors_in_layer) in
            deleted_neighbors.iter().enumerate().take(deleted_level + 1)
        {
            let neighbors_of_deleted = neighbors_in_layer.clone();

            // Remove deleted node from all its neighbors' neighbor lists
            for &neighbor_idx in &neighbors_of_deleted {
                if neighbor_idx < self.nodes.len()
                    && !self.nodes.is_deleted(neighbor_idx)
                    && layer < self.nodes.layers(neighbor_idx)
                {
                    self.nodes.with_links_mut(neighbor_idx, layer, |links| {
                        links.retain(|n| n as usize != deleted_idx)
                    });
                }
            }

            // Optionally reconnect orphaned neighbors to each other
            // This helps maintain graph connectivity
            if neighbors_of_deleted.len() >= 2 {
                // Connect neighbors to each other to maintain graph density
                for i in 0..neighbors_of_deleted.len() {
                    let neighbor_a = neighbors_of_deleted[i];
                    if neighbor_a >= self.nodes.len() || self.nodes.is_deleted(neighbor_a) {
                        continue;
                    }

                    // Find the closest neighbors from the deleted node's neighbors
                    let mut distances: Vec<(usize, f32)> = neighbors_of_deleted
                        .iter()
                        .filter(|&&n| {
                            n != neighbor_a && n < self.nodes.len() && !self.nodes.is_deleted(n)
                        })
                        .map(|&n| (n, self.node_distance(neighbor_a, n)))
                        .collect();

                    distances.sort_by(|a, b| a.1.total_cmp(&b.1));

                    // Add connections to maintain connectivity (limit to M/2 to avoid over-connection)
                    let max_new_connections = self.config.m / 2;
                    for (new_neighbor, _) in distances.into_iter().take(max_new_connections) {
                        if layer < self.nodes.layers(neighbor_a) {
                            let added = self
                                .nodes
                                .with_links_mut(neighbor_a, layer, |links| {
                                    !links.contains(new_neighbor as u32)
                                        && links.push(new_neighbor as u32)
                                })
                                .unwrap_or(false);
                            // One lock at a time here too: the reverse edge is taken after
                            // the first lock is released, so this path can never be the odd
                            // one out that deadlocks against a concurrent insert.
                            if added && layer < self.nodes.layers(new_neighbor) {
                                self.nodes.with_links_mut(new_neighbor, layer, |reverse| {
                                    if !reverse.contains(neighbor_a as u32) {
                                        reverse.push(neighbor_a as u32);
                                    }
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Find a new entry point if the current one is deleted
    fn find_new_entry_point(&mut self) {
        // Find the first non-deleted node with maximum level
        let mut best_idx = None;
        let mut best_level = 0;

        for idx in 0..self.nodes.len() {
            if !self.nodes.is_deleted(idx) && self.nodes.level(idx) >= best_level {
                best_idx = Some(idx);
                best_level = self.nodes.level(idx);
            }
        }

        self.entry_point = best_idx;
    }
}

impl crate::search::index_traits::SerializableIndex for HnswIndex {
    fn save(&self, path: &std::path::Path) -> crate::Result<()> {
        // "hnsw-v4": the column layout of `Nodes`. v2 and v3 are read, never written.
        crate::search::index_persistence::write_index(path, "hnsw-v4", self)
    }

    fn load(path: &std::path::Path) -> crate::Result<Self> {
        // The RNG is `#[serde(skip)]` and comes back seeded from `default_rng`, not from
        // `config.seed`. Level assignment for vectors inserted after a load therefore follows a
        // different sequence than it would have without the restart. That changes the graph
        // shape, not its correctness — HNSW makes no guarantee about which levels a node lands
        // on, only that search respects the levels it finds.
        // "hnsw-v2", not "hnsw": vectors in the file are now unit length for cosine, and a
        // v1 file read back would be scored against normalised queries — every distance quietly
        // wrong rather than loudly absent. A rejected kind makes the restore path warn and
        // rebuild from the rows instead.
        match crate::search::index_persistence::read_kind(path)?.as_str() {
            "hnsw-v4" => {
                let mut index: HnswIndex =
                    crate::search::index_persistence::read_index(path, "hnsw-v4")?;
                index.after_load();
                Ok(index)
            }
            "hnsw-v3" => {
                let old: LegacyV3 = crate::search::index_persistence::read_index(path, "hnsw-v3")?;
                Ok(Self::from_legacy(
                    old.config,
                    old.distance_metric,
                    old.nodes,
                    old.entry_point,
                    old.quantized,
                ))
            }
            _ => {
                let old: LegacyV2 = crate::search::index_persistence::read_index(path, "hnsw-v2")?;
                Ok(Self::from_legacy(
                    old.config,
                    old.distance_metric,
                    old.nodes,
                    old.entry_point,
                    None,
                ))
            }
        }
    }

    fn stats(&self) -> crate::search::index_traits::IndexStats {
        // Quantized nodes store no vector, so this is zero for them and the codes are counted
        // by `resident_bytes_without_vectors` instead.
        let vectors_bytes = self.nodes.vectors.len() * std::mem::size_of::<f32>();

        crate::search::index_traits::IndexStats {
            // `nodes` keeps tombstoned entries, so this counts live vectors rather than slots.
            total_vectors: self.nodes.live_count(),
            index_size_bytes: vectors_bytes + self.resident_bytes_without_vectors(),
            build_time_ms: 0,
        }
    }
}

#[cfg(test)]
mod reachability_tests {
    use super::*;

    /// Nodes nothing points at. A search walks edges from the entry point, so a node with no
    /// in-edges is in the graph and cannot be returned by any query — not at a larger `ef`,
    /// not at a larger `top_k`. Reachability is not a budget.
    fn orphans(index: &HnswIndex) -> Vec<usize> {
        let n = index.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut links = Vec::new();
        for from in 0..n {
            for layer in 0..index.nodes.layers(from) {
                index.nodes.copy_links(from, layer, &mut links);
                for &to in &links {
                    in_degree[to as usize] += 1;
                }
            }
        }
        (0..n)
            .filter(|&i| in_degree[i] == 0 && Some(i) != index.entry_point)
            .collect()
    }

    /// A clustered corpus leaves no node stranded, which is the case that has to keep working.
    ///
    /// **This is characterisation, not a guarantee.** The graph *does* strand a small
    /// fraction of points on some real corpora — about 2% of GIST-960 and 0.5% of a modern
    /// embedding corpus, measured — and `select_neighbors_heuristic` carries the numbers and
    /// the reason. What this pins is that the ordinary case stays at zero, so a change that
    /// started stranding nodes in bulk would be caught here, with the node named.
    ///
    /// It will not catch the existing shortfall: no synthetic corpus written for this
    /// reproduced it. The points that get stranded are mid-distance ones that lose every
    /// pruning contest, and the obvious synthesis — a far outlier — is *protected* by the
    /// heuristic, because its distance to the other candidates is about its distance to the
    /// node being linked, so it never reads as crowded.
    /// A node with its in-edges cut by hand is unreachable, and one repair pass brings it
    /// back -- found by a search for its own vector, which is the operation that fails on a
    /// stranded node. Cut by hand because no synthetic corpus reproduces the stranding (see
    /// `a_clustered_corpus_strands_nobody`); the mechanism is what this pins.
    #[test]
    fn a_repair_pass_gives_a_stranded_node_an_in_edge_back() {
        let mut index = HnswIndex::new(DistanceMetric::Euclidean, Some(HnswConfig::default()));
        let mut rng = StdRng::seed_from_u64(7);
        let mut vectors: Vec<DenseVector> = Vec::new();
        for i in 0..3_000 {
            let v: DenseVector = (0..16).map(|_| rng.gen::<f32>()).collect();
            vectors.push(v.clone());
            index.insert(format!("v{i:06}"), v).unwrap();
        }
        assert!(
            orphans(&index).is_empty(),
            "the corpus itself strands nobody"
        );

        // Strand ten nodes: remove each from every list that points at it, on every layer.
        let victims: Vec<usize> = (0..10).map(|k| 100 + k * 251).collect();
        let n = index.nodes.len();
        let mut links = Vec::new();
        for from in 0..n {
            for layer in 0..index.nodes.layers(from) {
                index.nodes.copy_links(from, layer, &mut links);
                if links.iter().any(|&l| victims.contains(&(l as usize))) {
                    let kept: Vec<u32> = links
                        .iter()
                        .copied()
                        .filter(|&l| !victims.contains(&(l as usize)))
                        .collect();
                    index
                        .nodes
                        .with_links_mut(from, layer, |slot| slot.set(&kept));
                }
            }
        }
        let stranded = orphans(&index);
        assert_eq!(stranded.len(), 10, "{stranded:?}");
        for &v in &victims {
            let got = index.search(&vectors[v], 1).unwrap();
            assert_ne!(
                got[0].0,
                format!("v{v:06}"),
                "a node with no in-edges should not be found by its own vector"
            );
        }

        let (found, repaired) = index.repair_reachability();
        assert_eq!((found, repaired), (10, 10));
        assert!(orphans(&index).is_empty(), "{:?}", orphans(&index));
        for &v in &victims {
            let got = index.search(&vectors[v], 1).unwrap();
            assert_eq!(
                got[0].0,
                format!("v{v:06}"),
                "reachable again by its own vector"
            );
        }
        // A second pass has nothing to do and changes nothing.
        assert_eq!(index.repair_reachability(), (0, 0));
    }

    #[test]
    fn a_clustered_corpus_strands_nobody() {
        let mut index = HnswIndex::new(DistanceMetric::Euclidean, Some(HnswConfig::default()));
        let mut rng = StdRng::seed_from_u64(99);
        for i in 0..2_000 {
            let centre = (i % 12) as f32;
            let v: DenseVector = (0..16)
                .map(|d| centre + rng.gen::<f32>() * 0.4 + (d as f32) * 0.01)
                .collect();
            index.insert(format!("v{i:06}"), v).unwrap();
        }
        let orphaned = orphans(&index);
        assert!(
            orphaned.is_empty(),
            "{} of {} nodes have no in-edges, so no search can reach them: {:?}",
            orphaned.len(),
            index.nodes.len(),
            &orphaned[..orphaned.len().min(10)]
        );
    }

    /// Every stored vector comes back when used as its own query, on a corpus that strands
    /// nobody.
    ///
    /// Compared by *distance*, not by id. Corpora have duplicate vectors — GIST has records
    /// whose three nearest neighbours are all at distance zero — and demanding the id back
    /// counts a correct tie as a failure, which is how the first measurement of the stranding
    /// came out twice as bad as it was.
    #[test]
    fn every_stored_vector_finds_itself_or_an_equally_near_twin() {
        let mut index = HnswIndex::new(DistanceMetric::Euclidean, Some(HnswConfig::default()));
        let mut rng = StdRng::seed_from_u64(7);
        let corpus: Vec<(RecordId, DenseVector)> = (0..2_000)
            .map(|i| {
                let centre = (i % 12) as f32;
                let v: DenseVector = (0..16)
                    .map(|d| centre + rng.gen::<f32>() * 0.4 + (d as f32) * 0.01)
                    .collect();
                (format!("v{i:06}"), v)
            })
            .collect();
        for (id, v) in corpus.clone() {
            index.insert(id, v).unwrap();
        }
        let mut missed = Vec::new();
        for (id, v) in corpus.iter().take(400) {
            let hits = index.search_with_ef(v, 1, 64).unwrap();
            let found = hits.first().map(|(_, d)| *d).unwrap_or(f32::MAX);
            if found > 1e-4 {
                missed.push((id.clone(), found));
            }
        }
        assert!(
            missed.is_empty(),
            "{} stored vectors did not come back when used as their own query, nearest \
             distance {:?}",
            missed.len(),
            &missed[..missed.len().min(5)]
        );
    }
}

#[cfg(test)]
mod batch_insert_tests {
    use super::*;

    fn corpus(n: usize, dim: usize, seed: u64) -> Vec<(RecordId, DenseVector)> {
        // A clustered distribution, not uniform noise: uniform random unit vectors are the
        // hardest case for any graph index and the least like real embeddings, so a
        // regression in graph quality shows up more honestly on something with structure.
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                let centre = (i % 16) as f32;
                let v: DenseVector = (0..dim)
                    .map(|d| centre + rng.gen::<f32>() * 0.5 + (d as f32) * 0.01)
                    .collect();
                (format!("v{i:06}"), v)
            })
            .collect()
    }

    fn index() -> HnswIndex {
        HnswIndex::new(DistanceMetric::Cosine, Some(HnswConfig::default()))
    }

    fn live_count(index: &HnswIndex) -> usize {
        index.nodes.live_count()
    }

    fn recall_against(
        index_a: &HnswIndex,
        index_b: &HnswIndex,
        queries: &[DenseVector],
        k: usize,
    ) -> (f64, f64) {
        // Both indexes answer the same queries; each is scored against the exact answer taken
        // from a brute-force pass over the same vectors.
        let mut hits_a = 0usize;
        let mut hits_b = 0usize;
        let mut total = 0usize;
        for q in queries {
            let exact: HashSet<String> = {
                let mut scored: Vec<(String, f32)> = (0..index_a.nodes.len())
                    .filter(|&i| !index_a.nodes.is_deleted(i))
                    .map(|i| {
                        (
                            index_a.nodes.id(i).to_string(),
                            index_a.calculate_distance(q, index_a.nodes.vector(i)),
                        )
                    })
                    .collect();
                scored.sort_by(|x, y| x.1.total_cmp(&y.1));
                scored.into_iter().take(k).map(|(id, _)| id).collect()
            };
            let got_a: HashSet<String> = index_a
                .search(q, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let got_b: HashSet<String> = index_b
                .search(q, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            hits_a += got_a.intersection(&exact).count();
            hits_b += got_b.intersection(&exact).count();
            total += k;
        }
        (hits_a as f64 / total as f64, hits_b as f64 / total as f64)
    }

    #[test]
    fn a_batched_build_is_as_good_as_a_serial_one() {
        let dim = 32;
        let items = corpus(12_000, dim, 0xB47C);

        let mut serial = index();
        for (id, v) in items.clone() {
            serial.insert(id, v).unwrap();
        }

        let mut batched = index();
        for chunk in items.chunks(500) {
            batched.insert_batch(chunk.to_vec()).unwrap();
        }

        assert_eq!(
            live_count(&serial),
            live_count(&batched),
            "the two paths stored different counts"
        );

        let queries: Vec<DenseVector> = items.iter().step_by(400).map(|(_, v)| v.clone()).collect();
        let (serial_recall, batched_recall) = recall_against(&serial, &batched, &queries, 10);

        println!("serial recall@10 {serial_recall:.4}, batched recall@10 {batched_recall:.4}");
        // Held to the serial path rather than to a fixed number: this is asking whether
        // batching cost anything, not whether HNSW is good.
        assert!(
            batched_recall >= serial_recall - 0.02,
            "batched build lost recall: serial {serial_recall:.4}, batched {batched_recall:.4}"
        );
    }

    #[test]
    fn a_batch_below_the_threshold_takes_the_serial_path() {
        // Small graphs fall back on purpose: the first records are what everything else hangs
        // off, and letting them plan without seeing each other is not a rounding error there.
        let items = corpus(64, 8, 7);
        let mut idx = index();
        idx.insert_batch(items.clone()).unwrap();
        assert_eq!(live_count(&idx), items.len());
        // Every record must be findable -- a fallback that silently dropped rows would still
        // pass a count check on an empty graph, so query for one.
        let (_, probe) = &items[13];
        let hits = idx.search(probe, 1).unwrap();
        assert_eq!(hits[0].0, items[13].0, "the fallback path lost a record");
    }

    #[tokio::test]
    async fn a_bulk_build_matches_a_serial_one() {
        // `bulk_build` is what `rebuild_index` calls, and it used to fall through to the
        // trait's default -- one `insert` at a time, on one core, for the whole table. The
        // override sends it through `insert_batch` instead, so this asks the question that
        // change raises: does building in chunks cost anything a search can see?
        use crate::search::index_traits::VectorIndex;
        let dim = 32;
        let items = corpus(12_000, dim, 0x8B1C);

        let mut serial = index();
        for (id, v) in items.clone() {
            serial.insert(id, v).unwrap();
        }

        let mut bulk = index();
        bulk.bulk_build(items.clone()).await.unwrap();

        assert_eq!(
            live_count(&serial),
            live_count(&bulk),
            "bulk_build stored a different number of records"
        );

        let queries: Vec<DenseVector> = items.iter().step_by(400).map(|(_, v)| v.clone()).collect();
        let (serial_recall, bulk_recall) = recall_against(&serial, &bulk, &queries, 10);
        println!("serial recall@10 {serial_recall:.4}, bulk recall@10 {bulk_recall:.4}");
        assert!(
            bulk_recall >= serial_recall - 0.02,
            "bulk build lost recall: serial {serial_recall:.4}, bulk {bulk_recall:.4}"
        );
    }

    #[test]
    fn a_duplicate_inside_one_batch_is_rejected() {
        // The serial path catches a repeat because the first copy is already in `id_to_idx`.
        // Nothing in a batch is, while the batch is being planned, so the batch has to check
        // itself.
        let mut idx = index();
        let mut items = corpus(11_000, 8, 3);
        let seeded = items[0].clone();
        items.push(seeded);
        assert!(
            idx.insert_batch(items).is_err(),
            "a repeated id inside one batch was accepted"
        );
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::*;
    use crate::search::index_traits::SerializableIndex;

    /// An index file written before the edge lists went behind locks, by the code at
    /// 25c279c: 200 clustered 16-d vectors, `HnswConfig::default()`, cosine.
    const HNSW_V2_GOLDEN: &[u8] = include_bytes!("testdata/hnsw_v2_golden.idx");

    #[test]
    fn hnsw_v2_files_still_load() {
        // `Vec<RwLock<Vec<usize>>>` is only safe to swap in for `Vec<Vec<usize>>` if it
        // encodes the same bytes -- otherwise every index file on every existing deployment
        // becomes unreadable on upgrade, and the kind would have to go to "hnsw-v3". bincode
        // is positional and parking_lot's serde impl is transparent, so it should be a
        // no-op. Should be is not a result: this reads a file written by the old struct.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("golden.idx");
        std::fs::write(&path, HNSW_V2_GOLDEN).unwrap();

        let loaded =
            HnswIndex::load(&path).expect("an hnsw-v2 file written before the lock change");

        assert_eq!(
            loaded.nodes.len(),
            200,
            "lost nodes across the format change"
        );
        assert_eq!(loaded.entry_point, Some(172), "entry point moved");
        assert_eq!(crate::search::index_traits::VectorIndex::len(&loaded), 200);

        let edges = loaded.nodes.total_edges();
        assert!(edges > 200, "edges did not survive the load: {edges}");

        for probe in [0usize, 57, 199] {
            let hits = loaded
                .search(&loaded.nodes.vector(probe).to_vec(), 1)
                .unwrap();
            assert_eq!(
                hits[0].0,
                loaded.nodes.id(probe),
                "loaded graph could not find node {probe}"
            );
        }

        // Saved again, the graph is "hnsw-v4": the v2 layout is read, never written.
        let again = dir.path().join("again.idx");
        loaded.save(&again).unwrap();
        assert_eq!(
            crate::search::index_persistence::read_kind(&again).unwrap(),
            "hnsw-v4"
        );
        let back = HnswIndex::load(&again).unwrap();
        assert_eq!(back.nodes.total_edges(), edges);
        for probe in [0usize, 57, 199] {
            assert_eq!(
                back.search(&back.nodes.vector(probe).to_vec(), 1).unwrap()[0].0,
                loaded.nodes.id(probe)
            );
        }
    }
}

#[cfg(test)]
mod adc_kernel_tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The unrolled and gathered kernels must agree with the plain loop on every code
    /// length, including the tails that are not a multiple of eight.
    #[test]
    fn kernels_agree_with_the_plain_loop() {
        let k = 256usize;
        let mut st = 0x9E37_79B9_7F4A_7C15_u64;
        for m in [3usize, 8, 13, 48, 96, 192] {
            let table: Vec<f32> = (0..m * k)
                .map(|_| (xorshift(&mut st) >> 40) as f32 / 1e6)
                .collect();
            let a: Vec<u8> = (0..m).map(|_| xorshift(&mut st) as u8).collect();
            let adc_bases: Vec<i32> = (0..m).map(|sub| (sub * k) as i32).collect();

            let plain: f32 = (0..m).map(|sub| table[sub * k + a[sub] as usize]).sum();
            for (name, got) in [
                ("scalar", adc_scalar(&table, &a, &adc_bases)),
                ("dispatched", adc_fn()(&table, &a, &adc_bases)),
            ] {
                assert!(
                    (got - plain).abs() <= 1e-3 * plain.abs().max(1.0),
                    "adc {name} m={m}: {got} vs {plain}"
                );
            }
        }
    }
}

#[cfg(test)]
mod quantized_tests {
    use super::*;
    use crate::search::index_traits::SerializableIndex;

    const DIM: usize = 64;

    /// Distinct unit vectors drawn around 40 shared centres, so the data has the structure a
    /// product quantizer relies on. Uniform noise has none: 8 subspaces of 32 centroids over
    /// 64 uniform dimensions lose most of the neighbourhood, and a first version of these tests
    /// on uniform data read 0.58 recall and looked like a bug in the walk. It was the data --
    /// the brute-force scan over the same codes scored the same. `centres_seed` is fixed so
    /// that query and corpus draws share the centres.
    fn unit_vectors(n: usize, seed: u64) -> Vec<DenseVector> {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let mut cstate = 0xC0FFEE_u64;
        let mut cnext = move || {
            cstate ^= cstate << 13;
            cstate ^= cstate >> 7;
            cstate ^= cstate << 17;
            (cstate >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let centres: Vec<DenseVector> = (0..40)
            .map(|_| (0..DIM).map(|_| cnext()).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centres[i % centres.len()];
                let mut v: DenseVector = (0..DIM).map(|d| c[d] + 0.15 * next()).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect()
    }

    /// recall@10 of a brute-force scan over the codes with an exact rerank of the best `r`:
    /// the most any walk on the same codes can reach.
    fn scan_bound(
        index: &HnswIndex,
        corpus: &[DenseVector],
        queries: &[DenseVector],
        r: usize,
    ) -> f64 {
        let store = index.quantized.as_ref().unwrap();
        let mut hits = 0;
        for q in queries {
            let truth = exact_top(corpus, q, 10);
            let QueryScorer::Adc(table) = index.scorer(q) else {
                panic!("quantized graph must give an ADC scorer")
            };
            let mut scored: Vec<(f32, usize)> = (0..corpus.len())
                .map(|i| (store.adc(&table, i), i))
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut reranked: Vec<(f32, usize)> = scored
                .iter()
                .take(r)
                .map(|&(_, i)| {
                    (
                        1.0 - crate::search::simd_distance::dot_product(q, &corpus[i]),
                        i,
                    )
                })
                .collect();
            reranked.sort_by(|a, b| a.0.total_cmp(&b.0));
            hits += reranked
                .iter()
                .take(10)
                .filter(|(_, i)| truth.contains(i))
                .count();
        }
        hits as f64 / (10 * queries.len()) as f64
    }

    fn exact_top(corpus: &[DenseVector], q: &DenseVector, k: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (1.0 - crate::search::simd_distance::dot_product(q, v), i))
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    fn build(corpus: &[DenseVector]) -> HnswIndex {
        let mut index = HnswIndex::new(
            DistanceMetric::Cosine,
            Some(HnswConfig::from_params(16, 100, 64)),
        );
        let items: Vec<(RecordId, DenseVector)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v.clone()))
            .collect();
        index.insert_batch(items).unwrap();
        index
    }

    fn trained_pq(corpus: &[DenseVector]) -> ProductQuantizer {
        // 16 subspaces of 4 dims, 64 centroids: fine enough that the codes keep the
        // neighbourhood (the scan bound below reads it), small enough to train on 3,000.
        let mut pq = ProductQuantizer::new(DIM, 16, 64, DistanceMetric::Euclidean).unwrap();
        pq.train(corpus, 10).unwrap();
        pq
    }

    /// recall@10 of `index` against brute force, reranking the walk's best `r` exactly.
    fn recall(
        index: &HnswIndex,
        corpus: &[DenseVector],
        queries: &[DenseVector],
        ef: usize,
        r: usize,
    ) -> f64 {
        let mut hits = 0;
        for q in queries {
            let truth = exact_top(corpus, q, 10);
            let walk = index.search_with_ef(q, r, ef).unwrap();
            let mut reranked: Vec<(f32, usize)> = walk
                .iter()
                .map(|(id, _)| {
                    let i: usize = id.parse().unwrap();
                    (
                        1.0 - crate::search::simd_distance::dot_product(q, &corpus[i]),
                        i,
                    )
                })
                .collect();
            reranked.sort_by(|a, b| a.0.total_cmp(&b.0));
            hits += reranked
                .iter()
                .take(10)
                .filter(|(_, i)| truth.contains(i))
                .count();
        }
        hits as f64 / (10 * queries.len()) as f64
    }

    #[test]
    fn a_quantized_walk_finds_what_the_exact_walk_finds() {
        let corpus = unit_vectors(3000, 0x9E3779B97F4A7C15);
        let queries = unit_vectors(50, 0xD1B54A32D192ED03);
        let mut index = build(&corpus);
        let exact = recall(&index, &corpus, &queries, 64, 10);
        assert!(exact > 0.9, "exact walk recall {exact}");

        index.quantize(trained_pq(&corpus), None).unwrap();
        assert!(index.is_quantized());
        assert!(
            index.nodes.vectors_dropped(),
            "quantizing must drop the stored vectors"
        );
        // Two properties, kept apart. The walk must find what the codes still hold: within
        // a few points of a brute-force scan over the same codes. And the codes plus a rerank
        // must land near the exact walk -- with a 200-candidate rerank here, because
        // 16 x 64-centroid codes on this synthetic data are coarser than the 192-byte codes
        // measured on real embeddings (where 500 candidates matched exact recall@100).
        let walk = recall(&index, &corpus, &queries, 256, 200);
        let bound = scan_bound(&index, &corpus, &queries, 200);
        assert!(
            walk >= bound - 0.05,
            "walk on codes with a 200-candidate rerank: {walk}; a brute-force scan over the \
             same codes reaches {bound}, so the walk is losing what the codes still have"
        );
        assert!(
            walk >= exact - 0.1,
            "walk on codes with a 200-candidate rerank: {walk}, exact walk: {exact}, scan \
             bound {bound}"
        );
    }

    #[test]
    fn inserts_after_quantizing_are_found() {
        let mut corpus = unit_vectors(3000, 0x2545F4914F6CDD1D);
        let mut index = build(&corpus);
        index.quantize(trained_pq(&corpus), None).unwrap();

        let more = unit_vectors(500, 0xA0761D6478BD642F);
        let items: Vec<(RecordId, DenseVector)> = more
            .iter()
            .enumerate()
            .map(|(i, v)| ((3000 + i).to_string(), v.clone()))
            .collect();
        index.insert_batch(items).unwrap();
        corpus.extend(more.iter().cloned());
        assert_eq!(index.quantized.as_ref().unwrap().codes.len(), 3500 * 16);

        // Each late vector, queried by itself and reranked exactly, comes back first.
        let mut found = 0;
        for (i, v) in more.iter().enumerate().take(100) {
            let walk = index.search_with_ef(v, 50, 64).unwrap();
            let best = walk
                .iter()
                .map(|(id, _)| id.parse::<usize>().unwrap())
                .min_by(|&a, &b| {
                    let da = 1.0 - crate::search::simd_distance::dot_product(v, &corpus[a]);
                    let db = 1.0 - crate::search::simd_distance::dot_product(v, &corpus[b]);
                    da.total_cmp(&db)
                })
                .unwrap();
            if best == 3000 + i {
                found += 1;
            }
        }
        assert!(
            found >= 95,
            "only {found}/100 late inserts found themselves"
        );
    }

    #[test]
    fn a_quantized_graph_round_trips_as_v4() {
        let corpus = unit_vectors(1500, 0x1234_5678_9ABC_DEF1);
        let dir = tempfile::tempdir().unwrap();

        let plain = build(&corpus);
        let plain_path = dir.path().join("plain.idx");
        plain.save(&plain_path).unwrap();
        assert_eq!(
            crate::search::index_persistence::read_kind(&plain_path).unwrap(),
            "hnsw-v4"
        );
        let plain_back = HnswIndex::load(&plain_path).unwrap();
        assert!(!plain_back.is_quantized());
        assert_eq!(plain_back.nodes.len(), 1500);

        let mut quantized = build(&corpus);
        quantized.quantize(trained_pq(&corpus), None).unwrap();
        let q_path = dir.path().join("quantized.idx");
        quantized.save(&q_path).unwrap();
        assert_eq!(
            crate::search::index_persistence::read_kind(&q_path).unwrap(),
            "hnsw-v4"
        );
        let back = HnswIndex::load(&q_path).unwrap();
        assert!(back.is_quantized());
        let store = back.quantized.as_ref().unwrap();
        assert_eq!(store.codes.len(), 1500 * 16);
        assert!(
            !store.flat.is_empty(),
            "the flat codebook is rebuilt on load, not stored"
        );
        for q in corpus.iter().take(20) {
            assert_eq!(
                quantized.search_with_ef(q, 10, 64).unwrap(),
                back.search_with_ef(q, 10, 64).unwrap(),
                "a loaded quantized graph must answer exactly as the one that was saved"
            );
        }
    }
}
