use crate::precomputed_query::PrecomputedQuery;
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
    use std::cell::RefCell;

    thread_local! {
        static BUFFER: RefCell<(u32, Vec<u32>)> = const { RefCell::new((0, Vec::new())) };
    }

    pub(super) struct Visited {
        generation: u32,
        stamps: Vec<u32>,
    }

    impl Visited {
        /// Take the thread's buffer, sized for `capacity` nodes, and start a new generation.
        pub(super) fn start(capacity: usize) -> Self {
            BUFFER.with(|cell| {
                let (generation, stamps) = &mut *cell.borrow_mut();
                let mut stamps = std::mem::take(stamps);
                if stamps.len() < capacity {
                    stamps.resize(capacity, 0);
                }
                // Wrapping past u32::MAX would make stale stamps read as current, so clear
                // once every four billion traversals rather than never.
                *generation = generation.wrapping_add(1);
                if *generation == 0 {
                    stamps.iter_mut().for_each(|s| *s = 0);
                    *generation = 1;
                }
                Visited {
                    generation: *generation,
                    stamps,
                }
            })
        }

        pub(super) fn contains(&self, idx: usize) -> bool {
            self.stamps.get(idx) == Some(&self.generation)
        }

        /// Mark `idx` visited, returning whether it already was.
        pub(super) fn insert(&mut self, idx: usize) -> bool {
            if idx >= self.stamps.len() {
                self.stamps.resize(idx + 1, 0);
            }
            let seen = self.stamps[idx] == self.generation;
            self.stamps[idx] = self.generation;
            seen
        }
    }

    impl Drop for Visited {
        fn drop(&mut self) {
            // Hand the allocation back so the next traversal on this thread reuses it.
            BUFFER.with(|cell| {
                cell.borrow_mut().1 = std::mem::take(&mut self.stamps);
            });
        }
    }
}

/// One vector and its outgoing edges, one edge list per layer.
///
/// The edge lists are behind a lock each because pruning a neighbour's list is where insert
/// spends most of its CPU (about 78% of the profile, all of it distance calculations), and a
/// single lock over the whole graph is what kept that on one core. Per-node locking is what
/// Qdrant, Weaviate and hnswlib all do for the same reason.
///
/// `RwLock` is not `Clone`, so neither is `Node` — nothing ever cloned a whole one; the
/// clones in this file are all of a single field. `serde(rename)` keeps the field's name in
/// self-describing formats, and parking_lot's `serde` impl is transparent, so the bincode
/// encoding is byte-for-byte what `Vec<Vec<usize>>` wrote. `hnsw_v2_files_still_load` pins
/// that against a file written before this change.
#[derive(Debug, Serialize, Deserialize)]
struct Node {
    id: RecordId,
    vector: DenseVector,
    level: usize,
    #[serde(rename = "neighbors")]
    links: Vec<RwLock<Vec<usize>>>,
    deleted: bool, // Tombstone flag for soft deletion
}

impl Node {
    /// A fresh node with `level + 1` empty edge lists.
    fn new(id: RecordId, vector: DenseVector, level: usize) -> Self {
        Node {
            id,
            vector,
            level,
            links: (0..=level).map(|_| RwLock::new(Vec::new())).collect(),
            deleted: false,
        }
    }

    fn layers(&self) -> usize {
        self.links.len()
    }
}

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

/// Below this many nodes a batch is inserted one record at a time. Early records are
/// what the rest of the graph hangs off, so letting a batch of them plan without seeing
/// each other costs real recall; past this size it does not.
const PARALLEL_PLAN_MIN_NODES: usize = 10_000;
/// Below this many records the rayon fan-out costs more than the serial loop.
const PARALLEL_PLAN_MIN_BATCH: usize = 16;

#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    #[serde(skip, default = "default_rng")]
    rng: StdRng,
    config: HnswConfig,
    distance_metric: DistanceMetric,
    nodes: Vec<Node>,
    id_to_idx: HashMap<RecordId, usize>,
    entry_point: Option<usize>,
}

fn default_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

impl HnswIndex {
    pub fn new(distance_metric: DistanceMetric, config: Option<HnswConfig>) -> Self {
        let config = config.unwrap_or_default();
        Self {
            rng: StdRng::seed_from_u64(config.seed),
            config,
            distance_metric,
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            entry_point: None,
        }
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

    fn calculate_distance(&self, a: &DenseVector, b: &DenseVector) -> f32 {
        // Cosine inputs are unit length by construction (see `prepare`), so the similarity is the
        // dot product and the distance is one minus it.
        if matches!(self.distance_metric, DistanceMetric::Cosine) {
            return 1.0 - crate::search::simd_distance::dot_product(a, b);
        }
        // ✅ Use AVX-512/AVX2/SSE optimized SIMD distance calculation via trait
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
    /// `candidates` must be sorted nearest-first, which is what `search_layer` returns.
    fn select_neighbors_heuristic(&self, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(m);
        for &(idx, dist_to_base) in candidates {
            if selected.len() >= m {
                break;
            }
            let crowded = selected.iter().any(|&kept| {
                self.calculate_distance(&self.nodes[idx].vector, &self.nodes[kept].vector)
                    < dist_to_base
            });
            if !crowded {
                selected.push(idx);
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
        query: &DenseVector,
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
        filter_idxs: Option<&HashSet<usize>>,
    ) -> Vec<(usize, f32)> {
        let mut visited = visited::Visited::start(self.nodes.len());
        let mut candidates = BinaryHeap::new(); // Min-heap for exploration
        let mut w = BinaryHeap::new(); // Max-heap for results

        // Initialize with entry points
        for idx in entry_points {
            if idx >= self.nodes.len() || self.nodes[idx].deleted {
                continue; // Skip deleted nodes
            }
            // An entry point that fails the filter is still the way into the graph: it goes on
            // the exploration queue, and only the *result* set is gated on the filter. Gating
            // both is how a filtered search returns nothing — with the queue empty the loop
            // below never runs, so a selective filter came back empty for every query whose
            // entry point happened not to match. At 2% selectivity that was 98% of queries.
            let dist = self.calculate_distance(query, &self.nodes[idx].vector);
            candidates.push(Candidate {
                idx,
                distance: dist,
            });
            if filter_idxs.is_none_or(|f| f.contains(&idx)) {
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
            let node = &self.nodes[current.idx];
            if layer < node.layers() {
                // The guard is held across the walk, which reads only `vector` and `deleted`
                // on the neighbours and takes no second lock — so a writer pruning some other
                // node cannot be waiting on this one.
                let links = node.links[layer].read();
                for &neighbor_idx in links.iter() {
                    if neighbor_idx >= self.nodes.len()
                        || visited.contains(neighbor_idx)
                        || self.nodes[neighbor_idx].deleted
                    {
                        continue;
                    }
                    // Mark visited regardless — prevents redundant re-visits
                    visited.insert(neighbor_idx);

                    let dist = self.calculate_distance(query, &self.nodes[neighbor_idx].vector);

                    // Always add to exploration candidates so graph can bridge through non-matching nodes
                    if dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY) || w.len() < ef
                    {
                        candidates.push(Candidate {
                            idx: neighbor_idx,
                            distance: dist,
                        });
                    }

                    // Only add to result window if this node passes the filter
                    if filter_idxs.is_none_or(|f| f.contains(&neighbor_idx))
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
            .filter(|r| !self.nodes[r.idx].deleted)
            .map(|r| (r.idx, r.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    /// Parallel search at layer 0 for improved throughput.
    /// Accepts the same `filter_idxs` parameter as `search_layer` for Acorn filtering.
    fn search_layer_parallel(
        &self,
        query: &DenseVector,
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
        filter_idxs: Option<&HashSet<usize>>,
    ) -> Vec<(usize, f32)> {
        let mut visited = visited::Visited::start(self.nodes.len());
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();

        // Initialize with entry points
        for idx in entry_points {
            if idx >= self.nodes.len() || self.nodes[idx].deleted {
                continue;
            }
            // Same rule as `search_layer`: the queue is unfiltered, the result set is not.
            let dist = self.calculate_distance(query, &self.nodes[idx].vector);
            candidates.push(Candidate {
                idx,
                distance: dist,
            });
            if filter_idxs.is_none_or(|f| f.contains(&idx)) {
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

            let node = &self.nodes[current.idx];
            if layer < node.layers() {
                // Copy the ids out and drop the guard before the `par_iter` below: handing a
                // lock guard to rayon means holding it for as long as the pool takes to
                // schedule, and a writer pruning this same node would block behind it.
                let candidate_neighbors: Vec<usize> = {
                    let links = node.links[layer].read();
                    links
                        .iter()
                        .filter(|&&neighbor_idx| {
                            neighbor_idx < self.nodes.len()
                                && !visited.contains(neighbor_idx)
                                && !self.nodes[neighbor_idx].deleted
                        })
                        .copied()
                        .collect()
                };

                // Mark all as visited before computing distances (prevents re-entry)
                for &idx in &candidate_neighbors {
                    visited.insert(idx);
                }

                // Now compute distances only for filter-matching nodes
                let neighbor_distances: Vec<_> = if candidate_neighbors.len() > 8 {
                    candidate_neighbors
                        .par_iter()
                        .filter(|&&idx| filter_idxs.is_none_or(|f| f.contains(&idx)))
                        .map(|&neighbor_idx| {
                            let dist =
                                self.calculate_distance(query, &self.nodes[neighbor_idx].vector);
                            (neighbor_idx, dist)
                        })
                        .collect()
                } else {
                    candidate_neighbors
                        .iter()
                        .filter(|&&idx| filter_idxs.is_none_or(|f| f.contains(&idx)))
                        .map(|&neighbor_idx| {
                            let dist =
                                self.calculate_distance(query, &self.nodes[neighbor_idx].vector);
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
                if filter_idxs.is_some() {
                    for &idx in candidate_neighbors
                        .iter()
                        .filter(|&&i| !filter_idxs.unwrap().contains(&i))
                    {
                        let dist = self.calculate_distance(query, &self.nodes[idx].vector);
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
            .filter(|r| !self.nodes[r.idx].deleted)
            .map(|r| (r.idx, r.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    /// Greedy search for upper layers (find single closest point)
    fn search_layer_closest(
        &self,
        query: &DenseVector,
        entry_points: Vec<usize>,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = visited::Visited::start(self.nodes.len());
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();

        // Initialize
        for idx in entry_points {
            if idx >= self.nodes.len() || self.nodes[idx].deleted {
                continue; // Skip deleted nodes
            }
            let dist = self.calculate_distance(query, &self.nodes[idx].vector);
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

            let node = &self.nodes[current.idx];
            if layer < node.layers() {
                let links = node.links[layer].read();
                for &neighbor_idx in links.iter() {
                    if neighbor_idx >= self.nodes.len()
                        || visited.contains(neighbor_idx)
                        || self.nodes[neighbor_idx].deleted
                    {
                        continue; // Skip deleted nodes
                    }
                    visited.insert(neighbor_idx);

                    let dist = self.calculate_distance(query, &self.nodes[neighbor_idx].vector);

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

        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes[entry].level;
        let mut curr_nearest = vec![entry];

        // Search from top layer to layer 1 (greedy search for closest point)
        for lc in (1..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(query, curr_nearest, lc);
        }

        // Rayon has a non-trivial scheduling cost for the small neighbor lists
        // in a normal HNSW graph (M=16 => at most 32 neighbors at layer 0).
        // The old `>1000 nodes` switch therefore made every 10K query slower
        // than the scalar traversal.  Keep the parallel path for filtered
        // searches, where it can avoid scoring rejected candidates, but use
        // the cache-friendly sequential loop for the common unfiltered case.
        let ef = ef_override.max(k);
        let candidates = self.search_layer(query, curr_nearest, ef, 0, None);

        // Return top k
        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes[idx].id.clone(), dist))
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
        let filter_idxs: HashSet<usize> = filter_ids
            .iter()
            .filter_map(|id| self.id_to_idx.get(id).copied())
            .collect();

        if filter_idxs.is_empty() {
            return Ok(Vec::new());
        }
        let query = &self.prepare(query.clone());

        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes[entry].level;
        let mut curr_nearest = vec![entry];

        // Upper layers: greedy traversal without filter (find good entry to layer 0)
        for lc in (1..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(query, curr_nearest, lc);
        }

        // Layer 0: apply Acorn filter
        let ef = ef_override.max(k);
        let candidates = if self.nodes.len() > 1000 {
            self.search_layer_parallel(query, curr_nearest, ef, 0, Some(&filter_idxs))
        } else {
            self.search_layer(query, curr_nearest, ef, 0, Some(&filter_idxs))
        };

        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes[idx].id.clone(), dist))
            .collect())
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
        let entry_level = self.nodes[entry].level;
        let mut curr_nearest = vec![entry];

        // Search upper layers
        for lc in (1..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(&precomputed.vector, curr_nearest, lc);
        }

        // Search layer 0 with precomputed optimization
        let ef = self.config.ef_search.max(k);
        let candidates = self.search_layer(&precomputed.vector, curr_nearest, ef, 0, None);

        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes[idx].id.clone(), dist))
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
    pub fn insert_batch(&mut self, items: Vec<(RecordId, DenseVector)>) -> crate::Result<()> {
        if items.len() < PARALLEL_PLAN_MIN_BATCH
            || self.nodes.len() < PARALLEL_PLAN_MIN_NODES
            || self.entry_point.is_none()
        {
            for (id, vector) in items {
                self.insert(id, vector)?;
            }
            return Ok(());
        }

        // Levels come from the RNG, which needs `&mut self`. Draw them here so that planning
        // needs nothing but `&self`. Duplicate ids are rejected against the graph and against
        // the rest of the batch — the serial path catches the second one because the first is
        // already in `id_to_idx`, and nothing in this batch is yet.
        let mut prepared: Vec<(RecordId, DenseVector, usize)> = Vec::with_capacity(items.len());
        let mut seen: HashSet<RecordId> = HashSet::with_capacity(items.len());
        for (id, vector) in items {
            if self.id_to_idx.contains_key(&id) || !seen.insert(id.clone()) {
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
        // plans against, and phases B and C use the same value so a plan and its application
        // never disagree about which layers were searched.
        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes[entry].level;

        // A. Plan. Read-only, `&self`, every core.
        let plans: Vec<Vec<Vec<usize>>> = prepared
            .par_iter()
            .map(|(_, vector, level)| self.plan_neighbors(vector, *level, entry, entry_level))
            .collect();

        // B. Append the nodes and their own forward edges. Serial `&mut self`, but pure
        // pushes -- for a 380-record group at 768 dimensions this is about a megabyte of
        // memcpy. The whole batch is appended before phase C starts, so `nodes` does not
        // grow while C runs and no index into it can be invalidated.
        let mut applied: Vec<(usize, usize)> = Vec::with_capacity(plans.len());
        let mut new_entry: Option<usize> = None;
        for ((id, vector, level), plan) in prepared.into_iter().zip(plans) {
            let idx = self.push_planned_node(id, vector, level, plan);
            if level > entry_level {
                // Last one wins, exactly as the one-at-a-time path did: every plan in this
                // batch was made against `entry`, so a mid-batch change would make some plans
                // disagree with the graph they were built for.
                new_entry = Some(idx);
            }
            applied.push((idx, level));
        }

        // C. Reverse edges and pruning. `&self`, one edge lock at a time, every core. This
        // is the phase worth parallelising: the pruning inside `link_back` is where the
        // insert path spent about 78% of its CPU, and it was serial.
        let graph: &Self = self;
        applied
            .par_iter()
            .for_each(|&(idx, level)| graph.link_back_all(idx, level, entry_level));

        if let Some(idx) = new_entry {
            self.entry_point = Some(idx);
        }
        Ok(())
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

        for lc in ((level + 1).min(entry_level + 1)..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(vector, curr_nearest, lc);
        }

        for lc in (0..=level.min(entry_level)).rev() {
            let candidates = self.search_layer(
                vector,
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
    ) -> usize {
        let idx = self.nodes.len();
        let mut node = Node::new(id.clone(), vector, level);
        for (lc, chosen) in neighbors_per_layer.into_iter().enumerate() {
            *node.links[lc].get_mut() = chosen;
        }
        self.nodes.push(node);
        self.id_to_idx.insert(id, idx);
        idx
    }

    /// Add the reverse edge on every neighbour `idx` chose, pruning any list that overflows.
    ///
    /// `&self`, so a batch can run this for many nodes at once.
    fn link_back_all(&self, idx: usize, level: usize, entry_level: usize) {
        for lc in 0..=level.min(entry_level) {
            // Copy the plan out and drop the guard before taking any write lock, so that no
            // thread ever holds two edge locks at once.
            let chosen = self.nodes[idx].links[lc].read().clone();
            for neighbor_idx in chosen {
                self.link_back(neighbor_idx, lc, idx);
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
        let neighbor = &self.nodes[neighbor_idx];
        if lc >= neighbor.layers() {
            return;
        }
        let max_neighbors = if lc == 0 {
            self.config.max_m0
        } else {
            self.config.m
        };

        let mut links = neighbor.links[lc].write();
        links.push(new_idx);
        if links.len() <= max_neighbors {
            return;
        }

        // Prune by the same heuristic used to choose links in the first place. Keeping the
        // `max_neighbors` closest instead would undo the diversity on exactly the nodes that
        // matter most -- the ones popular enough to overflow.
        let mut neighbors_with_dist: Vec<_> = links
            .iter()
            .map(|&n_idx| {
                let dist = self.calculate_distance(&neighbor.vector, &self.nodes[n_idx].vector);
                (n_idx, dist)
            })
            .collect();
        neighbors_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        *links = self.select_neighbors_heuristic(&neighbors_with_dist, max_neighbors);
    }

    pub fn insert(&mut self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        let vector = self.prepare(vector);
        if self.id_to_idx.contains_key(&id) {
            return Err(crate::Error::InvalidInput(format!(
                "ID {} already exists",
                id
            )));
        }

        let idx = self.nodes.len();
        let level = self.select_level();

        let mut node = Node::new(id.clone(), vector.clone(), level);

        if self.nodes.is_empty() || self.entry_point.is_none() {
            self.nodes.push(node);
            self.id_to_idx.insert(id, idx);
            self.entry_point = Some(idx);
            return Ok(());
        }

        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes[entry].level;
        let mut curr_nearest = vec![entry];

        // Search from top to target layer
        for lc in ((level + 1).min(entry_level + 1)..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(&vector, curr_nearest, lc);
        }

        // Insert at all levels from level to 0
        for lc in (0..=level.min(entry_level)).rev() {
            let candidates = self.search_layer(
                &vector,
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

            *node.links[lc].get_mut() = self.select_neighbors_heuristic(&candidates, m);

            // Update curr_nearest for next layer
            if lc > 0 {
                curr_nearest = candidates.into_iter().map(|(idx, _)| idx).collect();
            }
        }

        // Add node
        self.nodes.push(node);
        self.id_to_idx.insert(id, idx);

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
    async fn add(&mut self, id: String, vector: Vec<f32>) -> crate::Result<()> {
        self.insert(id, vector)
    }

    async fn add_batch(&mut self, items: Vec<(String, Vec<f32>)>) -> crate::Result<()> {
        self.insert_batch(items)
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
        let idx = *self
            .id_to_idx
            .get(id)
            .ok_or_else(|| crate::Error::NotFound(format!("ID {} not found", id)))?;

        // Mark node as deleted (tombstone)
        self.nodes[idx].deleted = true;

        // Remove from id mapping
        self.id_to_idx.remove(id);

        // Reconnect neighbors to maintain graph connectivity
        self.reconnect_neighbors(idx)?;

        // Update entry point if it was deleted
        if self.entry_point == Some(idx) {
            self.find_new_entry_point();
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.id_to_idx.len()
    }

    fn snapshot_to(&self, path: &std::path::Path) -> crate::Result<bool> {
        crate::search::index_traits::SerializableIndex::save(self, path)?;
        Ok(true)
    }

    fn is_empty(&self) -> bool {
        self.id_to_idx.is_empty()
    }
}

// Helper methods for HNSW index (not part of VectorIndex trait)
impl HnswIndex {
    /// Reconnect neighbors when a node is deleted
    fn reconnect_neighbors(&mut self, deleted_idx: usize) -> crate::Result<()> {
        // Clone the data we need before borrowing mutably
        let deleted_level = self.nodes[deleted_idx].level;
        let deleted_neighbors: Vec<Vec<usize>> = self.nodes[deleted_idx]
            .links
            .iter()
            .map(|l| l.read().clone())
            .collect();

        // For each layer where the deleted node had connections
        for (layer, neighbors_in_layer) in
            deleted_neighbors.iter().enumerate().take(deleted_level + 1)
        {
            let neighbors_of_deleted = neighbors_in_layer.clone();

            // Remove deleted node from all its neighbors' neighbor lists
            for &neighbor_idx in &neighbors_of_deleted {
                if neighbor_idx < self.nodes.len()
                    && !self.nodes[neighbor_idx].deleted
                    && layer < self.nodes[neighbor_idx].layers()
                {
                    self.nodes[neighbor_idx].links[layer]
                        .write()
                        .retain(|&n| n != deleted_idx);
                }
            }

            // Optionally reconnect orphaned neighbors to each other
            // This helps maintain graph connectivity
            if neighbors_of_deleted.len() >= 2 {
                // Connect neighbors to each other to maintain graph density
                for i in 0..neighbors_of_deleted.len() {
                    let neighbor_a = neighbors_of_deleted[i];
                    if neighbor_a >= self.nodes.len() || self.nodes[neighbor_a].deleted {
                        continue;
                    }

                    // Find the closest neighbors from the deleted node's neighbors
                    let mut distances: Vec<(usize, f32)> = neighbors_of_deleted
                        .iter()
                        .filter(|&&n| {
                            n != neighbor_a && n < self.nodes.len() && !self.nodes[n].deleted
                        })
                        .map(|&n| {
                            let dist = self.calculate_distance(
                                &self.nodes[neighbor_a].vector,
                                &self.nodes[n].vector,
                            );
                            (n, dist)
                        })
                        .collect();

                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    // Add connections to maintain connectivity (limit to M/2 to avoid over-connection)
                    let max_new_connections = self.config.m / 2;
                    for (new_neighbor, _) in distances.into_iter().take(max_new_connections) {
                        if layer < self.nodes[neighbor_a].layers() {
                            let mut links_a = self.nodes[neighbor_a].links[layer].write();
                            if !links_a.contains(&new_neighbor) {
                                links_a.push(new_neighbor);
                                // Released before the reverse edge is taken: one lock at a
                                // time here too, so this path can never be the odd one out
                                // that deadlocks against a concurrent insert.
                                drop(links_a);

                                // Add bidirectional edge
                                if layer < self.nodes[new_neighbor].layers() {
                                    let mut reverse = self.nodes[new_neighbor].links[layer].write();
                                    if !reverse.contains(&neighbor_a) {
                                        reverse.push(neighbor_a);
                                    }
                                }
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

        for (idx, node) in self.nodes.iter().enumerate() {
            if !node.deleted && node.level >= best_level {
                best_idx = Some(idx);
                best_level = node.level;
            }
        }

        self.entry_point = best_idx;
    }
}

impl crate::search::index_traits::SerializableIndex for HnswIndex {
    fn save(&self, path: &std::path::Path) -> crate::Result<()> {
        crate::search::index_persistence::write_index(path, "hnsw-v2", self)
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
        crate::search::index_persistence::read_index(path, "hnsw-v2")
    }

    fn stats(&self) -> crate::search::index_traits::IndexStats {
        let dimension = self.nodes.first().map(|n| n.vector.len()).unwrap_or(0);
        let vectors_bytes = self.nodes.len() * dimension * std::mem::size_of::<f32>();
        let graph_bytes: usize = self
            .nodes
            .iter()
            .map(|n| {
                n.links.iter().map(|l| l.read().len()).sum::<usize>() * std::mem::size_of::<usize>()
            })
            .sum();

        crate::search::index_traits::IndexStats {
            // `nodes` keeps tombstoned entries, so this counts live vectors rather than slots.
            total_vectors: self.nodes.iter().filter(|n| !n.deleted).count(),
            index_size_bytes: vectors_bytes + graph_bytes,
            build_time_ms: 0,
        }
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
        index.nodes.iter().filter(|n| !n.deleted).count()
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
                let mut scored: Vec<(String, f32)> = index_a
                    .nodes
                    .iter()
                    .filter(|n| !n.deleted)
                    .map(|n| (n.id.clone(), index_a.calculate_distance(q, &n.vector)))
                    .collect();
                scored.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
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
    /// c7e1cbe: 200 clustered 16-d vectors, `HnswConfig::default()`, cosine.
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
        assert_eq!(loaded.id_to_idx.len(), 200);

        // The edges have to survive too, not just the node count -- a graph that read back
        // with every edge list empty would still pass the assertions above.
        let edges: usize = loaded
            .nodes
            .iter()
            .map(|n| n.links.iter().map(|l| l.read().len()).sum::<usize>())
            .sum();
        assert!(edges > 200, "edges did not survive the load: {edges}");

        // And it has to answer queries: every node must find itself.
        for probe in [0usize, 57, 199] {
            let node = &loaded.nodes[probe];
            let hits = loaded.search(&node.vector, 1).unwrap();
            assert_eq!(
                hits[0].0, node.id,
                "loaded graph could not find node {probe}"
            );
        }

        // The nodes region has to come back byte for byte -- that is the part this change
        // could have altered. The whole file cannot be compared: `id_to_idx` is a `HashMap`,
        // bincode writes it in iteration order, and that order is randomised per process, so
        // a golden file diverges from a faithful rewrite at the byte where the map starts
        // (44,926 of 49,124 here) no matter what the node layout does.
        //
        // Offsets: a 23-byte envelope (8 magic + 4 version + 4 kind length + "hnsw-v2"),
        // then the struct's fields in declaration order with `rng` skipped.
        let prefix = bincode::serialize(&loaded.config).unwrap().len()
            + bincode::serialize(&loaded.distance_metric).unwrap().len();
        let start = 23 + prefix;
        let nodes_bytes = bincode::serialize(&loaded.nodes).unwrap();
        assert_eq!(
            &HNSW_V2_GOLDEN[start..start + nodes_bytes.len()],
            &nodes_bytes[..],
            "`Vec<RwLock<Vec<usize>>>` does not encode like the `Vec<Vec<usize>>` that wrote \
             this file -- the index kind must go to \"hnsw-v3\""
        );
    }
}
