use crate::types::{DenseVector, DistanceMetric, RecordId};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use crate::search::simd_distance::SimdDistance;
use crate::precomputed_query::PrecomputedQuery;



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
            ml: 1.0 / (2.0_f64.ln()),
            seed: 42,
            // Compatibility defaults
            max_connections: 16,
            level_multiplier: 1.0 / (2.0_f64.ln()),
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
            ml: 1.0 / (2.0_f64.ln()),
            seed: 42,
            // Compatibility
            max_connections: m,
            level_multiplier: 1.0 / (2.0_f64.ln()),
            max_level: 16,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    id: RecordId,
    vector: DenseVector,
    level: usize,
    neighbors: Vec<Vec<usize>>,
    deleted: bool,  // Tombstone flag for soft deletion
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
    
    fn calculate_distance(&self, a: &DenseVector, b: &DenseVector) -> f32 {
        // ✅ Use AVX-512/AVX2/SSE optimized SIMD distance calculation via trait
        self.distance_metric.calculate_simd(a, b)
            .unwrap_or_else(|_| {
                // Fallback to scalar calculation if SIMD fails
                match self.distance_metric {
                    DistanceMetric::Cosine => {
                        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                        1.0 - (dot / (norm_a * norm_b))
                    }
                    DistanceMetric::Euclidean => {
                        a.iter().zip(b.iter())
                            .map(|(x, y)| (x - y).powi(2))
                            .sum::<f32>()
                            .sqrt()
                    }
                    DistanceMetric::Dot => {
                        -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
                    }
                    _ => 0.0,
                }
            })
    }
    
    fn select_level(&mut self) -> usize {
        let mut level = 0;
        let ml = self.config.ml;
        while self.rng.gen::<f64>() < ml && level < 16 {
            level += 1;
        }
        level
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
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();  // Min-heap for exploration
        let mut w = BinaryHeap::new();  // Max-heap for results

        // Initialize with entry points
        for idx in entry_points {
            if idx >= self.nodes.len() || self.nodes[idx].deleted {
                continue;  // Skip deleted nodes
            }
            // Entry points that don't match the filter still serve as graph entry,
            // but we skip adding them to the result set w.
            if filter_idxs.is_none_or(|f| f.contains(&idx)) {
                let dist = self.calculate_distance(query, &self.nodes[idx].vector);
                candidates.push(Candidate { idx, distance: dist });
                w.push(Result { idx, distance: dist });
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
            if layer < node.neighbors.len() {
                for &neighbor_idx in &node.neighbors[layer] {
                    if neighbor_idx >= self.nodes.len() || visited.contains(&neighbor_idx) || self.nodes[neighbor_idx].deleted {
                        continue;
                    }
                    // Mark visited regardless — prevents redundant re-visits
                    visited.insert(neighbor_idx);

                    let dist = self.calculate_distance(query, &self.nodes[neighbor_idx].vector);

                    // Always add to exploration candidates so graph can bridge through non-matching nodes
                    if dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY) || w.len() < ef {
                        candidates.push(Candidate { idx: neighbor_idx, distance: dist });
                    }

                    // Only add to result window if this node passes the filter
                    if filter_idxs.is_none_or(|f| f.contains(&neighbor_idx))
                        && (dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY) || w.len() < ef)
                    {
                        w.push(Result { idx: neighbor_idx, distance: dist });
                        // Prune results if we have too many
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vector, filtering out deleted nodes
        let mut results: Vec<_> = w.into_iter()
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
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();

        // Initialize with entry points
        for idx in entry_points {
            if idx >= self.nodes.len() || self.nodes[idx].deleted {
                continue;
            }
            if filter_idxs.is_none_or(|f| f.contains(&idx)) {
                let dist = self.calculate_distance(query, &self.nodes[idx].vector);
                candidates.push(Candidate { idx, distance: dist });
                w.push(Result { idx, distance: dist });
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
            if layer < node.neighbors.len() {
                let neighbors = &node.neighbors[layer];

                // Parallel distance calculation for neighbors (layer 0 optimization);
                // filtered-out neighbours are excluded from distance computation entirely.
                let candidate_neighbors: Vec<usize> = neighbors.iter()
                    .filter(|&&neighbor_idx| {
                        neighbor_idx < self.nodes.len()
                            && !visited.contains(&neighbor_idx)
                            && !self.nodes[neighbor_idx].deleted
                    })
                    .copied()
                    .collect();

                // Mark all as visited before computing distances (prevents re-entry)
                for &idx in &candidate_neighbors {
                    visited.insert(idx);
                }

                // Now compute distances only for filter-matching nodes
                let neighbor_distances: Vec<_> = if candidate_neighbors.len() > 8 {
                    candidate_neighbors.par_iter()
                        .filter(|&&idx| filter_idxs.is_none_or(|f| f.contains(&idx)))
                        .map(|&neighbor_idx| {
                            let dist = self.calculate_distance(query, &self.nodes[neighbor_idx].vector);
                            (neighbor_idx, dist)
                        })
                        .collect()
                } else {
                    candidate_neighbors.iter()
                        .filter(|&&idx| filter_idxs.is_none_or(|f| f.contains(&idx)))
                        .map(|&neighbor_idx| {
                            let dist = self.calculate_distance(query, &self.nodes[neighbor_idx].vector);
                            (neighbor_idx, dist)
                        })
                        .collect()
                };

                // Process computed distances from filter-passing neighbors
                // (candidate_neighbors were already all marked visited above)
                for (neighbor_idx, dist) in neighbor_distances {
                    // These already passed the filter check in the collection above
                    if dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY) || w.len() < ef {
                        candidates.push(Candidate { idx: neighbor_idx, distance: dist });
                        w.push(Result { idx: neighbor_idx, distance: dist });
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }

                // Also add non-filter-matching candidates for graph traversal (bridges)
                if filter_idxs.is_some() {
                    for &idx in candidate_neighbors.iter().filter(|&&i| !filter_idxs.unwrap().contains(&i)) {
                        let dist = self.calculate_distance(query, &self.nodes[idx].vector);
                        if dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY) || candidates.len() < ef {
                            candidates.push(Candidate { idx, distance: dist });
                        }
                    }
                }
            }
        }

        let mut results: Vec<_> = w.into_iter()
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
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        
        // Initialize
        for idx in entry_points {
            if idx >= self.nodes.len() || self.nodes[idx].deleted {
                continue;  // Skip deleted nodes
            }
            let dist = self.calculate_distance(query, &self.nodes[idx].vector);
            candidates.push(Candidate { idx, distance: dist });
            w.push(Result { idx, distance: dist });
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
            if layer < node.neighbors.len() {
                for &neighbor_idx in &node.neighbors[layer] {
                    if neighbor_idx >= self.nodes.len() || visited.contains(&neighbor_idx) || self.nodes[neighbor_idx].deleted {
                        continue;  // Skip deleted nodes
                    }
                    visited.insert(neighbor_idx);

                    let dist = self.calculate_distance(query, &self.nodes[neighbor_idx].vector);
                    
                    // Update if better
                    if dist < w.peek().map(|r| r.distance).unwrap_or(f32::INFINITY) {
                        candidates.push(Candidate { idx: neighbor_idx, distance: dist });
                        w.clear();
                        w.push(Result { idx: neighbor_idx, distance: dist });
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
    pub fn search_with_ef(&self, query: &DenseVector, k: usize, ef_override: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        let entry = self.entry_point.unwrap();
        let entry_level = self.nodes[entry].level;
        let mut curr_nearest = vec![entry];

        // Search from top layer to layer 1 (greedy search for closest point)
        for lc in (1..=entry_level).rev() {
            curr_nearest = self.search_layer_closest(query, curr_nearest, lc);
        }

        // Automatic parallel search for large datasets (>1000 nodes)
        let ef = ef_override.max(k);
        let candidates = if self.nodes.len() > 1000 {
            self.search_layer_parallel(query, curr_nearest, ef, 0, None)
        } else {
            self.search_layer(query, curr_nearest, ef, 0, None)
        };

        // Return top k
        Ok(candidates.into_iter()
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

        Ok(candidates.into_iter()
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
        let candidates = if self.nodes.len() > 1000 {
            self.search_layer_parallel(&precomputed.vector, curr_nearest, ef, 0, None)
        } else {
            self.search_layer(&precomputed.vector, curr_nearest, ef, 0, None)
        };

        Ok(candidates.into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes[idx].id.clone(), dist))
            .collect())
    }
    
    pub fn insert(&mut self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        if self.id_to_idx.contains_key(&id) {
            return Err(crate::Error::InvalidInput(format!("ID {} already exists", id)));
        }
        
        let idx = self.nodes.len();
        let level = self.select_level();
        
        let mut node = Node {
            id: id.clone(),
            vector: vector.clone(),
            level,
            neighbors: vec![Vec::new(); level + 1],
            deleted: false,  // New nodes are not deleted
        };
        
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
            
            // Select M nearest neighbors
            let neighbors: Vec<usize> = candidates.iter()
                .take(m)
                .map(|(idx, _)| *idx)
                .collect();
            
            node.neighbors[lc] = neighbors.clone();
            
            // Update curr_nearest for next layer
            if lc > 0 {
                curr_nearest = candidates.into_iter()
                    .map(|(idx, _)| idx)
                    .collect();
            }
        }
        
        // Add node
        self.nodes.push(node);
        self.id_to_idx.insert(id, idx);
        
        // Add bidirectional links
        for lc in 0..=level.min(entry_level) {
            let neighbors = self.nodes[idx].neighbors[lc].clone();
            for neighbor_idx in neighbors {
                if lc < self.nodes[neighbor_idx].neighbors.len() {
                    self.nodes[neighbor_idx].neighbors[lc].push(idx);
                    
                    // Prune if needed
                    let max_neighbors = if lc == 0 {
                        self.config.max_m0
                    } else {
                        self.config.m
                    };
                    
                    if self.nodes[neighbor_idx].neighbors[lc].len() > max_neighbors {
                        // Prune to M closest
                        let neighbor_vec = &self.nodes[neighbor_idx].vector;
                        let mut neighbors_with_dist: Vec<_> = self.nodes[neighbor_idx].neighbors[lc]
                            .iter()
                            .map(|&n_idx| {
                                let dist = self.calculate_distance(neighbor_vec, &self.nodes[n_idx].vector);
                                (n_idx, dist)
                            })
                            .collect();
                        
                        neighbors_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        self.nodes[neighbor_idx].neighbors[lc] = neighbors_with_dist
                            .into_iter()
                            .take(max_neighbors)
                            .map(|(idx, _)| idx)
                            .collect();
                    }
                }
            }
        }
        
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
    
    async fn search(&self, query: &[f32], k: usize) -> crate::Result<Vec<(String, f32)>> {
        self.search(&query.to_vec(), k)
    }

    async fn search_with_ef_hint(&self, query: &[f32], k: usize, ef: usize) -> crate::Result<Vec<(String, f32)>> {
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
        let idx = *self.id_to_idx.get(id)
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
        let deleted_neighbors: Vec<Vec<usize>> = self.nodes[deleted_idx].neighbors.clone();

        // For each layer where the deleted node had connections
        for (layer, neighbors_in_layer) in deleted_neighbors.iter().enumerate().take(deleted_level + 1) {
            let neighbors_of_deleted = neighbors_in_layer.clone();

            // Remove deleted node from all its neighbors' neighbor lists
            for &neighbor_idx in &neighbors_of_deleted {
                if neighbor_idx < self.nodes.len()
                    && !self.nodes[neighbor_idx].deleted
                    && layer < self.nodes[neighbor_idx].neighbors.len()
                {
                    self.nodes[neighbor_idx].neighbors[layer].retain(|&n| n != deleted_idx);
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
                        .filter(|&&n| n != neighbor_a && n < self.nodes.len() && !self.nodes[n].deleted)
                        .map(|&n| {
                            let dist = self.calculate_distance(
                                &self.nodes[neighbor_a].vector,
                                &self.nodes[n].vector
                            );
                            (n, dist)
                        })
                        .collect();

                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    // Add connections to maintain connectivity (limit to M/2 to avoid over-connection)
                    let max_new_connections = self.config.m / 2;
                    for (new_neighbor, _) in distances.into_iter().take(max_new_connections) {
                        if layer < self.nodes[neighbor_a].neighbors.len() {
                            let neighbors = &mut self.nodes[neighbor_a].neighbors[layer];
                            if !neighbors.contains(&new_neighbor) {
                                neighbors.push(new_neighbor);

                                // Add bidirectional edge
                                if layer < self.nodes[new_neighbor].neighbors.len() {
                                    let reverse_neighbors = &mut self.nodes[new_neighbor].neighbors[layer];
                                    if !reverse_neighbors.contains(&neighbor_a) {
                                        reverse_neighbors.push(neighbor_a);
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
        crate::search::index_persistence::write_index(path, "hnsw", self)
    }

    fn load(path: &std::path::Path) -> crate::Result<Self> {
        // The RNG is `#[serde(skip)]` and comes back seeded from `default_rng`, not from
        // `config.seed`. Level assignment for vectors inserted after a load therefore follows a
        // different sequence than it would have without the restart. That changes the graph
        // shape, not its correctness — HNSW makes no guarantee about which levels a node lands
        // on, only that search respects the levels it finds.
        crate::search::index_persistence::read_index(path, "hnsw")
    }

    fn stats(&self) -> crate::search::index_traits::IndexStats {
        let dimension = self.nodes.first().map(|n| n.vector.len()).unwrap_or(0);
        let vectors_bytes = self.nodes.len() * dimension * std::mem::size_of::<f32>();
        let graph_bytes: usize = self
            .nodes
            .iter()
            .map(|n| n.neighbors.iter().map(|l| l.len()).sum::<usize>() * std::mem::size_of::<usize>())
            .sum();

        crate::search::index_traits::IndexStats {
            // `nodes` keeps tombstoned entries, so this counts live vectors rather than slots.
            total_vectors: self.nodes.iter().filter(|n| !n.deleted).count(),
            index_size_bytes: vectors_bytes + graph_bytes,
            build_time_ms: 0,
        }
    }
}
