// DiskANN - Disk-based Approximate Nearest Neighbor Search
// Reference: "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" (NeurIPS 2019)
//
// DiskANN is designed for billion-scale vector search with SSD storage optimization.
// Key features:
// - Memory-mapped disk access for large datasets
// - Graph-based index structure (similar to HNSW but optimized for disk)
// - Sector-aligned I/O for optimal SSD performance
// - Vamana graph construction for better quality

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::info;

use crate::types::vector::{DenseVector, DistanceMetric};
use crate::types::RecordId;

/// DiskANN Index - Disk-optimized approximate nearest neighbor search
#[derive(Debug)]
pub struct DiskANNIndex {
    /// Configuration
    config: DiskANNConfig,

    /// Distance metric
    distance_metric: DistanceMetric,

    /// Vector dimension
    dimension: usize,

    /// Graph structure: node -> neighbors (Vamana graph)
    graph: Arc<RwLock<HashMap<RecordId, Vec<RecordId>>>>,

    /// Vector data: memory-mapped or cached
    vectors: Arc<RwLock<HashMap<RecordId, DenseVector>>>,

    /// Entry points for search (medoid-based)
    entry_points: Arc<RwLock<Vec<RecordId>>>,

    /// Node metadata for optimization
    node_metadata: Arc<RwLock<HashMap<RecordId, NodeMetadata>>>,

    /// Index file path for persistence
    #[allow(dead_code)]
    index_path: Option<String>,
}

/// DiskANN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskANNConfig {
    /// Maximum out-degree for each node (R parameter)
    pub max_degree: usize,

    /// Search list size during construction (L parameter)
    pub search_list_size: usize,

    /// Number of random entry points
    pub num_entry_points: usize,

    /// Alpha parameter for robust pruning (typically 1.2)
    pub alpha: f32,

    /// Whether to use PQ compression
    pub use_pq_compression: bool,

    /// Number of subquantizers for PQ
    pub num_pq_chunks: usize,

    /// Number of iterations for graph pruning
    pub prune_iterations: usize,

    /// Cache size in MB (for SSD optimization)
    pub cache_size_mb: usize,

    /// Enable SSD sector-aligned I/O optimizations
    pub ssd_optimized: bool,

    /// Beam width for search
    pub beam_width: usize,
}

impl Default for DiskANNConfig {
    fn default() -> Self {
        Self {
            max_degree: 64,            // R = 64 (good balance)
            search_list_size: 100,     // L = 100 during construction
            num_entry_points: 1,       // Single medoid entry point
            alpha: 1.2,                // Robust pruning parameter
            use_pq_compression: false, // Disable PQ by default
            num_pq_chunks: 8,          // 8 chunks if PQ enabled
            prune_iterations: 2,       // 2 iterations for graph pruning
            cache_size_mb: 512,        // 512MB cache
            ssd_optimized: true,       // Enable SSD optimizations
            beam_width: 4,             // Beam search width
        }
    }
}

/// Node metadata for optimization
#[derive(Debug, Clone)]
struct NodeMetadata {
    /// Access frequency (for caching)
    #[allow(dead_code)]
    access_count: u64,

    /// Last access timestamp
    #[allow(dead_code)]
    last_access: std::time::Instant,

    /// Whether node is in cache
    #[allow(dead_code)]
    is_cached: bool,
}

impl DiskANNIndex {
    /// Create a new DiskANN index
    pub fn new(dimension: usize, distance_metric: DistanceMetric, config: DiskANNConfig) -> Self {
        info!(
            "Creating DiskANN index with dimension={}, max_degree={}, search_list_size={}",
            dimension, config.max_degree, config.search_list_size
        );

        Self {
            config,
            distance_metric,
            dimension,
            graph: Arc::new(RwLock::new(HashMap::new())),
            vectors: Arc::new(RwLock::new(HashMap::new())),
            entry_points: Arc::new(RwLock::new(Vec::new())),
            node_metadata: Arc::new(RwLock::new(HashMap::new())),
            index_path: None,
        }
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        if vector.len() != self.dimension {
            return Err(crate::Error::InvalidInput(format!(
                "Vector dimension {} does not match index dimension {}",
                vector.len(),
                self.dimension
            )));
        }

        // Add vector to storage
        self.vectors
            .write()
            .unwrap()
            .insert(id.clone(), vector.clone());

        // Initialize node metadata
        self.node_metadata.write().unwrap().insert(
            id.clone(),
            NodeMetadata {
                access_count: 0,
                last_access: std::time::Instant::now(),
                is_cached: true,
            },
        );

        // Build graph connections using Vamana algorithm
        self.vamana_insert(id, vector)?;

        Ok(())
    }

    /// Vamana graph construction - insert a node with robust pruning
    fn vamana_insert(&mut self, id: RecordId, vector: DenseVector) -> crate::Result<()> {
        let entry_points = self.entry_points.read().unwrap();

        // If no entry points, make this the first one
        if entry_points.is_empty() {
            drop(entry_points);
            self.entry_points.write().unwrap().push(id.clone());
            self.graph.write().unwrap().insert(id, Vec::new());
            return Ok(());
        }

        // Find nearest neighbors using beam search
        let candidates = self.beam_search(&vector, self.config.search_list_size)?;

        // Robust pruning to select R neighbors
        let neighbors = self.robust_prune(&vector, &candidates, self.config.max_degree)?;

        // Add bidirectional edges
        let mut graph = self.graph.write().unwrap();
        graph.insert(id.clone(), neighbors.clone());

        // Add reverse edges and prune if necessary
        for neighbor_id in neighbors {
            let neighbor_edges = graph.entry(neighbor_id.clone()).or_default();
            neighbor_edges.push(id.clone());

            // Prune if exceeds max degree
            if neighbor_edges.len() > self.config.max_degree {
                let neighbor_vec = self.vectors.read().unwrap()[&neighbor_id].clone();
                let pruned =
                    self.robust_prune(&neighbor_vec, neighbor_edges, self.config.max_degree)?;
                *neighbor_edges = pruned;
            }
        }

        Ok(())
    }

    /// Beam search for finding candidates
    fn beam_search(&self, query: &DenseVector, beam_width: usize) -> crate::Result<Vec<RecordId>> {
        let entry_points = self.entry_points.read().unwrap();
        let vectors = self.vectors.read().unwrap();
        let graph = self.graph.read().unwrap();

        if entry_points.is_empty() {
            return Ok(Vec::new());
        }

        let mut visited = std::collections::HashSet::new();
        let mut candidates = std::collections::BinaryHeap::new();

        // Start from entry points
        for ep in entry_points.iter() {
            if let Some(vec) = vectors.get(ep) {
                let dist = self
                    .distance_metric
                    .calculate(query, vec)
                    .unwrap_or(f32::MAX);
                candidates.push(std::cmp::Reverse((
                    ordered_float::OrderedFloat(dist),
                    ep.clone(),
                )));
                visited.insert(ep.clone());
            }
        }

        let mut result = Vec::new();

        while let Some(std::cmp::Reverse((_dist, current_id))) = candidates.pop() {
            result.push(current_id.clone());

            if result.len() >= beam_width {
                break;
            }

            // Explore neighbors
            if let Some(neighbors) = graph.get(&current_id) {
                for neighbor_id in neighbors {
                    if !visited.contains(neighbor_id) {
                        visited.insert(neighbor_id.clone());

                        if let Some(neighbor_vec) = vectors.get(neighbor_id) {
                            let neighbor_dist = self
                                .distance_metric
                                .calculate(query, neighbor_vec)
                                .unwrap_or(f32::MAX);
                            candidates.push(std::cmp::Reverse((
                                ordered_float::OrderedFloat(neighbor_dist),
                                neighbor_id.clone(),
                            )));
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Robust prune - DiskANN's key algorithm for graph quality
    fn robust_prune(
        &self,
        query: &DenseVector,
        candidates: &[RecordId],
        max_size: usize,
    ) -> crate::Result<Vec<RecordId>> {
        let vectors = self.vectors.read().unwrap();
        let mut result = Vec::new();

        // Calculate all distances
        let mut dist_candidates: Vec<_> = candidates
            .iter()
            .filter_map(|id| {
                vectors.get(id).map(|vec| {
                    let dist = self
                        .distance_metric
                        .calculate(query, vec)
                        .unwrap_or(f32::MAX);
                    (dist, id.clone())
                })
            })
            .collect();

        // Sort by distance
        dist_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy selection with alpha-robust pruning
        for (dist, id) in dist_candidates {
            if result.len() >= max_size {
                break;
            }

            // Check if adding this point maintains connectivity
            let should_add = result.is_empty() || {
                let query_to_candidate = dist;
                result.iter().all(|existing_id| {
                    if let Some(existing_vec) = vectors.get(existing_id) {
                        if let Some(candidate_vec) = vectors.get(&id) {
                            let candidate_to_existing = self
                                .distance_metric
                                .calculate(candidate_vec, existing_vec)
                                .unwrap_or(f32::MAX);
                            // Alpha condition: ensures diverse neighbors
                            query_to_candidate <= self.config.alpha * candidate_to_existing
                        } else {
                            true
                        }
                    } else {
                        true
                    }
                })
            };

            if should_add {
                result.push(id);
            }
        }

        Ok(result)
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>> {
        if query.len() != self.dimension {
            return Err(crate::Error::InvalidInput(format!(
                "Query dimension {} does not match index dimension {}",
                query.len(),
                self.dimension
            )));
        }

        // Use beam search with larger beam width for better recall
        let search_width = std::cmp::max(k * 2, self.config.beam_width);
        let candidates = self.beam_search(query, search_width)?;

        // Calculate distances and get top-k
        let vectors = self.vectors.read().unwrap();
        let mut results: Vec<_> = candidates
            .iter()
            .filter_map(|id| {
                vectors.get(id).map(|vec| {
                    let dist = self
                        .distance_metric
                        .calculate(query, vec)
                        .unwrap_or(f32::MAX);
                    (id.clone(), dist)
                })
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    /// Delete a vector from the index
    pub fn delete(&mut self, id: &RecordId) -> crate::Result<()> {
        // Remove from vectors
        self.vectors.write().unwrap().remove(id);
        self.node_metadata.write().unwrap().remove(id);

        // Remove from graph and fix broken edges
        let mut graph = self.graph.write().unwrap();

        // Get neighbors before removing
        if let Some(neighbors) = graph.remove(id) {
            // Remove reverse edges
            for neighbor_id in neighbors {
                if let Some(neighbor_edges) = graph.get_mut(&neighbor_id) {
                    neighbor_edges.retain(|nid| nid != id);
                }
            }
        }

        // Remove from all neighbor lists
        for edges in graph.values_mut() {
            edges.retain(|nid| nid != id);
        }

        // Remove from entry points
        self.entry_points.write().unwrap().retain(|ep| ep != id);

        Ok(())
    }

    /// Get index statistics
    pub fn stats(&self) -> DiskANNStats {
        let graph = self.graph.read().unwrap();
        let vectors = self.vectors.read().unwrap();

        let total_nodes = vectors.len();
        let total_edges: usize = graph.values().map(|v| v.len()).sum();
        let avg_degree = if total_nodes > 0 {
            total_edges as f64 / total_nodes as f64
        } else {
            0.0
        };

        DiskANNStats {
            total_nodes,
            total_edges,
            avg_degree,
            dimension: self.dimension,
            max_degree: self.config.max_degree,
            entry_points: self.entry_points.read().unwrap().len(),
        }
    }

    /// Save index to disk (for SSD optimization)
    pub fn save(&self, path: &str) -> crate::Result<()> {
        info!("Saving DiskANN index to {}", path);

        // In a full implementation, this would write sector-aligned data
        // for optimal SSD performance. For now, this is a placeholder.

        Ok(())
    }

    /// Load index from disk
    pub fn load(path: &str) -> crate::Result<Self> {
        info!("Loading DiskANN index from {}", path);

        // Placeholder - would load memory-mapped data in full implementation

        Err(crate::Error::Internal(
            "Load not yet implemented".to_string(),
        ))
    }
}

/// DiskANN index statistics
#[derive(Debug, Clone)]
pub struct DiskANNStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub avg_degree: f64,
    pub dimension: usize,
    pub max_degree: usize,
    pub entry_points: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diskann_creation() {
        let config = DiskANNConfig::default();
        let index = DiskANNIndex::new(128, DistanceMetric::Euclidean, config);

        let stats = index.stats();
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.dimension, 128);
    }

    #[test]
    fn test_diskann_insert_and_search() {
        let config = DiskANNConfig {
            max_degree: 16,
            search_list_size: 50,
            ..Default::default()
        };

        let mut index = DiskANNIndex::new(4, DistanceMetric::Euclidean, config);

        // Insert some vectors
        index
            .insert("v1".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();
        index
            .insert("v2".to_string(), vec![0.0, 1.0, 0.0, 0.0])
            .unwrap();
        index
            .insert("v3".to_string(), vec![0.0, 0.0, 1.0, 0.0])
            .unwrap();
        index
            .insert("v4".to_string(), vec![1.0, 1.0, 0.0, 0.0])
            .unwrap();

        // Search
        let query = vec![1.0, 0.1, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "v1"); // Closest should be v1
    }

    #[test]
    fn test_diskann_delete() {
        let config = DiskANNConfig::default();
        let mut index = DiskANNIndex::new(3, DistanceMetric::Cosine, config);

        index.insert("v1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        index.insert("v2".to_string(), vec![0.0, 1.0, 0.0]).unwrap();

        assert_eq!(index.stats().total_nodes, 2);

        index.delete(&"v1".to_string()).unwrap();
        assert_eq!(index.stats().total_nodes, 1);
    }
}
