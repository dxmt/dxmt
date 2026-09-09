pub mod index_persistence;
pub mod index_traits;
pub mod partitioned_ivf;
// Core search modules
pub mod bm25;
pub mod brute_force;
#[cfg(feature = "experimental-indexes")]
pub mod diskann;
pub mod diskann_enhanced;
#[cfg(feature = "text-bm25")]
pub mod full_text;
#[cfg(feature = "adaptive-index")]
pub mod hierarchical_clustering; // NEW: Hierarchical clustering for large-scale search
pub mod hnsw;
pub mod hnsw_pq;
pub mod index;
pub mod index_wrapper; // NEW: Interior mutability wrapper for Arc<dyn VectorIndex>
#[cfg(feature = "experimental-indexes")]
pub mod ivf;
pub mod quantization;
pub mod simd_distance;
pub mod sparse_index;

// Chinese tokenizer (Jieba)
#[cfg(feature = "text-bm25")]
pub mod jieba_tokenizer;

// Module exports

pub use brute_force::BruteForceIndex;
#[cfg(feature = "experimental-indexes")]
pub use diskann::DiskANNIndex;
#[cfg(feature = "adaptive-index")]
pub use hierarchical_clustering::{
    AdaptiveClusterAnalyzer, AdaptiveClusterConfig, AdaptiveClusterResult, ClusteringStats,
    HierarchicalClusteringConfig, HierarchicalClusteringIndex,
}; // NEW: Hierarchical clustering with adaptive sizing
pub use hnsw::HnswIndex;

pub use crate::types::record::SearchResult;
pub use index_traits::{IndexStats, SerializableIndex, VectorIndex};
pub use index_wrapper::{IndexCollection, IndexWrapper}; // NEW: Interior mutability wrappers

pub use simd_distance::SimdDistance;
