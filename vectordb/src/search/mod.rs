pub mod index_persistence;
pub mod index_traits;
pub mod partitioned_ivf;
// Core search modules
#[cfg(feature = "experimental-indexes")]
pub mod batch_operations;
#[cfg(feature = "experimental-indexes")]
pub mod binary_hnsw; // NEW: Binary HNSW with RaBitQ quantization
pub mod bm25;
pub mod brute_force;
#[cfg(feature = "adaptive-index")]
pub mod cluster_aware_search; // NEW: Cluster-aware search strategy
#[cfg(feature = "experimental-indexes")]
pub mod diskann;
pub mod diskann_enhanced;
#[cfg(feature = "text-bm25")]
pub mod full_text;
#[cfg(feature = "adaptive-index")]
pub mod hierarchical_clustering; // NEW: Hierarchical clustering for large-scale search
pub mod hnsw;
pub mod hnsw_pq;
#[cfg(feature = "adaptive-index")]
pub mod hybrid_rabitq_clustering; // NEW: Hybrid RaBitQ + Clustering (Phase 4)
#[cfg(feature = "text-bm25")]
pub mod hybrid_search;
pub mod index;
pub mod index_wrapper; // NEW: Interior mutability wrapper for Arc<dyn VectorIndex>
#[cfg(feature = "experimental-indexes")]
pub mod ivf;
#[cfg(feature = "experimental-indexes")]
pub mod ivf_pq;
#[cfg(feature = "experimental-indexes")]
pub mod lsh;
pub mod quantization;
#[cfg(feature = "experimental-indexes")]
pub mod scann;
pub mod simd_distance;
pub mod sparse_index;
#[cfg(feature = "experimental-indexes")]
pub mod trigram_index;

// Chinese tokenizer (Jieba)
#[cfg(feature = "text-bm25")]
pub mod jieba_tokenizer;

// Module exports

#[cfg(feature = "experimental-indexes")]
pub use binary_hnsw::{BinaryHnswConfig, BinaryHnswIndex}; // NEW: Binary HNSW
pub use brute_force::BruteForceIndex;
#[cfg(feature = "adaptive-index")]
pub use cluster_aware_search::{
    ClusterAwareSearchConfig, ClusterAwareSearchIndex, SearchStats, SearchStrategy,
}; // NEW: Cluster-aware search
#[cfg(feature = "experimental-indexes")]
pub use diskann::DiskANNIndex;
#[cfg(feature = "adaptive-index")]
pub use hierarchical_clustering::{
    AdaptiveClusterAnalyzer, AdaptiveClusterConfig, AdaptiveClusterResult, ClusteringStats,
    HierarchicalClusteringConfig, HierarchicalClusteringIndex,
}; // NEW: Hierarchical clustering with adaptive sizing
pub use hnsw::HnswIndex;
#[cfg(feature = "adaptive-index")]
pub use hybrid_rabitq_clustering::{
    BinaryCode, HybridIndexConfig, HybridIndexStats, HybridRaBitQClusteringIndex,
}; // NEW: Hybrid RaBitQ + Clustering (Phase 4)

pub use crate::types::record::SearchResult;
pub use index_traits::{IndexStats, SerializableIndex, VectorIndex};
pub use index_wrapper::{IndexCollection, IndexWrapper}; // NEW: Interior mutability wrappers
#[cfg(feature = "experimental-indexes")]
pub use ivf_pq::IvfPqIndex;
#[cfg(feature = "experimental-indexes")]
pub use lsh::LSHIndex;

pub use simd_distance::SimdDistance;
