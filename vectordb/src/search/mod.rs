pub mod index_persistence;
pub mod index_traits;
// Core search modules
pub mod auto_tuned;
pub mod batch_operations;
pub mod binary_hnsw; // NEW: Binary HNSW with RaBitQ quantization
pub mod bm25;
pub mod brute_force;
pub mod cluster_aware_search; // NEW: Cluster-aware search strategy
pub mod diskann;
pub mod diskann_enhanced;
pub mod fastscan;
pub mod full_text;
pub mod hierarchical_clustering; // NEW: Hierarchical clustering for large-scale search
pub mod hnsw;
pub mod hnsw_pq;
pub mod hybrid_rabitq_clustering; // NEW: Hybrid RaBitQ + Clustering (Phase 4)
pub mod hybrid_search;
pub mod index;
pub mod index_wrapper; // NEW: Interior mutability wrapper for Arc<dyn VectorIndex>
pub mod ivf;
pub mod ivf_pq;
pub mod lsh;
pub mod quantization;
pub mod scann;
pub mod simd_distance;
pub mod sparse_index;
pub mod trigram_index;

// Chinese tokenizer (Jieba)
#[cfg(feature = "text-bm25")]
pub mod jieba_tokenizer;

// Module exports

pub use binary_hnsw::{BinaryHnswConfig, BinaryHnswIndex}; // NEW: Binary HNSW
pub use brute_force::BruteForceIndex;
pub use cluster_aware_search::{
    ClusterAwareSearchConfig, ClusterAwareSearchIndex, SearchStats, SearchStrategy,
}; // NEW: Cluster-aware search
pub use diskann::DiskANNIndex;
pub use hierarchical_clustering::{
    AdaptiveClusterAnalyzer, AdaptiveClusterConfig, AdaptiveClusterResult, ClusteringStats,
    HierarchicalClusteringConfig, HierarchicalClusteringIndex,
}; // NEW: Hierarchical clustering with adaptive sizing
pub use hnsw::HnswIndex;
pub use hybrid_rabitq_clustering::{
    BinaryCode, HybridIndexConfig, HybridIndexStats, HybridRaBitQClusteringIndex,
}; // NEW: Hybrid RaBitQ + Clustering (Phase 4)

pub use crate::types::record::SearchResult;
pub use index_traits::{IndexStats, SerializableIndex, VectorIndex};
pub use index_wrapper::{IndexCollection, IndexWrapper}; // NEW: Interior mutability wrappers
pub use ivf_pq::IvfPqIndex;
pub use lsh::LSHIndex;

pub use simd_distance::SimdDistance;
