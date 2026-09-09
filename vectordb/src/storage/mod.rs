// Core storage module - simplified and clean

pub mod atomic_ops;
pub mod compression;
pub mod consistency_check; // New: Index consistency verification
pub mod factory;
pub mod index_selector;
pub mod intelligent_indexed_storage;
#[cfg(feature = "storage-lmdb")]
pub mod lmdb;
pub mod memory;
pub(crate) mod memory_guard;
pub mod object;
pub mod payload_index;
pub mod record_store;
pub mod tier_promotion;
pub mod tiered;
pub mod traits;
pub mod vector_file;
pub mod wal_entry; // New: WAL entry definitions
                   // Removed

// Re-export main types
pub use factory::{
    OptimizedCostConstraints, OptimizedStorageConfig, PerformanceTargets, StorageBackend,
    StorageFactory, StorageIntegration,
};
pub use intelligent_indexed_storage::IntelligentIndexedStorage;
#[cfg(feature = "storage-lmdb")]
pub use lmdb::{GraphEdge, GraphNode, LmdbStorage, TableMetadata, VectorRecord};
pub use memory::MemoryStorage;
pub use tier_promotion::{
    AccessPattern, Tier, TierPromotionConfig, TierPromotionPolicy, TierStats,
};
pub use tiered::TieredStorage;
pub use traits::{_Storage as Storage, StorageConfig, StorageType};

// Re-export types from types module
pub use crate::types::{
    record::{CreateRecordRequest, Record, SearchQuery, SearchResult},
    table::{Table, TableConfig},
};

// Re-export index selector types (moved from intelligent_indexed_storage)
pub use index_selector::{
    CostConstraints, ExpectedPerformance, HardwareCapabilities, IndexAlgorithm, IndexConfig,
    IndexDecision, IntelligentIndexSelector, PerformanceRequirements,
};
