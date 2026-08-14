// Core storage module - simplified and clean

pub mod atomic_ops;
pub mod backup;
pub mod background_demotion;  // New: Background demotion task
pub mod compression;
pub mod consistency;
pub mod consistency_check;  // New: Index consistency verification
pub mod encryption;
pub mod factory;
pub mod wal_entry;  // New: WAL entry definitions
pub mod hybrid_engine;
pub mod index_selector;
pub mod intelligent_indexed_storage;
pub mod lmdb;
pub mod memory;
pub mod mmap_index;
pub mod object;
pub mod payload_index;
pub mod tiered;
pub mod tier_promotion;
pub mod tombstone_tracker;
pub mod traits;
 // Removed

// Re-export main types
pub use backup::{BackupConfig, BackupManager, RestoreOptions};
pub use background_demotion::{
    BackgroundDemotionConfig, BackgroundDemotionTask,
    start_background_demotion_task
};
pub use factory::{
    StorageFactory, StorageBackend, OptimizedStorageConfig,
    PerformanceTargets, OptimizedCostConstraints, StorageIntegration
};
pub use hybrid_engine::{HybridStorageEngine, HybridConfig, ReadPreference, HybridStats};
pub use intelligent_indexed_storage::IntelligentIndexedStorage;
pub use lmdb::{LmdbStorage, VectorRecord, GraphNode, GraphEdge, TableMetadata};
pub use memory::MemoryStorage;
pub use mmap_index::{MmapIndexStorage, MmapHnswIndex, IndexMetadata};
pub use tier_promotion::{
    TierPromotionPolicy, TierPromotionConfig, AccessPattern,
    Tier, TierStats
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
    IndexAlgorithm, IndexConfig, PerformanceRequirements,
    CostConstraints, HardwareCapabilities, IndexDecision,
    ExpectedPerformance, IntelligentIndexSelector
};
