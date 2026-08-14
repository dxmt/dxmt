// Core storage module - simplified and clean

pub mod atomic_ops;
pub mod background_demotion; // New: Background demotion task
pub mod backup;
pub mod compression;
pub mod consistency;
pub mod consistency_check; // New: Index consistency verification
pub mod encryption;
pub mod factory;
pub mod hybrid_engine;
pub mod index_selector;
pub mod intelligent_indexed_storage;
pub mod lmdb;
pub mod memory;
pub mod mmap_index;
pub mod object;
pub mod payload_index;
pub mod tier_promotion;
pub mod tiered;
pub mod tombstone_tracker;
pub mod traits;
pub mod wal_entry; // New: WAL entry definitions
                   // Removed

// Re-export main types
pub use background_demotion::{
    start_background_demotion_task, BackgroundDemotionConfig, BackgroundDemotionTask,
};
pub use backup::{BackupConfig, BackupManager, RestoreOptions};
pub use factory::{
    OptimizedCostConstraints, OptimizedStorageConfig, PerformanceTargets, StorageBackend,
    StorageFactory, StorageIntegration,
};
pub use hybrid_engine::{HybridConfig, HybridStats, HybridStorageEngine, ReadPreference};
pub use intelligent_indexed_storage::IntelligentIndexedStorage;
pub use lmdb::{GraphEdge, GraphNode, LmdbStorage, TableMetadata, VectorRecord};
pub use memory::MemoryStorage;
pub use mmap_index::{IndexMetadata, MmapHnswIndex, MmapIndexStorage};
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
