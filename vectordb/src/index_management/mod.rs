//! Phase 3: Index Management - Safe index switching with data migration
//!
//! This module provides:
//! - IndexFactory: Create and configure index instances
//! - IndexManager: Manage active index atomically
//! - DataMigrator: Safe data copying between indexes
//! - TransactionLog: Recovery and durability

pub mod factory;
pub mod manager;
pub mod migrator;
pub mod transaction_log;
pub mod factory_impl;
pub mod errors;
pub mod monitoring;
pub mod production;
pub mod profiling;
pub mod benchmark_harness;
pub mod record_iterator;

// Tests
#[cfg(test)]
mod tests;

pub use factory::IndexFactory;
pub use manager::IndexManager;
pub use migrator::{DataMigrator, MigrationStats, MigrationProgress};
pub use transaction_log::{TransactionLog, LogEntry, LogEntryType};
pub use profiling::{OperationProfiler, OperationProfileHandle, PerformanceProfile, BenchmarkConfig, BenchmarkResult};
pub use benchmark_harness::{Phase3BenchmarkHarness, BenchmarkConfigBuilder, quick_benchmark};
pub use record_iterator::{IndexRecord, RecordIterator, SimpleRecordIterator, RecordIteratorBuilder, MigrationContext};

use crate::types::table::IndexType;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Metadata about an index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub index_type: IndexType,
    pub created_at: String,
    pub record_count: usize,
    pub memory_usage_mb: f64,
    pub last_switched: Option<String>,
}

/// Index switching result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchResult {
    pub from_type: IndexType,
    pub to_type: IndexType,
    pub records_migrated: usize,
    pub duration_ms: u128,
    pub success: bool,
}

/// Configuration for index management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManagementConfig {
    pub migration_chunk_size: usize,
    pub migration_timeout: Duration,
    pub enable_transaction_log: bool,
    pub transaction_log_path: String,
    pub enable_background_migration: bool,
    pub max_concurrent_migrations: usize,
}

impl Default for IndexManagementConfig {
    fn default() -> Self {
        Self {
            migration_chunk_size: 1000,
            migration_timeout: Duration::from_secs(300),
            enable_transaction_log: true,
            transaction_log_path: "/var/log/dxmt/index_migrations.log".to_string(),
            enable_background_migration: true,
            max_concurrent_migrations: 2,
        }
    }
}

// Removed duplicate inline tests module - tests are in tests.rs
