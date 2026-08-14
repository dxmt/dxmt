//! DXMT VectorDB - A lightweight, horizontally scalable vector database
//! with object storage native architecture.

pub mod api;
pub mod benchmarks;
pub mod config;
pub mod error;
pub mod search;
pub mod storage;
pub mod types;
pub mod utils;

// Stub modules (minimal implementation for compilation)
pub mod cluster;
pub mod embeddings;
pub mod multimodal;

// Hardware acceleration (GPU, SIMD, parallel)
pub mod acceleration;

// MCP Protocol Support (P0 Priority)
pub mod mcp;

// VectorQL - SQL-like query language

// DXQL - DXMT Query Language (SQL-like)
pub mod dxql;

// Python bindings
#[cfg(feature = "python")]
pub mod python;

// Schema version management

// Optimized worker pool
pub mod worker_pool;

// Adaptive indexing
pub mod adaptive_index;

// Index Management (Phase 3)
pub mod index_management;

// Core modules
pub mod async_wal; // Async Write-Ahead Log for non-blocking writes
pub mod consensus; // Raft consensus implementation
pub mod memory_pool; // Memory pool for vector allocation optimization
pub mod multi_tenant;
pub mod precomputed_query; // Precomputed query optimization for faster searches
pub mod wal; // Write-Ahead Log with batch flush optimization // Multi-tenancy: usage limits, quotas, federation

// Commented out - module file missing
// #[cfg(test)]
// mod debug_serialization;

// Re-export commonly used types
pub use error::{Error, Result};

pub use types::{
    record::{CreateRecordRequest, Record, SearchQuery, SearchResult},
    table::{Table, TableConfig},
    vector::{DenseVector, DistanceMetric, RecordId},
};
