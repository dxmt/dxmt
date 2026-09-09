//! DXMT VectorDB - A lightweight, horizontally scalable vector database
//! with object storage native architecture.

pub mod api;
#[cfg(feature = "benchmarks")]
pub mod benchmarks;
pub mod config;
pub mod error;
pub mod search;
pub mod storage;
pub mod types;
pub mod utils;

// Experimental modules are opt-in. Keeping them out of the default dependency graph is part of
// the supported single-node Alpha contract.
#[cfg(feature = "embeddings")]
pub mod embeddings;
#[cfg(feature = "multimodal")]
pub mod multimodal;

// Hardware acceleration (GPU, SIMD, parallel)
#[cfg(feature = "embeddings")]
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

// Adaptive indexing

// Index Management (Phase 3)

// Core modules
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
