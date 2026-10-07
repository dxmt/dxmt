//! DXMT VectorDB - a lightweight single-node vector database: WAL durability, HNSW-PQ
//! indexing, payload filtering and server-side embedding behind one HTTP API.

pub mod api;
#[cfg(feature = "benchmarks")]
pub mod benchmarks;
pub mod config;
pub mod error;
pub mod search;
pub mod storage;
pub mod types;
pub mod utils;

// Turning text into a vector, server-side. The rest of what used to be here -- four
// overlapping provider abstractions, an ONNX multimodal path returning placeholder vectors,
// and a GPU distance calculator whose feature did not compile -- was taken out of
// the crate on 2026-09-27. None of it was reachable and none of it was built by CI.
#[cfg(feature = "server-embedding")]
pub mod embeddings;

// MCP Protocol Support (P0 Priority)
pub mod mcp;

// VectorQL - SQL-like query language

// DXQL - DXMT Query Language (SQL-like)
pub mod dxql;

// Python bindings

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
