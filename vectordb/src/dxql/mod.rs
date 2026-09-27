//! There is one executor: [`executor_impl::StorageAwareExecutor`], which is what
//! `POST /query` uses.
//!
//! There used to be two. `dxql::executor::Executor` was a storage-free shell that returned an
//! empty result for every plan type by design, and `DxqlEngine` wired the parser, compiler and
//! optimizer to *it* rather than to the real one. Nothing constructed `DxqlEngine`, which is
//! the only reason it never shipped an empty answer to anybody -- but it was the obvious name
//! to reach for, and reaching for it would have produced a 200 with no records and no error.
//! Both were removed on 2026-09-27.

pub mod compiler;
pub mod executor_impl;
pub mod optimizer;
pub mod parser;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::RecordId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub records: Vec<ResultRecord>,
    pub metadata: QueryMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultRecord {
    pub id: RecordId,
    pub score: Option<f32>,
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    pub total_results: usize,
    /// Rows written by INSERT, UPDATE or DELETE, counted from the rows the statement matched
    /// rather than from what it asked for. Zero for SELECT, CREATE and DROP.
    pub affected_rows: usize,
    pub execution_time_ms: u64,
    pub plan_type: String,
}

#[derive(Debug, thiserror::Error)]
pub enum DxqlError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Compilation error: {0}")]
    CompilationError(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Type error: {0}")]
    TypeError(String),

    #[error("Unknown table: {0}")]
    UnknownTable(String),

    #[error("Unknown field: {0}")]
    UnknownField(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Valid DXQL that this implementation does not execute. Answered with 501, so a caller can
    /// tell "you wrote it wrong" (400) from "this is not done here" and is told what to use.
    #[error("Not supported: {0}")]
    Unsupported(String),
}
