//! DXQL Query Executor
//!
//! This module provides query execution capabilities. For full storage-aware execution,
//! use `executor_impl::StorageAwareExecutor` which has access to storage backends.
//!
//! This basic executor is used for simple query execution without storage integration.
//! For production use cases, integrate with StorageAwareExecutor.

// executor_impl is a sibling module, not a submodule
pub use crate::dxql::executor_impl::StorageAwareExecutor;

use crate::dxql::{DxqlError, QueryResult, QueryMetadata, compiler::QueryPlan};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Executor {
    _private: (),
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Execute a query plan without storage backend integration
    ///
    /// This is a simplified executor that returns empty results.
    /// For actual query execution with data, use StorageAwareExecutor instead.
    pub async fn execute(&self, plan: QueryPlan) -> Result<QueryResult, DxqlError> {
        let start = Instant::now();

        let (records, plan_type) = match plan {
            QueryPlan::Select(_select) => {
                // Simple executor returns empty - use StorageAwareExecutor for real execution
                (vec![], "SELECT")
            }
            QueryPlan::Insert(_) => (vec![], "INSERT"),
            QueryPlan::Update(_) => (vec![], "UPDATE"),
            QueryPlan::Delete(_) => (vec![], "DELETE"),
            QueryPlan::Create(_) => (vec![], "CREATE"),
            QueryPlan::Drop(_) => (vec![], "DROP"),
        };

        let execution_time_ms = start.elapsed().as_millis() as u64;
        let total_results = records.len();

        Ok(QueryResult {
            records,
            metadata: QueryMetadata {
                total_results,
                execution_time_ms,
                plan_type: plan_type.to_string(),
            },
        })
    }
}