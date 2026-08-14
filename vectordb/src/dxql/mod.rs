pub mod parser;
pub mod compiler;
pub mod executor;
pub mod executor_impl;
pub mod optimizer;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::RecordId;

#[derive(Debug, Clone)]
pub struct DxqlEngine {
    parser: parser::Parser,
    compiler: compiler::Compiler,
    optimizer: optimizer::Optimizer,
    executor: executor::Executor,
}

impl DxqlEngine {
    pub fn new() -> Self {
        Self {
            parser: parser::Parser::new(),
            compiler: compiler::Compiler::new(),
            optimizer: optimizer::Optimizer::new(),
            executor: executor::Executor::new(),
        }
    }

    pub async fn execute(&self, query: &str) -> Result<QueryResult, DxqlError> {
        let ast = self.parser.parse(query)?;

        let plan = self.compiler.compile(ast)?;

        let optimized_plan = self.optimizer.optimize(plan)?;

        let result = self.executor.execute(optimized_plan).await?;

        Ok(result)
    }
}

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
}