//! VectorQL - SQL-like query language for vector operations
//! 
//! Provides familiar SQL syntax for vector database operations,
//! making it easier for developers to adopt and integrate.

pub mod parser;
pub mod compiler;
pub mod optimizer;
pub mod executor;
pub mod ast;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// VectorQL query processor
pub struct VectorQL {
    parser: parser::Parser,
    compiler: compiler::Compiler,
    optimizer: optimizer::Optimizer,
    executor: executor::Executor,
}

impl VectorQL {
    pub fn new() -> Self {
        Self {
            parser: parser::Parser::new(),
            compiler: compiler::Compiler::new(),
            optimizer: optimizer::Optimizer::new(),
            executor: executor::Executor::new(),
        }
    }
    
    /// Parse and execute a VectorQL query
    pub async fn execute(&self, query: &str) -> Result<QueryResult, VectorQLError> {
        // Parse query into AST
        let ast = self.parser.parse(query)?;
        
        // Compile AST into query plan
        let plan = self.compiler.compile(ast)?;
        
        // Optimize query plan
        let optimized_plan = self.optimizer.optimize(plan)?;
        
        // Execute query plan
        let result = self.executor.execute(optimized_plan).await?;
        
        Ok(result)
    }
    
    /// Explain query execution plan
    pub fn explain(&self, query: &str) -> Result<ExplainResult, VectorQLError> {
        let ast = self.parser.parse(query)?;
        let ast_debug = format!("{:#?}", ast);
        let plan = self.compiler.compile(ast)?;
        let plan_debug = format!("{:#?}", plan);
        let optimized_plan = self.optimizer.optimize(plan)?;

        Ok(ExplainResult {
            original_query: query.to_string(),
            ast: ast_debug,
            logical_plan: plan_debug,
            physical_plan: format!("{:#?}", optimized_plan),
            estimated_cost: self.optimizer.estimate_cost(&optimized_plan),
        })
    }
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub rows: Vec<Row>,
    pub metadata: QueryMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub id: String,
    pub vector: Option<Vec<f32>>,
    pub score: Option<f32>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    pub total_rows: usize,
    pub execution_time_ms: u64,
    pub index_used: Option<String>,
    pub cache_hit: bool,
}

/// Explain result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainResult {
    pub original_query: String,
    pub ast: String,
    pub logical_plan: String,
    pub physical_plan: String,
    pub estimated_cost: f64,
}

/// VectorQL errors
#[derive(Debug, thiserror::Error)]
pub enum VectorQLError {
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Compilation error: {0}")]
    CompilationError(String),
    
    #[error("Optimization error: {0}")]
    OptimizationError(String),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
}

/// Example VectorQL queries:
/// 
/// ```sql
/// -- Basic similarity search
/// SELECT * FROM products 
/// WHERE SIMILARITY(embedding, [0.1, 0.2, ...]) > 0.8
/// ORDER BY SIMILARITY_SCORE DESC
/// LIMIT 10;
/// 
/// -- Hybrid search with filters
/// SELECT id, name, category, SIMILARITY_SCORE() as score
/// FROM documents
/// WHERE SIMILARITY(vector, EMBED('machine learning')) > 0.7
///   AND category IN ('AI', 'ML', 'Data Science')
///   AND created_date > '2024-01-01'
/// ORDER BY score DESC
/// LIMIT 20;
/// 
/// -- KNN search
/// SELECT KNN(embedding, [0.1, 0.2, ...], k=5) as neighbors
/// FROM vectors
/// WHERE status = 'active';
/// 
/// -- Aggregations
/// SELECT category, COUNT(*) as count, AVG_VECTOR(embedding) as centroid
/// FROM products
/// GROUP BY category
/// HAVING count > 10;
/// 
/// -- Join operations
/// SELECT p.*, r.rating
/// FROM products p
/// INNER JOIN reviews r ON SIMILARITY(p.embedding, r.embedding) > 0.9
/// WHERE r.rating >= 4;
/// 
/// -- Insert with embedding
/// INSERT INTO documents (id, text, embedding)
/// VALUES ('doc1', 'AI is transforming industries', EMBED('AI is transforming industries'));
/// 
/// -- Update vectors
/// UPDATE products 
/// SET embedding = EMBED(description)
/// WHERE embedding IS NULL;
/// 
/// -- Create index
/// CREATE INDEX idx_products_embedding ON products 
/// USING HNSW(embedding) 
/// WITH (m=16, ef_construction=200);
/// 
/// -- Materialized view
/// CREATE MATERIALIZED VIEW product_categories AS
/// SELECT category, AVG_VECTOR(embedding) as avg_embedding, COUNT(*) as count
/// FROM products
/// GROUP BY category;
/// ```
pub mod prelude {
    pub use super::{
        VectorQL, QueryResult, Row, QueryMetadata, 
        ExplainResult, VectorQLError
    };
}