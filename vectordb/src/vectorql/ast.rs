//! Abstract Syntax Tree for VectorQL

use serde::{Deserialize, Serialize};

/// Root AST node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Statement {
    Select(SelectStatement),
    Insert(InsertStatement),
    Update(UpdateStatement),
    Delete(DeleteStatement),
    CreateIndex(CreateIndexStatement),
    CreateView(CreateViewStatement),
    Explain(Box<Statement>),
}

/// SELECT statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectStatement {
    pub projection: Vec<SelectItem>,
    pub from: TableRef,
    pub where_clause: Option<Expression>,
    pub group_by: Vec<Expression>,
    pub having: Option<Expression>,
    pub order_by: Vec<OrderByItem>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectItem {
    Wildcard,
    Expression { expr: Expression, alias: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableRef {
    pub name: String,
    pub alias: Option<String>,
    pub join: Option<Box<JoinClause>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinClause {
    pub join_type: JoinType,
    pub table: TableRef,
    pub on: Expression,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
    Similarity, // Special join type for vector similarity
}

/// Expression nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    // Literals
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Vector(Vec<f32>),
    
    // Identifiers
    Column(String),
    QualifiedColumn { table: String, column: String },
    
    // Binary operations
    BinaryOp {
        left: Box<Expression>,
        op: BinaryOperator,
        right: Box<Expression>,
    },
    
    // Unary operations
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expression>,
    },
    
    // Vector functions
    Similarity {
        vector_col: Box<Expression>,
        query_vector: Box<Expression>,
    },
    KNN {
        vector_col: Box<Expression>,
        query_vector: Box<Expression>,
        k: usize,
    },
    Embed {
        text: Box<Expression>,
        model: Option<String>,
    },
    
    // Aggregate functions
    Aggregate {
        func: AggregateFunction,
        expr: Box<Expression>,
        distinct: bool,
    },
    
    // Special functions
    Function {
        name: String,
        args: Vec<Expression>,
    },
    
    // Subquery
    Subquery(Box<SelectStatement>),
    
    // CASE expression
    Case {
        expr: Option<Box<Expression>>,
        when_clauses: Vec<WhenClause>,
        else_clause: Option<Box<Expression>>,
    },
    
    // IN expression
    In {
        expr: Box<Expression>,
        list: Vec<Expression>,
        negated: bool,
    },
    
    // BETWEEN expression
    Between {
        expr: Box<Expression>,
        low: Box<Expression>,
        high: Box<Expression>,
        negated: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhenClause {
    pub condition: Expression,
    pub result: Expression,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinaryOperator {
    // Comparison
    Eq, Ne, Lt, Le, Gt, Ge,
    
    // Logical
    And, Or,
    
    // Arithmetic
    Plus, Minus, Multiply, Divide, Modulo,
    
    // String
    Like, NotLike,
    
    // Vector
    DotProduct, CosineSimilarity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Minus,
    Plus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    
    // Vector aggregates
    AvgVector,
    Centroid,
    Variance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderByItem {
    pub expr: Expression,
    pub asc: bool,
    pub nulls_first: Option<bool>,
}

/// INSERT statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertStatement {
    pub table: String,
    pub columns: Vec<String>,
    pub values: Vec<Vec<Expression>>,
    pub on_conflict: Option<ConflictClause>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictClause {
    pub columns: Vec<String>,
    pub action: ConflictAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictAction {
    DoNothing,
    Update(Vec<Assignment>),
}

/// UPDATE statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStatement {
    pub table: String,
    pub assignments: Vec<Assignment>,
    pub where_clause: Option<Expression>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    pub column: String,
    pub value: Expression,
}

/// DELETE statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteStatement {
    pub table: String,
    pub where_clause: Option<Expression>,
}

/// CREATE INDEX statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateIndexStatement {
    pub name: String,
    pub table: String,
    pub column: String,
    pub index_type: IndexType,
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    HNSW,
    IVF,
    IVFPQ,
    DiskANN,
    ScaNN,
    Flat,
}

/// CREATE MATERIALIZED VIEW statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateViewStatement {
    pub name: String,
    pub query: SelectStatement,
    pub materialized: bool,
    pub refresh_policy: Option<RefreshPolicy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefreshPolicy {
    OnCommit,
    OnDemand,
    Interval { seconds: u64 },
}

use std::collections::HashMap;