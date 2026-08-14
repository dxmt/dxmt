use pest::Parser as PestParser;
use pest_derive::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::dxql::DxqlError;

#[derive(Parser)]
#[grammar = "dxql/grammar.pest"]
pub struct DxqlParser;

#[derive(Debug, Clone)]
pub struct Parser {
    _private: (),
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn parse(&self, input: &str) -> Result<Statement, DxqlError> {
        let pairs = DxqlParser::parse(Rule::query, input)
            .map_err(|e| DxqlError::ParseError(e.to_string()))?;

        for pair in pairs {
            if pair.as_rule() == Rule::statement {
                return self.parse_statement(pair);
            }
        }

        Err(DxqlError::ParseError("No valid statement found".to_string()))
    }

    fn parse_statement(&self, pair: pest::iterators::Pair<Rule>) -> Result<Statement, DxqlError> {
        let inner = pair.into_inner().next()
            .ok_or_else(|| DxqlError::ParseError("Empty statement".to_string()))?;

        match inner.as_rule() {
            Rule::select_statement => self.parse_select(inner),
            Rule::insert_statement => self.parse_insert(inner),
            Rule::update_statement => self.parse_update(inner),
            Rule::delete_statement => self.parse_delete(inner),
            Rule::create_statement => self.parse_create(inner),
            Rule::drop_statement => self.parse_drop(inner),
            _ => Err(DxqlError::ParseError(format!("Unknown statement type: {:?}", inner.as_rule()))),
        }
    }

    fn parse_select(&self, pair: pest::iterators::Pair<Rule>) -> Result<Statement, DxqlError> {
        let mut select = SelectStatement {
            fields: vec![],
            from: String::new(),
            where_clause: None,
            order_by: None,
            limit: None,
            offset: None,
        };

        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::select_list => {
                    select.fields = self.parse_select_list(pair)?;
                }
                Rule::table_name => {
                    select.from = pair.as_str().to_string();
                }
                Rule::where_clause => {
                    select.where_clause = Some(self.parse_where_clause(pair)?);
                }
                Rule::order_by_clause => {
                    select.order_by = Some(self.parse_order_by(pair)?);
                }
                Rule::limit_clause => {
                    let (limit, offset) = self.parse_limit(pair)?;
                    select.limit = Some(limit);
                    select.offset = offset;
                }
                _ => {}
            }
        }

        Ok(Statement::Select(select))
    }

    fn parse_select_list(&self, pair: pest::iterators::Pair<Rule>) -> Result<Vec<SelectField>, DxqlError> {
        let mut fields = vec![];

        for pair in pair.into_inner() {
            if pair.as_str() == "*" {
                fields.push(SelectField::All);
            } else if pair.as_rule() == Rule::select_item {
                let mut field_name = String::new();
                let mut alias = None;

                for inner in pair.into_inner() {
                    match inner.as_rule() {
                        Rule::field_name => field_name = inner.as_str().to_string(),
                        Rule::alias => alias = Some(inner.as_str().to_string()),
                        _ => {}
                    }
                }

                fields.push(SelectField::Field {
                    name: field_name,
                    alias,
                });
            }
        }

        Ok(fields)
    }

    fn parse_where_clause(&self, pair: pest::iterators::Pair<Rule>) -> Result<WhereClause, DxqlError> {
        for pair in pair.into_inner() {
            if pair.as_rule() == Rule::condition {
                return self.parse_condition(pair);
            }
        }

        Err(DxqlError::ParseError("Invalid WHERE clause".to_string()))
    }

    fn parse_condition(&self, pair: pest::iterators::Pair<Rule>) -> Result<WhereClause, DxqlError> {
        for pair in pair.into_inner() {
            if pair.as_rule() == Rule::and_condition {
                return self.parse_and_condition(pair);
            }
        }

        Err(DxqlError::ParseError("Invalid condition".to_string()))
    }

    fn parse_and_condition(&self, pair: pest::iterators::Pair<Rule>) -> Result<WhereClause, DxqlError> {
        let mut conditions = vec![];

        for pair in pair.into_inner() {
            if pair.as_rule() == Rule::or_condition {
                conditions.push(self.parse_or_condition(pair)?);
            }
        }

        if conditions.len() == 1 {
            Ok(conditions.into_iter().next().unwrap())
        } else {
            Ok(WhereClause::And(conditions))
        }
    }

    fn parse_or_condition(&self, pair: pest::iterators::Pair<Rule>) -> Result<WhereClause, DxqlError> {
        let mut conditions = vec![];

        for pair in pair.into_inner() {
            if pair.as_rule() == Rule::primary_condition {
                conditions.push(self.parse_primary_condition(pair)?);
            }
        }

        if conditions.len() == 1 {
            Ok(conditions.into_iter().next().unwrap())
        } else {
            Ok(WhereClause::Or(conditions))
        }
    }

    fn parse_primary_condition(&self, pair: pest::iterators::Pair<Rule>) -> Result<WhereClause, DxqlError> {
        let inner = pair.into_inner().next()
            .ok_or_else(|| DxqlError::ParseError("Empty primary condition".to_string()))?;

        match inner.as_rule() {
            Rule::vector_similarity_condition => {
                Ok(WhereClause::VectorSimilarity {
                    field: "embedding".to_string(),
                    vector: vec![],
                    threshold: 0.8,
                })
            }
            Rule::text_match_condition => {
                Ok(WhereClause::TextMatch {
                    field: "text".to_string(),
                    query: String::new(),
                })
            }
            Rule::comparison_condition => {
                Ok(WhereClause::Comparison {
                    field: String::new(),
                    op: ComparisonOp::Equal,
                    value: Value::Null,
                })
            }
            Rule::condition => self.parse_condition(inner),
            _ => Err(DxqlError::ParseError(format!("Unknown condition type: {:?}", inner.as_rule()))),
        }
    }

    fn parse_order_by(&self, _pair: pest::iterators::Pair<Rule>) -> Result<OrderBy, DxqlError> {
        Ok(OrderBy::Field {
            name: "score".to_string(),
            direction: OrderDirection::Desc,
        })
    }

    fn parse_limit(&self, pair: pest::iterators::Pair<Rule>) -> Result<(usize, Option<usize>), DxqlError> {
        let mut limit = 10;
        let mut offset = None;

        for pair in pair.into_inner() {
            if pair.as_rule() == Rule::integer {
                if offset.is_none() {
                    limit = pair.as_str().parse()
                        .map_err(|_| DxqlError::ParseError("Invalid limit".to_string()))?;
                } else {
                    offset = Some(pair.as_str().parse()
                        .map_err(|_| DxqlError::ParseError("Invalid offset".to_string()))?);
                }
            }
        }

        Ok((limit, offset))
    }

    fn parse_insert(&self, _pair: pest::iterators::Pair<Rule>) -> Result<Statement, DxqlError> {
        Ok(Statement::Insert(InsertStatement {
            table: String::new(),
            fields: vec![],
            values: vec![],
        }))
    }

    fn parse_update(&self, _pair: pest::iterators::Pair<Rule>) -> Result<Statement, DxqlError> {
        Ok(Statement::Update(UpdateStatement {
            table: String::new(),
            assignments: HashMap::new(),
            where_clause: None,
        }))
    }

    fn parse_delete(&self, _pair: pest::iterators::Pair<Rule>) -> Result<Statement, DxqlError> {
        Ok(Statement::Delete(DeleteStatement {
            table: String::new(),
            where_clause: None,
        }))
    }

    fn parse_create(&self, _pair: pest::iterators::Pair<Rule>) -> Result<Statement, DxqlError> {
        Ok(Statement::Create(CreateStatement::Table {
            name: String::new(),
            columns: vec![],
        }))
    }

    fn parse_drop(&self, _pair: pest::iterators::Pair<Rule>) -> Result<Statement, DxqlError> {
        Ok(Statement::Drop(DropStatement {
            object_type: DropType::Table,
            name: String::new(),
        }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Statement {
    Select(SelectStatement),
    Insert(InsertStatement),
    Update(UpdateStatement),
    Delete(DeleteStatement),
    Create(CreateStatement),
    Drop(DropStatement),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectStatement {
    pub fields: Vec<SelectField>,
    pub from: String,
    pub where_clause: Option<WhereClause>,
    pub order_by: Option<OrderBy>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectField {
    All,
    Field { name: String, alias: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhereClause {
    And(Vec<WhereClause>),
    Or(Vec<WhereClause>),
    VectorSimilarity {
        field: String,
        vector: Vec<f32>,
        threshold: f32,
    },
    TextMatch {
        field: String,
        query: String,
    },
    GraphDistance {
        field: String,
        target: String,
        max_distance: usize,
    },
    Comparison {
        field: String,
        op: ComparisonOp,
        value: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Like,
    In,
    Between,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderBy {
    Field {
        name: String,
        direction: OrderDirection,
    },
    HybridScore {
        text_weight: f32,
        vector_weight: f32,
        graph_weight: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertStatement {
    pub table: String,
    pub fields: Vec<String>,
    pub values: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStatement {
    pub table: String,
    pub assignments: HashMap<String, Value>,
    pub where_clause: Option<WhereClause>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteStatement {
    pub table: String,
    pub where_clause: Option<WhereClause>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreateStatement {
    Table {
        name: String,
        columns: Vec<ColumnDef>,
    },
    Index {
        name: String,
        table: String,
        field: String,
        index_type: IndexType,
        options: HashMap<String, Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub constraints: Vec<ColumnConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Vector(usize),
    Text,
    Integer,
    Float,
    Boolean,
    Json,
    Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnConstraint {
    PrimaryKey,
    NotNull,
    Unique,
    Default(Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    Hnsw,
    Ivf,
    Flat,
    Bm25,
    Graph,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropStatement {
    pub object_type: DropType,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DropType {
    Table,
    Index,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    Text(String),
    Vector(Vec<f32>),
    Json(serde_json::Value),
    Parameter(String),
}