use crate::dxql::{DxqlError, parser::Statement};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Compiler {
    _private: (),
}

impl Compiler {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn compile(&self, ast: Statement) -> Result<QueryPlan, DxqlError> {
        match ast {
            Statement::Select(select) => self.compile_select(select),
            Statement::Insert(insert) => self.compile_insert(insert),
            Statement::Update(update) => self.compile_update(update),
            Statement::Delete(delete) => self.compile_delete(delete),
            Statement::Create(create) => self.compile_create(create),
            Statement::Drop(drop) => self.compile_drop(drop),
        }
    }

    fn compile_select(&self, select: crate::dxql::parser::SelectStatement) -> Result<QueryPlan, DxqlError> {
        let projections = select.fields.iter().filter_map(|f| match f {
            crate::dxql::parser::SelectField::All => None,
            crate::dxql::parser::SelectField::Field { name, alias } => Some(Projection {
                field: name.clone(),
                alias: alias.clone(),
            }),
        }).collect();

        let filters = if let Some(wc) = select.where_clause {
            self.compile_where(wc)
        } else {
            vec![]
        };

        Ok(QueryPlan::Select(SelectPlan {
            table: select.from,
            projections,
            filters,
            limit: select.limit,
            offset: select.offset,
        }))
    }

    fn compile_where(&self, wc: crate::dxql::parser::WhereClause) -> Vec<Filter> {
        match wc {
            crate::dxql::parser::WhereClause::Comparison { field, op, value } => {
                let op_str = match op {
                    crate::dxql::parser::ComparisonOp::Equal => "=",
                    crate::dxql::parser::ComparisonOp::NotEqual => "!=",
                    crate::dxql::parser::ComparisonOp::Less => "<",
                    crate::dxql::parser::ComparisonOp::LessEqual => "<=",
                    crate::dxql::parser::ComparisonOp::Greater => ">",
                    crate::dxql::parser::ComparisonOp::GreaterEqual => ">=",
                    crate::dxql::parser::ComparisonOp::Like => "LIKE",
                    crate::dxql::parser::ComparisonOp::In => "IN",
                    crate::dxql::parser::ComparisonOp::Between => "BETWEEN",
                };
                let json_val = match value {
                    crate::dxql::parser::Value::Null => serde_json::Value::Null,
                    crate::dxql::parser::Value::Boolean(b) => serde_json::Value::Bool(b),
                    crate::dxql::parser::Value::Integer(i) => serde_json::json!(i),
                    crate::dxql::parser::Value::Float(f) => serde_json::json!(f),
                    crate::dxql::parser::Value::Text(s) => serde_json::Value::String(s),
                    _ => serde_json::Value::Null,
                };
                vec![Filter { field, operation: op_str.to_string(), value: json_val }]
            }
            crate::dxql::parser::WhereClause::And(clauses) => {
                clauses.into_iter().flat_map(|c| self.compile_where(c)).collect()
            }
            _ => vec![],
        }
    }

    fn compile_insert(&self, insert: crate::dxql::parser::InsertStatement) -> Result<QueryPlan, DxqlError> {
        let value_obj: serde_json::Value = {
            let mut map = serde_json::Map::new();
            for (field, val) in insert.fields.iter().zip(insert.values.iter()) {
                let json_val = match val {
                    crate::dxql::parser::Value::Null => serde_json::Value::Null,
                    crate::dxql::parser::Value::Boolean(b) => serde_json::Value::Bool(*b),
                    crate::dxql::parser::Value::Integer(i) => serde_json::json!(i),
                    crate::dxql::parser::Value::Float(f) => serde_json::json!(f),
                    crate::dxql::parser::Value::Text(s) => serde_json::Value::String(s.clone()),
                    crate::dxql::parser::Value::Vector(v) => serde_json::json!(v),
                    crate::dxql::parser::Value::Json(j) => j.clone(),
                    crate::dxql::parser::Value::Parameter(p) => serde_json::Value::String(format!("${}", p)),
                };
                map.insert(field.clone(), json_val);
            }
            serde_json::Value::Object(map)
        };
        Ok(QueryPlan::Insert(InsertPlan {
            table: insert.table,
            values: vec![value_obj],
        }))
    }

    fn compile_update(&self, update: crate::dxql::parser::UpdateStatement) -> Result<QueryPlan, DxqlError> {
        let assignments = update.assignments.into_iter().map(|(field, val)| {
            let json_val = match val {
                crate::dxql::parser::Value::Null => serde_json::Value::Null,
                crate::dxql::parser::Value::Boolean(b) => serde_json::Value::Bool(b),
                crate::dxql::parser::Value::Integer(i) => serde_json::json!(i),
                crate::dxql::parser::Value::Float(f) => serde_json::json!(f),
                crate::dxql::parser::Value::Text(s) => serde_json::Value::String(s),
                _ => serde_json::Value::Null,
            };
            Assignment { field, value: json_val }
        }).collect();
        let filters = update.where_clause.map(|wc| self.compile_where(wc)).unwrap_or_default();
        Ok(QueryPlan::Update(UpdatePlan {
            table: update.table,
            assignments,
            filters,
        }))
    }

    fn compile_delete(&self, delete: crate::dxql::parser::DeleteStatement) -> Result<QueryPlan, DxqlError> {
        let filters = delete.where_clause.map(|wc| self.compile_where(wc)).unwrap_or_default();
        Ok(QueryPlan::Delete(DeletePlan {
            table: delete.table,
            filters,
        }))
    }

    fn compile_create(&self, create: crate::dxql::parser::CreateStatement) -> Result<QueryPlan, DxqlError> {
        match create {
            crate::dxql::parser::CreateStatement::Table { name, .. } => {
                Ok(QueryPlan::Create(CreatePlan::Table { name }))
            }
            crate::dxql::parser::CreateStatement::Index { name, .. } => {
                Ok(QueryPlan::Create(CreatePlan::Index { name }))
            }
        }
    }

    fn compile_drop(&self, drop: crate::dxql::parser::DropStatement) -> Result<QueryPlan, DxqlError> {
        let object_type = match drop.object_type {
            crate::dxql::parser::DropType::Table => "TABLE",
            crate::dxql::parser::DropType::Index => "INDEX",
        };
        Ok(QueryPlan::Drop(DropPlan {
            object_type: object_type.to_string(),
            name: drop.name,
        }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryPlan {
    Select(SelectPlan),
    Insert(InsertPlan),
    Update(UpdatePlan),
    Delete(DeletePlan),
    Create(CreatePlan),
    Drop(DropPlan),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectPlan {
    pub table: String,
    pub projections: Vec<Projection>,
    pub filters: Vec<Filter>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projection {
    pub field: String,
    pub alias: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filter {
    pub field: String,
    pub operation: String,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertPlan {
    pub table: String,
    pub values: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatePlan {
    pub table: String,
    pub assignments: Vec<Assignment>,
    pub filters: Vec<Filter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    pub field: String,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletePlan {
    pub table: String,
    pub filters: Vec<Filter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreatePlan {
    Table { name: String },
    Index { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropPlan {
    pub object_type: String,
    pub name: String,
}