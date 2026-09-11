//! DXQL AST to plan.
//!
//! What cannot be answered is refused here, by name, with `DxqlError::Unsupported` rather than
//! dropped. This compiler used to turn every OR, IN, BETWEEN and function condition into "no
//! filter", so a query asking for two cities would have returned the whole table.

use crate::dxql::parser::{
    ComparisonOp, CreateStatement, DataType, DeleteStatement, DropStatement, DropType,
    InsertStatement, OrderBy, OrderDirection, SelectField, SelectStatement, Statement,
    UpdateStatement, Value, WhereClause,
};
use crate::dxql::DxqlError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Compiler {
    _private: (),
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn compile(&self, ast: Statement) -> Result<QueryPlan, DxqlError> {
        match ast {
            Statement::Select(select) => compile_select(select),
            Statement::Insert(insert) => compile_insert(insert),
            Statement::Update(update) => compile_update(update),
            Statement::Delete(delete) => compile_delete(delete),
            Statement::Create(create) => compile_create(create),
            Statement::Drop(drop) => compile_drop(drop),
        }
    }
}

fn compile_select(select: SelectStatement) -> Result<QueryPlan, DxqlError> {
    let projections = select
        .fields
        .into_iter()
        .filter_map(|field| match field {
            SelectField::All => None,
            SelectField::Field { name, alias } => Some(Projection { field: name, alias }),
        })
        .collect();
    let filter = select.where_clause.map(compile_where).transpose()?;
    let order_by = select
        .order_by
        .into_iter()
        .map(|item| match item {
            OrderBy::Field { name, direction } => Ok(OrderKey {
                field: name,
                descending: direction == OrderDirection::Desc,
            }),
            OrderBy::HybridScore { .. } => Err(DxqlError::Unsupported(
                "ORDER BY hybrid_score(...): DXQL rows carry no scores to combine; \
                 use POST /tables/{name}/search/hybrid"
                    .to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(QueryPlan::Select(SelectPlan {
        table: select.from,
        projections,
        filter,
        order_by,
        limit: select.limit,
        offset: select.offset,
    }))
}

fn compile_where(clause: WhereClause) -> Result<FilterExpr, DxqlError> {
    Ok(match clause {
        WhereClause::And(terms) => FilterExpr::And(
            terms
                .into_iter()
                .map(compile_where)
                .collect::<Result<_, _>>()?,
        ),
        WhereClause::Or(terms) => FilterExpr::Or(
            terms
                .into_iter()
                .map(compile_where)
                .collect::<Result<_, _>>()?,
        ),
        WhereClause::Comparison { field, op, value } => FilterExpr::Compare(Filter {
            field,
            operation: operator(&op).to_string(),
            value: json(value)?,
        }),
        WhereClause::In { field, values } => FilterExpr::In {
            field,
            values: values.into_iter().map(json).collect::<Result<_, _>>()?,
        },
        WhereClause::Between { field, low, high } => FilterExpr::Between {
            field,
            low: json(low)?,
            high: json(high)?,
        },
        WhereClause::VectorSimilarity { .. } => {
            return Err(DxqlError::Unsupported(
                "vector_similarity(...) in WHERE: use POST /tables/{name}/search for vector search"
                    .to_string(),
            ))
        }
        WhereClause::TextMatch { .. } => return Err(DxqlError::Unsupported(
            "text_match(...) in WHERE: use POST /tables/{name}/search/text for full-text search"
                .to_string(),
        )),
        WhereClause::GraphDistance { .. } => {
            return Err(DxqlError::Unsupported(
                "graph_distance(...) in WHERE: there is no graph index".to_string(),
            ))
        }
    })
}

fn operator(op: &ComparisonOp) -> &'static str {
    match op {
        ComparisonOp::Equal => "=",
        ComparisonOp::NotEqual => "!=",
        ComparisonOp::Less => "<",
        ComparisonOp::LessEqual => "<=",
        ComparisonOp::Greater => ">",
        ComparisonOp::GreaterEqual => ">=",
        ComparisonOp::Like => "LIKE",
    }
}

/// A literal as the JSON a payload holds. Parameters are refused: `POST /query` takes a string
/// and nothing to bind them to.
fn json(value: Value) -> Result<serde_json::Value, DxqlError> {
    Ok(match value {
        Value::Null => serde_json::Value::Null,
        Value::Boolean(b) => serde_json::Value::Bool(b),
        Value::Integer(i) => serde_json::Value::from(i),
        Value::Float(f) => serde_json::Value::from(f),
        Value::Text(s) => serde_json::Value::String(s),
        Value::Vector(v) => serde_json::json!(v),
        Value::Parameter(p) => {
            return Err(DxqlError::Unsupported(format!(
                "query parameter {p}: POST /query has no parameter bindings; write the value inline"
            )))
        }
    })
}

/// `id` and `vector` are the record's own; every other column becomes a payload field.
fn compile_insert(insert: InsertStatement) -> Result<QueryPlan, DxqlError> {
    let mut row = InsertRow {
        id: None,
        vector: Vec::new(),
        payload: serde_json::Map::new(),
    };
    let mut has_vector = false;
    for (field, value) in insert.fields.into_iter().zip(insert.values) {
        let repeated = match field.as_str() {
            "id" => match value {
                Value::Text(id) => row.id.replace(id).is_some(),
                _ => {
                    return Err(DxqlError::CompilationError(
                        "id must be a string literal, e.g. 'doc-1'".to_string(),
                    ))
                }
            },
            "vector" => match value {
                Value::Vector(vector) => {
                    row.vector = vector;
                    std::mem::replace(&mut has_vector, true)
                }
                _ => {
                    return Err(DxqlError::CompilationError(
                        "vector must be a list of numbers, e.g. [0.1, 0.2]".to_string(),
                    ))
                }
            },
            _ => row.payload.insert(field.clone(), json(value)?).is_some(),
        };
        if repeated {
            return Err(DxqlError::CompilationError(format!(
                "column '{field}' is given twice"
            )));
        }
    }
    if !has_vector {
        return Err(DxqlError::CompilationError(
            "INSERT needs a vector column".to_string(),
        ));
    }
    Ok(QueryPlan::Insert(InsertPlan {
        table: insert.table,
        rows: vec![row],
    }))
}

fn compile_update(update: UpdateStatement) -> Result<QueryPlan, DxqlError> {
    let mut set_payload = serde_json::Map::new();
    let mut set_vector = None;
    for (field, value) in update.assignments {
        match field.as_str() {
            "id" => {
                return Err(DxqlError::CompilationError(
                    "id cannot be changed by UPDATE".to_string(),
                ))
            }
            "vector" => match value {
                Value::Vector(vector) => set_vector = Some(vector),
                _ => {
                    return Err(DxqlError::CompilationError(
                        "vector must be a list of numbers, e.g. [0.1, 0.2]".to_string(),
                    ))
                }
            },
            _ => {
                set_payload.insert(field.clone(), json(value)?);
            }
        }
    }
    Ok(QueryPlan::Update(UpdatePlan {
        table: update.table,
        set_payload,
        set_vector,
        filter: update.where_clause.map(compile_where).transpose()?,
    }))
}

fn compile_delete(delete: DeleteStatement) -> Result<QueryPlan, DxqlError> {
    Ok(QueryPlan::Delete(DeletePlan {
        table: delete.table,
        filter: delete.where_clause.map(compile_where).transpose()?,
    }))
}

/// The table's dimension is its one `VECTOR(n)` column. The other columns are accepted and
/// describe nothing the table enforces -- payloads are schemaless -- and the metric is cosine,
/// which is what `POST /tables` defaults to as well. This used to create every table with 128
/// dimensions whatever the statement said.
fn compile_create(create: CreateStatement) -> Result<QueryPlan, DxqlError> {
    match create {
        CreateStatement::Table { name, columns } => {
            // The grammar accepts any backtick-quoted identifier; a table name has the rules
            // `POST /tables` applies, or DXQL creates tables the REST API cannot address.
            crate::utils::validate_table_name(&name)
                .map_err(|e| DxqlError::CompilationError(e.to_string()))?;
            let dimensions: Vec<usize> = columns
                .iter()
                .filter_map(|column| match column.data_type {
                    DataType::Vector(n) => Some(n),
                    _ => None,
                })
                .collect();
            let dimension = match dimensions.as_slice() {
                [] => {
                    return Err(DxqlError::CompilationError(format!(
                        "CREATE TABLE {name} needs a VECTOR(n) column"
                    )))
                }
                [0] => {
                    return Err(DxqlError::CompilationError(
                        "VECTOR(0) is not a dimension".to_string(),
                    ))
                }
                [n] => *n,
                _ => {
                    return Err(DxqlError::Unsupported(
                        "more than one VECTOR column: create named vectors with POST /tables"
                            .to_string(),
                    ))
                }
            };
            Ok(QueryPlan::Create(CreatePlan::Table { name, dimension }))
        }
        CreateStatement::Index { .. } => Err(DxqlError::Unsupported(
            "CREATE INDEX: each table chooses its own index; rebuild it with \
             POST /tables/{name}/index/rebuild"
                .to_string(),
        )),
    }
}

fn compile_drop(drop: DropStatement) -> Result<QueryPlan, DxqlError> {
    match drop.object_type {
        DropType::Table => Ok(QueryPlan::Drop(DropPlan { table: drop.name })),
        DropType::Index => Err(DxqlError::Unsupported(
            "DROP INDEX: use DELETE /tables/{name}/index".to_string(),
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryPlan {
    Select(SelectPlan),
    Insert(InsertPlan),
    Update(UpdatePlan),
    Delete(DeletePlan),
    Create(CreatePlan),
    Drop(DropPlan),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectPlan {
    pub table: String,
    /// Empty means every field: the id and the whole payload.
    pub projections: Vec<Projection>,
    pub filter: Option<FilterExpr>,
    /// Applied before `offset` and `limit`. Ties, and the whole order when this is empty, fall
    /// back to the record id.
    pub order_by: Vec<OrderKey>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    pub field: String,
    pub alias: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderKey {
    pub field: String,
    pub descending: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterExpr {
    And(Vec<FilterExpr>),
    Or(Vec<FilterExpr>),
    Compare(Filter),
    In {
        field: String,
        values: Vec<serde_json::Value>,
    },
    /// Inclusive at both ends.
    Between {
        field: String,
        low: serde_json::Value,
        high: serde_json::Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    pub field: String,
    /// One of `=`, `!=`, `<`, `<=`, `>`, `>=`, `LIKE`.
    pub operation: String,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InsertPlan {
    pub table: String,
    pub rows: Vec<InsertRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InsertRow {
    /// Generated when the statement names none.
    pub id: Option<String>,
    pub vector: Vec<f32>,
    pub payload: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdatePlan {
    pub table: String,
    /// Merged into each matching record's payload; fields not named keep their values.
    pub set_payload: serde_json::Map<String, serde_json::Value>,
    pub set_vector: Option<Vec<f32>>,
    pub filter: Option<FilterExpr>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeletePlan {
    pub table: String,
    /// None deletes every record, as `DELETE FROM t` does in SQL.
    pub filter: Option<FilterExpr>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CreatePlan {
    Table { name: String, dimension: usize },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DropPlan {
    pub table: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dxql::parser::Parser;
    use serde_json::json;

    fn compile(sql: &str) -> Result<QueryPlan, DxqlError> {
        Compiler::new().compile(Parser::new().parse(sql)?)
    }

    #[test]
    fn or_in_and_between_become_filters_rather_than_disappearing() {
        let Ok(QueryPlan::Select(plan)) =
            compile("SELECT * FROM t WHERE city IN ('a') OR rank BETWEEN 1 AND 2")
        else {
            panic!("did not compile to a SELECT");
        };
        assert_eq!(
            plan.filter,
            Some(FilterExpr::Or(vec![
                FilterExpr::In {
                    field: "city".into(),
                    values: vec![json!("a")]
                },
                FilterExpr::Between {
                    field: "rank".into(),
                    low: json!(1),
                    high: json!(2)
                },
            ]))
        );
    }

    #[test]
    fn insert_puts_every_other_column_in_the_payload() {
        let Ok(QueryPlan::Insert(plan)) =
            compile("INSERT INTO t (city, id, vector, rank) VALUES ('Rome', 'n1', [1, 0], 9)")
        else {
            panic!("did not compile to an INSERT");
        };
        let payload = json!({"city": "Rome", "rank": 9});
        assert_eq!(
            plan.rows,
            vec![InsertRow {
                id: Some("n1".into()),
                vector: vec![1.0, 0.0],
                payload: payload.as_object().cloned().unwrap_or_default(),
            }]
        );
    }

    #[test]
    fn create_table_takes_its_dimension_from_the_vector_column() {
        assert!(matches!(
            compile("CREATE TABLE d (id TEXT, e VECTOR(4))"),
            Ok(QueryPlan::Create(CreatePlan::Table { dimension: 4, .. }))
        ));
        assert!(matches!(
            compile("CREATE TABLE d (id TEXT)"),
            Err(DxqlError::CompilationError(_))
        ));
    }

    #[test]
    fn what_dxql_cannot_answer_is_refused_by_name() {
        for sql in [
            "SELECT * FROM t WHERE vector_similarity(v, [1, 0]) > 0.5",
            "SELECT * FROM t WHERE text_match(body, 'x')",
            "SELECT * FROM t ORDER BY hybrid_score(0.5, 0.5, 0)",
            "SELECT * FROM t WHERE a = $1",
            "CREATE INDEX i ON t(v)",
            "DROP INDEX i",
        ] {
            assert!(
                matches!(compile(sql), Err(DxqlError::Unsupported(_))),
                "{sql}"
            );
        }
        for sql in [
            "UPDATE t SET id = 'x'",
            "INSERT INTO t (id) VALUES ('x')",
            "INSERT INTO t (id, vector, vector) VALUES ('x', [1], [2])",
        ] {
            assert!(
                matches!(compile(sql), Err(DxqlError::CompilationError(_))),
                "{sql}"
            );
        }
    }
}
