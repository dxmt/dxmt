//! DXQL text to AST, per `grammar.pest`.
//!
//! Until 2026-09-11 this parsed nothing. `parse` looked for `statement` among the top-level
//! pairs, where only `query` ever is, so every input -- `SELECT * FROM t` included -- failed with
//! "No valid statement found"; and past `parse_select`, every function returned a placeholder
//! with an empty table name. `POST /query` answered 400 to everything, and the only test that
//! touched it counted the 400 as the route being served.

use pest::iterators::Pair;
use pest::Parser as PestParser;
use pest_derive::Parser;
use serde::{Deserialize, Serialize};

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
        let query = DxqlParser::parse(Rule::query, input)
            .map_err(|e| DxqlError::ParseError(e.to_string()))?
            .next()
            .ok_or_else(|| DxqlError::ParseError("empty query".to_string()))?;
        // `query` is the one top-level pair; the statement is inside it.
        let statement = query
            .into_inner()
            .find(|pair| pair.as_rule() == Rule::statement)
            .ok_or_else(|| DxqlError::ParseError("No valid statement found".to_string()))?;
        parse_statement(statement)
    }
}

type ParseResult<T> = Result<T, DxqlError>;

fn parse_statement(pair: Pair<'_, Rule>) -> ParseResult<Statement> {
    let inner = first_inner(pair, "statement")?;
    match inner.as_rule() {
        Rule::select_statement => parse_select(inner),
        Rule::insert_statement => parse_insert(inner),
        Rule::update_statement => parse_update(inner),
        Rule::delete_statement => parse_delete(inner),
        Rule::create_statement => parse_create(inner),
        Rule::drop_statement => parse_drop(inner),
        other => Err(unexpected(other, "a statement")),
    }
}

fn first_inner<'i>(pair: Pair<'i, Rule>, context: &str) -> ParseResult<Pair<'i, Rule>> {
    pair.into_inner()
        .next()
        .ok_or_else(|| DxqlError::ParseError(format!("empty {context}")))
}

fn missing(what: &str) -> DxqlError {
    DxqlError::ParseError(format!("missing {what}"))
}

fn unexpected(rule: Rule, context: &str) -> DxqlError {
    DxqlError::ParseError(format!("unexpected {rule:?} in {context}"))
}

/// A table, field, alias or index name, without the backticks of a quoted identifier.
fn name(pair: &Pair<'_, Rule>) -> String {
    let text = pair.as_str();
    text.strip_prefix('`')
        .and_then(|t| t.strip_suffix('`'))
        .unwrap_or(text)
        .to_string()
}

fn parse_select(pair: Pair<'_, Rule>) -> ParseResult<Statement> {
    let mut select = SelectStatement {
        fields: Vec::new(),
        from: String::new(),
        where_clause: None,
        order_by: Vec::new(),
        limit: None,
        offset: None,
    };
    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::select_list => select.fields = parse_select_list(part),
            Rule::table_name => select.from = name(&part),
            Rule::where_clause => select.where_clause = Some(parse_where(part)?),
            Rule::order_by_clause => select.order_by = parse_order_by(part)?,
            Rule::limit_clause => {
                let (limit, offset) = parse_limit(part)?;
                select.limit = Some(limit);
                select.offset = offset;
            }
            other => return Err(unexpected(other, "SELECT")),
        }
    }
    Ok(Statement::Select(select))
}

fn parse_select_list(pair: Pair<'_, Rule>) -> Vec<SelectField> {
    pair.into_inner()
        .map(|item| match item.as_rule() {
            Rule::star => SelectField::All,
            _ => {
                let mut parts = item.into_inner();
                let field = parts.next().map(|p| name(&p)).unwrap_or_default();
                let alias = parts.next().map(|p| name(&p));
                SelectField::Field { name: field, alias }
            }
        })
        .collect()
}

fn parse_where(pair: Pair<'_, Rule>) -> ParseResult<WhereClause> {
    parse_condition(first_inner(pair, "WHERE")?)
}

fn parse_condition(pair: Pair<'_, Rule>) -> ParseResult<WhereClause> {
    let mut alternatives = pair
        .into_inner()
        .map(parse_and_condition)
        .collect::<ParseResult<Vec<_>>>()?;
    Ok(if alternatives.len() == 1 {
        alternatives.remove(0)
    } else {
        WhereClause::Or(alternatives)
    })
}

fn parse_and_condition(pair: Pair<'_, Rule>) -> ParseResult<WhereClause> {
    let mut terms = pair
        .into_inner()
        .map(parse_primary)
        .collect::<ParseResult<Vec<_>>>()?;
    Ok(if terms.len() == 1 {
        terms.remove(0)
    } else {
        WhereClause::And(terms)
    })
}

fn parse_primary(pair: Pair<'_, Rule>) -> ParseResult<WhereClause> {
    let inner = first_inner(pair, "condition")?;
    let rule = inner.as_rule();
    if rule == Rule::condition {
        return parse_condition(inner);
    }
    let mut parts = inner.into_inner();
    let mut next = |what: &str| parts.next().ok_or_else(|| missing(what));

    match rule {
        Rule::comparison_condition => Ok(WhereClause::Comparison {
            field: name(&next("field")?),
            op: parse_op(&next("operator")?)?,
            value: parse_value(next("value")?)?,
        }),
        Rule::in_condition => {
            let field = name(&next("field")?);
            let values = next("value list")?
                .into_inner()
                .map(parse_value)
                .collect::<ParseResult<Vec<_>>>()?;
            Ok(WhereClause::In { field, values })
        }
        Rule::between_condition => Ok(WhereClause::Between {
            field: name(&next("field")?),
            low: parse_value(next("lower bound")?)?,
            high: parse_value(next("upper bound")?)?,
        }),
        Rule::vector_similarity_condition => {
            let field = name(&next("field")?);
            let target = next("vector")?;
            let vector = match target.as_rule() {
                Rule::parameter => Value::Parameter(target.as_str().to_string()),
                _ => Value::Vector(parse_vector(target)?),
            };
            Ok(WhereClause::VectorSimilarity {
                field,
                vector,
                op: parse_op(&next("operator")?)?,
                threshold: parse_f64(&next("threshold")?)?,
            })
        }
        Rule::text_match_condition => Ok(WhereClause::TextMatch {
            field: name(&next("field")?),
            query: unquote(next("text")?.as_str()),
        }),
        Rule::graph_distance_condition => Ok(WhereClause::GraphDistance {
            field: name(&next("field")?),
            target: unquote(next("target")?.as_str()),
            op: parse_op(&next("operator")?)?,
            max_distance: parse_usize(&next("distance")?)?,
        }),
        other => Err(unexpected(other, "a condition")),
    }
}

fn parse_op(pair: &Pair<'_, Rule>) -> ParseResult<ComparisonOp> {
    Ok(match pair.as_str().to_ascii_uppercase().as_str() {
        "=" => ComparisonOp::Equal,
        "!=" | "<>" => ComparisonOp::NotEqual,
        "<" => ComparisonOp::Less,
        "<=" => ComparisonOp::LessEqual,
        ">" => ComparisonOp::Greater,
        ">=" => ComparisonOp::GreaterEqual,
        "LIKE" => ComparisonOp::Like,
        other => return Err(DxqlError::ParseError(format!("unknown operator {other}"))),
    })
}

fn parse_value(pair: Pair<'_, Rule>) -> ParseResult<Value> {
    let inner = first_inner(pair, "value")?;
    Ok(match inner.as_rule() {
        Rule::now_func => Value::Now,
        Rule::vector_value => Value::Vector(parse_vector(inner)?),
        Rule::string_literal => Value::Text(unquote(inner.as_str())),
        Rule::number => parse_number(inner.as_str())?,
        Rule::boolean => Value::Boolean(inner.as_str().eq_ignore_ascii_case("true")),
        Rule::null => Value::Null,
        Rule::parameter => Value::Parameter(inner.as_str().to_string()),
        other => return Err(unexpected(other, "a value")),
    })
}

fn parse_vector(pair: Pair<'_, Rule>) -> ParseResult<Vec<f32>> {
    pair.into_inner()
        .map(|n| parse_f64(&n).map(|x| x as f32))
        .collect()
}

fn parse_f64(pair: &Pair<'_, Rule>) -> ParseResult<f64> {
    pair.as_str()
        .parse()
        .map_err(|_| DxqlError::ParseError(format!("not a number: {}", pair.as_str())))
}

fn parse_usize(pair: &Pair<'_, Rule>) -> ParseResult<usize> {
    pair.as_str()
        .parse()
        .map_err(|_| DxqlError::ParseError(format!("not a count: {}", pair.as_str())))
}

/// An integer when it is written as one and fits, a float otherwise.
fn parse_number(text: &str) -> ParseResult<Value> {
    if !text.contains(['.', 'e', 'E']) {
        if let Ok(integer) = text.parse::<i64>() {
            return Ok(Value::Integer(integer));
        }
    }
    text.parse::<f64>()
        .map(Value::Float)
        .map_err(|_| DxqlError::ParseError(format!("not a number: {text}")))
}

/// `'it''s'` to `it's`, and the same for double quotes.
fn unquote(literal: &str) -> String {
    let quote = &literal[..1];
    literal[1..literal.len() - 1].replace(&quote.repeat(2), quote)
}

fn parse_order_by(pair: Pair<'_, Rule>) -> ParseResult<Vec<OrderBy>> {
    pair.into_inner()
        .map(|item| {
            let mut parts = item.into_inner();
            let first = parts.next().ok_or_else(|| missing("ORDER BY item"))?;
            if first.as_rule() == Rule::hybrid_score_func {
                let weights = first
                    .into_inner()
                    .map(|w| parse_f64(&w))
                    .collect::<ParseResult<Vec<_>>>()?;
                return match weights.as_slice() {
                    [text, vector, graph] => Ok(OrderBy::HybridScore {
                        text_weight: *text,
                        vector_weight: *vector,
                        graph_weight: *graph,
                    }),
                    _ => Err(missing("three hybrid_score weights")),
                };
            }
            let direction = match parts.next() {
                Some(d) if d.as_str().eq_ignore_ascii_case("desc") => OrderDirection::Desc,
                _ => OrderDirection::Asc,
            };
            Ok(OrderBy::Field {
                name: name(&first),
                direction,
            })
        })
        .collect()
}

fn parse_limit(pair: Pair<'_, Rule>) -> ParseResult<(usize, Option<usize>)> {
    // The first number is the limit and the second the offset. It used to test
    // `offset.is_none()` to tell them apart, which is true for both, so OFFSET overwrote LIMIT.
    let mut numbers = pair.into_inner();
    let limit = parse_usize(&numbers.next().ok_or_else(|| missing("LIMIT count"))?)?;
    let offset = numbers.next().map(|n| parse_usize(&n)).transpose()?;
    Ok((limit, offset))
}

fn parse_insert(pair: Pair<'_, Rule>) -> ParseResult<Statement> {
    let mut table = String::new();
    let mut fields = Vec::new();
    let mut values = Vec::new();
    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::table_name => table = name(&part),
            Rule::field_list => fields = part.into_inner().map(|f| name(&f)).collect(),
            Rule::value_list => {
                values = part
                    .into_inner()
                    .map(parse_value)
                    .collect::<ParseResult<Vec<_>>>()?
            }
            other => return Err(unexpected(other, "INSERT")),
        }
    }
    if fields.len() != values.len() {
        return Err(DxqlError::ParseError(format!(
            "INSERT names {} columns but gives {} values",
            fields.len(),
            values.len()
        )));
    }
    Ok(Statement::Insert(InsertStatement {
        table,
        fields,
        values,
    }))
}

fn parse_update(pair: Pair<'_, Rule>) -> ParseResult<Statement> {
    let mut update = UpdateStatement {
        table: String::new(),
        assignments: Vec::new(),
        where_clause: None,
    };
    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::table_name => update.table = name(&part),
            Rule::assignment_list => {
                for assignment in part.into_inner() {
                    let mut kv = assignment.into_inner();
                    let field = kv.next().map(|p| name(&p)).unwrap_or_default();
                    let value = parse_value(kv.next().ok_or_else(|| missing("assigned value"))?)?;
                    if update.assignments.iter().any(|(f, _)| *f == field) {
                        return Err(DxqlError::ParseError(format!(
                            "column '{field}' is assigned twice"
                        )));
                    }
                    update.assignments.push((field, value));
                }
            }
            Rule::where_clause => update.where_clause = Some(parse_where(part)?),
            other => return Err(unexpected(other, "UPDATE")),
        }
    }
    Ok(Statement::Update(update))
}

fn parse_delete(pair: Pair<'_, Rule>) -> ParseResult<Statement> {
    let mut delete = DeleteStatement {
        table: String::new(),
        where_clause: None,
    };
    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::table_name => delete.table = name(&part),
            Rule::where_clause => delete.where_clause = Some(parse_where(part)?),
            other => return Err(unexpected(other, "DELETE")),
        }
    }
    Ok(Statement::Delete(delete))
}

fn parse_create(pair: Pair<'_, Rule>) -> ParseResult<Statement> {
    let inner = first_inner(pair, "CREATE")?;
    match inner.as_rule() {
        Rule::create_table_statement => {
            let mut table = String::new();
            let mut columns = Vec::new();
            for part in inner.into_inner() {
                match part.as_rule() {
                    Rule::table_name => table = name(&part),
                    Rule::column_definition => columns.push(parse_column(part)?),
                    other => return Err(unexpected(other, "CREATE TABLE")),
                }
            }
            Ok(Statement::Create(CreateStatement::Table {
                name: table,
                columns,
            }))
        }
        Rule::create_index_statement => {
            let mut index = String::new();
            let mut table = String::new();
            let mut field = String::new();
            let mut index_type = None;
            let mut options = Vec::new();
            for part in inner.into_inner() {
                match part.as_rule() {
                    Rule::index_type => index_type = Some(parse_index_type(part.as_str())?),
                    Rule::index_name => index = name(&part),
                    Rule::table_name => table = name(&part),
                    Rule::field_name => field = name(&part),
                    Rule::index_options => {
                        for option in first_inner(part, "WITH")?.into_inner() {
                            let mut kv = option.into_inner();
                            let key = kv.next().map(|p| name(&p)).unwrap_or_default();
                            let value =
                                parse_value(kv.next().ok_or_else(|| missing("option value"))?)?;
                            options.push((key, value));
                        }
                    }
                    other => return Err(unexpected(other, "CREATE INDEX")),
                }
            }
            Ok(Statement::Create(CreateStatement::Index {
                name: index,
                table,
                field,
                index_type,
                options,
            }))
        }
        other => Err(unexpected(other, "CREATE")),
    }
}

fn parse_column(pair: Pair<'_, Rule>) -> ParseResult<ColumnDef> {
    let mut parts = pair.into_inner();
    let column = parts.next().map(|p| name(&p)).unwrap_or_default();
    let data_type = parse_data_type(parts.next().ok_or_else(|| missing("column type"))?)?;
    let constraints = parts
        .map(parse_constraint)
        .collect::<ParseResult<Vec<_>>>()?;
    Ok(ColumnDef {
        name: column,
        data_type,
        constraints,
    })
}

fn parse_data_type(pair: Pair<'_, Rule>) -> ParseResult<DataType> {
    let inner = first_inner(pair, "column type")?;
    Ok(match inner.as_rule() {
        Rule::vector_type => DataType::Vector(parse_usize(&first_inner(inner, "VECTOR")?)?),
        Rule::text_type => DataType::Text,
        Rule::integer_type => DataType::Integer,
        Rule::float_type => DataType::Float,
        Rule::boolean_type => DataType::Boolean,
        Rule::json_type => DataType::Json,
        Rule::timestamp_type => DataType::Timestamp,
        other => return Err(unexpected(other, "a column type")),
    })
}

fn parse_constraint(pair: Pair<'_, Rule>) -> ParseResult<ColumnConstraint> {
    let inner = first_inner(pair, "column constraint")?;
    Ok(match inner.as_rule() {
        Rule::primary_key => ColumnConstraint::PrimaryKey,
        Rule::not_null => ColumnConstraint::NotNull,
        Rule::unique => ColumnConstraint::Unique,
        Rule::default_value => {
            ColumnConstraint::Default(parse_value(first_inner(inner, "DEFAULT")?)?)
        }
        other => return Err(unexpected(other, "a column constraint")),
    })
}

fn parse_index_type(text: &str) -> ParseResult<IndexType> {
    Ok(match text.to_ascii_uppercase().as_str() {
        "HNSW" => IndexType::Hnsw,
        "IVF" => IndexType::Ivf,
        "FLAT" => IndexType::Flat,
        "BM25" => IndexType::Bm25,
        "GRAPH" => IndexType::Graph,
        other => return Err(DxqlError::ParseError(format!("unknown index type {other}"))),
    })
}

fn parse_drop(pair: Pair<'_, Rule>) -> ParseResult<Statement> {
    let mut parts = pair.into_inner();
    let kind = parts.next().ok_or_else(|| missing("TABLE or INDEX"))?;
    let object_type = if kind.as_str().eq_ignore_ascii_case("index") {
        DropType::Index
    } else {
        DropType::Table
    };
    let target = parts
        .next()
        .map(|p| name(&p))
        .ok_or_else(|| missing("name to drop"))?;
    Ok(Statement::Drop(DropStatement {
        object_type,
        name: target,
    }))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    Select(SelectStatement),
    Insert(InsertStatement),
    Update(UpdateStatement),
    Delete(DeleteStatement),
    Create(CreateStatement),
    Drop(DropStatement),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectStatement {
    pub fields: Vec<SelectField>,
    pub from: String,
    pub where_clause: Option<WhereClause>,
    pub order_by: Vec<OrderBy>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectField {
    All,
    Field { name: String, alias: Option<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WhereClause {
    And(Vec<WhereClause>),
    Or(Vec<WhereClause>),
    VectorSimilarity {
        field: String,
        vector: Value,
        op: ComparisonOp,
        threshold: f64,
    },
    TextMatch {
        field: String,
        query: String,
    },
    GraphDistance {
        field: String,
        target: String,
        op: ComparisonOp,
        max_distance: usize,
    },
    Comparison {
        field: String,
        op: ComparisonOp,
        value: Value,
    },
    In {
        field: String,
        values: Vec<Value>,
    },
    Between {
        field: String,
        low: Value,
        high: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Like,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderBy {
    Field {
        name: String,
        direction: OrderDirection,
    },
    HybridScore {
        text_weight: f64,
        vector_weight: f64,
        graph_weight: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InsertStatement {
    pub table: String,
    pub fields: Vec<String>,
    pub values: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateStatement {
    pub table: String,
    /// In the order written. Assigning the same column twice is a parse error.
    pub assignments: Vec<(String, Value)>,
    pub where_clause: Option<WhereClause>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeleteStatement {
    pub table: String,
    pub where_clause: Option<WhereClause>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CreateStatement {
    Table {
        name: String,
        columns: Vec<ColumnDef>,
    },
    Index {
        name: String,
        table: String,
        field: String,
        index_type: Option<IndexType>,
        options: Vec<(String, Value)>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub constraints: Vec<ColumnConstraint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Vector(usize),
    Text,
    Integer,
    Float,
    Boolean,
    Json,
    Timestamp,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnConstraint {
    PrimaryKey,
    NotNull,
    Unique,
    Default(Value),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexType {
    Hnsw,
    Ivf,
    Flat,
    Bm25,
    Graph,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DropStatement {
    pub object_type: DropType,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DropType {
    Table,
    Index,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    Text(String),
    Vector(Vec<f32>),
    Parameter(String),
    /// `NOW()`, unresolved. The parser records what was written and the compiler decides what
    /// it means, so the clock stays out of parsing and every `NOW()` in one statement resolves
    /// to the same instant rather than to whenever each was converted.
    Now,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(sql: &str) -> Statement {
        Parser::new()
            .parse(sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    fn select(sql: &str) -> SelectStatement {
        match parse(sql) {
            Statement::Select(select) => select,
            other => panic!("not a SELECT: {other:?}"),
        }
    }

    fn cmp(field: &str, op: ComparisonOp, value: Value) -> WhereClause {
        WhereClause::Comparison {
            field: field.to_string(),
            op,
            value,
        }
    }

    /// `NOW()` is the one function DXQL evaluates, and both places it is ordinarily written
    /// used to be parse errors: `SET updated_at = NOW()` and `DEFAULT NOW()`. Both statements
    /// were otherwise fully supported -- UPDATE compiles to a real plan and CREATE TABLE makes
    /// a real table -- so the only thing between them and working was the `value` rule having
    /// no alternative for a function call.
    #[test]
    fn now_is_a_value_wherever_a_value_goes() {
        let Statement::Update(update) =
            parse("UPDATE products SET price = 249.99, updated_at = NOW() WHERE id = 'prod_456'")
        else {
            panic!("not an UPDATE");
        };
        assert_eq!(
            update.assignments,
            vec![
                ("price".to_string(), Value::Float(249.99)),
                ("updated_at".to_string(), Value::Now),
            ],
            "the parser records NOW(), unresolved; the compiler decides what it means"
        );

        // In a column default, where nothing reads the value -- CREATE TABLE only looks for the
        // VECTOR(n) column -- so this only has to parse.
        let Statement::Create(_) = parse(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, embedding VECTOR(384) NOT NULL, \
             created_at TIMESTAMP DEFAULT NOW())",
        ) else {
            panic!("not a CREATE");
        };

        // And in a comparison, which comes free and is the third place SQL puts it.
        let Statement::Select(select) = parse("SELECT * FROM logs WHERE created_at < NOW()") else {
            panic!("not a SELECT");
        };
        assert_eq!(
            select.where_clause,
            Some(cmp("created_at", ComparisonOp::Less, Value::Now))
        );
    }

    /// Only `NOW()`. Anything else that looks like a function stays a parse error rather than
    /// becoming a silently-ignored field name.
    #[test]
    fn other_functions_are_still_refused_in_a_value() {
        assert!(Parser::new()
            .parse("UPDATE products SET name = UPPER('x') WHERE id = 'a'")
            .is_err());
    }

    fn text(s: &str) -> Value {
        Value::Text(s.to_string())
    }

    #[test]
    fn a_plain_select_parses() {
        let s = select("SELECT * FROM t");
        assert_eq!(s.fields, vec![SelectField::All]);
        assert_eq!(s.from, "t");
        assert_eq!(s.where_clause, None);
    }

    #[test]
    fn projections_aliases_limit_and_offset() {
        let s = select("SELECT id, rank AS r FROM t LIMIT 3 OFFSET 6");
        assert_eq!(
            s.fields,
            vec![
                SelectField::Field {
                    name: "id".into(),
                    alias: None
                },
                SelectField::Field {
                    name: "rank".into(),
                    alias: Some("r".into())
                },
            ]
        );
        assert_eq!((s.limit, s.offset), (Some(3), Some(6)));
    }

    #[test]
    fn and_binds_tighter_than_or() {
        let s = select("SELECT * FROM t WHERE a = 1 OR b = 2 AND c = 3");
        assert_eq!(
            s.where_clause,
            Some(WhereClause::Or(vec![
                cmp("a", ComparisonOp::Equal, Value::Integer(1)),
                WhereClause::And(vec![
                    cmp("b", ComparisonOp::Equal, Value::Integer(2)),
                    cmp("c", ComparisonOp::Equal, Value::Integer(3)),
                ]),
            ]))
        );
    }

    #[test]
    fn floats_and_two_character_operators_parse() {
        for (symbol, op) in [
            (">=", ComparisonOp::GreaterEqual),
            ("<=", ComparisonOp::LessEqual),
            ("!=", ComparisonOp::NotEqual),
            ("<>", ComparisonOp::NotEqual),
        ] {
            let s = select(&format!("SELECT * FROM t WHERE x {symbol} -2.5"));
            assert_eq!(s.where_clause, Some(cmp("x", op, Value::Float(-2.5))));
        }
    }

    #[test]
    fn in_between_like_and_parentheses() {
        let s = select(
            "SELECT * FROM t WHERE (city IN ('a', 'b') OR name LIKE 'x%') AND rank BETWEEN 2 AND 4",
        );
        assert_eq!(
            s.where_clause,
            Some(WhereClause::And(vec![
                WhereClause::Or(vec![
                    WhereClause::In {
                        field: "city".into(),
                        values: vec![text("a"), text("b")],
                    },
                    cmp("name", ComparisonOp::Like, text("x%")),
                ]),
                WhereClause::Between {
                    field: "rank".into(),
                    low: Value::Integer(2),
                    high: Value::Integer(4),
                },
            ]))
        );
    }

    #[test]
    fn order_by_is_not_mistaken_for_or() {
        let s = select("SELECT * FROM t WHERE rank = 1 ORDER BY city, rank DESC");
        assert_eq!(
            s.where_clause,
            Some(cmp("rank", ComparisonOp::Equal, Value::Integer(1)))
        );
        assert_eq!(
            s.order_by,
            vec![
                OrderBy::Field {
                    name: "city".into(),
                    direction: OrderDirection::Asc
                },
                OrderBy::Field {
                    name: "rank".into(),
                    direction: OrderDirection::Desc
                },
            ]
        );
    }

    #[test]
    fn keywords_are_case_insensitive_and_names_keep_their_case() {
        let s = select("select Id from `My Table` where City = 'London';");
        assert_eq!(s.from, "My Table");
        assert_eq!(
            s.fields,
            vec![SelectField::Field {
                name: "Id".into(),
                alias: None
            }]
        );
        assert_eq!(
            s.where_clause,
            Some(cmp("City", ComparisonOp::Equal, text("London")))
        );
    }

    #[test]
    fn string_literals_unescape_doubled_quotes() {
        let s = select("SELECT * FROM t WHERE a = 'it''s' AND b = ''");
        assert_eq!(
            s.where_clause,
            Some(WhereClause::And(vec![
                cmp("a", ComparisonOp::Equal, text("it's")),
                cmp("b", ComparisonOp::Equal, text("")),
            ]))
        );
    }

    #[test]
    fn insert_update_and_delete() {
        assert_eq!(
            parse("INSERT INTO t (id, vector, city) VALUES ('n1', [0.5, -1, 2e1], 'Rome')"),
            Statement::Insert(InsertStatement {
                table: "t".into(),
                fields: vec!["id".into(), "vector".into(), "city".into()],
                values: vec![
                    text("n1"),
                    Value::Vector(vec![0.5, -1.0, 20.0]),
                    text("Rome")
                ],
            })
        );
        assert_eq!(
            parse("UPDATE t SET city = 'Oslo', rank = 2 WHERE id = 'r2'"),
            Statement::Update(UpdateStatement {
                table: "t".into(),
                assignments: vec![
                    ("city".into(), text("Oslo")),
                    ("rank".into(), Value::Integer(2))
                ],
                where_clause: Some(cmp("id", ComparisonOp::Equal, text("r2"))),
            })
        );
        assert_eq!(
            parse("DELETE FROM t"),
            Statement::Delete(DeleteStatement {
                table: "t".into(),
                where_clause: None
            })
        );
    }

    #[test]
    fn create_and_drop() {
        assert_eq!(
            parse("CREATE TABLE docs (id TEXT PRIMARY KEY, embedding VECTOR(384) NOT NULL)"),
            Statement::Create(CreateStatement::Table {
                name: "docs".into(),
                columns: vec![
                    ColumnDef {
                        name: "id".into(),
                        data_type: DataType::Text,
                        constraints: vec![ColumnConstraint::PrimaryKey],
                    },
                    ColumnDef {
                        name: "embedding".into(),
                        data_type: DataType::Vector(384),
                        constraints: vec![ColumnConstraint::NotNull],
                    },
                ],
            })
        );
        assert_eq!(
            parse("CREATE HNSW INDEX idx ON docs(embedding) WITH (m = 16)"),
            Statement::Create(CreateStatement::Index {
                name: "idx".into(),
                table: "docs".into(),
                field: "embedding".into(),
                index_type: Some(IndexType::Hnsw),
                options: vec![("m".into(), Value::Integer(16))],
            })
        );
        assert_eq!(
            parse("DROP TABLE docs"),
            Statement::Drop(DropStatement {
                object_type: DropType::Table,
                name: "docs".into()
            })
        );
        assert_eq!(
            parse("drop index idx"),
            Statement::Drop(DropStatement {
                object_type: DropType::Index,
                name: "idx".into()
            })
        );
    }

    #[test]
    fn malformed_input_is_a_parse_error() {
        for sql in [
            "",
            "SELEC * FROM t",
            "SELECT * FROM",
            "INSERT INTO t (id, vector) VALUES ('a')",
            "UPDATE t SET a = 1, a = 2",
            "SELECT * FROM t WHERE x = 3.",
        ] {
            assert!(
                matches!(Parser::new().parse(sql), Err(DxqlError::ParseError(_))),
                "{sql:?} should not parse"
            );
        }
    }
}
