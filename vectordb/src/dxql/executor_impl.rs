//! Executes DXQL plans against storage.
//!
//! Every statement is answered by a scan: the table's ids are listed and the filter is evaluated
//! against each record. That is O(rows) per statement, and it is the honest cost of an arbitrary
//! predicate here -- an ANN search cannot enumerate "every row where city = 'London'", and the
//! zero-vector search this used to use silently missed rows beyond `top_k`.
//!
//! Without ORDER BY, rows come back in id order, so LIMIT/OFFSET pages are stable. With it, ties
//! are broken by id the same way, and a row missing a sort key comes last in either direction.

use crate::dxql::{compiler::*, DxqlError, QueryMetadata, QueryResult, ResultRecord};
use crate::storage::traits::_Storage;
use crate::types::record::{CreateRecordRequest, Record};
use crate::types::table::TableConfig;
use crate::types::vector::{DistanceMetric, RecordId};
use serde_json::Value as Json;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Executor that can access storage backends for query execution
pub struct StorageAwareExecutor {
    storage: Arc<dyn _Storage>,
}

impl StorageAwareExecutor {
    pub fn new(storage: Arc<dyn _Storage>) -> Self {
        Self { storage }
    }

    pub async fn execute(&self, plan: QueryPlan) -> Result<QueryResult, DxqlError> {
        let start = Instant::now();

        let (records, affected_rows, plan_type) = match plan {
            QueryPlan::Select(select) => (self.execute_select(select).await?, 0, "SELECT"),
            QueryPlan::Insert(insert) => (Vec::new(), self.execute_insert(insert).await?, "INSERT"),
            QueryPlan::Update(update) => (Vec::new(), self.execute_update(update).await?, "UPDATE"),
            QueryPlan::Delete(delete) => (Vec::new(), self.execute_delete(delete).await?, "DELETE"),
            QueryPlan::Create(create) => {
                self.execute_create(create).await?;
                (Vec::new(), 0, "CREATE")
            }
            QueryPlan::Drop(drop) => {
                self.execute_drop(drop).await?;
                (Vec::new(), 0, "DROP")
            }
        };

        Ok(QueryResult {
            metadata: QueryMetadata {
                total_results: records.len(),
                affected_rows,
                execution_time_ms: start.elapsed().as_millis() as u64,
                plan_type: plan_type.to_string(),
            },
            records,
        })
    }

    async fn execute_select(&self, select: SelectPlan) -> Result<Vec<ResultRecord>, DxqlError> {
        let rows = self
            .matching_records(&select.table, select.filter.as_ref())
            .await?;
        let rows = sort_rows(rows, &select.order_by);
        Ok(rows
            .iter()
            .skip(select.offset.unwrap_or(0))
            .take(select.limit.unwrap_or(usize::MAX))
            .map(|row| project(row, &select.projections))
            .collect())
    }

    async fn execute_insert(&self, insert: InsertPlan) -> Result<usize, DxqlError> {
        self.storage
            .get_table(&insert.table)
            .await
            .map_err(|e| storage_error(&insert.table, e))?;

        let mut inserted = 0;
        for row in insert.rows {
            let request = CreateRecordRequest {
                id: row.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                vector: row.vector,
                sparse_vector: None,
                payload: Some(Json::Object(row.payload)),
                named_vectors: None,
            };
            self.storage
                .insert_record(&insert.table, request)
                .await
                .map_err(|e| storage_error(&insert.table, e))?;
            inserted += 1;
        }
        Ok(inserted)
    }

    async fn execute_update(&self, update: UpdatePlan) -> Result<usize, DxqlError> {
        let rows = self
            .matching_records(&update.table, update.filter.as_ref())
            .await?;
        let updated = rows.len();

        for row in rows {
            let mut payload = match row.payload {
                Json::Object(map) => map,
                _ => serde_json::Map::new(),
            };
            payload.extend(
                update
                    .set_payload
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone())),
            );
            let request = CreateRecordRequest {
                id: row.id.clone(),
                vector: update.set_vector.clone().unwrap_or(row.vector),
                sparse_vector: row.sparse_vector,
                payload: Some(Json::Object(payload)),
                named_vectors: row.named_vectors,
            };
            self.storage
                .update_record(&update.table, &row.id, request)
                .await
                .map_err(|e| storage_error(&update.table, e))?;
        }
        Ok(updated)
    }

    async fn execute_delete(&self, delete: DeletePlan) -> Result<usize, DxqlError> {
        let ids: Vec<RecordId> = match &delete.filter {
            None => self
                .storage
                .list_all_record_ids(&delete.table)
                .await
                .map_err(|e| storage_error(&delete.table, e))?,
            Some(filter) => self
                .matching_records(&delete.table, Some(filter))
                .await?
                .into_iter()
                .map(|row| row.id)
                .collect(),
        };
        // Counted from the rows that matched, so a DELETE that finds nothing says 0.
        let deleted = ids.len();
        if deleted > 0 {
            self.storage
                .delete_records(&delete.table, ids)
                .await
                .map_err(|e| storage_error(&delete.table, e))?;
        }
        Ok(deleted)
    }

    async fn execute_create(&self, create: CreatePlan) -> Result<(), DxqlError> {
        let CreatePlan::Table { name, dimension } = create;
        let config = TableConfig {
            name,
            dimension,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Created via DXQL".to_string()),
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None, // Use global config
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
        };
        self.storage
            .create_table(config)
            .await
            .map_err(|e| DxqlError::ExecutionError(format!("Create table failed: {e}")))?;
        Ok(())
    }

    async fn execute_drop(&self, drop: DropPlan) -> Result<(), DxqlError> {
        self.storage
            .delete_table(&drop.table)
            .await
            .map_err(|e| storage_error(&drop.table, e))
    }

    /// The records of `table` that satisfy `filter`; all of them when there is none.
    async fn matching_records(
        &self,
        table: &str,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<Record>, DxqlError> {
        let ids = self
            .storage
            .list_all_record_ids(table)
            .await
            .map_err(|e| storage_error(table, e))?;

        let mut rows = Vec::new();
        for id in ids {
            // A record deleted between the listing and the read is no longer a match.
            let Ok(record) = self.storage.get_record(table, &id).await else {
                continue;
            };
            let keep = match filter {
                None => true,
                Some(filter) => matches(&record, filter),
            };
            if keep {
                rows.push(record);
            }
        }
        Ok(rows)
    }
}

/// A missing table is `UnknownTable` (404 over HTTP); anything else storage refuses is an
/// execution error.
fn storage_error(table: &str, error: crate::Error) -> DxqlError {
    match error {
        crate::Error::TableNotFound(_) => DxqlError::UnknownTable(table.to_string()),
        other => DxqlError::ExecutionError(other.to_string()),
    }
}

fn project(record: &Record, projections: &[Projection]) -> ResultRecord {
    let mut fields = HashMap::new();
    if projections.is_empty() {
        if let Json::Object(map) = &record.payload {
            fields.extend(map.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        // After the payload, so a payload key called `id` cannot overwrite the record's id.
        fields.insert("id".to_string(), Json::String(record.id.clone()));
    } else {
        for projection in projections {
            let value = match projection.field.as_str() {
                "id" => Json::String(record.id.clone()),
                "vector" => serde_json::json!(record.vector),
                field => record.payload.get(field).cloned().unwrap_or(Json::Null),
            };
            let output = projection
                .alias
                .clone()
                .unwrap_or_else(|| projection.field.clone());
            fields.insert(output, value);
        }
    }
    ResultRecord {
        id: record.id.clone(),
        score: None,
        fields,
    }
}

fn field_value<'a>(record: &'a Record, field: &str) -> Option<Cow<'a, Json>> {
    match field {
        "id" => Some(Cow::Owned(Json::String(record.id.clone()))),
        _ => record.payload.get(field).map(Cow::Borrowed),
    }
}

/// A missing field satisfies no condition, `!=` included -- SQL's NULL semantics. A row with no
/// `city` is not an answer to "city != 'London'".
fn matches(record: &Record, filter: &FilterExpr) -> bool {
    match filter {
        FilterExpr::And(terms) => terms.iter().all(|term| matches(record, term)),
        FilterExpr::Or(terms) => terms.iter().any(|term| matches(record, term)),
        FilterExpr::Compare(f) => field_value(record, &f.field)
            .is_some_and(|value| compare(&value, &f.operation, &f.value)),
        FilterExpr::In { field, values } => field_value(record, field).is_some_and(|value| {
            values
                .iter()
                .any(|candidate| values_equal(&value, candidate))
        }),
        FilterExpr::Between { field, low, high } => {
            field_value(record, field).is_some_and(|value| {
                matches!(
                    comparable(&value, low),
                    Some(Ordering::Greater | Ordering::Equal)
                ) && matches!(
                    comparable(&value, high),
                    Some(Ordering::Less | Ordering::Equal)
                )
            })
        }
    }
}

fn compare(value: &Json, operation: &str, target: &Json) -> bool {
    match operation {
        "=" => values_equal(value, target),
        "!=" => !values_equal(value, target),
        "<" => comparable(value, target) == Some(Ordering::Less),
        "<=" => matches!(
            comparable(value, target),
            Some(Ordering::Less | Ordering::Equal)
        ),
        ">" => comparable(value, target) == Some(Ordering::Greater),
        ">=" => matches!(
            comparable(value, target),
            Some(Ordering::Greater | Ordering::Equal)
        ),
        "LIKE" => match (value.as_str(), target.as_str()) {
            (Some(text), Some(pattern)) => like(text, pattern),
            _ => false,
        },
        _ => false,
    }
}

/// JSON equality, except that numbers compare by value. `serde_json` holds `3` and `3.0` to be
/// different, which would make `rank = 3.0` miss a payload that stores `3`.
fn values_equal(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Number(x), Json::Number(y)) => match (x.as_i64(), y.as_i64()) {
            (Some(i), Some(j)) => i == j,
            _ => x.as_f64() == y.as_f64(),
        },
        _ => a == b,
    }
}

/// Numbers compare as numbers and strings lexicographically; anything else is not ordered.
fn comparable(a: &Json, b: &Json) -> Option<Ordering> {
    match (a, b) {
        (Json::Number(x), Json::Number(y)) => x.as_f64()?.partial_cmp(&y.as_f64()?),
        (Json::String(x), Json::String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// SQL LIKE: `%` is any run of characters, `_` exactly one, anything else itself; case-sensitive.
/// It used to strip the `%`s and test `contains`, so '%don' also matched "Londonderry".
fn like(text: &str, pattern: &str) -> bool {
    let text: Vec<char> = text.chars().collect();
    let pattern: Vec<char> = pattern.chars().collect();
    let (mut t, mut p) = (0, 0);
    // Where to resume after the most recent `%`: the pattern position past it, and the text
    // position it has been tried against so far.
    let mut resume: Option<(usize, usize)> = None;
    while t < text.len() {
        if p < pattern.len() && pattern[p] == '%' {
            p += 1;
            resume = Some((p, t));
        } else if p < pattern.len() && (pattern[p] == '_' || pattern[p] == text[t]) {
            p += 1;
            t += 1;
        } else if let Some((after_percent, tried)) = resume {
            p = after_percent;
            t = tried + 1;
            resume = Some((after_percent, tried + 1));
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|&c| c == '%')
}

fn sort_rows(rows: Vec<Record>, order_by: &[OrderKey]) -> Vec<Record> {
    let mut keyed: Vec<(Vec<Option<Json>>, Record)> = rows
        .into_iter()
        .map(|row| {
            let keys = order_by
                .iter()
                .map(|key| {
                    field_value(&row, &key.field)
                        .map(Cow::into_owned)
                        .filter(|v| !v.is_null())
                })
                .collect();
            (keys, row)
        })
        .collect();

    keyed.sort_by(|(a_keys, a), (b_keys, b)| {
        for ((key, x), y) in order_by.iter().zip(a_keys).zip(b_keys) {
            let ordering = match (x, y) {
                (Some(x), Some(y)) => {
                    let ordering = order_values(x, y);
                    if key.descending {
                        ordering.reverse()
                    } else {
                        ordering
                    }
                }
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            };
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        a.id.cmp(&b.id)
    });

    keyed.into_iter().map(|(_, row)| row).collect()
}

/// A total order for sorting mixed values: numbers, then strings, then booleans, then the rest.
fn order_values(a: &Json, b: &Json) -> Ordering {
    fn kind(v: &Json) -> u8 {
        match v {
            Json::Number(_) => 0,
            Json::String(_) => 1,
            Json::Bool(_) => 2,
            _ => 3,
        }
    }
    match (a, b) {
        (Json::Bool(x), Json::Bool(y)) => x.cmp(y),
        _ => comparable(a, b).unwrap_or_else(|| kind(a).cmp(&kind(b))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    // Mock storage for testing
    struct MockStorage;

    #[async_trait::async_trait]
    impl _Storage for MockStorage {
        async fn create_table(
            &self,
            config: crate::types::table::TableConfig,
        ) -> crate::Result<crate::types::table::Table> {
            Ok(crate::types::table::Table {
                id: uuid::Uuid::new_v4(),
                name: config.name,
                dimension: config.dimension,
                distance_metric: config.distance_metric,
                description: None,
                index_type: crate::types::IndexType::BruteForce,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                record_count: 0,
                text_fields: None,
                enable_full_text_search: false,
                schema_version: 1,
                payload_schema: None,
                memory_optimization: None,
                named_vectors: None,
                hnsw: None,
                quantization: None,
                payload_index: None,
            })
        }

        async fn get_table(&self, _name: &str) -> crate::Result<crate::types::table::Table> {
            Ok(crate::types::table::Table {
                id: uuid::Uuid::new_v4(),
                name: "test".to_string(),
                dimension: 128,
                distance_metric: DistanceMetric::Cosine,
                description: None,
                index_type: crate::types::IndexType::BruteForce,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                record_count: 0,
                text_fields: None,
                enable_full_text_search: false,
                schema_version: 1,
                payload_schema: None,
                memory_optimization: None,
                named_vectors: None,
                hnsw: None,
                quantization: None,
                payload_index: None,
            })
        }

        async fn list_tables(&self) -> crate::Result<Vec<crate::types::table::Table>> {
            Ok(vec![])
        }

        async fn delete_table(&self, _name: &str) -> crate::Result<()> {
            Ok(())
        }

        async fn insert_record(
            &self,
            _tablename: &str,
            _request: crate::types::record::CreateRecordRequest,
        ) -> crate::Result<crate::types::record::Record> {
            Ok(record("test", serde_json::Value::Null))
        }

        async fn batch_insert_records(
            &self,
            _tablename: &str,
            _requests: Vec<crate::types::record::CreateRecordRequest>,
        ) -> crate::Result<Vec<crate::types::record::Record>> {
            Ok(vec![])
        }

        async fn search(
            &self,
            _tablename: &str,
            _query: crate::types::record::SearchQuery,
        ) -> crate::Result<crate::types::record::SearchResult> {
            Ok(crate::types::record::SearchResult {
                hits: vec![],
                total: 0,
                tookms: 0,
            })
        }

        async fn get_record(
            &self,
            _tablename: &str,
            _record_id: &crate::types::vector::RecordId,
        ) -> crate::Result<crate::types::record::Record> {
            Ok(record("test", json!({"field": "old_value"})))
        }

        async fn update_record(
            &self,
            _tablename: &str,
            _record_id: &crate::types::vector::RecordId,
            request: crate::types::record::CreateRecordRequest,
        ) -> crate::Result<crate::types::record::Record> {
            Ok(record(
                &request.id,
                request.payload.unwrap_or(serde_json::Value::Null),
            ))
        }

        async fn delete_record(
            &self,
            _tablename: &str,
            _record_id: &crate::types::vector::RecordId,
        ) -> crate::Result<()> {
            Ok(())
        }

        async fn insert_records(
            &self,
            _tablename: &str,
            _requests: Vec<crate::types::record::CreateRecordRequest>,
        ) -> crate::Result<Vec<crate::types::record::Record>> {
            Ok(vec![])
        }

        async fn delete_records(
            &self,
            _tablename: &str,
            _record_ids: Vec<crate::types::vector::RecordId>,
        ) -> crate::Result<()> {
            Ok(())
        }

        async fn backup(&self, _tablename: &str, _backup_path: &str) -> crate::Result<()> {
            Ok(())
        }

        async fn restore(&self, _tablename: &str, _backup_path: &str) -> crate::Result<()> {
            Ok(())
        }
    }

    fn record(id: &str, payload: serde_json::Value) -> Record {
        Record {
            id: id.to_string(),
            vector: vec![],
            sparse_vector: None,
            payload,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
            status: crate::types::record::RecordStatus::Active,
            named_vectors: None,
        }
    }

    fn cmp(field: &str, operation: &str, value: serde_json::Value) -> FilterExpr {
        FilterExpr::Compare(Filter {
            field: field.to_string(),
            operation: operation.to_string(),
            value,
        })
    }

    #[tokio::test]
    async fn test_create_table() {
        let executor = StorageAwareExecutor::new(Arc::new(MockStorage));
        let plan = QueryPlan::Create(CreatePlan::Table {
            name: "test_table".to_string(),
            dimension: 4,
        });
        assert!(executor.execute(plan).await.is_ok());
    }

    #[tokio::test]
    async fn test_drop_table() {
        let executor = StorageAwareExecutor::new(Arc::new(MockStorage));
        let plan = QueryPlan::Drop(DropPlan {
            table: "test_table".to_string(),
        });
        assert!(executor.execute(plan).await.is_ok());
    }

    #[tokio::test]
    async fn an_update_or_delete_that_matches_nothing_affects_nothing() {
        // The mock lists no ids, so there is nothing to match.
        let executor = StorageAwareExecutor::new(Arc::new(MockStorage));

        let mut set_payload = serde_json::Map::new();
        set_payload.insert("status".to_string(), json!("active"));
        let update = executor
            .execute(QueryPlan::Update(UpdatePlan {
                table: "test_table".to_string(),
                set_payload,
                set_vector: None,
                filter: None,
            }))
            .await
            .expect("update");
        assert_eq!(update.metadata.plan_type, "UPDATE");
        assert_eq!(update.metadata.affected_rows, 0);

        let delete = executor
            .execute(QueryPlan::Delete(DeletePlan {
                table: "test_table".to_string(),
                filter: None,
            }))
            .await
            .expect("delete");
        assert_eq!(delete.metadata.plan_type, "DELETE");
        assert_eq!(delete.metadata.affected_rows, 0);
    }

    #[test]
    fn an_integer_field_equals_the_same_number_written_as_a_float() {
        let r = record("r3", json!({"rank": 3}));
        assert!(matches(&r, &cmp("rank", "=", json!(3.0))));
        assert!(!matches(&r, &cmp("rank", "!=", json!(3.0))));
        assert!(matches(
            &r,
            &FilterExpr::In {
                field: "rank".into(),
                values: vec![json!(1), json!(3.0)]
            }
        ));
    }

    #[test]
    fn a_missing_field_satisfies_nothing_not_even_not_equal() {
        let r = record("r1", json!({"rank": 1}));
        for op in ["=", "!=", "<", ">", "LIKE"] {
            assert!(!matches(&r, &cmp("city", op, json!("London"))), "{op}");
        }
        assert!(matches(&r, &cmp("id", "=", json!("r1"))));
    }

    #[test]
    fn strings_order_and_between_is_inclusive() {
        let r = record("r5", json!({"city": "Berlin", "rank": 5}));
        assert!(matches(&r, &cmp("city", "<", json!("London"))));
        let between = |low, high| FilterExpr::Between {
            field: "rank".into(),
            low,
            high,
        };
        assert!(matches(&r, &between(json!(5), json!(6))));
        assert!(matches(&r, &between(json!(4), json!(5.0))));
        assert!(!matches(&r, &between(json!(6), json!(9))));
        assert!(!matches(&r, &between(json!("a"), json!("z"))));
    }

    #[test]
    fn and_or_nest() {
        let r = record("r2", json!({"city": "London", "tag": "beta"}));
        let london_or_paris = FilterExpr::Or(vec![
            cmp("city", "=", json!("London")),
            cmp("city", "=", json!("Paris")),
        ]);
        assert!(matches(
            &r,
            &FilterExpr::And(vec![
                london_or_paris.clone(),
                cmp("tag", "=", json!("beta"))
            ])
        ));
        assert!(!matches(
            &r,
            &FilterExpr::And(vec![london_or_paris, cmp("tag", "=", json!("alpha"))])
        ));
    }

    #[test]
    fn like_is_sql_like() {
        for (text, pattern, expected) in [
            ("London", "Lon%", true),
            ("London", "%don", true),
            ("Londonderry", "%don", false),
            ("London", "%on%", true),
            ("Paris", "Par_s", true),
            ("London", "ondon", false),
            ("London", "London", true),
            ("London", "L_n%n", true),
            ("", "%", true),
            ("a", "_", true),
            ("", "_", false),
            ("100%", "100%", true),
            ("abc", "a%c%", true),
            ("abc", "a%b", false),
        ] {
            assert_eq!(like(text, pattern), expected, "{text:?} LIKE {pattern:?}");
        }
    }

    #[test]
    fn rows_sort_by_their_keys_with_missing_values_last_and_ties_by_id() {
        let rows = || {
            vec![
                record("b", json!({"rank": 2})),
                record("a", json!({})),
                record("c", json!({"rank": 2})),
                record("d", json!({"rank": 1})),
            ]
        };
        let ids = |rows: Vec<Record>| rows.into_iter().map(|r| r.id).collect::<Vec<_>>();

        let ascending = OrderKey {
            field: "rank".into(),
            descending: false,
        };
        assert_eq!(ids(sort_rows(rows(), &[ascending])), ["d", "b", "c", "a"]);

        let descending = OrderKey {
            field: "rank".into(),
            descending: true,
        };
        assert_eq!(ids(sort_rows(rows(), &[descending])), ["b", "c", "d", "a"]);

        assert_eq!(ids(sort_rows(rows(), &[])), ["a", "b", "c", "d"]);
    }
}
