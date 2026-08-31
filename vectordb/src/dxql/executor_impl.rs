//! Actual DXQL query execution implementation
//!
//! Provides full query execution with support for:
//! - Vector searches with filtering
//! - Metadata filtering and projection
//! - Result ranking and limiting
//! - Multiple data types (dense vectors, sparse vectors, payloads)

use crate::dxql::{compiler::*, DxqlError, QueryMetadata, QueryResult, ResultRecord};
use crate::storage::traits::_Storage;
use crate::types::{record::CreateRecordRequest, vector::DistanceMetric};
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

        let (records, plan_type) = match plan {
            QueryPlan::Select(select) => {
                let records = self.execute_select(select).await?;
                (records, "SELECT")
            }
            QueryPlan::Insert(insert) => {
                self.execute_insert(insert).await?;
                (vec![], "INSERT")
            }
            QueryPlan::Update(update) => {
                self.execute_update(update).await?;
                (vec![], "UPDATE")
            }
            QueryPlan::Delete(delete) => {
                self.execute_delete(delete).await?;
                (vec![], "DELETE")
            }
            QueryPlan::Create(create) => {
                self.execute_create(create).await?;
                (vec![], "CREATE")
            }
            QueryPlan::Drop(drop) => {
                self.execute_drop(drop).await?;
                (vec![], "DROP")
            }
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

    async fn execute_select(&self, select: SelectPlan) -> Result<Vec<ResultRecord>, DxqlError> {
        // Enumerate matching records using the same deterministic scan as UPDATE/DELETE
        let matching_ids = self.find_by_filters(&select.table, &select.filters).await?;

        let offset = select.offset.unwrap_or(0);
        let limit = select.limit.unwrap_or(usize::MAX);

        let mut records = Vec::new();

        for (idx, id) in matching_ids.iter().enumerate() {
            if idx < offset {
                continue;
            }
            if records.len() >= limit {
                break;
            }

            let record = match self.storage.get_record(&select.table, id).await {
                Ok(r) => r,
                Err(_) => continue,
            };

            let mut fields = HashMap::new();

            if select.projections.is_empty() {
                // No projections — include id, vector, and all payload fields
                fields.insert(
                    "id".to_string(),
                    serde_json::Value::String(record.id.clone()),
                );
                if let serde_json::Value::Object(map) = &record.payload {
                    for (k, v) in map {
                        fields.insert(k.clone(), v.clone());
                    }
                }
            } else {
                for projection in &select.projections {
                    let alias = projection.alias.as_ref().unwrap_or(&projection.field);
                    let value = match projection.field.as_str() {
                        "id" => serde_json::Value::String(record.id.clone()),
                        "vector" => serde_json::json!(record.vector),
                        _ => record
                            .payload
                            .get(&projection.field)
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                    };
                    fields.insert(alias.clone(), value);
                }
            }

            records.push(ResultRecord {
                id: record.id.clone(),
                score: None,
                fields,
            });
        }

        Ok(records)
    }

    async fn execute_insert(&self, insert: InsertPlan) -> Result<(), DxqlError> {
        // Verify table exists
        self.storage
            .get_table(&insert.table)
            .await
            .map_err(|e| DxqlError::UnknownTable(format!("Table '{}': {}", insert.table, e)))?;

        // Process each value as a record
        for value in insert.values {
            if let serde_json::Value::Object(obj) = value {
                let vector = obj
                    .get("vector")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| {
                        let vec: Result<Vec<f32>, _> = arr
                            .iter()
                            .map(|v| v.as_f64().ok_or(()))
                            .map(|f| f.map(|x| x as f32))
                            .collect();
                        vec.ok()
                    })
                    .ok_or_else(|| {
                        DxqlError::ExecutionError("Invalid or missing vector field".to_string())
                    })?;

                let payload = obj
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));

                // Get id from object or generate a new UUID
                let id = obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                let request = crate::types::record::CreateRecordRequest {
                    id,
                    vector,
                    sparse_vector: None,
                    payload: Some(payload),
                    named_vectors: None,
                };

                self.storage
                    .insert_record(&insert.table, request)
                    .await
                    .map_err(|e| DxqlError::ExecutionError(format!("Insert failed: {}", e)))?;
            }
        }

        Ok(())
    }

    async fn execute_update(&self, update: UpdatePlan) -> Result<(), DxqlError> {
        // Find matching records via a broad search then filter in memory
        let candidates = self.find_by_filters(&update.table, &update.filters).await?;

        for record_id in candidates {
            // Fetch current record state
            let current = self
                .storage
                .get_record(&update.table, &record_id)
                .await
                .map_err(|e| DxqlError::ExecutionError(format!("Get record failed: {}", e)))?;

            // Apply each assignment to a mutable payload clone
            let mut payload = if let serde_json::Value::Object(map) = current.payload.clone() {
                map
            } else {
                serde_json::Map::new()
            };

            let mut new_vector: Option<Vec<f32>> = None;
            for assignment in &update.assignments {
                match assignment.field.as_str() {
                    "vector" => {
                        if let Some(arr) = assignment.value.as_array() {
                            let v: Option<Vec<f32>> =
                                arr.iter().map(|x| x.as_f64().map(|f| f as f32)).collect();
                            new_vector = v;
                        }
                    }
                    _ => {
                        payload.insert(assignment.field.clone(), assignment.value.clone());
                    }
                }
            }

            let updated_vector = new_vector.unwrap_or_else(|| current.vector.clone());
            let request = CreateRecordRequest {
                id: record_id.clone(),
                vector: updated_vector,
                sparse_vector: current.sparse_vector.clone(),
                payload: Some(serde_json::Value::Object(payload)),
                named_vectors: None,
            };

            self.storage
                .update_record(&update.table, &record_id, request)
                .await
                .map_err(|e| DxqlError::ExecutionError(format!("Update record failed: {}", e)))?;
        }

        Ok(())
    }

    async fn execute_delete(&self, delete: DeletePlan) -> Result<(), DxqlError> {
        let ids = self.find_by_filters(&delete.table, &delete.filters).await?;

        if !ids.is_empty() {
            self.storage
                .delete_records(&delete.table, ids)
                .await
                .map_err(|e| DxqlError::ExecutionError(format!("Delete records failed: {}", e)))?;
        }

        Ok(())
    }

    /// Scan the table and return record IDs that satisfy all DXQL filters.
    async fn find_by_filters(
        &self,
        table: &str,
        filters: &[Filter],
    ) -> Result<Vec<crate::types::vector::RecordId>, DxqlError> {
        // Enumerate all records and apply filters in memory.
        // This is O(N) but correct — ANN search with a zero vector is unreliable for
        // arbitrary distance metrics and silently misses records beyond top_k.
        let all_ids =
            self.storage.list_all_record_ids(table).await.map_err(|e| {
                DxqlError::ExecutionError(format!("list_all_record_ids failed: {}", e))
            })?;

        if filters.is_empty() {
            return Ok(all_ids);
        }

        // For each ID, fetch the record and evaluate all filters
        let mut matching_ids = Vec::new();
        for id in all_ids {
            let record = match self.storage.get_record(table, &id).await {
                Ok(r) => r,
                Err(_) => continue,
            };
            if Self::record_matches_filters(&record, filters) {
                matching_ids.push(id);
            }
        }
        Ok(matching_ids)
    }

    fn record_matches_filters(record: &crate::types::record::Record, filters: &[Filter]) -> bool {
        for f in filters {
            let field_value = match f.field.as_str() {
                "id" => Some(serde_json::Value::String(record.id.clone())),
                _ => record.payload.get(&f.field).cloned(),
            };
            let matches = match (field_value, &f.value) {
                (Some(fv), filter_val) => match f.operation.as_str() {
                    "=" => &fv == filter_val,
                    "!=" => &fv != filter_val,
                    ">" => fv
                        .as_f64()
                        .zip(filter_val.as_f64())
                        .map(|(a, b)| a > b)
                        .unwrap_or(false),
                    ">=" => fv
                        .as_f64()
                        .zip(filter_val.as_f64())
                        .map(|(a, b)| a >= b)
                        .unwrap_or(false),
                    "<" => fv
                        .as_f64()
                        .zip(filter_val.as_f64())
                        .map(|(a, b)| a < b)
                        .unwrap_or(false),
                    "<=" => fv
                        .as_f64()
                        .zip(filter_val.as_f64())
                        .map(|(a, b)| a <= b)
                        .unwrap_or(false),
                    "LIKE" => fv
                        .as_str()
                        .zip(filter_val.as_str())
                        .map(|(s, pat)| s.contains(&pat.replace('%', "")))
                        .unwrap_or(false),
                    _ => false,
                },
                (None, _) => false,
            };
            if !matches {
                return false;
            }
        }
        true
    }

    async fn execute_create(&self, create: CreatePlan) -> Result<(), DxqlError> {
        match create {
            CreatePlan::Table { name } => {
                // Create table with default configuration
                let config = crate::types::table::TableConfig {
                    name: name.clone(),
                    dimension: 128, // Default dimension
                    distance_metric: DistanceMetric::Cosine,
                    description: Some("Created via DXQL".to_string()),
                    index_type: None,
                    text_fields: None,
                    enable_full_text_search: false,
                    memory_optimization: None, // Use global config
                    payload_schema: None,
                    schema_version: 1,
                    named_vectors: None,
                };

                self.storage.create_table(config).await.map_err(|e| {
                    DxqlError::ExecutionError(format!("Create table failed: {}", e))
                })?;

                Ok(())
            }
            CreatePlan::Index { name } => Err(DxqlError::ExecutionError(format!(
                "CREATE INDEX '{}' not yet implemented",
                name
            ))),
        }
    }

    async fn execute_drop(&self, drop: DropPlan) -> Result<(), DxqlError> {
        match drop.object_type.to_uppercase().as_str() {
            "TABLE" => {
                self.storage
                    .delete_table(&drop.name)
                    .await
                    .map_err(|e| DxqlError::ExecutionError(format!("Drop table failed: {}", e)))?;
                Ok(())
            }
            _ => Err(DxqlError::ExecutionError(format!(
                "DROP {} not supported",
                drop.object_type
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            Ok(crate::types::record::Record {
                id: "test".to_string(),
                vector: vec![],
                sparse_vector: None,
                payload: serde_json::Value::Null,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
                status: crate::types::record::RecordStatus::Active,
                named_vectors: None,
            })
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
            Ok(crate::types::record::Record {
                id: "test".to_string(),
                vector: vec![0.0; 128],
                sparse_vector: None,
                payload: serde_json::json!({"field": "old_value"}),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
                status: crate::types::record::RecordStatus::Active,
                named_vectors: None,
            })
        }

        async fn update_record(
            &self,
            _tablename: &str,
            _record_id: &crate::types::vector::RecordId,
            request: crate::types::record::CreateRecordRequest,
        ) -> crate::Result<crate::types::record::Record> {
            Ok(crate::types::record::Record {
                id: request.id,
                vector: request.vector,
                sparse_vector: None,
                payload: request.payload.unwrap_or(serde_json::Value::Null),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 2,
                status: crate::types::record::RecordStatus::Active,
                named_vectors: None,
            })
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

    #[tokio::test]
    async fn test_create_table() {
        let storage = Arc::new(MockStorage);
        let executor = StorageAwareExecutor::new(storage);

        let plan = QueryPlan::Create(CreatePlan::Table {
            name: "test_table".to_string(),
        });

        let result = executor.execute(plan).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_drop_table() {
        let storage = Arc::new(MockStorage);
        let executor = StorageAwareExecutor::new(storage);

        let plan = QueryPlan::Drop(DropPlan {
            object_type: "TABLE".to_string(),
            name: "test_table".to_string(),
        });

        let result = executor.execute(plan).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_update_no_filter() {
        let storage = Arc::new(MockStorage);
        let executor = StorageAwareExecutor::new(storage);

        // search() returns empty hits → update loop is a no-op
        let plan = QueryPlan::Update(UpdatePlan {
            table: "test_table".to_string(),
            assignments: vec![Assignment {
                field: "status".to_string(),
                value: serde_json::Value::String("active".to_string()),
            }],
            filters: vec![],
        });

        let result = executor.execute(plan).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().metadata.plan_type, "UPDATE");
    }

    #[tokio::test]
    async fn test_delete_no_filter() {
        let storage = Arc::new(MockStorage);
        let executor = StorageAwareExecutor::new(storage);

        // search() returns empty hits → delete is a no-op
        let plan = QueryPlan::Delete(DeletePlan {
            table: "test_table".to_string(),
            filters: vec![],
        });

        let result = executor.execute(plan).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().metadata.plan_type, "DELETE");
    }

    #[test]
    fn test_record_matches_filters() {
        let record = crate::types::record::Record {
            id: "r1".to_string(),
            vector: vec![],
            sparse_vector: None,
            payload: serde_json::json!({ "age": 30, "name": "Alice" }),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
            status: crate::types::record::RecordStatus::Active,
            named_vectors: None,
        };

        let match_filters = vec![Filter {
            field: "age".to_string(),
            operation: "=".to_string(),
            value: serde_json::json!(30),
        }];
        assert!(StorageAwareExecutor::record_matches_filters(
            &record,
            &match_filters
        ));

        let no_match_filters = vec![Filter {
            field: "age".to_string(),
            operation: ">".to_string(),
            value: serde_json::json!(50),
        }];
        assert!(!StorageAwareExecutor::record_matches_filters(
            &record,
            &no_match_filters
        ));

        // Empty filters always match
        assert!(StorageAwareExecutor::record_matches_filters(&record, &[]));

        // LIKE filter
        let like_filter = vec![Filter {
            field: "name".to_string(),
            operation: "LIKE".to_string(),
            value: serde_json::json!("%Alice%"),
        }];
        assert!(StorageAwareExecutor::record_matches_filters(
            &record,
            &like_filter
        ));
    }
}
