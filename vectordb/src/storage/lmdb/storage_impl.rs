//! Storage trait implementation for LMDB backend
//!
//! Implements the `_Storage` trait to provide full vector database operations
//! with LMDB as the persistence backend.

use async_trait::async_trait;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{Error, Result};
use crate::storage::traits::_Storage;
use crate::types::{
    record::{CreateRecordRequest, Record, SearchQuery, SearchResult},
    table::{IndexType, Table, TableAlterRequest, TableConfig},
    vector::RecordId,
};

use super::{LmdbStorage, TableMetadata, VectorRecord};

/// LMDB-backed storage implementation
pub struct LmdbStorageImpl {
    inner: Arc<LmdbStorage>,
    // In-memory table catalog for fast lookups
    tables: Arc<RwLock<std::collections::HashMap<String, Table>>>,
}

impl LmdbStorageImpl {
    /// Create a new LMDB storage instance
    pub fn new(lmdb: LmdbStorage) -> Self {
        Self {
            inner: Arc::new(lmdb),
            tables: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Helper to convert between internal VectorRecord and public Record
    fn vector_record_to_record(vr: VectorRecord, _table_name: &str) -> Record {
        use crate::types::record::RecordStatus;
        use chrono::Utc;

        Record {
            id: vr.id,
            vector: vr.vector,
            sparse_vector: None,
            payload: vr.payload.unwrap_or_default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            version: 1,
            status: RecordStatus::Active,
            named_vectors: None,
        }
    }

    /// Helper to convert public Record to internal VectorRecord
    #[allow(dead_code)]
    fn record_to_vector_record(r: Record) -> VectorRecord {
        VectorRecord {
            id: r.id,
            vector: r.vector,
            payload: if r.payload.is_object() && !r.payload.as_object().unwrap().is_empty() {
                Some(r.payload)
            } else {
                None
            },
        }
    }
}

#[async_trait]
impl _Storage for LmdbStorageImpl {
    async fn create_table(&self, config: TableConfig) -> Result<Table> {
        // Create table metadata
        let metadata = TableMetadata {
            name: config.name.clone(),
            dimension: config.dimension,
            distance_metric: config.distance_metric,
            total_records: 0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Persist to LMDB (legacy minimal metadata for backward compat)
        self.inner
            .update_table_metadata(&config.name, &metadata)
            .map_err(|e| Error::storage(format!("Failed to create table: {}", e)))?;

        // Create Table object
        let table = Table {
            id: uuid::Uuid::new_v4(),
            name: config.name.clone(),
            dimension: config.dimension,
            distance_metric: config.distance_metric,
            description: config.description.clone(),
            index_type: config.index_type.unwrap_or(IndexType::BruteForce),
            created_at: metadata.created_at,
            updated_at: metadata.updated_at,
            record_count: 0,
            text_fields: config.text_fields,
            enable_full_text_search: config.enable_full_text_search,
            schema_version: config.schema_version,
            payload_schema: config.payload_schema,
            memory_optimization: config.memory_optimization,
            named_vectors: None,
        };

        // Persist full config so all fields survive a restart
        self.inner
            .store_full_table(&table)
            .map_err(|e| Error::storage(format!("Failed to persist full table config: {}", e)))?;

        // Cache in memory
        let mut tables = self.tables.write().await;
        tables.insert(config.name, table.clone());

        Ok(table)
    }

    async fn get_table(&self, name: &str) -> Result<Table> {
        // Try memory cache first
        {
            let tables = self.tables.read().await;
            if let Some(table) = tables.get(name) {
                return Ok(table.clone());
            }
        }

        // Try to load the full persisted config first (new path)
        let full_table_opt = self
            .inner
            .load_full_table(name)
            .map_err(|e| Error::storage(format!("Failed to load full table config: {}", e)))?;
        if let Some(mut table) = full_table_opt {
            // Sync record_count from legacy metadata for accuracy
            if let Ok(Some(meta)) = self.inner.get_table_metadata(name) {
                table.record_count = meta.total_records as u64;
            }
            let mut tables = self.tables.write().await;
            tables.insert(name.to_string(), table.clone());
            return Ok(table);
        }

        // Fallback: legacy minimal metadata (tables created before full-config persistence)
        let metadata = self
            .inner
            .get_table_metadata(name)
            .map_err(|e| Error::storage(format!("Failed to get table metadata: {}", e)))?
            .ok_or_else(|| Error::NotFound(format!("Table not found: {}", name)))?;

        let table = Table {
            id: uuid::Uuid::new_v4(),
            name: metadata.name.clone(),
            dimension: metadata.dimension,
            distance_metric: metadata.distance_metric,
            description: None,
            index_type: IndexType::BruteForce,
            updated_at: metadata.updated_at,
            created_at: metadata.created_at,
            record_count: metadata.total_records as u64,
            text_fields: None,
            enable_full_text_search: false,
            schema_version: 1,
            payload_schema: None,
            memory_optimization: None,
            named_vectors: None,
        };

        // Cache result
        let mut tables = self.tables.write().await;
        tables.insert(name.to_string(), table.clone());

        Ok(table)
    }

    async fn list_tables(&self) -> Result<Vec<Table>> {
        let tables = self.tables.read().await;
        Ok(tables.values().cloned().collect())
    }

    async fn delete_table(&self, name: &str) -> Result<()> {
        let mut tables = self.tables.write().await;
        tables.remove(name);
        // Note: Actual data cleanup from LMDB would require additional implementation
        Ok(())
    }

    async fn alter_table(&self, name: &str, req: TableAlterRequest) -> Result<Table> {
        let current = self.get_table(name).await?;
        let updated = current.apply_alter(&req);

        // Persist the updated full config
        self.inner
            .store_full_table(&updated)
            .map_err(|e| Error::storage(format!("Failed to persist altered table: {}", e)))?;

        // Update in-memory cache
        let mut tables = self.tables.write().await;
        tables.insert(name.to_string(), updated.clone());

        Ok(updated)
    }

    async fn insert_record(&self, tablename: &str, request: CreateRecordRequest) -> Result<Record> {
        // Verify table exists
        self.get_table(tablename).await?;

        // Use provided ID or generate a new one
        let record_id = if !request.id.is_empty() {
            request.id.clone()
        } else {
            uuid::Uuid::new_v4().to_string()
        };

        let vector_record = VectorRecord {
            id: record_id.clone(),
            vector: request.vector.clone(),
            payload: request.payload.clone(),
        };

        // Persist to LMDB
        self.inner
            .insert_vector(&vector_record)
            .map_err(|e| Error::storage(format!("Failed to insert record: {}", e)))?;

        // Update table metadata
        if let Ok(Some(mut meta)) = self
            .inner
            .get_table_metadata(tablename)
            .map_err(|e| Error::storage(format!("Failed to get metadata: {}", e)))
        {
            meta.total_records += 1;
            meta.updated_at = Utc::now();
            let _ = self.inner.update_table_metadata(tablename, &meta);
        }

        Ok(Record {
            id: record_id,
            vector: request.vector,
            sparse_vector: request.sparse_vector,
            payload: request.payload.unwrap_or_default(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
            status: crate::types::record::RecordStatus::Active,
            named_vectors: None,
        })
    }

    async fn batch_insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> Result<Vec<Record>> {
        let mut results = Vec::new();
        for request in requests {
            results.push(self.insert_record(tablename, request).await?);
        }
        Ok(results)
    }

    async fn search(&self, _tablename: &str, _query: SearchQuery) -> Result<SearchResult> {
        // This is a stub - full vector search requires index implementation
        // For now, return empty results
        tracing::warn!("Vector search not yet implemented for LMDB backend");

        Ok(SearchResult {
            hits: Vec::new(),
            total: 0,
            tookms: 0,
        })
    }

    async fn get_record(&self, tablename: &str, record_id: &RecordId) -> Result<Record> {
        // Verify table exists
        self.get_table(tablename).await?;

        let vector_record = self
            .inner
            .get_vector(record_id)
            .map_err(|e| Error::storage(format!("Failed to get record: {}", e)))?
            .ok_or_else(|| Error::NotFound(format!("Record not found: {}", record_id)))?;

        Ok(Self::vector_record_to_record(vector_record, tablename))
    }

    async fn update_record(
        &self,
        tablename: &str,
        record_id: &RecordId,
        request: CreateRecordRequest,
    ) -> Result<Record> {
        // Verify table and record exist; preserve original created_at and bump version
        self.get_table(tablename).await?;
        let existing = self.get_record(tablename, record_id).await?;

        let vector_record = VectorRecord {
            id: record_id.clone(),
            vector: request.vector.clone(),
            payload: request.payload.clone(),
        };

        // Update in LMDB
        self.inner
            .insert_vector(&vector_record)
            .map_err(|e| Error::storage(format!("Failed to update record: {}", e)))?;

        Ok(Record {
            id: record_id.clone(),
            vector: request.vector,
            sparse_vector: request.sparse_vector,
            payload: request.payload.unwrap_or_default(),
            created_at: existing.created_at,
            updated_at: chrono::Utc::now(),
            version: existing.version + 1,
            status: crate::types::record::RecordStatus::Active,
            named_vectors: None,
        })
    }

    async fn delete_record(&self, tablename: &str, record_id: &RecordId) -> Result<()> {
        // Verify table exists
        self.get_table(tablename).await?;

        // Delete from LMDB
        self.inner
            .delete_vector(record_id)
            .map_err(|e| Error::storage(format!("Failed to delete record: {}", e)))?;

        // Update table metadata
        if let Ok(Some(mut metadata)) = self.inner.get_table_metadata(tablename) {
            if metadata.total_records > 0 {
                metadata.total_records -= 1;
                metadata.updated_at = Utc::now();
                let _ = self.inner.update_table_metadata(tablename, &metadata);
            }
        }

        Ok(())
    }

    async fn insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> Result<Vec<Record>> {
        self.batch_insert_records(tablename, requests).await
    }

    async fn delete_records(&self, tablename: &str, record_ids: Vec<RecordId>) -> Result<()> {
        for record_id in record_ids {
            self.delete_record(tablename, &record_id).await?;
        }
        Ok(())
    }

    async fn backup(&self, _tablename: &str, _backup_path: &str) -> Result<()> {
        Err(Error::NotImplemented(
            "Backup not yet implemented for LMDB".to_string(),
        ))
    }

    async fn restore(&self, _tablename: &str, _backup_path: &str) -> Result<()> {
        Err(Error::NotImplemented(
            "Restore not yet implemented for LMDB".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_create_and_get_table() {
        let dir = tempdir().unwrap();
        let lmdb = LmdbStorage::new(dir.path(), 10 * 1024 * 1024).unwrap();
        let storage = LmdbStorageImpl::new(lmdb);

        let config = TableConfig {
            name: "test_table".to_string(),
            dimension: 128,
            distance_metric: crate::types::vector::DistanceMetric::Cosine,
            description: None,
            index_type: Some(IndexType::BruteForce),
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
        };

        let table = storage.create_table(config).await.unwrap();
        assert_eq!(table.name, "test_table");
        assert_eq!(table.dimension, 128);

        let retrieved = storage.get_table("test_table").await.unwrap();
        assert_eq!(retrieved.name, "test_table");
    }

    #[tokio::test]
    async fn test_insert_and_get_record() {
        let dir = tempdir().unwrap();
        let lmdb = LmdbStorage::new(dir.path(), 10 * 1024 * 1024).unwrap();
        let storage = LmdbStorageImpl::new(lmdb);

        // Create table
        let config = TableConfig {
            name: "test_table".to_string(),
            dimension: 128,
            distance_metric: crate::types::vector::DistanceMetric::Cosine,
            description: None,
            index_type: Some(IndexType::BruteForce),
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
        };
        storage.create_table(config).await.unwrap();

        // Insert record
        let request = CreateRecordRequest {
            id: "test_record_1".to_string(),
            vector: vec![0.1; 128],
            sparse_vector: None,
            payload: Some(serde_json::json!({"name": "test"})),
            named_vectors: None,
        };

        let record = storage.insert_record("test_table", request).await.unwrap();
        let record_id = record.id.clone();

        // Get record
        let retrieved = storage.get_record("test_table", &record_id).await.unwrap();
        assert_eq!(retrieved.id, record_id);
        assert_eq!(retrieved.vector.len(), 128);
    }

    #[tokio::test]
    async fn test_delete_record() {
        let dir = tempdir().unwrap();
        let lmdb = LmdbStorage::new(dir.path(), 10 * 1024 * 1024).unwrap();
        let storage = LmdbStorageImpl::new(lmdb);

        // Create table and insert record
        let config = TableConfig {
            name: "test_table".to_string(),
            dimension: 128,
            distance_metric: crate::types::vector::DistanceMetric::Cosine,
            description: None,
            index_type: Some(IndexType::BruteForce),
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
        };
        storage.create_table(config).await.unwrap();

        let request = CreateRecordRequest {
            id: "test_record_2".to_string(),
            vector: vec![0.1; 128],
            sparse_vector: None,
            payload: None,
            named_vectors: None,
        };

        let record = storage.insert_record("test_table", request).await.unwrap();
        let record_id = record.id;

        // Delete
        storage
            .delete_record("test_table", &record_id)
            .await
            .unwrap();

        // Verify deletion
        let result = storage.get_record("test_table", &record_id).await;
        assert!(result.is_err());
    }
}
