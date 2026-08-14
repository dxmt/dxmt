//! Storage trait implementation for RocksDB backend

use async_trait::async_trait;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::storage::traits::_Storage;
use crate::types::{
    record::{CreateRecordRequest, Record, SearchQuery, SearchResult},
    table::{Table, TableConfig},
    vector::RecordId,
};
use crate::error::{Error, Result};

use super::{RocksDbStorage, VectorRecord, TableMetadata};

pub struct RocksDbStorageImpl {
    inner: Arc<RocksDbStorage>,
    tables: Arc<RwLock<std::collections::HashMap<String, Table>>>,
}

impl RocksDbStorageImpl {
    pub fn new(rocksdb: RocksDbStorage) -> Self {
        Self {
            inner: Arc::new(rocksdb),
            tables: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }
}

#[async_trait]
impl _Storage for RocksDbStorageImpl {
    async fn create_table(&self, config: TableConfig) -> Result<Table> {
        let metadata = TableMetadata {
            name: config.name.clone(),
            dimension: config.dimension,
            total_records: 0,
        };

        self.inner
            .update_table_metadata(&config.name, &metadata)
            .map_err(|e| Error::storage(format!("Failed to create table: {}", e)))?;

        let table = Table {
            id: uuid::Uuid::new_v4(),
            name: config.name.clone(),
            dimension: config.dimension,
            distance_metric: config.distance_metric,
            description: config.description.clone(),
            index_type: config.index_type.unwrap_or(crate::types::IndexType::BruteForce),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            record_count: 0,
            text_fields: config.text_fields,
            enable_full_text_search: config.enable_full_text_search,
        };

        let mut tables = self.tables.write().await;
        tables.insert(config.name, table.clone());

        Ok(table)
    }

    async fn get_table(&self, name: &str) -> Result<Table> {
        {
            let tables = self.tables.read().await;
            if let Some(table) = tables.get(name) {
                return Ok(table.clone());
            }
        }

        let metadata = self.inner
            .get_table_metadata(name)
            .map_err(|e| Error::storage(format!("Failed to get table metadata: {}", e)))?
            .ok_or_else(|| Error::NotFound(format!("Table not found: {}", name)))?;

        let table = Table {
            id: uuid::Uuid::new_v4(),
            name: metadata.name.clone(),
            dimension: metadata.dimension,
            distance_metric: crate::types::vector::DistanceMetric::Cosine,
            description: None,
            index_type: crate::types::IndexType::BruteForce,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            record_count: metadata.total_records,
            text_fields: None,
            enable_full_text_search: false,
        };

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
        Ok(())
    }

    async fn insert_record(&self, tablename: &str, request: CreateRecordRequest) -> Result<Record> {
        self.get_table(tablename).await?;

        let record_id = RecordId::new();
        let vector_record = VectorRecord {
            id: record_id.clone(),
            vector: request.vector.clone(),
            payload: request.payload.clone(),
        };

        self.inner
            .insert_vector(&vector_record)
            .map_err(|e| Error::storage(format!("Failed to insert record: {}", e)))?;

        if let Ok(Some(mut metadata)) = self.inner.get_table_metadata(tablename) {
            metadata.total_records += 1;
            let _ = self.inner.update_table_metadata(tablename, &metadata);
        }

        Ok(Record {
            id: record_id,
            vector: request.vector,
            sparse_vector: None,
            payload: request.payload.unwrap_or_default(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
            status: crate::types::record::RecordStatus::Active,
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
        tracing::warn!("Vector search not yet implemented for RocksDB backend");
        Ok(SearchResult {
            hits: Vec::new(),
            total: 0,
            tookms: 0,
        })
    }

    async fn get_record(&self, tablename: &str, record_id: &RecordId) -> Result<Record> {
        self.get_table(tablename).await?;

        let vector_record = self.inner
            .get_vector(record_id)
            .map_err(|e| Error::storage(format!("Failed to get record: {}", e)))?
            .ok_or_else(|| Error::NotFound(format!("Record not found: {}", record_id)))?;

        Ok(Record {
            id: vector_record.id,
            vector: vector_record.vector,
            sparse_vector: None,
            payload: vector_record.payload.unwrap_or_default(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
            status: crate::types::record::RecordStatus::Active,
        })
    }

    async fn update_record(
        &self,
        tablename: &str,
        record_id: &RecordId,
        request: CreateRecordRequest,
    ) -> Result<Record> {
        self.get_table(tablename).await?;
        self.get_record(tablename, record_id).await?;

        let vector_record = VectorRecord {
            id: record_id.clone(),
            vector: request.vector.clone(),
            payload: request.payload.clone(),
        };

        self.inner
            .insert_vector(&vector_record)
            .map_err(|e| Error::storage(format!("Failed to update record: {}", e)))?;

        Ok(Record {
            id: record_id.clone(),
            vector: request.vector,
            sparse_vector: None,
            payload: request.payload.unwrap_or_default(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
            status: crate::types::record::RecordStatus::Active,
        })
    }

    async fn delete_record(&self, tablename: &str, record_id: &RecordId) -> Result<()> {
        self.get_table(tablename).await?;

        self.inner
            .delete_vector(record_id)
            .map_err(|e| Error::storage(format!("Failed to delete record: {}", e)))?;

        if let Ok(Some(mut metadata)) = self.inner.get_table_metadata(tablename) {
            if metadata.total_records > 0 {
                metadata.total_records -= 1;
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
        Err(Error::NotImplemented("Backup not yet implemented for RocksDB".to_string()))
    }

    async fn restore(&self, _tablename: &str, _backup_path: &str) -> Result<()> {
        Err(Error::NotImplemented("Restore not yet implemented for RocksDB".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_create_and_get_table() {
        let dir = tempdir().unwrap();
        let rocksdb = RocksDbStorage::new(dir.path()).unwrap();
        let storage = RocksDbStorageImpl::new(rocksdb);

        let config = TableConfig {
            name: "test_table".to_string(),
            dimension: 128,
            distance_metric: crate::types::vector::DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
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
        let rocksdb = RocksDbStorage::new(dir.path()).unwrap();
        let storage = RocksDbStorageImpl::new(rocksdb);

        let config = TableConfig {
            name: "test_table".to_string(),
            dimension: 128,
            distance_metric: crate::types::vector::DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
        };
        storage.create_table(config).await.unwrap();

        let request = CreateRecordRequest {
            vector: vec![0.1; 128],
            payload: Some(serde_json::json!({"name": "test"})),
        };

        let record = storage.insert_record("test_table", request).await.unwrap();
        let record_id = record.id.clone();

        let retrieved = storage
            .get_record("test_table", &record_id)
            .await
            .unwrap();
        assert_eq!(retrieved.id, record_id);
    }
}
