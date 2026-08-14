use crate::types::{
    record::{CreateRecordRequest, Record, SearchQuery, SearchResult},
    table::{Table, TableAlterRequest, TableConfig},
    vector::RecordId,
};
use async_trait::async_trait;

/// Storage trait for vector database operations
#[async_trait]
pub trait _Storage: Send + Sync {
    /// Create a new table
    async fn create_table(&self, config: TableConfig) -> crate::Result<Table>;

    /// Get table information
    async fn get_table(&self, name: &str) -> crate::Result<Table>;

    /// List all tables
    async fn list_tables(&self) -> crate::Result<Vec<Table>>;

    /// Delete a table
    async fn delete_table(&self, name: &str) -> crate::Result<()>;

    /// Alter table schema (safe mutations only: description, memory_optimization,
    /// enable_full_text_search toggle, add/drop payload field declarations).
    async fn alter_table(&self, name: &str, req: TableAlterRequest) -> crate::Result<Table> {
        let _ = (name, req);
        Err(crate::Error::NotImplemented(
            "alter_table not implemented for this backend".to_string(),
        ))
    }

    /// Insert a record
    async fn insert_record(
        &self,
        tablename: &str,
        request: CreateRecordRequest,
    ) -> crate::Result<Record>;

    /// Batch insert records
    async fn batch_insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>>;

    /// Search for similar vectors
    async fn search(&self, tablename: &str, query: SearchQuery) -> crate::Result<SearchResult>;

    /// Get a specific record
    async fn get_record(&self, tablename: &str, record_id: &RecordId) -> crate::Result<Record>;

    /// Update a record
    async fn update_record(
        &self,
        tablename: &str,
        record_id: &RecordId,
        request: CreateRecordRequest,
    ) -> crate::Result<Record>;

    /// Delete a record
    async fn delete_record(&self, tablename: &str, record_id: &RecordId) -> crate::Result<()>;

    /// Insert multiple records
    async fn insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>>;

    /// Delete multiple records
    async fn delete_records(&self, tablename: &str, record_ids: Vec<RecordId>)
        -> crate::Result<()>;

    /// Backup table data
    async fn backup(&self, tablename: &str, backup_path: &str) -> crate::Result<()>;

    /// Restore table data
    async fn restore(&self, tablename: &str, backup_path: &str) -> crate::Result<()>;

    /// Insert with storage API
    async fn insert(&self, tablename: &str, record: CreateRecordRequest) -> crate::Result<Record> {
        self.insert_record(tablename, record).await
    }

    /// Get table info
    async fn get_table_info(&self, tablename: &str) -> crate::Result<Table> {
        self.get_table(tablename).await
    }

    /// Delete by IDs
    async fn delete_by_ids(&self, tablename: &str, ids: Vec<RecordId>) -> crate::Result<()> {
        self.delete_records(tablename, ids).await
    }

    /// Delete by filter (stub implementation)
    async fn delete_by_filter(
        &self,
        _tablename: &str,
        _filter: serde_json::Value,
    ) -> crate::Result<()> {
        Err(crate::Error::NotImplemented(
            "delete_by_filter not implemented".to_string(),
        ))
    }

    /// Hybrid search (stub implementation)
    async fn hybrid_search(
        &self,
        tablename: &str,
        query: SearchQuery,
    ) -> crate::Result<SearchResult> {
        self.search(tablename, query).await
    }

    /// Get table schema (stub implementation)
    async fn get_table_schema(&self, tablename: &str) -> crate::Result<serde_json::Value> {
        let table = self.get_table(tablename).await?;
        Ok(serde_json::json!({
            "dimension": table.dimension,
            "distance_metric": table.distance_metric,
            "index_type": table.index_type
        }))
    }

    /// Get table stats (stub implementation)
    async fn get_table_stats(&self, tablename: &str) -> crate::Result<serde_json::Value> {
        let table = self.get_table(tablename).await?;
        Ok(serde_json::json!({
            "record_count": table.record_count,
            "dimension": table.dimension
        }))
    }

    /// List all record IDs in a table. Default: returns empty vec (override for real implementations).
    async fn list_all_record_ids(&self, _tablename: &str) -> crate::Result<Vec<RecordId>> {
        Ok(vec![])
    }

    /// Get sample vectors (stub implementation)
    async fn get_sample_vectors(
        &self,
        _tablename: &str,
        _limit: usize,
    ) -> crate::Result<Vec<Record>> {
        Ok(vec![])
    }

    /// Get metrics (stub implementation)
    async fn get_metrics(&self) -> crate::Result<serde_json::Value> {
        Ok(serde_json::json!({}))
    }
}

/// Storage configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct StorageConfig {
    pub storage_type: StorageType,
    pub path: Option<String>,
    pub cache_size_mb: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageType {
    Memory,
    Disk,
    Tiered,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            storage_type: StorageType::Memory,
            path: None,
            cache_size_mb: Some(1024),
        }
    }
}
