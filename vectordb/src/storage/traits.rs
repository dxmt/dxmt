use crate::types::{
    record::{CreateRecordRequest, HybridSearchRequest, Record, SearchQuery, SearchResult},
    table::{Table, TableAlterRequest, TableConfig},
    vector::RecordId,
};
use async_trait::async_trait;

/// Storage trait for vector database operations
#[async_trait]
// `async_trait` marks each generated method `#[must_use]` and returns a boxed future, which
// is must-use already; clippy 1.99's `double_must_use` fires on every method of the trait.
// The attribute is the macro's, not ours, and async-trait 0.1.89 is the newest release.
// `unknown_lints` keeps a toolchain older than 1.99, which has no such lint, quiet.
#[allow(unknown_lints, clippy::double_must_use)]
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

    /// Hybrid search: dense, sparse and text, fused.
    ///
    /// This took a `SearchQuery` and ran `search` on it, so the one caller -- the MCP
    /// `hybrid_search` tool -- ran a plain vector search under the name, and answered a
    /// text-only call with an empty page and a note. `IntelligentIndexedStorage` overrides it
    /// with the real fusion. A storage without text or sparse search answers a request that
    /// needs either with `NotImplemented`, rather than the dense half dressed up as a fusion.
    async fn hybrid_search(
        &self,
        tablename: &str,
        request: HybridSearchRequest,
    ) -> crate::Result<SearchResult> {
        match &request.vector {
            Some(vector) if request.text.is_none() && request.sparse_vector.is_none() => {
                self.search(
                    tablename,
                    SearchQuery {
                        vector: vector.clone(),
                        top_k: request.limit,
                        limit: request.limit,
                        filter: request.filter.clone(),
                        with_payload: request.with_payload,
                        with_vector: request.with_vector,
                        ..Default::default()
                    },
                )
                .await
            }
            _ => Err(crate::Error::NotImplemented(
                "this storage has no text or sparse search to fuse with".to_string(),
            )),
        }
    }

    /// Store a long text as chunks; see `storage::documents`. Only a storage with server-side
    /// embedding can, and the others say so.
    async fn put_document(
        &self,
        _tablename: &str,
        _doc_id: &str,
        _request: crate::storage::documents::PutDocumentRequest,
    ) -> crate::Result<crate::storage::documents::DocumentWritten> {
        Err(crate::Error::NotImplemented(
            "this storage does not chunk and embed documents".to_string(),
        ))
    }

    /// Delete every chunk of a document; how many there were.
    async fn delete_document(&self, _tablename: &str, _doc_id: &str) -> crate::Result<usize> {
        Err(crate::Error::NotImplemented(
            "this storage does not chunk and embed documents".to_string(),
        ))
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
