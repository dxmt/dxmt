use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::info;

use crate::storage::traits::_Storage;
use crate::types::{
    record::{CreateRecordRequest, Record, SearchHit, SearchQuery, SearchResult},
    table::{Table, TableConfig},
    vector::{DenseVector, RecordId},
};

/// 内存存储引擎
#[derive(Debug)]
pub struct MemoryStorage {
    tables: Arc<RwLock<HashMap<String, TableData>>>,
}

/// 表数据容器
#[derive(Debug)]
struct TableData {
    table: Table,
    records: HashMap<RecordId, Record>,
}

impl MemoryStorage {
    /// 创建新的内存存储实例
    pub fn new() -> Self {
        Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 创建表
    pub fn create_table(&self, config: TableConfig) -> crate::Result<Table> {
        let mut tables = self
            .tables
            .write()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        if tables.contains_key(&config.name) {
            return Err(crate::Error::TableAlreadyExists(config.name.clone()));
        }

        let table = Table::new(config)?;
        let table_data = TableData {
            table: table.clone(),
            records: HashMap::new(),
        };

        tables.insert(table.name.clone(), table_data);
        info!("Created table: {}", table.name);

        Ok(table)
    }

    /// 获取表信息
    pub fn get_table(&self, name: &str) -> crate::Result<Table> {
        let tables = self
            .tables
            .read()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        tables
            .get(name)
            .map(|data| data.table.clone())
            .ok_or_else(|| crate::Error::TableNotFound(name.to_string()))
    }

    /// 列出所有表
    pub fn list_tables(&self) -> crate::Result<Vec<Table>> {
        let tables = self
            .tables
            .read()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        Ok(tables.values().map(|data| data.table.clone()).collect())
    }

    /// 删除表
    pub fn delete_table(&self, name: &str) -> crate::Result<()> {
        let mut tables = self
            .tables
            .write()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        if tables.remove(name).is_some() {
            info!("Deleted table: {}", name);
            Ok(())
        } else {
            Err(crate::Error::TableNotFound(name.to_string()))
        }
    }

    /// 插入记录
    pub fn insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<RecordId>> {
        let mut tables = self
            .tables
            .write()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        let table_data = tables
            .get_mut(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let mut inserted_ids = Vec::new();

        for request in requests {
            let record = Record::new(request.clone(), table_data.table.dimension)?;
            table_data.records.insert(record.id.clone(), record.clone());
            inserted_ids.push(record.id);
        }

        // 更新记录计数
        table_data
            .table
            .update_record_count(table_data.records.len() as u64);

        info!(
            "Inserted {} records into table: {}",
            inserted_ids.len(),
            tablename
        );
        Ok(inserted_ids)
    }

    /// 获取单个记录
    pub fn get_record(&self, tablename: &str, id: &RecordId) -> crate::Result<Record> {
        let tables = self
            .tables
            .read()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        table_data
            .records
            .get(id)
            .cloned()
            .ok_or_else(|| crate::Error::RecordNotFound(id.to_string()))
    }

    /// 批量获取记录
    pub fn get_records(&self, tablename: &str, ids: &[RecordId]) -> crate::Result<Vec<Record>> {
        let tables = self
            .tables
            .read()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let mut records = Vec::new();
        for id in ids {
            if let Some(record) = table_data.records.get(id) {
                records.push(record.clone())
            }
        }

        Ok(records)
    }

    /// 更新记录
    pub fn update_record(
        &self,
        tablename: &str,
        id: &RecordId,
        vector: Option<DenseVector>,
        payload: Option<serde_json::Value>,
    ) -> crate::Result<()> {
        let mut tables = self
            .tables
            .write()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        let table_data = tables
            .get_mut(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let record = table_data
            .records
            .get_mut(id)
            .ok_or_else(|| crate::Error::RecordNotFound(id.to_string()))?;

        record.update(vector, None, payload, table_data.table.dimension, None)?;

        info!("Updated record {} in table: {}", id, tablename);
        Ok(())
    }

    /// 删除记录
    pub fn delete_records(&self, tablename: &str, ids: &[RecordId]) -> crate::Result<usize> {
        let mut tables = self
            .tables
            .write()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        let table_data = tables
            .get_mut(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        let mut deleted_count = 0;
        for id in ids {
            if table_data.records.remove(id).is_some() {
                deleted_count += 1
            }
        }

        // 更新记录计数
        table_data
            .table
            .update_record_count(table_data.records.len() as u64);

        info!(
            "Deleted {} records from table: {}",
            deleted_count, tablename
        );
        Ok(deleted_count)
    }

    /// 搜索向量
    pub fn search(&self, tablename: &str, query: SearchQuery) -> crate::Result<SearchResult> {
        let start_time = std::time::Instant::now();
        let tables = self
            .tables
            .read()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        let table_data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;

        // 验证查询向量维度
        if query.vector.len() != table_data.table.dimension {
            return Err(crate::Error::DimensionMismatch {
                expected: table_data.table.dimension,
                actual: query.vector.len(),
            });
        }

        let mut candidates: Vec<(RecordId, f32)> = Vec::new();

        // 暴力搜索
        for (id, record) in &table_data.records {
            // 应用过滤条件
            if let Some(filter) = &query.filter {
                if !record.matches_filter(filter) {
                    continue;
                }
            }

            // 计算距离
            let distance = table_data
                .table
                .distance_metric
                .calculate(&query.vector, &record.vector)?;

            candidates.push((id.clone(), distance));
        }

        // 排序并取top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(query.top_k);

        // 构建结果
        let mut results = Vec::new();
        for (id, score) in candidates {
            let record = &table_data.records[&id];
            let hit = SearchHit {
                id: id.clone(),
                score,
                payload: if query.with_payload {
                    Some(record.payload.clone())
                } else {
                    None
                },
                vector: if query.with_vector {
                    Some(record.vector.clone())
                } else {
                    None
                },
            };
            results.push(hit);
        }

        let query_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(SearchResult {
            total: results.len(),
            hits: results,
            tookms: query_time_ms,
        })
    }

    /// 获取存储统计信息
    pub fn get_stats(&self) -> crate::Result<StorageStats> {
        let tables = self
            .tables
            .read()
            .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

        let total_tables = tables.len();
        let total_records: u64 = tables.values().map(|data| data.records.len() as u64).sum();

        let memory_usage = self.estimate_memory_usage(&tables);

        Ok(StorageStats {
            total_tables,
            total_records,
            memory_usagebytes: memory_usage,
        })
    }

    /// 估算内存使用量
    fn estimate_memory_usage(&self, tables: &HashMap<String, TableData>) -> u64 {
        let mut total_size = 0u64;

        for table_data in tables.values() {
            // 表元数据大小
            total_size += 1000; // 粗略估算

            // 记录数据大小
            for record in table_data.records.values() {
                total_size += record.vector.len() as u64 * 4; // f32大小
                total_size += record.payload.to_string().len() as u64; // 粗略估算payload大小
                total_size += record.id.len() as u64
            }
        }

        total_size
    }
}

/// 存储统计信息
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_tables: usize,
    pub total_records: u64,
    pub memory_usagebytes: u64,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

// Implement the _Storage trait for MemoryStorage
#[async_trait]
impl _Storage for MemoryStorage {
    async fn create_table(&self, config: TableConfig) -> crate::Result<Table> {
        Ok(self.create_table(config)?)
    }

    async fn get_table(&self, name: &str) -> crate::Result<Table> {
        Ok(self.get_table(name)?)
    }

    async fn list_tables(&self) -> crate::Result<Vec<Table>> {
        Ok(self.list_tables()?)
    }

    async fn delete_table(&self, name: &str) -> crate::Result<()> {
        Ok(self.delete_table(name)?)
    }

    async fn insert_record(
        &self,
        tablename: &str,
        request: CreateRecordRequest,
    ) -> crate::Result<Record> {
        let ids = self.insert_records(tablename, vec![request.clone()])?;
        let id = ids
            .first()
            .ok_or_else(|| crate::Error::Internal("Failed to insert record".to_string()))?;
        self.get_record(tablename, id)
    }

    async fn batch_insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>> {
        let ids = self.insert_records(tablename, requests)?;
        Ok(self.get_records(tablename, &ids)?)
    }

    async fn search(&self, tablename: &str, query: SearchQuery) -> crate::Result<SearchResult> {
        Ok(self.search(tablename, query)?)
    }

    async fn get_record(&self, tablename: &str, record_id: &RecordId) -> crate::Result<Record> {
        Ok(self.get_record(tablename, record_id)?)
    }

    async fn update_record(
        &self,
        tablename: &str,
        record_id: &RecordId,
        request: CreateRecordRequest,
    ) -> crate::Result<Record> {
        self.update_record(tablename, record_id, Some(request.vector), request.payload)?;
        self.get_record(tablename, record_id)
    }

    async fn delete_record(&self, tablename: &str, record_id: &RecordId) -> crate::Result<()> {
        self.delete_records(tablename, std::slice::from_ref(record_id))?;
        Ok(())
    }

    async fn insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>> {
        self.batch_insert_records(tablename, requests).await
    }

    async fn delete_records(
        &self,
        tablename: &str,
        record_ids: Vec<RecordId>,
    ) -> crate::Result<()> {
        let _ = self.delete_records(tablename, &record_ids)?;
        Ok(())
    }

    async fn backup(&self, tablename: &str, backup_path: &str) -> crate::Result<()> {
        #[derive(Serialize)]
        struct TableSnapshot<'a> {
            table: &'a Table,
            records: Vec<&'a Record>,
        }

        let tables = self
            .tables
            .read()
            .map_err(|_| crate::Error::Internal("lock poisoned".to_string()))?;
        let data = tables
            .get(tablename)
            .ok_or_else(|| crate::Error::TableNotFound(tablename.to_string()))?;
        let snapshot = TableSnapshot {
            table: &data.table,
            records: data.records.values().collect(),
        };
        let json = serde_json::to_vec(&snapshot)
            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
        std::fs::write(backup_path, json)
            .map_err(|e| crate::Error::Internal(format!("backup write failed: {}", e)))?;
        Ok(())
    }

    async fn restore(&self, tablename: &str, backup_path: &str) -> crate::Result<()> {
        #[derive(Deserialize)]
        struct TableSnapshot {
            table: Table,
            records: Vec<Record>,
        }

        let json = std::fs::read(backup_path)
            .map_err(|e| crate::Error::Internal(format!("backup read failed: {}", e)))?;
        let snapshot: TableSnapshot = serde_json::from_slice(&json)
            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
        let mut tables = self
            .tables
            .write()
            .map_err(|_| crate::Error::Internal("lock poisoned".to_string()))?;
        let record_map: HashMap<RecordId, Record> = snapshot
            .records
            .into_iter()
            .map(|r| (r.id.clone(), r))
            .collect();
        tables.insert(
            tablename.to_string(),
            TableData {
                table: snapshot.table,
                records: record_map,
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::table::TableConfig;
    use crate::types::{table::IndexType, DistanceMetric};
    use crate::CreateRecordRequest;
    use crate::SearchQuery;

    fn create_test_table_config() -> TableConfig {
        TableConfig {
            name: "test_table".to_string(),
            dimension: 3,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Test table".to_string()),
            index_type: Some(IndexType::BruteForce),
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
        }
    }

    #[test]
    fn test_table_creation() -> Result<(), Box<dyn std::error::Error>> {
        let storage = MemoryStorage::new();
        let config = create_test_table_config();

        let table = storage.create_table(config)?;
        assert_eq!(table.name, "test_table");
        assert_eq!(table.dimension, 3);
        Ok(())
    }

    #[test]
    fn test_duplicate_table_creation() -> Result<(), Box<dyn std::error::Error>> {
        let storage = MemoryStorage::new();
        let config = create_test_table_config();

        storage.create_table(config.clone())?;
        let result = storage.create_table(config);

        assert!(matches!(result, Err(crate::Error::TableAlreadyExists(_))));
        Ok(())
    }

    #[test]
    fn test_record_operations() -> Result<(), Box<dyn std::error::Error>> {
        let storage = MemoryStorage::new();
        let config = create_test_table_config();

        storage.create_table(config)?;

        let request = CreateRecordRequest {
            id: "test_record".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            sparse_vector: None,
            payload: Some(serde_json::json!({"category": "test"})),
            named_vectors: None,
        };

        // 插入记录
        let ids = storage.insert_records("test_table", vec![request])?;
        assert_eq!(ids.len(), 1);

        // 获取记录
        let record = storage.get_record("test_table", &ids[0])?;
        assert_eq!(record.id.to_string(), "test_record");
        assert_eq!(record.vector, vec![0.1, 0.2, 0.3]);

        // 删除记录
        let deleted = storage.delete_records("test_table", &ids)?;
        assert_eq!(deleted, 1);
        Ok(())
    }

    #[test]
    fn test_vector_search() -> Result<(), Box<dyn std::error::Error>> {
        let storage = MemoryStorage::new();
        let config = create_test_table_config();

        storage.create_table(config)?;

        // 插入测试数据
        let requests = vec![
            CreateRecordRequest {
                id: "record1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                sparse_vector: None,
                payload: Some(serde_json::json!({"category": "A"})),
                named_vectors: None,
            },
            CreateRecordRequest {
                id: "record2".to_string(),
                vector: vec![0.0, 1.0, 0.0],
                sparse_vector: None,
                payload: Some(serde_json::json!({"category": "B"})),
                named_vectors: None,
            },
        ];

        storage.insert_records("test_table", requests)?;

        // 搜索
        let query = SearchQuery {
            vector: vec![1.0, 0.0, 0.0],
            top_k: 1,
            filter: None,
            with_payload: true,
            with_vector: false,
            limit: 1,
            ..Default::default()
        };

        let result = storage.search("test_table", query)?;
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].id.to_string(), "record1");
        Ok(())
    }
}
