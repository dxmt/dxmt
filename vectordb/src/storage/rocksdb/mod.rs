//! RocksDB storage backend for vector database
//!
//! Provides persistent storage using RocksDB as the backend engine.

use rocksdb::{DB, Options};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

use crate::types::RecordId;

const CF_VECTORS: &str = "vectors";
const CF_METADATA: &str = "metadata";
const CF_PAYLOADS: &str = "payloads";

#[derive(Clone)]
pub struct RocksDbStorage {
    db: Arc<DB>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: RecordId,
    pub vector: Vec<f32>,
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TableMetadata {
    pub name: String,
    pub dimension: usize,
    pub total_records: usize,
}

impl RocksDbStorage {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cfs = vec![CF_VECTORS, CF_METADATA, CF_PAYLOADS];
        let db = DB::open_cf_descriptors(&opts, path, cfs.iter().map(|cf| {
            (cf.to_string(), Options::default())
        }).collect())?;

        Ok(Self {
            db: Arc::new(db),
        })
    }

    pub fn insert_vector(&self, record: &VectorRecord) -> Result<(), Box<dyn std::error::Error>> {
        let cf = self.db.cf_handle(CF_VECTORS).ok_or("CF_VECTORS not found")?;
        let key = record.id.as_bytes();
        let value = bincode::serialize(record)?;
        self.db.put_cf(cf, key, &value)?;

        if let Some(ref payload) = record.payload {
            let cf_payloads = self.db.cf_handle(CF_PAYLOADS).ok_or("CF_PAYLOADS not found")?;
            let payload_value = serde_json::to_vec(payload)?;
            self.db.put_cf(cf_payloads, key, &payload_value)?;
        }

        Ok(())
    }

    pub fn get_vector(&self, id: &RecordId) -> Result<Option<VectorRecord>, Box<dyn std::error::Error>> {
        let cf = self.db.cf_handle(CF_VECTORS).ok_or("CF_VECTORS not found")?;
        let key = id.as_bytes();

        if let Some(value) = self.db.get_cf(cf, key)? {
            let record: VectorRecord = bincode::deserialize(&value)?;
            Ok(Some(record))
        } else {
            Ok(None)
        }
    }

    pub fn delete_vector(&self, id: &RecordId) -> Result<bool, Box<dyn std::error::Error>> {
        let cf = self.db.cf_handle(CF_VECTORS).ok_or("CF_VECTORS not found")?;
        let key = id.as_bytes();

        self.db.delete_cf(cf, key)?;

        if let Ok(cf_payloads) = self.db.cf_handle(CF_PAYLOADS).ok_or("CF_PAYLOADS not found") {
            let _ = self.db.delete_cf(cf_payloads, key);
        }

        Ok(true)
    }

    pub fn update_table_metadata(&self, table_name: &str, metadata: &TableMetadata) -> Result<(), Box<dyn std::error::Error>> {
        let cf = self.db.cf_handle(CF_METADATA).ok_or("CF_METADATA not found")?;
        let key = table_name.as_bytes();
        let value = bincode::serialize(metadata)?;
        self.db.put_cf(cf, key, &value)?;
        Ok(())
    }

    pub fn get_table_metadata(&self, table_name: &str) -> Result<Option<TableMetadata>, Box<dyn std::error::Error>> {
        let cf = self.db.cf_handle(CF_METADATA).ok_or("CF_METADATA not found")?;
        let key = table_name.as_bytes();

        if let Some(value) = self.db.get_cf(cf, key)? {
            let metadata: TableMetadata = bincode::deserialize(&value)?;
            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }

    pub fn compact(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.db.compact_range(None::<&[u8]>, None::<&[u8]>);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_rocksdb_basic_operations() {
        let dir = tempdir().unwrap();
        let storage = RocksDbStorage::new(dir.path()).unwrap();

        let record = VectorRecord {
            id: uuid::Uuid::new_v4().to_string(),
            vector: vec![1.0, 2.0, 3.0],
            payload: Some(serde_json::json!({"name": "test"})),
        };

        storage.insert_vector(&record).unwrap();

        let retrieved = storage.get_vector(&record.id).unwrap();
        assert!(retrieved.is_some());

        let deleted = storage.delete_vector(&record.id).unwrap();
        assert!(deleted);

        let retrieved = storage.get_vector(&record.id).unwrap();
        assert!(retrieved.is_none());
    }
}
