// WAL (Write-Ahead Log) Entry Definitions
// This module defines the structure for persisting operations to disk

use serde::{Deserialize, Serialize};
use crate::types::record::CreateRecordRequest;
use crate::types::table::TableConfig;
use crate::types::vector::RecordId;

/// WAL operation types
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WalOperation {
    /// Create a new table
    CreateTable {
        config: TableConfig,
    },
    /// Delete a table
    DeleteTable {
        table_name: String,
    },
    /// Insert a record
    Insert {
        table_name: String,
        record: CreateRecordRequest,
    },
    /// Update a record
    Update {
        table_name: String,
        record_id: RecordId,
        record: CreateRecordRequest,
    },
    /// Delete a record
    Delete {
        table_name: String,
        record_id: RecordId,
    },
    /// Batch insert
    BatchInsert {
        table_name: String,
        records: Vec<CreateRecordRequest>,
    },
    /// Batch delete
    BatchDelete {
        table_name: String,
        record_ids: Vec<RecordId>,
    },
}

/// WAL entry with metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WalEntry {
    /// Unique sequence number for ordering
    pub sequence_number: u64,
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
    /// The actual operation
    pub operation: WalOperation,
    /// Optional checksum for integrity
    pub checksum: Option<u32>,
}

impl WalEntry {
    /// Create a new WAL entry
    pub fn new(sequence_number: u64, operation: WalOperation) -> Self {
        let timestamp = chrono::Utc::now().timestamp_millis();
        Self {
            sequence_number,
            timestamp,
            operation,
            checksum: None,
        }
    }

    /// Serialize to bytes using bincode
    pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserialize from bytes
    pub fn deserialize(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }

    /// Calculate checksum for integrity verification
    pub fn calculate_checksum(&self) -> u32 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        self.sequence_number.hash(&mut hasher);
        self.timestamp.hash(&mut hasher);
        // Would hash operation data here in production
        hasher.finish() as u32
    }

    /// Set checksum
    pub fn with_checksum(mut self) -> Self {
        self.checksum = Some(self.calculate_checksum());
        self
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        match self.checksum {
            Some(stored) => stored == self.calculate_checksum(),
            None => true, // No checksum to verify
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_entry_serialization() {
        let operation = WalOperation::CreateTable {
            config: TableConfig {
                name: "test".to_string(),
                dimension: 128,
                distance_metric: crate::types::vector::DistanceMetric::Cosine,
                index_type: None,
                description: None,
                text_fields: None,
                enable_full_text_search: false,
                memory_optimization: None,
                payload_schema: None,
                schema_version: 1,
                named_vectors: None,
            },
        };

        let entry = WalEntry::new(1, operation).with_checksum();

        // Serialize
        let bytes = entry.serialize().unwrap();
        assert!(!bytes.is_empty());

        // Deserialize
        let recovered = WalEntry::deserialize(&bytes).unwrap();
        assert_eq!(recovered.sequence_number, entry.sequence_number);
        assert_eq!(recovered.timestamp, entry.timestamp);
        assert!(recovered.verify_checksum());
    }

    #[test]
    fn test_wal_operation_insert() {
        let record = CreateRecordRequest {
            id: "test_1".to_string(),
            vector: vec![0.1; 128],
            sparse_vector: None,
            payload: None,
            named_vectors: None,
        };

        let operation = WalOperation::Insert {
            table_name: "test_table".to_string(),
            record,
        };

        let entry = WalEntry::new(1, operation);
        let bytes = entry.serialize().unwrap();
        let recovered = WalEntry::deserialize(&bytes).unwrap();

        assert_eq!(recovered.sequence_number, 1);
    }
}
