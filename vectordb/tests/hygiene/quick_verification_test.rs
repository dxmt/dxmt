// Quick verification test for critical fixes
// This test verifies that the new modules compile and basic functionality works

use dxmt_vectordb::storage::consistency_check::{ConsistencyChecker, RepairReport};
use dxmt_vectordb::storage::wal_entry::{WalEntry, WalOperation};
use dxmt_vectordb::types::record::CreateRecordRequest;
use dxmt_vectordb::types::table::TableConfig;
use dxmt_vectordb::types::vector::DistanceMetric;
use std::collections::HashSet;

#[test]
fn test_wal_entry_creation() {
    let operation = WalOperation::CreateTable {
        config: TableConfig {
            name: "test_table".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            index_type: None,
            description: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        },
    };

    let entry = WalEntry::new(1, operation);
    assert_eq!(entry.sequence_number, 1);
    assert!(entry.timestamp > 0);
}

#[test]
fn test_wal_entry_serialization() {
    let record = CreateRecordRequest {
        id: "test_1".to_string(),
        vector: vec![0.1, 0.2, 0.3],
        sparse_vector: None,
        payload: None,
        named_vectors: None,
        text: None,
    };

    let operation = WalOperation::Insert {
        table_name: "test_table".to_string(),
        record,
    };

    let entry = WalEntry::new(1, operation).with_checksum();

    // Serialize
    let bytes = entry.serialize().expect("Serialization should succeed");
    assert!(!bytes.is_empty());

    // Deserialize
    let recovered = WalEntry::deserialize(&bytes).expect("Deserialization should succeed");
    assert_eq!(recovered.sequence_number, entry.sequence_number);
    assert!(recovered.verify_checksum());
}

#[test]
fn test_consistency_checker_consistent_state() {
    let mut records = HashSet::new();
    records.insert("id1".to_string());
    records.insert("id2".to_string());
    records.insert("id3".to_string());

    let mut index = HashSet::new();
    index.insert("id1".to_string());
    index.insert("id2".to_string());
    index.insert("id3".to_string());

    let report = ConsistencyChecker::build_report("test_table".to_string(), records, index);

    assert!(report.is_consistent);
    assert_eq!(report.total_records, 3);
    assert_eq!(report.indexed_records, 3);
    assert!(report.missing_from_index.is_empty());
    assert!(report.orphaned_in_index.is_empty());
    assert!(!report.has_issues());

    println!("{}", report.summary());
}

#[test]
fn test_consistency_checker_detects_missing() {
    let mut records = HashSet::new();
    records.insert("id1".to_string());
    records.insert("id2".to_string());
    records.insert("id3".to_string());

    let mut index = HashSet::new();
    index.insert("id1".to_string());
    index.insert("id2".to_string());

    let report = ConsistencyChecker::build_report("test_table".to_string(), records, index);

    assert!(!report.is_consistent);
    assert_eq!(report.missing_from_index.len(), 1);
    assert!(report.missing_from_index.contains(&"id3".to_string()));
    assert!(report.has_issues());

    println!("{}", report.summary());
}

#[test]
fn test_consistency_checker_detects_orphaned() {
    let mut records = HashSet::new();
    records.insert("id1".to_string());

    let mut index = HashSet::new();
    index.insert("id1".to_string());
    index.insert("id2".to_string());
    index.insert("id3".to_string());

    let report = ConsistencyChecker::build_report("test_table".to_string(), records, index);

    assert!(!report.is_consistent);
    assert_eq!(report.orphaned_in_index.len(), 2);
    assert!(report.has_issues());

    println!("{}", report.summary());
}

#[test]
fn test_repair_report_creation() {
    let mut report = RepairReport::new("test_table".to_string());
    report.records_added_to_index = 5;
    report.orphaned_entries_removed = 2;
    report = report.mark_success();

    assert!(report.success);
    assert!(report.error_message.is_none());
    assert_eq!(report.records_added_to_index, 5);
    assert_eq!(report.orphaned_entries_removed, 2);

    println!("{}", report.summary());
}

#[test]
fn test_repair_report_failure() {
    let report =
        RepairReport::new("test_table".to_string()).mark_failed("Simulated error".to_string());

    assert!(!report.success);
    assert!(report.error_message.is_some());

    println!("{}", report.summary());
}

#[test]
fn test_batch_wal_operations() {
    let records = vec![
        CreateRecordRequest {
            id: "batch_1".to_string(),
            vector: vec![0.1; 128],
            sparse_vector: None,
            payload: None,
            named_vectors: None,
            text: None,
        },
        CreateRecordRequest {
            id: "batch_2".to_string(),
            vector: vec![0.2; 128],
            sparse_vector: None,
            payload: None,
            named_vectors: None,
            text: None,
        },
    ];

    let operation = WalOperation::BatchInsert {
        table_name: "test_table".to_string(),
        records,
    };

    let entry = WalEntry::new(1, operation).with_checksum();
    let bytes = entry.serialize().expect("Should serialize");
    let recovered = WalEntry::deserialize(&bytes).expect("Should deserialize");

    assert_eq!(recovered.sequence_number, 1);
    assert!(recovered.verify_checksum());
}
