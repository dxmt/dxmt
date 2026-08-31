// Index Consistency Verification and Repair
// This module provides tools to detect and fix inconsistencies between
// the vector index and stored records

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Consistency check report
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConsistencyReport {
    /// Name of the checked table
    pub table_name: String,
    /// Total number of records in storage
    pub total_records: usize,
    /// Total number of entries in index
    pub indexed_records: usize,
    /// Records that exist but are not in the index
    pub missing_from_index: Vec<String>,
    /// Index entries that don't have corresponding records
    pub orphaned_in_index: Vec<String>,
    /// Whether the table is consistent
    pub is_consistent: bool,
    /// Timestamp of the check
    pub checked_at: String,
}

impl ConsistencyReport {
    /// Create a new consistency report
    pub fn new(table_name: String) -> Self {
        Self {
            table_name,
            total_records: 0,
            indexed_records: 0,
            missing_from_index: Vec::new(),
            orphaned_in_index: Vec::new(),
            is_consistent: true,
            checked_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Check if there are any issues
    pub fn has_issues(&self) -> bool {
        !self.missing_from_index.is_empty() || !self.orphaned_in_index.is_empty()
    }

    /// Get summary message
    pub fn summary(&self) -> String {
        if self.is_consistent {
            format!(
                "✅ Table '{}' is consistent: {} records, {} indexed",
                self.table_name, self.total_records, self.indexed_records
            )
        } else {
            format!(
                "❌ Table '{}' has issues: {} missing from index, {} orphaned",
                self.table_name,
                self.missing_from_index.len(),
                self.orphaned_in_index.len()
            )
        }
    }
}

/// Repair operation report
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RepairReport {
    /// Name of the repaired table
    pub table_name: String,
    /// Number of records added to index
    pub records_added_to_index: usize,
    /// Number of orphaned entries removed from index
    pub orphaned_entries_removed: usize,
    /// Whether repair was successful
    pub success: bool,
    /// Error message if repair failed
    pub error_message: Option<String>,
    /// Timestamp of repair
    pub repaired_at: String,
}

impl RepairReport {
    /// Create a new repair report
    pub fn new(table_name: String) -> Self {
        Self {
            table_name,
            records_added_to_index: 0,
            orphaned_entries_removed: 0,
            success: false,
            error_message: None,
            repaired_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Mark as successful
    pub fn mark_success(mut self) -> Self {
        self.success = true;
        self
    }

    /// Mark as failed with error
    pub fn mark_failed(mut self, error: String) -> Self {
        self.success = false;
        self.error_message = Some(error);
        self
    }

    /// Get summary message
    pub fn summary(&self) -> String {
        if self.success {
            format!(
                "✅ Repaired '{}': +{} to index, -{} orphaned",
                self.table_name, self.records_added_to_index, self.orphaned_entries_removed
            )
        } else {
            format!(
                "❌ Repair failed for '{}': {}",
                self.table_name,
                self.error_message
                    .as_ref()
                    .unwrap_or(&"Unknown error".to_string())
            )
        }
    }
}

/// Consistency checker helper
pub struct ConsistencyChecker;

impl ConsistencyChecker {
    /// Compare two sets and find differences
    pub fn compare_sets(
        record_ids: HashSet<String>,
        index_ids: HashSet<String>,
    ) -> (Vec<String>, Vec<String>) {
        let missing_from_index: Vec<String> = record_ids.difference(&index_ids).cloned().collect();

        let orphaned_in_index: Vec<String> = index_ids.difference(&record_ids).cloned().collect();

        (missing_from_index, orphaned_in_index)
    }

    /// Build consistency report from sets
    pub fn build_report(
        table_name: String,
        record_ids: HashSet<String>,
        index_ids: HashSet<String>,
    ) -> ConsistencyReport {
        let (missing_from_index, orphaned_in_index) =
            Self::compare_sets(record_ids.clone(), index_ids.clone());

        let is_consistent = missing_from_index.is_empty() && orphaned_in_index.is_empty();

        ConsistencyReport {
            table_name,
            total_records: record_ids.len(),
            indexed_records: index_ids.len(),
            missing_from_index,
            orphaned_in_index,
            is_consistent,
            checked_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistency_report_consistent() {
        let mut records = HashSet::new();
        records.insert("id1".to_string());
        records.insert("id2".to_string());

        let mut index = HashSet::new();
        index.insert("id1".to_string());
        index.insert("id2".to_string());

        let report = ConsistencyChecker::build_report("test_table".to_string(), records, index);

        assert!(report.is_consistent);
        assert_eq!(report.total_records, 2);
        assert_eq!(report.indexed_records, 2);
        assert!(report.missing_from_index.is_empty());
        assert!(report.orphaned_in_index.is_empty());
        assert!(!report.has_issues());
    }

    #[test]
    fn test_consistency_report_missing_from_index() {
        let mut records = HashSet::new();
        records.insert("id1".to_string());
        records.insert("id2".to_string());
        records.insert("id3".to_string());

        let mut index = HashSet::new();
        index.insert("id1".to_string());
        index.insert("id2".to_string());

        let report = ConsistencyChecker::build_report("test_table".to_string(), records, index);

        assert!(!report.is_consistent);
        assert_eq!(report.total_records, 3);
        assert_eq!(report.indexed_records, 2);
        assert_eq!(report.missing_from_index.len(), 1);
        assert!(report.missing_from_index.contains(&"id3".to_string()));
        assert!(report.has_issues());
    }

    #[test]
    fn test_consistency_report_orphaned() {
        let mut records = HashSet::new();
        records.insert("id1".to_string());

        let mut index = HashSet::new();
        index.insert("id1".to_string());
        index.insert("id2".to_string());
        index.insert("id3".to_string());

        let report = ConsistencyChecker::build_report("test_table".to_string(), records, index);

        assert!(!report.is_consistent);
        assert_eq!(report.total_records, 1);
        assert_eq!(report.indexed_records, 3);
        assert_eq!(report.orphaned_in_index.len(), 2);
        assert!(report.has_issues());
    }

    #[test]
    fn test_repair_report() {
        let mut report = RepairReport::new("test_table".to_string());
        report.records_added_to_index = 5;
        report.orphaned_entries_removed = 2;
        report = report.mark_success();

        assert!(report.success);
        assert!(report.error_message.is_none());
        assert_eq!(report.records_added_to_index, 5);
    }
}
