//! Record Iterator - Enables iteration over index records for migration
//!
//! This module provides utilities for iterating over records in indexes,
//! which is essential for data migration operations.

use crate::types::{RecordId, DenseVector};
use crate::error::Result;
use async_trait::async_trait;

/// A record in an index (ID + Vector)
#[derive(Clone, Debug)]
pub struct IndexRecord {
    pub id: RecordId,
    pub vector: DenseVector,
}

/// Iterator over index records
#[async_trait]
pub trait RecordIterator {
    /// Get next batch of records
    /// Returns empty Vec when iteration complete
    async fn next_batch(&mut self, batch_size: usize) -> Result<Vec<IndexRecord>>;

    /// Get total record count
    fn total_count(&self) -> usize;

    /// Get current position
    fn current_position(&self) -> usize;

    /// Reset iterator to beginning
    async fn reset(&mut self) -> Result<()>;
}

/// Builder for creating record iterators
pub struct RecordIteratorBuilder {
    batch_size: usize,
    skip_errors: bool,
}

impl RecordIteratorBuilder {
    pub fn new() -> Self {
        Self {
            batch_size: 1000,
            skip_errors: false,
        }
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn skip_errors(mut self, skip: bool) -> Self {
        self.skip_errors = skip;
        self
    }

    pub fn build(self) -> SimpleRecordIterator {
        SimpleRecordIterator::new(self.batch_size, self.skip_errors)
    }
}

impl Default for RecordIteratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple in-memory record iterator
#[allow(dead_code)]
pub struct SimpleRecordIterator {
    records: Vec<IndexRecord>,
    current_pos: usize,
    batch_size: usize,
    skip_errors: bool,
}

impl SimpleRecordIterator {
    pub fn new(batch_size: usize, skip_errors: bool) -> Self {
        Self {
            records: Vec::new(),
            current_pos: 0,
            batch_size,
            skip_errors,
        }
    }

    /// Add a record to the iterator
    pub fn add_record(&mut self, record: IndexRecord) {
        self.records.push(record);
    }

    /// Add multiple records
    pub fn add_records(&mut self, records: Vec<IndexRecord>) {
        self.records.extend(records);
    }
}

#[async_trait::async_trait]
impl RecordIterator for SimpleRecordIterator {
    async fn next_batch(&mut self, batch_size: usize) -> Result<Vec<IndexRecord>> {
        if self.current_pos >= self.records.len() {
            return Ok(Vec::new());
        }

        let end_pos = std::cmp::min(self.current_pos + batch_size, self.records.len());
        let batch = self.records[self.current_pos..end_pos].to_vec();
        self.current_pos = end_pos;

        Ok(batch)
    }

    fn total_count(&self) -> usize {
        self.records.len()
    }

    fn current_position(&self) -> usize {
        self.current_pos
    }

    async fn reset(&mut self) -> Result<()> {
        self.current_pos = 0;
        Ok(())
    }
}

/// Migration context tracking
#[derive(Debug, Clone)]
pub struct MigrationContext {
    pub total_records: usize,
    pub migrated_records: usize,
    pub failed_records: usize,
    pub skipped_records: usize,
    pub start_time: std::time::Instant,
}

impl MigrationContext {
    pub fn new(total_records: usize) -> Self {
        Self {
            total_records,
            migrated_records: 0,
            failed_records: 0,
            skipped_records: 0,
            start_time: std::time::Instant::now(),
        }
    }

    pub fn record_success(&mut self) {
        self.migrated_records += 1;
    }

    pub fn record_failure(&mut self) {
        self.failed_records += 1;
    }

    pub fn record_skip(&mut self) {
        self.skipped_records += 1;
    }

    pub fn elapsed_ms(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    pub fn progress_percent(&self) -> f64 {
        if self.total_records == 0 {
            100.0
        } else {
            ((self.migrated_records + self.failed_records) as f64 / self.total_records as f64)
                * 100.0
        }
    }

    pub fn throughput_rps(&self) -> f64 {
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        if elapsed_secs > 0.0 {
            self.migrated_records as f64 / elapsed_secs
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_iterator_builder() {
        let iter = RecordIteratorBuilder::new()
            .batch_size(500)
            .skip_errors(true)
            .build();

        assert_eq!(iter.batch_size, 500);
        assert!(iter.skip_errors);
    }

    #[tokio::test]
    async fn test_simple_record_iterator() {
        let mut iter = SimpleRecordIterator::new(10, false);

        // Add test records
        for i in 0..25 {
            let record = IndexRecord {
                id: format!("record_{}", i),
                vector: DenseVector::from(vec![1.0; 10]),
            };
            iter.add_record(record);
        }

        // Get first batch
        let batch1 = iter.next_batch(10).await.unwrap();
        assert_eq!(batch1.len(), 10);

        // Get second batch
        let batch2 = iter.next_batch(10).await.unwrap();
        assert_eq!(batch2.len(), 10);

        // Get remaining batch
        let batch3 = iter.next_batch(10).await.unwrap();
        assert_eq!(batch3.len(), 5);

        // Next should be empty
        let batch4 = iter.next_batch(10).await.unwrap();
        assert_eq!(batch4.len(), 0);
    }

    #[test]
    fn test_migration_context() {
        let mut ctx = MigrationContext::new(100);

        for _ in 0..80 {
            ctx.record_success();
        }
        for _ in 0..10 {
            ctx.record_failure();
        }
        for _ in 0..10 {
            ctx.record_skip();
        }

        assert_eq!(ctx.migrated_records, 80);
        assert_eq!(ctx.failed_records, 10);
        assert_eq!(ctx.skipped_records, 10);
        assert_eq!(ctx.progress_percent(), 90.0);
    }

    #[test]
    fn test_migration_context_throughput() {
        let mut ctx = MigrationContext::new(1000);

        for _ in 0..1000 {
            ctx.record_success();
        }

        let throughput = ctx.throughput_rps();
        assert!(throughput > 0.0, "Throughput should be positive");
    }
}
