//! TransactionLog - Durable logging for index switching operations
//!
//! Responsible for:
//! - Recording all switching operations
//! - Enabling recovery after failures
//! - Durability guarantees
//! - Crash recovery procedures

use crate::error::{Error, Result};
use crate::types::table::IndexType;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Type of transaction log entry
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogEntryType {
    BeginSwitch,
    DataMigrated,
    SwitchCommitted,
    SwitchRolledBack,
    Checkpoint,
}

/// A single entry in the transaction log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub entry_type: LogEntryType,
    pub timestamp: String,
    pub from_type: Option<IndexType>,
    pub to_type: Option<IndexType>,
    pub records_migrated: Option<usize>,
    pub reason: Option<String>,
}

impl LogEntry {
    fn new(entry_type: LogEntryType) -> Self {
        Self {
            entry_type,
            timestamp: Utc::now().to_rfc3339(),
            from_type: None,
            to_type: None,
            records_migrated: None,
            reason: None,
        }
    }

    fn with_types(mut self, from: &IndexType, to: &IndexType) -> Self {
        self.from_type = Some(*from);
        self.to_type = Some(*to);
        self
    }

    fn with_records(mut self, count: usize) -> Self {
        self.records_migrated = Some(count);
        self
    }

    fn with_reason(mut self, reason: String) -> Self {
        self.reason = Some(reason);
        self
    }
}

/// Transaction log for durability and recovery
pub struct TransactionLog {
    /// In-memory entries (write-ahead log)
    entries: Arc<RwLock<Vec<LogEntry>>>,

    /// Path to WAL file
    wal_path: PathBuf,

    /// Whether writes are durable (written to disk)
    enable_durability: bool,
}

impl TransactionLog {
    /// Create a new TransactionLog
    pub fn new(wal_path: PathBuf, enable_durability: bool) -> Self {
        Self {
            entries: Arc::new(RwLock::new(Vec::new())),
            wal_path,
            enable_durability,
        }
    }

    /// Log the beginning of an index switch
    pub async fn log_begin_switch(&self, from: &IndexType, to: &IndexType) -> Result<()> {
        let entry = LogEntry::new(LogEntryType::BeginSwitch).with_types(from, to);
        self.log_entry(entry).await
    }

    /// Log that data migration completed
    pub async fn log_data_migrated(&self, records: usize) -> Result<()> {
        let entry = LogEntry::new(LogEntryType::DataMigrated).with_records(records);
        self.log_entry(entry).await
    }

    /// Log that switch was committed
    pub async fn log_switch_committed(&self, new_type: &IndexType) -> Result<()> {
        let mut entry = LogEntry::new(LogEntryType::SwitchCommitted);
        entry.to_type = Some(*new_type);
        self.log_entry(entry).await
    }

    /// Log that switch was rolled back
    pub async fn log_switch_rolled_back(&self, reason: String) -> Result<()> {
        let entry = LogEntry::new(LogEntryType::SwitchRolledBack).with_reason(reason);
        self.log_entry(entry).await
    }

    /// Log a checkpoint
    pub async fn log_checkpoint(&self) -> Result<()> {
        let entry = LogEntry::new(LogEntryType::Checkpoint);
        self.log_entry(entry).await
    }

    /// Internal method to log an entry
    async fn log_entry(&self, entry: LogEntry) -> Result<()> {
        // Write to in-memory log
        let mut entries = self.entries.write().await;
        entries.push(entry.clone());

        // Write to disk if durability is enabled
        if self.enable_durability {
            self.write_to_disk(&entry).await?;
        }

        Ok(())
    }

    /// Write entry to disk (Write-Ahead Log)
    async fn write_to_disk(&self, entry: &LogEntry) -> Result<()> {
        use tokio::fs::OpenOptions;
        use tokio::io::AsyncWriteExt;

        // Step 1: Serialize entry to JSON
        let json_entry = serde_json::to_string(entry)
            .map_err(|e| Error::Serialization(format!("Failed to serialize log entry: {}", e)))?;

        // Step 2: Append to WAL file with newline separator
        use tokio::fs::File;
        let mut file: File = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal_path)
            .await
            .map_err(|e| Error::internal(format!("Failed to open WAL file: {}", e)))?;

        // Write entry as JSON line (JSON Lines format)
        file.write_all(json_entry.as_bytes())
            .await
            .map_err(|e| Error::internal(format!("Failed to write to WAL: {}", e)))?;

        file.write_all(b"\n")
            .await
            .map_err(|e| Error::internal(format!("Failed to write newline to WAL: {}", e)))?;

        // Step 3: Fsync for durability
        file.sync_all()
            .await
            .map_err(|e| Error::internal(format!("Failed to sync WAL file: {}", e)))?;

        Ok(())
    }

    /// Read all entries from log
    pub async fn read_entries(&self) -> Result<Vec<LogEntry>> {
        let entries = self.entries.read().await;
        Ok(entries.clone())
    }

    /// Recover from a previous switch operation
    ///
    /// This reads the log to determine:
    /// - Was a switch in progress?
    /// - Was it committed?
    /// - What needs to be recovered?
    pub async fn recover(&self) -> Result<Option<SwitchRecovery>> {
        let entries = self.read_entries().await?;

        // Find the most recent BeginSwitch
        for i in (0..entries.len()).rev() {
            if entries[i].entry_type == LogEntryType::BeginSwitch {
                // Check if there is a SwitchCommitted anywhere after this entry
                let committed = entries[i + 1..]
                    .iter()
                    .any(|e| e.entry_type == LogEntryType::SwitchCommitted);

                if !committed {
                    return Ok(Some(SwitchRecovery {
                        from_type: entries[i].from_type,
                        to_type: entries[i].to_type,
                        completed: committed,
                    }));
                }
            }
        }

        Ok(None)
    }

    /// Clear old entries (checkpoint)
    pub async fn truncate_before(&self, entry_count: usize) -> Result<()> {
        let mut entries = self.entries.write().await;
        let len = entries.len();
        if len > entry_count {
            entries.drain(0..len - entry_count);
            drop(entries); // Release lock before calling log_checkpoint
            self.log_checkpoint().await?;
        }
        Ok(())
    }

    /// Load entries from disk (for recovery on startup)
    pub async fn load_from_disk(&self) -> Result<()> {
        use tokio::fs;
        use tokio::io::{AsyncBufReadExt, BufReader};

        // Check if WAL file exists
        if !fs::try_exists(&self.wal_path).await.unwrap_or(false) {
            return Ok(()); // No WAL file, start fresh
        }

        // Open and read WAL file
        let file = fs::File::open(&self.wal_path)
            .await
            .map_err(|e| Error::internal(format!("Failed to open WAL file for recovery: {}", e)))?;

        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read entries line by line (JSON Lines format)
        let mut loaded_entries = Vec::new();
        loop {
            let line = match lines.next_line().await {
                Ok(Some(l)) => l,
                Ok(None) => break,
                Err(e) => return Err(Error::internal(format!("Failed to read WAL file: {}", e))),
            };

            if line.is_empty() {
                continue; // Skip empty lines
            }

            let entry: LogEntry = serde_json::from_str(&line)
                .map_err(|e| Error::internal(format!("Failed to deserialize log entry: {}", e)))?;

            loaded_entries.push(entry);
        }

        // Update in-memory log with loaded entries
        let mut entries = self.entries.write().await;
        *entries = loaded_entries;

        Ok(())
    }
}

/// Information about a switch that needs recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchRecovery {
    pub from_type: Option<IndexType>,
    pub to_type: Option<IndexType>,
    pub completed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_log_creation() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        assert_eq!(log.read_entries().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_log_begin_switch() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        let from = IndexType::Hnsw;
        let to = IndexType::Ivf;

        log.log_begin_switch(&from, &to).await.unwrap();
        let entries = log.read_entries().await.unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_type, LogEntryType::BeginSwitch);
    }

    #[tokio::test]
    async fn test_recovery() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        let from = IndexType::Hnsw;
        let to = IndexType::Ivf;

        log.log_begin_switch(&from, &to).await.unwrap();
        log.log_data_migrated(100).await.unwrap();
        // Note: didn't call log_switch_committed, so recovery should show incomplete

        let recovery = log.recover().await.unwrap();
        assert!(recovery.is_some());
        let recovery = recovery.unwrap();
        assert!(!recovery.completed);
    }

    #[tokio::test]
    async fn test_checkpoint() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        let from = IndexType::Hnsw;
        let to = IndexType::Ivf;

        log.log_begin_switch(&from, &to).await.unwrap();
        log.log_data_migrated(100).await.unwrap();
        log.log_switch_committed(&to).await.unwrap();
        log.log_checkpoint().await.unwrap();

        let entries = log.read_entries().await.unwrap();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[3].entry_type, LogEntryType::Checkpoint);
    }
}
