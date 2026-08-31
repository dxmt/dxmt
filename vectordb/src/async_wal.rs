//! Async Write-Ahead Log (WAL) implementation
//!
//! Provides asynchronous WAL functionality with background flushing
//! to minimize write latency and improve throughput.

use crate::error::{Error, Result};
use crate::wal::WalEntry;
use std::path::{Path, PathBuf};
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, oneshot};

/// Async WAL write request
enum WalRequest {
    /// Write a single entry
    WriteEntry {
        entry: WalEntry,
        response: oneshot::Sender<Result<u64>>,
    },
    /// Write multiple entries in batch
    WriteBatch {
        entries: Vec<WalEntry>,
        response: oneshot::Sender<Result<Vec<u64>>>,
    },
    /// Force flush pending writes
    ForceFlush {
        response: oneshot::Sender<Result<()>>,
    },
    /// Shutdown the writer
    Shutdown,
}

/// Async Write-Ahead Log manager
///
/// Uses a background task to handle WAL writes asynchronously,
/// reducing write latency and improving throughput.
#[derive(Clone)]
pub struct AsyncWalManager {
    /// Channel to send write requests to background task
    sender: mpsc::UnboundedSender<WalRequest>,
}

/// Internal state for the async WAL writer
struct WalWriter {
    wal_path: PathBuf,
    current_file: Option<File>,
    sequence_number: u64,
    write_buffer: Vec<u8>,
    batch_threshold: usize,
    compression_enabled: bool,
    compression_level: i32,
}

impl AsyncWalManager {
    /// Create a new async WAL manager
    pub async fn new<P: AsRef<Path>>(wal_dir: P) -> Result<Self> {
        Self::new_with_compression(wal_dir, false, 3).await
    }

    /// Create a new async WAL manager with compression settings
    pub async fn new_with_compression<P: AsRef<Path>>(
        wal_dir: P,
        enable_compression: bool,
        compression_level: i32,
    ) -> Result<Self> {
        let wal_path = wal_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        tokio::fs::create_dir_all(&wal_path)
            .await
            .map_err(Error::Io)?;

        // Create channel for async communication
        let (sender, receiver) = mpsc::unbounded_channel();

        // Create writer state
        let mut writer = WalWriter {
            wal_path: wal_path.clone(),
            current_file: None,
            sequence_number: 0,
            write_buffer: Vec::with_capacity(1024 * 1024), // 1MB buffer
            batch_threshold: 256 * 1024,                   // 256KB threshold
            compression_enabled: enable_compression,
            compression_level,
        };

        // Open the WAL file
        writer.open_current_file().await?;

        // Spawn background task to handle writes
        tokio::spawn(async move {
            writer.run(receiver).await;
        });

        Ok(AsyncWalManager { sender })
    }

    /// Write a single entry asynchronously
    pub async fn write_entry(&self, entry: WalEntry) -> Result<u64> {
        let (tx, rx) = oneshot::channel();

        self.sender
            .send(WalRequest::WriteEntry {
                entry,
                response: tx,
            })
            .map_err(|_| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "WAL writer task has stopped",
                ))
            })?;

        rx.await.map_err(|_| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "WAL writer response channel closed",
            ))
        })?
    }

    /// Write multiple entries in a batch asynchronously
    pub async fn write_entries_batch(&self, entries: Vec<WalEntry>) -> Result<Vec<u64>> {
        let (tx, rx) = oneshot::channel();

        self.sender
            .send(WalRequest::WriteBatch {
                entries,
                response: tx,
            })
            .map_err(|_| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "WAL writer task has stopped",
                ))
            })?;

        rx.await.map_err(|_| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "WAL writer response channel closed",
            ))
        })?
    }

    /// Force flush any pending writes
    pub async fn force_flush(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();

        self.sender
            .send(WalRequest::ForceFlush { response: tx })
            .map_err(|_| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "WAL writer task has stopped",
                ))
            })?;

        rx.await.map_err(|_| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "WAL writer response channel closed",
            ))
        })?
    }

    /// Shutdown the async WAL writer gracefully
    pub async fn shutdown(&self) -> Result<()> {
        self.sender.send(WalRequest::Shutdown).map_err(|_| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "WAL writer task has stopped",
            ))
        })?;

        Ok(())
    }
}

impl WalWriter {
    /// Main loop for the background writer task
    async fn run(&mut self, mut receiver: mpsc::UnboundedReceiver<WalRequest>) {
        while let Some(request) = receiver.recv().await {
            match request {
                WalRequest::WriteEntry { entry, response } => {
                    let result = self.write_entry_internal(entry).await;
                    let _ = response.send(result);
                }
                WalRequest::WriteBatch { entries, response } => {
                    let result = self.write_entries_batch_internal(entries).await;
                    let _ = response.send(result);
                }
                WalRequest::ForceFlush { response } => {
                    let result = self.flush_buffer().await;
                    let _ = response.send(result);
                }
                WalRequest::Shutdown => {
                    // Flush any pending writes before shutdown
                    let _ = self.flush_buffer().await;
                    break;
                }
            }
        }
    }

    /// Internal method to write a single entry
    async fn write_entry_internal(&mut self, entry: WalEntry) -> Result<u64> {
        let seq = self.sequence_number;
        self.sequence_number += 1;

        // Serialize using bincode
        let entry_bytes =
            bincode::serialize(&entry).map_err(|e| Error::Serialization(e.to_string()))?;

        // Compress if enabled
        let compressed = if self.compression_enabled {
            tokio::task::spawn_blocking({
                let bytes = entry_bytes.clone();
                let level = self.compression_level;
                move || zstd::encode_all(&bytes[..], level)
            })
            .await
            .map_err(|e| Error::Serialization(format!("Compression task failed: {}", e)))?
            .map_err(|e| Error::Serialization(format!("Compression failed: {}", e)))?
        } else {
            entry_bytes
        };

        // Append to buffer
        self.write_buffer.extend_from_slice(&seq.to_le_bytes());
        self.write_buffer
            .push(if self.compression_enabled { 1 } else { 0 });
        self.write_buffer
            .extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        self.write_buffer.extend_from_slice(&compressed);

        // Batch flush: only flush when buffer exceeds threshold
        if self.write_buffer.len() >= self.batch_threshold {
            self.flush_buffer().await?;
        }

        Ok(seq)
    }

    /// Internal method to write multiple entries
    async fn write_entries_batch_internal(&mut self, entries: Vec<WalEntry>) -> Result<Vec<u64>> {
        let mut seq_numbers = Vec::with_capacity(entries.len());

        for entry in entries {
            let seq = self.sequence_number;
            self.sequence_number += 1;

            // Serialize using bincode
            let entry_bytes =
                bincode::serialize(&entry).map_err(|e| Error::Serialization(e.to_string()))?;

            // Compress if enabled (in blocking task to avoid blocking async runtime)
            let compressed = if self.compression_enabled {
                tokio::task::spawn_blocking({
                    let bytes = entry_bytes.clone();
                    let level = self.compression_level;
                    move || zstd::encode_all(&bytes[..], level)
                })
                .await
                .map_err(|e| Error::Serialization(format!("Compression task failed: {}", e)))?
                .map_err(|e| Error::Serialization(format!("Compression failed: {}", e)))?
            } else {
                entry_bytes
            };

            // Append to buffer
            self.write_buffer.extend_from_slice(&seq.to_le_bytes());
            self.write_buffer
                .push(if self.compression_enabled { 1 } else { 0 });
            self.write_buffer
                .extend_from_slice(&(compressed.len() as u32).to_le_bytes());
            self.write_buffer.extend_from_slice(&compressed);

            seq_numbers.push(seq);
        }

        // Always flush after batch write
        self.flush_buffer().await?;

        Ok(seq_numbers)
    }

    /// Flush write buffer to disk asynchronously
    async fn flush_buffer(&mut self) -> Result<()> {
        if self.write_buffer.is_empty() {
            return Ok(());
        }

        if let Some(ref mut file) = self.current_file {
            file.write_all(&self.write_buffer)
                .await
                .map_err(Error::Io)?;
            file.flush().await.map_err(Error::Io)?;
        }

        self.write_buffer.clear();
        Ok(())
    }

    /// Open the current WAL file
    async fn open_current_file(&mut self) -> Result<()> {
        let wal_file_path = self.wal_path.join("wal.log");

        // If file exists, read the last sequence number
        if wal_file_path.exists() {
            let metadata = tokio::fs::metadata(&wal_file_path)
                .await
                .map_err(Error::Io)?;

            if metadata.len() > 0 {
                // Read last sequence number (simplified - in production would need full parsing)
                // For now, just open in append mode
                self.sequence_number = 0; // Would be read from file in production
            }
        }

        // Open file for appending
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_file_path)
            .await
            .map_err(Error::Io)?;

        self.current_file = Some(file);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_async_wal_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let wal = AsyncWalManager::new(temp_dir.path()).await.unwrap();

        // Write some entries
        let seq1 = wal
            .write_entry(WalEntry::CreateTable {
                table: "test".to_string(),
                dimension: 128,
            })
            .await
            .unwrap();

        let seq2 = wal
            .write_entry(WalEntry::Insert {
                table: "test".to_string(),
                record_id: "rec1".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                payload: None,
            })
            .await
            .unwrap();

        assert_eq!(seq1, 0);
        assert_eq!(seq2, 1);

        // Force flush
        wal.force_flush().await.unwrap();
    }

    #[tokio::test]
    async fn test_async_wal_batch_write() {
        let temp_dir = TempDir::new().unwrap();
        let wal = AsyncWalManager::new(temp_dir.path()).await.unwrap();

        let entries = vec![
            WalEntry::CreateTable {
                table: "test".to_string(),
                dimension: 128,
            },
            WalEntry::Insert {
                table: "test".to_string(),
                record_id: "rec1".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                payload: None,
            },
            WalEntry::Insert {
                table: "test".to_string(),
                record_id: "rec2".to_string(),
                vector: vec![4.0, 5.0, 6.0],
                payload: None,
            },
        ];

        let seq_numbers = wal.write_entries_batch(entries).await.unwrap();
        assert_eq!(seq_numbers.len(), 3);
        assert_eq!(seq_numbers, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn test_async_wal_graceful_shutdown() {
        let temp_dir = TempDir::new().unwrap();
        let wal = AsyncWalManager::new(temp_dir.path()).await.unwrap();

        // Write some entries
        wal.write_entry(WalEntry::CreateTable {
            table: "test".to_string(),
            dimension: 128,
        })
        .await
        .unwrap();

        // Shutdown gracefully (should flush pending writes)
        wal.shutdown().await.unwrap();
    }
}
