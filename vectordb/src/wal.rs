//! Write-Ahead Log (WAL) implementation for crash recovery
//! 
//! Provides basic WAL functionality for ensuring data consistency
//! across crashes and restarts.

use std::fs::{File, OpenOptions};
use std::io::{Write, Read};
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};

/// WAL entry types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    Insert {
        table: String,
        record_id: String,
        vector: Vec<f32>,
        payload: Option<serde_json::Value>,
    },
    Delete {
        table: String,
        record_id: String,
    },
    CreateTable {
        table: String,
        dimension: usize,
    },
    /// Table creation carrying the whole config.
    ///
    /// `CreateTable` above keeps only the name and dimension, which is not enough to rebuild a
    /// table on replay — `distance_metric`, `named_vectors`, `payload_schema`,
    /// `memory_optimization` and `enable_full_text_search` are all lost, and a table restored
    /// without its distance metric answers queries wrongly rather than failing. New writers
    /// should emit this variant; the old one is still read so existing logs replay.
    CreateTableWithConfig {
        config: crate::types::table::TableConfig,
    },
    DeleteTable {
        table: String,
    },
    Checkpoint {
        timestamp: u64,
    },
}

/// Write-Ahead Log manager
pub struct WalManager {
    wal_path: PathBuf,
    current_file: Option<File>,
    sequence_number: u64,

    /// Batch buffer for flush optimization
    write_buffer: Vec<u8>,
    /// Batch flush threshold (bytes)
    batch_threshold: usize,

    /// Enable compression (zstd)
    compression_enabled: bool,
    /// Compression level (1-21, 3 is default)
    compression_level: i32,
}

impl WalManager {
    /// Create a new WAL manager with default settings (no compression)
    pub fn new<P: AsRef<Path>>(wal_dir: P) -> Result<Self> {
        Self::new_with_compression(wal_dir, false, 3)
    }

    /// Create a new WAL manager with compression settings
    pub fn new_with_compression<P: AsRef<Path>>(
        wal_dir: P,
        enable_compression: bool,
        compression_level: i32,
    ) -> Result<Self> {
        let wal_path = wal_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&wal_path)
            .map_err(Error::Io)?;

        let mut manager = WalManager {
            wal_path,
            current_file: None,
            sequence_number: 0,
            write_buffer: Vec::with_capacity(1024 * 1024), // 1MB buffer
            batch_threshold: 256 * 1024, // 256KB threshold
            compression_enabled: enable_compression,
            compression_level,
        };

        manager.open_current_file()?;
        Ok(manager)
    }
    
    /// Write an entry to the WAL (using MessagePack + optional compression + batch flush)
    pub fn write_entry(&mut self, entry: WalEntry) -> Result<u64> {
        let seq = self.sequence_number;
        self.sequence_number += 1;

        // Serialize using MessagePack (supports dynamic types like serde_json::Value)
        let entry_bytes = rmp_serde::to_vec(&entry)
            .map_err(|e| Error::Serialization(e.to_string()))?;

        // Compress if enabled (zstd: -80% size)
        let compressed = if self.compression_enabled {
            zstd::encode_all(&entry_bytes[..], self.compression_level)
                .map_err(|e| Error::Serialization(format!("Compression failed: {}", e)))?
        } else {
            entry_bytes.clone()
        };

        // Write format: [seq: u64][compressed_flag: u8][len: u32][data: bytes]
        // Append to buffer instead of immediate write
        self.write_buffer.extend_from_slice(&seq.to_le_bytes());
        self.write_buffer.push(if self.compression_enabled { 1 } else { 0 });
        self.write_buffer.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        self.write_buffer.extend_from_slice(&compressed);

        // Batch flush: only flush when buffer exceeds threshold
        if self.write_buffer.len() >= self.batch_threshold {
            self.flush_buffer()?;
        }

        Ok(seq)
    }

    /// Write multiple entries in a batch (optimized for high throughput + compression)
    pub fn write_entries_batch(&mut self, entries: Vec<WalEntry>) -> Result<Vec<u64>> {
        let mut seq_numbers = Vec::with_capacity(entries.len());

        for entry in entries {
            let seq = self.sequence_number;
            self.sequence_number += 1;

            // Serialize using MessagePack
            let entry_bytes = rmp_serde::to_vec(&entry)
                .map_err(|e| Error::Serialization(e.to_string()))?;

            // Compress if enabled
            let compressed = if self.compression_enabled {
                zstd::encode_all(&entry_bytes[..], self.compression_level)
                    .map_err(|e| Error::Serialization(format!("Compression failed: {}", e)))?
            } else {
                entry_bytes
            };

            // Append to buffer
            self.write_buffer.extend_from_slice(&seq.to_le_bytes());
            self.write_buffer.push(if self.compression_enabled { 1 } else { 0 });
            self.write_buffer.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
            self.write_buffer.extend_from_slice(&compressed);

            seq_numbers.push(seq);
        }

        // Always flush after batch write
        self.flush_buffer()?;

        Ok(seq_numbers)
    }

    /// Flush write buffer to disk
    fn flush_buffer(&mut self) -> Result<()> {
        if self.write_buffer.is_empty() {
            return Ok(());
        }

        if let Some(ref mut file) = self.current_file {
            file.write_all(&self.write_buffer)
                .map_err(Error::Io)?;
            // `Write::flush` on a File is a no-op — it only drains a userspace buffer, and File
            // has none. Without sync_data the entries sit in the page cache, which survives
            // `kill -9` but not power loss. A write-ahead log that loses acknowledged writes on
            // power loss is not doing its job, so pay for the fsync here.
            file.sync_data()
                .map_err(Error::Io)?;
        }

        self.write_buffer.clear();
        Ok(())
    }

    /// Force flush any pending writes
    pub fn force_flush(&mut self) -> Result<()> {
        self.flush_buffer()
    }

    /// Sequence number the next entry will get.
    ///
    /// Recorded alongside an index snapshot so that recovery can replay only the tail of the
    /// log instead of the whole thing.
    pub fn next_sequence(&self) -> u64 {
        self.sequence_number
    }
    
    /// Replay WAL entries from a given sequence number (supports compression)
    pub fn replay_from(&self, from_seq: u64) -> Result<Vec<(u64, WalEntry)>> {
        let wal_file_path = self.wal_path.join("wal.log");
        if !wal_file_path.exists() {
            return Ok(Vec::new());
        }

        let mut file = File::open(&wal_file_path)
            .map_err(Error::Io)?;

        let mut entries = Vec::new();

        loop {
            // Read sequence number (8 bytes)
            let mut seq_bytes = [0u8; 8];
            match file.read_exact(&mut seq_bytes) {
                Ok(_) => {},
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(Error::Io(e)),
            }
            let seq = u64::from_le_bytes(seq_bytes);

            // Read compression flag (1 byte)
            let mut compressed_flag = [0u8; 1];
            file.read_exact(&mut compressed_flag)
                .map_err(Error::Io)?;
            let is_compressed = compressed_flag[0] == 1;

            // Read length (4 bytes)
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)
                .map_err(Error::Io)?;
            let len = u32::from_le_bytes(len_bytes) as usize;

            // Read data
            let mut data = vec![0u8; len];
            file.read_exact(&mut data)
                .map_err(Error::Io)?;

            // Decompress if needed
            let decompressed = if is_compressed {
                zstd::decode_all(&data[..])
                    .map_err(|e| Error::Serialization(format!("Decompression failed: {}", e)))?
            } else {
                data
            };

            // Deserialize using MessagePack
            if seq >= from_seq {
                let entry: WalEntry = rmp_serde::from_slice(&decompressed)
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                entries.push((seq, entry));
            }
        }

        Ok(entries)
    }
    
    /// Create a checkpoint and truncate old entries
    pub fn checkpoint(&mut self) -> Result<u64> {
        // Flush any pending writes before checkpoint
        self.flush_buffer()?;

        let checkpoint_seq = self.sequence_number;
        self.write_entry(WalEntry::Checkpoint {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        })?;

        // Ensure checkpoint is written to disk
        self.flush_buffer()?;

        Ok(checkpoint_seq)
    }
    
    /// Get the current sequence number
    pub fn current_sequence(&self) -> u64 {
        self.sequence_number
    }
    
    fn open_current_file(&mut self) -> Result<()> {
        let wal_file_path = self.wal_path.join("wal.log");

        // If file exists, find the last sequence number (with compression support)
        if wal_file_path.exists() {
            let mut file = File::open(&wal_file_path)
                .map_err(Error::Io)?;

            let mut last_seq = 0u64;
            loop {
                // Read sequence number (8 bytes)
                let mut seq_bytes = [0u8; 8];
                match file.read_exact(&mut seq_bytes) {
                    Ok(_) => {},
                    Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(_) => break, // Corrupted file, stop reading
                }
                last_seq = u64::from_le_bytes(seq_bytes);

                // Read compression flag (1 byte)
                let mut compressed_flag = [0u8; 1];
                if file.read_exact(&mut compressed_flag).is_err() {
                    break;
                }

                // Read length (4 bytes) and skip data
                let mut len_bytes = [0u8; 4];
                if file.read_exact(&mut len_bytes).is_err() {
                    break;
                }
                let len = u32::from_le_bytes(len_bytes) as i64;

                // Skip the data bytes
                if std::io::Seek::seek(&mut file, std::io::SeekFrom::Current(len)).is_err() {
                    break;
                }
            }

            self.sequence_number = last_seq + 1;
        }

        // Open file for appending
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_file_path)
            .map_err(Error::Io)?;

        self.current_file = Some(file);
        Ok(())
    }
}

impl Drop for WalManager {
    fn drop(&mut self) {
        // Ensure all pending writes are flushed on drop
        let _ = self.flush_buffer();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_wal_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(temp_dir.path()).unwrap();
        
        // Write some entries
        let seq1 = wal.write_entry(WalEntry::CreateTable {
            table: "test".to_string(),
            dimension: 128,
        }).unwrap();
        
        let seq2 = wal.write_entry(WalEntry::Insert {
            table: "test".to_string(),
            record_id: "rec1".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            payload: None,
        }).unwrap();
        
        assert_eq!(seq1, 0);
        assert_eq!(seq2, 1);

        // Flush buffer to disk before replay
        wal.force_flush().unwrap();

        // Replay entries
        let entries = wal.replay_from(0).unwrap();
        assert_eq!(entries.len(), 2);
    }
    
    #[test]
    fn test_wal_recovery() {
        let temp_dir = TempDir::new().unwrap();
        
        // First WAL instance
        {
            let mut wal = WalManager::new(temp_dir.path()).unwrap();
            wal.write_entry(WalEntry::CreateTable {
                table: "test".to_string(),
                dimension: 128,
            }).unwrap();
        }
        
        // Second WAL instance (simulating restart)
        {
            let wal = WalManager::new(temp_dir.path()).unwrap();
            assert_eq!(wal.current_sequence(), 1);
            
            let entries = wal.replay_from(0).unwrap();
            assert_eq!(entries.len(), 1);
        }
    }
}
