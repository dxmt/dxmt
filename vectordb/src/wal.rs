//! Write-Ahead Log (WAL) implementation for crash recovery
//!
//! Provides basic WAL functionality for ensuring data consistency
//! across crashes and restarts.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

/// A single WAL payload may be large, but an unbounded length from a corrupt header must not
/// turn recovery into a multi-gigabyte allocation. Larger frames are rejected explicitly.
const MAX_WAL_FRAME_BYTES: usize = 256 * 1024 * 1024;

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
    DeleteTable {
        table: String,
    },
    Checkpoint {
        timestamp: u64,
    },
    // Everything below this point was added after the original five variants. rmp-serde tags a
    // variant by *name* — `{"Delete": [...]}`, as the fixtures in `historical_messagepack_entries_still_decode`
    // show — so adding or reordering variants is safe, but renaming one, or changing the order
    // of the fields inside one, silently breaks every log already on disk.
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
    /// Upsert a record carrying the full request shape.
    ///
    /// The legacy `Insert` variant is kept for old logs, but it cannot restore sparse vectors
    /// or named vectors. New writers should use this variant for both inserts and updates.
    UpsertRecord {
        table: String,
        request: crate::types::record::CreateRecordRequest,
    },
    AlterTable {
        table: String,
        request: crate::types::table::TableAlterRequest,
    },
    /// Full record form used by current writers.
    ///
    /// `UpsertRecord` above preserves the request fields but regenerates timestamps and derives
    /// the version again during replay. Keep it readable for logs written during the transition;
    /// new logs carry the exact persisted record instead.
    UpsertRecordV2 {
        table: String,
        record: crate::types::record::Record,
    },
    /// A record whose vector is in the table's row file, at `row`, and not in the entry.
    ///
    /// With a WAL directory a table's full vectors live in `wal/vectors/<table>.f32`, appended
    /// and fsynced before the write is logged. Logging the vector again, as `UpsertRecordV2`
    /// does, wrote every vector to disk twice -- at 960 dimensions the entry was 3.8 KB of
    /// vector and 100 bytes of record. `record.vector` is empty here; replay reads `row`
    /// from the row file. The row and not the id is what the entry carries, because rows
    /// are never reused or renumbered, while the id's row in the record store moves on
    /// every update and goes away on a delete -- replaying `insert, delete, insert` by id
    /// would read the wrong row for the first entry and nothing for the last. Written by
    /// the insert and update paths for a table with a row file; a table whose vectors are
    /// in memory still writes `UpsertRecordV2`, and so does every table when the log is
    /// mirrored to object storage, where a node recovering from the bucket has no row file
    /// to read.
    UpsertRecordInRowFile {
        table: String,
        record: crate::types::record::Record,
        row: u64,
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

    /// Test-only switch used to exercise the storage read-only degradation path without
    /// depending on filesystem permissions or filling the host disk.
    #[cfg(feature = "test-failpoints")]
    fail_next_sync: bool,
    /// Write half of the next flush and then fail, as a full disk does.
    #[cfg(feature = "test-failpoints")]
    tear_next_write: bool,
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
        std::fs::create_dir_all(&wal_path).map_err(Error::Io)?;

        let mut manager = WalManager {
            wal_path,
            current_file: None,
            sequence_number: 0,
            write_buffer: Vec::with_capacity(1024 * 1024), // 1MB buffer
            batch_threshold: 256 * 1024,                   // 256KB threshold
            compression_enabled: enable_compression,
            compression_level,
            #[cfg(feature = "test-failpoints")]
            fail_next_sync: false,
            #[cfg(feature = "test-failpoints")]
            tear_next_write: false,
        };

        manager.open_current_file()?;
        // `open_current_file` scans the framed log and restores the next sequence number. Keep
        // payload decoding in `replay_from`: opening a WAL should still be possible when a
        // caller wants to inspect/report a corrupt frame (for example an invalid compression
        // flag), while recovery remains strict when it explicitly replays the log.
        Ok(manager)
    }

    /// Write an entry to the WAL (using MessagePack + optional compression + batch flush)
    pub fn write_entry(&mut self, entry: WalEntry) -> Result<u64> {
        // Serialize using MessagePack (supports dynamic types like serde_json::Value)
        let entry_bytes =
            rmp_serde::to_vec(&entry).map_err(|e| Error::Serialization(e.to_string()))?;

        // Compress if enabled (zstd: -80% size)
        let compressed = if self.compression_enabled {
            zstd::encode_all(&entry_bytes[..], self.compression_level)
                .map_err(|e| Error::Serialization(format!("Compression failed: {}", e)))?
        } else {
            entry_bytes
        };

        // Numbered only once there is a frame to number. Recovery refuses a log that starts
        // after the snapshot's replay point, so a number spent on an entry that never reached
        // the file would read as lost writes.
        let seq = self.sequence_number;
        self.sequence_number += 1;

        // Write format: [seq: u64][compressed_flag: u8][len: u32][data: bytes]
        // Append to buffer instead of immediate write
        self.write_buffer.extend_from_slice(&seq.to_le_bytes());
        self.write_buffer
            .push(if self.compression_enabled { 1 } else { 0 });
        self.write_buffer
            .extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        self.write_buffer.extend_from_slice(&compressed);

        // Batch flush: only flush when buffer exceeds threshold
        if self.write_buffer.len() >= self.batch_threshold {
            self.flush_buffer()?;
        }

        Ok(seq)
    }

    /// Write multiple entries in a batch (optimized for high throughput + compression)
    pub fn write_entries_batch(&mut self, entries: Vec<WalEntry>) -> Result<Vec<u64>> {
        // Every entry is encoded before any is numbered or buffered, for the reason given in
        // `write_entry`: a failure halfway through must not leave numbers without frames.
        let mut encoded = Vec::with_capacity(entries.len());
        for entry in entries {
            // Serialize using MessagePack
            let entry_bytes =
                rmp_serde::to_vec(&entry).map_err(|e| Error::Serialization(e.to_string()))?;

            // Compress if enabled
            let compressed = if self.compression_enabled {
                zstd::encode_all(&entry_bytes[..], self.compression_level)
                    .map_err(|e| Error::Serialization(format!("Compression failed: {}", e)))?
            } else {
                entry_bytes
            };
            encoded.push(compressed);
        }

        let mut seq_numbers = Vec::with_capacity(encoded.len());
        for compressed in encoded {
            let seq = self.sequence_number;
            self.sequence_number += 1;

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
        self.flush_buffer()?;

        Ok(seq_numbers)
    }

    /// Flush write buffer to disk
    fn flush_buffer(&mut self) -> Result<()> {
        if self.write_buffer.is_empty() {
            return Ok(());
        }

        // Take ownership before touching the file. Once a write or fsync reports an error the
        // caller puts storage into read-only mode; retaining the bytes here would make Drop retry
        // them and could append a complete frame twice after a successful `write_all` followed by
        // a failed `sync_data`.
        let pending = std::mem::take(&mut self.write_buffer);
        if let Some(ref mut file) = self.current_file {
            #[cfg(feature = "test-failpoints")]
            if self.tear_next_write {
                self.tear_next_write = false;
                file.write_all(&pending[..pending.len() / 2])
                    .map_err(Error::Io)?;
                return Err(Error::Io(std::io::Error::other("injected torn WAL write")));
            }
            file.write_all(&pending).map_err(Error::Io)?;
            #[cfg(feature = "test-failpoints")]
            if self.fail_next_sync {
                self.fail_next_sync = false;
                return Err(Error::Io(std::io::Error::other(
                    "injected WAL sync failure",
                )));
            }
            // `Write::flush` on a File is a no-op — it only drains a userspace buffer, and File
            // has none. Without sync_data the entries sit in the page cache, which survives
            // `kill -9` but not power loss. A write-ahead log that loses acknowledged writes on
            // power loss is not doing its job, so pay for the fsync here.
            file.sync_data().map_err(Error::Io)?;
        }

        Ok(())
    }

    /// Force flush any pending writes
    pub fn force_flush(&mut self) -> Result<()> {
        self.flush_buffer()
    }

    /// Inject one sync failure for storage failpoint tests.
    #[cfg(feature = "test-failpoints")]
    pub fn fail_next_sync_for_test(&mut self) {
        self.fail_next_sync = true;
    }

    /// Inject one torn write: half the pending bytes reach the file, then the flush fails.
    #[cfg(feature = "test-failpoints")]
    pub fn tear_next_write_for_test(&mut self) {
        self.tear_next_write = true;
    }

    /// Sequence number the next entry will get.
    ///
    /// Recorded alongside an index snapshot so that recovery can replay only the tail of the
    /// log instead of the whole thing.
    pub fn next_sequence(&self) -> u64 {
        self.sequence_number
    }

    /// Never hand out a sequence number below `next` again.
    ///
    /// Opening a log derives the next number by scanning the file, and a log that a snapshot
    /// has compacted to nothing scans as "start at 0". The snapshot manifest still says
    /// where the log left off, and recovery replays from there -- so a write numbered from
    /// 0 in the session after a restart is below the replay point, and the restart after
    /// that skips it. The storage layer passes the manifest's sequence here before it
    /// writes anything.
    pub fn advance_sequence_to(&mut self, next: u64) {
        self.sequence_number = self.sequence_number.max(next);
    }

    /// Replay WAL entries from a given sequence number (supports compression)
    /// The bytes of every frame from `from_seq` on, or `None` if the file is not shaped the way
    /// this expects.
    ///
    /// Headers only: sequence, compression flag and length, then a seek over the payload. No
    /// frame is decoded, which is the whole point -- see the caller.
    fn tail_bytes_from(&self, from_seq: u64) -> Result<Option<Vec<u8>>> {
        use std::io::SeekFrom;

        let wal_file_path = self.wal_path.join("wal.log");
        let file = match File::open(&wal_file_path) {
            Ok(file) => file,
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Some(Vec::new())),
            Err(e) => return Err(Error::Io(e)),
        };
        let file_len = file.metadata().map_err(Error::Io)?.len();
        // Buffered, and skipping with `seek_relative`, which stays inside the buffer instead of
        // issuing a syscall per frame. Unbuffered this was a read and a seek for every entry in
        // the log -- 320,000 syscalls for 160,000 entries -- and it cost 1.23 s under the WAL
        // mutex, which is the same stall this whole change is about, just further down.
        let mut file = std::io::BufReader::with_capacity(1 << 20, file);

        let mut offset = 0u64;
        loop {
            if offset >= file_len {
                // Every frame is below `from_seq`; nothing survives.
                return Ok(Some(Vec::new()));
            }
            let mut header = [0u8; 13];
            if file.read_exact(&mut header).is_err() {
                // A torn tail, which replay drops anyway: everything from here is going.
                return Ok(Some(Vec::new()));
            }
            let seq = u64::from_le_bytes(header[0..8].try_into().expect("8 bytes"));
            let compressed = header[8];
            let len = u32::from_le_bytes(header[9..13].try_into().expect("4 bytes")) as usize;
            if compressed > 1 || len > MAX_WAL_FRAME_BYTES {
                // Not a header this writer produced. Hand it to the decoding path, which
                // reports the problem rather than guessing past it.
                return Ok(None);
            }
            let payload_end = offset + 13 + len as u64;
            if payload_end > file_len {
                // The last frame is torn; it is below `from_seq` or it is the tail's first
                // frame, and either way there is nothing complete from here on.
                return Ok(Some(Vec::new()));
            }
            if seq >= from_seq {
                let mut bytes = Vec::with_capacity((file_len - offset) as usize);
                file.seek(SeekFrom::Start(offset)).map_err(Error::Io)?;
                file.read_to_end(&mut bytes).map_err(Error::Io)?;
                return Ok(Some(bytes));
            }
            file.seek_relative(len as i64).map_err(Error::Io)?;
            offset = payload_end;
        }
    }

    pub fn replay_from(&self, from_seq: u64) -> Result<Vec<(u64, WalEntry)>> {
        let wal_file_path = self.wal_path.join("wal.log");
        if !wal_file_path.exists() {
            return Ok(Vec::new());
        }

        let mut file = File::open(&wal_file_path).map_err(Error::Io)?;

        let mut entries = Vec::new();

        loop {
            // Read sequence number (8 bytes)
            let mut seq_bytes = [0u8; 8];
            match file.read_exact(&mut seq_bytes) {
                Ok(_) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(Error::Io(e)),
            }
            let seq = u64::from_le_bytes(seq_bytes);

            // A process can die between any two writes that make up the framed record. A
            // complete record is authoritative; an incomplete final frame is a torn tail and is
            // ignored so restart can recover all earlier acknowledged entries. Errors other than
            // EOF remain fatal because they indicate an unreadable file, not a crash boundary.
            // Read compression flag (1 byte)
            let mut compressed_flag = [0u8; 1];
            match file.read_exact(&mut compressed_flag) {
                Ok(_) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(Error::Io(e)),
            }
            if compressed_flag[0] > 1 {
                return Err(Error::Serialization(format!(
                    "Invalid WAL compression flag: {}",
                    compressed_flag[0]
                )));
            }
            let is_compressed = compressed_flag[0] == 1;

            // Read length (4 bytes)
            let mut len_bytes = [0u8; 4];
            match file.read_exact(&mut len_bytes) {
                Ok(_) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(Error::Io(e)),
            }
            let len = u32::from_le_bytes(len_bytes) as usize;
            if len > MAX_WAL_FRAME_BYTES {
                return Err(Error::Serialization(format!(
                    "WAL frame is too large: {} bytes (maximum {})",
                    len, MAX_WAL_FRAME_BYTES
                )));
            }

            // A complete header whose payload extends past EOF is a torn final frame. Check this
            // before allocating so a crash-written length cannot trigger a huge allocation.
            let payload_start = file.stream_position().map_err(Error::Io)?;
            let file_len = file.metadata().map_err(Error::Io)?.len();
            if payload_start.saturating_add(len as u64) > file_len {
                break;
            }

            // Read data
            let mut data = vec![0u8; len];
            match file.read_exact(&mut data) {
                Ok(_) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(Error::Io(e)),
            }

            // Decompress if needed
            let decompressed = if is_compressed {
                zstd::decode_all(&data[..])
                    .map_err(|e| Error::Serialization(format!("Decompression failed: {}", e)))?
            } else {
                data
            };

            // Deserialize even entries before `from_seq`: a snapshot may allow us to skip
            // applying them, but silently accepting a corrupt middle frame would make the
            // recovery frontier unknowable. Only retain entries at or after the requested point.
            let entry: WalEntry = rmp_serde::from_slice(&decompressed)
                .map_err(|e| Error::Serialization(e.to_string()))?;
            if seq >= from_seq {
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
                .as_secs(),
        })?;

        // Ensure checkpoint is written to disk
        self.flush_buffer()?;

        Ok(checkpoint_seq)
    }

    /// Drop every entry below `keep_from`, keeping the rest.
    ///
    /// Without this the log only grows: a snapshot makes the entries it covers redundant, but
    /// they stay on disk forever and every restart re-reads them. Returns how many bytes were
    /// reclaimed.
    ///
    /// Rewrites to a temporary file and renames it into place, so an interruption leaves the
    /// old log intact. Losing the log to a failed compaction would be worse than never
    /// compacting: the snapshot alone cannot account for writes made after it.
    pub fn truncate_before(&mut self, keep_from: u64) -> Result<u64> {
        self.flush_buffer()?;

        let wal_file_path = self.wal_path.join("wal.log");
        if !wal_file_path.exists() {
            return Ok(0);
        }
        let before = std::fs::metadata(&wal_file_path).map_err(Error::Io)?.len();

        // When the snapshot covers everything written, nothing survives the compaction and
        // the file can go without being read.
        //
        // `replay_from` deserialises every frame, including the ones below `keep_from` -- it
        // does that on purpose, so a corrupt frame in the middle cannot be silently skipped
        // and leave the recovery frontier unknowable. That reasoning does not apply to a log
        // that is about to be deleted in full: by definition there is no entry at or after
        // `keep_from` to lose. Measured on a 1M-row table, the scan it replaces read 3.97 GB
        // and spent 15 s doing it, and the transient allocations took resident memory from
        // 8.0 GB to 14.2 GB -- the whole of what a snapshot appeared to cost, once the record
        // copying was gone (design doc section 34).
        //
        // `sequence_number` is the next sequence to be handed out, so the highest written is
        // one below it.
        let nothing_survives = keep_from >= self.sequence_number;

        // The surviving tail, copied as bytes rather than decoded and re-encoded.
        //
        // This used to call `replay_from`, which deserialises *every* frame including the ones
        // below `keep_from` -- on purpose, so that a corrupt frame in the middle of a log being
        // recovered cannot be silently skipped. Compaction is the other case: the frames below
        // `keep_from` are covered by a snapshot that has just been read back and verified, so
        // nothing is lost by not looking at them, which is the same reasoning the
        // `nothing_survives` shortcut above already rests on.
        //
        // What it cost: the decode runs under the WAL mutex, and every write waits on that
        // mutex. Measured over HTTP with four tables of 40,000 rows and a writer running
        // throughout, a snapshot blocked writes for 1.28 s of its 1.55 s -- all of it here,
        // decoding 160,000 entries to keep the twenty that arrived while the snapshot ran.
        // The comment above puts the same scan at 15 s and 3.97 GB on a 1M-row table.
        //
        // Scanning frame headers and copying the tail verbatim is the same operation without
        // the decode: seek past each payload rather than parse it. A header that does not make
        // sense means the file is not what this expects, and the old path -- decode, and let it
        // report what it finds -- takes over.
        let tail = if nothing_survives {
            Some(Vec::new())
        } else {
            self.tail_bytes_from(keep_from)?
        };

        let tmp_path = self.wal_path.join("wal.log.compacting");
        {
            let mut tmp = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp_path)
                .map_err(Error::Io)?;

            if let Some(bytes) = tail {
                tmp.write_all(&bytes).map_err(Error::Io)?;
                tmp.sync_data().map_err(Error::Io)?;
                drop(tmp);
                self.current_file = None;
                std::fs::rename(&tmp_path, &wal_file_path).map_err(Error::Io)?;
                let next = self.sequence_number;
                self.open_current_file()?;
                self.sequence_number = self.sequence_number.max(next);
                let after = std::fs::metadata(&wal_file_path).map_err(Error::Io)?.len();
                return Ok(before.saturating_sub(after));
            }

            // The fallback: the header scan found something it did not understand, so every
            // frame is decoded and re-encoded, which is what this always did.
            for (seq, entry) in self.replay_from(keep_from)? {
                let bytes =
                    rmp_serde::to_vec(&entry).map_err(|e| Error::Serialization(e.to_string()))?;
                let payload = if self.compression_enabled {
                    zstd::encode_all(&bytes[..], self.compression_level)
                        .map_err(|e| Error::Serialization(format!("Compression failed: {}", e)))?
                } else {
                    bytes
                };
                tmp.write_all(&seq.to_le_bytes()).map_err(Error::Io)?;
                tmp.write_all(&[u8::from(self.compression_enabled)])
                    .map_err(Error::Io)?;
                tmp.write_all(&(payload.len() as u32).to_le_bytes())
                    .map_err(Error::Io)?;
                tmp.write_all(&payload).map_err(Error::Io)?;
            }
            tmp.sync_data().map_err(Error::Io)?;
        }

        // Close our append handle before swapping the file underneath it.
        self.current_file = None;
        std::fs::rename(&tmp_path, &wal_file_path).map_err(Error::Io)?;

        // `open_current_file` derives the next sequence number by scanning the file. If the
        // compaction removed every entry, that scan would restart numbering at 1 and the next
        // writes would collide with sequence numbers a snapshot manifest already refers to.
        // The counter only ever moves forward.
        let next = self.sequence_number;
        self.open_current_file()?;
        self.sequence_number = self.sequence_number.max(next);

        let after = std::fs::metadata(&wal_file_path).map_err(Error::Io)?.len();
        Ok(before.saturating_sub(after))
    }

    /// Get the current sequence number
    pub fn current_sequence(&self) -> u64 {
        self.sequence_number
    }

    fn open_current_file(&mut self) -> Result<()> {
        let wal_file_path = self.wal_path.join("wal.log");

        // If file exists, find the last sequence number (with compression support)
        if wal_file_path.exists() {
            let mut file = File::open(&wal_file_path).map_err(Error::Io)?;

            let mut last_seq = 0u64;
            let mut saw_complete_frame = false;
            let mut truncate_to = None;
            let file_len = file.metadata().map_err(Error::Io)?.len();
            loop {
                let frame_start = file.stream_position().map_err(Error::Io)?;
                // Read sequence number (8 bytes)
                let mut seq_bytes = [0u8; 8];
                match file.read_exact(&mut seq_bytes) {
                    Ok(_) => {}
                    Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        // A crash can leave only part of the next frame's sequence number.
                        truncate_to = Some(frame_start);
                        break;
                    }
                    Err(_) => break, // Corrupted file, stop reading
                }
                let seq = u64::from_le_bytes(seq_bytes);

                // Read compression flag (1 byte)
                let mut compressed_flag = [0u8; 1];
                match file.read_exact(&mut compressed_flag) {
                    Ok(_) => {}
                    Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        truncate_to = Some(frame_start);
                        break;
                    }
                    Err(_) => break,
                }
                // Invalid flags are a corruption error for replay, not a torn tail. Leave the
                // bytes intact so `replay_from` can report the precise failure.
                if compressed_flag[0] > 1 {
                    break;
                }

                // Read length (4 bytes) and skip data
                let mut len_bytes = [0u8; 4];
                match file.read_exact(&mut len_bytes) {
                    Ok(_) => {}
                    Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        truncate_to = Some(frame_start);
                        break;
                    }
                    Err(_) => break,
                }
                let len = u32::from_le_bytes(len_bytes) as u64;
                if len > MAX_WAL_FRAME_BYTES as u64 {
                    // A complete header with an impossible length is corruption, not a crash
                    // tail. Keep it for strict replay diagnostics.
                    break;
                }

                let payload_start = file.stream_position().map_err(Error::Io)?;
                if payload_start.saturating_add(len) > file_len {
                    // The header is present but the payload is not. Remove this final frame before
                    // opening the append handle; otherwise future writes would sit behind the
                    // torn bytes and replay would stop before reaching them forever.
                    truncate_to = Some(frame_start);
                    break;
                }
                std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(payload_start + len))
                    .map_err(Error::Io)?;
                saw_complete_frame = true;
                last_seq = seq;
            }

            drop(file);
            if let Some(offset) = truncate_to {
                let repair = OpenOptions::new()
                    .write(true)
                    .open(&wal_file_path)
                    .map_err(Error::Io)?;
                repair.set_len(offset).map_err(Error::Io)?;
                repair.sync_data().map_err(Error::Io)?;
            }

            self.sequence_number = if saw_complete_frame {
                last_seq.saturating_add(1)
            } else {
                0
            };
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
    fn truncate_before_drops_only_the_covered_prefix() {
        let dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(dir.path()).unwrap();

        for i in 0..10 {
            wal.write_entry(WalEntry::Delete {
                table: "t".to_string(),
                record_id: format!("r{}", i),
            })
            .unwrap();
        }
        wal.force_flush().unwrap();

        let reclaimed = wal.truncate_before(7).unwrap();
        assert!(reclaimed > 0, "compaction reclaimed nothing");

        let kept = wal.replay_from(0).unwrap();
        assert_eq!(kept.len(), 3, "expected entries 7,8,9 to survive");
        assert_eq!(kept.first().unwrap().0, 7);
    }

    /// Compaction keeps the tail byte for byte, and falls back when the file says something
    /// it does not understand.
    ///
    /// The tail is copied rather than decoded and re-encoded, so the property to hold is that
    /// what survives is *identical*, not merely equivalent: an entry that round-tripped
    /// through a different compression setting would still replay, and would still be a
    /// rewrite this is supposed to avoid.
    #[test]
    fn compaction_copies_the_surviving_tail_verbatim() {
        let dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(dir.path()).unwrap();
        for i in 0..40 {
            wal.write_entry(WalEntry::Delete {
                table: "t".to_string(),
                record_id: format!("r{}", i),
            })
            .unwrap();
        }
        wal.force_flush().unwrap();

        let keep_from = 30u64;
        let before: Vec<(u64, WalEntry)> = wal
            .replay_from(0)
            .unwrap()
            .into_iter()
            .filter(|(seq, _)| *seq >= keep_from)
            .collect();
        assert_eq!(before.len(), 10, "the fixture has to straddle the boundary");

        // Compared as the sequence numbers plus the debug rendering of each entry: `WalEntry`
        // has no `PartialEq`, and deriving one for a test would put a trait on the wire type
        // for the sake of an assertion.
        let render = |entries: &[(u64, WalEntry)]| -> Vec<String> {
            entries.iter().map(|(s, e)| format!("{s}:{e:?}")).collect()
        };
        wal.truncate_before(keep_from).unwrap();
        let after = wal.replay_from(0).unwrap();
        assert_eq!(
            render(&after),
            render(&before),
            "the surviving tail changed across a compaction"
        );

        // And the log is still writable, with sequence numbers that do not go backwards.
        let next = wal
            .write_entry(WalEntry::Delete {
                table: "t".to_string(),
                record_id: "after".to_string(),
            })
            .unwrap();
        assert!(
            next >= 40,
            "compaction rewound the sequence to {next}, colliding with entries a snapshot names"
        );
    }

    /// A header the scan cannot make sense of sends it to the decoding path rather than past it.
    #[test]
    fn a_frame_that_is_not_a_frame_falls_back_to_the_decoder() {
        let dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(dir.path()).unwrap();
        for i in 0..5 {
            wal.write_entry(WalEntry::Delete {
                table: "t".to_string(),
                record_id: format!("r{}", i),
            })
            .unwrap();
        }
        wal.force_flush().unwrap();

        // Byte 8 of the first frame is the compression flag, and 7 is neither 0 nor 1.
        let path = dir.path().join("wal.log");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8] = 7;
        std::fs::write(&path, &bytes).unwrap();

        // The decoding path is what reports it: an invalid flag is fatal there, and a
        // compaction that quietly carried on would be writing a log it cannot read back.
        let err = wal.truncate_before(2).unwrap_err();
        assert!(
            format!("{err}").contains("compression flag"),
            "expected the decoder's complaint, got {err}"
        );
    }

    #[test]
    fn the_shortcut_and_the_scan_agree_at_the_boundary() {
        // `truncate_before` skips reading the log when `keep_from` is at or past the next
        // sequence, because nothing can survive. That is the case a snapshot produces -- it
        // covers everything written, and reading 4 GB to find nothing cost 15 s and took
        // resident memory from 8.0 to 14.2 GB. The boundary is worth pinning from both sides:
        // one below it the scan still runs and still keeps the entry it should.
        for (offset, expected_kept) in [(0u64, 0usize), (1, 1)] {
            let dir = TempDir::new().unwrap();
            let mut wal = WalManager::new(dir.path()).unwrap();
            for i in 0..5 {
                wal.write_entry(WalEntry::Delete {
                    table: "t".to_string(),
                    record_id: format!("r{}", i),
                })
                .unwrap();
            }
            wal.force_flush().unwrap();

            let keep_from = wal.next_sequence() - offset;
            wal.truncate_before(keep_from).unwrap();
            let left = wal.replay_from(0).unwrap();
            assert_eq!(
                left.len(),
                expected_kept,
                "keep_from = next_sequence - {offset} should leave {expected_kept} entries, \
                 left {}",
                left.len()
            );
            if expected_kept > 0 {
                assert!(
                    left.iter().all(|(seq, _)| *seq >= keep_from),
                    "an entry below the cutoff survived"
                );
            }
        }
    }

    #[test]
    fn truncating_everything_does_not_rewind_the_sequence() {
        let dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(dir.path()).unwrap();

        for i in 0..5 {
            wal.write_entry(WalEntry::Delete {
                table: "t".to_string(),
                record_id: format!("r{}", i),
            })
            .unwrap();
        }
        wal.force_flush().unwrap();
        let next_before = wal.next_sequence();

        // Compacting past the end empties the log. Deriving the next sequence from an empty
        // file would restart at 1, and those numbers are already referenced by any snapshot
        // manifest taken earlier.
        wal.truncate_before(u64::MAX).unwrap();
        assert!(wal.replay_from(0).unwrap().is_empty());
        assert!(
            wal.next_sequence() >= next_before,
            "sequence went backwards: {} -> {}",
            next_before,
            wal.next_sequence()
        );

        let seq = wal
            .write_entry(WalEntry::Checkpoint { timestamp: 0 })
            .unwrap();
        assert!(seq >= next_before, "reused a sequence number: {}", seq);
    }

    #[test]
    fn test_wal_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(temp_dir.path()).unwrap();

        // Write some entries
        let seq1 = wal
            .write_entry(WalEntry::CreateTable {
                table: "test".to_string(),
                dimension: 128,
            })
            .unwrap();

        let seq2 = wal
            .write_entry(WalEntry::Insert {
                table: "test".to_string(),
                record_id: "rec1".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                payload: None,
            })
            .unwrap();

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
            })
            .unwrap();
        }

        // Second WAL instance (simulating restart)
        {
            let wal = WalManager::new(temp_dir.path()).unwrap();
            assert_eq!(wal.current_sequence(), 1);

            let entries = wal.replay_from(0).unwrap();
            assert_eq!(entries.len(), 1);
        }
    }

    #[test]
    fn truncated_final_frame_is_ignored() {
        let temp_dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(temp_dir.path()).unwrap();
        wal.write_entry(WalEntry::Delete {
            table: "t".to_string(),
            record_id: "complete".to_string(),
        })
        .unwrap();
        wal.write_entry(WalEntry::Delete {
            table: "t".to_string(),
            record_id: "torn".to_string(),
        })
        .unwrap();
        wal.force_flush().unwrap();
        drop(wal);

        let path = temp_dir.path().join("wal.log");
        let len = std::fs::metadata(&path).unwrap().len();
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len - 2).unwrap();

        let wal = WalManager::new(temp_dir.path()).unwrap();
        let entries = wal.replay_from(0).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, 0);
    }

    #[test]
    fn truncated_frame_header_is_ignored() {
        let temp_dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(temp_dir.path()).unwrap();
        wal.write_entry(WalEntry::Checkpoint { timestamp: 1 })
            .unwrap();
        wal.force_flush().unwrap();
        drop(wal);

        let path = temp_dir.path().join("wal.log");
        let len = std::fs::metadata(&path).unwrap().len();
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len + 3).unwrap();

        let wal = WalManager::new(temp_dir.path()).unwrap();
        let entries = wal.replay_from(0).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, 0);
    }

    #[test]
    fn appending_after_a_torn_final_frame_replaces_the_tail() {
        let temp_dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(temp_dir.path()).unwrap();
        wal.write_entry(WalEntry::Delete {
            table: "t".to_string(),
            record_id: "complete".to_string(),
        })
        .unwrap();
        wal.write_entry(WalEntry::Delete {
            table: "t".to_string(),
            record_id: "torn".to_string(),
        })
        .unwrap();
        wal.force_flush().unwrap();
        drop(wal);

        let path = temp_dir.path().join("wal.log");
        let len = std::fs::metadata(&path).unwrap().len();
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len - 2).unwrap();
        drop(file);

        let mut reopened = WalManager::new(temp_dir.path()).unwrap();
        let seq = reopened
            .write_entry(WalEntry::Delete {
                table: "t".to_string(),
                record_id: "replacement".to_string(),
            })
            .unwrap();
        reopened.force_flush().unwrap();

        let entries = reopened.replay_from(0).unwrap();
        assert_eq!(seq, 1);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].0, 1);
        match &entries[1].1 {
            WalEntry::Delete { record_id, .. } => assert_eq!(record_id, "replacement"),
            other => panic!("unexpected replacement entry: {other:?}"),
        }
    }

    #[test]
    fn corruption_in_a_middle_frame_is_fatal() {
        let temp_dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(temp_dir.path()).unwrap();
        wal.write_entry(WalEntry::Delete {
            table: "t".to_string(),
            record_id: "first".to_string(),
        })
        .unwrap();
        wal.write_entry(WalEntry::Delete {
            table: "t".to_string(),
            record_id: "second".to_string(),
        })
        .unwrap();
        wal.force_flush().unwrap();
        drop(wal);

        // The first frame starts with a 13-byte header. 0xc1 is the MessagePack "reserved"
        // marker, so deserializing this complete middle frame must fail rather than being treated
        // as a crash tail and skipped.
        let path = temp_dir.path().join("wal.log");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[13] = 0xc1;
        std::fs::write(&path, bytes).unwrap();

        let wal = WalManager::new(temp_dir.path()).unwrap();
        for from_seq in [0, 1] {
            let err = wal.replay_from(from_seq).unwrap_err();
            assert!(matches!(err, Error::Serialization(_)));
        }
    }

    #[test]
    fn invalid_compression_flag_is_fatal() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("wal.log");
        let mut file = File::create(&path).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.write_all(&[2u8]).unwrap();
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.sync_all().unwrap();

        let wal = WalManager::new(temp_dir.path()).unwrap();
        let err = wal.replay_from(0).unwrap_err();
        assert!(format!("{err}").contains("compression flag"));
    }

    #[cfg(feature = "test-failpoints")]
    #[test]
    fn failed_sync_does_not_duplicate_a_frame_on_drop() {
        let temp_dir = TempDir::new().unwrap();
        let mut wal = WalManager::new(temp_dir.path()).unwrap();
        wal.write_entry(WalEntry::Checkpoint { timestamp: 1 })
            .unwrap();
        wal.fail_next_sync_for_test();
        assert!(wal.force_flush().is_err());
        drop(wal);

        let reopened = WalManager::new(temp_dir.path()).unwrap();
        let entries = reopened.replay_from(0).unwrap();
        assert_eq!(
            entries.len(),
            1,
            "a failed sync must not duplicate the frame"
        );
        assert_eq!(entries[0].0, 0);
    }

    #[test]
    fn historical_messagepack_entries_still_decode() {
        // These bytes were emitted by the pre-V2 WAL writer. MessagePack uses the externally
        // tagged variant name, so this fixture also protects against accidentally changing the
        // old field shapes while adding new variants.
        let fixtures: &[(&[u8], &str)] = &[
            (
                &[
                    0x81, 0xa6, b'D', b'e', b'l', b'e', b't', b'e', 0x92, 0xa1, b't', 0xa1, b'r',
                ],
                "delete",
            ),
            (
                &[
                    0x81, 0xab, b'C', b'r', b'e', b'a', b't', b'e', b'T', b'a', b'b', b'l', b'e',
                    0x92, 0xa1, b't', 0x03,
                ],
                "create",
            ),
            (
                &[
                    0x81, 0xab, b'D', b'e', b'l', b'e', b't', b'e', b'T', b'a', b'b', b'l', b'e',
                    0x91, 0xa1, b't',
                ],
                "drop",
            ),
            (
                &[
                    0x81, 0xaa, b'C', b'h', b'e', b'c', b'k', b'p', b'o', b'i', b'n', b't', 0x91,
                    0x07,
                ],
                "checkpoint",
            ),
        ];

        for (bytes, kind) in fixtures {
            let entry: WalEntry = rmp_serde::from_slice(bytes).unwrap();
            match (kind, entry) {
                (&"delete", WalEntry::Delete { table, record_id }) => {
                    assert_eq!(table, "t");
                    assert_eq!(record_id, "r");
                }
                (&"create", WalEntry::CreateTable { table, dimension }) => {
                    assert_eq!((table, dimension), ("t".to_string(), 3));
                }
                (&"drop", WalEntry::DeleteTable { table }) => assert_eq!(table, "t"),
                (&"checkpoint", WalEntry::Checkpoint { timestamp }) => assert_eq!(timestamp, 7),
                _ => panic!(
                    "historical {} WAL fixture decoded as the wrong variant",
                    kind
                ),
            }
        }
    }

    #[test]
    fn sequence_continues_after_reopen() {
        let temp_dir = TempDir::new().unwrap();
        let first = {
            let mut wal = WalManager::new(temp_dir.path()).unwrap();
            let seq = wal
                .write_entry(WalEntry::Checkpoint { timestamp: 1 })
                .unwrap();
            wal.force_flush().unwrap();
            seq
        };

        let mut reopened = WalManager::new(temp_dir.path()).unwrap();
        let next = reopened
            .write_entry(WalEntry::Checkpoint { timestamp: 2 })
            .unwrap();
        assert_eq!(first, 0);
        assert_eq!(next, 1);
    }
}
