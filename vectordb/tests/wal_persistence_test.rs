#![cfg(feature = "storage-lmdb")]

//! WAL Persistence Test
//!
//! Tests that WAL entries are properly persisted to LMDB storage
//! and can be recovered after restart.

use dxmt_vectordb::{
    error::Result,
    storage::{
        atomic_ops::{AtomicStorageManager, StorageBackends},
        lmdb::{LmdbStorage, LmdbStorageImpl},
        traits::_Storage,
    },
    wal::{WalEntry, WalManager},
};
use std::sync::Arc;
use tempfile::TempDir;

/// Test WAL persistence to LMDB backend
#[tokio::test]
async fn test_wal_persistence_to_lmdb() -> Result<()> {
    // Create temporary directories for WAL and LMDB
    let wal_dir = TempDir::new().unwrap();
    let lmdb_dir = TempDir::new().unwrap();

    // Create LMDB storage with Storage trait implementation
    let lmdb = LmdbStorage::new(lmdb_dir.path(), 100 * 1024 * 1024).unwrap();
    let lmdb_storage = Arc::new(LmdbStorageImpl::new(lmdb));

    // Create WAL manager
    let wal = WalManager::new(wal_dir.path()).unwrap();

    // Create atomic storage manager with LMDB backend
    let backends = StorageBackends {
        #[cfg(feature = "storage-lmdb")]
        lmdb: Some(lmdb_storage.clone() as Arc<dyn _Storage>),
        #[cfg(feature = "storage-rocksdb")]
        rocksdb: None,
    };

    let manager = AtomicStorageManager::new(wal, backends);

    // Create a table via WAL
    let create_table_entry = WalEntry::CreateTable {
        table: "test_table".to_string(),
        dimension: 128,
    };

    // Write to WAL
    let tx_id = manager.begin_transaction().await?;
    manager.add_operation(&tx_id, create_table_entry).await?;
    manager.commit_transaction(&tx_id).await?;

    // Verify table was created in LMDB
    let table = lmdb_storage.get_table("test_table").await?;
    assert_eq!(table.name, "test_table");
    assert_eq!(table.dimension, 128);

    // Insert a record via WAL
    let insert_entry = WalEntry::Insert {
        table: "test_table".to_string(),
        record_id: "rec1".to_string(),
        vector: vec![0.1; 128],
        payload: Some(serde_json::json!({"name": "test"})),
    };

    let tx_id = manager.begin_transaction().await?;
    manager.add_operation(&tx_id, insert_entry).await?;
    manager.commit_transaction(&tx_id).await?;

    // Verify record was persisted to LMDB
    let record = lmdb_storage
        .get_record("test_table", &"rec1".to_string())
        .await?;
    assert_eq!(record.id, "rec1");
    assert_eq!(record.vector.len(), 128);
    assert_eq!(record.payload["name"], "test");

    println!("✅ WAL entries successfully persisted to LMDB");
    Ok(())
}

/// Test WAL replay on startup (data recovery)
#[tokio::test]
async fn test_wal_replay_on_startup() -> Result<()> {
    let wal_dir = TempDir::new().unwrap();
    let lmdb_dir = TempDir::new().unwrap();
    let wal_path = wal_dir.path().to_path_buf();
    let lmdb_path = lmdb_dir.path().to_path_buf();

    // === Phase 1: Write data with WAL ===
    {
        let lmdb = LmdbStorage::new(&lmdb_path, 100 * 1024 * 1024).unwrap();
        let lmdb_storage = Arc::new(LmdbStorageImpl::new(lmdb));

        let wal = WalManager::new(&wal_path).unwrap();

        let backends = StorageBackends {
            #[cfg(feature = "storage-lmdb")]
            lmdb: Some(lmdb_storage.clone() as Arc<dyn _Storage>),
            #[cfg(feature = "storage-rocksdb")]
            rocksdb: None,
        };

        let manager = AtomicStorageManager::new(wal, backends);

        // Create table
        let tx_id = manager.begin_transaction().await?;
        manager
            .add_operation(
                &tx_id,
                WalEntry::CreateTable {
                    table: "recovery_test".to_string(),
                    dimension: 64,
                },
            )
            .await?;
        manager.commit_transaction(&tx_id).await?;

        // Insert records
        for i in 0..5 {
            let tx_id = manager.begin_transaction().await?;
            manager
                .add_operation(
                    &tx_id,
                    WalEntry::Insert {
                        table: "recovery_test".to_string(),
                        record_id: format!("rec_{}", i),
                        vector: vec![i as f32 * 0.1; 64],
                        payload: Some(serde_json::json!({"index": i})),
                    },
                )
                .await?;
            manager.commit_transaction(&tx_id).await?;
        }

        // Verify data persisted
        let record = lmdb_storage
            .get_record("recovery_test", &"rec_3".to_string())
            .await?;
        assert_eq!(record.id, "rec_3");
        println!("✅ Phase 1: Data written successfully");

        // Drop manager to simulate shutdown
    }

    // === Phase 2: Restart and replay WAL ===
    {
        let lmdb = LmdbStorage::new(&lmdb_path, 100 * 1024 * 1024).unwrap();
        let lmdb_storage = Arc::new(LmdbStorageImpl::new(lmdb));

        let wal = WalManager::new(&wal_path).unwrap();

        let backends = StorageBackends {
            #[cfg(feature = "storage-lmdb")]
            lmdb: Some(lmdb_storage.clone() as Arc<dyn _Storage>),
            #[cfg(feature = "storage-rocksdb")]
            rocksdb: None,
        };

        let manager = AtomicStorageManager::new(wal, backends);

        // Replay WAL on startup
        let replayed_count = manager.replay_wal_on_startup().await?;
        println!("✅ Phase 2: Replayed {} WAL entries", replayed_count);

        // Verify all data is still accessible
        let table = lmdb_storage.get_table("recovery_test").await?;
        assert_eq!(table.name, "recovery_test");
        assert_eq!(table.dimension, 64);

        for i in 0..5 {
            let record = lmdb_storage
                .get_record("recovery_test", &format!("rec_{}", i))
                .await?;
            assert_eq!(record.payload["index"], i);
        }

        println!("✅ All data successfully recovered after restart");
    }

    Ok(())
}

/// Test that Delete operations are properly persisted
#[tokio::test]
async fn test_delete_persistence() -> Result<()> {
    let wal_dir = TempDir::new().unwrap();
    let lmdb_dir = TempDir::new().unwrap();

    let lmdb = LmdbStorage::new(lmdb_dir.path(), 100 * 1024 * 1024).unwrap();
    let lmdb_storage = Arc::new(LmdbStorageImpl::new(lmdb));

    let wal = WalManager::new(wal_dir.path()).unwrap();

    let backends = StorageBackends {
        #[cfg(feature = "storage-lmdb")]
        lmdb: Some(lmdb_storage.clone() as Arc<dyn _Storage>),
        #[cfg(feature = "storage-rocksdb")]
        rocksdb: None,
    };

    let manager = AtomicStorageManager::new(wal, backends);

    // Create table
    let tx_id = manager.begin_transaction().await?;
    manager
        .add_operation(
            &tx_id,
            WalEntry::CreateTable {
                table: "delete_test".to_string(),
                dimension: 32,
            },
        )
        .await?;
    manager.commit_transaction(&tx_id).await?;

    // Insert record
    let tx_id = manager.begin_transaction().await?;
    manager
        .add_operation(
            &tx_id,
            WalEntry::Insert {
                table: "delete_test".to_string(),
                record_id: "to_delete".to_string(),
                vector: vec![1.0; 32],
                payload: None,
            },
        )
        .await?;
    manager.commit_transaction(&tx_id).await?;

    // Verify insert
    let record = lmdb_storage
        .get_record("delete_test", &"to_delete".to_string())
        .await?;
    assert_eq!(record.id, "to_delete");

    // Delete record
    let tx_id = manager.begin_transaction().await?;
    manager
        .add_operation(
            &tx_id,
            WalEntry::Delete {
                table: "delete_test".to_string(),
                record_id: "to_delete".to_string(),
            },
        )
        .await?;
    manager.commit_transaction(&tx_id).await?;

    // Verify deletion persisted
    let result = lmdb_storage
        .get_record("delete_test", &"to_delete".to_string())
        .await;
    assert!(result.is_err(), "Record should be deleted");

    println!("✅ Delete operations successfully persisted");
    Ok(())
}

/// Test crash recovery with partial transactions
#[tokio::test]
async fn test_crash_recovery() -> Result<()> {
    let wal_dir = TempDir::new().unwrap();
    let lmdb_dir = TempDir::new().unwrap();
    let wal_path = wal_dir.path().to_path_buf();
    let lmdb_path = lmdb_dir.path().to_path_buf();

    // Write some WAL entries then "crash"
    {
        let lmdb = LmdbStorage::new(&lmdb_path, 100 * 1024 * 1024).unwrap();
        let lmdb_storage = Arc::new(LmdbStorageImpl::new(lmdb));
        let wal = WalManager::new(&wal_path).unwrap();

        let backends = StorageBackends {
            #[cfg(feature = "storage-lmdb")]
            lmdb: Some(lmdb_storage as Arc<dyn _Storage>),
            #[cfg(feature = "storage-rocksdb")]
            rocksdb: None,
        };

        let manager = AtomicStorageManager::new(wal, backends);

        // Create table
        let tx_id = manager.begin_transaction().await?;
        manager
            .add_operation(
                &tx_id,
                WalEntry::CreateTable {
                    table: "crash_test".to_string(),
                    dimension: 16,
                },
            )
            .await?;
        manager.commit_transaction(&tx_id).await?;

        // Insert some records
        let tx_id = manager.begin_transaction().await?;
        manager
            .add_operation(
                &tx_id,
                WalEntry::Insert {
                    table: "crash_test".to_string(),
                    record_id: "before_crash".to_string(),
                    vector: vec![1.0; 16],
                    payload: None,
                },
            )
            .await?;
        manager.commit_transaction(&tx_id).await?;

        // Simulate crash (drop without cleanup)
    }

    // Recover from crash
    {
        let lmdb = LmdbStorage::new(&lmdb_path, 100 * 1024 * 1024).unwrap();
        let lmdb_storage = Arc::new(LmdbStorageImpl::new(lmdb));
        let wal = WalManager::new(&wal_path).unwrap();

        let backends = StorageBackends {
            #[cfg(feature = "storage-lmdb")]
            lmdb: Some(lmdb_storage.clone() as Arc<dyn _Storage>),
            #[cfg(feature = "storage-rocksdb")]
            rocksdb: None,
        };

        let manager = AtomicStorageManager::new(wal, backends);

        // Recover using the existing method
        manager.recover_from_crash(0).await?;

        // Verify data recovered
        let table = lmdb_storage.get_table("crash_test").await?;
        assert_eq!(table.name, "crash_test");

        let record = lmdb_storage
            .get_record("crash_test", &"before_crash".to_string())
            .await?;
        assert_eq!(record.id, "before_crash");

        println!("✅ Successfully recovered from crash");
    }

    Ok(())
}
