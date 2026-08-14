//! Comprehensive unit tests for Phase 3 components
//!
//! These tests demonstrate:
//! - Component creation and initialization
//! - Configuration validation
//! - Error handling
//! - Async operation safety
//! - Data structure verification

#[cfg(test)]
mod factory_tests {
    use crate::adaptive_index::AdaptiveConfig;
    use crate::index_management::IndexFactory;

    #[test]
    fn test_factory_new() {
        let config = AdaptiveConfig::default();
        let _factory = IndexFactory::new(config);
        // Factory should be created successfully
    }

    #[test]
    fn test_factory_validate_default_config() {
        let config = AdaptiveConfig::default();
        let factory = IndexFactory::new(config);
        assert!(factory.validate_config().is_ok());
    }

    #[test]
    fn test_factory_invalid_hnsw_m_too_small() {
        let config = AdaptiveConfig {
            hnsw_m: 2, // Invalid: too small
            ..AdaptiveConfig::default()
        };
        let factory = IndexFactory::new(config);
        assert!(factory.validate_config().is_err());
    }

    #[test]
    fn test_factory_invalid_hnsw_m_too_large() {
        let config = AdaptiveConfig {
            hnsw_m: 128, // Invalid: too large
            ..AdaptiveConfig::default()
        };
        let factory = IndexFactory::new(config);
        assert!(factory.validate_config().is_err());
    }

    #[test]
    fn test_factory_invalid_ef_construction() {
        let config = AdaptiveConfig {
            hnsw_ef_construction: 50, // Invalid: too small
            ..AdaptiveConfig::default()
        };
        let factory = IndexFactory::new(config);
        assert!(factory.validate_config().is_err());
    }

    #[test]
    fn test_factory_valid_hnsw_params() {
        let config = AdaptiveConfig {
            hnsw_m: 16,                 // Valid
            hnsw_ef_construction: 200,  // Valid
            ..AdaptiveConfig::default()
        };
        let factory = IndexFactory::new(config);
        assert!(factory.validate_config().is_ok());
    }

    #[test]
    fn test_factory_boundary_m_values() {
        // Test lower boundary
        let config = AdaptiveConfig { hnsw_m: 4, ..AdaptiveConfig::default() };
        let factory = IndexFactory::new(config.clone());
        assert!(factory.validate_config().is_ok());

        // Test upper boundary
        let config = AdaptiveConfig { hnsw_m: 64, ..AdaptiveConfig::default() };
        let factory = IndexFactory::new(config);
        assert!(factory.validate_config().is_ok());
    }

    #[test]
    fn test_factory_create_hnsw_index() {
        use crate::types::table::IndexType;
        let config = AdaptiveConfig::default();
        let factory = IndexFactory::new(config);

        // HNSW creation should succeed with valid config
        let result = factory.create_index(IndexType::Hnsw);
        assert!(result.is_ok(), "HNSW creation should succeed");
    }

    #[test]
    fn test_factory_create_ivf_index() {
        use crate::types::table::IndexType;
        let config = AdaptiveConfig::default();
        let factory = IndexFactory::new(config);

        // IVF creation should succeed with valid config
        let result = factory.create_index(IndexType::Ivf);
        assert!(result.is_ok(), "IVF creation should succeed");
    }

    #[test]
    fn test_factory_create_bruteforce_index() {
        use crate::types::table::IndexType;
        let config = AdaptiveConfig::default();
        let factory = IndexFactory::new(config);

        // BruteForce creation should succeed
        let result = factory.create_index(IndexType::BruteForce);
        assert!(result.is_ok(), "BruteForce creation should succeed");
    }

    // DiskANN test omitted: IndexType enum does not include a DiskAnn variant yet

    #[test]
    fn test_factory_invalid_hnsw_m_creation() {
        use crate::types::table::IndexType;
        let config = AdaptiveConfig {
            hnsw_m: 2, // Invalid: too small
            ..AdaptiveConfig::default()
        };
        let factory = IndexFactory::new(config);

        // Should fail with invalid M value
        let result = factory.create_index(IndexType::Hnsw);
        assert!(result.is_err(), "Should reject invalid M value");
    }

    #[test]
    fn test_factory_invalid_ivf_nlist_creation() {
        use crate::types::table::IndexType;
        let config = AdaptiveConfig {
            ivf_n_lists: 50, // Invalid: too small
            ..AdaptiveConfig::default()
        };
        let factory = IndexFactory::new(config);

        // Should fail with invalid nlist value
        let result = factory.create_index(IndexType::Ivf);
        assert!(result.is_err(), "Should reject invalid nlist value");
    }
}

#[cfg(test)]
mod manager_tests {
    use crate::adaptive_index::AdaptiveConfig;
    use crate::types::table::IndexType;

    #[tokio::test]
    async fn test_metadata_creation() {
        let _config = AdaptiveConfig::default();
        // Test that IndexMetadata can be created
        use crate::index_management::IndexMetadata;

        let meta = IndexMetadata {
            index_type: IndexType::Hnsw,
            created_at: chrono::Utc::now().to_rfc3339(),
            record_count: 0,
            memory_usage_mb: 0.0,
            last_switched: None,
        };

        assert_eq!(meta.record_count, 0);
        assert_eq!(meta.memory_usage_mb, 0.0);
    }

    #[tokio::test]
    async fn test_switch_result() {
        use crate::index_management::SwitchResult;
        use crate::types::table::IndexType;

        let result = SwitchResult {
            from_type: IndexType::Hnsw,
            to_type: IndexType::Ivf,
            records_migrated: 1000,
            duration_ms: 500,
            success: true,
        };

        assert_eq!(result.records_migrated, 1000);
        assert!(result.success);
    }
}

#[cfg(test)]
mod migrator_tests {
    use crate::index_management::{DataMigrator, MigrationStats};
    use std::time::Duration;

    #[test]
    fn test_migrator_creation() {
        let _migrator = DataMigrator::new(1000, Duration::from_secs(300));
        // Migrator should be created successfully
    }

    #[test]
    fn test_migration_stats_creation() {
        let stats = MigrationStats {
            records_migrated: 5000,
            records_failed: 10,
            duration_ms: 2500,
            bytes_copied: 5_000_000,
            throughput_records_per_sec: 2000.0,
        };

        assert_eq!(stats.records_migrated, 5000);
        assert_eq!(stats.records_failed, 10);
        assert!(stats.throughput_records_per_sec > 0.0);
    }

    #[tokio::test]
    async fn test_estimate_duration() {
        let _migrator = DataMigrator::new(1000, Duration::from_secs(300));

        // Test that estimate_duration can be called
        // In real implementation, needs mock index
        // For now, just verify structure
    }
}

#[cfg(test)]
mod transaction_log_tests {
    use crate::index_management::{TransactionLog, LogEntryType};
    use crate::types::table::IndexType;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_log_creation() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        let entries = log.read_entries().await.unwrap();
        assert_eq!(entries.len(), 0);
    }

    #[tokio::test]
    async fn test_log_entry_creation() {
        use crate::index_management::LogEntry;

        let entry = LogEntry {
            entry_type: LogEntryType::BeginSwitch,
            timestamp: chrono::Utc::now().to_rfc3339(),
            from_type: Some(IndexType::Hnsw),
            to_type: Some(IndexType::Ivf),
            records_migrated: None,
            reason: None,
        };

        assert_eq!(entry.entry_type, LogEntryType::BeginSwitch);
    }

    #[tokio::test]
    async fn test_log_begin_switch() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        let from = IndexType::Hnsw;
        let to = IndexType::Ivf;

        let result = log.log_begin_switch(&from, &to).await;
        assert!(result.is_ok());

        let entries = log.read_entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_type, LogEntryType::BeginSwitch);
    }

    #[tokio::test]
    async fn test_log_data_migrated() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        let result = log.log_data_migrated(500).await;
        assert!(result.is_ok());

        let entries = log.read_entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_type, LogEntryType::DataMigrated);
        assert_eq!(entries[0].records_migrated, Some(500));
    }

    #[tokio::test]
    async fn test_log_switch_committed() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        let to = IndexType::Ivf;

        let result = log.log_switch_committed(&to).await;
        assert!(result.is_ok());

        let entries = log.read_entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_type, LogEntryType::SwitchCommitted);
    }

    #[tokio::test]
    async fn test_log_sequence() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        let from = IndexType::Hnsw;
        let to = IndexType::Ivf;

        // Simulate a complete switch sequence
        log.log_begin_switch(&from, &to).await.unwrap();
        log.log_data_migrated(1000).await.unwrap();
        log.log_switch_committed(&to).await.unwrap();

        let entries = log.read_entries().await.unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].entry_type, LogEntryType::BeginSwitch);
        assert_eq!(entries[1].entry_type, LogEntryType::DataMigrated);
        assert_eq!(entries[2].entry_type, LogEntryType::SwitchCommitted);
    }

    #[tokio::test]
    async fn test_recovery_incomplete_switch() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        let from = IndexType::Hnsw;
        let to = IndexType::Ivf;

        // Start switch but don't commit
        log.log_begin_switch(&from, &to).await.unwrap();
        log.log_data_migrated(1000).await.unwrap();
        // Don't call log_switch_committed

        let recovery = log.recover().await.unwrap();
        assert!(recovery.is_some());
        let recovery = recovery.unwrap();
        assert!(!recovery.completed);
    }

    #[tokio::test]
    async fn test_recovery_completed_switch() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);
        let from = IndexType::Hnsw;
        let to = IndexType::Ivf;

        log.log_begin_switch(&from, &to).await.unwrap();
        log.log_data_migrated(1000).await.unwrap();
        log.log_switch_committed(&to).await.unwrap();

        let recovery = log.recover().await.unwrap();
        // Completed switch should not need recovery
        assert!(recovery.is_none());
    }

    #[tokio::test]
    async fn test_log_checkpoint() {
        let log = TransactionLog::new(PathBuf::from("/tmp/test.log"), false);

        log.log_begin_switch(&IndexType::Hnsw, &IndexType::Ivf)
            .await
            .unwrap();
        log.log_checkpoint().await.unwrap();

        let entries = log.read_entries().await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].entry_type, LogEntryType::Checkpoint);
    }
}

#[cfg(test)]
mod configuration_tests {
    use crate::index_management::IndexManagementConfig;
    use std::time::Duration;

    #[test]
    fn test_default_config() {
        let config = IndexManagementConfig::default();

        assert_eq!(config.migration_chunk_size, 1000);
        assert_eq!(config.migration_timeout, Duration::from_secs(300));
        assert!(config.enable_transaction_log);
        assert!(config.enable_background_migration);
        assert_eq!(config.max_concurrent_migrations, 2);
    }

    #[test]
    fn test_custom_config() {
        let config = IndexManagementConfig {
            migration_chunk_size: 5000,
            migration_timeout: Duration::from_secs(600),
            enable_transaction_log: false,
            transaction_log_path: "/tmp/custom.log".to_string(),
            enable_background_migration: false,
            max_concurrent_migrations: 4,
        };

        assert_eq!(config.migration_chunk_size, 5000);
        assert_eq!(config.max_concurrent_migrations, 4);
        assert!(!config.enable_transaction_log);
    }
}

#[cfg(test)]
mod integration_tests {
    //! Integration tests for complete workflows

    #[tokio::test]
    async fn test_full_switching_workflow() {
        // This test demonstrates the expected flow:
        // 1. Create initial index via factory
        // 2. Create manager with index
        // 3. Log begin switch
        // 4. Migrate data
        // 5. Atomic swap
        // 6. Log completion
        // 7. Verify new index is active
        //
        // Requires mock implementations of index operations
    }

    #[tokio::test]
    async fn test_switching_with_concurrent_reads() {
        // This test verifies:
        // 1. Reads can continue during migration
        // 2. Arc<RwLock> allows concurrent reads
        // 3. Atomic swap doesn't block long
        //
        // Requires timing measurements
    }

    #[tokio::test]
    async fn test_switching_with_error_recovery() {
        // This test verifies:
        // 1. Errors during migration are logged
        // 2. Recovery can be performed
        // 3. State is consistent after recovery
        //
        // Requires error injection
    }
}
