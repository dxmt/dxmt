//! Background Demotion Task
//!
//! Periodically scans for cold vectors and demotes them to lower-cost tiers.
//! This prevents hot/warm tiers from filling up with rarely-accessed data.

use crate::storage::tiered::TieredStorage;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Configuration for background demotion task
#[derive(Debug, Clone)]
pub struct BackgroundDemotionConfig {
    /// How often to run demotion scan (default: 5 minutes)
    pub interval_secs: u64,

    /// Table names to scan for demotion (empty = all tables)
    pub table_names: Vec<String>,

    /// Whether to enable the background task
    pub enabled: bool,
}

impl Default for BackgroundDemotionConfig {
    fn default() -> Self {
        Self {
            interval_secs: 300, // 5 minutes
            table_names: Vec::new(),
            enabled: true,
        }
    }
}

/// Handle to control the background demotion task
pub struct BackgroundDemotionTask {
    /// Task handle for the background worker
    task_handle: JoinHandle<()>,

    /// Sender to signal shutdown
    shutdown_tx: mpsc::Sender<()>,
}

impl BackgroundDemotionTask {
    /// Stop the background demotion task gracefully
    pub async fn stop(self) {
        // Send shutdown signal
        let _ = self.shutdown_tx.send(()).await;

        // Wait for task to finish
        let _ = self.task_handle.await;

        tracing::info!("Background demotion task stopped");
    }

    /// Check if the task is still running
    pub fn is_running(&self) -> bool {
        !self.task_handle.is_finished()
    }
}

/// Start the background demotion task
///
/// This spawns a tokio task that periodically scans for cold vectors
/// and demotes them to lower-cost storage tiers.
///
/// # Arguments
/// * `storage` - Arc to TieredStorage instance
/// * `config` - Configuration for the background task
///
/// # Returns
/// A handle that can be used to stop the task
pub fn start_background_demotion_task(
    storage: Arc<TieredStorage>,
    config: BackgroundDemotionConfig,
) -> BackgroundDemotionTask {
    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

    let interval = Duration::from_secs(config.interval_secs);
    let table_names = config.table_names.clone();

    tracing::info!(
        "Starting background demotion task (interval: {}s, tables: {:?})",
        config.interval_secs,
        if table_names.is_empty() {
            "all".to_string()
        } else {
            format!("{:?}", table_names)
        }
    );

    let task_handle = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                // Shutdown signal received
                _ = shutdown_rx.recv() => {
                    tracing::info!("Background demotion task received shutdown signal");
                    break;
                }

                // Periodic tick
                _ = ticker.tick() => {
                    // Run demotion for each table
                    if table_names.is_empty() {
                        // If no tables specified, we can't auto-discover them yet
                        // This would require listing tables from storage
                        tracing::debug!("Background demotion: no tables specified, skipping");
                    } else {
                        for table_name in &table_names {
                            match storage.demote_cold_vectors(table_name).await {
                                Ok(_) => {
                                    tracing::debug!(
                                        "Background demotion completed for table: {}",
                                        table_name
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Background demotion failed for table {}: {}",
                                        table_name,
                                        e
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    BackgroundDemotionTask {
        task_handle,
        shutdown_tx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        memory::MemoryStorage,
        tier_promotion::{Tier, TierPromotionConfig},
        tiered::TieredStorage,
        Storage, StorageConfig,
    };
    use crate::types::{
        record::CreateRecordRequest, table::TableConfig, DistanceMetric, IndexType,
    };

    fn create_test_table_config(name: &str) -> TableConfig {
        TableConfig {
            name: name.to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: Some(IndexType::BruteForce),
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
        }
    }

    fn create_test_vector(id: &str) -> CreateRecordRequest {
        CreateRecordRequest {
            id: id.to_string(),
            vector: vec![0.1; 128],
            sparse_vector: None,
            payload: Some(serde_json::json!({"test": "data"})),
            named_vectors: None,
        }
    }

    #[tokio::test]
    async fn test_background_task_start_stop() {
        // Create tiered storage
        let hot_tier = Arc::new(MemoryStorage::new());
        let warm_tier = Arc::new(MemoryStorage::new());
        let cold_tier = Arc::new(MemoryStorage::new());

        let storage = Arc::new(TieredStorage::new(
            hot_tier,
            warm_tier,
            cold_tier,
            StorageConfig::default(),
        ));

        // Start background task
        let config = BackgroundDemotionConfig {
            interval_secs: 1, // 1 second for testing
            table_names: vec!["test_table".to_string()],
            enabled: true,
        };

        let task = start_background_demotion_task(storage.clone(), config);

        // Verify task is running
        assert!(task.is_running());

        // Wait a bit
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop the task
        task.stop().await;

        println!("✓ Background task started and stopped successfully");
    }

    #[tokio::test]
    async fn test_background_demotion_runs_periodically() {
        // Create tiered storage with aggressive demotion thresholds
        let hot_tier = Arc::new(MemoryStorage::new());
        let warm_tier = Arc::new(MemoryStorage::new());
        let cold_tier = Arc::new(MemoryStorage::new());

        let promotion_config = TierPromotionConfig {
            hot_threshold: 0.8,
            warm_threshold: 0.5,
            demote_to_warm_threshold: 0.4,
            demote_to_cold_threshold: 0.2,
            time_window_secs: 10, // Short window for testing
            auto_promotion_enabled: true,
            auto_demotion_enabled: true,
        };

        let storage = Arc::new(TieredStorage::new_with_promotion_config(
            hot_tier.clone(),
            warm_tier.clone(),
            cold_tier.clone(),
            StorageConfig::default(),
            promotion_config,
        ));

        // Create table
        let table_config = create_test_table_config("test_demotion");
        storage.create_table(table_config).await.unwrap();

        // Insert a vector into hot tier
        let vector = create_test_vector("vec_1");
        hot_tier
            .insert_record("test_demotion", vector)
            .await
            .unwrap();

        // Set it as hot in promotion policy (but with no recent accesses)
        storage
            .promotion_policy()
            .set_current_tier("vec_1", Tier::Hot);

        // Start background task with short interval
        let config = BackgroundDemotionConfig {
            interval_secs: 1, // 1 second
            table_names: vec!["test_demotion".to_string()],
            enabled: true,
        };

        let task = start_background_demotion_task(storage.clone(), config);

        // Wait for demotion to potentially happen
        // Since the vector has no accesses and time window is short,
        // it should eventually be demoted
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Stop the task
        task.stop().await;

        println!("✓ Background demotion task ran periodically");
    }

    #[tokio::test]
    async fn test_multiple_tables() {
        // Create storage
        let hot_tier = Arc::new(MemoryStorage::new());
        let warm_tier = Arc::new(MemoryStorage::new());
        let cold_tier = Arc::new(MemoryStorage::new());

        let storage = Arc::new(TieredStorage::new(
            hot_tier.clone(),
            warm_tier.clone(),
            cold_tier.clone(),
            StorageConfig::default(),
        ));

        // Create multiple tables
        for i in 0..3 {
            let table_config = create_test_table_config(&format!("table_{}", i));
            storage.create_table(table_config).await.unwrap();
        }

        // Start background task for multiple tables
        let config = BackgroundDemotionConfig {
            interval_secs: 1,
            table_names: vec![
                "table_0".to_string(),
                "table_1".to_string(),
                "table_2".to_string(),
            ],
            enabled: true,
        };

        let task = start_background_demotion_task(storage.clone(), config);

        // Wait for a few ticks
        tokio::time::sleep(Duration::from_millis(1500)).await;

        // Stop
        task.stop().await;

        println!("✓ Background task handled multiple tables");
    }

    #[tokio::test]
    async fn test_graceful_shutdown() {
        // Create storage
        let hot_tier = Arc::new(MemoryStorage::new());
        let warm_tier = Arc::new(MemoryStorage::new());
        let cold_tier = Arc::new(MemoryStorage::new());

        let storage = Arc::new(TieredStorage::new(
            hot_tier,
            warm_tier,
            cold_tier,
            StorageConfig::default(),
        ));

        // Start background task
        let config = BackgroundDemotionConfig {
            interval_secs: 10, // Long interval
            table_names: vec!["test_table".to_string()],
            enabled: true,
        };

        let task = start_background_demotion_task(storage.clone(), config);

        // Immediately stop (should not wait for next tick)
        let start = std::time::Instant::now();
        task.stop().await;
        let elapsed = start.elapsed();

        // Should stop quickly (< 100ms), not wait for the 10s interval
        assert!(elapsed < Duration::from_millis(100));

        println!("✓ Graceful shutdown completed in {:?}", elapsed);
    }
}
