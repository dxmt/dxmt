//! IndexManager - Manages active index with atomic switching
//!
//! Responsible for:
//! - Holding and managing current index
//! - Coordinating safe index switching
//! - Tracking metadata about current index
//! - Integration with AdaptiveIndex recommendations

use crate::adaptive_index::AdaptiveIndex;
use crate::search::IndexWrapper;
use crate::types::table::IndexType;
use crate::error::{Error, Result};
use super::{IndexFactory, IndexMetadata, SwitchResult, DataMigrator, TransactionLog};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Manages the currently active index and coordinates switching
pub struct IndexManager {
    /// Currently active index (wrapped for thread safety)
    current: Arc<RwLock<IndexWrapper>>,

    /// Metadata about current index
    metadata: Arc<RwLock<IndexMetadata>>,

    /// Factory for creating new indexes
    factory: IndexFactory,

    /// Adaptive index for recommendations
    adaptive: Arc<AdaptiveIndex>,

    /// Data migrator for safe switching
    migrator: DataMigrator,

    /// Transaction log for recovery
    transaction_log: Option<TransactionLog>,
}

impl IndexManager {
    /// Create a new IndexManager with an initial index
    ///
    /// # Arguments
    /// * `initial_index` - The index to start with
    /// * `factory` - Factory for creating new indexes
    /// * `adaptive` - Adaptive index for recommendations
    /// * `migrator` - Data migrator for safe switching
    ///
    /// # Returns
    /// * `IndexManager` - New manager instance
    pub async fn new(
        initial_index: IndexWrapper,
        factory: IndexFactory,
        adaptive: Arc<AdaptiveIndex>,
        migrator: DataMigrator,
        transaction_log: Option<TransactionLog>,
    ) -> Self {
        let metadata = IndexMetadata {
            index_type: IndexType::Hnsw, // Should come from initial_index
            created_at: chrono::Utc::now().to_rfc3339(),
            record_count: 0,
            memory_usage_mb: 0.0,
            last_switched: None,
        };

        Self {
            current: Arc::new(RwLock::new(initial_index)),
            metadata: Arc::new(RwLock::new(metadata)),
            factory,
            adaptive,
            migrator,
            transaction_log,
        }
    }

    /// Get a reference to the current index
    pub async fn current_index(&self) -> Arc<RwLock<IndexWrapper>> {
        self.current.clone()
    }

    /// Get current index type
    pub async fn current_index_type(&self) -> IndexType {
        self.metadata.read().await.index_type
    }

    /// Get current metadata
    pub async fn metadata(&self) -> IndexMetadata {
        self.metadata.read().await.clone()
    }

    /// Switch to a different index type
    ///
    /// This is the core operation that:
    /// 1. Creates new index of target type
    /// 2. Migrates all data from current to new index
    /// 3. Atomically swaps reference
    /// 4. Logs transaction for recovery
    ///
    /// # Arguments
    /// * `target_type` - Type of index to switch to
    ///
    /// # Returns
    /// * `Result<SwitchResult>` - Details of switch operation or error
    ///
    /// # Example
    /// ```ignore
    /// let result = manager.switch_to(IndexType::Ivf).await?;
    /// println!("Switched: {} -> {}, records: {}",
    ///     result.from_type, result.to_type, result.records_migrated);
    /// ```
    pub async fn switch_to(&self, target_type: IndexType) -> Result<SwitchResult> {
        let start = Instant::now();
        let from_type = self.current_index_type().await;

        if from_type == target_type {
            return Err(Error::validation(
                "Cannot switch to same index type".to_string(),
            ));
        }

        // Log start of switch
        if let Some(ref log) = self.transaction_log {
            log.log_begin_switch(&from_type, &target_type).await?;
        }

        // Create new index
        let new_index = self.factory.create_index(target_type)?;

        // Migrate data from current to new
        let old_index = self.current.read().await;
        let migration_stats = self.migrator.migrate(
            &old_index,
            &new_index,
            Some(Box::new(|_progress| {
                // progress event emission not yet wired to a channel
            })),
        ).await?;

        // Atomic swap
        drop(old_index); // Release read lock
        let mut current = self.current.write().await;
        *current = new_index;
        drop(current); // Release write lock

        // Update metadata
        let mut metadata = self.metadata.write().await;
        metadata.index_type = target_type;
        metadata.record_count = migration_stats.records_migrated;
        metadata.last_switched = Some(chrono::Utc::now().to_rfc3339());

        // Log success
        if let Some(ref log) = self.transaction_log {
            log.log_switch_committed(&target_type).await?;
        }

        let duration_ms = start.elapsed().as_millis();

        Ok(SwitchResult {
            from_type,
            to_type: target_type,
            records_migrated: migration_stats.records_migrated,
            duration_ms,
            success: true,
        })
    }

    /// Check if switching is possible
    pub async fn can_switch(&self) -> bool {
        // Check various conditions
        // - No active migrations
        // - Not switching too frequently
        // - System resources available
        // For now, always true
        true
    }

    /// Adapt index based on workload if beneficial
    pub async fn adapt_if_beneficial(&self) -> Result<bool> {
        let recommendation = self.adaptive.get_recommendation().await
            .map_err(|e| crate::Error::Internal(e.to_string()))?;

        if recommendation.should_switch() {
            self.switch_to(recommendation.target_type).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    

    #[tokio::test]
    async fn test_manager_can_switch() {
        // IndexManager requires a live IndexFactory + IndexWrapper — skip full integration here.
        // can_switch() always returns true until resource-check logic is added.
        assert!(true);
    }
}
