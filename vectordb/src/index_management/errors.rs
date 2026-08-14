//! Error handling and recovery patterns for Phase 3
//!
//! This module demonstrates:
//! - Common errors that can occur
//! - How to handle them
//! - Recovery procedures
//! - Error propagation patterns

use crate::error::Error;
use thiserror::Error;

/// Phase 3 specific errors
#[derive(Error, Debug)]
pub enum IndexManagementError {
    #[error("Index creation failed: {0}")]
    CreationFailed(String),

    #[error("Index switching failed: {0}")]
    SwitchingFailed(String),

    #[error("Data migration failed: {0}")]
    MigrationFailed(String),

    #[error("Migration timeout after {0}ms")]
    MigrationTimeout(u128),

    #[error("Recovery failed: {0}")]
    RecoveryFailed(String),

    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Invalid state transition: from {from} to {to}")]
    InvalidStateTransition { from: String, to: String },

    #[error("Lock acquisition timeout")]
    LockTimeout,

    #[error("Data integrity check failed: {0}")]
    IntegrityCheckFailed(String),
}

/// Convert to generic Error type
impl From<IndexManagementError> for Error {
    fn from(err: IndexManagementError) -> Self {
        Error::Internal(err.to_string())
    }
}

/// Common error handling patterns
pub mod patterns {

    /// Pattern 1: Handle creation errors gracefully
    ///
    /// # Example
    /// ```ignore
    /// pub fn create_index(&self, index_type: IndexType) -> Result<IndexWrapper> {
    ///     match index_type {
    ///         IndexType::Hnsw => {
    ///             self.create_hnsw_index()
    ///                 .map_err(|e| Error::Custom(
    ///                     format!("Failed to create HNSW index: {}", e)
    ///                 ))
    ///         }
    ///         IndexType::Ivf => {
    ///             self.create_ivf_index()
    ///                 .map_err(|e| Error::Custom(
    ///                     format!("Failed to create IVF index: {}", e)
    ///                 ))
    ///         }
    ///         _ => Err(Error::NotImplemented("Index type not implemented".to_string()))
    ///     }
    /// }
    /// ```
    pub const PATTERN_1: &str = "Handle creation errors gracefully";

    /// Pattern 2: Handle switching errors with recovery
    ///
    /// # Example
    /// ```ignore
    /// pub async fn switch_to(&self, target_type: IndexType) -> Result<SwitchResult> {
    ///     // Log start
    ///     if let Err(e) = self.transaction_log.log_begin_switch(&from, &to).await {
    ///         return Err(Error::Custom(format!("Failed to log switch: {}", e)));
    ///     }
    ///
    ///     // Create new index
    ///     let new_index = match self.factory.create_index(target_type.clone()) {
    ///         Ok(idx) => idx,
    ///         Err(e) => {
    ///             // Log failure
    ///             let _ = self.transaction_log
    ///                 .log_switch_rolled_back(format!("Creation failed: {}", e)).await;
    ///             return Err(e);
    ///         }
    ///     };
    ///
    ///     // Migrate data
    ///     if let Err(e) = self.migrate_data(&new_index).await {
    ///         // Log failure and rollback
    ///         let _ = self.transaction_log
    ///             .log_switch_rolled_back(format!("Migration failed: {}", e)).await;
    ///         return Err(e);
    ///     }
    ///
    ///     // Atomic swap
    ///     let result = self.atomic_swap(&new_index).await?;
    ///
    ///     // Log success
    ///     self.transaction_log.log_switch_committed(&target_type).await?;
    ///
    ///     Ok(result)
    /// }
    /// ```
    /// Pattern 3: Handle migration timeout
    ///
    /// # Example
    /// ```ignore
    /// pub async fn migrate_with_timeout(
    ///     &self,
    ///     from: &IndexWrapper,
    ///     to: &IndexWrapper,
    /// ) -> Result<MigrationStats> {
    ///     let timeout_duration = self.config.migration_timeout;
    ///
    ///     match tokio::time::timeout(
    ///         timeout_duration,
    ///         self.migrator.migrate(from, to, None)
    ///     ).await {
    ///         Ok(Ok(stats)) => Ok(stats),
    ///         Ok(Err(e)) => Err(e),
    ///         Err(_) => {
    ///             Err(Error::Custom(format!(
    ///                 "Migration timeout after {:?}",
    ///                 timeout_duration
    ///             )))
    ///         }
    ///     }
    /// }
    /// ```
    /// Pattern 4: Handle lock acquisition with timeout
    ///
    /// # Example
    /// ```ignore
    /// pub async fn get_current_index_safe(&self) -> Result<Arc<RwLock<IndexWrapper>>> {
    ///     match tokio::time::timeout(
    ///         Duration::from_secs(5),
    ///         self.current.read()
    ///     ).await {
    ///         Ok(_guard) => Ok(self.current.clone()),
    ///         Err(_) => Err(Error::Custom("Failed to acquire read lock".to_string()))
    ///     }
    /// }
    /// ```
    /// Pattern 5: Handle recovery after crash
    ///
    /// # Example
    /// ```ignore
    /// pub async fn recover_from_crash(&self) -> Result<()> {
    ///     match self.transaction_log.recover().await? {
    ///         None => {
    ///             // No incomplete switch, normal startup
    ///             Ok(())
    ///         }
    ///         Some(recovery) => {
    ///             if recovery.completed {
    ///                 // Switch was completed, nothing to do
    ///                 Ok(())
    ///             } else {
    ///                 // Switch was incomplete, need to recover
    ///                 self.rollback_incomplete_switch(&recovery).await
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    /// Pattern 6: Handle validation errors
    ///
    /// # Example
    /// ```ignore
    /// pub fn validate_config(&self) -> Result<()> {
    ///     // Collect all errors
    ///     let mut errors = Vec::new();
    ///
    ///     if self.config.hnsw_m < 4 {
    ///         errors.push("HNSW M too small".to_string());
    ///     }
    ///     if self.config.hnsw_ef_construction < 100 {
    ///         errors.push("HNSW ef_construction too small".to_string());
    ///     }
    ///
    ///     if errors.is_empty() {
    ///         Ok(())
    ///     } else {
    ///         Err(Error::Custom(
    ///             format!("Configuration validation failed: {}",
    ///                     errors.join("; "))
    ///         ))
    ///     }
    /// }
    /// ```
    pub const PATTERN_6: &str = "Handle validation errors";
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_creation_failed_error() {
        let err = IndexManagementError::CreationFailed("HNSW config invalid".to_string());
        assert!(err.to_string().contains("Index creation failed"));
    }

    #[test]
    fn test_switching_failed_error() {
        let err = IndexManagementError::SwitchingFailed("Migration failed".to_string());
        assert!(err.to_string().contains("Index switching failed"));
    }

    #[test]
    fn test_migration_timeout_error() {
        let err = IndexManagementError::MigrationTimeout(5000);
        assert!(err.to_string().contains("5000"));
    }

    #[test]
    fn test_invalid_state_transition_error() {
        let err = IndexManagementError::InvalidStateTransition {
            from: "HNSW".to_string(),
            to: "HNSW".to_string(),
        };
        assert!(err.to_string().contains("Invalid state transition"));
    }
}
