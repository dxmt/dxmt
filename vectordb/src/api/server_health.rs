//! Server Health Monitoring and Backpressure Management
//!
//! This module provides server health state tracking and backpressure mechanisms
//! to prevent overload and provide graceful degradation under heavy load.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Server health states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthState {
    /// Server is healthy and accepting all requests
    Healthy,
    /// Server is under moderate load, still accepting requests but with warnings
    Degraded,
    /// Server is under heavy load, applying backpressure
    Overloaded,
    /// Server is critically overloaded, rejecting non-critical requests
    Critical,
}

impl HealthState {
    /// Returns retry delay in seconds for this health state
    pub fn retry_after_seconds(&self) -> u64 {
        match self {
            HealthState::Healthy => 0,
            HealthState::Degraded => 1,
            HealthState::Overloaded => 2,
            HealthState::Critical => 5,
        }
    }

    /// Returns human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            HealthState::Healthy => "Server is operating normally",
            HealthState::Degraded => "Server is under moderate load",
            HealthState::Overloaded => "Server is overloaded, please retry after delay",
            HealthState::Critical => "Server is critically overloaded, please retry later",
        }
    }
}

/// Tracks ongoing operations for backpressure
#[derive(Debug)]
struct OperationMetrics {
    /// Number of ongoing insert operations
    ongoing_inserts: AtomicUsize,
    /// Number of ongoing search operations
    ongoing_searches: AtomicUsize,
    /// Number of ongoing delete operations
    ongoing_deletes: AtomicUsize,
    /// Number of ongoing table operations (create/delete)
    ongoing_table_ops: AtomicUsize,
    /// Total operations completed
    total_operations: AtomicU64,
    /// Last index build timestamp
    last_index_build: RwLock<Option<Instant>>,
    /// Is index building currently in progress
    index_building: AtomicBool,
    /// Number of recent errors
    recent_errors: AtomicUsize,
    /// Start of the current error burst. Error pressure decays after the configured cooldown;
    /// otherwise ten unrelated startup/client errors could permanently mark a healthy server as
    /// critical until restart.
    last_error_at: std::sync::Mutex<Option<Instant>>,
}

impl Default for OperationMetrics {
    fn default() -> Self {
        Self {
            ongoing_inserts: AtomicUsize::new(0),
            ongoing_searches: AtomicUsize::new(0),
            ongoing_deletes: AtomicUsize::new(0),
            ongoing_table_ops: AtomicUsize::new(0),
            total_operations: AtomicU64::new(0),
            last_index_build: RwLock::new(None),
            index_building: AtomicBool::new(false),
            recent_errors: AtomicUsize::new(0),
            last_error_at: std::sync::Mutex::new(None),
        }
    }
}

/// Configuration for health monitoring thresholds
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Maximum concurrent insert operations before degraded state
    pub max_concurrent_inserts: usize,
    /// Maximum concurrent operations total before overloaded state
    pub max_concurrent_operations: usize,
    /// Cooldown period after heavy operations (ms)
    pub cooldown_period_ms: u64,
    /// Maximum recent errors before critical state
    pub max_recent_errors: usize,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            max_concurrent_inserts: 50,
            max_concurrent_operations: 200,
            cooldown_period_ms: 2000, // 2 seconds - matches observed recovery time
            max_recent_errors: 10,
        }
    }
}

/// Server health monitor
pub struct ServerHealth {
    metrics: Arc<OperationMetrics>,
    config: HealthConfig,
}

impl Default for ServerHealth {
    fn default() -> Self {
        Self::new(HealthConfig::default())
    }
}

impl ServerHealth {
    /// Create a new server health monitor
    pub fn new(config: HealthConfig) -> Self {
        Self {
            metrics: Arc::new(OperationMetrics::default()),
            config,
        }
    }

    /// Get current health state
    pub async fn current_state(&self) -> HealthState {
        self.reset_stale_errors();
        let ongoing_inserts = self.metrics.ongoing_inserts.load(Ordering::Relaxed);
        let total_ongoing = self.total_ongoing_operations();
        let recent_errors = self.metrics.recent_errors.load(Ordering::Relaxed);
        let index_building = self.metrics.index_building.load(Ordering::Relaxed);

        // Check if we're in cooldown period after index build
        if let Some(last_build) = *self.metrics.last_index_build.read().await {
            let elapsed = Instant::now().duration_since(last_build);
            if elapsed < Duration::from_millis(self.config.cooldown_period_ms) {
                return HealthState::Overloaded;
            }
        }

        // Critical: too many recent errors or index building
        if recent_errors >= self.config.max_recent_errors || index_building {
            return HealthState::Critical;
        }

        // Overloaded: too many operations
        if total_ongoing >= self.config.max_concurrent_operations {
            return HealthState::Overloaded;
        }

        // Degraded: many insert operations
        if ongoing_inserts >= self.config.max_concurrent_inserts {
            return HealthState::Degraded;
        }

        HealthState::Healthy
    }

    /// Check if server should accept insert requests
    pub async fn should_accept_insert(&self) -> (bool, HealthState) {
        let state = self.current_state().await;
        let accept = match state {
            HealthState::Healthy | HealthState::Degraded => true,
            // Backpressure must be deterministic. Random admission made retries impossible to
            // reason about and could overload the same queue that triggered this state.
            HealthState::Overloaded => false,
            HealthState::Critical => false,
        };
        (accept, state)
    }

    /// Check if server should accept search requests
    pub async fn should_accept_search(&self) -> (bool, HealthState) {
        let state = self.current_state().await;
        let accept = match state {
            // Reads remain available while writes are being drained or durability is repaired.
            HealthState::Healthy
            | HealthState::Degraded
            | HealthState::Overloaded
            | HealthState::Critical => true,
        };
        (accept, state)
    }

    /// Start tracking an insert operation
    pub fn start_insert(&self) -> OperationGuard {
        self.metrics.ongoing_inserts.fetch_add(1, Ordering::Relaxed);
        OperationGuard {
            metrics: Arc::clone(&self.metrics),
            op_type: OperationType::Insert,
        }
    }

    /// Start tracking a search operation
    pub fn start_search(&self) -> OperationGuard {
        self.metrics
            .ongoing_searches
            .fetch_add(1, Ordering::Relaxed);
        OperationGuard {
            metrics: Arc::clone(&self.metrics),
            op_type: OperationType::Search,
        }
    }

    /// Start tracking a delete operation
    pub fn start_delete(&self) -> OperationGuard {
        self.metrics.ongoing_deletes.fetch_add(1, Ordering::Relaxed);
        OperationGuard {
            metrics: Arc::clone(&self.metrics),
            op_type: OperationType::Delete,
        }
    }

    /// Start tracking a table operation
    pub fn start_table_op(&self) -> OperationGuard {
        self.metrics
            .ongoing_table_ops
            .fetch_add(1, Ordering::Relaxed);
        OperationGuard {
            metrics: Arc::clone(&self.metrics),
            op_type: OperationType::TableOp,
        }
    }

    /// Record an error
    pub fn record_error(&self) {
        let now = Instant::now();
        if let Ok(mut last_error_at) = self.metrics.last_error_at.lock() {
            let expired = (*last_error_at)
                .map(|started| {
                    now.duration_since(started)
                        >= Duration::from_millis(self.config.cooldown_period_ms)
                })
                .unwrap_or(false);
            if expired {
                self.metrics.recent_errors.store(0, Ordering::Relaxed);
            }
            if expired || last_error_at.is_none() {
                *last_error_at = Some(now);
            }
        }
        self.metrics.recent_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Reset error counter (called periodically)
    pub fn reset_errors(&self) {
        self.metrics.recent_errors.store(0, Ordering::Relaxed);
        if let Ok(mut last_error_at) = self.metrics.last_error_at.lock() {
            *last_error_at = None;
        }
    }

    /// Mark index build started
    pub async fn mark_index_build_start(&self) {
        self.metrics.index_building.store(true, Ordering::Relaxed);
    }

    /// Mark index build completed
    pub async fn mark_index_build_end(&self) {
        self.metrics.index_building.store(false, Ordering::Relaxed);
        *self.metrics.last_index_build.write().await = Some(Instant::now());
    }

    /// Get total ongoing operations
    fn total_ongoing_operations(&self) -> usize {
        self.metrics.ongoing_inserts.load(Ordering::Relaxed)
            + self.metrics.ongoing_searches.load(Ordering::Relaxed)
            + self.metrics.ongoing_deletes.load(Ordering::Relaxed)
            + self.metrics.ongoing_table_ops.load(Ordering::Relaxed)
    }

    /// Get metrics for monitoring
    pub fn get_metrics(&self) -> HealthMetrics {
        self.reset_stale_errors();
        HealthMetrics {
            ongoing_inserts: self.metrics.ongoing_inserts.load(Ordering::Relaxed),
            ongoing_searches: self.metrics.ongoing_searches.load(Ordering::Relaxed),
            ongoing_deletes: self.metrics.ongoing_deletes.load(Ordering::Relaxed),
            ongoing_table_ops: self.metrics.ongoing_table_ops.load(Ordering::Relaxed),
            total_operations: self.metrics.total_operations.load(Ordering::Relaxed),
            recent_errors: self.metrics.recent_errors.load(Ordering::Relaxed),
            index_building: self.metrics.index_building.load(Ordering::Relaxed),
        }
    }

    fn reset_stale_errors(&self) {
        let stale = self
            .metrics
            .last_error_at
            .lock()
            .ok()
            .and_then(|last| *last)
            .map(|started| {
                started.elapsed() >= Duration::from_millis(self.config.cooldown_period_ms)
            })
            .unwrap_or(false);
        if stale {
            self.metrics.recent_errors.store(0, Ordering::Relaxed);
            if let Ok(mut last_error_at) = self.metrics.last_error_at.lock() {
                *last_error_at = None;
            }
        }
    }
}

/// Operation type for tracking
#[derive(Debug, Clone, Copy)]
enum OperationType {
    Insert,
    Search,
    Delete,
    TableOp,
}

/// RAII guard for operation tracking
pub struct OperationGuard {
    metrics: Arc<OperationMetrics>,
    op_type: OperationType,
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        // Decrement the appropriate counter
        match self.op_type {
            OperationType::Insert => {
                self.metrics.ongoing_inserts.fetch_sub(1, Ordering::Relaxed);
            }
            OperationType::Search => {
                self.metrics
                    .ongoing_searches
                    .fetch_sub(1, Ordering::Relaxed);
            }
            OperationType::Delete => {
                self.metrics.ongoing_deletes.fetch_sub(1, Ordering::Relaxed);
            }
            OperationType::TableOp => {
                self.metrics
                    .ongoing_table_ops
                    .fetch_sub(1, Ordering::Relaxed);
            }
        }
        // Increment total operations counter
        self.metrics
            .total_operations
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// Metrics snapshot for monitoring
#[derive(Debug, Clone)]
pub struct HealthMetrics {
    pub ongoing_inserts: usize,
    pub ongoing_searches: usize,
    pub ongoing_deletes: usize,
    pub ongoing_table_ops: usize,
    pub total_operations: u64,
    pub recent_errors: usize,
    pub index_building: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_state_progression() {
        let health = ServerHealth::default();

        // Initially healthy
        assert_eq!(health.current_state().await, HealthState::Healthy);

        // Add some operations
        let _guards: Vec<_> = (0..60).map(|_| health.start_insert()).collect();
        let state = health.current_state().await;
        assert!(matches!(
            state,
            HealthState::Degraded | HealthState::Overloaded
        ));

        // Drop guards
        drop(_guards);

        // Should recover to healthy
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(health.current_state().await, HealthState::Healthy);
    }

    #[tokio::test]
    async fn test_index_build_cooldown() {
        let config = HealthConfig {
            cooldown_period_ms: 100,
            ..Default::default()
        };
        let health = ServerHealth::new(config);

        // Mark index build
        health.mark_index_build_start().await;
        assert_eq!(health.current_state().await, HealthState::Critical);

        health.mark_index_build_end().await;
        assert_eq!(health.current_state().await, HealthState::Overloaded);

        // Wait for cooldown
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(health.current_state().await, HealthState::Healthy);
    }

    #[tokio::test]
    async fn test_error_tracking() {
        let config = HealthConfig {
            max_recent_errors: 3,
            ..Default::default()
        };
        let health = ServerHealth::new(config);

        // Record errors
        health.record_error();
        health.record_error();
        assert_eq!(health.current_state().await, HealthState::Healthy);

        health.record_error();
        assert_eq!(health.current_state().await, HealthState::Critical);

        // Reset errors
        health.reset_errors();
        assert_eq!(health.current_state().await, HealthState::Healthy);
    }
}
