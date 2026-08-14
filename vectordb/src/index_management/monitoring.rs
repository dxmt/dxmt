//! Monitoring and observability utilities for Phase 3
//!
//! Provides:
//! - Operation metrics and tracking
//! - Performance monitoring
//! - Event logging
//! - Error tracking

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Metrics for index operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    pub operation_name: String,
    pub start_time: String,
    pub duration_ms: u128,
    pub success: bool,
    pub error_message: Option<String>,
    pub records_affected: usize,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub total_duration_ms: u128,
    pub average_duration_ms: u128,
    pub min_duration_ms: u128,
    pub max_duration_ms: u128,
}

/// Monitoring service for tracking operations
pub struct MonitoringService {
    metrics: Arc<RwLock<Vec<OperationMetrics>>>,
    stats: Arc<RwLock<PerformanceStats>>,
}

impl MonitoringService {
    /// Create a new monitoring service
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(PerformanceStats {
                total_operations: 0,
                successful_operations: 0,
                failed_operations: 0,
                total_duration_ms: 0,
                average_duration_ms: 0,
                min_duration_ms: u128::MAX,
                max_duration_ms: 0,
            })),
        }
    }

    /// Record operation start
    pub async fn start_operation(&self, name: &str) -> OperationTimer {
        OperationTimer {
            name: name.to_string(),
            start: Instant::now(),
            metrics: self.metrics.clone(),
            stats: self.stats.clone(),
        }
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> PerformanceStats {
        self.stats.read().await.clone()
    }

    /// Get all metrics
    pub async fn get_metrics(&self) -> Vec<OperationMetrics> {
        self.metrics.read().await.clone()
    }

    /// Clear metrics
    pub async fn clear_metrics(&self) {
        self.metrics.write().await.clear();
    }

    /// Get metrics since duration ago
    pub async fn get_recent_metrics(&self, _since: Duration) -> Vec<OperationMetrics> {
        // In real implementation, parse timestamps and filter
        self.metrics.read().await.clone()
    }
}

/// Timer that automatically records operation metrics
pub struct OperationTimer {
    name: String,
    start: Instant,
    metrics: Arc<RwLock<Vec<OperationMetrics>>>,
    stats: Arc<RwLock<PerformanceStats>>,
}

impl OperationTimer {
    /// Mark operation as successful
    pub async fn finish_success(self, records_affected: usize) {
        let duration_ms = self.start.elapsed().as_millis();

        let metric = OperationMetrics {
            operation_name: self.name,
            start_time: chrono::Utc::now().to_rfc3339(),
            duration_ms,
            success: true,
            error_message: None,
            records_affected,
        };

        // Record metric
        self.metrics.write().await.push(metric);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_operations += 1;
        stats.successful_operations += 1;
        stats.total_duration_ms += duration_ms;
        stats.average_duration_ms = stats.total_duration_ms / (stats.total_operations as u128);
        stats.min_duration_ms = stats.min_duration_ms.min(duration_ms);
        stats.max_duration_ms = stats.max_duration_ms.max(duration_ms);
    }

    /// Mark operation as failed
    pub async fn finish_error(self, error: String) {
        let duration_ms = self.start.elapsed().as_millis();

        let metric = OperationMetrics {
            operation_name: self.name,
            start_time: chrono::Utc::now().to_rfc3339(),
            duration_ms,
            success: false,
            error_message: Some(error),
            records_affected: 0,
        };

        // Record metric
        self.metrics.write().await.push(metric);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_operations += 1;
        stats.failed_operations += 1;
        stats.total_duration_ms += duration_ms;
        if stats.total_operations > 0 {
            stats.average_duration_ms = stats.total_duration_ms / (stats.total_operations as u128);
        }
        stats.min_duration_ms = stats.min_duration_ms.min(duration_ms);
        stats.max_duration_ms = stats.max_duration_ms.max(duration_ms);
    }
}

impl Default for MonitoringService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod monitoring_tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_service_creation() {
        let monitoring = MonitoringService::new();
        let stats = monitoring.get_stats().await;
        assert_eq!(stats.total_operations as u128, 0);
    }

    #[tokio::test]
    async fn test_operation_success_tracking() {
        let monitoring = MonitoringService::new();

        let timer = monitoring.start_operation("test_op").await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        timer.finish_success(100).await;

        let stats = monitoring.get_stats().await;
        assert_eq!(stats.total_operations as u128, 1);
        assert_eq!(stats.successful_operations, 1);
        assert_eq!(stats.failed_operations, 0);
    }

    #[tokio::test]
    async fn test_operation_error_tracking() {
        let monitoring = MonitoringService::new();

        let timer = monitoring.start_operation("failing_op").await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        timer.finish_error("Test error".to_string()).await;

        let stats = monitoring.get_stats().await;
        assert_eq!(stats.total_operations as u128, 1);
        assert_eq!(stats.successful_operations, 0);
        assert_eq!(stats.failed_operations, 1);
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let monitoring = MonitoringService::new();

        let timer = monitoring.start_operation("op1").await;
        timer.finish_success(50).await;

        let metrics = monitoring.get_metrics().await;
        assert_eq!(metrics.len(), 1);
        assert!(metrics[0].success);
    }
}
