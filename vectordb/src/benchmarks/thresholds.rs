//! Performance Thresholds for Quality Assurance
//!
//! Defines and validates performance thresholds for different index types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance threshold for a specific index type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThreshold {
    /// Index type name
    pub index_type: String,
    /// Minimum recall@k value (0.0-1.0)
    pub min_recall_at_k: f64,
    /// k value for recall measurement
    pub recall_k: usize,
    /// Minimum queries per second
    pub min_qps: f64,
    /// Maximum p99 latency in milliseconds
    pub max_p99_latency_ms: f64,
    /// Minimum precision@k value (0.0-1.0)
    pub min_precision_at_k: Option<f64>,
    /// Minimum NDCG@k value (0.0-1.0)
    pub min_ndcg_at_k: Option<f64>,
}

impl PerformanceThreshold {
    /// Create default HNSW thresholds
    pub fn hnsw_default() -> Self {
        Self {
            index_type: "HNSW".to_string(),
            min_recall_at_k: 0.95,
            recall_k: 10,
            min_qps: 10_000.0,
            max_p99_latency_ms: 10.0,
            min_precision_at_k: Some(0.95),
            min_ndcg_at_k: Some(0.95),
        }
    }

    /// Create default DiskANN thresholds
    pub fn diskann_default() -> Self {
        Self {
            index_type: "DiskANN".to_string(),
            min_recall_at_k: 0.90,
            recall_k: 10,
            min_qps: 5_000.0,
            max_p99_latency_ms: 50.0,
            min_precision_at_k: Some(0.90),
            min_ndcg_at_k: Some(0.90),
        }
    }

    /// Create default IVF-PQ thresholds
    pub fn ivf_pq_default() -> Self {
        Self {
            index_type: "IVF-PQ".to_string(),
            min_recall_at_k: 0.85,
            recall_k: 10,
            min_qps: 8_000.0,
            max_p99_latency_ms: 20.0,
            min_precision_at_k: Some(0.85),
            min_ndcg_at_k: Some(0.85),
        }
    }

    /// Create default HNSW-PQ thresholds
    pub fn hnsw_pq_default() -> Self {
        Self {
            index_type: "HNSW-PQ".to_string(),
            min_recall_at_k: 0.90,
            recall_k: 10,
            min_qps: 12_000.0,
            max_p99_latency_ms: 15.0,
            min_precision_at_k: Some(0.90),
            min_ndcg_at_k: Some(0.90),
        }
    }

    /// Create default BruteForce thresholds
    pub fn brute_force_default() -> Self {
        Self {
            index_type: "BruteForce".to_string(),
            min_recall_at_k: 1.0, // Exact search
            recall_k: 10,
            min_qps: 100.0, // Much slower but accurate
            max_p99_latency_ms: 1000.0,
            min_precision_at_k: Some(1.0),
            min_ndcg_at_k: Some(1.0),
        }
    }
}

/// Threshold validation result
#[derive(Debug, Clone)]
pub struct ThresholdValidation {
    pub passed: bool,
    pub index_type: String,
    pub failures: Vec<String>,
    pub metrics: ValidationMetrics,
}

/// Metrics used for validation
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    pub recall_at_k: f64,
    pub qps: f64,
    pub p99_latency_ms: f64,
    pub precision_at_k: Option<f64>,
    pub ndcg_at_k: Option<f64>,
}

impl ThresholdValidation {
    /// Print validation report
    pub fn print_report(&self) {
        println!("\n=== Threshold Validation: {} ===", self.index_type);
        println!(
            "Status: {}",
            if self.passed {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        );

        println!("\nMetrics:");
        println!("  Recall@{}: {:.4}", 10, self.metrics.recall_at_k);
        println!("  QPS: {:.0}", self.metrics.qps);
        println!("  P99 Latency: {:.2}ms", self.metrics.p99_latency_ms);

        if let Some(precision) = self.metrics.precision_at_k {
            println!("  Precision@{}: {:.4}", 10, precision);
        }

        if let Some(ndcg) = self.metrics.ndcg_at_k {
            println!("  NDCG@{}: {:.4}", 10, ndcg);
        }

        if !self.failures.is_empty() {
            println!("\nFailures:");
            for failure in &self.failures {
                println!("  ❌ {}", failure);
            }
        }
    }
}

/// Threshold validator
#[derive(Debug, Clone)]
pub struct ThresholdValidator {
    thresholds: HashMap<String, PerformanceThreshold>,
}

impl ThresholdValidator {
    /// Create a new validator with default thresholds
    pub fn new() -> Self {
        let mut thresholds = HashMap::new();

        thresholds.insert("HNSW".to_string(), PerformanceThreshold::hnsw_default());
        thresholds.insert(
            "DiskANN".to_string(),
            PerformanceThreshold::diskann_default(),
        );
        thresholds.insert("IVF-PQ".to_string(), PerformanceThreshold::ivf_pq_default());
        thresholds.insert(
            "HNSW-PQ".to_string(),
            PerformanceThreshold::hnsw_pq_default(),
        );
        thresholds.insert(
            "BruteForce".to_string(),
            PerformanceThreshold::brute_force_default(),
        );

        Self { thresholds }
    }

    /// Add or update a threshold
    pub fn set_threshold(&mut self, index_type: String, threshold: PerformanceThreshold) {
        self.thresholds.insert(index_type, threshold);
    }

    /// Get threshold for an index type
    pub fn get_threshold(&self, index_type: &str) -> Option<&PerformanceThreshold> {
        self.thresholds.get(index_type)
    }

    /// Validate metrics against thresholds
    pub fn validate(&self, index_type: &str, metrics: ValidationMetrics) -> ThresholdValidation {
        let threshold = match self.get_threshold(index_type) {
            Some(t) => t,
            None => {
                return ThresholdValidation {
                    passed: false,
                    index_type: index_type.to_string(),
                    failures: vec![format!("No threshold defined for {}", index_type)],
                    metrics,
                };
            }
        };

        let mut failures = Vec::new();

        // Validate recall
        if metrics.recall_at_k < threshold.min_recall_at_k {
            failures.push(format!(
                "Recall@{} {:.4} < threshold {:.4}",
                threshold.recall_k, metrics.recall_at_k, threshold.min_recall_at_k
            ));
        }

        // Validate QPS
        if metrics.qps < threshold.min_qps {
            failures.push(format!(
                "QPS {:.0} < threshold {:.0}",
                metrics.qps, threshold.min_qps
            ));
        }

        // Validate latency
        if metrics.p99_latency_ms > threshold.max_p99_latency_ms {
            failures.push(format!(
                "P99 latency {:.2}ms > threshold {:.2}ms",
                metrics.p99_latency_ms, threshold.max_p99_latency_ms
            ));
        }

        // Validate precision (if specified)
        if let (Some(min_precision), Some(actual_precision)) =
            (threshold.min_precision_at_k, metrics.precision_at_k)
        {
            if actual_precision < min_precision {
                failures.push(format!(
                    "Precision@{} {:.4} < threshold {:.4}",
                    threshold.recall_k, actual_precision, min_precision
                ));
            }
        }

        // Validate NDCG (if specified)
        if let (Some(min_ndcg), Some(actual_ndcg)) = (threshold.min_ndcg_at_k, metrics.ndcg_at_k) {
            if actual_ndcg < min_ndcg {
                failures.push(format!(
                    "NDCG@{} {:.4} < threshold {:.4}",
                    threshold.recall_k, actual_ndcg, min_ndcg
                ));
            }
        }

        ThresholdValidation {
            passed: failures.is_empty(),
            index_type: index_type.to_string(),
            failures,
            metrics,
        }
    }

    /// Validate multiple index types and return summary
    pub fn validate_batch(
        &self,
        results: Vec<(String, ValidationMetrics)>,
    ) -> Vec<ThresholdValidation> {
        results
            .into_iter()
            .map(|(index_type, metrics)| self.validate(&index_type, metrics))
            .collect()
    }

    /// Export thresholds to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.thresholds)
    }

    /// Import thresholds from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let thresholds = serde_json::from_str(json)?;
        Ok(Self { thresholds })
    }
}

impl Default for ThresholdValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_threshold() {
        let threshold = PerformanceThreshold::hnsw_default();
        assert_eq!(threshold.index_type, "HNSW");
        assert_eq!(threshold.min_recall_at_k, 0.95);
        assert_eq!(threshold.min_qps, 10_000.0);
    }

    #[test]
    fn test_validator_creation() {
        let validator = ThresholdValidator::new();
        assert!(validator.get_threshold("HNSW").is_some());
        assert!(validator.get_threshold("DiskANN").is_some());
        assert!(validator.get_threshold("IVF-PQ").is_some());
    }

    #[test]
    fn test_validation_pass() {
        let validator = ThresholdValidator::new();

        let metrics = ValidationMetrics {
            recall_at_k: 0.96,
            qps: 12_000.0,
            p99_latency_ms: 8.0,
            precision_at_k: Some(0.96),
            ndcg_at_k: Some(0.96),
        };

        let result = validator.validate("HNSW", metrics);
        assert!(result.passed);
        assert!(result.failures.is_empty());
    }

    #[test]
    fn test_validation_fail_recall() {
        let validator = ThresholdValidator::new();

        let metrics = ValidationMetrics {
            recall_at_k: 0.90, // Below threshold (0.95)
            qps: 12_000.0,
            p99_latency_ms: 8.0,
            precision_at_k: None,
            ndcg_at_k: None,
        };

        let result = validator.validate("HNSW", metrics);
        assert!(!result.passed);
        assert_eq!(result.failures.len(), 1);
        assert!(result.failures[0].contains("Recall"));
    }

    #[test]
    fn test_validation_fail_qps() {
        let validator = ThresholdValidator::new();

        let metrics = ValidationMetrics {
            recall_at_k: 0.96,
            qps: 5_000.0, // Below threshold (10,000)
            p99_latency_ms: 8.0,
            precision_at_k: None,
            ndcg_at_k: None,
        };

        let result = validator.validate("HNSW", metrics);
        assert!(!result.passed);
        assert_eq!(result.failures.len(), 1);
        assert!(result.failures[0].contains("QPS"));
    }

    #[test]
    fn test_validation_fail_latency() {
        let validator = ThresholdValidator::new();

        let metrics = ValidationMetrics {
            recall_at_k: 0.96,
            qps: 12_000.0,
            p99_latency_ms: 15.0, // Above threshold (10.0)
            precision_at_k: None,
            ndcg_at_k: None,
        };

        let result = validator.validate("HNSW", metrics);
        assert!(!result.passed);
        assert_eq!(result.failures.len(), 1);
        assert!(result.failures[0].contains("latency"));
    }

    #[test]
    fn test_validation_multiple_failures() {
        let validator = ThresholdValidator::new();

        let metrics = ValidationMetrics {
            recall_at_k: 0.90,    // Fail
            qps: 5_000.0,         // Fail
            p99_latency_ms: 15.0, // Fail
            precision_at_k: None,
            ndcg_at_k: None,
        };

        let result = validator.validate("HNSW", metrics);
        assert!(!result.passed);
        assert_eq!(result.failures.len(), 3);
    }

    #[test]
    fn test_validation_unknown_index() {
        let validator = ThresholdValidator::new();

        let metrics = ValidationMetrics {
            recall_at_k: 0.96,
            qps: 12_000.0,
            p99_latency_ms: 8.0,
            precision_at_k: None,
            ndcg_at_k: None,
        };

        let result = validator.validate("UnknownIndex", metrics);
        assert!(!result.passed);
        assert!(result.failures[0].contains("No threshold defined"));
    }

    #[test]
    fn test_batch_validation() {
        let validator = ThresholdValidator::new();

        let results = vec![
            (
                "HNSW".to_string(),
                ValidationMetrics {
                    recall_at_k: 0.96,
                    qps: 12_000.0,
                    p99_latency_ms: 8.0,
                    precision_at_k: None,
                    ndcg_at_k: None,
                },
            ),
            (
                "DiskANN".to_string(),
                ValidationMetrics {
                    recall_at_k: 0.92,
                    qps: 6_000.0,
                    p99_latency_ms: 40.0,
                    precision_at_k: None,
                    ndcg_at_k: None,
                },
            ),
        ];

        let validations = validator.validate_batch(results);
        assert_eq!(validations.len(), 2);
        assert!(validations[0].passed);
        assert!(validations[1].passed);
    }

    #[test]
    fn test_json_export_import() {
        let validator = ThresholdValidator::new();

        let json = validator.to_json().unwrap();
        let validator2 = ThresholdValidator::from_json(&json).unwrap();

        assert_eq!(
            validator.get_threshold("HNSW").unwrap().min_recall_at_k,
            validator2.get_threshold("HNSW").unwrap().min_recall_at_k
        );
    }

    #[test]
    fn test_print_report() {
        let validator = ThresholdValidator::new();

        let metrics = ValidationMetrics {
            recall_at_k: 0.90,
            qps: 5_000.0,
            p99_latency_ms: 15.0,
            precision_at_k: Some(0.92),
            ndcg_at_k: Some(0.91),
        };

        let result = validator.validate("HNSW", metrics);
        result.print_report(); // Should not panic
    }
}
