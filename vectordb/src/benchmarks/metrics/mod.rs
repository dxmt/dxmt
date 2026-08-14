//! Metrics for evaluating vector search quality
//!
//! This module provides comprehensive evaluation metrics for approximate
//! nearest neighbor (ANN) search, including:
//! - Recall@k: Proportion of true nearest neighbors found
//! - Precision@k: Fraction of retrieved results that are relevant
//! - NDCG@k: Normalized Discounted Cumulative Gain (ranking quality)
//! - MAP@k: Mean Average Precision (precision + recall balance)

pub mod evaluator;
pub mod map;
pub mod ndcg;
pub mod precision;
pub mod recall;

pub use evaluator::{MetricsEvaluator, MetricsReport};
pub use map::MapEvaluator;
pub use ndcg::NdcgEvaluator;
pub use precision::PrecisionEvaluator;
pub use recall::{RecallEvaluator, RecallStatistics};

/// Unified statistics structure for all metrics
#[derive(Debug, Clone, Default)]
pub struct MetricStatistics {
    pub mean: f64,
    pub median: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub p95: f64,
    pub p99: f64,
}

impl MetricStatistics {
    /// Calculate statistics from a list of scores
    pub fn from_scores(scores: &[f64]) -> Self {
        if scores.is_empty() {
            return Self::default();
        }

        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let median = sorted[sorted.len() / 2];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        // Calculate standard deviation
        let variance = sorted
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / sorted.len() as f64;
        let stddev = variance.sqrt();

        // Calculate percentiles
        let p95_idx = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);
        let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        let p95 = sorted[p95_idx];
        let p99 = sorted[p99_idx];

        Self {
            mean,
            median,
            stddev,
            min,
            max,
            p95,
            p99,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_statistics() {
        let scores = vec![0.5, 0.7, 0.6, 0.9, 0.8];
        let stats = MetricStatistics::from_scores(&scores);

        assert!((stats.mean - 0.7).abs() < 0.01);
        assert_eq!(stats.median, 0.7);
        assert_eq!(stats.min, 0.5);
        assert_eq!(stats.max, 0.9);
    }

    #[test]
    fn test_empty_scores() {
        let stats = MetricStatistics::from_scores(&[]);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.median, 0.0);
    }
}
