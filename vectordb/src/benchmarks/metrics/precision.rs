//! Precision@k metric for search quality evaluation
//!
//! Precision@k measures the fraction of retrieved results that are relevant.
//! It is calculated as: Precision@k = (# relevant results in top-k) / k
//!
//! Example:
//! - Ground truth nearest neighbors: [1, 2, 3, 4, 5]
//! - Search results: [1, 2, 6, 3, 7]
//! - Precision@5 = 3/5 = 0.6 (3 out of 5 results are in ground truth)

use super::MetricStatistics;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct PrecisionEvaluator;

impl PrecisionEvaluator {
    /// Calculate Precision@k for a single query
    ///
    /// # Arguments
    /// * `retrieved` - IDs of retrieved results (in order)
    /// * `ground_truth` - IDs of ground truth nearest neighbors
    /// * `k` - Number of top results to consider
    ///
    /// # Returns
    /// Precision score in range [0.0, 1.0]
    pub fn precision_at_k(retrieved: &[String], ground_truth: &[String], k: usize) -> f64 {
        if k == 0 || retrieved.is_empty() {
            return 0.0;
        }

        let truth_set: HashSet<_> = ground_truth.iter().collect();
        let top_k = retrieved.iter().take(k);

        let relevant_count = top_k.filter(|id| truth_set.contains(id)).count();

        relevant_count as f64 / k.min(retrieved.len()) as f64
    }

    /// Calculate Precision@k for multiple queries
    ///
    /// # Arguments
    /// * `results` - Vector of (retrieved_ids, ground_truth_ids) for each query
    /// * `k` - Number of top results to consider
    ///
    /// # Returns
    /// Statistics of Precision@k across all queries
    pub fn evaluate(results: &[(Vec<String>, Vec<String>)], k: usize) -> MetricStatistics {
        let precisions: Vec<f64> = results
            .iter()
            .map(|(retrieved, truth)| Self::precision_at_k(retrieved, truth, k))
            .collect();

        MetricStatistics::from_scores(&precisions)
    }

    /// Calculate Precision@k at multiple k values
    ///
    /// # Arguments
    /// * `retrieved` - IDs of retrieved results
    /// * `ground_truth` - IDs of ground truth nearest neighbors
    /// * `k_values` - List of k values to evaluate
    ///
    /// # Returns
    /// Vector of (k, precision) pairs
    pub fn precision_at_k_multi(
        retrieved: &[String],
        ground_truth: &[String],
        k_values: &[usize],
    ) -> Vec<(usize, f64)> {
        k_values
            .iter()
            .map(|&k| (k, Self::precision_at_k(retrieved, ground_truth, k)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_perfect() {
        let retrieved = vec!["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = retrieved.clone();

        let precision = PrecisionEvaluator::precision_at_k(&retrieved, &ground_truth, 5);
        assert!((precision - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_precision_partial() {
        let retrieved = vec!["1", "2", "6", "3", "7"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = vec!["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let precision = PrecisionEvaluator::precision_at_k(&retrieved, &ground_truth, 5);
        assert!((precision - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_precision_zero() {
        let retrieved = vec!["6", "7", "8", "9", "10"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = vec!["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let precision = PrecisionEvaluator::precision_at_k(&retrieved, &ground_truth, 5);
        assert!((precision - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_precision_at_different_k() {
        let retrieved = vec!["1", "2", "6", "3", "7"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = vec!["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let p1 = PrecisionEvaluator::precision_at_k(&retrieved, &ground_truth, 1);
        let p3 = PrecisionEvaluator::precision_at_k(&retrieved, &ground_truth, 3);
        let p5 = PrecisionEvaluator::precision_at_k(&retrieved, &ground_truth, 5);

        assert!((p1 - 1.0).abs() < 1e-6); // 1/1
        assert!((p3 - 2.0 / 3.0).abs() < 1e-6); // 2/3
        assert!((p5 - 0.6).abs() < 1e-6); // 3/5
    }

    #[test]
    fn test_precision_multi_k() {
        let retrieved = vec!["1", "2", "6", "3", "7"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = vec!["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let results = PrecisionEvaluator::precision_at_k_multi(&retrieved, &ground_truth, &[1, 3, 5]);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(results[1].0, 3);
        assert!((results[1].1 - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(results[2].0, 5);
        assert!((results[2].1 - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_multiple_queries() {
        let results = vec![
            (
                vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
                vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
            ),
            (
                vec!["1", "5", "3"].iter().map(|s| s.to_string()).collect(),
                vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
            ),
        ];

        let stats = PrecisionEvaluator::evaluate(&results, 3);
        assert!((stats.mean - 5.0 / 6.0).abs() < 1e-6); // (3/3 + 2/3) / 2
    }

    #[test]
    fn test_empty_results() {
        let retrieved: Vec<String> = vec![];
        let ground_truth = vec!["1", "2", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let precision = PrecisionEvaluator::precision_at_k(&retrieved, &ground_truth, 5);
        assert_eq!(precision, 0.0);
    }
}
