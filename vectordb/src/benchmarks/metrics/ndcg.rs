//! NDCG@k (Normalized Discounted Cumulative Gain) metric
//!
//! NDCG@k measures the quality of ranking by considering both relevance and position.
//! Higher-ranked relevant results contribute more to the score.
//!
//! Formula:
//! - DCG@k = Σ(i=1 to k) relevance[i] / log2(i + 1)
//! - IDCG@k = DCG@k for perfect ranking
//! - NDCG@k = DCG@k / IDCG@k
//!
//! Example:
//! - Retrieved: [1, 6, 2, 7, 3] (relevance: [1, 0, 1, 0, 1])
//! - DCG@5 = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5) + 1/log2(6) = 2.13
//! - IDCG@5 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 2.13
//! - NDCG@5 = 2.13 / 2.13 = 1.0

use super::MetricStatistics;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct NdcgEvaluator;

impl NdcgEvaluator {
    /// Calculate NDCG@k for a single query
    ///
    /// # Arguments
    /// * `retrieved` - IDs of retrieved results (in order)
    /// * `ground_truth` - IDs of ground truth nearest neighbors
    /// * `k` - Number of top results to consider
    ///
    /// # Returns
    /// NDCG score in range [0.0, 1.0]
    pub fn ndcg_at_k(retrieved: &[String], ground_truth: &[String], k: usize) -> f64 {
        if k == 0 || retrieved.is_empty() || ground_truth.is_empty() {
            return 0.0;
        }

        let truth_set: HashSet<_> = ground_truth.iter().collect();

        // Calculate DCG (Discounted Cumulative Gain)
        let dcg: f64 = retrieved
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, id)| {
                let relevance = if truth_set.contains(id) { 1.0 } else { 0.0 };
                relevance / (i as f64 + 2.0).log2()
            })
            .sum();

        // Calculate IDCG (Ideal DCG) - assume all top-k are relevant
        let ideal_k = k.min(ground_truth.len());
        let idcg: f64 = (0..ideal_k).map(|i| 1.0 / (i as f64 + 2.0).log2()).sum();

        if idcg == 0.0 {
            0.0
        } else {
            dcg / idcg
        }
    }

    /// Calculate NDCG@k for multiple queries
    ///
    /// # Arguments
    /// * `results` - Vector of (retrieved_ids, ground_truth_ids) for each query
    /// * `k` - Number of top results to consider
    ///
    /// # Returns
    /// Statistics of NDCG@k across all queries
    pub fn evaluate(results: &[(Vec<String>, Vec<String>)], k: usize) -> MetricStatistics {
        let ndcg_scores: Vec<f64> = results
            .iter()
            .map(|(retrieved, truth)| Self::ndcg_at_k(retrieved, truth, k))
            .collect();

        MetricStatistics::from_scores(&ndcg_scores)
    }

    /// Calculate NDCG@k at multiple k values
    ///
    /// # Arguments
    /// * `retrieved` - IDs of retrieved results
    /// * `ground_truth` - IDs of ground truth nearest neighbors
    /// * `k_values` - List of k values to evaluate
    ///
    /// # Returns
    /// Vector of (k, ndcg) pairs
    pub fn ndcg_at_k_multi(
        retrieved: &[String],
        ground_truth: &[String],
        k_values: &[usize],
    ) -> Vec<(usize, f64)> {
        k_values
            .iter()
            .map(|&k| (k, Self::ndcg_at_k(retrieved, ground_truth, k)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndcg_perfect() {
        let retrieved = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = retrieved.clone();

        let ndcg = NdcgEvaluator::ndcg_at_k(&retrieved, &ground_truth, 5);
        assert!((ndcg - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ndcg_partial() {
        let retrieved = ["1", "6", "2", "7", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ndcg = NdcgEvaluator::ndcg_at_k(&retrieved, &ground_truth, 5);

        // Manual calculation:
        // DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5) + 1/log2(6)
        //     = 1.0 + 0.0 + 0.5 + 0.0 + 0.387 = 1.887
        // IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5) + 1/log2(6)
        //      = 1.0 + 0.631 + 0.5 + 0.431 + 0.387 = 2.949
        // NDCG = 1.887 / 2.949 = 0.640

        assert!((ndcg - 0.64).abs() < 0.01);
    }

    #[test]
    fn test_ndcg_zero() {
        let retrieved = ["6", "7", "8", "9", "10"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ndcg = NdcgEvaluator::ndcg_at_k(&retrieved, &ground_truth, 5);
        assert!((ndcg - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ndcg_position_matters() {
        // Better ranking (relevant results earlier)
        let retrieved1 = ["1", "2", "6", "7", "8"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        // Worse ranking (relevant results later)
        let retrieved2 = ["6", "7", "8", "1", "2"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ground_truth = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ndcg1 = NdcgEvaluator::ndcg_at_k(&retrieved1, &ground_truth, 5);
        let ndcg2 = NdcgEvaluator::ndcg_at_k(&retrieved2, &ground_truth, 5);

        // NDCG should be higher when relevant results are ranked higher
        assert!(ndcg1 > ndcg2);
    }

    #[test]
    fn test_ndcg_multi_k() {
        let retrieved = ["1", "6", "2", "7", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let results = NdcgEvaluator::ndcg_at_k_multi(&retrieved, &ground_truth, &[1, 3, 5]);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 1e-6); // Perfect at k=1
        assert_eq!(results[1].0, 3);
        assert_eq!(results[2].0, 5);
    }

    #[test]
    fn test_evaluate_multiple_queries() {
        let results = vec![
            (
                ["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
                ["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
            ),
            (
                ["1", "6", "2"].iter().map(|s| s.to_string()).collect(),
                ["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
            ),
        ];

        let stats = NdcgEvaluator::evaluate(&results, 3);
        assert!(stats.mean > 0.8); // Both queries have good NDCG
    }

    #[test]
    fn test_empty_results() {
        let retrieved: Vec<String> = vec![];
        let ground_truth = ["1", "2", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ndcg = NdcgEvaluator::ndcg_at_k(&retrieved, &ground_truth, 5);
        assert_eq!(ndcg, 0.0);
    }
}
