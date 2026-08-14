//! MAP@k (Mean Average Precision) metric
//!
//! MAP@k measures both precision and recall by averaging precision
//! at each relevant result position.
//!
//! Formula:
//! - AP@k = (1/min(k, |relevant|)) * Σ(i=1 to k) Precision@i * relevance[i]
//! - MAP@k = mean(AP@k) across all queries
//!
//! Example:
//! - Retrieved: [1, 6, 2, 7, 3] with ground truth [1, 2, 3, 4, 5]
//! - Relevant positions: 1, 3, 5
//! - Precision at relevant positions: P@1=1.0, P@3=2/3, P@5=3/5
//! - AP@5 = (1/3) * (1.0 + 2/3 + 3/5) = 0.756

use super::MetricStatistics;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct MapEvaluator;

impl MapEvaluator {
    /// Calculate Average Precision@k for a single query
    ///
    /// # Arguments
    /// * `retrieved` - IDs of retrieved results (in order)
    /// * `ground_truth` - IDs of ground truth nearest neighbors
    /// * `k` - Number of top results to consider
    ///
    /// # Returns
    /// Average Precision score in range [0.0, 1.0]
    pub fn average_precision_at_k(retrieved: &[String], ground_truth: &[String], k: usize) -> f64 {
        if k == 0 || retrieved.is_empty() || ground_truth.is_empty() {
            return 0.0;
        }

        let truth_set: HashSet<_> = ground_truth.iter().collect();
        let mut relevant_count = 0;
        let mut precision_sum = 0.0;

        for (i, id) in retrieved.iter().take(k).enumerate() {
            if truth_set.contains(id) {
                relevant_count += 1;
                let precision_at_i = relevant_count as f64 / (i + 1) as f64;
                precision_sum += precision_at_i;
            }
        }

        // Divide by the number of relevant items found in the top-k retrieved results
        if relevant_count == 0 {
            0.0
        } else {
            precision_sum / relevant_count as f64
        }
    }

    /// Calculate MAP@k for multiple queries
    ///
    /// # Arguments
    /// * `results` - Vector of (retrieved_ids, ground_truth_ids) for each query
    /// * `k` - Number of top results to consider
    ///
    /// # Returns
    /// Statistics of MAP@k across all queries
    pub fn evaluate(results: &[(Vec<String>, Vec<String>)], k: usize) -> MetricStatistics {
        let ap_scores: Vec<f64> = results
            .iter()
            .map(|(retrieved, truth)| Self::average_precision_at_k(retrieved, truth, k))
            .collect();

        MetricStatistics::from_scores(&ap_scores)
    }

    /// Calculate MAP@k at multiple k values
    ///
    /// # Arguments
    /// * `retrieved` - IDs of retrieved results
    /// * `ground_truth` - IDs of ground truth nearest neighbors
    /// * `k_values` - List of k values to evaluate
    ///
    /// # Returns
    /// Vector of (k, map) pairs
    pub fn map_at_k_multi(
        retrieved: &[String],
        ground_truth: &[String],
        k_values: &[usize],
    ) -> Vec<(usize, f64)> {
        k_values
            .iter()
            .map(|&k| (k, Self::average_precision_at_k(retrieved, ground_truth, k)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_perfect() {
        let retrieved = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = retrieved.clone();

        let ap = MapEvaluator::average_precision_at_k(&retrieved, &ground_truth, 5);
        assert!((ap - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_map_partial() {
        let retrieved = ["1", "6", "2", "7", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ap = MapEvaluator::average_precision_at_k(&retrieved, &ground_truth, 5);

        // Manual calculation:
        // Relevant positions: 1 (pos 1), 2 (pos 3), 3 (pos 5)
        // Precision@1 = 1/1 = 1.0
        // Precision@3 = 2/3 = 0.667
        // Precision@5 = 3/5 = 0.6
        // AP@5 = (1.0 + 0.667 + 0.6) / 3 = 0.756

        assert!((ap - 0.756).abs() < 0.01);
    }

    #[test]
    fn test_map_zero() {
        let retrieved = ["6", "7", "8", "9", "10"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ap = MapEvaluator::average_precision_at_k(&retrieved, &ground_truth, 5);
        assert!((ap - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_map_position_matters() {
        // Better ranking (relevant results earlier)
        let retrieved1 = ["1", "2", "3", "6", "7"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        // Worse ranking (relevant results later)
        let retrieved2 = ["6", "7", "1", "2", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ground_truth = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ap1 = MapEvaluator::average_precision_at_k(&retrieved1, &ground_truth, 5);
        let ap2 = MapEvaluator::average_precision_at_k(&retrieved2, &ground_truth, 5);

        // MAP should be higher when relevant results are ranked higher
        assert!(ap1 > ap2);
    }

    #[test]
    fn test_map_early_relevant() {
        // All relevant results at the beginning
        let retrieved = ["1", "2", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = ["1", "2", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ap = MapEvaluator::average_precision_at_k(&retrieved, &ground_truth, 3);

        // Precision@1 = 1/1, Precision@2 = 2/2, Precision@3 = 3/3
        // AP = (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert!((ap - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_map_multi_k() {
        let retrieved = ["1", "6", "2", "7", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ground_truth = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let results = MapEvaluator::map_at_k_multi(&retrieved, &ground_truth, &[1, 3, 5]);

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

        let stats = MapEvaluator::evaluate(&results, 3);
        assert!(stats.mean > 0.8); // Both queries have good MAP
    }

    #[test]
    fn test_empty_results() {
        let retrieved: Vec<String> = vec![];
        let ground_truth = ["1", "2", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let ap = MapEvaluator::average_precision_at_k(&retrieved, &ground_truth, 5);
        assert_eq!(ap, 0.0);
    }
}
