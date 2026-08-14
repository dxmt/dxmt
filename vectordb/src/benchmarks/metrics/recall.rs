//! Recall@k metric for ANN search evaluation
//!
//! Recall measures the proportion of true nearest neighbors that are
//! successfully retrieved in the top-k results.
//!
//! Formula: Recall@k = |retrieved ∩ ground_truth| / k
//!
//! Example:
//! ```
//! use dxmt_vectordb::benchmarks::metrics::RecallEvaluator;
//!
//! let evaluator = RecallEvaluator::new(10);
//!
//! let retrieved = vec!["v1".to_string(), "v2".to_string(), "v5".to_string()];
//! let ground_truth = vec!["v1".to_string(), "v2".to_string(), "v3".to_string()];
//!
//! let recall = evaluator.calculate_recall(&retrieved, &ground_truth);
//! assert_eq!(recall, 0.2); // 2 matches out of 10 (k=10)
//! ```

use std::collections::HashSet;
use serde::{Deserialize, Serialize};

/// Evaluates recall@k for ANN search results
#[derive(Debug, Clone)]
pub struct RecallEvaluator {
    /// Number of neighbors to consider (k)
    k: usize,
}

impl RecallEvaluator {
    /// Create a new recall evaluator with specified k
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        Self { k }
    }

    /// Calculate recall@k for a single query
    ///
    /// # Arguments
    /// * `retrieved` - IDs returned by the search algorithm (may be > k, will be truncated)
    /// * `ground_truth` - True nearest neighbor IDs (may be > k, will be truncated)
    ///
    /// # Returns
    /// Recall value between 0.0 and 1.0
    pub fn calculate_recall<T>(&self, retrieved: &[T], ground_truth: &[T]) -> f64
    where
        T: Eq + std::hash::Hash,
    {
        // Take only top-k from each list
        let retrieved_k: HashSet<_> = retrieved.iter().take(self.k).collect();
        let ground_truth_k: HashSet<_> = ground_truth.iter().take(self.k).collect();

        // Count intersection
        let intersection = retrieved_k.intersection(&ground_truth_k).count();

        // Recall = intersection / k
        intersection as f64 / self.k as f64
    }

    /// Calculate recall@k for multiple queries and return statistics
    ///
    /// # Arguments
    /// * `results` - Retrieved IDs for each query
    /// * `ground_truths` - Ground truth IDs for each query
    ///
    /// # Returns
    /// Statistics across all queries
    pub fn evaluate_batch<T>(
        &self,
        results: &[Vec<T>],
        ground_truths: &[Vec<T>],
    ) -> RecallStatistics
    where
        T: Eq + std::hash::Hash,
    {
        assert_eq!(
            results.len(),
            ground_truths.len(),
            "Number of results and ground truths must match"
        );

        let mut recalls: Vec<f64> = results
            .iter()
            .zip(ground_truths.iter())
            .map(|(retrieved, truth)| self.calculate_recall(retrieved, truth))
            .collect();

        RecallStatistics::from_values(&mut recalls, self.k)
    }
}

/// Statistics for recall evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallStatistics {
    /// Value of k used for evaluation
    pub k: usize,
    /// Number of queries evaluated
    pub num_queries: usize,
    /// Mean recall across all queries
    pub mean: f64,
    /// Median recall
    pub median: f64,
    /// Minimum recall observed
    pub min: f64,
    /// Maximum recall observed
    pub max: f64,
    /// Standard deviation
    pub stddev: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

impl RecallStatistics {
    /// Calculate statistics from a vector of recall values
    ///
    /// # Arguments
    /// * `recalls` - Recall values (will be sorted in-place)
    /// * `k` - The k value used for recall calculation
    pub fn from_values(recalls: &mut [f64], k: usize) -> Self {
        if recalls.is_empty() {
            return Self::default_for_k(k);
        }

        // Sort for percentile calculations
        recalls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let num_queries = recalls.len();
        let mean = recalls.iter().sum::<f64>() / num_queries as f64;

        let median = if num_queries.is_multiple_of(2) {
            let mid = num_queries / 2;
            (recalls[mid - 1] + recalls[mid]) / 2.0
        } else {
            recalls[num_queries / 2]
        };

        let min = recalls[0];
        let max = recalls[num_queries - 1];

        let variance = recalls
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / num_queries as f64;
        let stddev = variance.sqrt();

        let p95 = percentile(recalls, 95.0);
        let p99 = percentile(recalls, 99.0);

        Self {
            k,
            num_queries,
            mean,
            median,
            min,
            max,
            stddev,
            p95,
            p99,
        }
    }

    /// Create default statistics for k (all zeros)
    fn default_for_k(k: usize) -> Self {
        Self {
            k,
            num_queries: 0,
            mean: 0.0,
            median: 0.0,
            min: 0.0,
            max: 0.0,
            stddev: 0.0,
            p95: 0.0,
            p99: 0.0,
        }
    }

    /// Format statistics as a human-readable string
    pub fn format(&self) -> String {
        format!(
            "Recall@{}: {:.4} (mean), {:.4} (median), {:.4} (min), {:.4} (max), {:.4} (stddev), {:.4} (p95), {:.4} (p99) over {} queries",
            self.k,
            self.mean,
            self.median,
            self.min,
            self.max,
            self.stddev,
            self.p95,
            self.p99,
            self.num_queries
        )
    }

    /// Format as percentage string
    pub fn format_percent(&self) -> String {
        format!(
            "Recall@{}: {:.2}% (mean), {:.2}% (median), {:.2}% (p95), {:.2}% (p99)",
            self.k,
            self.mean * 100.0,
            self.median * 100.0,
            self.p95 * 100.0,
            self.p99 * 100.0
        )
    }
}

impl std::fmt::Display for RecallStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

/// Calculate percentile from sorted values
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let idx = ((sorted_values.len() - 1) as f64 * p / 100.0).ceil() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_perfect() {
        let evaluator = RecallEvaluator::new(10);
        let retrieved = vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"];
        let ground_truth = vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"];

        let recall = evaluator.calculate_recall(&retrieved, &ground_truth);
        assert!((recall - 1.0).abs() < 1e-6, "Perfect recall should be 1.0");
    }

    #[test]
    fn test_recall_partial() {
        let evaluator = RecallEvaluator::new(10);
        let retrieved = vec!["v1", "v2", "v5", "v6", "v11", "v12", "v13", "v14", "v15", "v16"];
        let ground_truth = vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"];

        let recall = evaluator.calculate_recall(&retrieved, &ground_truth);
        assert_eq!(recall, 0.4); // 4 matches out of 10
    }

    #[test]
    fn test_recall_zero() {
        let evaluator = RecallEvaluator::new(10);
        let retrieved = vec!["v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20"];
        let ground_truth = vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"];

        let recall = evaluator.calculate_recall(&retrieved, &ground_truth);
        assert_eq!(recall, 0.0);
    }

    #[test]
    fn test_recall_truncates_to_k() {
        let evaluator = RecallEvaluator::new(5);

        // Even though we provide 10 items, only first 5 should be considered
        let retrieved = vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"];
        let ground_truth = vec!["v1", "v2", "v3", "v4", "v5", "v100", "v101", "v102", "v103", "v104"];

        let recall = evaluator.calculate_recall(&retrieved, &ground_truth);
        assert_eq!(recall, 1.0); // All first 5 match
    }

    #[test]
    fn test_recall_statistics() {
        let evaluator = RecallEvaluator::new(10);

        let results = vec![
            vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"],
            vec!["v1", "v2", "v3", "v4", "v5", "v11", "v12", "v13", "v14", "v15"],
            vec!["v1", "v2", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18"],
        ];

        let ground_truths = vec![
            vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"],
            vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"],
            vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"],
        ];

        let stats = evaluator.evaluate_batch(&results, &ground_truths);

        assert_eq!(stats.k, 10);
        assert_eq!(stats.num_queries, 3);
        assert_eq!(stats.min, 0.2); // Query 3: 2/10
        assert_eq!(stats.max, 1.0); // Query 1: 10/10
        assert_eq!(stats.mean, (1.0 + 0.5 + 0.2) / 3.0);
    }

    #[test]
    fn test_recall_statistics_empty() {
        let mut empty: Vec<f64> = vec![];
        let stats = RecallStatistics::from_values(&mut empty, 10);

        assert_eq!(stats.k, 10);
        assert_eq!(stats.num_queries, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&values, 50.0), 6.0);
        assert_eq!(percentile(&values, 95.0), 10.0);
        assert_eq!(percentile(&values, 100.0), 10.0);
    }

    #[test]
    fn test_format_output() {
        let mut recalls = vec![0.95, 0.97, 0.93, 0.98, 0.96];
        let stats = RecallStatistics::from_values(&mut recalls, 10);

        let formatted = stats.format_percent();
        assert!(formatted.contains("95.80%")); // mean
        assert!(formatted.contains("Recall@10"));
    }
}
