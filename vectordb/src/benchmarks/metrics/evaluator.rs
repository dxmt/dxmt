//! Unified metrics evaluator for comprehensive search quality assessment
//!
//! Provides a single interface to evaluate all quality metrics together.

use super::{
    MetricStatistics, NdcgEvaluator, PrecisionEvaluator, MapEvaluator,
};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MetricsReport {
    pub k: usize,
    pub precision: MetricStatistics,
    pub ndcg: MetricStatistics,
    pub map: MetricStatistics,
}

impl MetricsReport {
    /// Print a formatted report
    pub fn print(&self) {
        println!("\n=== Quality Metrics Report @k={} ===\n", self.k);

        println!("Precision@{}:", self.k);
        Self::print_stats(&self.precision);

        println!("\nNDCG@{}:", self.k);
        Self::print_stats(&self.ndcg);

        println!("\nMAP@{}:", self.k);
        Self::print_stats(&self.map);
    }

    fn print_stats(stats: &MetricStatistics) {
        println!("  Mean:   {:.4}", stats.mean);
        println!("  Median: {:.4}", stats.median);
        println!("  StdDev: {:.4}", stats.stddev);
        println!("  Min:    {:.4}", stats.min);
        println!("  Max:    {:.4}", stats.max);
        println!("  P95:    {:.4}", stats.p95);
        println!("  P99:    {:.4}", stats.p99);
    }
}

#[derive(Debug, Clone)]
pub struct MetricsEvaluator;

impl MetricsEvaluator {
    /// Evaluate all metrics for a single k value
    ///
    /// # Arguments
    /// * `results` - Vector of (retrieved_ids, ground_truth_ids) for each query
    /// * `k` - Number of top results to consider
    ///
    /// # Returns
    /// Complete metrics report
    pub fn evaluate(results: &[(Vec<String>, Vec<String>)], k: usize) -> MetricsReport {
        MetricsReport {
            k,
            precision: PrecisionEvaluator::evaluate(results, k),
            ndcg: NdcgEvaluator::evaluate(results, k),
            map: MapEvaluator::evaluate(results, k),
        }
    }

    /// Evaluate all metrics at multiple k values
    ///
    /// # Arguments
    /// * `results` - Vector of (retrieved_ids, ground_truth_ids) for each query
    /// * `k_values` - List of k values to evaluate
    ///
    /// # Returns
    /// Map of k -> MetricsReport
    pub fn evaluate_multi(
        results: &[(Vec<String>, Vec<String>)],
        k_values: &[usize],
    ) -> HashMap<usize, MetricsReport> {
        k_values
            .iter()
            .map(|&k| (k, Self::evaluate(results, k)))
            .collect()
    }

    /// Generate a comprehensive benchmark report
    ///
    /// # Arguments
    /// * `results` - Vector of (retrieved_ids, ground_truth_ids) for each query
    /// * `k_values` - List of k values to evaluate
    /// * `index_name` - Name of the index being evaluated
    ///
    /// # Returns
    /// Formatted report string
    pub fn generate_report(
        results: &[(Vec<String>, Vec<String>)],
        k_values: &[usize],
        index_name: &str,
    ) -> String {
        let mut report = String::new();

        report.push_str("\n╔═══════════════════════════════════════════╗\n");
        report.push_str(&format!("║  Quality Metrics Report: {}  ║\n", index_name));
        report.push_str("╚═══════════════════════════════════════════╝\n\n");

        report.push_str(&format!("Total Queries: {}\n\n", results.len()));

        // Table header
        report.push_str("┌──────┬───────────┬───────────┬───────────┐\n");
        report.push_str("│  k   │ Precision │   NDCG    │    MAP    │\n");
        report.push_str("├──────┼───────────┼───────────┼───────────┤\n");

        for &k in k_values {
            let metrics = Self::evaluate(results, k);
            report.push_str(&format!(
                "│ {:4} │   {:.4}   │   {:.4}   │   {:.4}   │\n",
                k, metrics.precision.mean, metrics.ndcg.mean, metrics.map.mean
            ));
        }

        report.push_str("└──────┴───────────┴───────────┴───────────┘\n\n");

        // Detailed statistics for largest k
        if let Some(&max_k) = k_values.iter().max() {
            let metrics = Self::evaluate(results, max_k);

            report.push_str(&format!("Detailed Statistics @k={}:\n\n", max_k));

            report.push_str("Precision:\n");
            report.push_str(&format!("  Mean:   {:.4}\n", metrics.precision.mean));
            report.push_str(&format!("  Median: {:.4}\n", metrics.precision.median));
            report.push_str(&format!("  StdDev: {:.4}\n", metrics.precision.stddev));
            report.push_str(&format!("  P95:    {:.4}\n", metrics.precision.p95));
            report.push_str(&format!("  P99:    {:.4}\n\n", metrics.precision.p99));

            report.push_str("NDCG:\n");
            report.push_str(&format!("  Mean:   {:.4}\n", metrics.ndcg.mean));
            report.push_str(&format!("  Median: {:.4}\n", metrics.ndcg.median));
            report.push_str(&format!("  StdDev: {:.4}\n", metrics.ndcg.stddev));
            report.push_str(&format!("  P95:    {:.4}\n", metrics.ndcg.p95));
            report.push_str(&format!("  P99:    {:.4}\n\n", metrics.ndcg.p99));

            report.push_str("MAP:\n");
            report.push_str(&format!("  Mean:   {:.4}\n", metrics.map.mean));
            report.push_str(&format!("  Median: {:.4}\n", metrics.map.median));
            report.push_str(&format!("  StdDev: {:.4}\n", metrics.map.stddev));
            report.push_str(&format!("  P95:    {:.4}\n", metrics.map.p95));
            report.push_str(&format!("  P99:    {:.4}\n", metrics.map.p99));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_comprehensive() {
        let results = vec![
            (
                vec!["1", "2", "3", "4", "5"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                vec!["1", "2", "3", "4", "5"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            (
                vec!["1", "6", "2", "7", "3"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                vec!["1", "2", "3", "4", "5"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
        ];

        let report = MetricsEvaluator::evaluate(&results, 5);

        assert_eq!(report.k, 5);
        assert!(report.precision.mean > 0.5);
        assert!(report.ndcg.mean > 0.5);
        assert!(report.map.mean > 0.5);
    }

    #[test]
    fn test_evaluate_multi() {
        let results = vec![
            (
                vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
                vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
            ),
        ];

        let reports = MetricsEvaluator::evaluate_multi(&results, &[1, 3, 5]);

        assert_eq!(reports.len(), 3);
        assert!(reports.contains_key(&1));
        assert!(reports.contains_key(&3));
        assert!(reports.contains_key(&5));
    }

    #[test]
    fn test_generate_report() {
        let results = vec![
            (
                vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
                vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
            ),
        ];

        let report = MetricsEvaluator::generate_report(&results, &[1, 3, 5], "HNSW");

        assert!(report.contains("HNSW"));
        assert!(report.contains("Precision"));
        assert!(report.contains("NDCG"));
        assert!(report.contains("MAP"));
    }

    #[test]
    fn test_print_report() {
        let results = vec![
            (
                vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
                vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect(),
            ),
        ];

        let report = MetricsEvaluator::evaluate(&results, 3);
        report.print(); // Should not panic
    }
}
