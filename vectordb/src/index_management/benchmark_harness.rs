//! Benchmarking harness for Phase 3 components
//!
//! Provides a test framework for measuring:
//! - Index creation performance
//! - Data migration throughput
//! - Query performance
//! - Memory usage patterns
//! - Concurrency characteristics

use crate::index_management::{BenchmarkConfig, BenchmarkResult, OperationProfiler};

/// Main benchmark harness for Phase 3 components
pub struct Phase3BenchmarkHarness {
    profiler: OperationProfiler,
    config: BenchmarkConfig,
}

impl Phase3BenchmarkHarness {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            profiler: OperationProfiler::new(),
            config,
        }
    }

    /// Run index creation benchmark
    pub async fn benchmark_index_creation<F>(&self, create_fn: F) -> Vec<BenchmarkResult>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>>>>,
    {
        let mut results = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = create_fn().await;
        }

        // Benchmark
        for iteration in 0..self.config.benchmark_iterations {
            let handle = self.profiler.start_operation("index_creation");

            match create_fn().await {
                Ok(_) => {
                    handle.finish_success(1);
                }
                Err(e) => {
                    handle.finish_error(&e);
                }
            }

            if let Some(profile) = self.profiler.get_profile("index_creation") {
                if let Some(snapshot) = profile.snapshots.last() {
                    results.push(BenchmarkResult {
                        operation_name: "index_creation".to_string(),
                        record_count: 0,
                        vector_dimension: 0,
                        iterations: iteration + 1,
                        total_duration_ms: profile.avg_duration_ms,
                        avg_duration_per_iteration_ms: profile.avg_duration_ms,
                        throughput_records_per_sec: snapshot.throughput_ops_per_sec,
                    });
                }
            }
        }

        results
    }

    /// Run migration benchmark at various scales
    pub async fn benchmark_migration_scaling<F>(&self, migrate_fn: F) -> Vec<BenchmarkResult>
    where
        F: Fn(usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize, String>>>>,
    {
        let mut results = Vec::new();

        for &record_count in &self.config.record_counts {
            // Warmup
            for _ in 0..self.config.warmup_iterations {
                let _ = migrate_fn(record_count).await;
            }

            // Benchmark
            for _ in 0..self.config.benchmark_iterations {
                let op_name = format!("migrate_{}", record_count);
                let handle = self.profiler.start_operation(&op_name);

                match migrate_fn(record_count).await {
                    Ok(migrated) => {
                        handle.finish_success(migrated);
                    }
                    Err(e) => {
                        handle.finish_error(&e);
                    }
                }
            }

            if let Some(profile) = self
                .profiler
                .get_profile(&format!("migrate_{}", record_count))
            {
                let throughput = profile
                    .snapshots
                    .last()
                    .map(|s| s.throughput_ops_per_sec)
                    .unwrap_or(0.0);

                results.push(BenchmarkResult {
                    operation_name: "migration".to_string(),
                    record_count,
                    vector_dimension: 0,
                    iterations: self.config.benchmark_iterations,
                    total_duration_ms: profile.avg_duration_ms
                        * self.config.benchmark_iterations as f64,
                    avg_duration_per_iteration_ms: profile.avg_duration_ms,
                    throughput_records_per_sec: throughput,
                });
            }
        }

        results
    }

    /// Run query performance benchmark
    pub async fn benchmark_query_performance<F>(&self, query_fn: F) -> Vec<BenchmarkResult>
    where
        F: Fn(usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize, String>>>>,
    {
        let mut results = Vec::new();

        for &record_count in &self.config.record_counts {
            // Warmup
            for _ in 0..self.config.warmup_iterations {
                let _ = query_fn(record_count).await;
            }

            // Benchmark
            for _ in 0..self.config.benchmark_iterations {
                let op_name = format!("query_{}", record_count);
                let handle = self.profiler.start_operation(&op_name);

                match query_fn(record_count).await {
                    Ok(results_count) => {
                        handle.finish_success(results_count);
                    }
                    Err(e) => {
                        handle.finish_error(&e);
                    }
                }
            }

            if let Some(profile) = self
                .profiler
                .get_profile(&format!("query_{}", record_count))
            {
                results.push(BenchmarkResult {
                    operation_name: "query".to_string(),
                    record_count,
                    vector_dimension: 0,
                    iterations: self.config.benchmark_iterations,
                    total_duration_ms: profile.avg_duration_ms
                        * self.config.benchmark_iterations as f64,
                    avg_duration_per_iteration_ms: profile.avg_duration_ms,
                    throughput_records_per_sec: profile
                        .snapshots
                        .last()
                        .map(|s| s.throughput_ops_per_sec)
                        .unwrap_or(0.0),
                });
            }
        }

        results
    }

    /// Generate comparative benchmark report
    pub fn generate_report(&self, results: &[BenchmarkResult]) -> String {
        let mut report = String::from("=== Phase 3 Benchmark Report ===\n\n");

        for result in results {
            report.push_str(&format!("{}\n", result.format()));
        }

        report.push_str("\n=== Summary Statistics ===\n");

        // Group by operation
        let mut operations: std::collections::HashMap<String, Vec<&BenchmarkResult>> =
            std::collections::HashMap::new();

        for result in results {
            operations
                .entry(result.operation_name.clone())
                .or_default()
                .push(result);
        }

        for (op_name, op_results) in operations {
            let avg_duration: f64 = op_results
                .iter()
                .map(|r| r.avg_duration_per_iteration_ms)
                .sum::<f64>()
                / op_results.len() as f64;
            let max_duration = op_results
                .iter()
                .map(|r| r.avg_duration_per_iteration_ms)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            report.push_str(&format!(
                "\n{}:\n  Avg: {:.2}ms | Max: {:.2}ms | Iterations: {}\n",
                op_name,
                avg_duration,
                max_duration,
                op_results.len()
            ));
        }

        report
    }

    /// Get profiler for custom measurements
    pub fn profiler(&self) -> &OperationProfiler {
        &self.profiler
    }
}

/// Helper for creating standardized benchmark configurations
pub struct BenchmarkConfigBuilder {
    config: BenchmarkConfig,
}

impl BenchmarkConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    pub fn warmup_iterations(mut self, iterations: usize) -> Self {
        self.config.warmup_iterations = iterations;
        self
    }

    pub fn benchmark_iterations(mut self, iterations: usize) -> Self {
        self.config.benchmark_iterations = iterations;
        self
    }

    pub fn record_counts(mut self, counts: Vec<usize>) -> Self {
        self.config.record_counts = counts;
        self
    }

    pub fn vector_dimensions(mut self, dimensions: Vec<usize>) -> Self {
        self.config.vector_dimensions = dimensions;
        self
    }

    pub fn build(self) -> BenchmarkConfig {
        self.config
    }
}

impl Default for BenchmarkConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick benchmark for smoke testing
pub async fn quick_benchmark<F>(
    operation_name: &str,
    iterations: usize,
    operation: F,
) -> BenchmarkResult
where
    F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize, String>>>>,
{
    let profiler = OperationProfiler::new();

    for _ in 0..iterations {
        let handle = profiler.start_operation(operation_name);

        match operation().await {
            Ok(count) => {
                handle.finish_success(count);
            }
            Err(e) => {
                handle.finish_error(&e);
            }
        }
    }

    let profile = profiler.get_profile(operation_name).unwrap();

    BenchmarkResult {
        operation_name: operation_name.to_string(),
        record_count: 0,
        vector_dimension: 0,
        iterations,
        total_duration_ms: profile.avg_duration_ms * iterations as f64,
        avg_duration_per_iteration_ms: profile.avg_duration_ms,
        throughput_records_per_sec: profile
            .snapshots
            .last()
            .map(|s| s.throughput_ops_per_sec)
            .unwrap_or(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_builder() {
        let config = BenchmarkConfigBuilder::new()
            .warmup_iterations(5)
            .benchmark_iterations(20)
            .record_counts(vec![1000, 10000])
            .build();

        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.benchmark_iterations, 20);
        assert_eq!(config.record_counts.len(), 2);
    }

    #[test]
    fn test_benchmark_result_formatting() {
        let result = BenchmarkResult {
            operation_name: "test_op".to_string(),
            record_count: 1000,
            vector_dimension: 256,
            iterations: 10,
            total_duration_ms: 100.0,
            avg_duration_per_iteration_ms: 10.0,
            throughput_records_per_sec: 100.0,
        };

        let formatted = result.format();
        assert!(formatted.contains("test_op"));
        assert!(formatted.contains("1000x256"));
        assert!(formatted.contains("10.00ms/iter"));
    }

    #[tokio::test]
    async fn test_quick_benchmark() {
        let result = quick_benchmark("test_operation", 5, || Box::pin(async { Ok(100) })).await;

        assert_eq!(result.operation_name, "test_operation");
        assert_eq!(result.iterations, 5);
        assert!(result.avg_duration_per_iteration_ms >= 0.0);
    }
}
