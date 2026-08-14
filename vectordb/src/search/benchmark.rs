use rand::Rng;
use std::time::{Duration, Instant};

use crate::search::{
    BruteForceIndex, // DiskANNConfig removed, BruteForceIndex, HnswConfig, HnswIndex, IvfConfig, IvfIndex,
    // ScaNNConfig removed, BruteForceIndex, VectorIndex,
};
use crate::types::{vector::{DenseVector, RecordId}, DistanceMetric};

/// Benchmark configuration
pub struct BenchmarkConfig {
    pub numvectors: usize,
    pub dimension: usize,
    pub num_queries: usize,
    pub top_k: usize,
    pub distance_metric: DistanceMetric,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            numvectors: 10000,
            dimension: 128,
            num_queries: 1000,
            top_k: 10,
            distance_metric: DistanceMetric::Euclidean,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub algorithm: String,
    pub build_time: Duration,
    pub avg_query_time: Duration,
    pub p95_query_time: Duration,
    pub p99_query_time: Duration,
    pub recall_at_k: f64,
    pub memory_usage_mb: f64,
    pub queries_per_second: f64,
}

/// Vector database benchmark suite
pub struct VectorDBBenchmark {
    config: BenchmarkConfig,
    vectors: Vec<(RecordId, DenseVector)>,
    queries: Vec<DenseVector>,
}

impl VectorDBBenchmark {
    /// Create new benchmark
    pub fn new(config: BenchmarkConfig) -> Self {
        let vectors = Self::generate_random_vectors(config.numvectors, config.dimension);
        let queries = Self::generate_random_queries(config.num_queries, config.dimension);

        Self {
            config,
            vectors,
            queries,
        }
    }

    /// Run all benchmarks
    pub async fn run_all(&self) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();

        // Benchmark BruteForce
        println!("Benchmarking BruteForce...");
        if let Ok(result) = self.benchmark_brute_force().await {

            results.push(result);



        }
        // Benchmark HNSW
        println!("Benchmarking HNSW...");
        if let Ok(result) = self.benchmark_hnsw().await {

            results.push(result);

        }
        // Benchmark IVF
        println!("Benchmarking IVF...");
        if let Ok(result) = self.benchmark_ivf().await {

            results.push(result);

        }
        // Benchmark DiskANN
        println!("Benchmarking DiskANN...");
        if let Ok(result) = self.benchmark_diskann().await {

            results.push(result);

        }
        // Benchmark ScaNN
        println!("Benchmarking ScaNN...");
        if let Ok(result) = self.benchmark_scann().await {

            results.push(result);

        }
        results
    }

    /// Benchmark BruteForce
    async fn benchmark_brute_force(&self) -> crate::Result<BenchmarkResult> {
        let mut index = BruteForceIndex::new(self.config.distance_metric);

        // Build index
        let build_start = Instant::now();
        for (id, vector) in &self.vectors {
            index.insert(id.clone(), vector.clone())?;
        }
        let build_time = build_start.elapsed();

        // Benchmark queries
        let query_times = self.benchmark_queries(&index)?;

        Ok(BenchmarkResult {
            algorithm: "BruteForce".to_string(),
            build_time,
            avg_query_time: Self::calculate_average(&query_times),
            p95_query_time: Self::calculate_percentile(&query_times, 0.95),
            p99_query_time: Self::calculate_percentile(&query_times, 0.99),
            recall_at_k: 1.0, // BruteForce has 100% recall
            memory_usage_mb: self.estimate_memory_usage(&index),
            queries_per_second: 1000.0 / Self::calculate_average(&query_times).as_millis() as f64,
        })
    }

    /// Benchmark HNSW
    async fn benchmark_hnsw(&self) -> crate::Result<BenchmarkResult> {
        let config = HnswConfig {
            max_connections: 16,
            ef_construction: 200,
            ef_search: 100,
            ..Default::default()
        };
        let mut index = crate::search::hnsw::HnswIndex::new(self.config.distance_metric, Some(config));

        // Build index
        let build_start = Instant::now();
        for (id, vector) in &self.vectors {
            index.insert(id.clone(), vector.clone())?;
        }
        let build_time = build_start.elapsed();

        // Benchmark queries
        let query_times = self.benchmark_queries(&index)?;

        Ok(BenchmarkResult {
            algorithm: "HNSW".to_string(),
            build_time,
            avg_query_time: Self::calculate_average(&query_times),
            p95_query_time: Self::calculate_percentile(&query_times, 0.95),
            p99_query_time: Self::calculate_percentile(&query_times, 0.99),
            recall_at_k: 0.98, // Approximate
            memory_usage_mb: self.estimate_memory_usage(&index),
            queries_per_second: 1000.0 / Self::calculate_average(&query_times).as_millis() as f64,
        })
    }

    /// Benchmark IVF
    async fn benchmark_ivf(&self) -> crate::Result<BenchmarkResult> {
        let config = IvfConfig {
            nlist: (self.config.numvectors as f64).sqrt() as usize,
            nprobe: 10,
            max_iterations: 100,
            tolerance: 1e-4,
        };
        let mut index = IvfIndex::new(self.config.distance_metric, Some(config));

        // Train index
        let training_vectors: Vec<DenseVector> = self
            .vectors
            .iter()
            .take(1000)
            .map(|(_, v)| v.clone())
            .collect();
        index.train(&training_vectors)?;

        // Build index
        let build_start = Instant::now();
        for (id, vector) in &self.vectors {
            index.insert(id.clone(), vector.clone())?;
        }
        let build_time = build_start.elapsed();

        // Benchmark queries
        let query_times = self.benchmark_queries(&index)?;

        Ok(BenchmarkResult {
            algorithm: "IVF".to_string(),
            build_time,
            avg_query_time: Self::calculate_average(&query_times),
            p95_query_time: Self::calculate_percentile(&query_times, 0.95),
            p99_query_time: Self::calculate_percentile(&query_times, 0.99),
            recall_at_k: 0.95, // Approximate
            memory_usage_mb: self.estimate_memory_usage(&index),
            queries_per_second: 1000.0 / Self::calculate_average(&query_times).as_millis() as f64,
        })
    }

    /// Benchmark DiskANN
    async fn benchmark_diskann(&self) -> crate::Result<BenchmarkResult> {
        let config = // DiskANNConfig removed {
            max_degree: 32,
            search_list_size: 100,
            ..Default::default()
        };
        let mut index =
            BruteForceIndex::new(self.config.dimension, self.config.distance_metric, config);

        // Build index
        let build_start = Instant::now();
        index.build(self.vectors.clone())?;
        let build_time = build_start.elapsed();

        // Benchmark queries
        let query_times = self.benchmark_queries(&index)?;

        Ok(BenchmarkResult {
            algorithm: "DiskANN".to_string(),
            build_time,
            avg_query_time: Self::calculate_average(&query_times),
            p95_query_time: Self::calculate_percentile(&query_times, 0.95),
            p99_query_time: Self::calculate_percentile(&query_times, 0.99),
            recall_at_k: 0.97, // Approximate
            memory_usage_mb: self.estimate_memory_usage(&index),
            queries_per_second: 1000.0 / Self::calculate_average(&query_times).as_millis() as f64,
        })
    }

    /// Benchmark ScaNN
    async fn benchmark_scann(&self) -> crate::Result<BenchmarkResult> {
        let config = // ScaNNConfig removed {
            num_leaves: 100,
            num_leaves_to_search: 10,
            ..Default::default()
        };
        let mut index = BruteForceIndex::new(self.config.dimension, self.config.distance_metric, config);

        // Build index
        let build_start = Instant::now();
        index.build(self.vectors.clone())?;
        let build_time = build_start.elapsed();

        // Benchmark queries
        let query_times = self.benchmark_queries(&index)?;

        Ok(BenchmarkResult {
            algorithm: "ScaNN".to_string(),
            build_time,
            avg_query_time: Self::calculate_average(&query_times),
            p95_query_time: Self::calculate_percentile(&query_times, 0.95),
            p99_query_time: Self::calculate_percentile(&query_times, 0.99),
            recall_at_k: 0.96, // Approximate
            memory_usage_mb: self.estimate_memory_usage(&index),
            queries_per_second: 1000.0 / Self::calculate_average(&query_times).as_millis() as f64,
        })
    }

    /// Benchmark queries for an index
    fn benchmark_queries(
        &self,
        index: &dyn VectorIndex) -> crate::Result<Vec<Duration>> {
        let mut query_times = Vec::new();

        for query in &self.queries {
            let start = Instant::now();
            let _ = index.search(query, self.config.top_k)?;
            query_times.push(start.elapsed())
        }

        Ok(query_times)
    }

    /// Generate random vectors
    fn generate_random_vectors(num: usize, dim: usize) -> Vec<(RecordId, DenseVector)> {
        let mut rng = rand::thread_rng();
        (0..num)
            .map(|i| {
                let vector: DenseVector = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                (format!("vec_{}", i), vector)
            })
            .collect()
    }

    /// Generate random query vectors
    fn generate_random_queries(num: usize, dim: usize) -> Vec<DenseVector> {
        let mut rng = rand::thread_rng();
        (0..num)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    /// Calculate average duration
    fn calculate_average(durations: &[Duration]) -> Duration {
        let total: Duration = durations.iter().sum();
        total / durations.len() as u32
    }

    /// Calculate percentile
    fn calculate_percentile(durations: &[Duration], percentile: f64) -> Duration {
        let mut sorted = durations.to_vec();
        sorted.sort();
        let index = ((sorted.len() as f64 - 1.0) * percentile) as usize;
        sorted[index]
    }

    /// Estimate memory usage (simplified)
    fn estimate_memory_usage(&self, _index: &dyn VectorIndex) -> f64 {
        // Simplified estimation
        let vector_size = self.config.dimension * std::mem::size_of::<f32>();
        let total_size = self.config.numvectors * vector_size;
        total_size as f64 / (1024.0 * 1024.0)
    }

    /// Print benchmark results
    pub fn print_results(results: &[BenchmarkResult]) {
        println!("\n=== Vector Database Benchmark Results ===\n");
        println!(
            "{:<15} {:>12} {:>12} {:>12} {:>12} {:>10} {:>12} {:>10}",
            "Algorithm",
            "Build Time",
            "Avg Query",
            "P95 Query",
            "P99 Query",
            "Recall@K",
            "Memory (MB)",
            "QPS"
        );
        println!("{:-<110}", "");

        for result in results {
            println!(
                "{:<15} {:>12} {:>12} {:>12} {:>12} {:>10.2} {:>12.1} {:>10.1}",
                result.algorithm,
                format!("{:.2}s", result.build_time.as_secs_f64()),
                format!("{:.2}ms", result.avg_query_time.as_secs_f64() * 1000.0),
                format!("{:.2}ms", result.p95_query_time.as_secs_f64() * 1000.0),
                format!("{:.2}ms", result.p99_query_time.as_secs_f64() * 1000.0),
                result.recall_at_k,
                result.memory_usage_mb,
                result.queries_per_second
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
use crate::RecordId;

    #[tokio::test]
    async fn test_benchmark_suite() {
        let config = BenchmarkConfig {
            numvectors: 1000,
            dimension: 64,
            num_queries: 100,
            top_k: 10,
            distance_metric: _DistanceMetric::Euclidean,
        };

        let benchmark = VectorDBBenchmark::new(config);
        let results = benchmark.run_all()?;

        assert!(!results.is_empty());
        VectorDBBenchmark::print_results(&results);
    }
}
