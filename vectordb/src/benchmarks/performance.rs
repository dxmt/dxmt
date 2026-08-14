//! Performance benchmarking suite for DXMT VectorDB

use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use rand::prelude::*;
use statrs::statistics::Statistics;

use crate::types::{DenseVector, Record, RecordId, Payload};
use crate::acceleration::{
    PerformanceOptimizer, GpuAccelerator, SimdAccelerator, 
    ParallelExecutor, MemoryOptimizer, SimdLevel
};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub vector_dimensions: Vec<usize>,
    pub dataset_sizes: Vec<usize>,
    pub batch_sizes: Vec<usize>,
    pub k_values: Vec<usize>,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub enable_gpu: bool,
    pub enable_simd: bool,
    pub enable_parallel: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            vector_dimensions: vec![128, 256, 512, 768, 1024, 1536],
            dataset_sizes: vec![1000, 10000, 100000, 1000000],
            batch_sizes: vec![1, 10, 100, 1000],
            k_values: vec![10, 50, 100],
            iterations: 100,
            warmup_iterations: 10,
            enable_gpu: true,
            enable_simd: true,
            enable_parallel: true,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub config: BenchmarkConfig,
    pub system_info: SystemInfo,
    pub tests: Vec<TestResult>,
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub simd_level: String,
    pub gpu_name: Option<String>,
    pub os: String,
    pub rust_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub dimension: usize,
    pub dataset_size: usize,
    pub batch_size: usize,
    pub backend: String,
    pub operation: String,
    pub latency_ms: LatencyStats,
    pub throughput: ThroughputStats,
    pub memory_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub stddev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStats {
    pub ops_per_sec: f64,
    pub vectors_per_sec: f64,
    pub mb_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_tests: usize,
    pub total_duration_sec: f64,
    pub best_latency_ms: f64,
    pub best_throughput_ops: f64,
    pub gpu_speedup: f64,
    pub simd_speedup: f64,
    pub recommendations: Vec<String>,
}

/// Performance benchmark runner
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    optimizer: Arc<PerformanceOptimizer>,
    results: Vec<TestResult>,
}

impl BenchmarkRunner {
    pub async fn new(config: BenchmarkConfig) -> Self {
        let optimizer = Arc::new(PerformanceOptimizer::new().await);
        
        Self {
            config,
            optimizer,
            results: Vec::new(),
        }
    }
    
    pub async fn run_all(&mut self) -> BenchmarkResults {
        let start = Instant::now();
        let system_info = self.collect_system_info().await;
        
        println!("🚀 Starting DXMT VectorDB Performance Benchmarks");
        println!("   System: {} cores, {:.1} GB RAM", system_info.cpu_cores, system_info.memory_gb);
        println!("   SIMD: {}", system_info.simd_level);
        if let Some(gpu) = &system_info.gpu_name {
            println!("   GPU: {}", gpu);
        }
        println!();
        
        // Run benchmarks for each configuration
        for dimension in &self.config.vector_dimensions {
            for dataset_size in &self.config.dataset_sizes {
                for batch_size in &self.config.batch_sizes {
                    // Skip very large configurations
                    if dimension * dataset_size > 10_000_000 {
                        continue;
                    }
                    
                    println!("Testing: dim={}, dataset={}, batch={}", 
                             dimension, dataset_size, batch_size);
                    
                    // Test different backends
                    if self.config.enable_gpu {
                        self.benchmark_gpu(*dimension, *dataset_size, *batch_size).await;
                    }
                    
                    if self.config.enable_simd {
                        self.benchmark_simd(*dimension, *dataset_size, *batch_size).await;
                    }
                    
                    if self.config.enable_parallel {
                        self.benchmark_parallel(*dimension, *dataset_size, *batch_size).await;
                    }
                    
                    // Always benchmark baseline
                    self.benchmark_baseline(*dimension, *dataset_size, *batch_size).await;
                }
            }
        }
        
        let total_duration = start.elapsed().as_secs_f64();
        let summary = self.generate_summary(total_duration);
        
        BenchmarkResults {
            config: self.config.clone(),
            system_info,
            tests: self.results.clone(),
            summary,
        }
    }
    
    async fn benchmark_gpu(&mut self, dimension: usize, dataset_size: usize, batch_size: usize) {
        if let Some(gpu) = &self.optimizer.capabilities().gpu_accelerator {
            let vectors = self.generate_vectors(dimension, dataset_size);
            let queries = self.generate_vectors(dimension, batch_size);
            
            // Warmup
            for _ in 0..self.config.warmup_iterations {
                let _ = gpu.cosine_similarity_batch(&queries, &vectors).await;
            }
            
            // Measure
            let mut latencies = Vec::new();
            let start = Instant::now();
            
            for _ in 0..self.config.iterations {
                let iter_start = Instant::now();
                let _ = gpu.cosine_similarity_batch(&queries, &vectors).await;
                latencies.push(iter_start.elapsed().as_secs_f64() * 1000.0);
            }
            
            let total_duration = start.elapsed();
            
            self.results.push(TestResult {
                name: format!("gpu_cosine_similarity_{}x{}", dimension, dataset_size),
                dimension,
                dataset_size,
                batch_size,
                backend: "GPU".to_string(),
                operation: "cosine_similarity".to_string(),
                latency_ms: Self::calculate_latency_stats(&latencies),
                throughput: Self::calculate_throughput_stats(
                    batch_size * dataset_size,
                    total_duration,
                    dimension * 4 * dataset_size,
                ),
                memory_mb: (dimension * dataset_size * 4) as f64 / 1_048_576.0,
            });
        }
    }
    
    async fn benchmark_simd(&mut self, dimension: usize, dataset_size: usize, batch_size: usize) {
        let simd = SimdAccelerator::new(SimdLevel::detect());
        let vectors = self.generate_vectors(dimension, dataset_size);
        let queries = self.generate_vectors(dimension, batch_size);
        
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            for query in &queries {
                let _ = simd.batch_cosine_similarity(query, &vectors);
            }
        }
        
        // Measure
        let mut latencies = Vec::new();
        let start = Instant::now();
        
        for _ in 0..self.config.iterations {
            let iter_start = Instant::now();
            for query in &queries {
                let _ = simd.batch_cosine_similarity(query, &vectors);
            }
            latencies.push(iter_start.elapsed().as_secs_f64() * 1000.0);
        }
        
        let total_duration = start.elapsed();
        
        self.results.push(TestResult {
            name: format!("simd_cosine_similarity_{}x{}", dimension, dataset_size),
            dimension,
            dataset_size,
            batch_size,
            backend: format!("SIMD_{:?}", simd.level()),
            operation: "cosine_similarity".to_string(),
            latency_ms: Self::calculate_latency_stats(&latencies),
            throughput: Self::calculate_throughput_stats(
                batch_size * dataset_size,
                total_duration,
                dimension * 4 * dataset_size,
            ),
            memory_mb: (dimension * dataset_size * 4) as f64 / 1_048_576.0,
        });
    }
    
    async fn benchmark_parallel(&mut self, dimension: usize, dataset_size: usize, batch_size: usize) {
        let parallel = ParallelExecutor::new(num_cpus::get());
        let vectors = self.generate_vectors(dimension, dataset_size);
        let queries = self.generate_vectors(dimension, batch_size);
        
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            for query in &queries {
                let _ = parallel.parallel_cosine_similarity(query, &vectors);
            }
        }
        
        // Measure
        let mut latencies = Vec::new();
        let start = Instant::now();
        
        for _ in 0..self.config.iterations {
            let iter_start = Instant::now();
            for query in &queries {
                let _ = parallel.parallel_cosine_similarity(query, &vectors);
            }
            latencies.push(iter_start.elapsed().as_secs_f64() * 1000.0);
        }
        
        let total_duration = start.elapsed();
        
        self.results.push(TestResult {
            name: format!("parallel_cosine_similarity_{}x{}", dimension, dataset_size),
            dimension,
            dataset_size,
            batch_size,
            backend: format!("Parallel_{}_threads", num_cpus::get()),
            operation: "cosine_similarity".to_string(),
            latency_ms: Self::calculate_latency_stats(&latencies),
            throughput: Self::calculate_throughput_stats(
                batch_size * dataset_size,
                total_duration,
                dimension * 4 * dataset_size,
            ),
            memory_mb: (dimension * dataset_size * 4) as f64 / 1_048_576.0,
        });
    }
    
    async fn benchmark_baseline(&mut self, dimension: usize, dataset_size: usize, batch_size: usize) {
        let vectors = self.generate_vectors(dimension, dataset_size);
        let queries = self.generate_vectors(dimension, batch_size);
        
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            for query in &queries {
                for vector in &vectors {
                    let _ = Self::cosine_similarity_scalar(query, vector);
                }
            }
        }
        
        // Measure
        let mut latencies = Vec::new();
        let start = Instant::now();
        
        for _ in 0..self.config.iterations {
            let iter_start = Instant::now();
            for query in &queries {
                for vector in &vectors {
                    let _ = Self::cosine_similarity_scalar(query, vector);
                }
            }
            latencies.push(iter_start.elapsed().as_secs_f64() * 1000.0);
        }
        
        let total_duration = start.elapsed();
        
        self.results.push(TestResult {
            name: format!("baseline_cosine_similarity_{}x{}", dimension, dataset_size),
            dimension,
            dataset_size,
            batch_size,
            backend: "Scalar".to_string(),
            operation: "cosine_similarity".to_string(),
            latency_ms: Self::calculate_latency_stats(&latencies),
            throughput: Self::calculate_throughput_stats(
                batch_size * dataset_size,
                total_duration,
                dimension * 4 * dataset_size,
            ),
            memory_mb: (dimension * dataset_size * 4) as f64 / 1_048_576.0,
        });
    }
    
    fn generate_vectors(&self, dimension: usize, count: usize) -> Vec<DenseVector> {
        let mut rng = thread_rng();
        (0..count)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect()
    }
    
    fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
    
    fn calculate_latency_stats(latencies: &[f64]) -> LatencyStats {
        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let variance = latencies.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / latencies.len() as f64;
        let stddev = variance.sqrt();
        
        LatencyStats {
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            mean,
            median: sorted[sorted.len() / 2],
            p95: sorted[(sorted.len() as f64 * 0.95) as usize],
            p99: sorted[(sorted.len() as f64 * 0.99) as usize],
            stddev,
        }
    }
    
    fn calculate_throughput_stats(
        operations: usize,
        duration: Duration,
        bytes: usize,
    ) -> ThroughputStats {
        let secs = duration.as_secs_f64();
        
        ThroughputStats {
            ops_per_sec: operations as f64 / secs,
            vectors_per_sec: operations as f64 / secs,
            mb_per_sec: (bytes as f64 / 1_048_576.0) / secs,
        }
    }
    
    async fn collect_system_info(&self) -> SystemInfo {
        let capabilities = self.optimizer.capabilities();
        
        SystemInfo {
            cpu_model: Self::get_cpu_model(),
            cpu_cores: capabilities.cpu_cores,
            memory_gb: capabilities.total_memory_gb,
            simd_level: format!("{:?}", capabilities.simd_level),
            gpu_name: capabilities.gpu_devices.first()
                .map(|d| d.name.clone()),
            os: std::env::consts::OS.to_string(),
            rust_version: "1.70.0".to_string(),
        }
    }
    
    fn get_cpu_model() -> String {
        #[cfg(target_os = "linux")]
        {
            if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in cpuinfo.lines() {
                    if line.starts_with("model name") {
                        return line.split(':')
                            .nth(1)
                            .map(|s| s.trim().to_string())
                            .unwrap_or_else(|| "Unknown CPU".to_string());
                    }
                }
            }
        }
        "Unknown CPU".to_string()
    }
    
    fn generate_summary(&self, total_duration: f64) -> BenchmarkSummary {
        let mut best_latency = f64::MAX;
        let mut best_throughput = 0.0;
        let mut gpu_speedups = Vec::new();
        let mut simd_speedups = Vec::new();
        
        // Calculate speedups
        for result in &self.results {
            best_latency = best_latency.min(result.latency_ms.mean);
            best_throughput = best_throughput.max(result.throughput.ops_per_sec);
            
            // Find baseline for comparison
            let baseline = self.results.iter()
                .find(|r| {
                    r.backend == "Scalar" &&
                    r.dimension == result.dimension &&
                    r.dataset_size == result.dataset_size
                })
                .map(|r| r.latency_ms.mean);
            
            if let Some(baseline_latency) = baseline {
                match result.backend.as_str() {
                    "GPU" => gpu_speedups.push(baseline_latency / result.latency_ms.mean),
                    backend if backend.starts_with("SIMD") => {
                        simd_speedups.push(baseline_latency / result.latency_ms.mean)
                    },
                    _ => {}
                }
            }
        }
        
        let gpu_speedup = if gpu_speedups.is_empty() {
            1.0
        } else {
            gpu_speedups.iter().sum::<f64>() / gpu_speedups.len() as f64
        };
        
        let simd_speedup = if simd_speedups.is_empty() {
            1.0
        } else {
            simd_speedups.iter().sum::<f64>() / simd_speedups.len() as f64
        };
        
        // Generate recommendations
        let mut recommendations = Vec::new();
        
        if gpu_speedup > 2.0 {
            recommendations.push(format!(
                "GPU acceleration provides {:.1}x speedup - recommended for large datasets",
                gpu_speedup
            ));
        }
        
        if simd_speedup > 1.5 {
            recommendations.push(format!(
                "SIMD provides {:.1}x speedup - enable for all deployments",
                simd_speedup
            ));
        }
        
        if best_latency < 10.0 {
            recommendations.push("Excellent latency achieved - suitable for real-time applications".to_string());
        }
        
        BenchmarkSummary {
            total_tests: self.results.len(),
            total_duration_sec: total_duration,
            best_latency_ms: best_latency,
            best_throughput_ops: best_throughput,
            gpu_speedup,
            simd_speedup,
            recommendations,
        }
    }
}

/// Benchmark report generator
pub struct BenchmarkReport;

impl BenchmarkReport {
    pub fn generate_markdown(results: &BenchmarkResults) -> String {
        let mut report = String::new();
        
        report.push_str("# DXMT VectorDB Performance Benchmark Report\n\n");
        
        // System info
        report.push_str("## System Information\n\n");
        report.push_str(&format!("- **CPU**: {} ({} cores)\n", 
            results.system_info.cpu_model, results.system_info.cpu_cores));
        report.push_str(&format!("- **Memory**: {:.1} GB\n", results.system_info.memory_gb));
        report.push_str(&format!("- **SIMD**: {}\n", results.system_info.simd_level));
        if let Some(gpu) = &results.system_info.gpu_name {
            report.push_str(&format!("- **GPU**: {}\n", gpu));
        }
        report.push_str(&format!("- **OS**: {}\n", results.system_info.os));
        report.push_str("\n");
        
        // Summary
        report.push_str("## Summary\n\n");
        report.push_str(&format!("- **Total Tests**: {}\n", results.summary.total_tests));
        report.push_str(&format!("- **Duration**: {:.1}s\n", results.summary.total_duration_sec));
        report.push_str(&format!("- **Best Latency**: {:.2}ms\n", results.summary.best_latency_ms));
        report.push_str(&format!("- **Best Throughput**: {:.0} ops/sec\n", results.summary.best_throughput_ops));
        report.push_str(&format!("- **GPU Speedup**: {:.1}x\n", results.summary.gpu_speedup));
        report.push_str(&format!("- **SIMD Speedup**: {:.1}x\n", results.summary.simd_speedup));
        report.push_str("\n");
        
        // Recommendations
        if !results.summary.recommendations.is_empty() {
            report.push_str("## Recommendations\n\n");
            for rec in &results.summary.recommendations {
                report.push_str(&format!("- {}\n", rec));
            }
            report.push_str("\n");
        }
        
        // Detailed results table
        report.push_str("## Detailed Results\n\n");
        report.push_str("| Backend | Dimension | Dataset | Batch | Mean Latency (ms) | P95 (ms) | Throughput (ops/s) |\n");
        report.push_str("|---------|-----------|---------|-------|-------------------|----------|--------------------|\n");
        
        for result in &results.tests {
            report.push_str(&format!(
                "| {} | {} | {} | {} | {:.2} | {:.2} | {:.0} |\n",
                result.backend,
                result.dimension,
                result.dataset_size,
                result.batch_size,
                result.latency_ms.mean,
                result.latency_ms.p95,
                result.throughput.ops_per_sec
            ));
        }
        
        report
    }
    
    pub fn save_json(results: &BenchmarkResults, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(results)?;
        std::fs::write(path, json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_benchmark_runner() {
        let config = BenchmarkConfig {
            vector_dimensions: vec![128],
            dataset_sizes: vec![100],
            batch_sizes: vec![10],
            k_values: vec![10],
            iterations: 10,
            warmup_iterations: 2,
            enable_gpu: false,
            enable_simd: true,
            enable_parallel: true,
        };
        
        let mut runner = BenchmarkRunner::new(config).await;
        let results = runner.run_all().await;
        
        assert!(!results.tests.is_empty());
        assert!(results.summary.total_tests > 0);
    }
}