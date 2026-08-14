//! Performance profiling utilities for Phase 3 index management components
//!
//! This module provides tools for measuring and analyzing performance of:
//! - Index creation and initialization
//! - Data migration operations
//! - Index switching overhead
//! - Query performance across different index types
//! - Memory usage and resource consumption

use std::collections::HashMap;
use std::time::Instant;

#[cfg(test)]
use std::time::Duration;
use std::sync::Arc;
use parking_lot::RwLock;

#[derive(Clone, Debug)]
pub struct ProfileSnapshot {
    pub name: String,
    pub timestamp: Instant,
    pub duration_ms: f64,
    pub memory_mb: f64,
    pub cpu_percent: f64,
    pub throughput_ops_per_sec: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct PerformanceProfile {
    pub operation_name: String,
    pub snapshots: Vec<ProfileSnapshot>,
    pub min_duration_ms: f64,
    pub max_duration_ms: f64,
    pub avg_duration_ms: f64,
    pub p50_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub p99_duration_ms: f64,
}

impl PerformanceProfile {
    /// Calculate percentile from sorted durations
    fn calculate_percentile(durations: &[f64], percentile: f64) -> f64 {
        if durations.is_empty() {
            return 0.0;
        }
        let index = ((percentile / 100.0) * (durations.len() as f64 - 1.0)).ceil() as usize;
        durations[index.min(durations.len() - 1)]
    }

    /// Generate summary statistics from snapshots
    pub fn summarize(&self) -> String {
        format!(
            "{}:\n  Min: {:.2}ms | Max: {:.2}ms | Avg: {:.2}ms\n  P50: {:.2}ms | P95: {:.2}ms | P99: {:.2}ms",
            self.operation_name,
            self.min_duration_ms,
            self.max_duration_ms,
            self.avg_duration_ms,
            self.p50_duration_ms,
            self.p95_duration_ms,
            self.p99_duration_ms
        )
    }
}

/// Profiler for measuring operation performance
pub struct OperationProfiler {
    profiles: Arc<RwLock<HashMap<String, PerformanceProfile>>>,
    active_operations: Arc<RwLock<HashMap<String, Instant>>>,
}

impl OperationProfiler {
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
            active_operations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start profiling an operation
    pub fn start_operation(&self, name: &str) -> OperationProfileHandle {
        let mut active = self.active_operations.write();
        active.insert(name.to_string(), Instant::now());

        OperationProfileHandle {
            name: name.to_string(),
            start: Instant::now(),
            profiler: self.profiles.clone(),
            operations: self.active_operations.clone(),
        }
    }

    /// Get profile for an operation
    pub fn get_profile(&self, name: &str) -> Option<PerformanceProfile> {
        let profiles = self.profiles.read();
        profiles.get(name).cloned()
    }

    /// Get all profiles
    pub fn get_all_profiles(&self) -> Vec<PerformanceProfile> {
        let profiles = self.profiles.read();
        profiles.values().cloned().collect()
    }

    /// Clear all profiles
    pub fn clear_profiles(&self) {
        let mut profiles = self.profiles.write();
        profiles.clear();
    }

    /// Generate full profiling report
    pub fn generate_report(&self) -> String {
        let profiles = self.profiles.read();
        if profiles.is_empty() {
            return "No profiling data available".to_string();
        }

        let mut report = String::from("=== Performance Profiling Report ===\n\n");

        for profile in profiles.values() {
            report.push_str(&profile.summarize());
            report.push('\n');
        }

        report
    }
}

impl Default for OperationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for tracking a profiled operation
pub struct OperationProfileHandle {
    name: String,
    start: Instant,
    profiler: Arc<RwLock<HashMap<String, PerformanceProfile>>>,
    operations: Arc<RwLock<HashMap<String, Instant>>>,
}

impl OperationProfileHandle {
    /// Record successful operation completion
    pub fn finish_success(self, throughput_ops: usize) {
        let duration_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let throughput = if duration_ms > 0.0 {
            (throughput_ops as f64) / (duration_ms / 1000.0)
        } else {
            0.0
        };

        let snapshot = ProfileSnapshot {
            name: self.name.clone(),
            timestamp: Instant::now(),
            duration_ms,
            memory_mb: 0.0, // requires platform memory API (e.g. /proc/self/statm on Linux)
            cpu_percent: 0.0, // requires platform CPU API (e.g. /proc/self/stat on Linux)
            throughput_ops_per_sec: throughput,
            metadata: HashMap::new(),
        };

        self.record_snapshot(snapshot);

        let mut active = self.operations.write();
        active.remove(&self.name);
    }

    /// Record operation failure
    pub fn finish_error(self, error: &str) {
        let duration_ms = self.start.elapsed().as_secs_f64() * 1000.0;

        let mut metadata = HashMap::new();
        metadata.insert("error".to_string(), error.to_string());

        let snapshot = ProfileSnapshot {
            name: self.name.clone(),
            timestamp: Instant::now(),
            duration_ms,
            memory_mb: 0.0,
            cpu_percent: 0.0,
            throughput_ops_per_sec: 0.0,
            metadata,
        };

        self.record_snapshot(snapshot);

        let mut active = self.operations.write();
        active.remove(&self.name);
    }

    fn record_snapshot(&self, snapshot: ProfileSnapshot) {
        let mut profiles = self.profiler.write();

        profiles
            .entry(snapshot.name.clone())
            .or_insert_with(|| {
                PerformanceProfile {
                    operation_name: snapshot.name.clone(),
                    snapshots: vec![snapshot.clone()],
                    min_duration_ms: snapshot.duration_ms,
                    max_duration_ms: snapshot.duration_ms,
                    avg_duration_ms: snapshot.duration_ms,
                    p50_duration_ms: snapshot.duration_ms,
                    p95_duration_ms: snapshot.duration_ms,
                    p99_duration_ms: snapshot.duration_ms,
                }
            });

        let profile = profiles.get_mut(&snapshot.name).unwrap();
        profile.snapshots.push(snapshot.clone());

        // Recalculate statistics
        let durations: Vec<f64> = profile.snapshots.iter().map(|s| s.duration_ms).collect();
        let mut sorted_durations = durations.clone();
        sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        profile.min_duration_ms = *sorted_durations.first().unwrap_or(&0.0);
        profile.max_duration_ms = *sorted_durations.last().unwrap_or(&0.0);
        profile.avg_duration_ms = durations.iter().sum::<f64>() / durations.len() as f64;
        profile.p50_duration_ms = PerformanceProfile::calculate_percentile(&sorted_durations, 50.0);
        profile.p95_duration_ms = PerformanceProfile::calculate_percentile(&sorted_durations, 95.0);
        profile.p99_duration_ms = PerformanceProfile::calculate_percentile(&sorted_durations, 99.0);
    }
}

/// Benchmark configuration for comparative testing
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub benchmark_iterations: usize,
    pub record_counts: Vec<usize>,
    pub vector_dimensions: Vec<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            benchmark_iterations: 10,
            record_counts: vec![1000, 10000, 100000],
            vector_dimensions: vec![256, 512, 1024],
        }
    }
}

/// Results from a benchmark run
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    pub operation_name: String,
    pub record_count: usize,
    pub vector_dimension: usize,
    pub iterations: usize,
    pub total_duration_ms: f64,
    pub avg_duration_per_iteration_ms: f64,
    pub throughput_records_per_sec: f64,
}

impl BenchmarkResult {
    pub fn format(&self) -> String {
        format!(
            "{} ({}x{}): {:.2}ms/iter, {:.0} records/sec",
            self.operation_name,
            self.record_count,
            self.vector_dimension,
            self.avg_duration_per_iteration_ms,
            self.throughput_records_per_sec
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = OperationProfiler::new();
        assert!(profiler.get_all_profiles().is_empty());
    }

    #[test]
    fn test_operation_profiling() {
        let profiler = OperationProfiler::new();

        {
            let handle = profiler.start_operation("test_op");
            std::thread::sleep(Duration::from_millis(10));
            handle.finish_success(100);
        }

        let profile = profiler.get_profile("test_op");
        assert!(profile.is_some());
        let profile = profile.unwrap();
        assert!(profile.avg_duration_ms >= 10.0);
    }

    #[test]
    fn test_percentile_calculation() {
        let durations = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p50 = PerformanceProfile::calculate_percentile(&durations, 50.0);
        let p95 = PerformanceProfile::calculate_percentile(&durations, 95.0);

        assert!(p50 > 0.0);
        assert!(p95 >= p50);
    }

    #[test]
    fn test_error_handling() {
        let profiler = OperationProfiler::new();

        {
            let handle = profiler.start_operation("failing_op");
            handle.finish_error("test error");
        }

        let profile = profiler.get_profile("failing_op");
        assert!(profile.is_some());
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 3);
        assert_eq!(config.benchmark_iterations, 10);
        assert!(!config.record_counts.is_empty());
    }
}
