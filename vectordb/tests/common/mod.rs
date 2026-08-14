/// Common test utilities and markers
///
/// This module provides test categorization macros and shared utilities
/// to make it easy to run specific subsets of tests.

/// Test category markers
pub mod markers {
    /// Marker for fast unit tests (< 100ms)
    pub const FAST: &str = "fast";

    /// Marker for unit tests
    pub const UNIT: &str = "unit";

    /// Marker for integration tests
    pub const INTEGRATION: &str = "integration";

    /// Marker for benchmark tests
    pub const BENCHMARK: &str = "benchmark";

    /// Marker for property-based tests
    pub const PROPERTY: &str = "property";

    /// Marker for slow tests (> 1s)
    pub const SLOW: &str = "slow";
}

/// Assert helpers for floating point comparisons
pub mod assertions {
    /// Assert two floats are approximately equal (within epsilon)
    pub fn assert_float_eq(a: f32, b: f32, epsilon: f32) {
        assert!(
            (a - b).abs() < epsilon,
            "Values not equal: {} vs {} (epsilon: {})",
            a,
            b,
            epsilon
        );
    }

    /// Assert two vectors are approximately equal (within epsilon)
    pub fn assert_vec_eq(a: &[f32], b: &[f32], epsilon: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "Vector lengths don't match: {} vs {}",
            a.len(),
            b.len()
        );

        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (av - bv).abs() < epsilon,
                "Vectors differ at index {}: {} vs {} (epsilon: {})",
                i,
                av,
                bv,
                epsilon
            );
        }
    }

    /// Assert recall is above threshold
    pub fn assert_recall(recall: f64, min_threshold: f64) {
        assert!(
            recall >= min_threshold,
            "Recall {} is below minimum threshold {}",
            recall,
            min_threshold
        );
    }
}

/// Performance measurement helpers
pub mod perf {
    use std::time::{Duration, Instant};

    /// Measure execution time of a function
    pub fn measure<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Measure async execution time
    pub async fn measure_async<F, Fut, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let start = Instant::now();
        let result = f().await;
        let duration = start.elapsed();
        (result, duration)
    }

    /// Calculate QPS from operations and duration
    pub fn calculate_qps(operations: usize, duration: Duration) -> f64 {
        operations as f64 / duration.as_secs_f64()
    }

    /// Format duration in human-readable format
    pub fn format_duration(duration: Duration) -> String {
        let micros = duration.as_micros();
        if micros < 1_000 {
            format!("{}μs", micros)
        } else if micros < 1_000_000 {
            format!("{:.2}ms", micros as f64 / 1_000.0)
        } else {
            format!("{:.2}s", duration.as_secs_f64())
        }
    }
}

/// Statistics helpers
pub mod stats {
    /// Calculate mean of a slice
    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    /// Calculate median of a slice (requires sorted input)
    pub fn median(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = values.len() / 2;

        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    /// Calculate percentile (requires sorted input)
    pub fn percentile(values: &[f64], p: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let idx = ((values.len() - 1) as f64 * p / 100.0).ceil() as usize;
        values[idx.min(values.len() - 1)]
    }

    /// Calculate standard deviation
    pub fn stddev(values: &[f64]) -> f64 {
        let m = mean(values);
        let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_eq() {
        assertions::assert_float_eq(1.0, 1.0001, 0.001);
        assertions::assert_float_eq(0.5, 0.50001, 0.001);
    }

    #[test]
    fn test_vec_eq() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0001, 2.0001, 3.0001];
        assertions::assert_vec_eq(&a, &b, 0.001);
    }

    #[test]
    fn test_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(stats::mean(&values), 3.0);

        let mut values_for_median = values.clone();
        assert_eq!(stats::median(&mut values_for_median), 3.0);

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(stats::percentile(&sorted, 50.0), 3.0);
        assert_eq!(stats::percentile(&sorted, 100.0), 5.0);
    }

    #[test]
    fn test_perf_measure() {
        use std::thread;
        use std::time::Duration;

        let (result, duration) = perf::measure(|| {
            thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_format_duration() {
        use std::time::Duration;

        assert_eq!(perf::format_duration(Duration::from_micros(500)), "500μs");
        assert_eq!(perf::format_duration(Duration::from_micros(1500)), "1.50ms");
        assert_eq!(perf::format_duration(Duration::from_millis(1500)), "1.50s");
    }
}
