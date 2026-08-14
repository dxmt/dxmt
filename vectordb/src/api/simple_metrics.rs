// Metrics and monitoring for Simple Mode (Collections API)
// Provides insights into collection usage and performance

use prometheus::{
    register_counter_vec, register_histogram_vec, register_gauge_vec,
    CounterVec, HistogramVec, GaugeVec,
};
use lazy_static::lazy_static;
use std::time::Instant;

lazy_static! {
    /// Counter for collection operations
    pub static ref COLLECTION_OPS_COUNTER: CounterVec = register_counter_vec!(
        "simple_mode_collection_operations_total",
        "Total number of collection operations in simple mode",
        &["operation", "collection", "status"]
    ).expect("Failed to register collection ops counter");

    /// Histogram for operation latencies
    pub static ref OPERATION_LATENCY_HISTOGRAM: HistogramVec = register_histogram_vec!(
        "simple_mode_operation_duration_seconds",
        "Duration of operations in simple mode",
        &["operation", "collection"],
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    ).expect("Failed to register operation latency histogram");

    /// Gauge for collection sizes
    pub static ref COLLECTION_SIZE_GAUGE: GaugeVec = register_gauge_vec!(
        "simple_mode_collection_size",
        "Current size of collections in simple mode",
        &["collection", "metric_type"]
    ).expect("Failed to register collection size gauge");

    /// Counter for search queries
    pub static ref SEARCH_QUERY_COUNTER: CounterVec = register_counter_vec!(
        "simple_mode_search_queries_total",
        "Total number of search queries in simple mode",
        &["collection", "strategy"]
    ).expect("Failed to register search query counter");

    /// Histogram for search result counts
    pub static ref SEARCH_RESULTS_HISTOGRAM: HistogramVec = register_histogram_vec!(
        "simple_mode_search_results_count",
        "Number of results returned per search query",
        &["collection"],
        vec![0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    ).expect("Failed to register search results histogram");

    /// Counter for auto-embedding operations
    pub static ref AUTO_EMBED_COUNTER: CounterVec = register_counter_vec!(
        "simple_mode_auto_embeddings_total",
        "Total number of auto-embedding operations",
        &["type", "status"]
    ).expect("Failed to register auto-embed counter");

    /// Gauge for memory usage
    pub static ref MEMORY_USAGE_GAUGE: GaugeVec = register_gauge_vec!(
        "simple_mode_memory_usage_bytes",
        "Memory usage in simple mode",
        &["type"]
    ).expect("Failed to register memory usage gauge");

    /// Counter for strategy selections
    pub static ref STRATEGY_SELECTION_COUNTER: CounterVec = register_counter_vec!(
        "simple_mode_strategy_selections_total",
        "Auto-selected search strategies",
        &["strategy", "reason"]
    ).expect("Failed to register strategy selection counter");
}

/// Timer guard for automatic metric recording
pub struct MetricTimer {
    start: Instant,
    operation: String,
    collection: String,
}

impl MetricTimer {
    /// Create a new metric timer
    pub fn new(operation: &str, collection: &str) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.to_string(),
            collection: collection.to_string(),
        }
    }
}

impl Drop for MetricTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_secs_f64();
        OPERATION_LATENCY_HISTOGRAM
            .with_label_values(&[&self.operation, &self.collection])
            .observe(duration);
    }
}

/// Record a collection operation
pub fn record_collection_operation(operation: &str, collection: &str, success: bool) {
    let status = if success { "success" } else { "failure" };
    COLLECTION_OPS_COUNTER
        .with_label_values(&[operation, collection, status])
        .inc();
}

/// Update collection size metrics
pub fn update_collection_size(collection: &str, documents: usize, embeddings: usize, memory_bytes: usize) {
    COLLECTION_SIZE_GAUGE
        .with_label_values(&[collection, "documents"])
        .set(documents as f64);

    COLLECTION_SIZE_GAUGE
        .with_label_values(&[collection, "embeddings"])
        .set(embeddings as f64);

    COLLECTION_SIZE_GAUGE
        .with_label_values(&[collection, "memory_bytes"])
        .set(memory_bytes as f64);
}

/// Record a search query
pub fn record_search_query(collection: &str, strategy: &str, num_results: usize) {
    SEARCH_QUERY_COUNTER
        .with_label_values(&[collection, strategy])
        .inc();

    SEARCH_RESULTS_HISTOGRAM
        .with_label_values(&[collection])
        .observe(num_results as f64);
}

/// Record auto-embedding operation
pub fn record_auto_embedding(embed_type: &str, count: usize, success: bool) {
    let status = if success { "success" } else { "failure" };
    AUTO_EMBED_COUNTER
        .with_label_values(&[embed_type, status])
        .inc_by(count as f64);
}

/// Update memory usage metrics
pub fn update_memory_usage(collections_bytes: usize, cache_bytes: usize, index_bytes: usize) {
    MEMORY_USAGE_GAUGE
        .with_label_values(&["collections"])
        .set(collections_bytes as f64);

    MEMORY_USAGE_GAUGE
        .with_label_values(&["cache"])
        .set(cache_bytes as f64);

    MEMORY_USAGE_GAUGE
        .with_label_values(&["index"])
        .set(index_bytes as f64);
}

/// Record strategy selection
pub fn record_strategy_selection(strategy: &str, reason: &str) {
    STRATEGY_SELECTION_COUNTER
        .with_label_values(&[strategy, reason])
        .inc();
}

/// Get current metrics summary for simple mode
pub fn get_metrics_summary() -> serde_json::Value {
    use prometheus::{Encoder, TextEncoder};

    // Gather Prometheus metrics
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();

    // Parse and summarize key metrics
    let metrics_text = String::from_utf8(buffer).unwrap();

    // Extract simple mode metrics
    let mut summary = serde_json::json!({
        "mode": "simple",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "collections": {},
        "operations": {},
        "memory": {},
        "strategies": {}
    });

    // Parse metrics text to extract values (simplified)
    for line in metrics_text.lines() {
        if line.starts_with("simple_mode_") && !line.starts_with("#") {
            // Parse metric line (simplified parsing)
            if let Some(name_end) = line.find('{') {
                let metric_name = &line[..name_end];
                let value_start = line.rfind(' ').unwrap_or(line.len() - 1) + 1;
                if let Ok(value) = line[value_start..].parse::<f64>() {
                    // Add to summary based on metric type
                    match metric_name {
                        "simple_mode_collection_operations_total" => {
                            summary["operations"]["total"] = serde_json::json!(value);
                        }
                        "simple_mode_search_queries_total" => {
                            summary["operations"]["searches"] = serde_json::json!(value);
                        }
                        "simple_mode_memory_usage_bytes" => {
                            summary["memory"]["bytes"] = serde_json::json!(value);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    summary
}

/// Middleware for automatic metric collection
pub struct MetricsMiddleware;

impl MetricsMiddleware {
    /// Wrap an operation with metrics collection
    pub async fn track<F, T>(
        operation: &str,
        collection: &str,
        f: F,
    ) -> Result<T, Box<dyn std::error::Error>>
    where
        F: std::future::Future<Output = Result<T, Box<dyn std::error::Error>>>,
    {
        let _timer = MetricTimer::new(operation, collection);

        match f.await {
            Ok(result) => {
                record_collection_operation(operation, collection, true);
                Ok(result)
            }
            Err(e) => {
                record_collection_operation(operation, collection, false);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        // Test collection operation recording
        record_collection_operation("create", "test_collection", true);
        record_collection_operation("create", "test_collection", false);

        // Test collection size update
        update_collection_size("test_collection", 100, 100, 1024 * 1024);

        // Test search query recording
        record_search_query("test_collection", "brute_force", 10);

        // Test auto-embedding recording
        record_auto_embedding("batch", 5, true);

        // Test memory usage update
        update_memory_usage(1024 * 1024, 512 * 1024, 256 * 1024);

        // Test strategy selection recording
        record_strategy_selection("HNSW", "medium_dataset");

        // Verify metrics are recorded (basic check)
        let summary = get_metrics_summary();
        assert!(summary.is_object());
        assert_eq!(summary["mode"], "simple");
    }

    #[tokio::test]
    async fn test_metrics_middleware() {
        let result = MetricsMiddleware::track(
            "test_operation",
            "test_collection",
            async { Ok::<_, Box<dyn std::error::Error>>(42) },
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }
}