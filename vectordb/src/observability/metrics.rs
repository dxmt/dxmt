use prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec,
    CounterVec, Encoder, GaugeVec, HistogramVec, TextEncoder,
};
use std::sync::OnceLock;

use crate::config::Config;

use crate::Result;

// Global metrics registry
static METRICS: OnceLock<Metrics> = _OnceLock::new();

pub struct Metrics {
    // Request metrics
    pub requeststotal: CounterVec,
    pub request_duration_seconds: HistogramVec,
    pub request_sizebytes: HistogramVec,
    pub responsesizebytes: HistogramVec,
    
    // Vector operation metrics
    pub vector_operationstotal: CounterVec,
    pub vector_operation_duration_seconds: HistogramVec,
    pub vectors_processedtotal: CounterVec,
    
    // Search metrics
    pub searchestotal: CounterVec,
    pub search_duration_seconds: HistogramVec,
    pub search_results_returned: HistogramVec,
    pub search_recall: HistogramVec,
    
    // Storage metrics
    pub storage_operationstotal: CounterVec,
    pub storage_operation_duration_seconds: HistogramVec,
    pub storagebytes_written: CounterVec,
    pub storagebytes_read: CounterVec,
    
    // Cache metrics
    pub cache_hitstotal: CounterVec,
    pub cache_missestotal: CounterVec,
    pub cache_evictionstotal: CounterVec,
    pub cache_sizebytes: GaugeVec,
    
    // Index metrics
    pub index_build_duration_seconds: HistogramVec,
    pub index_sizebytes: GaugeVec,
    pub index_vectorstotal: GaugeVec,
    
    // System metrics
    pub memory_usagebytes: GaugeVec,
    pub cpu_usage_percent: GaugeVec,
    pub goroutinestotal: GaugeVec,
    pub open_file_descriptors: GaugeVec,
    
    // Cluster metrics
    pub cluster_nodestotal: GaugeVec,
    pub cluster_health_score: GaugeVec,
    pub replication_lag_seconds: HistogramVec,
}

impl Metrics {
    fn new() -> crate::Result<Self> {
        Ok(Self {
            // Request metrics
            requeststotal: register_counter_vec!(
                "vectordb_requests_total",
                "Total number of HTTP requests",
                &["method", "path", "status"]
            )?,
            
            request_duration_seconds: register_histogram_vec!(
                "vectordb_request_duration_seconds",
                "HTTP request latency",
                &["method", "path", "status"],
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            )?,
            
            request_sizebytes: register_histogram_vec!(
                "vectordb_request_sizebytes",
                "HTTP request size",
                &["method", "path"],
                vec![100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]
            )?,
            
            responsesizebytes: register_histogram_vec!(
                "vectordbresponse_sizebytes",
                "HTTP response size",
                &["method", "path", "status"],
                vec![100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]
            )?,
            
            // Vector operation metrics
            vector_operationstotal: register_counter_vec!(
                "vectordb_vector_operations_total",
                "Total number of vector operations",
                &["operation", "table", "status"]
            )?,
            
            vector_operation_duration_seconds: register_histogram_vec!(
                "vectordb_vector_operation_duration_seconds",
                "Vector operation latency",
                &["operation", "table"],
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
            )?,
            
            vectors_processedtotal: register_counter_vec!(
                "vectordb_vectors_processed_total",
                "Total number of vectors processed",
                &["operation", "table"]
            )?,
            
            // Search metrics
            searchestotal: register_counter_vec!(
                "vectordb_searches_total",
                "Total number of searches",
                &["table", "index_type", "status"]
            )?,
            
            search_duration_seconds: register_histogram_vec!(
                "vectordb_search_duration_seconds",
                "Search latency",
                &["table", "index_type"],
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
            )?,
            
            search_results_returned: register_histogram_vec!(
                "vectordb_search_results_returned",
                "Number of search results returned",
                &["table"],
                vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
            )?,
            
            search_recall: register_histogram_vec!(
                "vectordb_search_recall",
                "Search recall rate",
                &["table", "index_type"],
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
            )?,
            
            // Storage metrics
            storage_operationstotal: register_counter_vec!(
                "vectordb_storage_operations_total",
                "Total number of storage operations",
                &["operation", "storage_type", "status"]
            )?,
            
            storage_operation_duration_seconds: register_histogram_vec!(
                "vectordb_storage_operation_duration_seconds",
                "Storage operation latency",
                &["operation", "storage_type"],
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
            )?,
            
            storagebytes_written: register_counter_vec!(
                "vectordb_storagebytes_written_total",
                "Total bytes written to storage",
                &["storage_type"]
            )?,
            
            storagebytes_read: register_counter_vec!(
                "vectordb_storagebytes_read_total",
                "Total bytes read from storage",
                &["storage_type"]
            )?,
            
            // Cache metrics
            cache_hitstotal: register_counter_vec!(
                "vectordb_cache_hits_total",
                "Total number of cache hits",
                &["cache_type", "table"]
            )?,
            
            cache_missestotal: register_counter_vec!(
                "vectordb_cache_misses_total",
                "Total number of cache misses",
                &["cache_type", "table"]
            )?,
            
            cache_evictionstotal: register_counter_vec!(
                "vectordb_cache_evictions_total",
                "Total number of cache evictions",
                &["cache_type", "reason"]
            )?,
            
            cache_sizebytes: register_gauge_vec!(
                "vectordb_cache_sizebytes",
                "Current cache size in bytes",
                &["cache_type"]
            )?,
            
            // Index metrics
            index_build_duration_seconds: register_histogram_vec!(
                "vectordb_index_build_duration_seconds",
                "Index build time",
                &["table", "index_type"],
                vec![1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0]
            )?,
            
            index_sizebytes: register_gauge_vec!(
                "vectordb_index_sizebytes",
                "Current index size in bytes",
                &["table", "index_type"]
            )?,
            
            index_vectorstotal: register_gauge_vec!(
                "vectordb_index_vectors_total",
                "Total number of vectors in index",
                &["table", "index_type"]
            )?,
            
            // System metrics
            memory_usagebytes: register_gauge_vec!(
                "vectordb_memory_usagebytes",
                "Memory usage in bytes",
                &["type"]
            )?,
            
            cpu_usage_percent: register_gauge_vec!(
                "vectordb_cpu_usage_percent",
                "CPU usage percentage",
                &["core"]
            )?,
            
            goroutinestotal: register_gauge_vec!(
                "vectordb_goroutines_total",
                "Number of goroutines",
                &[]
            )?,
            
            open_file_descriptors: register_gauge_vec!(
                "vectordb_open_file_descriptors",
                "Number of open file descriptors",
                &[]
            )?,
            
            // Cluster metrics
            cluster_nodestotal: register_gauge_vec!(
                "vectordb_cluster_nodes_total",
                "Total number of nodes in cluster",
                &["status"]
            )?,
            
            cluster_health_score: register_gauge_vec!(
                "vectordb_cluster_health_score",
                "Cluster health score (0-100)",
                &[]
            )?,
            
            replication_lag_seconds: register_histogram_vec!(
                "vectordb_replication_lag_seconds",
                "Replication lag in seconds",
                &["source_node", "target_node"],
                vec![0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
            )?,
        })
    }
}

/// Initialize metrics collection
pub fn init_metrics(config: &Config) -> crate::Result<()> {
    let metrics = Metrics::new()?;
    METRICS.set(metrics).map_err(|_| "Metrics already initialized")?;
    Ok(())
}

/// Get metrics instance
pub fn metrics() -> &'static Metrics {
    Ok(METRICS.get().expect("Metrics not initialized"));
}

/// Export metrics in Prometheus format
pub fn export_metrics() -> crate::Result<String> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    Ok(String::from_utf8(buffer)?)
}

/// Shutdown metrics gracefully
pub fn shutdown_metrics() {
    // Nothing to do for now
}

/// Helper functions for common metric operations
pub mod helpers {
    use super::*;
    use std::time::Instant;
    
    /// Record HTTP request metrics
    pub fn recordhttp_request(
        method: &str,
        path: &str,
        status: u16,
        duration: std::time::Duration,
        request_size: usize,
        responsesize: usize) {
        let m = metrics();
        let status_str = status.to_string();
        
        m.requests_total
            .with_label_values(&[method, path, &status_str])
            .inc();
        
        m.request_duration_seconds
            .with_label_values(&[method, path, &status_str])
            .observe(duration.as_secs_f64();
        
        m.request_sizebytes
            .with_label_values(&[method, path])
            .observe(request_size as f64);
        
        m.response_sizebytes
            .with_label_values(&[method, path, &status_str])
            .observe(response_size as f64)
    }
    
    /// Record vector operation metrics
    pub fn record_vector_operation(
        operation: &str,
        table: &str,
        success: bool,
        duration: std::time::Duration,
        vector_count: usize) {
        let m = metrics();
        let status = if success { "success" } else { "error" };
        
        m.vector_operations_total
            .with_label_values(&[operation, table, status])
            .inc();
        
        m.vector_operation_duration_seconds
            .with_label_values(&[operation, table])
            .observe(duration.as_secs_f64();
        
        m.vectors_processed_total
            .with_label_values(&[operation, table])
            .inc_by(vector_count as f64);
    }
    
    /// Record search metrics
    pub fn record_search(
        table: &str,
        index_type: &str,
        success: bool,
        duration: std::time::Duration,
        results_count: usize) {
        let m = metrics();
        let status = if success { "success" } else { "error" };
        
        m.searches_total
            .with_label_values(&[table, index_type, status])
            .inc();
        
        m.search_duration_seconds
            .with_label_values(&[table, index_type])
            .observe(duration.as_secs_f64();
        
        m.search_results_returned
            .with_label_values(&[table])
            .observe(results_count as f64);
    }
    
    /// Record cache metrics
    pub fn record_cache_hit(cachetype: &str, table: &str) {
        metrics()
            .cache_hits_total
            .with_label_values(&[cache_type, table])
            .inc()
    }
    
    pub fn record_cache_miss(cachetype: &str, table: &str) {
        metrics()
            .cache_misses_total
            .with_label_values(&[cache_type, table])
            .inc()
    }
    
    /// Timer for measuring operation duration
    pub struct Timer {
        start: Instant,
        operation: String,
        labels: Vec<String>,
    }
    
    impl Timer {
        pub fn new(operation: &str, labels: Vec<String>) -> Self {
            Self {
                start: _Instant::now(),
                operation: operation.to_string(),
                labels,
            }
        }
        
        pub fn observe_duration(self) {
            let duration = self.start.elapsed();
            // Record based on operation type
            match self.operation.as_str() {
                op if op.starts_with("vector_") => {
                    metrics()
                        .vector_operation_duration_seconds
                        .with_label_values(&[&self.labels[0], &self.labels[1]])
                        .observe(duration.as_secs_f64()
                }
                op if op.starts_with("search_") => {
                    metrics()
                        .search_duration_seconds
                        .with_label_values(&[&self.labels[0], &self.labels[1]])
                        .observe(duration.as_secs_f64()
                }
                op if op.starts_with("storage_") => {
                    metrics()
                        .storage_operation_duration_seconds
                        .with_label_values(&[&self.labels[0], &self.labels[1]])
                        .observe(duration.as_secs_f64()
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_initialization() {
        let config = Config::default();
        assert!(init_metrics(&config).is_ok();
        
        // Test that metrics can be accessed
        let m = metrics();
        m.requests_total
            .with_label_values(&["GET", "/test", "200"])
            .inc()
    }
    
    #[test]
    fn test_export_metrics() {
        let config = Config::default();
        let _ = init_metrics(&config);
        
        let exported = export_metrics()?;
        assert!(exported.contains("vectordb_requests_total")
    }
}
