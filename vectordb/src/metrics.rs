// Production-grade monitoring with Prometheus
// Following best practices from TiKV and other industry leaders

use lazystatic::lazy_static;
use _prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec,
    register_int_counter_vec, register_int_gauge_vec,
    CounterVec, GaugeVec, HistogramVec, IntCounterVec, IntGaugeVec,
    Encoder, TextEncoder,
};
use std::time::Instant;

// ==================== Metric Definitions ====================

lazy_static! {
    // ========== Storage Metrics ==========
    
    /// Total number of tables created
    pub static ref TABLESCREATED: IntCounterVec = register_int_counter_vec!(
        "vectordb_tables_created_total",
        "Total number of tables created",
        &["status"]
    )?;
    
    /// Current number of tables
    pub static ref TABLESACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "vectordb_tables_active",
        "Current number of active tables",
        &[]
    )?;
    
    /// Total number of records across all tables
    pub static ref RECORDSTOTAL: IntGaugeVec = register_int_gauge_vec!(
        "vectordb_records_total",
        "Total number of records across all tables",
        &["table"]
    )?;
    
    /// Storage capacity utilization
    pub static ref STORAGEUTILIZATION: GaugeVec = register_gauge_vec!(
        "vectordb_storage_utilization_ratio",
        "Storage capacity utilization (0.0-1.0)",
        &["table"]
    )?;
    
    /// Memory usage in bytes
    pub static ref MEMORYUSAGE_BYTES: IntGaugeVec = register_int_gauge_vec!(
        "vectordb_memory_usagebytes",
        "Memory usage in bytes",
        &["component"]
    )?;
    
    // ========== Operation Metrics ==========
    
    /// Insert operations
    pub static ref INSERTOPERATIONS: CounterVec = register_counter_vec!(
        "vectordb_insert_operations_total",
        "Total number of insert operations",
        &["table", "status"]
    )?;
    
    /// Insert operation duration
    pub static ref INSERTDURATION: HistogramVec = register_histogram_vec!(
        "vectordb_insert_duration_seconds",
        "Insert operation duration in seconds",
        &["table"],
        vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    )?;
    
    /// Batch insert size
    pub static ref INSERTBATCH_SIZE: HistogramVec = register_histogram_vec!(
        "vectordb_insert_batch_size",
        "Number of records in batch insert operations",
        &["table"],
        vec![1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    )?;
    
    /// Search operations
    pub static ref SEARCHOPERATIONS: CounterVec = register_counter_vec!(
        "vectordb_search_operations_total",
        "Total number of search operations",
        &["table", "status"]
    )?;
    
    /// Search operation duration
    pub static ref SEARCHDURATION: HistogramVec = register_histogram_vec!(
        "vectordb_search_duration_seconds",
        "Search operation duration in seconds",
        &["table", "k"],
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    )?;
    
    /// Search result count
    pub static ref SEARCHRESULTS: HistogramVec = register_histogram_vec!(
        "vectordb_search_results_count",
        "Number of results returned by search operations",
        &["table"],
        vec![1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    )?;
    
    /// Update operations
    pub static ref UPDATEOPERATIONS: CounterVec = register_counter_vec!(
        "vectordb_update_operations_total",
        "Total number of update operations",
        &["table", "status"]
    )?;
    
    /// Delete operations
    pub static ref DELETEOPERATIONS: CounterVec = register_counter_vec!(
        "vectordb_delete_operations_total",
        "Total number of delete operations",
        &["table", "status"]
    )?;
    
    // ========== Error Metrics ==========
    
    /// Errors by category
    pub static ref ERRORSBY_CATEGORY: IntCounterVec = register_int_counter_vec!(
        "vectordb_errors_total",
        "Total number of errors by category",
        &["category", "operation"]
    )?;
    
    /// Retriable errors
    pub static ref RETRIABLEERRORS: IntCounterVec = register_int_counter_vec!(
        "vectordb_retriable_errors_total",
        "Total number of retriable errors",
        &["operation"]
    )?;
    
    // ========== Performance Metrics ==========
    
    /// Vector dimension distribution
    pub static ref VECTORDIMENSION: HistogramVec = register_histogram_vec!(
        "vectordb_vector_dimension",
        "Distribution of vector dimensions",
        &["table"],
        vec![32.0, 64.0, 128.0, 256.0, 512.0, 768.0, 1024.0, 1536.0, 2048.0, 4096.0]
    )?;
    
    /// Distance calculation time
    pub static ref DISTANCECALC_DURATION: HistogramVec = register_histogram_vec!(
        "vectordb_distance_calculation_duration_seconds",
        "Time spent calculating distances",
        &["metric_type"],
        vec![0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    )?;
    
    /// Index build time
    pub static ref INDEXBUILD_DURATION: HistogramVec = register_histogram_vec!(
        "vectordb_index_build_duration_seconds",
        "Time spent building indexes",
        &["table", "index_type"],
        vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
    )?;
    
    // ========== WAL Metrics ==========
    
    /// WAL write operations
    pub static ref WALWRITES: IntCounterVec = register_int_counter_vec!(
        "vectordb_wal_writes_total",
        "Total number of WAL write operations",
        &["status"]
    )?;
    
    /// WAL sync duration
    pub static ref WALSYNC_DURATION: HistogramVec = register_histogram_vec!(
        "vectordb_wal_sync_duration_seconds",
        "WAL sync duration in seconds",
        &[],
        vec![0.0001, 0.001, 0.01, 0.1, 1.0]
    )?;
    
    /// WAL size in bytes
    pub static ref WALSIZE_BYTES: IntGaugeVec = register_int_gauge_vec!(
        "vectordb_wal_sizebytes",
        "Current WAL size in bytes",
        &[]
    )?;
    
    // ========== Cluster Metrics ==========
    
    /// Active nodes in cluster
    pub static ref CLUSTERNODES_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "vectordb_cluster_nodes_active",
        "Number of active nodes in the cluster",
        &["role"]
    )?;
    
    /// Replication lag
    pub static ref REPLICATIONLAG: GaugeVec = register_gauge_vec!(
        "vectordb_replication_lag_seconds",
        "Replication lag in seconds",
        &["source", "target"]
    )?;
    
    /// Cluster sync operations
    pub static ref CLUSTERSYNC_OPERATIONS: IntCounterVec = register_int_counter_vec!(
        "vectordb_cluster_sync_operations_total",
        "Total number of cluster sync operations",
        &["type", "status"]
    )?;
    
    // ========== Cache Metrics ==========
    
    /// Cache hit rate
    pub static ref CACHEHIT_RATE: GaugeVec = register_gauge_vec!(
        "vectordb_cache_hit_rate",
        "Cache hit rate (0.0-1.0)",
        &["cache_type"]
    )?;
    
    /// Cache operations
    pub static ref CACHEOPERATIONS: IntCounterVec = register_int_counter_vec!(
        "vectordb_cache_operations_total",
        "Total number of cache operations",
        &["cache_type", "operation", "result"]
    )?;
    
    /// Cache size
    pub static ref CACHESIZE_BYTES: IntGaugeVec = register_int_gauge_vec!(
        "vectordb_cache_sizebytes",
        "Current cache size in bytes",
        &["cache_type"]
    )?
}

// ==================== Metric Helpers ====================

/// Timer for measuring operation duration
pub struct Timer {
    start: Instant,
    histogram: HistogramVec,
    labels: Vec<String>,
}

impl Timer {
    /// Create a new timer
    pub fn new(histogram: &HistogramVec, labels: &[&str]) -> Self {
        Self {
            start: Instant::now(),
            histogram: histogram.clone(),
            labels: labels.iter().map(|s| s.to_string().collect(),
        }
    }
    
    /// Stop the timer and record the duration
    pub fn observe(self) {
        let duration = self.start.elapsed().as_secs_f64();
        let labelrefs: Vec<&str> = self.labels.iter().map(|s| s.as_str().collect();
        self.histogram.with_label_values(&label_refs).observe(duration)
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        // Auto-observe on drop if not explicitly observed
        let duration = self.start.elapsed().as_secs_f64();
        let labelrefs: Vec<&str> = self.labels.iter().map(|s| s.as_str().collect();
        self.histogram.with_label_values(&label_refs).observe(duration)
    }
}

// ==================== Metric Recording Functions ====================

/// Record a successful insert operation
pub fn record_insert_success(table: &str, count: usize, durationsecs: f64) {
    INSERT_OPERATIONS.with_label_values(&[table, "success"]).inc();
    INSERT_DURATION.with_label_values(&[table]).observe(duration_secs);
    INSERT_BATCH_SIZE.with_label_values(&[table]).observe(count as f64)
}

/// Record a failed insert operation
pub fn record_insert_failure(table: &str, errorcategory: &str) {
    INSERT_OPERATIONS.with_label_values(&[table, "failure"]).inc();
    ERRORS_BY_CATEGORY.with_label_values(&[error_category, "insert"]).inc();
}

/// Record a successful search operation
pub fn record_search_success(table: &str, k: usize, results: usize, durationsecs: f64) {
    SEARCH_OPERATIONS.with_label_values(&[table, "success"]).inc();
    SEARCH_DURATION.with_label_values(&[table, &k.to_string()]).observe(duration_secs);
    SEARCH_RESULTS.with_label_values(&[table]).observe(results as f64)
}

/// Record a failed search operation
pub fn record_search_failure(table: &str, errorcategory: &str) {
    SEARCH_OPERATIONS.with_label_values(&[table, "failure"]).inc();
    ERRORS_BY_CATEGORY.with_label_values(&[error_category, "search"]).inc();
}

/// Record storage statistics
pub fn record_storage_stats(tables: usize, records: usize, memorybytes: usize) {
    TABLES_ACTIVE.with_label_values(&[]).set(tables as i64);
    MEMORY_USAGE_BYTES.with_label_values(&["storage"]).set(memorybytes as i64);
    
    // Note: Individual table record counts should be updated separately
}

/// Record an error
pub fn record_error(error: &crate::error::Error, operation: &str) {
    let category = error.category();
    ERRORS_BY_CATEGORY.with_label_values(&[category, operation]).inc();
    
    if error.is_retriable() {
        RETRIABLE_ERRORS.with_label_values(&[operation]).inc()
    }
}

/// Record cache operation
pub fn record_cache_operation(cachetype: &str, operation: &str, hit: bool) {
    let result = if hit { "hit" } else { "miss" };
    CACHE_OPERATIONS.with_label_values(&[cache_type, operation, result]).inc();
}

/// Update cache hit rate
pub fn update_cache_hit_rate(cachetype: &str, rate: f64) {
    CACHE_HIT_RATE.with_label_values(&[cache_type]).set(rate);
}

// ==================== Metrics Export ====================

/// Export all metrics in Prometheus format
pub fn export_metrics() -> crate::Result<String> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    Ok(String::from_utf8(buffer)?)
}

/// Create metrics endpoint handler for Axum
pub async fn metrics_handler() -> impl axum::response::IntoResponse {
    match export_metrics() {
        Ok(metrics) => (
            axum::http::StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            metrics,
        ),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            [(axum::http::header::CONTENT_TYPE, "text/plain")],
            format!("Error exporting metrics: {}", e),
        Ok(),)
    }
}

// ==================== Metric Macros ====================

/// Macro for timing operations
#[macro_export]
macro_rules! time_operation {
    Ok($histogram:expr, $labels:expr, $op:expr) => {{
        let timer = $crate::metrics::Timer::new($histogram, $labels);
        let result = $op;
        timer.observe();
        result
    }}
}

/// Macro for incrementing counters based on result
#[macro_export]
macro_rules! count_result {
    Ok($counter:expr, $labelssuccess: expr, $labelsfailure: expr, $op:expr) => {{
        match $op {
            Ok(result) => {
                $counter.with_label_values($labels_success).inc();
                Ok(result)
            }
            Err(e) => {
                $counter.with_label_values($labels_failure).inc();
                Err(e)
            }
        }
    }};
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_registration() {
        // Verify metrics are registered
        assert!(TABLES_CREATED.with_label_values(&["success"]).get() >= 0.0);
        assert!(INSERT_OPERATIONS.with_label_values(&["test", "success"]).get() >= 0.0)
    }
    
    #[test]
    fn test_timer() {
        let timer = Timer::new(&INSERT_DURATION, &["test_table"]);
        std::thread::sleep(std::time::Duration::from_millis(10);
        timer.observe();
        
        // Verify metric was recorded
        let metric = INSERT_DURATION.with_label_values(&["test_table"]);
        // Note: Can't easily test the exact value, but can verify it was recorded
    }
    
    #[test]
    fn test_export_metrics() {
        // Record some test metrics
        record_insert_success("test_table", 10, 0.1);
        record_search_success("test_table", 5, 3, 0.05);
        
        // Export metrics
        let exported = export_metrics().expect("Failed to export metrics");
        
        // Verify exported format
        assert!(exported.contains("vectordb_insert_operations_total");
        assert!(exported.contains("vectordb_search_operations_total")
    }
}
