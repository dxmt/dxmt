use std::sync::Arc;
use std::time::Duration;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use prometheus::{
    Encoder, TextEncoder, Counter, Gauge, Histogram, HistogramOpts,
    register_counter, register_gauge, register_histogram,
};

use crate::types::{};
use crate::Result;
use crate::error::Error;

/// Metrics snapshot for external use
#[derive(Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub vectors_per_second: f64,
    pub queries_per_second: f64,
    pub average_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
}

/// Comprehensive monitoring for horizontal scaling optimizations
pub struct ScalingMetricsCollector {
    /// Object storage metrics
    object_storage_metrics: ObjectStorageMetrics,
    /// Write buffer metrics
    write_buffer_metrics: WriteBufferMetrics,
    /// Prefetch metrics
    prefetch_metrics: PrefetchMetrics,
    /// Thread-per-core metrics
    thread_metrics: ThreadMetrics,
    /// Cache metrics
    cache_metrics: CacheMetrics,
    /// System-wide metrics
    system_metrics: SystemMetrics,
    /// Metrics aggregator
    aggregator: Arc<RwLock<MetricsAggregator>>,
}

/// Object storage operation metrics
struct ObjectStorageMetrics {
    // Batch operations
    batch_gets: Counter,
    batch_puts: Counter,
    batch_deletes: Counter,
    batch_size_histogram: Histogram,
    
    // Parallelization
    concurrent_operations: Gauge,
    parallel_fetch_latency: Histogram,
    
    // Efficiency
    bytessaved_batching: Counter,
    api_calls_saved: Counter,
}

/// Write buffer performance metrics
struct WriteBufferMetrics {
    // Buffer state
    buffer_size: Gauge,
    buffer_items: Gauge,
    
    // Operations
    writes_buffered: Counter,
    writes_flushed: Counter,
    flush_operations: Counter,
    
    // Efficiency
    bytescoalesced: Counter,
    compression_ratio: Gauge,
    flush_latency: Histogram,
}

/// Prefetch system metrics
struct PrefetchMetrics {
    // Cache performance
    prefetchhits: Counter,
    prefetch_misses: Counter,
    cache_hit_rate: Gauge,
    
    // Prediction accuracy
    sequential_predictions: Counter,
    correlation_predictions: Counter,
    prediction_accuracy: Gauge,
    
    // Resource usage
    cache_memorybytes: Gauge,
    prefetch_queue_depth: Gauge,
}

/// Thread-per-core system metrics
struct ThreadMetrics {
    // Worker performance
    operations_per_worker: Vec<Counter>,
    worker_utilization: Vec<Gauge>,
    
    // Work stealing
    steal_attempts: Counter,
    steal_successes: Counter,
    steal_efficiency: Gauge,
    
    // Latency
    operation_latency: Histogram,
    queue_wait_time: Histogram,
}

/// Distributed cache metrics
struct CacheMetrics {
    // L1 (local) cache
    l1hits: Counter,
    l1_misses: Counter,
    l1_evictions: Counter,
    l1_memorybytes: Gauge,
    
    // L2 (distributed) cache
    l2hits: Counter,
    l2_misses: Counter,
    l2_latency: Histogram,
    
    // Overall efficiency
    cache_hit_rate: Gauge,
    write_through_operations: Counter,
}

/// System-wide performance metrics
struct SystemMetrics {
    // Throughput
    vectors_per_second: Gauge,
    queries_per_second: Gauge,
    
    // Latency percentiles
    p50_latency: Gauge,
    p95_latency: Gauge,
    p99_latency: Gauge,
    
    // Resource utilization
    cpu_utilization: Gauge,
    memory_utilization: Gauge,
    network_bandwidth: Gauge,
}

/// Aggregates metrics over time windows
struct MetricsAggregator {
    /// Time series data
    time_series: HashMap<String, TimeSeries>,
    /// Anomaly detection
    anomaly_detector: AnomalyDetector,
    /// Performance analyzer
    performance_analyzer: PerformanceAnalyzer,
}

#[derive(Clone)]
struct TimeSeries {
    metric_name: String,
    data_points: Vec<DataPoint>,
    window_size: Duration,
}

#[derive(Clone)]
struct DataPoint {
    timestamp: DateTime<Utc>,
    value: f64,
    tags: HashMap<String, String>,
}

struct AnomalyDetector {
    /// Z-score threshold for anomaly detection
    z_score_threshold: f64,
    /// Historical baselines
    baselines: HashMap<String, BaselineStats>,
}

#[derive(Clone)]
struct BaselineStats {
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
}

struct PerformanceAnalyzer {
    /// Performance recommendations
    recommendations: Vec<PerformanceRecommendation>,
    /// Bottleneck detection
    bottlenecks: Vec<Bottleneck>,
}

#[derive(Clone, Serialize, Deserialize)]
struct PerformanceRecommendation {
    severity: Severity,
    component: String,
    issue: String,
    recommendation: String,
    impact: String,
}

#[derive(Clone, Serialize, Deserialize)]
enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Serialize, Deserialize)]
struct Bottleneck {
    component: String,
    metric: String,
    current_value: f64,
    threshold: f64,
    impact: String,
}

// Helper macro to convert prometheus errors
macro_rules! register_metric {
    Ok($metric:expr) => {
        $metric.map_err(|e| crate::Error::Internal { 
            format!("Failed to register metric: {}", e) 
       ))
    };
}

impl ScalingMetricsCollector {
    pub fn new() -> crate::Result<Self> {
        // Initialize Prometheus metrics
        let object_storage_metrics = ObjectStorageMetrics {
            batch_gets: register_metric!(register_counter!("vectordb_batch_gets_total", "Total batch GET operations")?,
            batch_puts: register_metric!(register_counter!("vectordb_batch_puts_total", "Total batch PUT operations")?,
            batch_deletes: register_metric!(register_counter!("vectordb_batch_deletes_total", "Total batch DELETE operations")?,
            batch_size_histogram: register_metric!(register_histogram!(
                _HistogramOpts::new("vectordb_batch_size", "Batch operation sizes")
                    .buckets(vec![1.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
            )?,
            concurrent_operations: register_metric!(register_gauge!("vectordb_concurrent_ops", "Current concurrent operations")?,
            parallel_fetch_latency: register_metric!(register_histogram!(
                _HistogramOpts::new("vectordb_parallel_fetch_latency", "Parallel fetch latency in ms")
                    .buckets(vec![1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
            )?,
            bytessaved_batching: register_metric!(register_counter!("vectordbbytes_saved_batching", "Bytes saved through batching")?,
            api_calls_saved: register_metric!(register_counter!("vectordb_api_calls_saved", "API calls saved through batching")?;
        };
        
        let write_buffer_metrics = WriteBufferMetrics {
            buffer_size: register_metric!(register_gauge!("vectordb_write_bufferbytes", "Write buffer size in bytes")?,
            buffer_items: register_metric!(register_gauge!("vectordb_write_buffer_items", "Items in write buffer")?,
            writes_buffered: register_metric!(register_counter!("vectordb_writes_buffered_total", "Total writes buffered")?,
            writes_flushed: register_metric!(register_counter!("vectordb_writes_flushed_total", "Total writes flushed")?,
            flush_operations: register_metric!(register_counter!("vectordb_flush_operations_total", "Total flush operations")?,
            bytescoalesced: register_metric!(register_counter!("vectordbbytes_coalesced", "Bytes saved through coalescing")?,
            compression_ratio: register_metric!(register_gauge!("vectordb_compression_ratio", "Current compression ratio")?,
            flush_latency: register_metric!(register_histogram!(
                _HistogramOpts::new("vectordb_flush_latency", "Flush operation latency in ms")
                    .buckets(vec![10.0, 50.0, 100.0, 500.0, 1000.0])
            )?,
        };
        
        let prefetch_metrics = PrefetchMetrics {
            prefetchhits: register_metric!(register_counter!("vectordb_prefetch_hits_total", "Total prefetch cache hits")?,
            prefetch_misses: register_metric!(register_counter!("vectordb_prefetch_misses_total", "Total prefetch cache misses")?,
            cache_hit_rate: register_metric!(register_gauge!("vectordb_prefetch_hit_rate", "Prefetch cache hit rate")?,
            sequential_predictions: register_metric!(register_counter!("vectordb_sequential_predictions", "Sequential pattern predictions")?,
            correlation_predictions: register_metric!(register_counter!("vectordb_correlation_predictions", "Correlation-based predictions")?,
            prediction_accuracy: register_metric!(register_gauge!("vectordb_prediction_accuracy", "Prediction accuracy rate")?,
            cache_memorybytes: register_metric!(register_gauge!("vectordb_prefetch_cachebytes", "Prefetch cache memory usage")?,
            prefetch_queue_depth: register_metric!(register_gauge!("vectordb_prefetch_queue_depth", "Current prefetch queue depth")?;
        };
        
        let num_workers = num_cpus::get();
        let thread_metrics = ThreadMetrics {
            operations_per_worker: (0..num_workers)
                .map(|i| register_metric!(register_counter!(
                    format!("vectordb_worker_{}_ops_total", i),
                    format!("Total operations for worker {}", i)
                ))
                .collect::<Result<Vec<_>>>()?,
            worker_utilization: (0..num_workers)
                .map(|i| register_metric!(register_gauge!(
                    format!("vectordb_worker_{}_utilization", i),
                    format!("CPU utilization for worker {}", i)
                ))
                .collect::<Result<Vec<_>>>()?,
            steal_attempts: register_metric!(register_counter!("vectordb_steal_attempts_total", "Total work steal attempts")?,
            steal_successes: register_metric!(register_counter!("vectordb_steal_successes_total", "Successful work steals")?,
            steal_efficiency: register_metric!(register_gauge!("vectordb_steal_efficiency", "Work stealing efficiency")?,
            operation_latency: register_metric!(register_histogram!(
                _HistogramOpts::new("vectordb_operation_latency", "Operation latency in microseconds")
                    .buckets(vec![10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0])
            )?,
            queue_wait_time: register_metric!(register_histogram!(
                _HistogramOpts::new("vectordb_queue_wait_time", "Queue wait time in microseconds")
                    .buckets(vec![1.0, 10.0, 100.0, 1000.0])
            )?;
       );
        
        let cache_metrics = CacheMetrics {
            l1hits: register_metric!(register_counter!("vectordb_l1_cache_hits_total", "L1 cache hits")?,
            l1_misses: register_metric!(register_counter!("vectordb_l1_cache_misses_total", "L1 cache misses")?,
            l1_evictions: register_metric!(register_counter!("vectordb_l1_cache_evictions_total", "L1 cache evictions")?,
            l1_memorybytes: register_metric!(register_gauge!("vectordb_l1_cachebytes", "L1 cache memory usage")?,
            l2hits: register_metric!(register_counter!("vectordb_l2_cache_hits_total", "L2 cache hits")?,
            l2_misses: register_metric!(register_counter!("vectordb_l2_cache_misses_total", "L2 cache misses")?,
            l2_latency: register_metric!(register_histogram!(
                _HistogramOpts::new("vectordb_l2_cache_latency", "L2 cache latency in ms")
                    .buckets(vec![1.0, 5.0, 10.0, 50.0, 100.0])
            )?,
            cache_hit_rate: register_metric!(register_gauge!("vectordb_overall_cache_hit_rate", "Overall cache hit rate")?,
            write_through_operations: register_metric!(register_counter!("vectordb_write_through_ops_total", "Write-through operations")?;
        };
        
        let system_metrics = SystemMetrics {
            vectors_per_second: register_metric!(register_gauge!("vectordb_vectors_per_second", "Current vectors/second throughput")?,
            queries_per_second: register_metric!(register_gauge!("vectordb_queries_per_second", "Current queries/second")?,
            p50_latency: register_metric!(register_gauge!("vectordb_p50_latency_ms", "50th percentile latency")?,
            p95_latency: register_metric!(register_gauge!("vectordb_p95_latency_ms", "95th percentile latency")?,
            p99_latency: register_metric!(register_gauge!("vectordb_p99_latency_ms", "99th percentile latency")?,
            cpu_utilization: register_metric!(register_gauge!("vectordb_cpu_utilization", "CPU utilization percentage")?,
            memory_utilization: register_metric!(register_gauge!("vectordb_memory_utilization", "Memory utilization percentage")?,
            network_bandwidth: register_metric!(register_gauge!("vectordb_network_bandwidth_mbps", "Network bandwidth in Mbps")?;
        };
        
        let aggregator = Arc::new(RwLock::new(MetricsAggregator {
            time_series: HashMap::new(),
            anomaly_detector: AnomalyDetector {
                _z_score_threshold: 3.0,
                baselines: HashMap::new(),
            },
            performance_analyzer: PerformanceAnalyzer {
                _recommendations: Vec::new(),
                bottlenecks: Vec::new(),
            },
        });
        
        Ok(Self {
            object_storage_metrics,
            write_buffer_metrics,
            prefetch_metrics,
            thread_metrics,
            cache_metrics,
            system_metrics,
            aggregator,
        })
    }
    
    /// Record batch operation metrics
    pub fn record_batch_operation(&self, operation: &str, batchsize: usize, latency_ms: f64) {
        match operation {
            "get" => self.object_storage_metrics.batch_gets.inc(),
            "put" => self.object_storage_metrics.batch_puts.inc(),
            "delete" => self.object_storage_metrics.batch_deletes.inc(),
            _ => {}
        }
        
        self.object_storage_metrics.batch_size_histogram.observe(batch_size as f64);
        self.object_storage_metrics.parallel_fetch_latency.observe(latency_ms);
        
        // Calculate efficiency
        let api_calls_saved = batch_size.saturating_sub(1);
        self.object_storage_metrics.api_calls_saved.inc_by(api_calls_saved as f64);
    }
    
    /// Record write buffer metrics
    pub fn record_write_buffer_stats(&self, stats: &crate::storage::write_buffer::WriteBufferStats) {
        self.write_buffer_metrics.writes_buffered.inc_by(stats.buffered_records as f64);
        self.write_buffer_metrics.writes_flushed.inc_by(stats.flushed_records as f64);
        self.write_buffer_metrics.flush_operations.inc_by(stats.flushed_batches as f64);
        self.write_buffer_metrics.bytes_coalesced.inc_by(stats.bufferedbytes as f64);
        
        if stats.bufferedbytes > 0 {

        
            let compression_ratio = 0.0; // No compression_savings field available
            self.write_buffer_metrics.compression_ratio.set(compression_ratio);

        }
    }
    
    /// Record prefetch metrics
    pub fn record_prefetch_stats(&self, stats: &crate::storage::prefetch_manager::PrefetchStats) {
        self.prefetch_metrics.prefetch_hits.inc_by(stats.prefetch_hits as f64);
        self.prefetch_metrics.prefetch_misses.inc_by(stats.prefetch_misses as f64);
        self.prefetch_metrics.cache_hit_rate.set(stats.hit_rate);
        self.prefetch_metrics.cache_memorybytes.set(stats.bytes_prefetched as f64);
    }
    
    /// Record thread-per-core metrics
    pub fn record_thread_stats(&self, stats: &crate::distributed::worker::thread_per_core_optimized::SystemStats) {
        // Update worker metrics
        for (i, worker_stat) in stats.worker_stats.iter().enumerate() {
            if i < self.thread_metrics.operations_per_worker.len() {
                self.thread_metrics.operations_per_worker[i].inc_by(worker_stat.operations as f64);
                
                let utilization = if worker_stat.idle_cycles > 0 {
                    1.0 - (worker_stat.idle_cycles as f64 / (worker_stat.operations + worker_stat.idle_cycles) as f64)
                } else {
                    1.0
                };
                self.thread_metrics.worker_utilization[i].set(utilization);
            }
            
            self.thread_metrics.steal_attempts.inc_by(worker_stat.steal_attempts as f64);
            self.thread_metrics.steal_successes.inc_by(worker_stat.steal_successes as f64);
        }
        
        // Calculate steal efficiency
        let total_attempts: u64 = stats.worker_stats.iter().map(|w| w.steal_attempts).sum();
        let total_successes: u64 = stats.worker_stats.iter().map(|w| w.steal_successes).sum();
        if total_attempts > 0 {

            let efficiency = total_successes as f64 / total_attempts as f64;
            self.thread_metrics.steal_efficiency.set(efficiency);

        }
        // Record average latency
        if stats.average_latency_us > 0 {
            self.thread_metrics.operation_latency.observe(stats.average_latency_us as f64)
        }
    }
    
    /// Analyze performance and generate recommendations
    pub async fn analyze_performance(&self) -> PerformanceReport {
        let aggregator = self.aggregator.write()?;
        
        // Detect anomalies
        let anomalies = aggregator.anomaly_detector.detect_anomalies();
        
        // Analyze bottlenecks
        let bottlenecks = self.detect_bottlenecks();
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&bottlenecks);
        
        PerformanceReport {
            timestamp: chrono::Utc::now(),
            anomalies,
            bottlenecks,
            recommendations,
            summary: self.generate_summary(),
        }
    }
    
    fn detect_bottlenecks(&self) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();
        
        // Check object storage batching efficiency
        let batch_gets = self.object_storage_metrics.batch_gets.get();
        let api_calls_saved = self.object_storage_metrics.api_calls_saved.get();
        
        if batch_gets > 0.0 && api_calls_saved / batch_gets < 10.0 {
            bottlenecks.push(Bottleneck {
                component: "ObjectStorage".to_string(),
                metric: "batch_efficiency".to_string(),
                current_value: api_calls_saved / batch_gets,
                threshold: 10.0,
                impact: "Low batch sizes reducing efficiency".to_string(),
            })
        }
        
        // Check cache hit rate
        let cache_hit_rate = self.cache_metrics.cache_hit_rate.get();
        if cache_hit_rate < 0.8 {
            bottlenecks.push(Bottleneck {
                component: "Cache".to_string(),
                metric: "hit_rate".to_string(),
                current_value: cache_hit_rate,
                threshold: 0.8,
                impact: "High cache miss rate increasing latency".to_string(),
            })
        }
        
        // Check work stealing efficiency
        let steal_efficiency = self.thread_metrics.steal_efficiency.get();
        if steal_efficiency < 0.5 {
            bottlenecks.push(Bottleneck {
                component: "ThreadPerCore".to_string(),
                metric: "steal_efficiency".to_string(),
                current_value: steal_efficiency,
                threshold: 0.5,
                impact: "Poor work distribution across cores".to_string(),
            })
        }
        
        bottlenecks
    }
    
    fn generate_recommendations(&self, bottlenecks: &[Bottleneck]) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();
        
        for bottleneck in bottlenecks {
            let recommendation = match bottleneck.component.as_str() {
                "ObjectStorage" => PerformanceRecommendation {
                    severity: _Severity::Medium,
                    component: bottleneck.component.clone(),
                    issue: "Small batch sizes".to_string(),
                    recommendation: "Increase batch_size configuration to 100-1000".to_string(),
                    impact: "Could improve throughput by 5-10x".to_string(),
                },
                "Cache" => PerformanceRecommendation {
                    severity: _Severity::High,
                    component: bottleneck.component.clone(),
                    issue: "Low cache hit rate".to_string(),
                    recommendation: "Increase cache size or improve prefetch predictions".to_string(),
                    impact: "Could reduce average latency by 50%".to_string(),
                },
                "ThreadPerCore" => PerformanceRecommendation {
                    severity: _Severity::Medium,
                    component: bottleneck.component.clone(),
                    issue: "Inefficient work stealing".to_string(),
                    recommendation: "Adjust steal_attempts or review workload distribution".to_string(),
                    impact: "Could improve CPU utilization by 20%".to_string(),
                },
                _ => continue,
            };
            
            recommendations.push(recommendation);
        }
        
        recommendations
    }
    
    fn generate_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            vectors_per_second: self.system_metrics.vectors_per_second.get(),
            queries_per_second: self.system_metrics.queries_per_second.get(),
            average_latency_ms: self.system_metrics.p50_latency.get(),
            cache_hit_rate: self.cache_metrics.cache_hit_rate.get(),
            cpu_utilization: self.system_metrics.cpu_utilization.get(),
            memory_utilization: self.system_metrics.memory_utilization.get(),
        }
    }
    
    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = vec![];
        encoder.encode(&metric_families, &mut buffer)?;
        String::from_utf8(buffer)?
    }
    
    /// Take a snapshot of current metrics
    pub async fn take_snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            timestamp: chrono::Utc::now(),
            vectors_per_second: self.system_metrics.vectors_per_second.get(),
            queries_per_second: self.system_metrics.queries_per_second.get(),
            average_latency_ms: self.system_metrics.p50_latency.get(),
            cache_hit_rate: self.cache_metrics.cache_hit_rate.get(),
            cpu_utilization: self.system_metrics.cpu_utilization.get(),
            memory_utilization: self.system_metrics.memory_utilization.get(),
        }
    }
}

impl AnomalyDetector {
    fn detect_anomalies(&self) -> Vec<Anomaly> {
        let anomalies = Vec::new();
        
        // Simplified anomaly detection
        // In production, would use statistical methods
        
        anomalies
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub anomalies: Vec<Anomaly>,
    pub bottlenecks: Vec<Bottleneck>,
    pub recommendations: Vec<PerformanceRecommendation>,
    pub summary: PerformanceSummary,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub metric: String,
    pub value: f64,
    pub expected_range: (f64, f64),
    pub severity: Severity,
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub vectors_per_second: f64,
    pub queries_per_second: f64,
    pub average_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_metrics_collection() {
        let collector = ScalingMetricsCollector::new()?;
        
        // Record some metrics
        collector.record_batch_operation("get", 100, 50.0);
        
        // Check Prometheus export
        let prometheus_output = collector.export_prometheus();
        assert!(prometheus_output.contains("vectordb_batch_gets_total");
        
        // Generate performance report
        let report = collector.analyze_performance()?;
        assert!(report.timestamp.elapsed() < Duration::from_secs(1)
    }
}
