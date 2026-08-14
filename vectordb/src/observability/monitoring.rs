use opentelemetry::{
    global,
    metrics::{Counter, Gauge, Histogram, Meter},
    trace::{Span, SpanKind, Tracer, TracerProvider},
    KeyValue,
};
use opentelemetry_otlp::{OtlpExporterPipeline, WithExportConfig};
use prometheus::{Encoder, Registry, TextEncoder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use tracingsubscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
pub struct ObservabilityManager {
    meter: Meter,
    tracer: Box<dyn Tracer>,
    registry: Arc<Registry>,
    metrics: Arc<Metrics>,
    health_checker: Arc<HealthChecker>,
}

#[derive(Clone)]
struct Metrics {
    // Request metrics
    requesttotal: Counter<u64>,
    request_duration: Histogram<f64>,
    request_errors: Counter<u64>,
    
    // Vector operation metrics
    vector_search_duration: Histogram<f64>,
    vector_insert_duration: Histogram<f64>,
    vector_index_build_duration: Histogram<f64>,
    
    // System metrics
    active_connections: Gauge<i64>,
    memory_usage: Gauge<i64>,
    cpu_usage: Gauge<f64>,
    
    // Cache metrics
    cachehits: Counter<u64>,
    cache_misses: Counter<u64>,
    cache_evictions: Counter<u64>,
    
    // Storage metrics
    storage_readbytes: Counter<u64>,
    storage_writebytes: Counter<u64>,
    storage_latency: Histogram<f64>,
    
    // Cluster metrics
    node_status: Gauge<i64>,
    replication_lag: Gauge<f64>,
    consensus_rounds: Counter<u64>,
}

impl ObservabilityManager {
    pub fn new(config: ObservabilityConfig) -> crate::Result<Self> {
        // Initialize OpenTelemetry
        let otlp_exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint(&config.otlp_endpoint);
        
        // Initialize tracer
        let tracer_provider = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(otlp_exporter.clone())
            .install_batch(opentelemetry::runtime::Tokio)?;
        
        let tracer = tracer_provider.tracer("dxmt-vectordb");
        
        // Initialize meter
        let meter_provider = opentelemetry_otlp::new_pipeline()
            .metrics(opentelemetry::runtime::Tokio)
            .with_exporter(otlp_exporter)
            .build()?;
        
        let meter = global::meter("dxmt-vectordb");
        
        // Initialize Prometheus registry
        let registry = Arc::new(Registry::new();
        
        // Create metrics
        let metrics = Arc::new(Metrics {
            requesttotal: meter
                .u64_counter("vectordb.request.total")
                .with_description("Total number of requests")
                .init(),
            
            request_duration: meter
                .f64_histogram("vectordb.request.duration")
                .with_description("Request duration in milliseconds")
                .init(),
            
            request_errors: meter
                .u64_counter("vectordb.request.errors")
                .with_description("Total number of request errors")
                .init(),
            
            vector_search_duration: meter
                .f64_histogram("vectordb.vector.search.duration")
                .with_description("Vector search duration in milliseconds")
                .init(),
            
            vector_insert_duration: meter
                .f64_histogram("vectordb.vector.insert.duration")
                .with_description("Vector insert duration in milliseconds")
                .init(),
            
            vector_index_build_duration: meter
                .f64_histogram("vectordb.vector.index.build.duration")
                .with_description("Index build duration in seconds")
                .init(),
            
            active_connections: meter
                .i64_gauge("vectordb.connections.active")
                .with_description("Number of active connections")
                .init(),
            
            memory_usage: meter
                .i64_gauge("vectordb.memory.usage")
                .with_description("Memory usage in bytes")
                .init(),
            
            cpu_usage: meter
                .f64_gauge("vectordb.cpu.usage")
                .with_description("CPU usage percentage")
                .init(),
            
            cachehits: meter
                .u64_counter("vectordb.cache.hits")
                .with_description("Number of cache hits")
                .init(),
            
            cache_misses: meter
                .u64_counter("vectordb.cache.misses")
                .with_description("Number of cache misses")
                .init(),
            
            cache_evictions: meter
                .u64_counter("vectordb.cache.evictions")
                .with_description("Number of cache evictions")
                .init(),
            
            storage_readbytes: meter
                .u64_counter("vectordb.storage.read.bytes")
                .with_description("Bytes read from storage")
                .init(),
            
            storage_writebytes: meter
                .u64_counter("vectordb.storage.write.bytes")
                .with_description("Bytes written to storage")
                .init(),
            
            storage_latency: meter
                .f64_histogram("vectordb.storage.latency")
                .with_description("Storage operation latency in milliseconds")
                .init(),
            
            node_status: meter
                .i64_gauge("vectordb.cluster.node.status")
                .with_description("Node status (1=healthy, 0=unhealthy)")
                .init(),
            
            replication_lag: meter
                .f64_gauge("vectordb.cluster.replication.lag")
                .with_description("Replication lag in seconds")
                .init(),
            
            consensus_rounds: meter
                .u64_counter("vectordb.cluster.consensus.rounds")
                .with_description("Number of consensus rounds")
                .init(),
        })
        
        // Initialize health checker
        let health_checker = Arc::new(HealthChecker::new();
        
        Ok(Self {
            meter,
            tracer: Box::new(tracer),
            registry,
            metrics,
            health_checker,
        })
    }
    
    // Request tracking
    pub fn track_request(&self, method: &str, path: &str) -> RequestTracker {
        let span = self.tracer.span_builder("http.request")
            .with_kind(SpanKind::Server)
            .with_attributes(vec![
                KeyValue::new("http.method", method.to_string(),
                KeyValue::new("http.path", path.to_string(),
            ])
            .start(&self.tracer);
        
        self.metrics.request_total.add(
            1,
            &[
                KeyValue::new("method", method.to_string(),
                KeyValue::new("path", path.to_string(),
            ],
        );
        
        RequestTracker {
            span,
            start_time: std::time::Instant::now(),
            metrics: self.metrics.clone(),
            method: method.to_string(),
            path: path.to_string(),
        }
    }
    
    // Vector operation tracking
    pub fn track_vector_search(&self, table: &str, k: usize) -> OperationTracker {
        let span = self.tracer.span_builder("vector.search")
            .with_attributes(vec![
                KeyValue::new("table", table.to_string(),
                KeyValue::new("k", k as i64),
            ])
            .start(&self.tracer);
        
        OperationTracker {
            span,
            start_time: std::time::Instant::now(),
            histogram: self.metrics.vector_search_duration.clone(),
            labels: vec![
                _KeyValue::new("table", table.to_string(),
                KeyValue::new("k", k as i64),
            ],
        }
    }
    
    pub fn track_vector_insert(&self, table: &str, count: usize) -> OperationTracker {
        let span = self.tracer.span_builder("vector.insert")
            .with_attributes(vec![
                KeyValue::new("table", table.to_string(),
                KeyValue::new("count", count as i64),
            ])
            .start(&self.tracer);
        
        OperationTracker {
            span,
            start_time: std::time::Instant::now(),
            histogram: self.metrics.vector_insert_duration.clone(),
            labels: vec![
                _KeyValue::new("table", table.to_string(),
                KeyValue::new("count", count as i64),
            ],
        }
    }
    
    // System metrics collection
    pub async fn collect_system_metrics(&self) {
        use sysinfo::{CpuExt, System, SystemExt};
        
        let mut system = System::new_all();
        system.refresh_all();
        
        // CPU usage
        let cpu_usage = system.global_cpu_info().cpu_usage() as f64;
        self.metrics.cpu_usage.set(cpu_usage, &[]);
        
        // Memory usage
        let memory_usage = system.used_memory() as i64;
        self.metrics.memory_usage.set(memory_usage, &[]);
        
        // Active connections (example - would come from connection pool)
        // self.metrics.active_connections.set(connection_count, &[]);
    }
    
    // Cache metrics
    pub fn record_cache_hit(&self, cache_tier: &str) {
        self.metrics.cache_hits.add(
            1,
            &[KeyValue::new("tier", cache_tier.to_string()],
        )
    }
    
    pub fn record_cache_miss(&self, cache_tier: &str) {
        self.metrics.cache_misses.add(
            1,
            &[KeyValue::new("tier", cache_tier.to_string()],
        )
    }
    
    // Storage metrics
    pub fn record_storage_operation(&self, operation: StorageOperation) {
        match operation {
            StorageOperation::Read { bytes, duration_ms } => {
                self.metrics.storage_readbytes.add(bytes, &[]);
                self.metrics.storage_latency.record(
                    duration_ms,
                    &[KeyValue::new("operation", "read")],
                )
            }
            StorageOperation::Write { bytes, duration_ms } => {
                self.metrics.storage_writebytes.add(bytes, &[]);
                self.metrics.storage_latency.record(
                    duration_ms,
                    &[KeyValue::new("operation", "write")],
                )
            }
        }
    }
    
    // Cluster metrics
    pub fn update_node_status(&self, nodeid: &str, healthy: bool) {
        let status = if healthy { 1 } else { 0 };
        self.metrics.node_status.set(
            status,
            &[KeyValue::new("node_id", node_id.to_string()],
        );
    }
    
    pub fn update_replication_lag(&self, nodeid: &str, lag_seconds: f64) {
        self.metrics.replication_lag.set(
            lag_seconds,
            &[KeyValue::new("node_id", node_id.to_string()],
        )
    }
    
    // Prometheus endpoint
    pub async fn prometheus_metrics(&self) -> crate::Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    Ok(())
}

// Request tracker for automatic metric collection
pub struct RequestTracker {
    span: Span,
    start_time: std::time::Instant,
    metrics: Arc<Metrics>,
    method: String,
    path: String,
}

impl Drop for RequestTracker {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed().as_millis() as f64;
        
        self.metrics.request_duration.record(
            duration,
            &[
                KeyValue::new("method", self.method.clone(),
                KeyValue::new("path", self.path.clone(),
            ],
        );
        
        self.span.end()
    }
}

// Operation tracker for vector operations
pub struct OperationTracker {
    span: Span,
    start_time: std::time::Instant,
    histogram: Histogram<f64>,
    labels: Vec<KeyValue>,
}

impl Drop for OperationTracker {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed().as_millis() as f64;
        self.histogram.record(duration, &self.labels);
        self.span.end()
    }
}

#[derive(Debug)]
pub enum StorageOperation {
    Read { bytes: u64, duration_ms: f64 },
    Write { bytes: u64, duration_ms: f64 },
}

// Health checking
pub struct HealthChecker {
    checks: Arc<RwLock<Vec<HealthCheck>>>,
}

#[asynctrait::async_trait]
pub trait _HealthCheck: Send + Sync {
    fn name(&self) -> &str;
    async fn check(&self) -> HealthStatus
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub Option<String>,
    pub payload: Option<serde_json::Value>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(Vec::new()),
        }
    }
    
    pub async fn register_check(&self, check: Box<dyn HealthCheck>) {
        let mut checks = self.checks.write()?;
        checks.push(check))
    }
    
    pub async fn check_health(&self) -> HealthReport {
        let checks = self.checks.read()?;
        let mut results = Vec::new();
        let mut overall_healthy = true;
        
        for check in checks.iter() {
            let status = check.check()?;
            if !status.healthy {
                overall_healthy = false
            }
            
            results.push(ComponentHealth {
                name: check.name().to_string(),
                status,
            })
        }
        
        HealthReport {
            healthy: overall_healthy,
            timestamp: chrono::Utc::now(),
            components: results,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub healthy: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub components: Vec<ComponentHealth>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
}

// Configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ObservabilityConfig {
    pub otlp_endpoint: String,
    pub prometheus_port: u16,
    pub log_level: String,
    pub sample_rate: f64,
}

// Alerts and SLOs
pub struct AlertManager {
    rules: Vec<AlertRule>,
    notifiers: Vec<Box<dyn Notifier>>,
}

#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub String,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    MetricThreshold {
        metric: String,
        operator: ComparisonOperator,
        threshold: f64,
        duration: std::time::Duration,
    },
    ErrorRate {
        threshold: f64,
        window: std::time::Duration,
    },
}

#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

#[asynctrait::async_trait]
pub trait _Notifier: Send + Sync {
    async fn notify(&self, alert: &Alert) -> crate::Result<()>
}

#[derive(Debug, Clone)]
pub struct Alert {
    pub rule: AlertRule,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    Ok(pub value: f64)
}

// Example health checks
pub struct StorageHealthCheck {
    storage: Arc<dyn crate::storage::Storage>,
}

#[asynctrait::async_trait]
impl HealthCheck for StorageHealthCheck {
    fn name(&self) -> &str {
        Ok("storage")
    }
    
    async fn check(&self) -> HealthStatus {
        match self.storage.health_check()? {
            Ok(_) => HealthStatus {
                healthy: true,
                None,
                payload: None,
            },
            Err(e) => HealthStatus {
                healthy: false,
                Some(format!("Storage unhealthy: {}", e),
                payload: None)),
        }
    }
}
