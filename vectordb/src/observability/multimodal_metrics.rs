use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::config::multimodal::MonitoringConfig;

use crate::Result;

/// Comprehensive multi-modal metrics collector
#[derive(Debug)]
pub struct MultimodalMetricsCollector {
    config: MonitoringConfig,
    metrics: Arc<RwLock<MultimodalMetrics>>,
    embedding_metrics: Arc<RwLock<HashMap<String, EmbeddingModelMetrics>>>,
    cache_metrics: Arc<RwLock<CacheMetrics>>,
    alerts: Arc<RwLock<Vec<Alert>>>,
    start_time: Instant,
}

/// Overall multi-modal system metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultimodalMetrics {
    /// Total number of embedding requests processed
    pub total_requests: u64,
    
    /// Number of successful requests
    pub successful_requests: u64,
    
    /// Number of failed requests
    pub failed_requests: u64,
    
    /// Total processing time across all requests (milliseconds)
    pub total_processing_time_ms: u64,
    
    /// Average latency in milliseconds
    pub average_latency_ms: f64,
    
    /// P95 latency in milliseconds
    pub p95_latency_ms: f64,
    
    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,
    
    /// Requests per minute
    pub requests_per_minute: f64,
    
    /// Current error rate (0.0 - 1.0)
    pub error_rate: f64,
    
    /// Throughput in embeddings per second
    pub throughput_eps: f64,
    
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    
    /// GPU utilization percentage (if available)
    pub gpu_utilization_percent: Option<f64>,
    
    /// GPU memory usage in MB (if available)
    pub gpu_memory_usage_mb: Option<f64>,
    
    /// Number of active models
    pub active_models: usize,
    
    /// Uptime in seconds
    pub uptime_seconds: u64,
    
    /// Last updated timestamp
    pub last_updated: u64,
}

/// Metrics for individual embedding models
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingModelMetrics {
    /// Model name
    pub model_name: String,
    
    /// Model type (openai, clip, whisper)
    pub model_type: String,
    
    /// Supported modalities
    pub modalities: Vec<String>,
    
    /// Number of requests processed
    pub requests_processed: u64,
    
    /// Number of successful requests
    pub successful_requests: u64,
    
    /// Number of failed requests
    pub failed_requests: u64,
    
    /// Average processing time per request (ms)
    pub avg_processing_time_ms: f64,
    
    /// P95 processing time (ms)
    pub p95_processing_time_ms: f64,
    
    /// Total embeddings generated
    pub total_embeddings_generated: u64,
    
    /// Average batch size
    pub avg_batch_size: f64,
    
    /// Model load time (ms)
    pub model_load_time_ms: u64,
    
    /// Last used timestamp
    pub last_used: u64,
    
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    
    /// Process restarts
    pub process_restarts: u64,
    
    /// Current status
    pub status: ModelStatus,
    
    /// Latency histogram (buckets in ms -> count)
    pub latency_histogram: HashMap<u64, u64>,
}

/// Cache performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// L1 (memory) cache metrics
    pub l1_cache: CacheTierMetrics,
    
    /// L2 (Redis) cache metrics
    pub l2_cache: Option<CacheTierMetrics>,
    
    /// L3 (disk) cache metrics
    pub l3_cache: Option<CacheTierMetrics>,
    
    /// Overall cache hit rate
    pub overall_hit_rate: f64,
    
    /// Cache promotion events (L3->L2->L1)
    pub promotion_events: u64,
    
    /// Cache demotion events (L1->L2->L3)
    pub demotion_events: u64,
    
    /// Cache eviction events
    pub eviction_events: u64,
}

/// Metrics for individual cache tiers
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheTierMetrics {
    /// Total requests to this tier
    pub requests: u64,
    
    /// Cache hits
    pub hits: u64,
    
    /// Cache misses
    pub misses: u64,
    
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
    
    /// Average lookup time (ms)
    pub avg_lookup_time_ms: f64,
    
    /// Current size in entries
    pub current_size: usize,
    
    /// Maximum capacity
    pub max_capacity: usize,
    
    /// Memory usage in MB
    pub memory_usage_mb: f64,
}

/// Model status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    Unloaded,
    Loading,
    Ready,
    Error,
    Restarting,
}

impl Default for ModelStatus {
    fn default() -> Self {
        ModelStatus::Unloaded
    }
}

/// Alert structure for monitoring events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub String,
    pub metric_name: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub timestamp: u64,
    pub resolved: bool,
}

/// Types of alerts that can be generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighErrorRate,
    HighLatency,
    HighMemoryUsage,
    LowCacheHitRate,
    ModelFailure,
    ProcessRestart,
    GPUOutOfMemory,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl MultimodalMetricsCollector {
    /// Create a new metrics collector
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(MultimodalMetrics::default()),
            embedding_metrics: Arc::new(RwLock::new(HashMap::new()),
            cache_metrics: Arc::new(RwLock::new(CacheMetrics::default()),
            alerts: Arc::new(RwLock::new(Vec::new()),
            start_time: _Instant::now(),
        }
    }
    
    /// Start the metrics collection background task
    pub async fn start(&self) -> crate::Result<()> {
        if !self.config.enabled {
            info!("Multi-modal metrics collection is disabled");
            return Ok((;
        }
        
        info!("Starting multi-modal metrics collection");
        
        let metrics = self.metrics.clone();
        let start_time = self.start_time;
        let collection_interval = Duration::from_secs(self.config.collection_interval);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(collection_interval);
            
            loop {
                interval.tick()?;
                
                // Update uptime
                let mut metrics_guard = metrics.write()?;
                metrics_guard.uptime_seconds = start_time.elapsed().as_secs();
                metrics_guard.last_updated = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    ?
                    .as_secs()
            }
        })
        
        // Start alerting if enabled
        if self.config.alerting.enabled {
    self.start_alerting()?)?;
}
        // Start health checks if enabled
        if self.config.health_checks.check_models || self.config.health_checks.check_cache {
    self.start_health_checks()?)?;
}
        Ok(())
    }
    
    /// Record a new embedding request
    pub async fn record_embedding_request(
        &self,
        model_name: &str,
        model_type: &str,
        modality: &str,
        processing_time: Duration,
        batchsize: usize,
        success: bool) {
        let processing_time_ms = processing_time.as_millis() as u64;
        
        // Update overall metrics
        {
            let mut metrics = self.metrics.write()?;
            metrics.total_requests += 1;
            metrics.total_processing_time_ms += processing_time_ms;
            
            if success {
                metrics.successful_requests += 1
            } else {
                metrics.failed_requests += 1
            }
            
            // Calculate derived metrics
            metrics.error_rate = metrics.failed_requests as f64 / metrics.total_requests as f64;
            metrics.average_latency_ms = metrics.total_processing_time_ms as f64 / metrics.total_requests as f64;
        }
        
        // Update model-specific metrics
        {
            let mut embedding_metrics = self.embedding_metrics.write()?;
            let model_metrics = embedding_metrics
                .entry(model_name.to_string())
                .or_insert_with(|| EmbeddingModelMetrics {
                    model_name: model_name.to_string(),
                    model_type: model_type.to_string(),
                    modalities: vec![modality.to_string()],
                    status: _ModelStatus::Ready,
                    ..Default::default()
                })
            
            model_metrics.requests_processed += 1;
            model_metrics.total_embeddings_generated += batch_size as u64;
            model_metrics.last_used = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                ?
                .as_secs();
            
            if success {
                model_metrics.successful_requests += 1
            } else {
                model_metrics.failed_requests += 1
            }
            
            // Update latency metrics
            let total_time = model_metrics.avg_processing_time_ms * (model_metrics.requests_processed - 1) as f64;
            model_metrics.avg_processing_time_ms = 
                (total_time + processing_time_ms as f64) / model_metrics.requests_processed as f64;
            
            // Update batch size
            let total_batch = model_metrics.avg_batch_size * (model_metrics.requests_processed - 1) as f64;
            model_metrics.avg_batch_size = 
                (total_batch + batch_size as f64) / model_metrics.requests_processed as f64;
            
            // Update latency histogram
            let bucket = Self::get_latency_bucket(processing_time_ms);
            *model_metrics.latency_histogram.entry(bucket).or_insert(0) += 1;
        }
        
        // Check for alerts
        if self.config.alerting.enabled {

            self.check_alerts()?;

        }
    }
    
    /// Record cache hit/miss events
    pub async fn record_cache_event(
        &self,
        tier: &str,
        hit: bool,
        lookup_time: Duration) {
        let mut cache_metrics = self.cache_metrics.write()?;
        
        let tier_metrics = match tier {
            "l1" => &mut cache_metrics.l1_cache,
            "l2" => cache_metrics.l2_cache.get_or_insert_with(Default::default),
            "l3" => cache_metrics.l3_cache.get_or_insert_with(Default::default),
            _ => return,
        };
        
        tier_metrics.requests += 1;
        
        if hit {
            tier_metrics.hits += 1
        } else {
            tier_metrics.misses += 1
        }
        
        tier_metrics.hit_rate = tier_metrics.hits as f64 / tier_metrics.requests as f64;
        
        // Update average lookup time
        let total_time = tier_metrics.avg_lookup_time_ms * (tier_metrics.requests - 1) as f64;
        tier_metrics.avg_lookup_time_ms = 
            (total_time + lookup_time.as_millis() as f64) / tier_metrics.requests as f64;
        
        // Update overall hit rate
        let total_hits = cache_metrics.l1_cache.hits + 
            cache_metrics.l2_cache.as_ref().map_or(0, |c| c.hits) +
            cache_metrics.l3_cache.as_ref().map_or(0, |c| c.hits);
        let total_requests = cache_metrics.l1_cache.requests + 
            cache_metrics.l2_cache.as_ref().map_or(0, |c| c.requests) +
            cache_metrics.l3_cache.as_ref().map_or(0, |c| c.requests);
        
        if total_requests > 0 {
            cache_metrics.overall_hit_rate = total_hits as f64 / total_requests as f64
        }
    }
    
    /// Record model status change
    pub async fn record_model_status(&self, model_name: &str, status: ModelStatus) {
        let mut embedding_metrics = self.embedding_metrics.write()?;
        if let Some(model_metrics) = embedding_metrics.get_mut(model_name) {
            model_metrics.status = status
        }
    }
    
    /// Record model restart
    pub async fn record_model_restart(&self, model_name: &str) {
        let mut embedding_metrics = self.embedding_metrics.write()?;
        if let Some(model_metrics) = embedding_metrics.get_mut(model_name) {
            model_metrics.process_restarts += 1
        }
        
        // Generate alert
        if self.config.alerting.enabled {
            let alert = Alert {
                id: format!("restart_{}_{}", model_name, SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                alert_type: _AlertType::ProcessRestart,
                severity: _AlertSeverity::Warning,
                format!("Model {} has been restarted", model_name),
                metric_name: "process_restarts".to_string(),
                metric_value: 1.0,
                threshold: 0.0,
                timestamp: _SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                resolved: false,
           };
            
            let mut alerts = self.alerts.write()?;
            alerts.push(alert);
        }
    }
    
    /// Get current metrics snapshot
    pub async fn get_metrics(&self) -> MultimodalMetrics {
        self.metrics.read()??.clone()
    }
    
    /// Get model-specific metrics
    pub async fn get_model_metrics(&self, model_name: &str) -> Option<EmbeddingModelMetrics> {
        self.embedding_metrics.read()??.get(model_name).cloned()
    }
    
    /// Get all model metrics
    pub async fn get_all_model_metrics(&self) -> HashMap<String, EmbeddingModelMetrics> {
        self.embedding_metrics.read()??.clone()
    }
    
    /// Get cache metrics
    pub async fn get_cache_metrics(&self) -> CacheMetrics {
        self.cache_metrics.read()??.clone()
    }
    
    /// Get recent alerts
    pub async fn get_alerts(&self, limit: Option<usize>) -> Vec<Alert> {
        let alerts = self.alerts.read()?;
        let mut sorted_alerts = alerts.clone();
        sorted_alerts.sort_by(|a, b| b.timestamp.cmp(&a.timestamp);
        
        if let Some(limit) = limit {
            sorted_alerts.into_iter().take(limit).collect()
        } else {
            sorted_alerts
        }
    }
    
    /// Start alerting background task
    async fn start_alerting(&self) -> crate::Result<()> {
        let metrics = self.metrics.clone();
        let embedding_metrics = self.embedding_metrics.clone();
        let cache_metrics = self.cache_metrics.clone();
        let alerts = self.alerts.clone();
        let alerting_config = self.config.alerting.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30); // Check every 30 seconds
            
            loop {
                interval.tick()?;
                
                // Check error rate
                let metrics_guard = metrics.read()?;
                if metrics_guard.error_rate > alerting_config.error_rate_threshold as f64 {
                    let alert = Alert {
                        id: format!("error_rate_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                        alert_type: _AlertType::HighErrorRate,
                        severity: _AlertSeverity::Critical,
                        format!("High error rate detected: {:.2}%", metrics_guard.error_rate * 100.0),
                        metric_name: "error_rate".to_string(),
                        metric_value: metrics_guard.error_rate,
                        threshold: alerting_config.error_rate_threshold as f64,
                        timestamp: _SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                        resolved: false,
                   };
                    
                    let mut alerts_guard = alerts.write()?;
                    alerts_guard.push(alert);
                }
                
                // Check latency
                if metrics_guard.p95_latency_ms > alerting_config.latency_threshold_ms as f64 {
                    let alert = Alert {
                        id: format!("latency_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                        alert_type: _AlertType::HighLatency,
                        severity: _AlertSeverity::Warning,
                        format!("High latency detected: {:.2}ms (P95)", metrics_guard.p95_latency_ms),
                        metric_name: "p95_latency_ms".to_string(),
                        metric_value: metrics_guard.p95_latency_ms,
                        threshold: alerting_config.latency_threshold_ms as f64,
                        timestamp: _SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                        resolved: false,
                   };
                    
                    let mut alerts_guard = alerts.write()?;
                    alerts_guard.push(alert);
                }
                drop(metrics_guard);
                
                // Check cache hit rate
                let cache_guard = cache_metrics.read()?;
                if cache_guard.overall_hit_rate < 0.5 { // Alert if hit rate below 50%
                    let alert = Alert {
                        id: format!("cache_hit_rate_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                        alert_type: _AlertType::LowCacheHitRate,
                        severity: _AlertSeverity::Warning,
                        format!("Low cache hit rate: {:.2}%", cache_guard.overall_hit_rate * 100.0),
                        metric_name: "cache_hit_rate".to_string(),
                        metric_value: cache_guard.overall_hit_rate,
                        threshold: 0.5,
                        timestamp: _SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                        resolved: false,
                   };
                    
                    let mut alerts_guard = alerts.write()?;
                    alerts_guard.push(alert);
                }
            }
        })
        
        Ok(())
    }
    
    /// Start health check background task
    async fn start_health_checks(&self) -> crate::Result<()> {
        let embedding_metrics = self.embedding_metrics.clone();
        let health_config = self.config.health_checks.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(health_config.interval);
            
            loop {
                interval.tick()?;
                
                if health_config.check_models {
                    let metrics_guard = embedding_metrics.read()?;
                    for (model_name, metrics) in metrics_guard.iter() {
                        // Check if model hasn't been used recently
                        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
                        if now - metrics.last_used > 300 && metrics.status == ModelStatus::Ready {
                            debug!("Model {} inactive for {} seconds", model_name, now - metrics.last_used);
                        }
                        
                        // Check for high error rates per model
                        if metrics.requests_processed > 10 {
                            let error_rate = metrics.failed_requests as f64 / metrics.requests_processed as f64;
                            if error_rate > 0.1 { // 10% error rate threshold
                                warn!("Model {} has high error rate: {:.2}%", model_name, error_rate * 100.0);
                            }
                        }
                    }
                }
            }
        })
        
        Ok(())
    }
    
    /// Check for alert conditions
    async fn check_alerts(&self) {
        // This method is called after each request, so we do lightweight checks here
        // Heavy checks are done in the background alerting task
    }
    
    /// Get latency bucket for histogram
    fn get_latency_bucket(latencyms: u64) -> u64 {
        match latency_ms {
            0..=10 => 10,
            11..=50 => 50,
            51..=100 => 100,
            101..=500 => 500,
            501..=1000 => 1000,
            1001..=5000 => 5000,
            _ => 10000,
        }
    }
}

/// Metrics export format for monitoring systems
#[derive(Debug, Serialize)]
pub struct MetricsExport {
    pub multimodal: MultimodalMetrics,
    pub models: HashMap<String, EmbeddingModelMetrics>,
    pub cache: CacheMetrics,
    pub alerts: Vec<Alert>,
    pub export_timestamp: u64,
}

impl MultimodalMetricsCollector {
    /// Export all metrics in a structured format
    pub async fn export_metrics(&self) -> MetricsExport {
        MetricsExport {
            multimodal: self.get_metrics()??,
            models: self.get_all_model_metrics()??,
            cache: self.get_cache_metrics()??,
            alerts: self.get_alerts(Some(100)??, // Last 100 alerts
            export_timestamp: _SystemTime::now()
                .duration_since(UNIX_EPOCH)
                ?
                .as_secs(),
        }
    }
    
    /// Generate Prometheus-format metrics
    pub async fn to_prometheus(&self) -> String {
        let mut output = String::new();
        
        let metrics = self.get_metrics()?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let model_metrics = self.get_all_model_metrics()?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let cache_metrics = self.get_cache_metrics()?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        
        // Overall metrics
        output.push_str(&format!("# HELP multimodal_requests_total Total number of multimodal requests\n");
        output.push_str(&format!("# TYPE multimodal_requests_total counter\n");
        output.push_str(&format!("multimodal_requests_total {}\n", metrics.total_requests);
        
        output.push_str(&format!("# HELP multimodal_error_rate Current error rate\n");
        output.push_str(&format!("# TYPE multimodal_error_rate gauge\n");
        output.push_str(&format!("multimodal_error_rate {}\n", metrics.error_rate);
        
        output.push_str(&format!("# HELP multimodal_latency_avg_ms Average latency in milliseconds\n");
        output.push_str(&format!("# TYPE multimodal_latency_avg_ms gauge\n");
        output.push_str(&format!("multimodal_latency_avg_ms {}\n", metrics.average_latency_ms);
        
        // Model-specific metrics
        for (model_name, model_metrics) in model_metrics {
            output.push_str(&format!(
                "# HELP model_requests_total{{model=\"{}\"}} Total requests for model\n",
                model_name
            );
            output.push_str(&format!(
                "model_requests_total{{model=\"{}\"}} {}\n",
                model_name, model_metrics.requests_processed
            );
            
            output.push_str(&format!(
                "model_latency_avg_ms{{model=\"{}\"}} {}\n",
                model_name, model_metrics.avg_processing_time_ms
            );
        }
        
        // Cache metrics
        output.push_str(&format!("# HELP cache_hit_rate Overall cache hit rate\n");
        output.push_str(&format!("# TYPE cache_hit_rate gauge\n");
        output.push_str(&format!("cache_hit_rate {}\n", cache_metrics.overall_hit_rate);
        
        output
    }
}
