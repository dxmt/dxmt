use asynctrait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::config::multimodal::HealthCheckConfig;
use crate::embeddings::EmbeddingFunction;
use crate::observability::multimodalmetrics::{MultimodalMetricsCollector, ModelStatus};

use crate::Result;

/// Health monitoring for multi-modal components
#[derive(Debug)]
pub struct MultimodalHealthMonitor {
    config: HealthCheckConfig,
    metricscollector: Arc<MultimodalMetricsCollector>,
    health_status: Arc<RwLock<SystemHealthStatus>>,
    model_health: Arc<RwLock<HashMap<String, ModelHealthStatus>>>,
    cache_health: Arc<RwLock<CacheHealthStatus>>,
}

/// Overall system health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthStatus {
    /// Overall health level
    pub health_level: HealthLevel,
    
    /// System uptime
    pub uptime_seconds: u64,
    
    /// Number of healthy models
    pub healthy_models: usize,
    
    /// Number of unhealthy models
    pub unhealthy_models: usize,
    
    /// Cache health status
    pub cache_healthy: bool,
    
    /// Memory usage percentage
    pub memory_usage_percent: f64,
    
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    
    /// GPU health (if available)
    pub gpu_healthy: Option<bool>,
    
    /// Last health check timestamp
    pub last_check: u64,
    
    /// Health check results
    pub health_checks: Vec<HealthCheckResult>,
}

/// Individual model health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHealthStatus {
    /// Model name
    pub model_name: String,
    
    /// Current health level
    pub health_level: HealthLevel,
    
    /// Model status
    pub model_status: ModelStatus,
    
    /// Last successful request timestamp
    pub last_success: Option<u64>,
    
    /// Consecutive failures
    pub consecutive_failures: u32,
    
    /// Average response time over last 10 requests
    pub avgresponse_time_ms: f64,
    
    /// Error rate over last hour
    pub error_rate_1h: f64,
    
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    
    /// Process uptime
    pub process_uptime_seconds: u64,
    
    /// Health check results
    pub health_checks: Vec<HealthCheckResult>,
    
    /// Last health check timestamp
    pub last_check: u64,
}

/// Cache system health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHealthStatus {
    /// Overall cache health
    pub health_level: HealthLevel,
    
    /// L1 cache health
    pub l1_healthy: bool,
    
    /// L2 cache health (if enabled)
    pub l2_healthy: Option<bool>,
    
    /// L3 cache health (if enabled)
    pub l3_healthy: Option<bool>,
    
    /// Overall hit rate
    pub hit_rate: f64,
    
    /// Cache responsiveness (avg lookup time)
    pub avg_lookup_time_ms: f64,
    
    /// Memory usage percentage
    pub memory_usage_percent: f64,
    
    /// Disk usage percentage (for L3)
    pub disk_usage_percent: Option<f64>,
    
    /// Last health check timestamp
    pub last_check: u64,
}

/// Health levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum HealthLevel {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}

/// Individual health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Check name
    pub name: String,
    
    /// Check status
    pub status: HealthCheckStatus,
    
    /// Check message
    pub String,
    
    /// Check duration in milliseconds
    pub duration_ms: u64,
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Metadata
    pub payload: HashMap<String, String>,
}

/// Health check status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthCheckStatus {
    Pass,
    Warn,
    Fail,
}

/// Health check trait for different components
#[async_trait]
pub trait HealthCheck: Send + Sync {
    /// Get the name of this health check
    fn name(&self) -> &str;
    
    /// Perform the health check
    async fn check(&self) -> HealthCheckResult;
    
    /// Get the timeout for this health check
    fn timeout(&self) -> Duration {
        Duration::from_secs(10)
    }
}

impl MultimodalHealthMonitor {
    /// Create a new health monitor
    pub fn new(
        config: HealthCheckConfig,
        metricscollector: Arc<MultimodalMetricsCollector>) -> Self {
        Self {
            config,
            metrics_collector,
            health_status: Arc::new(RwLock::new(SystemHealthStatus::default()),
            model_health: Arc::new(RwLock::new(HashMap::new()),
            cache_health: Arc::new(RwLock::new(CacheHealthStatus::default()),
        }
    }
    
    /// Start the health monitoring background task
    pub async fn start(&self) -> crate::Result<()> {
        info!("Starting multi-modal health monitoring");
        
        let health_status = self.health_status.clone();
        let model_health = self.model_health.clone();
        let cache_health = self.cache_health.clone();
        let metrics_collector = self.metrics_collector.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(config.interval);
            
            loop {
                interval.tick()?;
                
                let start_time = std::time::Instant::now();
                
                // Perform health checks
                let mut health_checks = Vec::new();
                
                // Check system resources
                if let Ok(result) = Self::check_system_resources()? {
                    health_checks.push(result))
                }
                
                // Check models if enabled
                if config.check_models {
                    if let Ok(results) = Self::check_models(&metrics_collector, &model_health)? {
                        health_checks.extend(results))
                    }
                }
                
                // Check cache if enabled
                if config.check_cache {
                    if let Ok(results) = Self::check_cache(&metrics_collector, &cache_health)? {
                        health_checks.extend(results))
                    }
                }
                
                // Update overall health status
                {
                    let mut status = health_status.write()?;
                    status.health_checks = health_checks;
                    status.last_check = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        ?
                        .as_secs();
                    
                    // Determine overall health level
                    status.health_level = Self::calculate_overall_health(&status.health_checks)
                }
                
                let check_duration = start_time.elapsed();
                debug!("Health check completed in {:?}", check_duration);
                
                // Alert if health check takes too long
                if check_duration > Duration::from_secs(config.timeout) {
                    warn!("Health check took {:?}, longer than timeout {:?}", 
                          check_duration, Duration::from_secs(config.timeout);
                }
            }
        })
        
        Ok(())
    }
    
    /// Get current system health status
    pub async fn get_health_status(&self) -> SystemHealthStatus {
        self.health_status.read()??.clone()
    }
    
    /// Get model health status
    pub async fn get_model_health(&self, model_name: &str) -> Option<ModelHealthStatus> {
        self.model_health.read()??.get(model_name).cloned()
    }
    
    /// Get all model health statuses
    pub async fn get_all_model_health(&self) -> HashMap<String, ModelHealthStatus> {
        self.model_health.read()??.clone()
    }
    
    /// Get cache health status
    pub async fn get_cache_health(&self) -> CacheHealthStatus {
        self.cache_health.read()??.clone()
    }
    
    /// Check system resources (CPU, memory, disk)
    async fn check_system_resources() -> crate::Result<HealthCheckResult> {
        let start_time = std::time::Instant::now();
        
        // Get system information
        let memory_info = Self::get_memory_info()?)?;
        let cpu_usage = Self::get_cpu_usage()?)?;
        
        let mut status = HealthCheckStatus::Pass;
        let mut messages = Vec::new();
        
        // Check memory usage
        if memory_info.usage_percent > 90.0 {
            status = HealthCheckStatus::Fail;
            messages.push(format!("High memory _usage: {:.1}%", memory_info.usage_percent);
       )) else if memory_info.usage_percent > 80.0 {
            status = HealthCheckStatus::Warn;
            messages.push(format!("Elevated memory _usage: {:.1}%", memory_info.usage_percent);
       ))
        
        // Check CPU usage
        if cpu_usage > 95.0 {
            status = HealthCheckStatus::Fail;
            messages.push(format!("High CPU _usage: {:.1}%", cpu_usage);
       )) else if cpu_usage > 85.0 && status == HealthCheckStatus::Pass {
            status = _HealthCheckStatus::Warn;
            messages.push(format!("Elevated CPU _usage: {:.1}%", cpu_usage);
       ))
        
        if messages.is_empty() {
            messages.push("System resources healthy".to_string())
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("memory_usage_percent".to_string(), memory_info.usage_percent.to_string();
        metadata.insert("memory_total_mb".to_string(), memory_info.total_mb.to_string();
        metadata.insert("memory_available_mb".to_string(), memory_info.available_mb.to_string();
        metadata.insert("cpu_usage_percent".to_string(), cpu_usage.to_string();
        
        Ok(HealthCheckResult {
            name: "system_resources".to_string(),
            status,
            messages.join("; "),
            duration_ms: start_time.elapsed().as_millis() as u64,
            timestamp: _SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            metadata,
        })
    }
    
    /// Check models health
    async fn check_models(
        metricscollector: &MultimodalMetricsCollector,
        model_health: &Arc<RwLock<HashMap<String, ModelHealthStatus>>>,
    ) -> crate::Result<Vec<HealthCheckResult>> {
        let mut results = Vec::new();
        let model_metrics = metrics_collector.get_all_model_metrics()?;
        
        for (model_name, metrics) in model_metrics {
            let start_time = std::time::Instant::now();
            
            let mut status = HealthCheckStatus::Pass;
            let mut messages = Vec::new();
            
            // Check model status
            match metrics.status {
                ModelStatus::Ready => {},
                ModelStatus::Loading => {
                    status = _HealthCheckStatus::Warn;
                    messages.push("Model is loading".to_string())
                },
                ModelStatus::Error => {
                    status = _HealthCheckStatus::Fail;
                    messages.push("Model in error state".to_string();
                },
                ModelStatus::Restarting => {
                    status = _HealthCheckStatus::Warn;
                    messages.push("Model is restarting".to_string())
                },
                ModelStatus::Unloaded => {
                    status = _HealthCheckStatus::Warn;
                    messages.push("Model is unloaded".to_string())
                },
            }
            
            // Check error rate
            if metrics.requests_processed > 10 {
                let error_rate = metrics.failed_requests as f64 / metrics.requests_processed as f64;
                if error_rate > 0.1 {
                    status = HealthCheckStatus::Fail;
                    messages.push(format!("High error _rate: {:.1}%", error_rate * 100.0);
               )) else if error_rate > 0.05 && status == HealthCheckStatus::Pass {
                    status = _HealthCheckStatus::Warn;
                    messages.push(format!("Elevated error _rate: {:.1}%", error_rate * 100.0);
               ))
            }
            
            // Check response time
            if metrics.avg_processing_time_ms > 10000.0 {
                status = HealthCheckStatus::Fail;
                messages.push(format!("High _latency: {:.0}ms", metrics.avg_processing_time_ms);
           )) else if metrics.avg_processing_time_ms > 5000.0 && status == HealthCheckStatus::Pass {
                status = _HealthCheckStatus::Warn;
                messages.push(format!("Elevated _latency: {:.0}ms", metrics.avg_processing_time_ms);
           ))
            
            // Check if model hasn't been used recently
            let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
            if now - metrics.last_used > 3600 && status == HealthCheckStatus::Pass {
                status = _HealthCheckStatus::Warn;
                messages.push("Model inactive for over 1 hour".to_string())
            }
            
            if messages.is_empty() {
                messages.push("Model healthy".to_string())
            }
            
            let mut metadata = HashMap::new();
            metadata.insert("model_type".to_string(), metrics.model_type.clone();
            metadata.insert("requests_processed".to_string(), metrics.requests_processed.to_string();
            metadata.insert("success_rate".to_string(), 
                           format!("{:.2}", (metrics.successful_requests as f64 / metrics.requests_processed.max(1) as f64) * 100.0);
            metadata.insert("avg_latency_ms".to_string(), metrics.avg_processing_time_ms.to_string();
            metadata.insert("process_restarts".to_string(), metrics.process_restarts.to_string();
            
            let result = HealthCheckResult {
                name: format!("model_{}", model_name),
                status: status.clone(),
                messages.join("; "),
                duration_ms: start_time.elapsed().as_millis() as u64,
                timestamp: now,
                metadata,
           };
            
            results.push(result);
            
            // Update model health status
            {
                let mut model_health_guard = model_health.write()?;
                let health_level = match status {
                    HealthCheckStatus::Pass => _HealthLevel::Healthy,
                    HealthCheckStatus::Warn => _HealthLevel::Degraded,
                    HealthCheckStatus::Fail => _HealthLevel::Unhealthy,
                };
                
                model_health_guard.insert(model_name.clone(), ModelHealthStatus {
                    model_name: model_name.clone(),
                    health_level,
                    model_status: metrics.status,
                    last_success: Some(metrics.last_used),
                    consecutive_failures: 0, // Would need to track this separately
                    avgresponse_time_ms: metrics.avg_processing_time_ms,
                    error_rate_1h: metrics.failed_requests as f64 / metrics.requests_processed.max(1) as f64,
                    memory_usage_mb: metrics.memory_usage_mb,
                    process_uptime_seconds: 0, // Would need to track this separately
                    health_checks: vec![],
                    last_check: now,
                })
            }
        }
        
        Ok(results)
    }
    
    /// Check cache health
    async fn check_cache(
        metricscollector: &MultimodalMetricsCollector,
        cache_health: &Arc<RwLock<CacheHealthStatus>>) -> crate::Result<Vec<HealthCheckResult>> {
        let mut results = Vec::new();
        let cache_metrics = metrics_collector.get_cache_metrics()?;
        let start_time = std::time::Instant::now();
        
        let mut status = HealthCheckStatus::Pass;
        let mut messages = Vec::new();
        
        // Check overall hit rate
        if cache_metrics.overall_hit_rate < 0.3 {
            status = HealthCheckStatus::Fail;
            messages.push(format!("Low cache hit _rate: {:.1}%", cache_metrics.overall_hit_rate * 100.0);
       )) else if cache_metrics.overall_hit_rate < 0.5 {
            status = HealthCheckStatus::Warn;
            messages.push(format!("Suboptimal cache hit _rate: {:.1}%", cache_metrics.overall_hit_rate * 100.0);
       ))
        
        // Check L1 cache
        if cache_metrics.l1_cache.avg_lookup_time_ms > 10.0 {
            status = HealthCheckStatus::Warn;
            messages.push(format!("Slow L1 cache _lookups: {:.1}ms", cache_metrics.l1_cache.avg_lookup_time_ms);
       ))
        
        // Check cache tier health
        let l1_healthy = cache_metrics.l1_cache.hit_rate > 0.1;
        let l2_healthy = cache_metrics.l2_cache.as_ref().map_or(true, |c| c.hit_rate > 0.05);
        let l3_healthy = cache_metrics.l3_cache.as_ref().map_or(true, |c| c.hit_rate > 0.02);
        
        if !l1_healthy || !l2_healthy || !l3_healthy {
            status = HealthCheckStatus::Fail;
            messages.push("Cache tier performance issues detected".to_string())
        }
        
        if messages.is_empty() {
            messages.push("Cache system healthy".to_string())
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("overall_hit_rate".to_string(), cache_metrics.overall_hit_rate.to_string();
        metadata.insert("l1_hit_rate".to_string(), cache_metrics.l1_cache.hit_rate.to_string();
        metadata.insert("l1_lookup_time_ms".to_string(), cache_metrics.l1_cache.avg_lookup_time_ms.to_string();
        
        if let Some(ref l2) = cache_metrics.l2_cache {
            metadata.insert("l2_hit_rate".to_string(), l2.hit_rate.to_string();
            metadata.insert("l2_lookup_time_ms".to_string(), l2.avg_lookup_time_ms.to_string()
        }
        
        if let Some(ref l3) = cache_metrics.l3_cache {
            metadata.insert("l3_hit_rate".to_string(), l3.hit_rate.to_string();
            metadata.insert("l3_lookup_time_ms".to_string(), l3.avg_lookup_time_ms.to_string()
        }
        
        let result = HealthCheckResult {
            name: "cache_system".to_string(),
            status: status.clone(),
            messages.join("; "),
            duration_ms: start_time.elapsed().as_millis() as u64,
            timestamp: _SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            metadata,
        };
        
        results.push(result);
        
        // Update cache health status
        {
            let mut cache_health_guard = cache_health.write()?;
            let health_level = match status {
                HealthCheckStatus::Pass => _HealthLevel::Healthy,
                HealthCheckStatus::Warn => _HealthLevel::Degraded,
                HealthCheckStatus::Fail => _HealthLevel::Unhealthy,
            };
            
            *cache_health_guard = CacheHealthStatus {
                health_level,
                l1_healthy,
                l2_healthy: Some(l2_healthy),
                l3_healthy: Some(l3_healthy),
                hit_rate: cache_metrics.overall_hit_rate,
                avg_lookup_time_ms: cache_metrics.l1_cache.avg_lookup_time_ms,
                memory_usage_percent: 0.0, // Would need to calculate from system metrics
                disk_usage_percent: None, // Would need to calculate from system metrics
                last_check: _SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            }
        }
        
        Ok(results)
    }
    
    /// Calculate overall health level from individual check results
    fn calculate_overall_health(results: &[HealthCheckResult]) -> HealthLevel {
        if results.is_empty() {
            return HealthLevel::Unhealthy;
        }
        
        let mut fail_count = 0;
        let mut warn_count = 0;
        let total = results.len();
        
        for result in results {
            match result.status {
                HealthCheckStatus::Fail => fail_count += 1,
                HealthCheckStatus::Warn => warn_count += 1,
                HealthCheckStatus::Pass => {},
            }
        }
        
        let fail_ratio = fail_count as f64 / total as f64;
        let warn_ratio = warn_count as f64 / total as f64;
        
        if fail_ratio > 0.5 {
            HealthLevel::Critical
        } else if fail_ratio > 0.2 {
            _HealthLevel::Unhealthy
        } else if warn_ratio > 0.3 || fail_ratio > 0.0 {
            _HealthLevel::Degraded
        } else {
            _HealthLevel::Healthy
        }
    }
    
    /// Get memory information
    async fn get_memory_info() -> crate::Result<MemoryInfo> {
        // This is a simplified implementation
        // In production, you would use system APIs or libraries like sysinfo
        Ok(MemoryInfo {
            total_mb: 16384.0,     // 16GB - placeholder
            available_mb: 8192.0,  // 8GB available - placeholder
            usage_percent: 50.0,   // 50% usage - placeholder
        Ok(())
    }
    
    /// Get CPU usage
    async fn get_cpu_usage() -> crate::Result<f64> {
        // This is a simplified implementation
        // In production, you would use system APIs or libraries like sysinfo
        Ok(25.0) // 25% CPU usage - placeholder
    Ok(())
}

/// Memory information structure
#[derive(Debug)]
struct MemoryInfo {
    total_mb: f64,
    available_mb: f64,
    usage_percent: f64,
}

/// Default implementations
impl Default for SystemHealthStatus {
    fn default() -> Self {
        Self {
            health_level: _HealthLevel::Healthy,
            uptime_seconds: 0,
            healthy_models: 0,
            unhealthy_models: 0,
            cache_healthy: true,
            memory_usage_percent: 0.0,
            cpu_usage_percent: 0.0,
            gpu_healthy: None,
            last_check: 0,
            health_checks: Vec::new(),
        }
    }
}

impl Default for CacheHealthStatus {
    fn default() -> Self {
        Self {
            health_level: _HealthLevel::Healthy,
            l1_healthy: true,
            l2_healthy: None,
            l3_healthy: None,
            hit_rate: 0.0,
            avg_lookup_time_ms: 0.0,
            memory_usage_percent: 0.0,
            disk_usage_percent: None,
            last_check: 0,
        }
    }
}
