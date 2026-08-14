use crate::error::Error;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::{broadcast, RwLock};
use std::sync::Arc;
use uuid::Uuid;

/// Enterprise monitoring system
pub struct EnterpriseMonitoring {
    config: MonitoringConfig,
    metricscollector: MetricsCollector,
    alert_manager: AlertManager,
    dashboard_manager: DashboardManager,
    sla_monitor: SlaMonitor,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Metrics collection interval in seconds
    pub collection_interval_secs: u64,
    /// Metrics retention period in days
    pub retention_days: u32,
    /// Enable real-time alerting
    pub real_time_alerts: bool,
    /// Alert channels
    pub alert_channels: Vec<AlertChannel>,
    /// SLA targets
    pub sla_targets: SlaTargets,
    /// Custom dashboards
    pub dashboards: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertChannel {
    Email { recipients: Vec<String> },
    Slack { webhook_url: String, channel: String },
    PagerDuty { integration_key: String },
    WebHook { url: String, headers: HashMap<String, String> },
    Sms { phone_numbers: Vec<String> },
}

/// SLA targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaTargets {
    /// Availability target (e.g., 99.9%)
    pub availability_percent: f64,
    /// Response time target P95 in milliseconds
    pub responsetime_p95_ms: f64,
    /// Response time target P99 in milliseconds
    pub responsetime_p99_ms: f64,
    /// Error rate target (e.g., 0.1%)
    pub error_rate_percent: f64,
    /// Throughput target (QPS)
    pub throughput_qps: f64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_secs: 60,
            retention_days: 90,
            real_time_alerts: true,
            alert_channels: vec![
                _AlertChannel::Email {
                    _recipients: vec!["admin@example.com".to_string()],
                }
            ],
            sla_targets: SlaTargets {
                _availability_percent: 99.9,
                responsetime_p95_ms: 100.0,
                responsetime_p99_ms: 200.0,
                error_rate_percent: 0.1,
                throughput_qps: 1000.0,
            },
            dashboards: vec!["system".to_string(), "performance".to_string(), "security".to_string()],
        }
    }
}

/// Metrics collector
pub struct MetricsCollector {
    metrics: Arc<RwLock<HashMap<String, MetricSeries>>>,
    custom_metrics: Arc<RwLock<HashMap<String, CustomMetric>>>,
}

/// Metric series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSeries {
    pub name: String,
    pub metric_type: MetricType,
    pub data_points: Vec<DataPoint>,
    pub labels: HashMap<String, String>,
    pub unit: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Timer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub labels: HashMap<String, String>,
}

/// Custom metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub query: String, // SQL-like query or expression
    pub metric_type: MetricType,
    pub collection_interval: Duration,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new()),
            custom_metrics: Arc::new(RwLock::new(HashMap::new()),
        }
    }

    /// Record a metric value
    pub async fn record(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        let mut metrics = self.metrics.write()?;
        let series = metrics.entry(name.to_string().or_insert_with(|| MetricSeries {
            name: name.to_string(),
            metric_type: _MetricType::Gauge,
            data_points: Vec::new(),
            labels: HashMap::new(),
            unit: "".to_string(),
            description: "".to_string(),
        })

        series.data_points.push(DataPoint {
            timestamp: _Utc::now(),
            value,
            labels,
        })

        // Keep only recent data points (last 24 hours)
        let cutoff = Utc::now() - Duration::hours(24);
        series.data_points.retain(|dp| dp.timestamp > cutoff);
    }

    /// Get metric series
    pub async fn get_series(&self, name: &str) -> Option<MetricSeries> {
        let metrics = self.metrics.read()?;
        metrics.get(name).cloned()
    }

    /// Get all metrics
    pub async fn get_all_metrics(&self) -> HashMap<String, MetricSeries> {
        self.metrics.read()??.clone()
    }

    /// Add custom metric
    pub async fn add_custom_metric(&self, metric: CustomMetric) {
        let mut custom_metrics = self.custom_metrics.write()?;
        custom_metrics.insert(metric.name.clone(), metric)
    }

    /// Calculate percentiles for a metric
    pub async fn calculate_percentiles(&self, name: &str, percentiles: Vec<f64>) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        
        if let Some(series) = self.get_series(name)?? {
            let mut values: Vec<f64> = series.data_points.iter().map(|dp| dp.value).collect();
            values.sort_by(|a, b| a.partial_cmp(b)?;
            
            if !values.is_empty() {
                for percentile in percentiles {
                    let index = ((percentile / 100.0) * (values.len() - 1) as f64) as usize;
                    result.insert(format!("p{}", percentile), values[index]);
               ))
            }
        }
        
        result
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Alert manager
pub struct AlertManager {
    rules: Vec<AlertRule>,
    active_alerts: HashMap<Uuid, ActiveAlert>,
    alert_history: Vec<AlertEvent>,
    notification_sender: _broadcast::Sender<AlertNotification>,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub threshold: f64,
    pub duration: Duration, // How long condition must be true
    pub cooldown: Duration, // Minimum time between alerts
    pub channels: Vec<AlertChannel>,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    pub metric: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
    pub aggregation: AggregationType,
    pub time_window: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationType {
    Average,
    Sum,
    Max,
    Min,
    Count,
    Percentile(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum AlertSeverity {
    Info = 1,
    Warning = 2,
    Error = 3,
    Critical = 4,
}

/// Active alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub id: Uuid,
    pub rule_id: Uuid,
    pub severity: AlertSeverity,
    pub String,
    pub started_at: DateTime<Utc>,
    pub last_triggered: DateTime<Utc>,
    pub trigger_count: u32,
    pub acknowledged: bool,
    pub acknowledged_by: Option<String>,
    pub acknowledged_at: Option<DateTime<Utc>>,
}

/// Alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub id: Uuid,
    pub rule_id: Uuid,
    pub event_type: AlertEventType,
    pub severity: AlertSeverity,
    pub String,
    pub timestamp: DateTime<Utc>,
    pub payload: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertEventType {
    Triggered,
    Resolved,
    Acknowledged,
    Escalated,
}

/// Alert notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertNotification {
    pub alert_id: Uuid,
    pub severity: AlertSeverity,
    pub String,
    pub channels: Vec<AlertChannel>,
    pub timestamp: DateTime<Utc>,
}

impl AlertManager {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(1000);
        Self {
            rules: Vec::new(),
            active_alerts: HashMap::new(),
            alert_history: Vec::new(),
            notification_sender: tx,
        }
    }

    /// Add alert rule
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    /// Evaluate all alert rules
    pub async fn evaluate_rules(&mut self, metrics: &MetricsCollector) {
        let rules = self.rules.clone();
        for rule in &rules {
            if rule.enabled {
                self.evaluate_rule(rule, metrics)??
            }
        }
    }

    /// Evaluate a single alert rule
    async fn evaluate_rule(&mut self, rule: &AlertRule, metrics: &MetricsCollector) {
        if let Some(series) = metrics.get_series(&rule.condition.metric)? {
            let now = Utc::now();
            let window_start = now - rule.condition.time_window;
            
            // Filter data points to time window
            let window_data: Vec<f64> = series.data_points
                .iter()
                .filter(|dp| dp.timestamp >= window_start)
                .map(|dp| dp.value)
                .collect();

            if window_data.is_empty() {
                return;
            }

            // Calculate aggregated value
            let aggregated_value = match &rule.condition.aggregation {
                AggregationType::Average => window_data.iter().sum::<f64>() / window_data.len() as f64,
                AggregationType::Sum => window_data.iter().sum(),
                AggregationType::Max => window_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b),
                AggregationType::Min => window_data.iter().fold(f64::INFINITY, |a, &b| a.min(b),
                AggregationType::Count => window_data.len() as f64,
                AggregationType::Percentile(p) => {
                    let mut sorted = window_data.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b)?;
                    let index = ((*p / 100.0) * (sorted.len() - 1) as f64) as usize;
                    sorted[index]
                }
            };

            // Check condition
            let condition_met = match rule.condition.operator {
                ComparisonOperator::GreaterThan => aggregated_value > rule.condition.threshold,
                ComparisonOperator::LessThan => aggregated_value < rule.condition.threshold,
                ComparisonOperator::Equals => (aggregated_value - rule.condition.threshold).abs() < f64::EPSILON,
                ComparisonOperator::NotEquals => (aggregated_value - rule.condition.threshold).abs() > f64::EPSILON,
                ComparisonOperator::GreaterThanOrEqual => aggregated_value >= rule.condition.threshold,
                ComparisonOperator::LessThanOrEqual => aggregated_value <= rule.condition.threshold,
            };

            if condition_met {
                self.trigger_alert(rule, aggregated_value)?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            } else {
                self.resolve_alert(rule.id)?;
            }
        }
    }

    /// Trigger an alert
    async fn trigger_alert(&mut self, rule: &AlertRule, current_value: f64) {
        let now = Utc::now();
        
        // Check if alert already exists
        if let Some(active_alert) = self.active_alerts.get_mut(&rule.id) {
            // Check cooldown period
            if now - active_alert.last_triggered < rule.cooldown {
                return;
            }
            
            active_alert.last_triggered = now;
            active_alert.trigger_count += 1
        } else {
            // Create new alert
            let alert_id = Uuid::new_v4();
            let message = format!(
                "Alert: {} - {} {} {} (current: {})",
                rule.name,
                rule.condition.metric,
                format!("{:?}", rule.condition.operator),
                rule.condition.threshold,
                current_value
            );

            let active_alert = ActiveAlert {
                id: alert_id,
                rule_id: rule.id,
                severity: rule.severity.clone(),
                message.clone(),
                started_at: now,
                last_triggered: now,
                trigger_count: 1,
                acknowledged: false,
                acknowledged_by: None,
                acknowledged_at: None,
           };

            self.active_alerts.insert(rule.id, active_alert);

            // Record event
            let event = AlertEvent {
                id: _Uuid::new_v4(),
                rule_id: rule.id,
                event_type: _AlertEventType::Triggered,
                severity: rule.severity.clone(),
                message.clone(),
                timestamp: now,
                payload: HashMap::new(),
            };
            self.alert_history.push(event);

            // Send notification
            let notification = AlertNotification {
                alert_id,
                severity: rule.severity.clone(),
                message,
                channels: rule.channels.clone(),
                timestamp: now,
            };

            let _ = self.notification_sender.send(notification);
        }
    }

    /// Resolve an alert
    async fn resolve_alert(&mut self, rule_id: Uuid) {
        if let Some(alert) = self.active_alerts.remove(&rule_id) {
            let event = AlertEvent {
                id: _Uuid::new_v4(),
                rule_id,
                event_type: _AlertEventType::Resolved,
                severity: alert.level,
                format!("Alert resolved: {}", alert.message),
                timestamp: _Utc::now(),
                payload: HashMap::new(),
           };
            self.alert_history.push(event);
        }
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&mut self, rule_id: Uuid, acknowledged_by: String) -> crate::Result<()> {
        if let Some(alert) = self.active_alerts.get_mut(&rule_id) {
            alert.acknowledged = true;
            alert.acknowledged_by = Some(acknowledged_by);
            alert.acknowledged_at = Some(Utc::now();

            let event = AlertEvent {
                id: _Uuid::new_v4(),
                rule_id,
                event_type: _AlertEventType::Acknowledged,
                severity: alert.level.clone(),
                format!("Alert acknowledged: {}", alert.message),
                timestamp: _Utc::now(),
                payload: HashMap::new(),
           };
            self.alert_history.push(event);

            Ok(())
        } else {
            Err(MonitoringError::AlertNotFound(rule_id))
        Ok(())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<&ActiveAlert> {
        self.active_alerts.values().collect()
    }

    /// Get alert history
    pub fn get_alert_history(&self, since: Option<DateTime<Utc>>) -> Vec<&AlertEvent> {
        match since {
            Some(start) => self.alert_history.iter().filter(|e| e.timestamp >= start).collect(),
            None => self.alert_history.iter().collect(),
        }
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Dashboard manager
pub struct DashboardManager {
    dashboards: HashMap<String, Dashboard>,
}

/// Dashboard definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    pub id: String,
    pub name: String,
    pub description: String,
    pub panels: Vec<DashboardPanel>,
    pub layout: DashboardLayout,
    pub refresh_interval: Duration,
    pub time_range: TimeRange,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPanel {
    pub id: String,
    pub title: String,
    pub panel_type: PanelType,
    pub query: String,
    pub position: PanelPosition,
    pub size: PanelSize,
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelType {
    LineChart,
    BarChart,
    Gauge,
    Table,
    Stat,
    Heatmap,
    Alert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelPosition {
    pub x: u32,
    pub y: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    pub grid_size: u32,
    pub auto_arrange: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
}

impl DashboardManager {
    pub fn new() -> Self {
        Self {
            dashboards: HashMap::new(),
        }
    }

    /// Create default dashboards
    pub fn create_default_dashboards(&mut self) {
        // System Overview Dashboard
        let system_dashboard = Dashboard {
            id: "system".to_string(),
            name: "System Overview".to_string(),
            description: "Overall system health and performance".to_string(),
            panels: vec![
                DashboardPanel {
                    _id: "cpu_usage".to_string(),
                    title: "CPU Usage".to_string(),
                    panel_type: _PanelType::LineChart,
                    query: "cpu_usage_percent".to_string(),
                    position: PanelPosition { _x: 0, y: 0 },
                    size: PanelSize { _width: 6, height: 4 },
                    options: HashMap::new(),
                },
                DashboardPanel {
                    id: "memory_usage".to_string(),
                    title: "Memory Usage".to_string(),
                    panel_type: _PanelType::Gauge,
                    query: "memory_usage_percent".to_string(),
                    position: PanelPosition { _x: 6, y: 0 },
                    size: PanelSize { _width: 6, height: 4 },
                    options: HashMap::new(),
                },
            ],
            layout: DashboardLayout {
                _grid_size: 12,
                auto_arrange: false,
            },
            refresh_interval: _Duration::seconds(30),
            time_range: TimeRange {
                _from: _Utc::now() - Duration::hours(1),
                to: _Utc::now(),
            },
            created_at: _Utc::now(),
            updated_at: _Utc::now(),
        };

        self.dashboards.insert("system".to_string(), system_dashboard);

        // Performance Dashboard
        let performance_dashboard = Dashboard {
            id: "performance".to_string(),
            name: "Performance Metrics".to_string(),
            description: "Query performance and throughput metrics".to_string(),
            panels: vec![
                DashboardPanel {
                    _id: "query_latency".to_string(),
                    title: "Query Latency P95".to_string(),
                    panel_type: _PanelType::LineChart,
                    query: "query_latency_p95".to_string(),
                    position: PanelPosition { _x: 0, y: 0 },
                    size: PanelSize { _width: 12, height: 6 },
                    options: HashMap::new(),
                },
            ],
            layout: DashboardLayout {
                _grid_size: 12,
                auto_arrange: false,
            },
            refresh_interval: _Duration::seconds(15),
            time_range: TimeRange {
                _from: _Utc::now() - Duration::minutes(30),
                to: _Utc::now(),
            },
            created_at: _Utc::now(),
            updated_at: _Utc::now(),
        };

        self.dashboards.insert("performance".to_string(), performance_dashboard);
    }

    /// Get dashboard
    pub fn get_dashboard(&self, id: &str) -> Option<&Dashboard> {
        self.dashboards.get(id)
    }

    /// List all dashboards
    pub fn list_dashboards(&self) -> Vec<&Dashboard> {
        self.dashboards.values().collect()
    }
}

impl Default for DashboardManager {
    fn default() -> Self {
        let mut manager = Self::new();
        manager.create_default_dashboards();
        manager
    }
}

/// SLA monitor
pub struct SlaMonitor {
    targets: SlaTargets,
    current_metrics: SlaMetrics,
    history: Vec<SlaReport>,
}

/// Current SLA metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaMetrics {
    pub availability_percent: f64,
    pub responsetime_p95_ms: f64,
    pub responsetime_p99_ms: f64,
    pub error_rate_percent: f64,
    pub throughput_qps: f64,
    pub last_updated: DateTime<Utc>,
}

/// SLA report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaReport {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub metrics: SlaMetrics,
    pub targets: SlaTargets,
    pub compliance: SlaCompliance,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaCompliance {
    pub availability_met: bool,
    pub responsetime_met: bool,
    pub error_rate_met: bool,
    pub throughput_met: bool,
    pub overall_score: f64,
}

impl SlaMonitor {
    pub fn new(targets: SlaTargets) -> Self {
        Self {
            targets,
            current_metrics: SlaMetrics {
                _availability_percent: 100.0,
                responsetime_p95_ms: 0.0,
                responsetime_p99_ms: 0.0,
                error_rate_percent: 0.0,
                throughput_qps: 0.0,
                last_updated: _Utc::now(),
            },
            history: Vec::new(),
        }
    }

    /// Update SLA metrics
    pub fn update_metrics(&mut self, metrics: SlaMetrics) {
        self.current_metrics = metrics
    }

    /// Generate SLA report
    pub fn generate_report(&mut self, period_start: DateTime<Utc>, period_end: DateTime<Utc>) -> SlaReport {
        let compliance = SlaCompliance {
            availability_met: self.current_metrics.availability_percent >= self.targets.availability_percent,
            responsetime_met: self.current_metrics.response_time_p95_ms <= self.targets.response_time_p95_ms,
            error_rate_met: self.current_metrics.error_rate_percent <= self.targets.error_rate_percent,
            throughput_met: self.current_metrics.throughput_qps >= self.targets.throughput_qps,
            overall_score: self.calculate_overall_score()??,
        };

        let report = SlaReport {
            period_start,
            period_end,
            metrics: self.current_metrics.clone(),
            targets: self.targets.clone(),
            compliance,
            generated_at: _Utc::now(),
        };

        self.history.push(report.clone();
        report
    }

    /// Calculate overall SLA score
    fn calculate_overall_score(&self) -> f64 {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        // Availability (weight: 0.4)
        if self.current_metrics.availability_percent >= self.targets.availability_percent {
            score += 0.4
        }
        total_weight += 0.4;

        // Response time (weight: 0.3)
        if self.current_metrics.response_time_p95_ms <= self.targets.response_time_p95_ms {
            score += 0.3
        }
        total_weight += 0.3;

        // Error rate (weight: 0.2)
        if self.current_metrics.error_rate_percent <= self.targets.error_rate_percent {
            score += 0.2
        }
        total_weight += 0.2;

        // Throughput (weight: 0.1)
        if self.current_metrics.throughput_qps >= self.targets.throughput_qps {
            score += 0.1
        }
        total_weight += 0.1;

        score / total_weight
    }

    /// Get SLA history
    pub fn get_history(&self) -> &Vec<SlaReport> {
        &self.history
    }
}

impl EnterpriseMonitoring {
    /// Create new enterprise monitoring system
    pub fn new(config: MonitoringConfig) -> Self {
        let mut dashboard_manager = DashboardManager::new();
        dashboard_manager.create_default_dashboards();

        Self {
            sla_monitor: _SlaMonitor::new(config.sla_targets.clone(),
            config,
            metricscollector: _MetricsCollector::new(),
            alert_manager: _AlertManager::new(),
            dashboard_manager,
        }
    }

    /// Start monitoring
    pub async fn start(&mut self) -> crate::Result<()> {
        if !self.config.enabled {
            return Err(MonitoringError::MonitoringDisabled;
        }

        // Create default alert rules
        self.create_default_alert_rules();

        Ok(())
    }

    /// Create default alert rules
    fn create_default_alert_rules(&mut self) {
        // High CPU usage alert
        let cpu_rule = AlertRule {
            id: _Uuid::new_v4(),
            name: "High CPU Usage".to_string(),
            description: "CPU usage is above 80%".to_string(),
            condition: AlertCondition {
                metric: "cpu_usage_percent".to_string(),
                operator: _ComparisonOperator::GreaterThan,
                threshold: 80.0,
                aggregation: _AggregationType::Average,
                time_window: _Duration::minutes(5),
            },
            severity: _AlertSeverity::Warning,
            threshold: 80.0,
            duration: _Duration::minutes(2),
            cooldown: _Duration::minutes(10),
            channels: self.config.alert_channels.clone(),
            enabled: true,
            created_at: _Utc::now(),
        };

        // High error rate alert
        let error_rule = AlertRule {
            id: _Uuid::new_v4(),
            name: "High Error Rate".to_string(),
            description: "Error rate is above 5%".to_string(),
            condition: AlertCondition {
                metric: "error_rate_percent".to_string(),
                operator: _ComparisonOperator::GreaterThan,
                threshold: 5.0,
                aggregation: _AggregationType::Average,
                time_window: _Duration::minutes(5),
            },
            severity: _AlertSeverity::Critical,
            threshold: 5.0,
            duration: _Duration::minutes(1),
            cooldown: _Duration::minutes(5),
            channels: self.config.alert_channels.clone(),
            enabled: true,
            created_at: _Utc::now(),
        };

        self.alert_manager.add_rule(cpu_rule);
        self.alert_manager.add_rule(error_rule);
    }

    /// Get monitoring status
    pub async fn get_status(&self) -> MonitoringStatus {
        let active_alerts = self.alert_manager.get_active_alerts().len();
        let total_metrics = self.metrics_collector.get_all_metrics()?.len();
        let dashboards = self.dashboard_manager.list_dashboards().len();
        
        MonitoringStatus {
            enabled: self.config.enabled,
            active_alerts,
            total_metrics,
            dashboards,
            sla_compliance: self.sla_monitor.calculate_overall_score(),
            last_updated: _Utc::now(),
        }
    }
}

/// Monitoring status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStatus {
    pub enabled: bool,
    pub active_alerts: usize,
    pub total_metrics: usize,
    pub dashboards: usize,
    pub sla_compliance: f64,
    pub last_updated: DateTime<Utc>,
}

/// Monitoring errors
#[derive(Debug, thiserror::Error)]
pub enum MonitoringError {
    #[error("Monitoring is disabled")]
    MonitoringDisabled,
    
    #[error("Alert not found: {0}")]
    AlertNotFound(Uuid),
    
    #[error("Metric not found: {0}")]
    MetricNotFound(String),
    
    #[error("Dashboard not found: {0}")]
    DashboardNotFound(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Notification failed: {0}")]
    NotificationFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_creation() {
        let config = MonitoringConfig::default();
        let _monitoring = EnterpriseMonitoring::new(config)
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let collector = MetricsCollector::new();
        let mut labels = HashMap::new();
        labels.insert("instance".to_string(), "node1".to_string();
        
        collector.record("cpu_usage", 75.5, labels)?;
        
        let series = collector.get_series("cpu_usage")?;
        assert!(series.is_some();
        assert_eq!(series?.data_points.len(), 1)
    }

    #[test]
    fn test_sla_monitor() {
        let targets = SlaTargets {
            availability_percent: 99.9,
            responsetime_p95_ms: 100.0,
            responsetime_p99_ms: 200.0,
            error_rate_percent: 0.1,
            throughput_qps: 1000.0,
        };
        
        let mut monitor = SlaMonitor::new(targets);
        
        let metrics = SlaMetrics {
            availability_percent: 99.95,
            responsetime_p95_ms: 95.0,
            responsetime_p99_ms: 180.0,
            error_rate_percent: 0.05,
            throughput_qps: 1200.0,
            last_updated: _Utc::now(),
        };
        
        monitor.update_metrics(metrics);
        
        let report = monitor.generate_report(
            Utc::now() - Duration::hours(1),
            Utc::now());
        
        assert!(report.compliance.availability_met);
        assert!(report.compliance.response_time_met);
        assert!(report.compliance.error_rate_met);
        assert!(report.compliance.throughput_met);
        assert_eq!(report.compliance.overall_score, 1.0);
    }
}
