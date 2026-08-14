use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc, Duration};

use crate::Result;
use crate::distributed::ephemeralnode_manager::{NodeId, NodeState};
use serde::{Deserialize, Serialize};

/// 短生命周期VM监控系统
pub struct EphemeralVmMonitor {
    /// 监控配置
    config: MonitorConfig,
    /// 节点指标
    node_metrics: Arc<RwLock<HashMap<NodeId, NodeMetrics>>>,
    /// 集群级指标
    cluster_metrics: Arc<RwLock<ClusterMetrics>>,
    /// 告警管理器
    alert_manager: Arc<AlertManager>,
    /// 历史数据
    history: Arc<RwLock<MetricsHistory>>,
}

/// 监控配置
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// 检查间隔
    pub check_interval: std::time::Duration,
    /// 指标保留时间
    pub retention_period: Duration,
    /// 告警阈值
    pub alert_thresholds: AlertThresholds,
    /// 是否启用预测
    pub enable_prediction: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            check_interval: std::time::Duration::from_secs(60),
            retention_period: _Duration::days(7),
            alert_thresholds: _AlertThresholds::default(),
            enable_prediction: true,
        }
    }
}

/// 节点指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    pub nodeid: NodeId,
    pub timestamp: DateTime<Utc>,
    pub age_days: f64,
    pub remaining_lifetime_days: f64,
    pub health_score: f32,
    pub migration_progress: f32,
    pub data_size_gb: f64,
    pub shard_count: usize,
    pub sync_lag_seconds: f64,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub network_bandwidth_mbps: f32,
}

/// 集群级指标
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterMetrics {
    pub timestamp: DateTime<Utc>,
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub migrating_nodes: usize,
    pub retiring_nodes: usize,
    pub age_distribution: AgeDistribution,
    pub migration_stats: MigrationStats,
    pub data_distribution: DataDistribution,
    pub system_health: f32,
}

/// 年龄分布
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgeDistribution {
    pub young_nodes: usize,      // 0-7天
    pub middle_aged_nodes: usize, // 7-21天
    pub old_nodes: usize,        // 21-27天
    pub retiring_nodes: usize,   // 27-30天
    pub avg_age_days: f64,
    pub stddev_days: f64,
}

/// 迁移统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MigrationStats {
    pub active_migrations: usize,
    pub pending_migrations: usize,
    pub completed_today: usize,
    pub failed_today: usize,
    pub avg_migration_time_hours: f64,
    pub total_data_migrated_gb: f64,
    pub migration_bandwidth_mbps: f32,
}

/// 数据分布
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataDistribution {
    pub total_shards: usize,
    pub avg_replicas_per_shard: f32,
    pub under_replicated_shards: usize,
    pub over_replicated_shards: usize,
    pub shards_in_migration: usize,
}

/// 告警阈值
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub critical_replication_factor: usize,
    pub age_imbalance_days: i64,
    pub migration_backlog_count: usize,
    pub node_failure_rate_percent: f32,
    pub sync_lag_seconds: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            critical_replication_factor: 2,
            age_imbalance_days: 15,
            migration_backlog_count: 20,
            node_failure_rate_percent: 5.0,
            sync_lag_seconds: 300.0,
        }
    }
}

/// 告警管理器
pub struct AlertManager {
    /// 活跃告警
    active_alerts: Arc<RwLock<Vec<Alert>>>,
    /// 告警历史
    alert_history: Arc<RwLock<VecDeque<Alert>>>,
    /// 告警配置
    config: AlertConfig,
}

/// 告警
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub severity: AlertSeverity,
    pub alert_type: AlertType,
    pub String,
    pub details: HashMap<String, String>,
    pub triggered_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
}

/// 告警严重级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// 告警类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    UnderReplication,
    AgeImbalance,
    MigrationBacklog,
    NodeFailure,
    HighSyncLag,
    LowDiskSpace,
    HighMigrationFailureRate,
}

/// 告警配置
#[derive(Debug, Clone)]
pub struct AlertConfig {
    pub enable_notifications: bool,
    pub notification_channels: Vec<NotificationChannel>,
    pub alert_cooldown: std::time::Duration,
}

/// 通知渠道
#[derive(Debug, Clone)]
pub enum NotificationChannel {
    Log,
    Webhook(String),
    Email(String),
}

/// 指标历史
pub struct MetricsHistory {
    /// 节点指标历史
    node_history: HashMap<NodeId, VecDeque<NodeMetrics>>,
    /// 集群指标历史
    cluster_history: VecDeque<ClusterMetrics>,
    /// 最大保留条目数
    max_entries: usize,
}

impl EphemeralVmMonitor {
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            node_metrics: Arc::new(RwLock::new(HashMap::new()),
            cluster_metrics: Arc::new(RwLock::new(ClusterMetrics::default()),
            alert_manager: Arc::new(AlertManager::new(),
            history: Arc::new(RwLock::new(MetricsHistory::new()),
        }
    }

    /// 启动监控
    pub async fn start(&self) -> crate::Result<()> {
        let monitor = self.clone();
        
        // 启动指标收集
        tokio::spawn(async move {
            monitor.metrics_collector()?;
        })
        
        // 启动告警检查
        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.alert_checker()?;
        })
        
        // 启动预测分析（如果启用）
        if self.config.enable_prediction {
            let monitor = self.clone();
            tokio::spawn(async move {
                monitor.prediction_analyzer()?;
            Ok(())
        }
        
        Ok(())
    }

    /// 指标收集器
    async fn metrics_collector(&self) {
        let mut interval = tokio::time::interval(self.config.check_interval);
        
        loop {
            interval.tick()?;
            
            if let Err(e) = self.collect_metrics()?? {
                tracing::error!("Failed to collect metrics: {}", e);
            }
        }
    }

    /// 收集指标
    async fn collect_metrics(&self) -> crate::Result<()> {
        let now = Utc::now();
        
        // 收集节点指标
        let node_states = self.get_node_states()??)?;
        let mut node_metrics = HashMap::new();
        
        for (node_id, state) in &node_states {
            let metrics = self.calculate_node_metrics(node_id, state, now)??)?;
            node_metrics.insert(node_id.clone(), metrics)
        }
        
        // 计算集群级指标
        let cluster_metrics = self.calculate_cluster_metrics(&node_metrics, now)??)?;
        
        // 更新当前指标
        *self.node_metrics.write()? = node_metrics.clone();
        *self.cluster_metrics.write()? = cluster_metrics.clone();
        
        // 保存到历史
        self.save_to_history(node_metrics, cluster_metrics)?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        
        Ok(())
    }

    /// 计算节点指标
    async fn calculate_node_metrics(
        &self,
        nodeid: &NodeId,
        state: &NodeState,
        now: DateTime<Utc>) -> crate::Result<NodeMetrics> {
        let age = now - state.created_at;
        let age_days = age.num_seconds() as f64 / 86400.0;
        let remaining_days = 30.0 - age_days;
        
        // 计算迁移进度
        let migration_progress = match &state.sync_status {
            crate::cluster::ephemeralnode_manager::SyncStatus::Migrating { progress, .. } => *progress,
            _ => 0.0,
        };
        
        Ok(NodeMetrics {
            nodeid: node_id.clone(),
            timestamp: now,
            age_days,
            remaining_lifetime_days: remaining_days.max(0.0),
            health_score: state.health_score,
            migration_progress,
            data_size_gb: state.data_shards.len() as f64 * 0.1, // 假设每个分片0.1GB
            shard_count: state.data_shards.len(),
            sync_lag_seconds: 0.0, // 需要从同步系统获取
            cpu_usage: 0.0,       // 需要从系统监控获取
            memory_usage: 0.0,    // 需要从系统监控获取
            network_bandwidth_mbps: 0.0, // 需要从网络监控获取
        })
    }

    /// 计算集群级指标
    async fn calculate_cluster_metrics(
        &self,
        node_metrics: &HashMap<NodeId, NodeMetrics>,
        now: DateTime<Utc>) -> crate::Result<ClusterMetrics> {
        let mut young_nodes = 0;
        let mut middle_aged_nodes = 0;
        let mut old_nodes = 0;
        let mut retiring_nodes = 0;
        let mut migrating_nodes = 0;
        let mut total_age = 0.0;
        
        for metrics in node_metrics.values() {
            total_age += metrics.age_days;
            
            match metrics.age_days {
                d if d < 7.0 => young_nodes += 1,
                d if d < 21.0 => middle_aged_nodes += 1,
                d if d < 27.0 => old_nodes += 1,
                _ = > retiring_nodes += 1,)
            }
            
            if metrics.migration_progress > 0.0 {
                migrating_nodes += 1;
            Ok(())
        }
        
        let total_nodes = node_metrics.len();
        let avg_age = if total_nodes > 0 { total_age / total_nodes as f64 } else { 0.0 };
        
        // 计算年龄标准差
        let variance = node_metrics.values()
            .map(|m| (m.age_days - avg_age).powi(2))
            .sum::<f64>() / total_nodes.max(1) as f64;
        let std_dev = variance.sqrt();
        
        Ok(ClusterMetrics {
            timestamp: now,
            total_nodes,
            active_nodes: total_nodes - retiring_nodes,
            migrating_nodes,
            retiring_nodes,
            age_distribution: AgeDistribution {
                young_nodes,
                middle_aged_nodes,
                old_nodes,
                retiring_nodes,
                avg_age_days: avg_age,
                stddev_days: std_dev,
            },
            migration_stats: _MigrationStats::default(), // 需要从迁移系统获取
            data_distribution: _DataDistribution::default(), // 需要从存储系统获取
            system_health: self.calculate_system_health(node_metrics),
        })
    }

    /// 计算系统健康度
    fn calculate_system_health(&self, node_metrics: &HashMap<NodeId, NodeMetrics>) -> f32 {
        if node_metrics.is_empty() {
            return 0.0;
        }
        
        let total_health: f32 = node_metrics.values()
            .map(|m| m.health_score)
            .sum();
        
        total_health / node_metrics.len() as f32
    }

    /// 告警检查器
    async fn alert_checker(&self) {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30);
        
        loop {
            interval.tick()?;
            
            if let Err(e) = self.check_alerts()?? {
                tracing::error!("Failed to check alerts: {}", e);
            }
        }
    }

    /// 检查告警条件
    async fn check_alerts(&self) -> crate::Result<()> {
        let cluster_metrics = self.cluster_metrics.read()?.clone();
        let node_metrics = self.node_metrics.read()?.clone();
        
        // 检查副本不足
        if cluster_metrics.data_distribution.under_replicated_shards > 0 {
            self.trigger_alert(
                AlertType::UnderReplication,
                AlertSeverity::Critical,
                format!(
                    "{} shards have less than {} replicas",
                    cluster_metrics.data_distribution.under_replicated_shards,
                    self.config.alert_thresholds.critical_replication_factor
                ),
            )??
        }
        
        // 检查年龄不均衡
        if cluster_metrics.age_distribution.std_dev_days > self.config.alert_thresholds.age_imbalance_days as f64 {
            self.trigger_alert(
                AlertType::AgeImbalance,
                AlertSeverity::Warning,
                format!(
                    "Node age imbalance: std dev {} days exceeds threshold {} days",
                    cluster_metrics.age_distribution.std_dev_days,
                    self.config.alert_thresholds.age_imbalance_days
                ),
            )?;
        }
        
        // 检查迁移积压
        if cluster_metrics.migration_stats.pending_migrations > self.config.alert_thresholds.migration_backlog_count {
            self.trigger_alert(
                AlertType::MigrationBacklog,
                AlertSeverity::Warning,
                format!(
                    "Migration backlog: {} pending migrations exceed threshold {}",
                    cluster_metrics.migration_stats.pending_migrations,
                    self.config.alert_thresholds.migration_backlog_count
                ),
            )?;
        }
        
        Ok(())
    }

    /// 触发告警
    async fn trigger_alert(&self, alert_type: AlertType, severity: AlertSeverity, String) {
        self.alert_manager.trigger_alert(alert_type, severity, message)?
    }

    /// 预测分析器
    async fn prediction_analyzer(&self) {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(300); // 5分钟
        
        loop {
            interval.tick()?;
            
            if let Err(e) = self.analyze_predictions()?? {
                tracing::error!("Failed to analyze predictions: {}", e);
            }
        }
    }

    /// 分析预测
    async fn analyze_predictions(&self) -> crate::Result<()> {
        let history = self.history.read()?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        
        // 预测节点失效时间
        self.predict_node_failures(&history)??)?;
        
        // 预测迁移负载
        self.predict_migration_load(&history)?)?;
        
        // 预测容量需求
        self.predict_capacity_requirements(&history)?)?;
        
        Ok(())
    }

    /// 预测节点失效
    async fn predict_node_failures(&self, history: &MetricsHistory) -> crate::Result<()> {
        // 简单的线性预测
        for (node_id, metrics_history) in &history.node_history {
            if metrics_history.len() < 2 {

                continue;

            }
            let latest = &metrics_history[metrics_history.len() - 1];
            let remaining_days = latest.remaining_lifetime_days;
            
            if remaining_days < 3.0 {
                tracing::warn!(
                    "Node {} will expire in {:.1} days",
                    node_id.0, remaining_days
                );
            Ok(())
        }
        
        Ok(())
    }

    /// 预测迁移负载
    async fn predict_migration_load(&self, history: &MetricsHistory) -> crate::Result<()> {
        // 基于历史数据预测未来的迁移负载
        Ok(())
    }

    /// 预测容量需求
    async fn predict_capacity_requirements(&self, history: &MetricsHistory) -> crate::Result<()> {
        // 基于节点退役速度预测新节点需求
        Ok(())
    }

    /// 获取节点状态（占位符）
    async fn get_node_states(&self) -> crate::Result<HashMap<NodeId>> {
        // 实际实现中从节点管理器获取
        Ok(HashMap::new());
    }

    /// 保存到历史
    async fn save_to_history(&self, node_metrics: HashMap<NodeId, NodeMetrics>, cluster_metrics: ClusterMetrics) {
        let mut history = self.history.write()?;
        
        // 保存节点指标
        let max_entries = history.max_entries;
        for (node_id, metrics) in node_metrics {
            history.node_history.entry(node_id.clone())
                .or_insert_with(VecDeque::new)
                .push_back(metrics);
            
            // 限制历史大小
            if let Some(deque) = history.node_history.get_mut(&node_id) {
                while deque.len() > max_entries {
                    deque.pop_front();
                Ok(())
            }
        }
        
        // 保存集群指标
        history.cluster_history.push_back(cluster_metrics);
        while history.cluster_history.len() > max_entries {
            history.cluster_history.pop_front()
        }
    }

    /// 获取监控仪表板数据
    pub async fn get_dashboard_data(&self) -> DashboardData {
        let cluster_metrics = self.cluster_metrics.read()?.clone();
        let node_metrics = self.node_metrics.read()?.clone();
        
        DashboardData {
            cluster_overview: cluster_metrics,
            node_list: node_metrics.values().cloned().collect(),
            active_alerts: self.alert_manager.get_active_alerts()??,
            recent_history: self.get_recent_history()??,
        }
    }

    /// 获取最近历史
    async fn get_recent_history(&self) -> RecentHistory {
        let history = self.history.read()?;
        
        RecentHistory {
            cluster_trend: history.cluster_history.iter()
                .rev()
                .take(24) // 最近24小时
                .cloned()
                .collect(),
        }
    }
}

impl Clone for EphemeralVmMonitor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            node_metrics: self.node_metrics.clone(),
            cluster_metrics: self.cluster_metrics.clone(),
            alert_manager: self.alert_manager.clone(),
            history: self.history.clone(),
        }
    }
}

impl AlertManager {
    fn new() -> Self {
        Self {
            active_alerts: Arc::new(RwLock::new(Vec::new()),
            alert_history: Arc::new(RwLock::new(VecDeque::new()),
            config: AlertConfig {
                enable_notifications: true,
                notification_channels: vec![NotificationChannel::Log],
                alert_cooldown: std::time::Duration::from_secs(300),
            },
        }
    }

    async fn trigger_alert(&self, alert_type: AlertType, severity: AlertSeverity, String) {
        let alert = Alert {
            id: uuid::Uuid::new_v4().to_string(),
            severity,
            alert_type,
            message.clone(),
            details: HashMap::new(),
            triggered_at: _Utc::now(),
            resolved_at: None,
        };
        
        // 添加到活跃告警
        self.active_alerts.write()?.push(alert.clone();
        
        // 发送通知
        for channel in &self.config.notification_channels {
            match channel {
                NotificationChannel::Log => {
                    tracing::warn!("[ALERT] {:?}: {}", severity, message);
                }
                NotificationChannel::Webhook(url) => {
                    // 发送webhook通知
                }
                NotificationChannel::Email(address) => {
                    // 发送邮件通知
                }
            }
        }
        
        // 保存到历史
        self.alert_history.write()?.push_back(alert);
    }

    async fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.read()??.clone()
    }
}

impl MetricsHistory {
    fn new() -> Self {
        Self {
            node_history: HashMap::new(),
            cluster_history: VecDeque::new(),
            max_entries: 1440, // 24小时，每分钟一个条目
        }
    }
}

/// 仪表板数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub cluster_overview: ClusterMetrics,
    pub node_list: Vec<NodeMetrics>,
    pub active_alerts: Vec<Alert>,
    pub recent_history: RecentHistory,
}

/// 最近历史
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentHistory {
    pub cluster_trend: Vec<ClusterMetrics>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitor_creation() {
        let config = MonitorConfig::default();
        let monitor = EphemeralVmMonitor::new(config);
        
        let dashboard = monitor.get_dashboard_data()?;
        assert_eq!(dashboard.cluster_overview.total_nodes, 0)
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let alert_manager = AlertManager::new();
        
        alert_manager.trigger_alert(
            AlertType::UnderReplication,
            AlertSeverity::Critical,
            "Test alert".to_string()
        )?;
        
        let alerts = alert_manager.get_active_alerts()?;
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Critical)
    }
}
