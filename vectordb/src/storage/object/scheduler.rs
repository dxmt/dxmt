use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Notify, RwLock};
use tokio::time::interval;
use tracing::{debug, info, warn};

use super::tiered::{StorageTier, TieredStorage};
use super::traits::{ObjectStorage, ObjectStorageResult};
use crate::storage::object::ObjectStorageError;

/// 数据访问模式
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// 对象键
    pub key: String,
    /// 访问次数
    pub access_count: u64,
    /// 最后访问时间
    pub last_access: DateTime<Utc>,
    /// 对象大小
    pub size: u64,
    /// 当前层级
    pub current_tier: StorageTier,
    /// 访问时间序列（用于趋势分析）
    pub access_timeline: VecDeque<DateTime<Utc>>,
}

impl AccessPattern {
    pub fn new(key: String, size: u64, tier: StorageTier) -> Self {
        Self {
            key,
            access_count: 1,
            last_access: Utc::now(),
            size,
            current_tier: tier,
            access_timeline: VecDeque::with_capacity(100), // 保留最近100次访问
        }
    }

    /// 记录新的访问
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_access = Utc::now();

        self.access_timeline.push_back(self.last_access);
        if self.access_timeline.len() > 100 {
            self.access_timeline.pop_front();
        }
    }

    /// 计算访问频率 (次/小时)
    pub fn access_frequency(&self) -> f64 {
        let now = Utc::now();
        let one_hour_ago = now - chrono::Duration::hours(1);

        let recent_accesses = self
            .access_timeline
            .iter()
            .filter(|&&access_time| access_time > one_hour_ago)
            .count();

        recent_accesses as f64
    }

    /// 计算热度分数
    pub fn hotness_score(&self) -> f64 {
        let frequency = self.access_frequency();
        let time_since_access = Utc::now()
            .signed_duration_since(self.last_access)
            .num_minutes() as f64;

        // 基础热度 = 频率 / 时间衰减 / 大小权重
        let time_decay = 1.0 / (time_since_access + 1.0).ln_1p();
        let size_factor = 1.0 / (self.size as f64).ln_1p();

        frequency * time_decay * size_factor
    }

    /// 预测未来访问概率
    pub fn predict_future_access(&self) -> f64 {
        if self.access_timeline.len() < 2 {
            return 0.5; // 默认概率
        }

        // 简单的线性趋势分析
        let recent_frequency = self.access_frequency();
        let total_frequency = self.access_count as f64
            / Utc::now()
                .signed_duration_since(self.access_timeline[0])
                .num_hours()
                .max(1) as f64;

        // 如果最近访问频率高于历史平均，预测概率更高
        if recent_frequency > total_frequency {
            0.8
        } else if recent_frequency > total_frequency * 0.5 {
            0.6
        } else {
            0.3
        }
    }
}

/// 迁移任务
#[derive(Debug, Clone)]
pub struct MigrationTask {
    pub key: String,
    pub from_tier: StorageTier,
    pub to_tier: StorageTier,
    pub priority: f64,
    pub created_at: DateTime<Utc>,
}

/// 调度策略
#[derive(Debug, Clone)]
pub struct SchedulingPolicy {
    /// 内存层热度阈值
    pub memory_hotness_threshold: f64,
    /// SSD层热度阈值
    pub ssd_hotness_threshold: f64,
    /// 冷数据降级阈值
    pub cold_data_threshold: f64,
    /// 预测性迁移开关
    pub predictive_migration: bool,
    /// 批量迁移大小
    pub batch_migration_size: usize,
    /// 成本优化权重
    pub cost_optimization_weight: f64,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        Self {
            memory_hotness_threshold: 2.0,
            ssd_hotness_threshold: 1.0,
            cold_data_threshold: 0.1,
            predictive_migration: true,
            batch_migration_size: 10,
            cost_optimization_weight: 0.3,
        }
    }
}

/// 数据调度器
pub struct DataScheduler {
    /// 存储引用
    storage: Arc<TieredStorage>,
    /// 访问模式数据
    access_patterns: Arc<RwLock<HashMap<String, AccessPattern>>>,
    /// 迁移任务队列
    migration_queue: Arc<RwLock<VecDeque<MigrationTask>>>,
    /// 调度策略
    policy: SchedulingPolicy,
    /// 调度间隔
    schedule_interval: Duration,
    /// 停止信号
    stop_signal: Arc<Notify>,
    /// 统计信息
    stats: Arc<RwLock<SchedulerStats>>,
}

/// 调度器统计信息
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub total_migrations: u64,
    pub successful_migrations: u64,
    pub failed_migrations: u64,
    pub bytesmigrated: u64,
    pub cost_savings: f64,
    pub last_schedule_time: Option<DateTime<Utc>>,
    pub average_schedule_duration: Duration,
}

impl DataScheduler {
    /// 创建新的数据调度器
    pub fn new(storage: Arc<TieredStorage>, policy: SchedulingPolicy) -> Self {
        Self {
            storage,
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
            migration_queue: Arc::new(RwLock::new(VecDeque::new())),
            policy,
            schedule_interval: Duration::from_secs(300), // 5分钟
            stop_signal: Arc::new(Notify::new()),
            stats: Arc::new(RwLock::new(SchedulerStats::default())),
        }
    }

    /// 记录数据访问
    pub async fn record_access(&self, key: &str, size: u64, tier: StorageTier) {
        let mut patterns = self.access_patterns.write().await;

        match patterns.get_mut(key) {
            Some(pattern) => {
                pattern.record_access();
                pattern.current_tier = tier
            }
            None => {
                patterns.insert(
                    key.to_string(),
                    AccessPattern::new(key.to_string(), size, tier),
                );
            }
        }
    }

    /// 启动调度器
    pub async fn start(&self) -> ObjectStorageResult<()> {
        info!("Starting data scheduler with policy: {:?}", self.policy);

        let mut schedule_timer = interval(self.schedule_interval);
        let stop_signal = Arc::clone(&self.stop_signal);

        loop {
            tokio::select! {
                _ = schedule_timer.tick() => {
                    let start_time = Instant::now();

                    if let Err(e) = self.run_scheduling_cycle().await {
                        warn!("Scheduling cycle failed: {}", e);
                    }

                    // 更新统计信息
                    let mut stats = self.stats.write().await;
                    stats.last_schedule_time = Some(Utc::now());
                    stats.average_schedule_duration = start_time.elapsed();
                }
                _ = stop_signal.notified() => {
                    info!("Data scheduler stopped");
                    break;
                }
            }
        }

        Ok(())
    }

    /// 停止调度器
    pub fn stop(&self) {
        self.stop_signal.notify_one()
    }

    /// 执行一轮调度
    async fn run_scheduling_cycle(&self) -> ObjectStorageResult<()> {
        debug!("Starting scheduling cycle");

        // 1. 分析访问模式
        self.analyze_access_patterns().await?;

        // 2. 生成迁移任务
        self.generate_migration_tasks().await?;

        // 3. 执行迁移任务
        self.execute_migration_tasks().await?;

        // 4. 清理过期数据
        self.cleanup_old_patterns().await.map_err(|e| ObjectStorageError::InternalError { message: format!("Cleanup failed: {:?}", e) })?;

        debug!("Scheduling cycle completed");
        Ok(())
    }

    /// 分析访问模式
    async fn analyze_access_patterns(&self) -> ObjectStorageResult<()> {
        let patterns = self.access_patterns.read().await;

        debug!("Analyzing {} access patterns", patterns.len());

        let mut hot_objects = 0;
        let mut warm_objects = 0;
        let mut cold_objects = 0;

        for pattern in patterns.values() {
            let hotness = pattern.hotness_score();

            if hotness > self.policy.memory_hotness_threshold {
                hot_objects += 1
            } else if hotness > self.policy.ssd_hotness_threshold {
                warm_objects += 1
            } else {
                cold_objects += 1
            }
        }

        debug!(
            "Objects classification: hot={}, warm={}, cold={}",
            hot_objects, warm_objects, cold_objects
        );

        Ok(())
    }

    /// 生成迁移任务
    async fn generate_migration_tasks(&self) -> ObjectStorageResult<()> {
        let patterns = self.access_patterns.read().await;
        let mut queue = self.migration_queue.write().await;

        for pattern in patterns.values() {
            let hotness = pattern.hotness_score();
            let prediction = if self.policy.predictive_migration {
                pattern.predict_future_access()
            } else {
                0.5
            };

            // 决定目标层级
            let target_tier = self.determine_target_tier(pattern, hotness, prediction);

            if target_tier != pattern.current_tier {
                let priority = self.calculate_migration_priority(pattern, hotness, prediction);

                let task = MigrationTask {
                    key: pattern.key.clone(),
                    from_tier: pattern.current_tier,
                    to_tier: target_tier,
                    priority,
                    created_at: Utc::now(),
                };

                queue.push_back(task);
            }
        }

        // 按优先级排序
        let mut tasks: Vec<_> = queue.drain(..).collect();
        tasks.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        queue.extend(tasks);

        debug!("Generated {} migration tasks", queue.len());
        Ok(())
    }

    /// 确定目标存储层级
    fn determine_target_tier(
        &self,
        pattern: &AccessPattern,
        hotness: f64,
        prediction: f64) -> StorageTier {
        // 考虑对象大小限制
        let memory_size_limit = 1024 * 1024; // 1MB
        let ssd_size_limit = 100 * 1024 * 1024; // 100MB

        if hotness > self.policy.memory_hotness_threshold
            && pattern.size < memory_size_limit
            && prediction > 0.7
        {
            StorageTier::Memory
        } else if hotness > self.policy.ssd_hotness_threshold
            && pattern.size < ssd_size_limit
            && prediction > 0.4
        {
            StorageTier::Ssd
        } else {
            StorageTier::ObjectStorage
        }
    }

    /// 计算迁移优先级
    fn calculate_migration_priority(
        &self,
        pattern: &AccessPattern,
        hotness: f64,
        prediction: f64) -> f64 {
        let hotness_weight = 0.4;
        let prediction_weight = 0.3;
        let cost_weight = self.policy.cost_optimization_weight;

        // 计算成本因子（提升到更快层级的成本效益）
        let cost_factor = match (
            pattern.current_tier,
            self.determine_target_tier(pattern, hotness, prediction),
        ) {
            (StorageTier::ObjectStorage, StorageTier::Memory) => 2.0,
            (StorageTier::ObjectStorage, StorageTier::Ssd) => 1.5,
            (StorageTier::Ssd, StorageTier::Memory) => 1.2,
            (StorageTier::Memory, StorageTier::Ssd) => -0.5,
            (StorageTier::Ssd, StorageTier::ObjectStorage) => -1.0,
            (StorageTier::Memory, StorageTier::ObjectStorage) => -1.5,
            _ => 0.0,
        };

        hotness * hotness_weight + prediction * prediction_weight + cost_factor * cost_weight
    }

    /// 执行迁移任务
    async fn execute_migration_tasks(&self) -> ObjectStorageResult<()> {
        let mut queue = self.migration_queue.write().await;
        let mut executed = 0;
        let batch_size = self.policy.batch_migration_size;

        while executed < batch_size && !queue.is_empty() {
            if let Some(task) = queue.pop_front() {
                if let Err(e) = self.execute_single_migration(&task).await {
                    warn!("Migration failed for {}: {}", task.key, e);

                    let mut stats = self.stats.write().await;
                    stats.failed_migrations += 1;
                } else {
                    debug!(
                        "Successfully migrated {} from {:?} to {:?}",
                        task.key, task.from_tier, task.to_tier
                    );

                    let mut stats = self.stats.write().await;
                    stats.successful_migrations += 1;
                    stats.total_migrations += 1;
                }

                executed += 1;
            }
        }

        debug!("Executed {} migration tasks", executed);
        Ok(())
    }

    /// 执行单个迁移任务
    async fn execute_single_migration(&self, task: &MigrationTask) -> ObjectStorageResult<()> {
        // 这里应该实现实际的迁移逻辑
        // 由于我们当前的 TieredStorage 已经自动处理分层，
        // 这里主要是触发数据的重新分层

        // 模拟迁移过程
        match (task.from_tier, task.to_tier) {
            (StorageTier::ObjectStorage, StorageTier::Memory)
            | (StorageTier::ObjectStorage, StorageTier::Ssd) => {
                // 触发数据访问，让TieredStorage自动缓存
                let _ = self.storage.get_object(&task.key).await?;
            }
            (StorageTier::Memory, StorageTier::ObjectStorage)
            | (StorageTier::Ssd, StorageTier::ObjectStorage) => {
                // 对于降级，我们需要从缓存中移除
                // 这需要在TieredStorage中添加相应的方法
                debug!("Would evict {} from cache", task.key);
            }
            _ => {
                // 其他迁移情况
                debug!(
                    "Migration from {:?} to {:?} not implemented",
                    task.from_tier, task.to_tier
                );
            }
        }

        // 更新访问模式中的层级信息
        let mut patterns = self.access_patterns.write().await;
        if let Some(pattern) = patterns.get_mut(&task.key) {
            pattern.current_tier = task.to_tier
        }

        Ok(())
    }

    /// 清理过期的访问模式数据
    async fn cleanup_old_patterns(&self) -> crate::Result<()> {
        let mut patterns = self.access_patterns.write().await;
        let cutoff_time = Utc::now() - chrono::Duration::days(7); // 保留7天的数据

        patterns.retain(|_, pattern| pattern.last_access > cutoff_time);

        debug!("Cleaned up old patterns, {} remaining", patterns.len());
        Ok(())
    }

    /// 获取调度器统计信息
    pub async fn get_stats(&self) -> crate::Result<SchedulerStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    /// 获取当前访问模式摘要
    pub async fn get_access_summary(&self) -> crate::Result<HashMap<String, f64>> {
        let patterns = self.access_patterns.read().await;

        Ok(patterns
            .iter()
            .map(|(key, pattern)| (key.clone(), pattern.hotness_score()))
            .collect())
    }
}
