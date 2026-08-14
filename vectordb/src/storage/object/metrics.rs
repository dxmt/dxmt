use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use super::compression::CompressionType;
use super::tiered::StorageTier;

/// 性能指标收集器
#[derive(Debug)]
pub struct MetricsCollector {
    /// 操作统计
    operation_stats: Arc<RwLock<HashMap<String, OperationMetrics>>>,
    /// 存储层统计
    tier_stats: Arc<RwLock<HashMap<StorageTier, TierMetrics>>>,
    /// 压缩统计
    compression_stats: Arc<RwLock<HashMap<CompressionType, CompressionMetrics>>>,
    /// 全局统计
    global_stats: Arc<RwLock<GlobalMetrics>>,
    /// 历史统计点
    historical_stats: Arc<RwLock<Vec<TimestampedMetrics>>>,
}

/// 操作指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    pub operation_name: String,
    pub total_count: u64,
    pub success_count: u64,
    pub error_count: u64,
    pub total_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub avg_duration: Duration,
    pub p95_duration: Duration,
    pub p99_duration: Duration,
    pub bytesprocessed: u64,
    pub last_operation_time: DateTime<Utc>,
    pub recent_durations: Vec<Duration>, // 最近100次操作的延迟
}

impl Default for OperationMetrics {
    fn default() -> Self {
        Self {
            operation_name: String::new(),
            total_count: 0,
            success_count: 0,
            error_count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            avg_duration: Duration::ZERO,
            p95_duration: Duration::ZERO,
            p99_duration: Duration::ZERO,
            bytesprocessed: 0,
            last_operation_time: chrono::Utc::now(),
            recent_durations: Vec::with_capacity(100),
        }
    }
}

impl OperationMetrics {
    pub fn record_operation(&mut self, duration: Duration, bytes: u64, success: bool) {
        self.total_count += 1;

        if success {
            self.success_count += 1
        } else {
            self.error_count += 1;
        }

        self.total_duration += duration;
        self.bytesprocessed += bytes;
        self.last_operation_time = chrono::Utc::now();

        // 更新最小/最大延迟
        if duration < self.min_duration {
            self.min_duration = duration
        }
        if duration > self.max_duration {
            self.max_duration = duration
        }

        // 记录最近的延迟
        self.recent_durations.push(duration);
        if self.recent_durations.len() > 100 {

            self.recent_durations.remove(0);

        }
        // 更新平均延迟
        if self.total_count > 0 {
            self.avg_duration = self.total_duration / self.total_count as u32
        }

        // 计算百分位数
        self.calculate_percentiles();
    }

    fn calculate_percentiles(&mut self) {
        if self.recent_durations.is_empty() {
            return;
        }

        let mut sorted_durations = self.recent_durations.clone();
        sorted_durations.sort();

        let len = sorted_durations.len();
        if len > 0 {
            let p95_index = (len as f64 * 0.95) as usize;
            let p99_index = (len as f64 * 0.99) as usize;

            self.p95_duration = sorted_durations[p95_index.min(len - 1)];
            self.p99_duration = sorted_durations[p99_index.min(len - 1)]
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.success_count as f64 / self.total_count as f64 * 100.0
        }
    }

    pub fn throughput_mbps(&self) -> f64 {
        if self.total_duration.as_secs_f64() > 0.0 {
            self.bytesprocessed as f64 / (1024.0 * 1024.0) / self.total_duration.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// 存储层指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierMetrics {
    pub tier: StorageTier,
    pub read_ops: u64,
    pub write_ops: u64,
    pub delete_ops: u64,
    pub cachehits: u64,
    pub cache_misses: u64,
    pub bytesread: u64,
    pub byteswritten: u64,
    pub objects_stored: u64,
    pub total_storagebytes: u64,
    pub avg_object_size: u64,
    pub evictions: u64,
    pub promotions: u64, // 从低层提升到此层的次数
}

impl Default for TierMetrics {
    fn default() -> Self {
        Self {
            tier: StorageTier::Memory,
            read_ops: 0,
            write_ops: 0,
            delete_ops: 0,
            cachehits: 0,
            cache_misses: 0,
            bytesread: 0,
            byteswritten: 0,
            objects_stored: 0,
            total_storagebytes: 0,
            avg_object_size: 0,
            evictions: 0,
            promotions: 0,
        }
    }
}

impl TierMetrics {
    pub fn cache_hit_rate(&self) -> f64 {
        let total_accesses = self.cachehits + self.cache_misses;
        if total_accesses == 0 {
            0.0
        } else {
            self.cachehits as f64 / total_accesses as f64 * 100.0
        }
    }

    pub fn update_avg_object_size(&mut self) {
        if let Some(avg) = self.total_storagebytes.checked_div(self.objects_stored) {
            self.avg_object_size = avg
        }
    }
}

/// 压缩指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetrics {
    pub compression_type: CompressionType,
    pub compression_ops: u64,
    pub decompression_ops: u64,
    pub total_originalbytes: u64,
    pub total_compressedbytes: u64,
    pub compression_time: Duration,
    pub decompression_time: Duration,
    pub compression_failures: u64,
    pub decompression_failures: u64,
}

impl Default for CompressionMetrics {
    fn default() -> Self {
        Self {
            compression_type: CompressionType::None,
            compression_ops: 0,
            decompression_ops: 0,
            total_originalbytes: 0,
            total_compressedbytes: 0,
            compression_time: Duration::ZERO,
            decompression_time: Duration::ZERO,
            compression_failures: 0,
            decompression_failures: 0,
        }
    }
}

impl CompressionMetrics {
    pub fn overall_compression_ratio(&self) -> f64 {
        if self.total_originalbytes == 0 {
            1.0
        } else {
            self.total_compressedbytes as f64 / self.total_originalbytes as f64
        }
    }

    pub fn space_savedbytes(&self) -> u64 {
        self.total_originalbytes
            .saturating_sub(self.total_compressedbytes)
    }

    pub fn space_saved_percentage(&self) -> f64 {
        (1.0 - self.overall_compression_ratio()) * 100.0
    }

    pub fn avg_compression_speed_mbps(&self) -> f64 {
        if self.compression_time.as_secs_f64() > 0.0 {
            self.total_originalbytes as f64
                / (1024.0 * 1024.0)
                / self.compression_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// 全局指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetrics {
    pub uptime: Duration,
    pub total_objects: u64,
    pub totalbytes_stored: u64,
    pub total_requests: u64,
    pub total_errors: u64,
    pub data_migrations: u64,
    pub cost_savings: f64, // 预估成本节省 (美元)
    pub memory_usagebytes: u64,
    pub disk_usagebytes: u64,
    pub networkbytes_in: u64,
    pub networkbytes_out: u64,
    pub start_time: DateTime<Utc>,
}

impl Default for GlobalMetrics {
    fn default() -> Self {
        Self {
            uptime: Duration::ZERO,
            total_objects: 0,
            totalbytes_stored: 0,
            total_requests: 0,
            total_errors: 0,
            data_migrations: 0,
            cost_savings: 0.0,
            memory_usagebytes: 0,
            disk_usagebytes: 0,
            networkbytes_in: 0,
            networkbytes_out: 0,
            start_time: chrono::Utc::now(),
        }
    }
}

impl GlobalMetrics {
    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_errors as f64 / self.total_requests as f64 * 100.0
        }
    }

    pub fn update_uptime(&mut self) {
        let now = chrono::Utc::now();
        self.uptime = now
            .signed_duration_since(self.start_time)
            .to_std()
            .unwrap_or(Duration::ZERO)
    }
}

/// 带时间戳的指标快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedMetrics {
    pub timestamp: DateTime<Utc>,
    pub global_metrics: GlobalMetrics,
    pub tier_metrics: HashMap<StorageTier, TierMetrics>,
    pub compression_metrics: HashMap<CompressionType, CompressionMetrics>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        let mut tier_stats = HashMap::new();
        tier_stats.insert(
            StorageTier::Memory,
            TierMetrics {
                tier: StorageTier::Memory,
                ..Default::default()
            },
        );
        tier_stats.insert(
            StorageTier::Ssd,
            TierMetrics {
                tier: StorageTier::Ssd,
                ..Default::default()
            },
        );
        tier_stats.insert(
            StorageTier::ObjectStorage,
            TierMetrics {
                tier: StorageTier::ObjectStorage,
                ..Default::default()
            },
        );

        Self {
            operation_stats: Arc::new(RwLock::new(HashMap::new())),
            tier_stats: Arc::new(RwLock::new(tier_stats)),
            compression_stats: Arc::new(RwLock::new(HashMap::new())),
            global_stats: Arc::new(RwLock::new(GlobalMetrics::default())),
            historical_stats: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// 记录操作指标
    pub async fn record_operation(
        &self,
        operation: &str,
        duration: Duration,
        bytes: u64,
        success: bool) {
        let mut operation_stats = self.operation_stats.write().await;
        let metrics = operation_stats
            .entry(operation.to_string())
            .or_insert_with(|| OperationMetrics {
                operation_name: operation.to_string(),
                ..Default::default()
            });

        metrics.record_operation(duration, bytes, success);

        // 更新全局统计
        let mut global_stats = self.global_stats.write().await;
        global_stats.total_requests += 1;
        if !success {

            global_stats.total_errors += 1;

        }
        global_stats.update_uptime();
    }

    /// 记录存储层指标
    pub async fn record_tier_operation(
        &self,
        tier: StorageTier,
        operation: &str,
        bytes: u64,
        cache_hit: Option<bool>) {
        let mut tier_stats = self.tier_stats.write().await;
        let metrics = tier_stats.get_mut(&tier);

        if let Some(metrics) = metrics {
            match operation {
                "read" => {
                    metrics.read_ops += 1;
                    metrics.bytesread += bytes;
                }
                "write" => {
                    metrics.write_ops += 1;
                    metrics.byteswritten += bytes;
                    metrics.objects_stored += 1;
                    metrics.total_storagebytes += bytes;
                    metrics.update_avg_object_size();
                }
                "delete" => {
                    metrics.delete_ops += 1;
                    if metrics.objects_stored > 0 {
                        metrics.objects_stored -= 1;
                        metrics.total_storagebytes = metrics.total_storagebytes.saturating_sub(bytes);
                        metrics.update_avg_object_size();
                    }
                }
                _ => {}
            }

            if let Some(hit) = cache_hit {
                if hit {
                    metrics.cachehits += 1
                } else {
                    metrics.cache_misses += 1
                }
            }
        }
    }

    /// 记录压缩指标
    pub async fn record_compression(
        &self,
        compression_type: CompressionType,
        operation: &str,
        originalbytes: u64,
        compressedbytes: u64,
        duration: Duration,
        success: bool) {
        let mut compression_stats = self.compression_stats.write().await;
        let metrics = compression_stats
            .entry(compression_type)
            .or_insert_with(|| CompressionMetrics {
                compression_type,
                ..Default::default()
            });

        match operation {
            "compress" => {
                metrics.compression_ops += 1;
                metrics.compression_time += duration;
                if success {
                    metrics.total_originalbytes += originalbytes;
                    metrics.total_compressedbytes += compressedbytes
                } else {
                    metrics.compression_failures += 1
                }
            }
            "decompress" => {
                metrics.decompression_ops += 1;
                metrics.decompression_time += duration;
                if !success {
                    metrics.decompression_failures += 1
                }
            }
            _ => {}
        }
    }

    /// 记录数据迁移
    pub async fn record_migration(&self, from_tier: StorageTier, to_tier: StorageTier, bytes: u64) {
        // 记录迁移统计
        let mut global_stats = self.global_stats.write().await;
        global_stats.data_migrations += 1;

        // 记录层级统计
        let mut tier_stats = self.tier_stats.write().await;

        // 源层级减少
        if let Some(from_metrics) = tier_stats.get_mut(&from_tier) {

            if from_metrics.objects_stored > 0 {
                from_metrics.objects_stored -= 1;
                from_metrics.total_storagebytes =
                    from_metrics.total_storagebytes.saturating_sub(bytes);
                from_metrics.evictions += 1;
                from_metrics.update_avg_object_size();

        }
        }

        // 目标层级增加
        if let Some(to_metrics) = tier_stats.get_mut(&to_tier) {

            to_metrics.objects_stored += 1;
            to_metrics.total_storagebytes += bytes;
            to_metrics.promotions += 1;
            to_metrics.update_avg_object_size();

        }
    }

    /// 获取操作统计
    pub async fn get_operation_stats(&self) -> HashMap<String, OperationMetrics> {
        self.operation_stats.read().await.clone()
    }

    /// 获取存储层统计
    pub async fn get_tier_stats(&self) -> HashMap<StorageTier, TierMetrics> {
        self.tier_stats.read().await.clone()
    }

    /// 获取压缩统计
    pub async fn get_compression_stats(&self) -> HashMap<CompressionType, CompressionMetrics> {
        self.compression_stats.read().await.clone()
    }

    /// 获取全局统计
    pub async fn get_global_stats(&self) -> crate::Result<GlobalMetrics> {
        let mut global_stats = self.global_stats.write().await;
        global_stats.update_uptime();
        Ok(global_stats.clone())
    }

    /// 创建当前指标快照
    pub async fn create_snapshot(&self) -> TimestampedMetrics {
        TimestampedMetrics {
            timestamp: chrono::Utc::now(),
            global_metrics: self.get_global_stats().await.unwrap_or_default(),
            tier_metrics: self.get_tier_stats().await,
            compression_metrics: self.get_compression_stats().await,
        }
    }

    /// 保存历史快照
    pub async fn save_snapshot(&self) {
        let snapshot = self.create_snapshot().await;
        let mut historical_stats = self.historical_stats.write().await;

        historical_stats.push(snapshot);

        // 只保留最近100个快照
        if historical_stats.len() > 100 {

            historical_stats.remove(0);

        }
    }

    /// 获取历史快照
    pub async fn get_historical_snapshots(&self) -> Vec<TimestampedMetrics> {
        self.historical_stats.read().await.clone()
    }

    /// 生成详细报告
    pub async fn generate_report(&self) -> MetricsReport {
        let global_stats = self.get_global_stats().await.unwrap_or_default();
        let operation_stats = self.get_operation_stats().await;
        let tier_stats = self.get_tier_stats().await;
        let compression_stats = self.get_compression_stats().await;

        let summary = self
            .generate_summary(&operation_stats, &tier_stats, &compression_stats)
            .await;

        MetricsReport {
            timestamp: chrono::Utc::now(),
            global_stats,
            operation_stats,
            tier_stats,
            compression_stats,
            summary,
    }
    }

    async fn generate_summary(
        &self,
        operation_stats: &HashMap<String, OperationMetrics>,
        tier_stats: &HashMap<StorageTier, TierMetrics>,
        compression_stats: &HashMap<CompressionType, CompressionMetrics>,
    ) -> MetricsSummary {
        // 计算总体统计
        let total_operations: u64 = operation_stats.values().map(|m| m.total_count).sum();
        let total_errors: u64 = operation_stats.values().map(|m| m.error_count).sum();
        let avg_success_rate = if total_operations > 0 {
            (total_operations - total_errors) as f64 / total_operations as f64 * 100.0
        } else {
            0.0
        };

        // 计算平均延迟
        let avg_latency = if !operation_stats.is_empty() {
            let total_avg_duration: Duration =
                operation_stats.values().map(|m| m.avg_duration).sum();
            total_avg_duration / operation_stats.len() as u32
        } else {
            Duration::ZERO
        };

        // 计算总体缓存命中率
        let (total_hits, total_misses) = tier_stats
            .values()
            .fold((0u64, 0u64), |(hits, misses), metrics| {
                (hits + metrics.cachehits, misses + metrics.cache_misses)
            });

        let overall_cache_hit_rate = if total_hits + total_misses > 0 {
            total_hits as f64 / (total_hits + total_misses) as f64 * 100.0
        } else {
            0.0
        };

        // 计算总压缩节省
        let total_compression_savings: u64 = compression_stats
            .values()
            .map(|m| m.space_savedbytes())
            .sum();

        MetricsSummary {
            total_operations,
            avg_success_rate,
            avg_latency,
            overall_cache_hit_rate,
            total_compression_savings,
            performance_grade: self.calculate_performance_grade(
                avg_success_rate,
                overall_cache_hit_rate,
                avg_latency,
            )
    }
    }

    fn calculate_performance_grade(
        &self,
        success_rate: f64,
        cache_hit_rate: f64,
        avg_latency: Duration) -> String {
        let mut score = 0;

        // 成功率评分 (40%)
        if success_rate >= 99.9 {
            score += 40
        } else if success_rate >= 99.0 {
            score += 35
        } else if success_rate >= 95.0 {
            score += 25
        } else {
            score += 10
        }

        // 缓存命中率评分 (35%)
        if cache_hit_rate >= 90.0 {
            score += 35
        } else if cache_hit_rate >= 80.0 {
            score += 30
        } else if cache_hit_rate >= 70.0 {
            score += 20
        } else {
            score += 10
        }

        // 延迟评分 (25%)
        let latency_ms = avg_latency.as_millis();
        if latency_ms <= 10 {
            score += 25
        } else if latency_ms <= 50 {
            score += 20
        } else if latency_ms <= 100 {
            score += 15
        } else {
            score += 5
        }

        match score {
            90..=100 => "A+ (优秀)".to_string(),
            80..=89 => "A (良好)".to_string(),
            70..=79 => "B (一般)".to_string(),
            60..=69 => "C (较差)".to_string(),
            _ => "D (需要优化)".to_string(),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// 指标报告
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsReport {
    pub timestamp: DateTime<Utc>,
    pub global_stats: GlobalMetrics,
    pub operation_stats: HashMap<String, OperationMetrics>,
    pub tier_stats: HashMap<StorageTier, TierMetrics>,
    pub compression_stats: HashMap<CompressionType, CompressionMetrics>,
    pub summary: MetricsSummary,
}

/// 指标摘要
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct MetricsSummary {
    pub total_operations: u64,
    pub avg_success_rate: f64,
    pub avg_latency: Duration,
    pub overall_cache_hit_rate: f64,
    pub total_compression_savings: u64,
    pub performance_grade: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_metrics_collection() -> Result<(), Box<dyn std::error::Error>> {
        let collector = MetricsCollector::new();

        // 记录一些操作
        collector
            .record_operation("get_object", Duration::from_millis(10), 1024, true)
            .await;
        collector
            .record_operation("put_object", Duration::from_millis(25), 2048, true)
            .await;
        collector
            .record_operation("get_object", Duration::from_millis(5), 512, false)
            .await;

        let stats = collector.get_operation_stats().await;

        assert!(stats.contains_key("get_object"));
        assert!(stats.contains_key("put_object"));

        let get_stats = &stats["get_object"];
        assert_eq!(get_stats.total_count, 2);
        assert_eq!(get_stats.success_count, 1);
        assert_eq!(get_stats.error_count, 1);
        assert_eq!(get_stats.success_rate(), 50.0);
        Ok(())
    }

    #[tokio::test]
    async fn test_tier_metrics() -> Result<(), Box<dyn std::error::Error>> {
        let collector = MetricsCollector::new();

        // 记录存储层操作
        collector
            .record_tier_operation(StorageTier::Memory, "read", 1024, Some(true))
            .await;
        collector
            .record_tier_operation(StorageTier::Memory, "read", 512, Some(false))
            .await;
        collector
            .record_tier_operation(StorageTier::Ssd, "write", 2048, None)
            .await;

        let tier_stats = collector.get_tier_stats().await;

        let memory_stats = &tier_stats[&StorageTier::Memory];
        assert_eq!(memory_stats.read_ops, 2);
        assert_eq!(memory_stats.cachehits, 1);
        assert_eq!(memory_stats.cache_misses, 1);
        assert_eq!(memory_stats.cache_hit_rate(), 50.0);

        let ssd_stats = &tier_stats[&StorageTier::Ssd];
        assert_eq!(ssd_stats.write_ops, 1);
        assert_eq!(ssd_stats.objects_stored, 1);
        Ok(())
    }
}
