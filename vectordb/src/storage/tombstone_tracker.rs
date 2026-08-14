/// 墓碑标记系统 - 跟踪已删除记录，确保搜索时不返回已删除数据
use crate::types::RecordId;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 纯 Rust 实现的 Bloom filter（双哈希：FNV-1a + DJB2）
/// 预期误判率约为 `fp_rate`，位数组大小由 `m = -n*ln(p) / ln(2)^2` 公式确定。
struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
}

impl BloomFilter {
    #[allow(dead_code)]
    fn new() -> Self {
        Self::with_rate(1024, 0.01)
    }

    /// 根据预期元素数量和误判率创建 Bloom filter
    fn with_rate(expected_items: usize, fp_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        let p = fp_rate.clamp(1e-10, 1.0 - 1e-10);
        let m = (-n * p.ln() / (2.0_f64.ln().powi(2))).ceil() as usize;
        let num_bits = m.max(64);
        let k = ((num_bits as f64 / n) * 2.0_f64.ln()).ceil() as usize;
        let num_hashes = k.clamp(1, 32);
        let words = num_bits.div_ceil(64);
        Self {
            bits: vec![0u64; words],
            num_bits,
            num_hashes,
        }
    }

    /// FNV-1a 32-bit hash seeded with `seed`
    fn fnv1a(data: &[u8], seed: u32) -> u32 {
        let mut h: u32 = 2_166_136_261u32.wrapping_add(seed.wrapping_mul(16_777_619));
        for &b in data {
            h ^= b as u32;
            h = h.wrapping_mul(16_777_619);
        }
        h
    }

    /// DJB2 hash seeded with `seed`
    fn djb2(data: &[u8], seed: u32) -> u32 {
        let mut h: u32 = 5381u32.wrapping_add(seed);
        for &b in data {
            h = h.wrapping_shl(5).wrapping_add(h).wrapping_add(b as u32);
        }
        h
    }

    fn bit_index(&self, data: &[u8], i: usize) -> usize {
        let h1 = Self::fnv1a(data, i as u32) as usize;
        let h2 = Self::djb2(data, i as u32) as usize;
        (h1.wrapping_add(i.wrapping_mul(h2))) % self.num_bits
    }

    fn insert(&mut self, item: &[u8]) {
        for i in 0..self.num_hashes {
            let idx = self.bit_index(item, i);
            self.bits[idx / 64] |= 1u64 << (idx % 64);
        }
    }

    fn contains(&self, item: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let idx = self.bit_index(item, i);
            if self.bits[idx / 64] & (1u64 << (idx % 64)) == 0 {
                return false;
            }
        }
        true
    }

    fn clear(&mut self) {
        self.bits.iter_mut().for_each(|w| *w = 0);
    }
}
use dashmap::DashMap;

/// 删除记录信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TombstoneEntry {
    /// 记录ID
    pub id: RecordId,
    /// 删除时间
    pub deleted_at: DateTime<Utc>,
    /// 删除原因（可选）
    pub reason: Option<String>,
    /// 表名
    pub table_name: String,
}

/// 墓碑标记配置
#[derive(Debug, Clone)]
pub struct TombstoneConfig {
    /// Bloom Filter预期元素数量
    pub expected_elements: usize,
    /// Bloom Filter错误概率
    pub false_positive_rate: f32,
    /// 墓碑记录保留时间（默认7天）
    pub retention_period: Duration,
    /// 自动清理间隔（默认1小时）
    pub cleanup_interval: Duration,
    /// 是否启用持久化
    pub enable_persistence: bool,
    /// 持久化文件路径
    pub persistence_path: Option<std::path::PathBuf>,
}

impl Default for TombstoneConfig {
    fn default() -> Self {
        Self {
            expected_elements: 1_000_000, // 预期100万删除记录
            false_positive_rate: 0.01,    // 1%错误率
            retention_period: Duration::days(7),
            cleanup_interval: Duration::hours(1),
            enable_persistence: false,
            persistence_path: None,
        }
    }
}

/// 墓碑记录跟踪器
pub struct TombstoneTracker {
    /// 配置
    config: TombstoneConfig,
    /// Bloom Filter用于快速检查
    bloom_filter: Arc<RwLock<BloomFilter>>,
    /// 精确的删除记录映射（使用DashMap支持并发）
    tombstones: Arc<DashMap<RecordId, TombstoneEntry>>,
    /// 按表分组的统计信息
    table_stats: Arc<RwLock<HashMap<String, TombstoneStats>>>,
    /// 清理任务句柄
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

/// 表级别的墓碑统计信息
#[derive(Debug, Clone, Default)]
pub struct TombstoneStats {
    /// 删除记录总数
    pub total_deleted: u64,
    /// 最后删除时间
    pub last_deleted_at: Option<DateTime<Utc>>,
    /// 待清理记录数
    pub pending_cleanup: u64,
}

/// 批量操作结果
#[derive(Debug)]
pub struct BatchOperationResult {
    pub processed: usize,
    pub skipped: usize,
    pub errors: Vec<String>,
}

impl TombstoneTracker {
    /// 创建新的墓碑追踪器
    pub fn new(config: TombstoneConfig) -> crate::Result<Self> {
        // 创建Bloom Filter
        let bloom_filter =
            BloomFilter::with_rate(config.expected_elements, config.false_positive_rate as f64);

        let tracker = Self {
            config,
            bloom_filter: Arc::new(RwLock::new(bloom_filter)),
            tombstones: Arc::new(DashMap::new()),
            table_stats: Arc::new(RwLock::new(HashMap::new())),
            cleanup_handle: None,
        };

        Ok(tracker)
    }

    /// 启动自动清理任务
    pub fn start_cleanup_task(&mut self) {
        let tombstones = self.tombstones.clone();
        let bloom_filter = self.bloom_filter.clone();
        let table_stats = self.table_stats.clone();
        let cleanup_interval = self.config.cleanup_interval;
        let retention_period = self.config.retention_period;

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(
                cleanup_interval.num_seconds() as u64,
            ));

            loop {
                interval.tick().await;

                let cutoff_time = Utc::now() - retention_period;
                let mut cleaned_count = 0;
                let mut table_updates: HashMap<String, u64> = HashMap::new();

                // 清理过期的墓碑记录
                tombstones.retain(|_id, entry| {
                    if entry.deleted_at < cutoff_time {
                        *table_updates.entry(entry.table_name.clone()).or_insert(0) += 1;
                        cleaned_count += 1;
                        false // 删除此记录
                    } else {
                        true // 保留此记录
                    }
                });

                // 更新统计信息
                if cleaned_count > 0 {
                    let mut stats = table_stats.write().await;
                    for (table_name, count) in table_updates {
                        if let Some(table_stat) = stats.get_mut(&table_name) {
                            table_stat.pending_cleanup =
                                table_stat.pending_cleanup.saturating_sub(count);
                        }
                    }

                    tracing::info!("Cleaned {} expired tombstone records", cleaned_count);

                    // 如果清理了大量记录，重建Bloom Filter
                    if cleaned_count > 10000 {
                        Self::rebuild_bloom_filter(&bloom_filter, &tombstones).await;
                    }
                }
            }
        });

        self.cleanup_handle = Some(handle);
    }

    /// 标记记录为已删除
    pub async fn mark_deleted(
        &self,
        id: RecordId,
        table_name: String,
        reason: Option<String>,
    ) -> crate::Result<()> {
        let entry = TombstoneEntry {
            id: id.clone(),
            deleted_at: Utc::now(),
            reason,
            table_name: table_name.clone(),
        };

        // 添加到Bloom Filter
        {
            let mut bloom = self.bloom_filter.write().await;
            bloom.insert(id.to_string().as_bytes());
        }

        // 添加到精确映射
        self.tombstones.insert(id, entry);

        // 更新统计信息
        {
            let mut stats = self.table_stats.write().await;
            let table_stat = stats
                .entry(table_name)
                .or_insert_with(TombstoneStats::default);
            table_stat.total_deleted += 1;
            table_stat.last_deleted_at = Some(Utc::now());
            table_stat.pending_cleanup += 1;
        }

        Ok(())
    }

    /// 批量标记记录为已删除
    pub async fn mark_deleted_batch(
        &self,
        ids: Vec<RecordId>,
        table_name: String,
        reason: Option<String>,
    ) -> crate::Result<BatchOperationResult> {
        let mut processed = 0;
        let mut skipped = 0;
        let errors = Vec::new();

        let deleted_at = Utc::now();

        // 批量更新Bloom Filter
        {
            let mut bloom = self.bloom_filter.write().await;
            for id in &ids {
                let key = id.to_string();
                bloom.insert(key.as_bytes());
            }
        }

        // 批量添加到映射
        for id in ids {
            if self.tombstones.contains_key(&id) {
                skipped += 1;
                continue;
            }

            let entry = TombstoneEntry {
                id: id.clone(),
                deleted_at,
                reason: reason.clone(),
                table_name: table_name.clone(),
            };

            self.tombstones.insert(id, entry);
            processed += 1;
        }

        // 更新统计信息
        if processed > 0 {
            let mut stats = self.table_stats.write().await;
            let table_stat = stats
                .entry(table_name)
                .or_insert_with(TombstoneStats::default);
            table_stat.total_deleted += processed as u64;
            table_stat.last_deleted_at = Some(deleted_at);
            table_stat.pending_cleanup += processed as u64;
        }

        Ok(BatchOperationResult {
            processed,
            skipped,
            errors,
        })
    }

    /// 检查记录是否已删除（快速检查）
    pub async fn is_deleted_fast(&self, id: &RecordId) -> bool {
        // 首先使用Bloom Filter进行快速检查
        let bloom = self.bloom_filter.read().await;
        if !bloom.contains(id.to_string().as_bytes()) {
            return false; // 肯定没有删除
        }

        // Bloom Filter可能有误报，需要精确检查
        drop(bloom); // 释放读锁
        self.tombstones.contains_key(id)
    }

    /// 检查记录是否已删除（精确检查）
    pub fn is_deleted_exact(&self, id: &RecordId) -> bool {
        self.tombstones.contains_key(id)
    }

    /// 获取删除记录的详细信息
    pub fn get_tombstone(&self, id: &RecordId) -> Option<TombstoneEntry> {
        self.tombstones.get(id).map(|entry| entry.clone())
    }

    /// 批量检查记录是否已删除
    pub async fn are_deleted_batch(&self, ids: &[RecordId]) -> Vec<bool> {
        let mut results = Vec::with_capacity(ids.len());

        // 首先用Bloom Filter批量检查
        let bloom = self.bloom_filter.read().await;
        let bloom_results: Vec<bool> = ids
            .iter()
            .map(|id| {
                let key = id.to_string();
                bloom.contains(key.as_bytes())
            })
            .collect();
        drop(bloom);

        // 对Bloom Filter返回true的记录进行精确检查
        for (i, &bloom_result) in bloom_results.iter().enumerate() {
            if !bloom_result {
                results.push(false); // Bloom Filter说不存在，肯定不存在
            } else {
                results.push(self.tombstones.contains_key(&ids[i])); // 精确检查
            }
        }

        results
    }

    /// 过滤搜索结果，移除已删除的记录
    pub async fn filter_deleted_records<T>(
        &self,
        results: Vec<(RecordId, T)>,
    ) -> Vec<(RecordId, T)> {
        if results.is_empty() {
            return results;
        }

        let ids: Vec<_> = results.iter().map(|(id, _)| id.clone()).collect();
        let deleted_flags = self.are_deleted_batch(&ids).await;

        results
            .into_iter()
            .zip(deleted_flags)
            .filter_map(
                |(result, is_deleted)| {
                    if !is_deleted {
                        Some(result)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// 获取表的墓碑统计信息
    pub async fn get_table_stats(&self, table_name: &str) -> Option<TombstoneStats> {
        let stats = self.table_stats.read().await;
        stats.get(table_name).cloned()
    }

    /// 获取所有表的墓碑统计信息
    pub async fn get_all_stats(&self) -> HashMap<String, TombstoneStats> {
        let stats = self.table_stats.read().await;
        stats.clone()
    }

    /// 手动触发清理过期记录
    pub async fn cleanup_expired(&self) -> usize {
        let cutoff_time = Utc::now() - self.config.retention_period;
        let mut cleaned_count = 0;
        let mut table_updates: HashMap<String, u64> = HashMap::new();

        self.tombstones.retain(|_id, entry| {
            if entry.deleted_at < cutoff_time {
                *table_updates.entry(entry.table_name.clone()).or_insert(0) += 1;
                cleaned_count += 1;
                false
            } else {
                true
            }
        });

        // 更新统计信息
        if cleaned_count > 0 {
            let mut stats = self.table_stats.write().await;
            for (table_name, count) in table_updates {
                if let Some(table_stat) = stats.get_mut(&table_name) {
                    table_stat.pending_cleanup = table_stat.pending_cleanup.saturating_sub(count);
                }
            }

            // 重建Bloom Filter
            Self::rebuild_bloom_filter(&self.bloom_filter, &self.tombstones).await;
        }

        cleaned_count
    }

    /// 重建Bloom Filter
    async fn rebuild_bloom_filter(
        bloom_filter: &Arc<RwLock<BloomFilter>>,
        tombstones: &Arc<DashMap<RecordId, TombstoneEntry>>,
    ) {
        let mut bloom = bloom_filter.write().await;
        bloom.clear();

        for entry in tombstones.iter() {
            bloom.insert(entry.key().to_string().as_bytes());
        }

        tracing::info!(
            "Rebuilt Bloom filter with {} tombstone records",
            tombstones.len()
        );
    }

    /// 获取内存使用统计
    pub fn get_memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            tombstone_count: self.tombstones.len(),
            bloom_filter_bits: self.config.expected_elements * 10, // 估算
            estimated_bytes: self.tombstones.len() * std::mem::size_of::<TombstoneEntry>(),
        }
    }

    /// 清除所有墓碑记录（谨慎使用）
    pub async fn clear_all(&self) {
        self.tombstones.clear();

        {
            let mut bloom = self.bloom_filter.write().await;
            bloom.clear();
        }

        {
            let mut stats = self.table_stats.write().await;
            stats.clear();
        }

        tracing::warn!("All tombstone records have been cleared");
    }

    /// 恢复记录（移除墓碑标记）
    pub async fn restore_record(&self, id: &RecordId) -> bool {
        if let Some((_, entry)) = self.tombstones.remove(id) {
            // 更新统计信息
            let mut stats = self.table_stats.write().await;
            if let Some(table_stat) = stats.get_mut(&entry.table_name) {
                table_stat.total_deleted = table_stat.total_deleted.saturating_sub(1);
                table_stat.pending_cleanup = table_stat.pending_cleanup.saturating_sub(1);
            }

            // 注意：不从Bloom Filter中删除，因为Bloom Filter不支持删除
            // 这意味着快速检查可能会有误报，但精确检查会准确

            true
        } else {
            false
        }
    }
}

/// 内存使用统计
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub tombstone_count: usize,
    pub bloom_filter_bits: usize,
    pub estimated_bytes: usize,
}

impl Drop for TombstoneTracker {
    fn drop(&mut self) {
        if let Some(handle) = self.cleanup_handle.take() {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RecordId;

    #[tokio::test]
    async fn test_basic_tombstone_operations() -> crate::Result<()> {
        let config = TombstoneConfig::default();
        let tracker = TombstoneTracker::new(config)?;

        let id = RecordId::from("test_record_1");
        let table_name = "test_table".to_string();

        // 检查记录最初未删除
        assert!(!tracker.is_deleted_fast(&id).await);
        assert!(!tracker.is_deleted_exact(&id));

        // 标记为删除
        tracker
            .mark_deleted(
                id.clone(),
                table_name.clone(),
                Some("test deletion".to_string()),
            )
            .await?;

        // 验证记录已删除
        assert!(tracker.is_deleted_fast(&id).await);
        assert!(tracker.is_deleted_exact(&id));

        // 获取墓碑信息
        let tombstone = tracker.get_tombstone(&id);
        assert!(tombstone.is_some());
        assert_eq!(tombstone.unwrap().table_name, table_name);

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_operations() -> crate::Result<()> {
        let config = TombstoneConfig::default();
        let tracker = TombstoneTracker::new(config)?;

        let ids = vec![
            RecordId::from("record_1"),
            RecordId::from("record_2"),
            RecordId::from("record_3"),
        ];
        let table_name = "test_table".to_string();

        // 批量标记删除
        let result = tracker
            .mark_deleted_batch(ids.clone(), table_name, None)
            .await?;
        assert_eq!(result.processed, 3);
        assert_eq!(result.skipped, 0);

        // 批量检查
        let deleted_flags = tracker.are_deleted_batch(&ids).await;
        assert_eq!(deleted_flags, vec![true, true, true]);

        Ok(())
    }

    #[tokio::test]
    async fn test_filter_search_results() -> crate::Result<()> {
        let config = TombstoneConfig::default();
        let tracker = TombstoneTracker::new(config)?;

        let results = vec![
            (RecordId::from("record_1"), 0.5f32),
            (RecordId::from("record_2"), 0.7f32),
            (RecordId::from("record_3"), 0.3f32),
        ];

        // 标记record_2为删除
        tracker
            .mark_deleted(RecordId::from("record_2"), "test_table".to_string(), None)
            .await?;

        // 过滤结果
        let filtered = tracker.filter_deleted_records(results).await;

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, RecordId::from("record_1"));
        assert_eq!(filtered[1].0, RecordId::from("record_3"));

        Ok(())
    }

    #[tokio::test]
    async fn test_restore_record() -> crate::Result<()> {
        let config = TombstoneConfig::default();
        let tracker = TombstoneTracker::new(config)?;

        let id = RecordId::from("test_record");

        // 标记删除
        tracker
            .mark_deleted(id.clone(), "test_table".to_string(), None)
            .await?;
        assert!(tracker.is_deleted_exact(&id));

        // 恢复记录
        let restored = tracker.restore_record(&id).await;
        assert!(restored);
        assert!(!tracker.is_deleted_exact(&id));

        // 再次恢复应该返回false
        let restored_again = tracker.restore_record(&id).await;
        assert!(!restored_again);

        Ok(())
    }
}
