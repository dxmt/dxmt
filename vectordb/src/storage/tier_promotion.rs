use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

/// 访问模式跟踪器
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AccessPattern {
    /// 最近访问时间线（保留最近100次访问）
    access_timeline: VecDeque<DateTime<Utc>>,
    /// 总访问次数
    total_accesses: u64,
    /// 首次访问时间
    first_access: DateTime<Utc>,
    /// 最后访问时间
    last_access: DateTime<Utc>,
}

impl AccessPattern {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            access_timeline: VecDeque::new(),
            total_accesses: 0,
            first_access: now,
            last_access: now,
        }
    }

    /// 记录一次访问
    pub fn record_access(&mut self) {
        let now = Utc::now();
        self.total_accesses += 1;
        self.last_access = now;

        // 保留最近100次访问
        self.access_timeline.push_back(now);
        if self.access_timeline.len() > 100 {
            self.access_timeline.pop_front();
        }
    }

    /// 计算热度分数（0.0-1.0）
    /// 基于：访问频率 × 时间衰减 × 访问新鲜度
    pub fn hotness_score(&self, time_window: Duration) -> f64 {
        if self.total_accesses == 0 {
            return 0.0;
        }

        let now = Utc::now();

        // 1. 计算访问频率（时间窗口内的访问次数）
        let recent_accesses = self
            .access_timeline
            .iter()
            .filter(|t| now.signed_duration_since(**t) <= time_window)
            .count();

        let frequency = recent_accesses as f64 / 100.0; // 归一化到 0-1

        // 2. 时间衰减（距离最后访问的时间）
        let time_since_last = now.signed_duration_since(self.last_access);
        let time_decay = if time_since_last > time_window {
            0.0
        } else {
            1.0 - (time_since_last.num_seconds() as f64 / time_window.num_seconds() as f64)
        };

        // 3. 综合热度分数
        frequency * 0.7 + time_decay * 0.3
    }

    /// 获取总访问次数
    pub fn total_accesses(&self) -> u64 {
        self.total_accesses
    }

    /// 获取最后访问时间
    pub fn last_access(&self) -> DateTime<Utc> {
        self.last_access
    }
}

impl Default for AccessPattern {
    fn default() -> Self {
        Self::new()
    }
}

/// 层级枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    Hot,   // Memory
    Warm,  // SSD
    Cold,  // S3
}

/// 向量提升策略配置
#[derive(Debug, Clone)]
pub struct TierPromotionConfig {
    /// 提升到热层的热度阈值（0.0-1.0）
    pub hot_threshold: f64,
    /// 提升到温层的热度阈值（0.0-1.0）
    pub warm_threshold: f64,
    /// 降级到温层的热度阈值（低于此值从热层降级）
    pub demote_to_warm_threshold: f64,
    /// 降级到冷层的热度阈值（低于此值从温层降级）
    pub demote_to_cold_threshold: f64,
    /// 热度计算的时间窗口（秒）
    pub time_window_secs: i64,
    /// 自动提升是否启用
    pub auto_promotion_enabled: bool,
    /// 自动降级是否启用
    pub auto_demotion_enabled: bool,
}

impl Default for TierPromotionConfig {
    fn default() -> Self {
        Self {
            hot_threshold: 0.7,
            warm_threshold: 0.4,
            demote_to_warm_threshold: 0.5,
            demote_to_cold_threshold: 0.2,
            time_window_secs: 3600, // 1小时
            auto_promotion_enabled: true,
            auto_demotion_enabled: true,
        }
    }
}

/// 向量提升策略管理器
pub struct TierPromotionPolicy {
    /// 配置
    config: TierPromotionConfig,
    /// 访问模式跟踪（vector_id -> AccessPattern）
    access_patterns: Arc<RwLock<HashMap<String, AccessPattern>>>,
    /// 当前向量所在层级（vector_id -> Tier）
    vector_tiers: Arc<RwLock<HashMap<String, Tier>>>,
}

impl TierPromotionPolicy {
    pub fn new(config: TierPromotionConfig) -> Self {
        Self {
            config,
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
            vector_tiers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 记录向量访问
    pub fn record_access(&self, vector_id: &str) {
        let mut patterns = self.access_patterns.write().unwrap();
        patterns
            .entry(vector_id.to_string())
            .or_default()
            .record_access();
    }

    /// 计算向量热度分数
    pub fn get_hotness_score(&self, vector_id: &str) -> f64 {
        let patterns = self.access_patterns.read().unwrap();
        patterns
            .get(vector_id)
            .map(|p| {
                p.hotness_score(Duration::seconds(self.config.time_window_secs))
            })
            .unwrap_or(0.0)
    }

    /// 获取向量当前所在层级
    pub fn get_current_tier(&self, vector_id: &str) -> Option<Tier> {
        let tiers = self.vector_tiers.read().unwrap();
        tiers.get(vector_id).copied()
    }

    /// 设置向量当前所在层级
    pub fn set_current_tier(&self, vector_id: &str, tier: Tier) {
        let mut tiers = self.vector_tiers.write().unwrap();
        tiers.insert(vector_id.to_string(), tier);
    }

    /// 判断是否应该提升到热层
    pub fn should_promote_to_hot(&self, vector_id: &str) -> bool {
        if !self.config.auto_promotion_enabled {
            return false;
        }

        let hotness = self.get_hotness_score(vector_id);
        let current_tier = self.get_current_tier(vector_id);

        match current_tier {
            Some(Tier::Hot) => false, // 已经在热层
            Some(Tier::Warm) | Some(Tier::Cold) | None => {
                hotness >= self.config.hot_threshold
            }
        }
    }

    /// 判断是否应该提升到温层
    pub fn should_promote_to_warm(&self, vector_id: &str) -> bool {
        if !self.config.auto_promotion_enabled {
            return false;
        }

        let hotness = self.get_hotness_score(vector_id);
        let current_tier = self.get_current_tier(vector_id);

        match current_tier {
            Some(Tier::Hot) | Some(Tier::Warm) => false, // 已经在热层或温层
            Some(Tier::Cold) | None => hotness >= self.config.warm_threshold,
        }
    }

    /// 判断是否应该降级到温层
    pub fn should_demote_to_warm(&self, vector_id: &str) -> bool {
        if !self.config.auto_demotion_enabled {
            return false;
        }

        let hotness = self.get_hotness_score(vector_id);
        let current_tier = self.get_current_tier(vector_id);

        match current_tier {
            Some(Tier::Hot) => hotness < self.config.demote_to_warm_threshold,
            _ => false,
        }
    }

    /// 判断是否应该降级到冷层
    pub fn should_demote_to_cold(&self, vector_id: &str) -> bool {
        if !self.config.auto_demotion_enabled {
            return false;
        }

        let hotness = self.get_hotness_score(vector_id);
        let current_tier = self.get_current_tier(vector_id);

        match current_tier {
            Some(Tier::Warm) => hotness < self.config.demote_to_cold_threshold,
            _ => false,
        }
    }

    /// 获取建议的目标层级
    pub fn get_recommended_tier(&self, vector_id: &str) -> Tier {
        let hotness = self.get_hotness_score(vector_id);

        if hotness >= self.config.hot_threshold {
            Tier::Hot
        } else if hotness >= self.config.warm_threshold {
            Tier::Warm
        } else {
            Tier::Cold
        }
    }

    /// 获取需要提升的向量列表（vector_id, 目标层级）
    pub fn get_vectors_to_promote(&self) -> Vec<(String, Tier)> {
        let patterns = self.access_patterns.read().unwrap();
        let mut promotions = Vec::new();

        for vector_id in patterns.keys() {
            if self.should_promote_to_hot(vector_id) {
                promotions.push((vector_id.clone(), Tier::Hot));
            } else if self.should_promote_to_warm(vector_id) {
                promotions.push((vector_id.clone(), Tier::Warm));
            }
        }

        promotions
    }

    /// 获取需要降级的向量列表（vector_id, 目标层级）
    pub fn get_vectors_to_demote(&self) -> Vec<(String, Tier)> {
        let patterns = self.access_patterns.read().unwrap();
        let mut demotions = Vec::new();

        for vector_id in patterns.keys() {
            if self.should_demote_to_cold(vector_id) {
                demotions.push((vector_id.clone(), Tier::Cold));
            } else if self.should_demote_to_warm(vector_id) {
                demotions.push((vector_id.clone(), Tier::Warm));
            }
        }

        demotions
    }

    /// 清理旧的访问模式（超过N天没有访问的）
    pub fn cleanup_old_patterns(&self, days: i64) {
        let mut patterns = self.access_patterns.write().unwrap();
        let mut tiers = self.vector_tiers.write().unwrap();

        let cutoff = Utc::now() - Duration::days(days);

        patterns.retain(|vector_id, pattern| {
            let keep = pattern.last_access() > cutoff;
            if !keep {
                // 同时清理层级信息
                tiers.remove(vector_id);
            }
            keep
        });
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> TierStats {
        let patterns = self.access_patterns.read().unwrap();
        let tiers = self.vector_tiers.read().unwrap();

        let mut hot_count = 0;
        let mut warm_count = 0;
        let mut cold_count = 0;

        for tier in tiers.values() {
            match tier {
                Tier::Hot => hot_count += 1,
                Tier::Warm => warm_count += 1,
                Tier::Cold => cold_count += 1,
            }
        }

        TierStats {
            total_vectors: patterns.len(),
            hot_tier_count: hot_count,
            warm_tier_count: warm_count,
            cold_tier_count: cold_count,
            total_accesses: patterns.values().map(|p| p.total_accesses()).sum(),
        }
    }
}

/// 层级统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierStats {
    pub total_vectors: usize,
    pub hot_tier_count: usize,
    pub warm_tier_count: usize,
    pub cold_tier_count: usize,
    pub total_accesses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_pattern_hotness() {
        let mut pattern = AccessPattern::new();

        // 记录多次访问
        for _ in 0..10 {
            pattern.record_access();
        }

        let score = pattern.hotness_score(Duration::hours(1));
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_tier_promotion_policy() {
        let config = TierPromotionConfig::default();
        let policy = TierPromotionPolicy::new(config);

        // 记录多次访问
        for _ in 0..20 {
            policy.record_access("vec1");
        }

        // 设置当前在冷层
        policy.set_current_tier("vec1", Tier::Cold);

        // 检查是否应该提升
        let should_promote = policy.should_promote_to_hot("vec1")
            || policy.should_promote_to_warm("vec1");
        assert!(should_promote);
    }

    #[test]
    fn test_get_recommended_tier() {
        let config = TierPromotionConfig::default();
        let policy = TierPromotionPolicy::new(config);

        // 高频访问 (need frequency >= 1.0 so hotness = 0.7*1.0 + 0.3*1.0 >= 0.7 threshold)
        for _ in 0..100 {
            policy.record_access("hot_vec");
        }

        // 中频访问 (need hotness >= 0.4 but < 0.7: frequency=0.2 → 0.2*0.7+1.0*0.3=0.44)
        for _ in 0..20 {
            policy.record_access("warm_vec");
        }

        // 低频访问
        policy.record_access("cold_vec");

        assert_eq!(policy.get_recommended_tier("hot_vec"), Tier::Hot);
        assert_eq!(policy.get_recommended_tier("warm_vec"), Tier::Warm);
        assert_eq!(policy.get_recommended_tier("cold_vec"), Tier::Cold);
    }
}
