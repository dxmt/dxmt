//! Payload索引 - 用于元数据过滤的性能优化
//!
//! payload 索引，支持：
//! - Cardinality估计（filter选择性评估）
//! - 倒排索引加速查找
//! - 智能PRE-FILTERING决策

use crate::types::{
    record::{Condition, Filter, FilterOperation},
    vector::RecordId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Payload索引 - 为每个字段维护倒排索引
#[derive(Debug, Clone)]
pub struct PayloadIndex {
    /// 字段名 -> 字段索引
    field_indices: HashMap<String, FieldIndex>,
    /// 总记录数
    total_records: usize,
}

/// 单个字段的索引
#[derive(Debug, Clone)]
struct FieldIndex {
    /// 值 -> 包含该值的记录ID集合
    value_to_ids: HashMap<ValueKey, HashSet<RecordId>>,
    /// 字段类型（推断）
    #[allow(dead_code)]
    field_type: FieldType,
    /// 不同值的数量
    distinct_count: usize,
    /// 字段统计信息（用于更准确的基数估计）
    stats: FieldStats,
}

/// 字段统计信息
#[derive(Debug, Clone)]
struct FieldStats {
    /// 数值字段的最小值
    numeric_min: Option<f64>,
    /// 数值字段的最大值
    numeric_max: Option<f64>,
    /// 字符串前缀集合（用于StartsWith估计，最多保留1000个前缀）
    string_prefixes: HashSet<String>,
    /// 总采样数（用于估计）
    sample_count: usize,
}

/// 值的键（用于HashMap）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ValueKey {
    String(String),
    Number(i64),         // 整数
    Float(OrderedFloat), // 浮点数（需要特殊处理）
    Bool(bool),
    Null,
}

/// 可排序的浮点数包装器
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct OrderedFloat(i64); // 将f64转为i64存储（乘以1000000保留6位小数）

impl From<f64> for OrderedFloat {
    fn from(f: f64) -> Self {
        OrderedFloat((f * 1_000_000.0) as i64)
    }
}

impl From<OrderedFloat> for f64 {
    fn from(of: OrderedFloat) -> Self {
        of.0 as f64 / 1_000_000.0
    }
}

/// 字段类型（推断）
#[derive(Debug, Clone, PartialEq)]
enum FieldType {
    String,
    Number,
    Boolean,
    Mixed, // 混合类型
}

impl FieldStats {
    fn new() -> Self {
        Self {
            numeric_min: None,
            numeric_max: None,
            string_prefixes: HashSet::new(),
            sample_count: 0,
        }
    }

    /// 更新数值统计
    fn update_numeric(&mut self, value: f64) {
        self.numeric_min = Some(self.numeric_min.map_or(value, |min| min.min(value)));
        self.numeric_max = Some(self.numeric_max.map_or(value, |max| max.max(value)));
        self.sample_count += 1;
    }

    /// 更新字符串统计（收集前缀）
    fn update_string(&mut self, value: &str) {
        // 收集2-4字符的前缀用于StartsWith估计
        // IMPORTANT: Use .chars() to handle multi-byte UTF-8 characters correctly
        if self.string_prefixes.len() < 1000 {
            // Get first 2 characters (by character count, not bytes)
            let char_count = value.chars().count();
            if char_count >= 2 {
                let prefix: String = value.chars().take(2).collect();
                self.string_prefixes.insert(prefix);
            }
            // Get first 3 characters (by character count, not bytes)
            if char_count >= 3 {
                let prefix: String = value.chars().take(3).collect();
                self.string_prefixes.insert(prefix);
            }
        }
        self.sample_count += 1;
    }

    /// 估计范围查询的选择性
    fn estimate_range_selectivity(&self, op: &FilterOperation) -> f64 {
        match (self.numeric_min, self.numeric_max) {
            (Some(min), Some(max)) if max > min => {
                let range = max - min;
                match op {
                    FilterOperation::GreaterThan(v) | FilterOperation::GreaterThanOrEqual(v) => {
                        if let Some(threshold) = Self::extract_numeric(v) {
                            if threshold <= min {
                                return 1.0; // 所有值都满足
                            } else if threshold >= max {
                                return 0.01; // 几乎没有值满足
                            } else {
                                return ((max - threshold) / range).max(0.01);
                            }
                        }
                    }
                    FilterOperation::LessThan(v) | FilterOperation::LessThanOrEqual(v) => {
                        if let Some(threshold) = Self::extract_numeric(v) {
                            if threshold >= max {
                                return 1.0; // 所有值都满足
                            } else if threshold <= min {
                                return 0.01; // 几乎没有值满足
                            } else {
                                return ((threshold - min) / range).max(0.01);
                            }
                        }
                    }
                    _ => {}
                }
                0.3 // 默认保守估计
            }
            _ => 0.3, // 没有统计信息时使用保守估计
        }
    }

    /// 估计字符串匹配的选择性
    fn estimate_string_selectivity(&self, op: &FilterOperation) -> f64 {
        match op {
            FilterOperation::StartsWith(prefix) => {
                if self.string_prefixes.is_empty() || self.sample_count == 0 {
                    return 0.2; // 默认估计
                }
                // 计算有多少前缀匹配
                let prefix_lower = prefix.to_lowercase();
                let matching = self
                    .string_prefixes
                    .iter()
                    .filter(|p| {
                        prefix_lower.starts_with(&p.to_lowercase())
                            || p.to_lowercase().starts_with(&prefix_lower)
                    })
                    .count();

                if matching == 0 {
                    return 0.01; // 没有匹配的前缀
                }

                // 基于匹配前缀比例估计
                let ratio = matching as f64 / self.string_prefixes.len() as f64;
                ratio.clamp(0.01, 0.5)
            }
            FilterOperation::Contains(_) => {
                // Contains更难估计，使用保守值
                0.25
            }
            FilterOperation::EndsWith(_) => {
                // EndsWith也使用保守值（需要后缀索引才能更准确）
                0.15
            }
            _ => 0.2,
        }
    }

    /// 从JSON值提取数值
    fn extract_numeric(value: &serde_json::Value) -> Option<f64> {
        match value {
            serde_json::Value::Number(n) => n.as_f64(),
            _ => None,
        }
    }
}

/// Cardinality估计结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardinalityEstimate {
    /// 预期匹配的记录数（最小值）
    pub min: usize,
    /// 预期匹配的记录数（最大值）
    pub max: usize,
    /// 预期匹配的记录数（期望值）
    pub exp: usize,
    /// 选择性（0.0-1.0，越小越selective）
    pub selectivity: f64,
}

impl PayloadIndex {
    /// 创建新的PayloadIndex
    pub fn new() -> Self {
        Self {
            field_indices: HashMap::new(),
            total_records: 0,
        }
    }

    /// 添加记录到索引
    pub fn insert_record(&mut self, record_id: &RecordId, payload: &serde_json::Value) {
        self.total_records += 1;

        if let Some(obj) = payload.as_object() {
            for (key, value) in obj {
                self.index_field(key, value, record_id);
            }
        }
    }

    /// 更新记录索引
    pub fn update_record(
        &mut self,
        record_id: &RecordId,
        old_payload: &serde_json::Value,
        new_payload: &serde_json::Value,
    ) {
        // 先删除旧的索引
        self.remove_record(record_id, old_payload);
        // 再添加新的索引
        self.insert_record(record_id, new_payload);
    }

    /// 从索引中删除记录
    pub fn remove_record(&mut self, record_id: &RecordId, payload: &serde_json::Value) {
        if self.total_records > 0 {
            self.total_records -= 1;
        }

        if let Some(obj) = payload.as_object() {
            for (key, value) in obj {
                if let Some(field_index) = self.field_indices.get_mut(key) {
                    if let Some(value_key) = Self::to_value_key(value) {
                        if let Some(ids) = field_index.value_to_ids.get_mut(&value_key) {
                            ids.remove(record_id);
                            if ids.is_empty() {
                                field_index.value_to_ids.remove(&value_key);
                                field_index.distinct_count = field_index.value_to_ids.len();
                            }
                        }
                    }
                }
            }
        }
    }

    /// 估计filter的cardinality
    pub fn estimate_cardinality(&self, filter: &Filter) -> CardinalityEstimate {
        // 处理must条件（AND）
        let must_estimate = if let Some(conditions) = &filter.must {
            self.estimate_conditions_and(conditions)
        } else {
            CardinalityEstimate {
                min: self.total_records,
                max: self.total_records,
                exp: self.total_records,
                selectivity: 1.0,
            }
        };

        // 处理must_not条件（NOT）
        let must_not_estimate = if let Some(conditions) = &filter.must_not {
            let pos_estimate = self.estimate_conditions_or(conditions);
            CardinalityEstimate {
                min: self.total_records.saturating_sub(pos_estimate.max),
                max: self.total_records.saturating_sub(pos_estimate.min),
                exp: self.total_records.saturating_sub(pos_estimate.exp),
                selectivity: 1.0 - pos_estimate.selectivity,
            }
        } else {
            // No must_not conditions - return full selectivity
            CardinalityEstimate {
                min: self.total_records,
                max: self.total_records,
                exp: self.total_records,
                selectivity: 1.0,
            }
        };

        // 处理should条件（OR）
        let should_estimate = if let Some(conditions) = &filter.should {
            if conditions.is_empty() {
                must_estimate.clone()
            } else {
                self.estimate_conditions_or(conditions)
            }
        } else {
            must_estimate.clone()
        };

        // 组合must和must_not（取交集，近似为相乘）
        let combined_min = must_estimate.min.min(must_not_estimate.min);
        let combined_max = must_estimate.max.min(must_not_estimate.max);
        let combined_exp = (must_estimate.exp as f64 * must_not_estimate.selectivity) as usize;

        // 如果有should条件，取并集（取最大值）
        let final_estimate =
            if filter.should.is_some() && !filter.should.as_ref().unwrap().is_empty() {
                CardinalityEstimate {
                    min: combined_min.max(should_estimate.min),
                    max: combined_max.max(should_estimate.max),
                    exp: combined_exp.max(should_estimate.exp),
                    selectivity: (combined_exp.max(should_estimate.exp) as f64
                        / self.total_records.max(1) as f64),
                }
            } else {
                CardinalityEstimate {
                    min: combined_min,
                    max: combined_max,
                    exp: combined_exp,
                    selectivity: combined_exp as f64 / self.total_records.max(1) as f64,
                }
            };

        final_estimate
    }

    /// 获取满足filter的记录ID集合（用于PRE-FILTERING）
    /// `true` when no record can satisfy this filter, decided from the index alone and
    /// without building an id set.
    ///
    /// A `must` condition matching none of its field's indexed values matches no record, and
    /// one failing `must` empties the whole filter whatever else it carries: `should` only
    /// adds a requirement, `must_not` only removes rows. Walking the distinct values costs the
    /// field's cardinality and clones nothing, where [`Self::get_matching_ids`] materialises
    /// every id of every condition before intersecting -- on a 20,000-row table a `<= 600`
    /// accepting 18,000 of them built an 18,000-string set to answer a filter whose result is
    /// empty, which is how the first version of this shortcut came out slower than the scan it
    /// replaced.
    ///
    /// Only conditions whose matches are all indexed can prove anything here; see
    /// [`Self::condition_is_exact`]. `id` is refused because
    /// [`crate::types::record::Record::matches_filter`] reads the record's own id, while this
    /// index only ever knows a payload field of that name.
    pub fn cannot_match_anything(&self, filter: &Filter) -> bool {
        let Some(must) = &filter.must else {
            return false;
        };
        // Grouped by key, because conditions on the same key all judge the same value and so
        // have to be tested together. `bucket >= 400` and `bucket <= 600` each accept plenty
        // of rows on their own; what is empty is their intersection, and it is empty exactly
        // when no indexed value falls between them.
        let mut by_key: HashMap<&str, Vec<&Condition>> = HashMap::new();
        for condition in must {
            // A condition that cannot prove anything is left out rather than refusing the
            // whole filter: proving that a subset of the `must` is unsatisfiable proves the
            // whole `must` is.
            if condition.key == "id" || !Self::condition_is_exact(condition) {
                continue;
            }
            by_key
                .entry(condition.key.as_str())
                .or_default()
                .push(condition);
        }
        by_key.into_iter().any(|(key, conditions)| {
            match self.field_indices.get(key) {
                // No record carries a scalar under this key, so none can match a condition
                // that needs one.
                None => true,
                Some(field) => !field.value_to_ids.iter().any(|(value_key, ids)| {
                    !ids.is_empty()
                        && conditions
                            .iter()
                            .all(|c| Self::value_key_matches(value_key, &c.operation))
                }),
            }
        })
    }

    /// Whether every record this condition accepts carries a value the index holds.
    ///
    /// This index keys the scalars [`Self::to_value_key`] can encode, so a record whose value
    /// for the key is an array or an object is absent from it. That only matters for a
    /// condition such a record could satisfy -- and most cannot: a range needs `as_f64`, a
    /// substring test needs `as_str`, an equality against a scalar needs a scalar. `NotEqual`
    /// and `NotIn` are the two that can, because an array is not equal to 3, and they are what
    /// makes this index an approximation of `matches_filter` rather than a restatement of it.
    fn condition_is_exact(condition: &Condition) -> bool {
        match &condition.operation {
            FilterOperation::Equal(value) => Self::to_value_key(value).is_some(),
            FilterOperation::In(values) => values.iter().all(|v| Self::to_value_key(v).is_some()),
            FilterOperation::GreaterThan(target)
            | FilterOperation::GreaterThanOrEqual(target)
            | FilterOperation::LessThan(target)
            | FilterOperation::LessThanOrEqual(target) => target.as_f64().is_some(),
            FilterOperation::Contains(_)
            | FilterOperation::StartsWith(_)
            | FilterOperation::EndsWith(_) => true,
            FilterOperation::NotEqual(_) | FilterOperation::NotIn(_) => false,
        }
    }

    /// Whether one indexed value satisfies the condition.
    ///
    /// The single place the comparison lives, so that asking "does anything match?" and asking
    /// "which records match?" cannot drift apart.
    fn value_key_matches(value_key: &ValueKey, operation: &FilterOperation) -> bool {
        match operation {
            FilterOperation::Equal(value) => {
                Self::to_value_key(value).is_some_and(|k| &k == value_key)
            }
            FilterOperation::NotEqual(value) => {
                Self::to_value_key(value).is_some_and(|k| &k != value_key)
            }
            FilterOperation::In(values) => values
                .iter()
                .filter_map(Self::to_value_key)
                .any(|k| &k == value_key),
            // An unencodable target is dropped from the exclusion set rather than excluding
            // everything, which is what the id-building version does too.
            FilterOperation::NotIn(values) => !values
                .iter()
                .filter_map(Self::to_value_key)
                .any(|k| &k == value_key),
            FilterOperation::GreaterThan(target) => Self::as_number(value_key)
                .zip(target.as_f64())
                .is_some_and(|(v, t)| v > t),
            FilterOperation::GreaterThanOrEqual(target) => Self::as_number(value_key)
                .zip(target.as_f64())
                .is_some_and(|(v, t)| v >= t),
            FilterOperation::LessThan(target) => Self::as_number(value_key)
                .zip(target.as_f64())
                .is_some_and(|(v, t)| v < t),
            FilterOperation::LessThanOrEqual(target) => Self::as_number(value_key)
                .zip(target.as_f64())
                .is_some_and(|(v, t)| v <= t),
            FilterOperation::Contains(substr) => {
                matches!(value_key, ValueKey::String(s) if s.contains(substr.as_str()))
            }
            FilterOperation::StartsWith(prefix) => {
                matches!(value_key, ValueKey::String(s) if s.starts_with(prefix.as_str()))
            }
            FilterOperation::EndsWith(suffix) => {
                matches!(value_key, ValueKey::String(s) if s.ends_with(suffix.as_str()))
            }
        }
    }

    /// The number an indexed value compares as, for the range operations.
    fn as_number(value_key: &ValueKey) -> Option<f64> {
        match value_key {
            ValueKey::Number(v) => Some(*v as f64),
            ValueKey::Float(f) => Some((*f).into()),
            _ => None,
        }
    }

    pub fn get_matching_ids(&self, filter: &Filter) -> HashSet<RecordId> {
        let mut result = HashSet::new();

        // 处理must条件（AND）
        if let Some(conditions) = &filter.must {
            if conditions.is_empty() {
                return result;
            }

            // 获取第一个条件的结果
            if let Some(first) = conditions.first() {
                result = self.get_condition_ids(first);
            }

            // 与其他条件取交集
            for condition in conditions.iter().skip(1) {
                let ids = self.get_condition_ids(condition);
                result.retain(|id| ids.contains(id));
                if result.is_empty() {
                    break; // 早期退出
                }
            }
        }

        // 处理must_not条件（NOT）- 从结果中移除
        if let Some(conditions) = &filter.must_not {
            for condition in conditions {
                let ids = self.get_condition_ids(condition);
                result.retain(|id| !ids.contains(id));
            }
        }

        // 处理should条件（OR）- 添加到结果
        if let Some(conditions) = &filter.should {
            if !conditions.is_empty() && filter.must.is_none() {
                // 如果没有must条件，should条件作为初始集合
                for condition in conditions {
                    let ids = self.get_condition_ids(condition);
                    result.extend(ids);
                }
            }
        }

        result
    }

    /// 估计多个条件的AND组合
    fn estimate_conditions_and(&self, conditions: &[Condition]) -> CardinalityEstimate {
        if conditions.is_empty() {
            return CardinalityEstimate {
                min: self.total_records,
                max: self.total_records,
                exp: self.total_records,
                selectivity: 1.0,
            };
        }

        // AND条件：选择性相乘
        let mut combined_selectivity = 1.0;
        let mut min_count = self.total_records;
        let mut max_count = self.total_records;

        for condition in conditions {
            let estimate = self.estimate_single_condition(condition);
            combined_selectivity *= estimate.selectivity;
            min_count = min_count.min(estimate.min);
            max_count = max_count.min(estimate.max);
        }

        CardinalityEstimate {
            min: min_count,
            max: max_count,
            exp: (self.total_records as f64 * combined_selectivity) as usize,
            selectivity: combined_selectivity,
        }
    }

    /// 估计多个条件的OR组合
    fn estimate_conditions_or(&self, conditions: &[Condition]) -> CardinalityEstimate {
        if conditions.is_empty() {
            return CardinalityEstimate {
                min: 0,
                max: 0,
                exp: 0,
                selectivity: 0.0,
            };
        }

        // OR条件：选择性相加（但不超过1.0）
        let mut combined_selectivity = 0.0;
        let mut total_min = 0;
        let mut total_max = 0;

        for condition in conditions {
            let estimate = self.estimate_single_condition(condition);
            combined_selectivity = (combined_selectivity + estimate.selectivity).min(1.0);
            total_min += estimate.min;
            total_max += estimate.max;
        }

        // OR的max不能超过总记录数
        total_max = total_max.min(self.total_records);
        total_min = total_min.min(self.total_records);

        CardinalityEstimate {
            min: total_min,
            max: total_max,
            exp: (self.total_records as f64 * combined_selectivity) as usize,
            selectivity: combined_selectivity,
        }
    }

    /// 估计单个条件的选择性
    fn estimate_single_condition(&self, condition: &Condition) -> CardinalityEstimate {
        let field_index = match self.field_indices.get(&condition.key) {
            Some(idx) => idx,
            None => {
                // No record carries this field, so nothing can match it. The comment here
                // used to call 0.5 a conservative estimate; it is the opposite. The index
                // holds every field any record has written, so a missing key is not missing
                // information -- it is the answer.
                //
                // What the old estimate cost: a filter on an absent field reported half the
                // table as matching, the search took that for a broad filter and ran post
                // filtering over 600 fetched candidates, and one such query on a 5M x 1536
                // table took 16.6 s to return nothing while holding the table read lock for
                // 4.6 s. The same query without a filter takes 3.6 ms
                // (design/DISK_INDEX.md 5.5.1).
                return CardinalityEstimate {
                    min: 0,
                    max: 0,
                    exp: 0,
                    selectivity: 0.0,
                };
            }
        };

        match &condition.operation {
            FilterOperation::Equal(value) => {
                if let Some(value_key) = Self::to_value_key(value) {
                    if let Some(ids) = field_index.value_to_ids.get(&value_key) {
                        let count = ids.len();
                        return CardinalityEstimate {
                            min: count,
                            max: count,
                            exp: count,
                            selectivity: count as f64 / self.total_records.max(1) as f64,
                        };
                    }
                }
                // 值不存在
                CardinalityEstimate {
                    min: 0,
                    max: 0,
                    exp: 0,
                    selectivity: 0.0,
                }
            }
            FilterOperation::NotEqual(value) => {
                let equal_estimate = self.estimate_single_condition(&Condition {
                    key: condition.key.clone(),
                    operation: FilterOperation::Equal(value.clone()),
                });
                CardinalityEstimate {
                    min: self.total_records.saturating_sub(equal_estimate.max),
                    max: self.total_records.saturating_sub(equal_estimate.min),
                    exp: self.total_records.saturating_sub(equal_estimate.exp),
                    selectivity: 1.0 - equal_estimate.selectivity,
                }
            }
            FilterOperation::In(values) => {
                let mut total_count = 0;
                for value in values {
                    if let Some(value_key) = Self::to_value_key(value) {
                        if let Some(ids) = field_index.value_to_ids.get(&value_key) {
                            total_count += ids.len();
                        }
                    }
                }
                // In操作可能有重复ID，实际count可能更少
                total_count = total_count.min(self.total_records);
                CardinalityEstimate {
                    min: total_count.saturating_sub(values.len().saturating_sub(1)),
                    max: total_count,
                    exp: total_count,
                    selectivity: total_count as f64 / self.total_records.max(1) as f64,
                }
            }
            FilterOperation::GreaterThan(_)
            | FilterOperation::LessThan(_)
            | FilterOperation::GreaterThanOrEqual(_)
            | FilterOperation::LessThanOrEqual(_) => {
                // 范围查询：使用统计信息进行智能估计
                let selectivity = if let Some(field_index) = self.field_indices.get(&condition.key)
                {
                    field_index
                        .stats
                        .estimate_range_selectivity(&condition.operation)
                } else {
                    0.3 // 字段不存在，使用保守估计
                };
                let estimate = (self.total_records as f64 * selectivity) as usize;
                CardinalityEstimate {
                    min: 0,
                    max: self.total_records,
                    exp: estimate,
                    selectivity,
                }
            }
            FilterOperation::Contains(_)
            | FilterOperation::StartsWith(_)
            | FilterOperation::EndsWith(_) => {
                // 字符串匹配：使用统计信息进行智能估计
                let selectivity = if let Some(field_index) = self.field_indices.get(&condition.key)
                {
                    field_index
                        .stats
                        .estimate_string_selectivity(&condition.operation)
                } else {
                    0.2 // 字段不存在，使用保守估计
                };
                let estimate = (self.total_records as f64 * selectivity) as usize;
                CardinalityEstimate {
                    min: 0,
                    max: self.total_records,
                    exp: estimate,
                    selectivity,
                }
            }
            _ => {
                // 其他操作：保守估计50%
                CardinalityEstimate {
                    min: 0,
                    max: self.total_records,
                    exp: self.total_records / 2,
                    selectivity: 0.5,
                }
            }
        }
    }

    /// 获取满足单个条件的记录ID
    fn get_condition_ids(&self, condition: &Condition) -> HashSet<RecordId> {
        let field_index = match self.field_indices.get(&condition.key) {
            Some(idx) => idx,
            None => return HashSet::new(),
        };

        // `Equal` and `In` name the values they want, so they are lookups. Everything else has
        // to be asked of each value in turn, through the one predicate in
        // [`Self::value_key_matches`] -- this used to be nine near-identical loops, and nine
        // copies of a comparison is nine chances for the "what matches" answer to disagree
        // with the "does anything match" answer.
        match &condition.operation {
            FilterOperation::Equal(value) => {
                return Self::to_value_key(value)
                    .and_then(|key| field_index.value_to_ids.get(&key))
                    .cloned()
                    .unwrap_or_default();
            }
            FilterOperation::In(values) => {
                let mut result = HashSet::new();
                for value in values {
                    if let Some(ids) =
                        Self::to_value_key(value).and_then(|key| field_index.value_to_ids.get(&key))
                    {
                        result.extend(ids.iter().cloned());
                    }
                }
                return result;
            }
            _ => {}
        }

        let mut result = HashSet::new();
        for (value_key, ids) in &field_index.value_to_ids {
            if Self::value_key_matches(value_key, &condition.operation) {
                result.extend(ids.iter().cloned());
            }
        }
        result
    }

    /// 为字段建立索引
    fn index_field(&mut self, key: &str, value: &serde_json::Value, record_id: &RecordId) {
        let value_key = match Self::to_value_key(value) {
            Some(vk) => vk,
            None => return, // 无法索引的类型（如数组、对象）
        };

        let field_index = self
            .field_indices
            .entry(key.to_string())
            .or_insert_with(|| FieldIndex {
                value_to_ids: HashMap::new(),
                field_type: Self::infer_field_type(value),
                distinct_count: 0,
                stats: FieldStats::new(),
            });

        field_index
            .value_to_ids
            .entry(value_key)
            .or_default()
            .insert(record_id.clone());

        field_index.distinct_count = field_index.value_to_ids.len();

        // 更新字段统计信息
        match value {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    field_index.stats.update_numeric(f);
                }
            }
            serde_json::Value::String(s) => {
                field_index.stats.update_string(s);
            }
            _ => {}
        }
    }

    /// 将JSON值转换为索引键
    fn to_value_key(value: &serde_json::Value) -> Option<ValueKey> {
        match value {
            serde_json::Value::String(s) => Some(ValueKey::String(s.clone())),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(ValueKey::Number(i))
                } else {
                    n.as_f64().map(|f| ValueKey::Float(OrderedFloat::from(f)))
                }
            }
            serde_json::Value::Bool(b) => Some(ValueKey::Bool(*b)),
            serde_json::Value::Null => Some(ValueKey::Null),
            _ => None, // 数组和对象暂不索引
        }
    }

    /// 推断字段类型
    fn infer_field_type(value: &serde_json::Value) -> FieldType {
        match value {
            serde_json::Value::String(_) => FieldType::String,
            serde_json::Value::Number(_) => FieldType::Number,
            serde_json::Value::Bool(_) => FieldType::Boolean,
            _ => FieldType::Mixed,
        }
    }

    /// 获取索引统计信息
    pub fn get_stats(&self) -> PayloadIndexStats {
        PayloadIndexStats {
            total_records: self.total_records,
            indexed_fields: self.field_indices.len(),
            field_details: self
                .field_indices
                .iter()
                .map(|(name, idx)| (name.clone(), idx.distinct_count))
                .collect(),
        }
    }
}

/// PayloadIndex统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayloadIndexStats {
    pub total_records: usize,
    pub indexed_fields: usize,
    pub field_details: HashMap<String, usize>,
}

impl Default for PayloadIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod estimator_repro {
    use super::*;
    use crate::types::record::{Condition, Filter, FilterOperation};

    /// A range pair with nothing between it is provably empty, and provable without touching
    /// a single record id.
    ///
    /// Neither half is empty on its own -- `>= 400` accepts the 1000s and `<= 600` accepts
    /// the 0s -- so this only works because conditions on one key are tested against one
    /// value together. The first version grouped nothing and proved nothing here.
    #[test]
    fn a_range_with_no_value_between_it_is_provably_empty() {
        let mut index = PayloadIndex::new();
        for i in 0..1_000u32 {
            let bucket = if i % 10 == 0 { 1000 } else { 0 };
            index.insert_record(&i.to_string(), &serde_json::json!({ "bucket": bucket }));
        }

        let between = |lo, hi| Filter {
            must: Some(vec![
                Condition {
                    key: "bucket".to_string(),
                    operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(lo)),
                },
                Condition {
                    key: "bucket".to_string(),
                    operation: FilterOperation::LessThanOrEqual(serde_json::json!(hi)),
                },
            ]),
            must_not: None,
            should: None,
        };

        assert!(
            index.cannot_match_anything(&between(400, 600)),
            "no bucket is between 400 and 600"
        );
        // And the ranges that do contain a value are not claimed empty.
        assert!(!index.cannot_match_anything(&between(0, 600)));
        assert!(!index.cannot_match_anything(&between(600, 2000)));
        assert!(!index.cannot_match_anything(&between(0, 2000)));

        // The claim agrees with what the ids say, which is the property that matters.
        assert!(index.get_matching_ids(&between(400, 600)).is_empty());
        assert_eq!(index.get_matching_ids(&between(600, 2000)).len(), 100);

        // A negation proves nothing: an array-valued record satisfies it without being
        // indexed, so the shortcut has to decline however empty the index looks.
        let not_equal = Filter {
            must: Some(vec![Condition {
                key: "absent".to_string(),
                operation: FilterOperation::NotEqual(serde_json::json!(1)),
            }]),
            must_not: None,
            should: None,
        };
        assert!(!index.cannot_match_anything(&not_equal));

        // But a range on a field no record carries is empty, and that one is provable.
        let absent_range = Filter {
            must: Some(vec![Condition {
                key: "absent".to_string(),
                operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(1)),
            }]),
            must_not: None,
            should: None,
        };
        assert!(index.cannot_match_anything(&absent_range));
    }

    fn index_with_ids(n: usize) -> PayloadIndex {
        let mut idx = PayloadIndex::new();
        for i in 0..n {
            idx.insert_record(&i.to_string(), &serde_json::json!({ "id_num": i }));
        }
        idx
    }

    fn ge(value: usize) -> Filter {
        Filter {
            must: Some(vec![Condition {
                key: "id_num".to_string(),
                operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(value)),
            }]),
            must_not: None,
            should: None,
        }
    }

    /// The state production was actually in: records loaded, but none carrying the field the
    /// filter selects on, because the harness only writes it when asked to. The server logged
    /// `selectivity=50.00%` there, and a query that could not match anything took 16.6 s.
    #[test]
    fn an_absent_field_estimates_zero_not_half_the_table() {
        let mut idx = PayloadIndex::new();
        for i in 0..100_000usize {
            // Exactly what the unfiltered load writes: an empty payload.
            idx.insert_record(&i.to_string(), &serde_json::json!({}));
        }
        let est = idx.estimate_cardinality(&ge(50_000));
        let matched = idx.get_matching_ids(&ge(50_000)).len();
        println!(
            "\n  field absent: estimate {:.2}%  exp {}  actual matches {}",
            est.selectivity * 100.0,
            est.exp,
            matched
        );
        assert_eq!(matched, 0, "no record carries the field");
        // The estimate is the guard, not the result. A search on an absent field returns an
        // empty page whatever the estimate says -- post-filtering 600 candidates that all
        // fail is still correct -- so correctness tests cannot catch this. What the estimate
        // decides is the *route*: 0 sends the search to the precomputed-empty-set path,
        // while 0.5 sent it to post-filtering, which on a 5M x 1536 table took 16.6 s to
        // return nothing while holding the table read lock for 4.6 s.
        assert_eq!(est.exp, 0, "an absent field cannot match any row");
        assert_eq!(
            est.selectivity, 0.0,
            "an absent field must estimate zero, not half the table"
        );
    }

    /// What the estimator says about `id_num >= t` over a uniform 0..n, which is the shape
    /// every VectorDBBench filtered case uses. Printed rather than asserted: this is the
    /// measurement that decides what to fix.
    #[test]
    fn what_the_estimator_says_about_a_uniform_field() {
        let n = 100_000usize;
        let idx = index_with_ids(n);
        println!("\n  n = {n}");
        for pct in [1usize, 10, 50, 90, 99] {
            let t = n * pct / 100;
            let truth = (n - t) as f64 / n as f64;
            let est = idx.estimate_cardinality(&ge(t));
            let matched = idx.get_matching_ids(&ge(t)).len();
            println!(
                "  id_num >= {t:>6}: truth {:>6.2}%  estimate {:>6.2}%  (exp {:>7})  actual matches {matched:>6}",
                truth * 100.0,
                est.selectivity * 100.0,
                est.exp,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_payload_index_insert_and_cardinality() {
        let mut index = PayloadIndex::new();

        // 插入100条记录，其中10条category=A, 90条category=B
        for i in 0..100 {
            let record_id = format!("rec_{}", i);
            let category = if i < 10 { "A" } else { "B" };
            let payload = json!({
                "category": category,
                "score": i
            });
            index.insert_record(&record_id, &payload);
        }

        // Debug: Check index state
        assert_eq!(index.total_records, 100);
        println!("Total records: {}", index.total_records);
        if let Some(field_idx) = index.field_indices.get("category") {
            println!(
                "Category field distinct count: {}",
                field_idx.distinct_count
            );
            println!(
                "Category field value_to_ids keys: {:?}",
                field_idx.value_to_ids.keys().collect::<Vec<_>>()
            );
            for (key, ids) in &field_idx.value_to_ids {
                println!("  {:?}: {} records", key, ids.len());
            }
        }

        // 测试Equal过滤 - category=A (10%)
        let filter = Filter {
            must: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::Equal(json!("A")),
            }]),
            must_not: None,
            should: None,
        };

        let estimate = index.estimate_cardinality(&filter);
        println!(
            "Estimate: exp={}, selectivity={}",
            estimate.exp, estimate.selectivity
        );
        assert_eq!(estimate.exp, 10);
        assert!((estimate.selectivity - 0.1).abs() < 0.01);

        // 测试In过滤 - category in [A, B] (100%)
        let filter_in = Filter {
            must: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::In(vec![json!("A"), json!("B")]),
            }]),
            must_not: None,
            should: None,
        };

        let estimate_in = index.estimate_cardinality(&filter_in);
        assert_eq!(estimate_in.exp, 100);
        assert!((estimate_in.selectivity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_get_matching_ids() {
        let mut index = PayloadIndex::new();

        for i in 0..10 {
            let record_id = format!("rec_{}", i);
            let payload = json!({"category": if i < 5 { "A" } else { "B" }});
            index.insert_record(&record_id, &payload);
        }

        let filter = Filter {
            must: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::Equal(json!("A")),
            }]),
            must_not: None,
            should: None,
        };

        let matching_ids = index.get_matching_ids(&filter);
        assert_eq!(matching_ids.len(), 5);
        assert!(matching_ids.contains("rec_0"));
        assert!(matching_ids.contains("rec_4"));
        assert!(!matching_ids.contains("rec_5"));
    }
}
