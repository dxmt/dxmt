use crate::types::{DenseVector, Payload, RecordId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::types::{sparse_vector::SparseVector, vector::VectorValidator};

/// 记录状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecordStatus {
    /// 活跃状态
    Active,
    /// 已删除（软删除）
    Deleted,
    /// 正在处理中
    Processing,
}

/// 向量记录
#[derive(Debug, Clone)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct Record {
    pub id: RecordId,
    pub vector: DenseVector,
    pub sparse_vector: Option<SparseVector>,
    /// Named vectors (multi-vector support). Key = vector space name.
    pub named_vectors: Option<std::collections::HashMap<String, DenseVector>>,
    pub payload: Payload,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// 版本号，用于乐观锁控制
    pub version: u64,
    /// 记录状态
    pub status: RecordStatus,
}

/// 用于序列化的Record结构体，避免serde_json::Value的bincode问题
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableRecord {
    id: RecordId,
    vector: DenseVector,
    sparse_vector: Option<SparseVector>,
    #[serde(default)]
    named_vectors: Option<std::collections::HashMap<String, DenseVector>>,
    payload_json: String, // 将Payload序列化为JSON字符串
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    version: u64,
    status: RecordStatus,
}

/// The same shape as [`SerializableRecord`], borrowing instead of owning.
///
/// `Record::serialize` used to build a `SerializableRecord`, which meant cloning every field
/// -- the 3 KB vector included -- for each record on its way to disk. A snapshot of a million
/// records spent about 3 GB of allocation on copies that existed only to be written and
/// dropped, and that showed up as resident memory for as long as the allocator held the
/// pages: with the record map already shared and the writer already streaming, a snapshot
/// still took the peak from 8.1 GB to 13.5 GB (design doc section 34).
///
/// Field order and types match `SerializableRecord` exactly, because bincode writes fields in
/// order and a `&T` writes as a `T` -- the file is the same file. `payload_json` is still
/// built: the payload is stored as a JSON string, and that is the format, not a copy.
#[derive(Serialize)]
struct SerializableRecordRef<'a> {
    id: &'a RecordId,
    vector: &'a DenseVector,
    sparse_vector: &'a Option<SparseVector>,
    named_vectors: &'a Option<std::collections::HashMap<String, DenseVector>>,
    payload_json: String,
    created_at: &'a DateTime<Utc>,
    updated_at: &'a DateTime<Utc>,
    version: u64,
    status: &'a RecordStatus,
}

impl Serialize for Record {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let serializable = SerializableRecordRef {
            id: &self.id,
            vector: &self.vector,
            sparse_vector: &self.sparse_vector,
            named_vectors: &self.named_vectors,
            payload_json: serde_json::to_string(&self.payload)
                .map_err(serde::ser::Error::custom)?,
            created_at: &self.created_at,
            updated_at: &self.updated_at,
            version: self.version,
            status: &self.status,
        };
        serializable.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Record {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let serializable = SerializableRecord::deserialize(deserializer)?;
        let payload =
            serde_json::from_str(&serializable.payload_json).map_err(serde::de::Error::custom)?;

        Ok(Record {
            id: serializable.id,
            vector: serializable.vector,
            sparse_vector: serializable.sparse_vector,
            named_vectors: serializable.named_vectors,
            payload,
            created_at: serializable.created_at,
            updated_at: serializable.updated_at,
            version: serializable.version,
            status: serializable.status,
        })
    }
}

/// 创建记录的请求
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct CreateRecordRequest {
    pub id: RecordId,
    pub vector: DenseVector,
    pub sparse_vector: Option<SparseVector>,
    /// Named vectors for multi-vector tables (key = vector space name).
    #[serde(default)]
    pub named_vectors: Option<std::collections::HashMap<String, DenseVector>>,
    pub payload: Option<Payload>,
}

/// 更新记录的请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRecordRequest {
    pub vector: Option<DenseVector>,
    pub sparse_vector: Option<SparseVector>,
    pub payload: Option<Payload>,
    /// 期望的版本号，用于乐观锁
    pub expected_version: Option<u64>,
}

/// Search accuracy/speed tradeoff mode.
///
/// Four of the five are the same knob at different settings: they scale HNSW's `ef_search`,
/// the traversal budget. `Exact` is not that knob at all -- it skips the graph and scores
/// every live row, which is the only way to get an answer with no recall loss in it.
///
/// That distinction is the reason `Exact` exists. `Accurate` is 4× ef and measures 0.9798 to
/// 0.9912 recall depending on the corpus; it is the most the graph will give you, not the
/// truth. Without a mode that returns the truth, "what is my recall on my own data" can only
/// be answered by the benchmark harness against a dataset with published ground truth --
/// never against the data someone actually loaded.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// Minimum ef_search = top_k. Fastest, lower recall on difficult queries.
    Fast,
    /// Default ef_search from index config. Balanced.
    #[default]
    Normal,
    /// 4× default ef_search. Higher recall, slower.
    Accurate,
    /// Dynamically scale ef based on top_k and filter selectivity (DynamicEF).
    Auto,
    /// Score every live row. No index, no approximation, no `ef` -- this is the ground truth
    /// the other four are approximating, and it costs a full pass over the table's vectors.
    ///
    /// Refused above `EXACT_SCAN_MAX` rows rather than run for minutes; see
    /// `IntelligentIndexedStorage::exact_scan_max`.
    Exact,
}

impl SearchMode {
    /// Whether this mode bypasses the index entirely.
    ///
    /// The search path branches on this before it computes an `ef`, because `ef` is a budget
    /// for a graph traversal and this mode does not traverse a graph.
    pub fn is_exhaustive(&self) -> bool {
        matches!(self, SearchMode::Exact)
    }

    /// Compute the effective ef_search value given the configured default and top_k.
    pub fn ef_search(&self, configured_ef: usize, top_k: usize) -> usize {
        match self {
            SearchMode::Fast => top_k.max(1),
            SearchMode::Normal => configured_ef.max(top_k),
            SearchMode::Accurate => (configured_ef * 4).max(top_k),
            // Auto without selectivity: fall back to Accurate for safety
            SearchMode::Auto => (configured_ef * 4).max(top_k),
            // Never reached -- `is_exhaustive` diverts this mode before any ef is computed.
            // Answering `top_k` rather than panicking keeps a future caller that forgets the
            // check merely wrong about speed, not dead.
            SearchMode::Exact => top_k.max(1),
        }
    }

    /// Dynamic ef calculation: scales proportionally to top_k / selectivity.
    ///
    /// A selective filter discards most of what the traversal visits, so the budget has to
    /// grow to compensate: `8 × (top_k / selectivity)`. Capped at 4× `configured_ef` so a
    /// near-zero selectivity estimate cannot turn one query into a full scan.
    ///
    /// `filter_selectivity` is in [0.0, 1.0]; 1.0 = no filter, 0.01 = 1% of records match.
    pub fn ef_search_adaptive(
        &self,
        configured_ef: usize,
        top_k: usize,
        filter_selectivity: f64,
    ) -> usize {
        match self {
            SearchMode::Auto => {
                let effective_selectivity = filter_selectivity.clamp(0.001, 1.0);
                let base = (8.0 * top_k as f64 / effective_selectivity).ceil() as usize;
                base.max(top_k).min(configured_ef * 4)
            }
            other => other.ef_search(configured_ef, top_k),
        }
    }
}

/// 搜索查询
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct SearchQuery {
    pub vector: DenseVector,
    #[serde(default = "default_top_k", alias = "k")]
    pub top_k: usize,
    pub filter: Option<Filter>,
    #[serde(default = "default_true")]
    pub with_payload: bool,
    #[serde(default)]
    pub with_vector: bool,
    #[serde(default = "default_top_k")]
    pub limit: usize,
    /// Search accuracy/speed mode. Defaults to Normal.
    #[serde(default)]
    pub search_mode: SearchMode,
    /// Target named vector space. None = use the default (unnamed) vector.
    #[serde(default)]
    pub vector_name: Option<String>,
}

/// 混合搜索查询
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct HybridSearchQuery {
    /// 密集向量查询
    pub dense_vector: Option<DenseVector>,
    /// 稀疏向量查询  
    pub sparse_vector: Option<SparseVector>,
    /// 文本查询
    pub text_query: Option<String>,
    // `alpha`, `beta` and `gamma` used to sit here: per-component weights for a fusion this
    // endpoint does not perform. It serves a lone dense vector and refuses anything carrying a
    // sparse or text component, so the three were documented knobs on a path that cannot run.
    // The weights that are read live on `HybridSearchRequest`, at POST /search/hybrid.
    /// 返回结果数量
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// 过滤条件
    pub filter: Option<Filter>,
    #[serde(default = "default_true")]
    pub with_payload: bool,
    #[serde(default)]
    pub with_vector: bool,
}

/// 全文搜索请求
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct FullTextSearchRequest {
    /// 搜索查询文本
    pub query: String,
    /// 要搜索的字段（如果不指定则搜索所有已配置的文本字段）
    pub fields: Option<Vec<String>>,
    /// 返回结果数量
    #[serde(default = "default_top_k")]
    pub limit: usize,
    /// 偏移量（用于分页）
    #[serde(default)]
    pub offset: usize,
    /// 是否启用高亮
    #[serde(default)]
    pub highlight: bool,
    /// 模糊搜索配置
    pub fuzzy: Option<FuzzyConfig>,
    /// 过滤条件
    pub filter: Option<Filter>,
}

/// 模糊搜索配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct FuzzyConfig {
    /// 最大编辑距离（Levenshtein距离）
    #[serde(default = "default_max_distance")]
    pub max_distance: u8,
    /// 前缀长度（前N个字符必须精确匹配）
    #[serde(default)]
    pub prefix_length: usize,
}

fn default_max_distance() -> u8 {
    2
}

/// 全文搜索结果
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct FullTextSearchResult {
    pub hits: Vec<FullTextSearchHit>,
    pub total: usize,
    pub took_ms: u64,
}

/// 全文搜索命中结果
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct FullTextSearchHit {
    pub id: RecordId,
    pub score: f32,
    pub payload: Option<Payload>,
    /// 高亮的片段（如果启用了高亮）
    pub highlights: Option<Vec<String>>,
}

/// 混合搜索融合方法
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub enum FusionMethod {
    /// 加权求和（默认方法）
    /// 使用 Min-Max 归一化后加权求和，适合大数据集
    WeightedSum,
    /// 倒数排名融合（Reciprocal Rank Fusion）
    /// 基于排名而非分数融合，对分数尺度不敏感，适合小数据集
    ReciprocalRankFusion,
    /// Distribution-Based Score Fusion（分布归一化融合）
    /// 对每路结果独立做 (s-mean)/std 归一化后求和，无需调参 alpha。
    /// 适合 dense+sparse+text 三路融合，消除量纲不一致。
    #[default]
    Dbsf,
}

/// 混合搜索请求（向量 + 稀疏向量 + 文本）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct HybridSearchRequest {
    /// 稠密向量查询（可选）
    pub vector: Option<DenseVector>,
    /// 稀疏向量查询（可选）— 与 SparseIndex 配合做 MIPS
    pub sparse_vector: Option<crate::types::sparse_vector::SparseVector>,
    /// 文本查询（可选）
    pub text: Option<String>,
    /// 向量搜索权重 (0.0-1.0)，仅 WeightedSum 时有效
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,
    /// 融合方法，默认 Dbsf
    #[serde(default)]
    pub fusion_method: FusionMethod,
    /// RRF 参数 k（仅当使用 ReciprocalRankFusion 时有效）
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    /// 返回结果数量
    #[serde(default = "default_top_k")]
    pub limit: usize,
    /// 过滤条件
    pub filter: Option<Filter>,
    /// 是否返回payload
    #[serde(default = "default_true")]
    pub with_payload: bool,
    /// 是否返回vector
    #[serde(default)]
    pub with_vector: bool,
}

fn default_vector_weight() -> f32 {
    0.7
}

fn default_rrf_k() -> f32 {
    60.0
}

/// 批量搜索查询
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSearchQuery {
    pub queries: Vec<SearchQuery>,
    #[serde(default)]
    pub parallel: bool,
}

/// 范围搜索查询
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeSearchQuery {
    pub vector: DenseVector,
    pub min_score: Option<f32>,
    pub max_score: Option<f32>,
    pub max_results: Option<usize>,
    pub filter: Option<Filter>,
    #[serde(default = "default_true")]
    pub with_payload: bool,
    #[serde(default)]
    pub with_vector: bool,
}

fn default_top_k() -> usize {
    10
}

fn default_true() -> bool {
    true
}

/// 搜索结果
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    pub total: usize,
    pub tookms: u64,
}

/// 搜索命中结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: RecordId,
    pub score: f32,
    pub payload: Option<Payload>,
    pub vector: Option<DenseVector>,
}

/// 过滤条件
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct Filter {
    pub must: Option<Vec<Condition>>,
    pub must_not: Option<Vec<Condition>>,
    pub should: Option<Vec<Condition>>,
}

/// 单个过滤条件
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct Condition {
    pub key: String,
    pub operation: FilterOperation,
}

/// 过滤操作
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[serde(tag = "type", content = "value")]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub enum FilterOperation {
    Equal(serde_json::Value),
    NotEqual(serde_json::Value),
    GreaterThan(serde_json::Value),
    GreaterThanOrEqual(serde_json::Value),
    LessThan(serde_json::Value),
    LessThanOrEqual(serde_json::Value),
    In(Vec<serde_json::Value>),
    NotIn(Vec<serde_json::Value>),
    Contains(String),
    StartsWith(String),
    EndsWith(String),
}

impl Record {
    /// 创建新记录
    pub fn new(request: CreateRecordRequest, dimension: usize) -> crate::Result<Self> {
        // 验证向量
        VectorValidator::validate(&request.vector, dimension)?;

        let now = chrono::Utc::now();
        Ok(Record {
            id: request.id,
            vector: request.vector,
            sparse_vector: request.sparse_vector,
            named_vectors: request.named_vectors,
            payload: request
                .payload
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
            created_at: now,
            updated_at: now,
            version: 1, // 新记录版本从1开始
            status: RecordStatus::Active,
        })
    }

    /// 更新记录（带版本控制）
    pub fn update(
        &mut self,
        vector: Option<DenseVector>,
        sparse_vector: Option<SparseVector>,
        payload: Option<Payload>,
        dimension: usize,
        expected_version: Option<u64>,
    ) -> crate::Result<()> {
        // 乐观锁检查
        if let Some(expected) = expected_version {
            if self.version != expected {
                return Err(crate::error::Error::InvalidInput(format!(
                    "Version mismatch: expected {}, current {}",
                    expected, self.version
                )));
            }
        }

        if let Some(new_vector) = vector {
            VectorValidator::validate(&new_vector, dimension)?;
            self.vector = new_vector
        }

        if let Some(new_sparse_vector) = sparse_vector {
            self.sparse_vector = Some(new_sparse_vector)
        }

        if let Some(new_payload) = payload {
            self.payload = new_payload
        }

        self.updated_at = chrono::Utc::now();
        self.version += 1; // 增加版本号
        Ok(())
    }

    /// 软删除记录
    pub fn soft_delete(&mut self, expected_version: Option<u64>) -> crate::Result<()> {
        // 乐观锁检查
        if let Some(expected) = expected_version {
            if self.version != expected {
                return Err(crate::error::Error::InvalidInput(format!(
                    "Version mismatch: expected {}, current {}",
                    expected, self.version
                )));
            }
        }

        self.status = RecordStatus::Deleted;
        self.updated_at = chrono::Utc::now();
        self.version += 1;
        Ok(())
    }

    /// 检查记录是否为活跃状态
    pub fn is_active(&self) -> bool {
        self.status == RecordStatus::Active
    }

    /// 检查是否满足过滤条件
    pub fn matches_filter(&self, filter: &Filter) -> bool {
        // 必须满足的条件
        if let Some(must_conditions) = &filter.must {
            for condition in must_conditions {
                if !self.matches_condition(condition) {
                    return false;
                }
            }
        }

        // 不能满足的条件
        if let Some(must_not_conditions) = &filter.must_not {
            for condition in must_not_conditions {
                if self.matches_condition(condition) {
                    return false;
                }
            }
        }

        // 应该满足的条件（至少一个）
        if let Some(should_conditions) = &filter.should {
            if !should_conditions.is_empty() {
                let mut any_match = false;
                for condition in should_conditions {
                    if self.matches_condition(condition) {
                        any_match = true;
                        break;
                    }
                }
                if !any_match {
                    return false;
                }
            }
        }

        true
    }

    /// 检查是否满足单个条件
    fn matches_condition(&self, condition: &Condition) -> bool {
        let owned = self.get_field_value(&condition.key);
        let value = owned.as_deref();
        // `as_f64` on the field once, not once per branch: every numeric comparison below
        // wants the same number.
        let number = || value.and_then(serde_json::Value::as_f64);
        let text = || value.and_then(serde_json::Value::as_str);

        match &condition.operation {
            FilterOperation::Equal(target) => value == Some(target),
            FilterOperation::NotEqual(target) => value != Some(target),
            FilterOperation::GreaterThan(target) => {
                matches!((number(), target.as_f64()), (Some(v), Some(t)) if v > t)
            }
            FilterOperation::GreaterThanOrEqual(target) => {
                matches!((number(), target.as_f64()), (Some(v), Some(t)) if v >= t)
            }
            FilterOperation::LessThan(target) => {
                matches!((number(), target.as_f64()), (Some(v), Some(t)) if v < t)
            }
            FilterOperation::LessThanOrEqual(target) => {
                matches!((number(), target.as_f64()), (Some(v), Some(t)) if v <= t)
            }
            FilterOperation::In(targets) => value.is_some_and(|v| targets.contains(v)),
            FilterOperation::NotIn(targets) => value.is_some_and(|v| !targets.contains(v)),
            FilterOperation::Contains(substr) => {
                text().is_some_and(|s| s.contains(substr.as_str()))
            }
            FilterOperation::StartsWith(prefix) => {
                text().is_some_and(|s| s.starts_with(prefix.as_str()))
            }
            FilterOperation::EndsWith(suffix) => {
                text().is_some_and(|s| s.ends_with(suffix.as_str()))
            }
        }
    }

    /// 获取字段值
    ///
    /// Borrowed, not cloned. Post-filtering calls this once per condition per record, and a
    /// filtered search that has to fall back to scanning the table calls it for every row it
    /// holds: the clone that used to be here allocated a `Value` (and, for a string field, its
    /// bytes) tens of thousands of times to answer a comparison that never needed to own
    /// anything. `id` is the one key with no `Value` to borrow, so it is the one that still
    /// builds one.
    fn get_field_value(&self, key: &str) -> Option<std::borrow::Cow<'_, serde_json::Value>> {
        match key {
            "id" => Some(std::borrow::Cow::Owned(serde_json::Value::String(
                self.id.clone(),
            ))),
            _ => match &self.payload {
                serde_json::Value::Object(map) => map.get(key).map(std::borrow::Cow::Borrowed),
                _ => None,
            },
        }
    }
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            vector: vec![],
            top_k: 10,
            filter: None,
            with_payload: true,
            with_vector: false,
            limit: 10,
            search_mode: SearchMode::Normal,
            vector_name: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_creation() -> Result<(), Box<dyn std::error::Error>> {
        let request = CreateRecordRequest {
            id: "test_record".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            sparse_vector: None,
            payload: Some(serde_json::json!({"category": "test"})),
            named_vectors: None,
        };

        let record = Record::new(request, 3)?;
        assert_eq!(record.id.to_string(), "test_record");
        assert_eq!(record.vector, vec![0.1, 0.2, 0.3]);
        Ok(())
    }

    #[test]
    fn test_filter_matching() -> Result<(), Box<dyn std::error::Error>> {
        let request = CreateRecordRequest {
            id: "filter_test".to_string(),
            vector: vec![1.0, 2.0],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "category": "electronics",
                "price": 99.99
            })),
            named_vectors: None,
        };

        let record = Record::new(request, 2)?;

        let filter = Filter {
            must: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::Equal(serde_json::Value::String(
                    "electronics".to_string(),
                )),
            }]),
            must_not: None,
            should: None,
        };

        assert!(record.matches_filter(&filter));
        Ok(())
    }

    #[test]
    fn test_filter_matching_by_id() -> Result<(), Box<dyn std::error::Error>> {
        let record = Record::new(
            CreateRecordRequest {
                id: "rec-42".to_string(),
                vector: vec![1.0, 0.0],
                sparse_vector: None,
                payload: Some(serde_json::json!({"x": 1})),
                named_vectors: None,
            },
            2,
        )?;

        let match_filter = Filter {
            must: Some(vec![Condition {
                key: "id".to_string(),
                operation: FilterOperation::Equal(serde_json::json!("rec-42")),
            }]),
            must_not: None,
            should: None,
        };
        assert!(record.matches_filter(&match_filter), "should match own id");

        let no_match_filter = Filter {
            must: Some(vec![Condition {
                key: "id".to_string(),
                operation: FilterOperation::Equal(serde_json::json!("other-id")),
            }]),
            must_not: None,
            should: None,
        };
        assert!(
            !record.matches_filter(&no_match_filter),
            "should not match different id"
        );
        Ok(())
    }

    #[test]
    fn test_filter_must_not_and_should() -> Result<(), Box<dyn std::error::Error>> {
        let record = Record::new(
            CreateRecordRequest {
                id: "r1".to_string(),
                vector: vec![1.0, 0.0],
                sparse_vector: None,
                payload: Some(serde_json::json!({"category": "food", "price": 5.0})),
                named_vectors: None,
            },
            2,
        )?;

        // must_not: should exclude record with category "food"
        let must_not = Filter {
            must: None,
            must_not: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::Equal(serde_json::json!("food")),
            }]),
            should: None,
        };
        assert!(
            !record.matches_filter(&must_not),
            "must_not: category=food should exclude this record"
        );

        // should: at least one must match — "food" OR "drink"
        let should_match = Filter {
            must: None,
            must_not: None,
            should: Some(vec![
                Condition {
                    key: "category".to_string(),
                    operation: FilterOperation::Equal(serde_json::json!("food")),
                },
                Condition {
                    key: "category".to_string(),
                    operation: FilterOperation::Equal(serde_json::json!("drink")),
                },
            ]),
        };
        assert!(
            record.matches_filter(&should_match),
            "should: 'food' satisfies the OR condition"
        );

        // should: none match → excluded
        let should_no_match = Filter {
            must: None,
            must_not: None,
            should: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::Equal(serde_json::json!("drink")),
            }]),
        };
        assert!(
            !record.matches_filter(&should_no_match),
            "should: no match should exclude"
        );

        // numeric GreaterThan
        let gt_filter = Filter {
            must: Some(vec![Condition {
                key: "price".to_string(),
                operation: FilterOperation::GreaterThan(serde_json::json!(3.0)),
            }]),
            must_not: None,
            should: None,
        };
        assert!(
            record.matches_filter(&gt_filter),
            "price 5.0 > 3.0 should match"
        );

        let gt_no_match = Filter {
            must: Some(vec![Condition {
                key: "price".to_string(),
                operation: FilterOperation::GreaterThan(serde_json::json!(10.0)),
            }]),
            must_not: None,
            should: None,
        };
        assert!(
            !record.matches_filter(&gt_no_match),
            "price 5.0 > 10.0 should not match"
        );
        Ok(())
    }

    #[test]
    fn test_search_mode_ef_search() {
        let configured_ef = 100;

        // Fast: ef = top_k (minimum scan)
        assert_eq!(SearchMode::Fast.ef_search(configured_ef, 10), 10);
        assert_eq!(SearchMode::Fast.ef_search(configured_ef, 50), 50);

        // Normal: ef = max(configured_ef, top_k)
        assert_eq!(SearchMode::Normal.ef_search(configured_ef, 10), 100);
        assert_eq!(SearchMode::Normal.ef_search(configured_ef, 200), 200);

        // Accurate: ef = max(4 * configured_ef, top_k)
        assert_eq!(SearchMode::Accurate.ef_search(configured_ef, 10), 400);
        assert_eq!(SearchMode::Accurate.ef_search(configured_ef, 500), 500);
    }

    #[test]
    fn test_search_mode_auto_dynamic_ef() {
        let configured_ef = 100;

        // Auto with no filter (selectivity=1.0): ef = 8*k/1.0, capped at 4*configured_ef
        assert_eq!(
            SearchMode::Auto.ef_search_adaptive(configured_ef, 10, 1.0),
            80
        ); // 8*10=80

        // Auto with 10% selectivity: ef = 8*10/0.1 = 800, capped at 400
        assert_eq!(
            SearchMode::Auto.ef_search_adaptive(configured_ef, 10, 0.1),
            400
        );

        // Auto with 1% selectivity: ef = 8*10/0.01 = 8000, capped at 400
        assert_eq!(
            SearchMode::Auto.ef_search_adaptive(configured_ef, 10, 0.01),
            400
        );

        // Auto with 50% selectivity: ef = 8*10/0.5 = 160, capped at 400
        assert_eq!(
            SearchMode::Auto.ef_search_adaptive(configured_ef, 10, 0.5),
            160
        );

        // Auto with large top_k: ef must be at least top_k
        assert_eq!(
            SearchMode::Auto.ef_search_adaptive(configured_ef, 500, 1.0),
            400
        ); // 8*500=4000 → capped at 400

        // Non-Auto modes delegate to ef_search (selectivity param ignored)
        assert_eq!(
            SearchMode::Normal.ef_search_adaptive(configured_ef, 10, 0.01),
            SearchMode::Normal.ef_search(configured_ef, 10)
        );
        assert_eq!(
            SearchMode::Fast.ef_search_adaptive(configured_ef, 10, 0.01),
            SearchMode::Fast.ef_search(configured_ef, 10)
        );
    }

    #[test]
    fn test_search_mode_serde() {
        assert_eq!(
            serde_json::to_string(&SearchMode::Fast).unwrap(),
            "\"fast\""
        );
        assert_eq!(
            serde_json::to_string(&SearchMode::Accurate).unwrap(),
            "\"accurate\""
        );
        let m: SearchMode = serde_json::from_str("\"normal\"").unwrap();
        assert_eq!(m, SearchMode::Normal);
    }

    #[test]
    fn test_search_query_default_search_mode() {
        let q = SearchQuery::default();
        assert_eq!(q.search_mode, SearchMode::Normal);
    }
}

#[cfg(test)]
mod serialize_borrow_tests {
    use super::*;

    /// Borrowing on the way to disk must not change what lands there.
    ///
    /// `Record::serialize` builds a `SerializableRecordRef` rather than cloning every field
    /// into a `SerializableRecord` -- for a snapshot of a million records that was about 3 GB
    /// of copies made only to be written and dropped. It is only safe because bincode writes
    /// a struct's fields in order and writes a `&T` exactly as a `T`, so the two shapes
    /// produce the same bytes. That is the sort of thing that is true until someone reorders
    /// a field, which is what this test is for: every snapshot and WAL entry already on disk
    /// depends on it.
    #[test]
    fn the_borrowing_shape_writes_the_same_bytes_as_the_owning_one() {
        let record = Record {
            id: "rec-42".to_string(),
            vector: vec![0.5, -0.25, 0.125, 0.0, 1.5],
            sparse_vector: None,
            named_vectors: None,
            payload: serde_json::json!({"bucket": 3, "name": "x", "flag": true}),
            created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
                .unwrap()
                .into(),
            updated_at: DateTime::parse_from_rfc3339("2026-01-02T03:04:05.678Z")
                .unwrap()
                .into(),
            version: 9,
            status: RecordStatus::Active,
        };
        let payload_json = serde_json::to_string(&record.payload).unwrap();

        let owning = SerializableRecord {
            id: record.id.clone(),
            vector: record.vector.clone(),
            sparse_vector: record.sparse_vector.clone(),
            named_vectors: record.named_vectors.clone(),
            payload_json: payload_json.clone(),
            created_at: record.created_at,
            updated_at: record.updated_at,
            version: record.version,
            status: record.status.clone(),
        };
        let borrowing = SerializableRecordRef {
            id: &record.id,
            vector: &record.vector,
            sparse_vector: &record.sparse_vector,
            named_vectors: &record.named_vectors,
            payload_json,
            created_at: &record.created_at,
            updated_at: &record.updated_at,
            version: record.version,
            status: &record.status,
        };

        assert_eq!(
            bincode::serialize(&owning).unwrap(),
            bincode::serialize(&borrowing).unwrap(),
            "the borrowing shape must write the bytes the owning one did"
        );
        // And the record itself, which now goes through the borrowing shape, agrees with both.
        assert_eq!(
            bincode::serialize(&record).unwrap(),
            bincode::serialize(&owning).unwrap(),
            "a Record must still write what a SerializableRecord writes"
        );
        // Round trip, so the reader side is covered too.
        let back: Record = bincode::deserialize(&bincode::serialize(&record).unwrap()).unwrap();
        assert_eq!(back.id, record.id);
        assert_eq!(back.vector, record.vector);
        assert_eq!(back.payload, record.payload);
        assert_eq!(back.version, record.version);
    }
}
