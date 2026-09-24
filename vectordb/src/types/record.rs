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
#[serde(deny_unknown_fields)]
pub struct CreateRecordRequest {
    pub id: RecordId,
    pub vector: DenseVector,
    pub sparse_vector: Option<SparseVector>,
    /// Named vectors for multi-vector tables (key = vector space name).
    #[serde(default)]
    pub named_vectors: Option<std::collections::HashMap<String, DenseVector>>,
    pub payload: Option<Payload>,
}

// `UpdateRecordRequest` was here too, with `sparse_vector` and an `expected_version` for
// optimistic locking. Nothing constructed or deserialised it: `PUT /records/{id}` takes the
// `UpdateRecordRequest` in `api::handlers`, which has neither field and has `merge_mode` and
// `upsert` instead. Two live-looking types under one name, and a reader who found this one
// would have believed the endpoint supports a version check it has never had.

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
#[serde(deny_unknown_fields)]
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
    /// Return only these payload keys, instead of the whole payload.
    ///
    /// `with_payload` is all-or-nothing, and a payload is often mostly the chunk of text the
    /// embedding came from: a page of 100 hits can be a megabyte of JSON to deliver two fields
    /// a caller wanted. An empty list is not "no fields" — leave the field out, or set
    /// `with_payload` to false, both of which say so plainly.
    ///
    /// A key that is not in a record's payload is simply absent from the projection; it is not
    /// an error, because a payload has no declared shape.
    #[serde(default)]
    pub payload_fields: Option<Vec<String>>,
    /// Drop hits scoring below this, after scoring and before `top_k` is applied.
    ///
    /// Scores here are similarities — higher is nearer, cosine in [-1, 1], `1/(1 + d)` for
    /// euclidean — so this is a floor, not a distance ceiling. Every comparable database has
    /// one, and without it a caller asking for ten neighbours of a query with three good
    /// matches gets seven bad ones and has to know the metric's scale to throw them away.
    ///
    /// It cannot make a search find *more*: the index is still asked for `top_k`, and the
    /// threshold only removes. A page that comes back short means that is all there was above
    /// the floor.
    #[serde(default)]
    pub score_threshold: Option<f32>,
}

/// 混合搜索查询
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

/// The payload a caller asked for: the whole thing, or only the named keys.
///
/// Taking the keys rather than removing the others keeps the cost proportional to what is
/// returned instead of to what is stored, which is the point of asking.
pub fn project_payload(payload: &Payload, fields: Option<&[String]>) -> Payload {
    match fields {
        None => payload.clone(),
        Some(fields) => {
            let mut out = serde_json::Map::with_capacity(fields.len());
            if let Some(object) = payload.as_object() {
                for field in fields {
                    if let Some(value) = object.get(field) {
                        out.insert(field.clone(), value.clone());
                    }
                }
            }
            serde_json::Value::Object(out)
        }
    }
}

/// 搜索命中结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: RecordId,
    pub score: f32,
    pub payload: Option<Payload>,
    pub vector: Option<DenseVector>,
}

/// One group of hits from `POST /tables/{t}/search/groups`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct SearchGroup {
    /// The value of `group_by` every hit in this group holds. Groups keep the order of their
    /// best hit, so the first group is the one the best hit belongs to.
    pub group_id: serde_json::Value,
    /// Up to `group_size` hits, best first.
    pub hits: Vec<SearchHit>,
    /// How many of the candidates fetched fell in this group, which can exceed `hits.len()`.
    pub matched: usize,
}

/// What a grouped search returns.
///
/// The counts are not decoration. Grouping is done over a *window* of the nearest neighbours,
/// not over the table: a group whose every member is further away than the window reaches is
/// not in the answer, and no amount of grouping can know that. `candidates_fetched` says how
/// wide the window was, so a caller can tell a result from a truncation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct SearchGroupsResult {
    pub groups: Vec<SearchGroup>,
    /// Nearest neighbours scored to fill the groups. Grown until enough groups were found or
    /// the cap was reached; when it equals the cap, and `groups` is short, the answer is
    /// bounded by the cap rather than by the data.
    pub candidates_fetched: usize,
    /// Candidates dropped because they carry no value under `group_by`. A record with no value
    /// there belongs to no group; reported rather than silently discarded, because a corpus
    /// where the field is missing on half the rows otherwise looks like a half-empty index.
    pub ungrouped: usize,
    pub tookms: u64,
}

/// 过滤条件
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct Filter {
    pub must: Option<Vec<Condition>>,
    pub must_not: Option<Vec<Condition>>,
    pub should: Option<Vec<Condition>>,
}

impl Filter {
    /// Refuse a condition whose target is an array or an object.
    ///
    /// `Equal`, `NotEqual`, `In` and `NotIn` compare a field against scalar values. The payload
    /// index keys scalars only, so it cannot answer `tags == ["a", "b"]`, while
    /// [`Record::matches_filter`] compares whole JSON values and can. Which one answered decided
    /// the result: `records/list`, a selective search and `search_mode: "exact"` returned no
    /// records, a broad search and full-text search returned the right ones. Answering it
    /// exactly would mean keying every array of every payload, for a query shape Qdrant refuses
    /// too — its match takes a keyword, an integer or a bool.
    ///
    /// So it is refused here, before any path is chosen, and every endpoint says the same
    /// thing: that this filter is not supported, rather than an answer that depends on routing.
    pub fn validate(&self) -> Result<(), String> {
        let all = [&self.must, &self.must_not, &self.should];
        for condition in all.into_iter().flatten().flatten() {
            let composite = |v: &serde_json::Value| v.is_array() || v.is_object();
            let (operation, bad) = match &condition.operation {
                FilterOperation::Equal(v) => ("Equal", composite(v)),
                FilterOperation::NotEqual(v) => ("NotEqual", composite(v)),
                FilterOperation::In(vs) => ("In", vs.iter().any(composite)),
                FilterOperation::NotIn(vs) => ("NotIn", vs.iter().any(composite)),
                _ => continue,
            };
            if bad {
                return Err(format!(
                    "filter on \"{}\": {operation} compares against strings, numbers, booleans \
                     or null, and was given an array or an object. Matching a whole array or \
                     object is not supported; filter on a scalar field instead.",
                    condition.key
                ));
            }
        }
        Ok(())
    }
}

/// 单个过滤条件
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct Condition {
    pub key: String,
    pub operation: FilterOperation,
}

/// A test applied to one payload field — or, under the key `id`, to the record's own id.
///
/// When the field holds an array, a test is satisfied if any element satisfies it, so
/// `tags == "rag"` matches `tags: ["rag", "llm"]`. `NotEqual` and `NotIn` are the complements:
/// they exclude a record if any element matches.
///
/// Numbers compare as numbers, so `5` and `5.0` are equal. The targets of `Equal`, `NotEqual`,
/// `In` and `NotIn` must be strings, numbers, booleans or null: an array or an object is refused
/// with `INVALID_INPUT`.
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[serde(tag = "type", content = "value")]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub enum FilterOperation {
    /// The field equals this value. A record lacking the field does not match.
    Equal(serde_json::Value),
    /// The field does not equal this value — for an array, no element does. A record lacking
    /// the field matches.
    NotEqual(serde_json::Value),
    /// The field is a number greater than this one.
    GreaterThan(serde_json::Value),
    /// The field is a number greater than or equal to this one.
    GreaterThanOrEqual(serde_json::Value),
    /// The field is a number less than this one.
    LessThan(serde_json::Value),
    /// The field is a number less than or equal to this one.
    LessThanOrEqual(serde_json::Value),
    /// The field equals one of these values. A record lacking the field does not match.
    In(Vec<serde_json::Value>),
    /// The field is present and equals none of these values — for an array, no element does. A
    /// record lacking the field does *not* match, unlike `NotEqual`; an empty array is present.
    NotIn(Vec<serde_json::Value>),
    /// The field is a string containing this one.
    Contains(String),
    /// The field is a string starting with this one.
    StartsWith(String),
    /// The field is a string ending with this one.
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

        match &condition.operation {
            // The negations are complements of the positive forms, so the array rule below
            // carries over without a second definition: `tags != "a"` is true exactly when no
            // element is `"a"`. A missing field satisfies `NotEqual` and not `NotIn`, as before.
            FilterOperation::NotEqual(target) => {
                !value.is_some_and(|v| any_element(v, |x| same_json(x, target)))
            }
            FilterOperation::NotIn(targets) => {
                value.is_some_and(|v| !any_element(v, |x| targets.iter().any(|t| same_json(x, t))))
            }
            positive => value.is_some_and(|v| any_element(v, |x| scalar_matches(x, positive))),
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
            payload_fields: None,
            score_threshold: None,
        }
    }
}

/// Whether a field satisfies a test: the value itself, or — when the field holds an array —
/// any one of its elements.
///
/// `tags: ["a", "b"]` has to match `tags == "a"`. That is how Qdrant answers it (an array
/// matches when any element does, `ValueChecker::_check`), Chroma and Weaviate offer it through
/// `$contains` and `ContainsAny`, and it is what filtering on tags means to anyone doing it.
/// This compared the whole array with `"a"` and never matched. One level only: an array inside
/// an array is an element that is itself an array, and matches nothing scalar — Qdrant does not
/// flatten either.
fn any_element(value: &serde_json::Value, test: impl Fn(&serde_json::Value) -> bool) -> bool {
    match value {
        serde_json::Value::Array(elements) => elements.iter().any(test),
        scalar => test(scalar),
    }
}

/// A positive operation against one value.
fn scalar_matches(value: &serde_json::Value, operation: &FilterOperation) -> bool {
    let number = || value.as_f64();
    let text = || value.as_str();
    match operation {
        FilterOperation::Equal(target) => same_json(value, target),
        FilterOperation::In(targets) => targets.iter().any(|t| same_json(value, t)),
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
        FilterOperation::Contains(substr) => text().is_some_and(|s| s.contains(substr.as_str())),
        FilterOperation::StartsWith(prefix) => {
            text().is_some_and(|s| s.starts_with(prefix.as_str()))
        }
        FilterOperation::EndsWith(suffix) => text().is_some_and(|s| s.ends_with(suffix.as_str())),
        // Answered as complements in `matches_condition`, never per element.
        FilterOperation::NotEqual(_) | FilterOperation::NotIn(_) => {
            unreachable!("negations are complements of the positive forms")
        }
    }
}

/// Equality between a stored value and a filter's target, with numbers compared as numbers.
///
/// `serde_json::Value`'s own equality keeps integers and floats apart, so `5 == 5.0` was false
/// here — while the payload index, which keys every number as an `f64`, said true, and the
/// range operations below have always compared as `f64`. The same filter then answered one way
/// in `records/list` and another in a search. Numbers are numbers: JSON has one kind, and a
/// JavaScript client cannot send `5.0` at all (`JSON.stringify(5.0)` is `"5"`) while a Python
/// one sends exactly that, so strict equality made the same query depend on which SDK asked.
fn same_json(value: &serde_json::Value, target: &serde_json::Value) -> bool {
    match (value.as_f64(), target.as_f64()) {
        (Some(a), Some(b)) if value.is_number() && target.is_number() => a == b,
        _ => value == target,
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

#[cfg(test)]
mod filter_validation {
    use super::*;
    use serde_json::json;

    fn with(op: FilterOperation) -> Condition {
        Condition {
            key: "tags".into(),
            operation: op,
        }
    }

    #[test]
    fn an_array_or_object_target_is_refused_in_every_list() {
        let composite = [
            FilterOperation::Equal(json!(["a", "b"])),
            FilterOperation::NotEqual(json!({"a": 1})),
            FilterOperation::In(vec![json!("a"), json!(["a"])]),
            FilterOperation::NotIn(vec![json!({"a": 1})]),
        ];
        for op in composite {
            for filter in [
                Filter {
                    must: Some(vec![with(op.clone())]),
                    must_not: None,
                    should: None,
                },
                Filter {
                    must: None,
                    must_not: Some(vec![with(op.clone())]),
                    should: None,
                },
                Filter {
                    must: None,
                    must_not: None,
                    should: Some(vec![with(op.clone())]),
                },
            ] {
                let error = filter
                    .validate()
                    .expect_err(&format!("{op:?} was accepted"));
                assert!(
                    error.contains("\"tags\""),
                    "the message names the field: {error}"
                );
            }
        }
    }

    /// Scalars, null included, and every operation that does not compare whole values.
    #[test]
    fn everything_else_is_accepted() {
        for op in [
            FilterOperation::Equal(json!("a")),
            FilterOperation::Equal(json!(5.0)),
            FilterOperation::Equal(json!(null)),
            FilterOperation::NotEqual(json!(true)),
            FilterOperation::In(vec![json!("a"), json!(1), json!(null)]),
            FilterOperation::NotIn(vec![]),
            FilterOperation::Contains("a".into()),
            FilterOperation::GreaterThan(json!(1)),
        ] {
            let filter = Filter {
                must: Some(vec![with(op.clone())]),
                must_not: None,
                should: None,
            };
            assert_eq!(filter.validate(), Ok(()), "{op:?} was refused");
        }
        let empty = Filter {
            must: None,
            must_not: None,
            should: None,
        };
        assert_eq!(empty.validate(), Ok(()));
    }
}
