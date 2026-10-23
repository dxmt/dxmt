use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

use crate::types::vector::DistanceMetric;

/// 创建表的请求（兼容API）
pub type CreateTableRequest = TableConfig;

/// 表的基本信息
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct Table {
    pub id: Uuid,
    pub name: String,
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub record_count: u64,
    pub index_type: IndexType,
    /// 全文搜索字段配置
    pub text_fields: Option<Vec<TextFieldConfig>>,
    /// 是否启用全文搜索
    pub enable_full_text_search: bool,
    /// Schema version, incremented on each alter_table call
    #[serde(default = "default_schema_version")]
    pub schema_version: u64,
    /// Optional declared payload schema
    #[serde(default)]
    pub payload_schema: Option<Vec<FieldDef>>,
    /// Memory optimization setting persisted with the table
    #[serde(default)]
    pub memory_optimization: Option<bool>,
    /// Named vector spaces configuration (persisted for index recreation)
    #[serde(default)]
    pub named_vectors: Option<Vec<VectorSpaceConfig>>,
    /// Graph parameters this table was created with, persisted so a rebuild after a restart
    /// builds the same graph rather than the server's current defaults.
    #[serde(default)]
    pub hnsw: Option<HnswParams>,
    /// Quantization parameters this table was created with, persisted for the same reason.
    #[serde(default)]
    pub quantization: Option<QuantizationParams>,
    /// Which payload fields this table indexes for filtering. Unset means every field.
    #[serde(default)]
    pub payload_index: Option<PayloadIndexPolicy>,
    /// How this table turns text into a vector, if it does. Unset means the caller sends
    /// vectors, which is what every table did before this existed.
    #[serde(default)]
    pub embedding: Option<EmbeddingConfig>,
    /// A cross-encoder provider a search may rerank its page with; see [`RerankConfig`].
    #[serde(default)]
    pub rerank: Option<RerankConfig>,
    /// recall@10 this table's `normal` search mode should reach, measured against the
    /// table's own exact scan. Unset means the server's default (`DXMT_RECALL_TARGET`,
    /// 0.95); `0` turns calibration off for this table.
    #[serde(default)]
    pub recall_target: Option<f64>,
    /// What the last calibration found: the `ef_search` that meets the target on this
    /// table's own data, and the curve it was chosen from. Derived from the records, so a
    /// restart may lose it; the next sweep measures again.
    #[serde(default)]
    pub recall: Option<RecallCalibration>,
}

/// One calibration of a table's traversal budget against its own exact scan.
///
/// HNSW's recall at a given `ef` is a property of the corpus: on glove-100, M=16, hnswlib and
/// this graph both give 0.92 at ef 512 and 0.97 at 2,048, while a 768-dimension embedding
/// corpus clears 0.98 at 512. A fixed budget therefore under-delivers on the hard corpus or
/// over-spends on the easy one. The calibration samples stored vectors, scores them exactly,
/// walks the graph at a doubling series of budgets, and keeps the smallest that meets the
/// target. The queries are points a third of the way between pairs of stored rows: a stored
/// row itself is a node whose true neighbours are its own edges and flatters the graph badly
/// (0.90 where held-out queries got 0.21, on 128-d noise), while the interpolated point is
/// off the graph like a real query and tracks held-out recall within a few points, erring
/// low. The target is a floor on that measurement, not a promise about every query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct RecallCalibration {
    /// The recall@10 asked for.
    pub target: f64,
    /// The budget `normal` uses now; `accurate` is four times it.
    pub ef_search: usize,
    /// recall@10 measured at `ef_search` on the sample.
    pub measured: f32,
    /// Whether `measured` reached `target`. False means the largest budget tried fell short
    /// and `ef_search` is that largest budget.
    pub met: bool,
    /// Queries in the sample.
    pub sample: usize,
    /// Rows in the table when measured; the sweep measures again once the table has doubled.
    pub at_rows: u64,
    pub at: DateTime<Utc>,
    /// Every budget tried, smallest first.
    pub curve: Vec<CalibrationPoint>,
}

/// One point on a calibration curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct CalibrationPoint {
    pub ef_search: usize,
    pub recall: f32,
    /// Median single-query latency inside the server at this budget, milliseconds.
    pub p50_ms: f32,
}

/// 索引类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub enum IndexType {
    /// 暴力搜索
    #[default]
    BruteForce,
    /// HNSW索引
    Hnsw,
    /// IVF索引
    Ivf,
    /// 优化的HNSW索引
    #[serde(rename = "optimized_hnsw")]
    OptimizedHnsw,
    /// LSH索引
    Lsh,
    /// IVF-PQ索引
    #[serde(rename = "ivf_pq")]
    IvfPq,
    /// 乘积量化 HNSW
    #[serde(rename = "hnsw_pq")]
    HnswPq,
    /// 磁盘常驻图索引
    #[serde(rename = "enhanced_diskann")]
    EnhancedDiskAnn,
    /// 分区 IVF，分区常驻对象存储、按需加载
    #[serde(rename = "partitioned_ivf")]
    PartitionedIvf,
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexType::BruteForce => write!(f, "brute_force"),
            IndexType::Hnsw => write!(f, "hnsw"),
            IndexType::Ivf => write!(f, "ivf"),
            IndexType::OptimizedHnsw => write!(f, "optimized_hnsw"),
            IndexType::Lsh => write!(f, "lsh"),
            IndexType::HnswPq => write!(f, "hnsw_pq"),
            IndexType::EnhancedDiskAnn => write!(f, "enhanced_diskann"),
            IndexType::PartitionedIvf => write!(f, "partitioned_ivf"),
            IndexType::IvfPq => write!(f, "ivf_pq"),
        }
    }
}

/// 文本分析器类型
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub enum TextAnalyzer {
    /// 标准分析器（默认）
    #[default]
    Standard,
    /// 简单分析器（仅分词）
    Simple,
    /// 空格分析器
    Whitespace,
    /// 中文分析器 (Jieba分词)
    Chinese,
    /// 语言特定分析器（如 "en", "zh", "ja"）
    Language(String),
}

/// 文本源配置
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub enum TextSource {
    /// 从元数据字段提取（如 metadata["description"]）
    MetadataField(String),
    /// 专用文本字段
    Dedicated(String),
}

/// 文本字段配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct TextFieldConfig {
    /// 字段名称
    pub name: String,
    /// 文本来源
    pub source: TextSource,
    /// 分析器类型
    #[serde(default)]
    pub analyzer: TextAnalyzer,
    /// 权重加成（用于搜索排序）
    #[serde(default = "default_boost")]
    pub boost: f32,
}

fn default_boost() -> f32 {
    1.0
}

/// Payload field type for schema definition
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub enum FieldType {
    String,
    Number,
    Bool,
    /// Any JSON value (no type enforcement)
    #[default]
    Any,
}

/// A single payload field declaration (optional schema enforcement)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct FieldDef {
    pub name: String,
    #[serde(default)]
    pub field_type: FieldType,
    #[serde(default = "default_true")]
    pub nullable: bool,
    pub default_value: Option<serde_json::Value>,
}

fn default_true() -> bool {
    true
}

/// Request to alter an existing table (only safe mutations)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct TableAlterRequest {
    pub description: Option<String>,
    pub memory_optimization: Option<bool>,
    /// If changed to true, triggers async full-text index build
    pub enable_full_text_search: Option<bool>,
    /// Add new payload field declarations to the schema
    #[serde(default)]
    pub add_fields: Vec<FieldDef>,
    /// Drop payload field declarations (records keep data, schema drops tracking)
    #[serde(default)]
    pub drop_fields: Vec<String>,
    /// Replace which payload fields are indexed for filtering. The payload index is rebuilt
    /// on the next read that needs it, so a narrowed policy frees its memory then.
    #[serde(default)]
    pub payload_index: Option<PayloadIndexPolicy>,
    /// Repoint this table's embedding: a new `base_url`, `api_key_env`, `field` or
    /// `batch_size`.
    ///
    /// **`model` may not change.** Every vector in the table came from the named model, and a
    /// query embedded by a different one is not a worse answer but an unrelated one — arriving
    /// as a normal ranked page with no sign anything is wrong. Moving a corpus to a new model
    /// means re-embedding it, which is a new table. What this is for is the endpoint moving,
    /// which is a real thing that happens and which otherwise needs the table recreated.
    #[serde(default)]
    pub embedding: Option<EmbeddingConfig>,
    /// Set this table's recall target (see [`Table::recall_target`]); `0` turns calibration
    /// off. Any previous calibration is discarded and the next sweep measures against the
    /// new target.
    #[serde(default)]
    pub recall_target: Option<f64>,
    /// A cross-encoder provider a search may rerank its page with; see [`RerankConfig`].
    ///
    /// **Last, and new fields go after it.** This struct is logged in `AlterTable` WAL entries,
    /// which `rmp_serde::to_vec` writes as an array: a field's position is its identity there.
    /// Inserted before `recall_target`, it made a 1.2 log's recall target decode as a rerank
    /// config and the replay fail. `alter_request_logged_by_1_2_replays` pins the order.
    #[serde(default)]
    pub rerank: Option<RerankConfig>,
    /// Replace the table's full-text fields -- which payload fields are indexed, and with which
    /// analyzer -- and rebuild its text index from the records. Turns full-text search on if it
    /// was off. The current index keeps answering while the new one is built, and writes made
    /// meanwhile reach both; the request returns once the new index has taken over.
    ///
    /// **Last, and new fields go after it** (see `rerank`).
    #[serde(default)]
    pub text_fields: Option<Vec<TextFieldConfig>>,
}

/// Named vector space configuration (multi-vector support)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct VectorSpaceConfig {
    /// Name of this vector space (e.g. "text_embedding", "image_embedding")
    pub name: String,
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
}

/// Graph build parameters for a table, overriding the defaults for this table only.
///
/// `m` and `ef_construction` decide the graph, and they cannot be recovered afterwards: a
/// search can look harder (`search_mode`), but no amount of searching adds edges the build
/// never made. They were a global default and an environment variable, which is the wrong
/// grain — a 5,000-row lookup table and a 10M-row corpus on one server want different graphs,
/// and the environment cannot say so per table.
///
/// Unset fields keep the server's defaults. They are persisted with the table, so a rebuild
/// and a restart use the same numbers the table was built with.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct HnswParams {
    /// Edges per node. Higher is a denser graph: better recall, more memory, slower build.
    pub m: Option<usize>,
    /// Candidates considered while building. Higher is a better graph and a slower ingest.
    pub ef_construction: Option<usize>,
    /// Starting traversal budget at query time. `search_mode` scales it per query, and the
    /// server widens it as the table grows.
    pub ef_search: Option<usize>,
}

/// Product-quantization parameters for a table, overriding the defaults for this table only.
///
/// These decide what the codes cost and what they lose. The defaults are calibrated on
/// 768-dimension corpora (four dimensions per subspace, 256 centroids); a different width or a
/// different accuracy target wants different numbers, and until now the only way to say so was
/// to change them for every table on the server.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct QuantizationParams {
    /// Subspaces a vector is split into; one byte of code per subspace at 256 centroids.
    /// Rounded to a divisor of the dimension.
    pub num_subvectors: Option<usize>,
    /// Centroids per subspace. 256 keeps a code to one byte.
    pub num_centroids: Option<usize>,
    /// Rows at which the graph trains its codebooks and switches to codes. Below it the index
    /// is an exact HNSW graph, so this is also how to say "never quantize this table": set it
    /// higher than the table will ever grow.
    pub quantize_at_rows: Option<usize>,
    /// Keep the 4-bit block-scan codes beside the 8-bit ones (default true). They cost about
    /// the 8-bit code's bytes again and make a filtered search over a large match set a flat
    /// scan; turn them off for a table that is never filtered, or for a server whose cores
    /// are all spoken for -- the scan is parallel and buys latency with CPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fast_scan: Option<bool>,
}

/// Which wire shape the rerank provider speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum RerankFormat {
    /// `POST {base_url}/rerank` with `{model, query, documents, top_n}`, answered by
    /// `{results: [{index, relevance_score}]}`: Cohere's v2 API, Jina, Voyage, and most
    /// hosted rerankers.
    #[default]
    Cohere,
    /// `POST {base_url}/rerank` with `{query, texts}`, answered by `[{index, score}]`:
    /// HuggingFace text-embeddings-inference serving a cross-encoder.
    Tei,
}

/// How this table re-scores a candidate page with a cross-encoder, when a search asks.
///
/// The server has no model of its own, as with `embedding`; it calls a provider with the
/// query and each candidate's text and reorders the page by what comes back. A search opts
/// in per request (`rerank` on `/search`, `/search/text`, `/search/hybrid`) and the table
/// says where the model is. The key is never in the table: `api_key_env` names a variable of
/// the server process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct RerankConfig {
    /// Model name, passed to the provider verbatim (`rerank-v3.5`, `jina-reranker-v2-base-multilingual`,
    /// `BAAI/bge-reranker-v2-m3`, …).
    pub model: String,
    /// Base URL of the provider, without the trailing `/rerank`.
    pub base_url: String,
    /// Wire shape; defaults to `cohere`.
    #[serde(default)]
    pub format: RerankFormat,
    /// Environment variable holding the API key. Defaults to `RERANK_API_KEY`; a provider
    /// that needs no key is served by naming a variable that is unset.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// The payload field holding each record's text. Defaults to the table's embedding
    /// field, else `text`.
    #[serde(default)]
    pub field: Option<String>,
    /// Candidates fetched and sent to the provider when a request does not say. Defaults to
    /// five times the page.
    #[serde(default)]
    pub top_n: Option<usize>,
}

impl RerankConfig {
    pub fn api_key_env(&self) -> &str {
        self.api_key_env.as_deref().unwrap_or("RERANK_API_KEY")
    }
}

/// How this table turns text into a vector, so a caller can write and search with text.
///
/// Without this, every client has to run an embedding model itself and keep it in step with
/// whatever the table was loaded with — and the failure when it drifts is not an error but a
/// worse answer. A table that names its model closes that: the text goes in, the server calls
/// the model, and every client and every protocol gets the same vectors.
///
/// **One provider shape: the OpenAI `/embeddings` endpoint.** It is the de-facto interface, so
/// one implementation reaches OpenAI, Jina, Voyage, text-embeddings-inference, Infinity,
/// ollama, vLLM, LM Studio and llama.cpp — `base_url` is the whole difference between them.
/// A second provider here would be a second thing to keep working for no new reach.
///
/// **The key is never in this struct**, so it is never in a request body, a snapshot, an
/// export or a `GET /tables/{t}` response. `api_key_env` names an environment variable of the
/// server process; the value is read at call time.
///
/// **A provider that will not answer is an error, not a zero vector.** The hashed bag of words
/// that used to stand in for a model is gone for that reason: a wrong vector is stored, ranks
/// badly forever, and looks exactly like a working system.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// Model name, passed to the provider verbatim (`text-embedding-3-small`,
    /// `BAAI/bge-m3`, `nomic-embed-text`, …).
    pub model: String,
    /// Base URL of an OpenAI-compatible API, without the trailing `/embeddings`.
    /// Defaults to `https://api.openai.com/v1`.
    #[serde(default)]
    pub base_url: Option<String>,
    /// The payload field holding the text to embed. Defaults to `text`.
    ///
    /// A record's top-level `text` is written here, so the text a hit came from is in the
    /// payload that comes back — which is what a retrieval caller needs next, and what it
    /// would otherwise have to store a second copy of.
    #[serde(default)]
    pub field: Option<String>,
    /// Environment variable holding the API key. Defaults to `OPENAI_API_KEY`; a provider that
    /// needs no key (a local one, usually) is served by setting it to a variable that is unset.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Texts per request to the provider. Defaults to 64.
    ///
    /// Providers cap both the number of inputs and the total tokens per call, and the token cap
    /// is the one that bites: 64 chunks of a document is a normal request, 2048 is not.
    #[serde(default)]
    pub batch_size: Option<usize>,
}

impl EmbeddingConfig {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";
    pub const DEFAULT_FIELD: &'static str = "text";
    pub const DEFAULT_API_KEY_ENV: &'static str = "OPENAI_API_KEY";
    pub const DEFAULT_BATCH_SIZE: usize = 64;

    pub fn base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .map(|u| u.trim_end_matches('/'))
            .unwrap_or(Self::DEFAULT_BASE_URL)
    }

    pub fn field(&self) -> &str {
        self.field.as_deref().unwrap_or(Self::DEFAULT_FIELD)
    }

    pub fn api_key_env(&self) -> &str {
        self.api_key_env
            .as_deref()
            .unwrap_or(Self::DEFAULT_API_KEY_ENV)
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
            .unwrap_or(Self::DEFAULT_BATCH_SIZE)
            .clamp(1, 2048)
    }
}

/// Which payload fields this table keeps a filter index for.
///
/// Every payload field is indexed by default, which is what makes a filter fast without anyone
/// declaring anything — Qdrant and Weaviate both require the declaration, and a filter on an
/// undeclared field there is a full scan. The cost of doing it for free is memory and rebuild
/// time, and it is not evenly spread: on a 10M-row table one field holding a distinct string
/// per row cost 11 s of a 32 s index rebuild, against 3.5 s for a four-valued one. A field
/// nobody filters on is paying that for nothing.
///
/// So a table can narrow it. `only` names the fields to index, and `except` names fields never
/// to index; a field is indexed when `only` is unset or names it, and `except` does not.
///
/// **A filter on a field this policy excludes is refused**, not answered. The index is the only
/// thing that answers a filter here — there is no scan fallback — so the alternative to an
/// error is an empty result that looks exactly like "nothing matched". Re-index the field with
/// `PUT /tables/{t}/payload_index/{field}` and the filter works again.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct PayloadIndexPolicy {
    /// When set, the only payload fields that are indexed. Unset means every field.
    #[serde(default)]
    pub only: Option<Vec<String>>,
    /// Fields never indexed, whether or not `only` names them.
    #[serde(default)]
    pub except: Option<Vec<String>>,
}

impl PayloadIndexPolicy {
    /// Whether a payload field is indexed under this policy.
    ///
    /// Names match subtrees: `except: ["blob"]` also excludes `blob.text`, and
    /// `only: ["meta"]` also indexes `meta.lang`. A nested payload is indexed path by path, so
    /// a policy that matched only whole paths would make `except` useless against the field
    /// that actually costs something -- the object with twenty keys under it -- and would make
    /// `only` name every leaf by hand.
    pub fn indexes(&self, field: &str) -> bool {
        fn covers(names: &[String], field: &str) -> bool {
            names.iter().any(|name| {
                field == name
                    || (field.len() > name.len()
                        && field.starts_with(name.as_str())
                        && field.as_bytes()[name.len()] == b'.')
            })
        }
        if self.except.as_ref().is_some_and(|f| covers(f, field)) {
            return false;
        }
        match &self.only {
            Some(only) => covers(only, field),
            None => true,
        }
    }

    /// True when this policy indexes everything, which is the default and the fast path.
    pub fn indexes_everything(&self) -> bool {
        self.only.is_none() && self.except.as_ref().is_none_or(|f| f.is_empty())
    }

    /// Start indexing `field` again: out of `except`, and into `only` when there is one.
    pub fn include(&mut self, field: &str) {
        if let Some(except) = &mut self.except {
            except.retain(|n| n != field);
        }
        if let Some(only) = &mut self.only {
            if !only.iter().any(|n| n == field) {
                only.push(field.to_string());
            }
        }
    }

    /// The first field in `filter` that this policy does not index, if any.
    ///
    /// `id` is not a payload field — a condition on it is answered from the id map — so it is
    /// always answerable.
    pub fn unanswerable_field(&self, filter: &crate::types::record::Filter) -> Option<String> {
        if self.indexes_everything() {
            return None;
        }
        let walk = |conditions: &Option<Vec<crate::types::record::Condition>>| {
            conditions.as_ref().and_then(|list| {
                list.iter()
                    .find(|c| c.key != "id" && !self.indexes(&c.key))
                    .map(|c| c.key.clone())
            })
        };
        walk(&filter.must)
            .or_else(|| walk(&filter.should))
            .or_else(|| walk(&filter.must_not))
    }

    /// Stop indexing `field`.
    pub fn exclude(&mut self, field: &str) {
        if let Some(only) = &mut self.only {
            only.retain(|n| n != field);
        }
        let except = self.except.get_or_insert_with(Vec::new);
        if !except.iter().any(|n| n == field) {
            except.push(field.to_string());
        }
    }
}

/// 创建表的配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct TableConfig {
    pub name: String,
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
    pub description: Option<String>,
    pub index_type: Option<IndexType>,
    /// 全文搜索字段配置
    pub text_fields: Option<Vec<TextFieldConfig>>,
    /// 是否启用全文搜索
    #[serde(default)]
    pub enable_full_text_search: bool,
    /// 是否启用内存优化（可选，None = 使用全局配置）
    #[serde(default)]
    pub memory_optimization: Option<bool>,
    /// Optional payload schema for field tracking (does not enforce types on existing records)
    #[serde(default)]
    pub payload_schema: Option<Vec<FieldDef>>,
    /// Schema version, incremented on each alter_table call
    #[serde(default = "default_schema_version")]
    pub schema_version: u64,
    /// Graph parameters for this table, overriding the server defaults.
    #[serde(default)]
    pub hnsw: Option<HnswParams>,
    /// Quantization parameters for this table, overriding the server defaults.
    #[serde(default)]
    pub quantization: Option<QuantizationParams>,
    /// Which payload fields to keep a filter index for. Unset indexes every field.
    #[serde(default)]
    pub payload_index: Option<PayloadIndexPolicy>,
    /// Optional named vector spaces (multi-vector support).
    /// When set, `dimension` serves as the default vector space named "default".
    #[serde(default)]
    pub named_vectors: Option<Vec<VectorSpaceConfig>>,
    /// Model to embed text with, so records and queries may carry text instead of a vector.
    #[serde(default)]
    pub embedding: Option<EmbeddingConfig>,
    /// A cross-encoder provider a search may rerank its page with; see [`RerankConfig`].
    #[serde(default)]
    pub rerank: Option<RerankConfig>,
}

fn default_schema_version() -> u64 {
    1
}

impl Table {
    /// The namespace `table_id` hashes names in. Arbitrary, fixed, and never to be changed:
    /// changing it renames every table's id and orphans everything keyed by one.
    const ID_NAMESPACE: uuid::Uuid = uuid::Uuid::from_bytes([
        0x64, 0x78, 0x6d, 0x74, 0x2d, 0x74, 0x61, 0x62, 0x6c, 0x65, 0x2d, 0x69, 0x64, 0x76, 0x35,
        0x00,
    ]);

    /// A table's id, derived from its name rather than drawn at random.
    ///
    /// It used to be `Uuid::new_v4()`, which made the id a property of the *process* and not of
    /// the table: recovery rebuilds a table through `TableConfig`, `TableConfig` carries no id,
    /// and so every table got a new one on every restart. Two things were keyed by it and both
    /// broke silently.
    ///
    /// The full-text index lives at `/tmp/dxmt_full_text_{id}`, so after a restart the engine
    /// opened a *different, empty* directory: the table still reported
    /// `enable_full_text_search: true`, `/search/text` still answered 200, and it returned
    /// nothing at all. On this machine that had left 135 orphaned index directories behind five
    /// tables. Partitioned-IVF segments are keyed by it too; there the damage is a leak rather
    /// than a wrong answer, because the sweep only deletes under `partitions/{id}/` and after a
    /// restart that prefix is empty — so old segments are simply never collected.
    ///
    /// A name is what actually identifies a table across a restart: it is the map key, it is
    /// what the API takes, and it is what the WAL records. Deriving the id from it makes the
    /// invariant structural rather than something a future recovery path has to remember to
    /// preserve — including the WAL replay path, which has no id to preserve.
    ///
    /// The cost is that a dropped and recreated table gets its old id back, so anything keyed by
    /// id must be cleaned up on drop. `drop_table` does that.
    pub fn table_id(name: &str) -> uuid::Uuid {
        uuid::Uuid::new_v5(&Self::ID_NAMESPACE, name.as_bytes())
    }

    /// 创建新表
    pub fn new(config: TableConfig) -> crate::Result<Self> {
        let now = chrono::Utc::now();
        let id = Self::table_id(&config.name);

        Ok(Table {
            id,
            name: config.name,
            dimension: config.dimension,
            distance_metric: config.distance_metric,
            description: config.description,
            created_at: now,
            updated_at: now,
            record_count: 0,
            index_type: config.index_type.unwrap_or(IndexType::BruteForce),
            text_fields: config.text_fields,
            enable_full_text_search: config.enable_full_text_search,
            schema_version: config.schema_version,
            payload_schema: config.payload_schema,
            memory_optimization: config.memory_optimization,
            named_vectors: config.named_vectors,
            // Persisted, so a rebuild after a restart builds the graph this table asked for
            // rather than whatever the server defaults to that day.
            hnsw: config.hnsw,
            quantization: config.quantization,
            payload_index: config.payload_index,
            embedding: config.embedding,
            rerank: config.rerank,
            recall_target: None,
            recall: None,
        })
    }

    /// Apply an alter request, returning a new Table with schema_version incremented.
    pub fn apply_alter(&self, req: &TableAlterRequest) -> Self {
        let mut updated = self.clone();
        updated.updated_at = chrono::Utc::now();
        updated.schema_version += 1;

        if let Some(desc) = &req.description {
            updated.description = Some(desc.clone());
        }
        if let Some(mo) = req.memory_optimization {
            updated.memory_optimization = Some(mo);
        }
        if let Some(fields) = &req.text_fields {
            updated.text_fields = Some(fields.clone());
            updated.enable_full_text_search = true;
        }
        if let Some(fts) = req.enable_full_text_search {
            updated.enable_full_text_search = fts;
        }
        if let Some(policy) = &req.payload_index {
            updated.payload_index = Some(policy.clone());
        }
        if let Some(embedding) = &req.embedding {
            updated.embedding = Some(embedding.clone());
        }
        if let Some(rerank) = &req.rerank {
            updated.rerank = Some(rerank.clone());
        }
        if let Some(target) = req.recall_target {
            updated.recall_target = Some(target);
            updated.recall = None;
        }

        // Mutate payload_schema
        let mut schema = updated.payload_schema.unwrap_or_default();
        for field in &req.add_fields {
            if !schema.iter().any(|f| f.name == field.name) {
                schema.push(field.clone());
            }
        }
        schema.retain(|f| !req.drop_fields.contains(&f.name));
        updated.payload_schema = if schema.is_empty() {
            None
        } else {
            Some(schema)
        };

        updated
    }

    /// 更新记录计数
    pub fn update_record_count(&mut self, new_count: u64) {
        self.record_count = new_count;
        self.updated_at = chrono::Utc::now()
    }

    /// 获取表的统计信息
    pub fn get_stats(&self) -> TableStats {
        TableStats {
            table_id: self.id,
            tablename: self.name.clone(),
            record_count: self.record_count,
            dimension: self.dimension,
            distance_metric: self.distance_metric,
            index_type: self.index_type,
            created_at: self.created_at,
            updated_at: self.updated_at,
            enable_full_text_search: self.enable_full_text_search,
            text_fields_count: self.text_fields.as_ref().map(|f| f.len()).unwrap_or(0),
        }
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Table[{}] {} ({}D, {})",
            self.id, self.name, self.dimension, self.index_type
        )
    }
}

/// 表的统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStats {
    pub table_id: Uuid,
    pub tablename: String,
    pub record_count: u64,
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
    pub index_type: IndexType,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub enable_full_text_search: bool,
    pub text_fields_count: usize,
}

impl std::str::FromStr for IndexType {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.to_lowercase().as_str() {
            "brute_force" => Ok(IndexType::BruteForce),
            "hnsw" => Ok(IndexType::Hnsw),
            "ivf" => Ok(IndexType::Ivf),
            "optimized_hnsw" => Ok(IndexType::OptimizedHnsw),
            "lsh" => Ok(IndexType::Lsh),
            "ivf_pq" | "ivfpq" => Ok(IndexType::IvfPq),
            _ => Err(crate::error::Error::Internal(format!(
                "Unknown index type: {}",
                s
            ))),
        }
    }
}

/// 表模式类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum TableSchema {
    /// 探索性模式 - 灵活的schema，适合原型开发
    #[default]
    Exploratory,
    /// 结构化模式 - 严格的schema，适合生产环境
    Structured,
    /// 企业级模式 - 带有高级功能的严格schema
    Enterprise,
}

impl std::fmt::Display for TableSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TableSchema::Exploratory => write!(f, "exploratory"),
            TableSchema::Structured => write!(f, "structured"),
            TableSchema::Enterprise => write!(f, "enterprise"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct TableInfoResponse {
    pub table: Table,
    pub record_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct TableWithCount {
    pub table: Table,
    pub record_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `PATCH /tables` logged by 1.2 must replay on this binary. The WAL encodes the request
    /// as a MessagePack array, so the frozen struct below is 1.2's field order, and the live
    /// type has to read it field for field -- the case that broke was a recall target landing
    /// in a field added in the middle.
    #[test]
    fn alter_request_logged_by_1_2_replays() {
        #[derive(Serialize)]
        struct AlterRequestV1_2 {
            description: Option<String>,
            memory_optimization: Option<bool>,
            enable_full_text_search: Option<bool>,
            add_fields: Vec<FieldDef>,
            drop_fields: Vec<String>,
            payload_index: Option<PayloadIndexPolicy>,
            embedding: Option<EmbeddingConfig>,
            recall_target: Option<f64>,
        }
        let logged = rmp_serde::to_vec(&AlterRequestV1_2 {
            description: Some("before".into()),
            memory_optimization: None,
            enable_full_text_search: Some(true),
            add_fields: Vec::new(),
            drop_fields: vec!["old".into()],
            payload_index: None,
            embedding: None,
            recall_target: Some(0.97),
        })
        .unwrap();
        let replayed: TableAlterRequest = rmp_serde::from_slice(&logged)
            .expect("a 1.2 AlterTable entry must decode as the current request");
        assert_eq!(replayed.recall_target, Some(0.97));
        assert_eq!(replayed.description.as_deref(), Some("before"));
        assert_eq!(replayed.drop_fields, vec!["old".to_string()]);
        assert!(replayed.rerank.is_none());

        // And the new field survives its own round trip in the array form.
        let now = TableAlterRequest {
            recall_target: Some(0.9),
            rerank: Some(RerankConfig {
                model: "m".into(),
                base_url: "http://127.0.0.1:1".into(),
                format: RerankFormat::Tei,
                api_key_env: None,
                field: None,
                top_n: Some(20),
            }),
            ..Default::default()
        };
        let back: TableAlterRequest =
            rmp_serde::from_slice(&rmp_serde::to_vec(&now).unwrap()).unwrap();
        assert_eq!(back.recall_target, Some(0.9));
        assert_eq!(back.rerank, now.rerank);
    }

    #[test]
    fn test_table_creation() -> crate::Result<()> {
        let config = TableConfig {
            name: "test_table".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Test table".to_string()),
            index_type: Some(IndexType::BruteForce),
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        let table = Table::new(config)?;
        assert_eq!(table.name, "test_table");
        assert_eq!(table.dimension, 128);
        assert_eq!(table.distance_metric, DistanceMetric::Cosine);
        assert_eq!(table.record_count, 0);
        Ok(())
    }

    #[test]
    fn test_table_stats() -> crate::Result<()> {
        let config = TableConfig {
            name: "stats_table".to_string(),
            dimension: 256,
            distance_metric: DistanceMetric::Euclidean,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        let table = Table::new(config)?;
        let stats = table.get_stats();

        assert_eq!(stats.tablename, "stats_table");
        assert_eq!(stats.dimension, 256);
        assert_eq!(stats.record_count, 0);
        assert!(!stats.enable_full_text_search);
        assert_eq!(stats.text_fields_count, 0);
        Ok(())
    }

    #[test]
    fn test_table_with_memory_optimization() -> crate::Result<()> {
        // 测试显式启用内存优化
        let config_enabled = TableConfig {
            name: "optimized_table".to_string(),
            dimension: 768,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: Some(true),
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        let table = Table::new(config_enabled)?;
        assert_eq!(table.name, "optimized_table");

        // 测试显式禁用内存优化
        let config_disabled = TableConfig {
            name: "precision_table".to_string(),
            dimension: 512,
            distance_metric: DistanceMetric::Euclidean,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: Some(false),
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        let table2 = Table::new(config_disabled)?;
        assert_eq!(table2.name, "precision_table");

        // 测试使用全局配置（None）
        let config_default = TableConfig {
            name: "default_table".to_string(),
            dimension: 384,
            distance_metric: DistanceMetric::Dot,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        let table3 = Table::new(config_default)?;
        assert_eq!(table3.name, "default_table");

        Ok(())
    }
}
