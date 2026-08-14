use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::fmt;

use crate::types::{vector::DistanceMetric};

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
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexType::BruteForce => write!(f, "brute_force"),
            IndexType::Hnsw => write!(f, "hnsw"),
            IndexType::Ivf => write!(f, "ivf"),
            IndexType::OptimizedHnsw => write!(f, "optimized_hnsw"),
            IndexType::Lsh => write!(f, "lsh"),
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
}

/// Named vector space configuration (Qdrant-style multi-vector support)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct VectorSpaceConfig {
    /// Name of this vector space (e.g. "text_embedding", "image_embedding")
    pub name: String,
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
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
    /// Optional named vector spaces (multi-vector support, Qdrant-style).
    /// When set, `dimension` serves as the default vector space named "default".
    #[serde(default)]
    pub named_vectors: Option<Vec<VectorSpaceConfig>>,
}

fn default_schema_version() -> u64 {
    1
}

impl Table {
    /// 创建新表
    pub fn new(config: TableConfig) -> crate::Result<Self> {
        let now = chrono::Utc::now();

        Ok(Table {
            id: uuid::Uuid::new_v4(),
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
        if let Some(fts) = req.enable_full_text_search {
            updated.enable_full_text_search = fts;
        }

        // Mutate payload_schema
        let mut schema = updated.payload_schema.unwrap_or_default();
        for field in &req.add_fields {
            if !schema.iter().any(|f| f.name == field.name) {
                schema.push(field.clone());
            }
        }
        schema.retain(|f| !req.drop_fields.contains(&f.name));
        updated.payload_schema = if schema.is_empty() { None } else { Some(schema) };

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
        write!(f, "Table[{}] {} ({}D, {})",
               self.id, self.name, self.dimension, self.index_type)
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
        };

        let table = Table::new(config)?;
        let stats = table.get_stats();

        assert_eq!(stats.tablename, "stats_table");
        assert_eq!(stats.dimension, 256);
        assert_eq!(stats.record_count, 0);
        assert_eq!(stats.enable_full_text_search, false);
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
        };

        let table3 = Table::new(config_default)?;
        assert_eq!(table3.name, "default_table");

        Ok(())
    }
}
