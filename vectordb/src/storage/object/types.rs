use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 对象元数据信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    /// 对象键
    pub key: String,
    /// 对象大小（字节）
    pub size: u64,
    /// 最后修改时间
    pub last_modified: DateTime<Utc>,
    /// ETag（实体标签）
    pub etag: Option<String>,
    /// 内容类型
    pub content_type: Option<String>,
    /// 自定义元数据
    pub payload: HashMap<String, String>,
}

/// 对象存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStorageConfig {
    /// 存储提供商类型
    pub provider: StorageProvider,
    /// 存储桶名称
    pub bucket: String,
    /// 区域
    pub region: Option<String>,
    /// 端点URL（用于MinIO等自建服务）
    pub endpoint: Option<String>,
    /// 访问密钥ID
    pub access_key_id: Option<String>,
    /// 访问密钥秘密
    pub secret_access_key: Option<String>,
    /// 会话令牌（临时凭证）
    pub session_token: Option<String>,
    /// 路径前缀
    pub prefix: Option<String>,
}

impl Default for ObjectStorageConfig {
    fn default() -> Self {
        Self {
            provider: StorageProvider::Filesystem,
            bucket: "default".to_string(),
            region: None,
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            prefix: None,
        }
    }
}

impl From<crate::config::ObjectStorageConfig> for ObjectStorageConfig {
    fn from(config: crate::config::ObjectStorageConfig) -> Self {
        let provider = match config.provider.as_str() {
            "s3" => StorageProvider::S3,
            "minio" => StorageProvider::MinIO,
            "gcs" => StorageProvider::GCS,
            "azure" => StorageProvider::Azure,
            "filesystem" => StorageProvider::Filesystem,
            _ => StorageProvider::Filesystem, // Default or error handling
        };

        ObjectStorageConfig {
            provider,
            bucket: config.bucket,
            region: config.region,
            endpoint: config.endpoint,
            access_key_id: config.access_key_id,
            secret_access_key: config.secret_access_key,
            session_token: None, // Not available in config::ObjectStorageConfig
            prefix: None,        // Not available in config::ObjectStorageConfig
        }
    }
}

/// 存储提供商类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StorageProvider {
    /// AWS S3
    S3,
    /// MinIO
    MinIO,
    /// Google Cloud Storage
    GCS,
    /// Azure Blob Storage
    Azure,
    /// 本地文件系统
    Filesystem,
}

/// 对象信息（用于列表操作）
#[derive(Debug, Clone)]
pub struct ObjectInfo {
    /// 对象键
    pub key: String,
    /// 对象大小
    pub size: u64,
    /// 最后修改时间
    pub last_modified: DateTime<Utc>,
    /// ETag
    pub etag: Option<String>,
}

/// 分片上传信息
#[derive(Debug, Clone)]
pub struct MultipartUpload {
    /// 上传ID
    pub upload_id: String,
    /// 对象键
    pub key: String,
    /// 已上传的分片
    pub parts: Vec<PartInfo>,
}

/// 分片信息
#[derive(Debug, Clone)]
pub struct PartInfo {
    /// 分片号（从1开始）
    pub part_number: u32,
    /// 分片ETag
    pub etag: String,
    /// 分片大小
    pub size: u64,
}

/// 列表对象的选项
#[derive(Debug, Clone, Default)]
pub struct ListObjectsOptions {
    /// 前缀过滤
    pub prefix: Option<String>,
    /// 最大返回数量
    pub max_keys: Option<i32>,
    /// 继续标记（用于分页）
    pub continuation_token: Option<String>,
    /// 分隔符
    pub delimiter: Option<String>,
}

/// 列表对象的结果
#[derive(Debug, Clone)]
pub struct ListObjectsResult {
    /// 对象列表
    pub objects: Vec<ObjectInfo>,
    /// 公共前缀（当使用分隔符时）
    pub common_prefixes: Vec<String>,
    /// 是否截断（是否还有更多结果）
    pub is_truncated: bool,
    /// 下一页的继续标记
    pub next_continuation_token: Option<String>,
}
