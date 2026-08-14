use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use std::sync::Arc;

use super::errors::ObjectStorageError;
use super::types::{ListObjectsOptions, ListObjectsResult, ObjectMetadata, PartInfo};

pub type ObjectStorageResult<T> = Result<T, ObjectStorageError>;

/// 对象存储抽象接口
///
/// 提供统一的对象存储操作接口，支持不同的存储后端
#[async_trait]
pub trait ObjectStorage: Send + Sync {
    /// 上传对象
    ///
    /// # 参数
    /// * `key` - 对象键
    /// * `data` - 对象数据
    /// * `content_type` - 内容类型（可选）
    async fn put_object(
        &self,
        key: &str,
        data: Bytes,
        content_type: Option<&str>,
    ) -> ObjectStorageResult<()>;

    /// 下载对象
    ///
    /// # 参数
    /// * `key` - 对象键
    ///
    /// # 返回
    /// 对象数据，如果对象不存在则返回ObjectNotFound错误
    async fn get_object(&self, key: &str) -> ObjectStorageResult<Vec<u8>>;

    /// 获取对象的部分内容
    ///
    /// # 参数
    /// * `key` - 对象键
    /// * `start` - 起始字节位置（包含）
    /// * `end` - 结束字节位置（不包含）
    async fn get_object_range(
        &self,
        key: &str,
        start: u64,
        end: u64,
    ) -> ObjectStorageResult<Vec<u8>>;

    /// 删除对象
    ///
    /// # 参数
    /// * `key` - 对象键
    async fn delete_object(&self, key: &str) -> ObjectStorageResult<()>;

    /// 批量删除对象
    ///
    /// # 参数
    /// * `keys` - 对象键列表
    ///
    /// # 返回
    /// 成功删除的对象键列表
    async fn delete_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<String>>;

    /// 检查对象是否存在
    ///
    /// # 参数
    /// * `key` - 对象键
    async fn object_exists(&self, key: &str) -> ObjectStorageResult<bool>;

    /// 获取对象元数据
    ///
    /// # 参数
    /// * `key` - 对象键
    async fn head_object(&self, key: &str) -> ObjectStorageResult<ObjectMetadata>;

    /// 获取对象大小
    ///
    /// # 参数
    /// * `key` - 对象键
    async fn get_object_size(&self, key: &str) -> ObjectStorageResult<u64> {
        let metadata = self.head_object(key).await?;
        Ok(metadata.size)
    }

    /// 列出对象
    ///
    /// # 参数
    /// * `options` - 列表选项
    async fn list_objects(
        &self,
        options: &ListObjectsOptions,
    ) -> ObjectStorageResult<ListObjectsResult>;

    /// 复制对象
    ///
    /// # 参数
    /// * `source_key` - 源对象键
    /// * `dest_key` - 目标对象键
    async fn copy_object(&self, source_key: &str, dest_key: &str) -> ObjectStorageResult<()>;

    /// 批量上传对象
    ///
    /// # 参数
    /// * `objects` - (键, 数据) 元组列表
    async fn put_objects(
        &self,
        objects: Vec<(String, Vec<u8>)>,
    ) -> ObjectStorageResult<Vec<String>>;

    /// 批量下载对象
    ///
    /// # 参数
    /// * `keys` - 对象键列表
    ///
    /// # 返回
    /// (键, 数据) 元组列表，不存在的对象会被跳过
    async fn get_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<(String, Vec<u8>)>>;

    /// 流式上传对象
    ///
    /// # 参数
    /// * `key` - 对象键
    /// * `stream` - 数据流
    /// * `content_length` - 内容长度（可选）
    /// * `content_type` - 内容类型（可选）
    async fn put_stream(
        &self,
        key: &str,
        stream: BoxStream<'static, Result<Bytes, std::io::Error>>,
        content_length: Option<u64>,
        content_type: Option<&str>,
    ) -> ObjectStorageResult<()>;

    /// 流式下载对象
    ///
    /// # 参数
    /// * `key` - 对象键
    ///
    /// # 返回
    /// 数据流
    async fn get_stream(
        &self,
        key: &str,
    ) -> ObjectStorageResult<BoxStream<'static, Result<Bytes, ObjectStorageError>>>;

    /// 开始分片上传
    ///
    /// # 参数
    /// * `key` - 对象键
    /// * `content_type` - 内容类型（可选）
    ///
    /// # 返回
    /// 上传ID
    async fn create_multipart_upload(
        &self,
        key: &str,
        content_type: Option<&str>,
    ) -> ObjectStorageResult<String>;

    /// 上传分片
    ///
    /// # 参数
    /// * `key` - 对象键
    /// * `upload_id` - 上传ID
    /// * `part_number` - 分片号（从1开始）
    /// * `data` - 分片数据
    ///
    /// # 返回
    /// 分片信息
    async fn upload_part(
        &self,
        key: &str,
        upload_id: &str,
        part_number: u32,
        data: &[u8],
    ) -> ObjectStorageResult<PartInfo>;

    /// 完成分片上传
    ///
    /// # 参数
    /// * `key` - 对象键
    /// * `upload_id` - 上传ID
    /// * `parts` - 已上传的分片列表
    async fn complete_multipart_upload(
        &self,
        key: &str,
        upload_id: &str,
        parts: &[PartInfo],
    ) -> ObjectStorageResult<()>;

    /// 取消分片上传
    ///
    /// # 参数
    /// * `key` - 对象键
    /// * `upload_id` - 上传ID
    async fn abort_multipart_upload(&self, key: &str, upload_id: &str) -> ObjectStorageResult<()>;

    /// 获取存储统计信息
    async fn get_storage_stats(&self) -> ObjectStorageResult<StorageStats>;

    /// 健康检查
    async fn health_check(&self) -> ObjectStorageResult<()>;
}

/// 存储统计信息
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// 存储桶名称
    pub bucket: String,
    /// 对象总数
    pub object_count: u64,
    /// 总大小（字节）
    pub total_size: u64,
    /// 最近访问时间
    pub last_access: Option<chrono::DateTime<chrono::Utc>>,
}

/// 对象存储工厂
pub trait ObjectStorageFactory: Send + Sync {
    /// 创建对象存储实例
    fn create(&self) -> ObjectStorageResult<Arc<dyn ObjectStorage>>;
}
