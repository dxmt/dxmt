use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::stream::{self, BoxStream, StreamExt};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use super::errors::ObjectStorageError;
use super::traits::{ObjectStorage, ObjectStorageResult, StorageStats};
use super::types::{ListObjectsOptions, ListObjectsResult, ObjectInfo, ObjectMetadata, PartInfo};

/// 本地文件系统存储实现
///
/// 用于开发和测试环境，将对象存储在本地文件系统中
#[derive(Debug)]
pub struct FilesystemStorage {
    /// 根目录路径
    root_path: PathBuf,
    /// 存储桶名称
    bucket: String,
}

impl FilesystemStorage {
    /// 创建新的文件系统存储实例
    ///
    /// # 参数
    /// * `root_path` - 存储根目录路径
    /// * `bucket` - 存储桶名称
    pub fn new<P: AsRef<Path>>(rootpath: P, bucket: &str) -> ObjectStorageResult<Self> {
        let root_path = rootpath.as_ref().to_path_buf();

        Ok(Self {
            root_path,
            bucket: bucket.to_string(),
        })
    }

    /// 获取对象的完整文件路径
    fn get_object_path(&self, key: &str) -> PathBuf {
        self.root_path.join(&self.bucket).join(key)
    }

    /// 确保目录存在
    async fn ensure_dir_exists(&self, path: &Path) -> ObjectStorageResult<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| ObjectStorageError::InternalError {
                    message: format!("Directory creation failed: {:?}", e),
                })?;
        }
        Ok(())
    }

    /// 获取文件元数据
    async fn get_file_metadata(&self, path: &Path) -> ObjectStorageResult<std::fs::Metadata> {
        match fs::metadata(path).await {
            Ok(metadata) => Ok(metadata),
            Err(err) => Err(if err.kind() == std::io::ErrorKind::NotFound {
                ObjectStorageError::ObjectNotFound {
                    key: path.to_string_lossy().to_string(),
                }
            } else {
                ObjectStorageError::IoError {
                    message: err.to_string(),
                }
            }),
        }
    }

    /// 计算ETag（使用文件大小和修改时间的哈希）
    fn calculate_etag(&self, payload: &std::fs::Metadata) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        payload.len().hash(&mut hasher);
        if let Ok(modified) = payload.modified() {
            if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                duration.as_secs().hash(&mut hasher)
            }
        }
        format!("{:x}", hasher.finish())
    }
}

#[async_trait]
impl ObjectStorage for FilesystemStorage {
    async fn put_object(
        &self,
        key: &str,
        data: Bytes,
        _content_type: Option<&str>,
    ) -> ObjectStorageResult<()> {
        let path = self.get_object_path(key);
        self.ensure_dir_exists(&path).await?;

        fs::write(&path, data.as_ref())
            .await
            .map_err(|e| ObjectStorageError::IoError {
                message: e.to_string(),
            })?;
        Ok(())
    }

    async fn get_object(&self, key: &str) -> ObjectStorageResult<Vec<u8>> {
        let path = self.get_object_path(key);
        fs::read(&path).await.map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                ObjectStorageError::ObjectNotFound {
                    key: key.to_string(),
                }
            } else {
                ObjectStorageError::IoError {
                    message: err.to_string(),
                }
            }
        })
    }

    async fn get_object_range(
        &self,
        key: &str,
        start: u64,
        end: u64,
    ) -> ObjectStorageResult<Vec<u8>> {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        let path = self.get_object_path(key);
        let mut file = fs::File::open(&path).await.map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                ObjectStorageError::ObjectNotFound {
                    key: key.to_string(),
                }
            } else {
                ObjectStorageError::IoError {
                    message: err.to_string(),
                }
            }
        })?;

        file.seek(std::io::SeekFrom::Start(start))
            .await
            .map_err(|e| ObjectStorageError::IoError {
                message: e.to_string(),
            })?;
        let read_len = (end - start) as usize;
        let mut buffer = vec![0u8; read_len];
        let bytes_read = file
            .read(&mut buffer)
            .await
            .map_err(|e| ObjectStorageError::IoError {
                message: e.to_string(),
            })?;
        buffer.truncate(bytes_read);

        Ok(buffer)
    }

    async fn delete_object(&self, key: &str) -> ObjectStorageResult<()> {
        let path = self.get_object_path(key);
        if path.exists() {
            fs::remove_file(&path)
                .await
                .map_err(|e| ObjectStorageError::IoError {
                    message: e.to_string(),
                })?;
        }
        Ok(())
    }

    async fn delete_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<String>> {
        let mut deleted = Vec::new();

        for key in keys {
            if self.delete_object(key).await.is_ok() {
                deleted.push(key.clone());
            }
        }

        Ok(deleted)
    }

    async fn object_exists(&self, key: &str) -> ObjectStorageResult<bool> {
        let path = self.get_object_path(key);
        Ok(path.exists())
    }

    async fn head_object(&self, key: &str) -> ObjectStorageResult<ObjectMetadata> {
        let path = self.get_object_path(key);
        let metadata =
            self.get_file_metadata(&path)
                .await
                .map_err(|e| ObjectStorageError::IoError {
                    message: e.to_string(),
                })?;

        let last_modified = metadata
            .modified()
            .map(DateTime::<Utc>::from)
            .unwrap_or_else(|_| Utc::now());

        Ok(ObjectMetadata {
            key: key.to_string(),
            size: metadata.len(),
            last_modified,
            etag: Some(self.calculate_etag(&metadata)),
            content_type: Some("application/octet-stream".to_string()),
            payload: HashMap::new(),
        })
    }

    async fn list_objects(
        &self,
        options: &ListObjectsOptions,
    ) -> ObjectStorageResult<ListObjectsResult> {
        let bucket_path = self.root_path.join(&self.bucket);
        let mut objects = Vec::new();
        let mut common_prefixes = Vec::new();

        if !bucket_path.exists() {
            return Ok(ListObjectsResult {
                objects,
                common_prefixes,
                is_truncated: false,
                next_continuation_token: None,
            });
        }

        let prefix = options.prefix.as_deref().unwrap_or("");
        let delimiter = options.delimiter.as_deref();
        let max_keys = options.max_keys.unwrap_or(1000) as usize;

        // 递归遍历目录
        self.collect_objects(
            &bucket_path,
            prefix,
            delimiter,
            &mut objects,
            &mut common_prefixes,
        )
        .await
        .map_err(|e| ObjectStorageError::IoError {
            message: e.to_string(),
        })?;

        // 排序并限制数量
        objects.sort_by(|a, b| a.key.cmp(&b.key));
        let is_truncated = objects.len() > max_keys;
        if is_truncated {
            objects.truncate(max_keys);
        }

        common_prefixes.sort();
        common_prefixes.dedup();

        Ok(ListObjectsResult {
            objects,
            common_prefixes,
            is_truncated,
            next_continuation_token: None,
        })
    }

    async fn copy_object(&self, source_key: &str, dest_key: &str) -> ObjectStorageResult<()> {
        let source_path = self.get_object_path(source_key);
        let dest_path = self.get_object_path(dest_key);

        self.ensure_dir_exists(&dest_path).await?;
        fs::copy(&source_path, &dest_path)
            .await
            .map_err(|e| ObjectStorageError::IoError {
                message: e.to_string(),
            })?;

        Ok(())
    }

    async fn put_objects(
        &self,
        objects: Vec<(String, Vec<u8>)>,
    ) -> ObjectStorageResult<Vec<String>> {
        let mut uploaded = Vec::new();

        for (key, data) in objects {
            if self.put_object(&key, Bytes::from(data), None).await.is_ok() {
                uploaded.push(key);
            }
        }

        Ok(uploaded)
    }

    async fn get_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<(String, Vec<u8>)>> {
        let mut results = Vec::new();

        for key in keys {
            if let Ok(data) = self.get_object(key).await {
                results.push((key.clone(), data));
            }
        }

        Ok(results)
    }

    async fn put_stream(
        &self,
        key: &str,
        mut stream: BoxStream<'static, Result<Bytes, std::io::Error>>,
        _content_length: Option<u64>,
        _content_type: Option<&str>,
    ) -> ObjectStorageResult<()> {
        let path = self.get_object_path(key);
        self.ensure_dir_exists(&path).await?;

        let mut file = fs::File::create(&path)
            .await
            .map_err(|e| ObjectStorageError::IoError {
                message: e.to_string(),
            })?;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ObjectStorageError::IoError {
                message: e.to_string(),
            })?;
            file.write_all(&chunk)
                .await
                .map_err(|e| ObjectStorageError::IoError {
                    message: e.to_string(),
                })?;
        }

        file.flush().await?;
        Ok(())
    }

    async fn get_stream(
        &self,
        key: &str,
    ) -> ObjectStorageResult<BoxStream<'static, Result<Bytes, ObjectStorageError>>> {
        let data = self.get_object(key).await?;
        let stream = stream::iter(vec![Bytes::from(data)]);
        Ok(Box::pin(stream.map(Ok)))
    }

    async fn create_multipart_upload(
        &self,
        _key: &str,
        _content_type: Option<&str>,
    ) -> ObjectStorageResult<String> {
        // 为简化实现，生成一个简单的上传ID
        use uuid::Uuid;
        Ok(Uuid::new_v4().to_string())
    }

    async fn upload_part(
        &self,
        _key: &str,
        upload_id: &str,
        part_number: u32,
        data: &[u8],
    ) -> ObjectStorageResult<PartInfo> {
        // 将分片存储为临时文件
        let part_path = self
            .root_path
            .join(&self.bucket)
            .join("_multipart")
            .join(upload_id)
            .join(format!("part-{}", part_number));

        self.ensure_dir_exists(&part_path).await.map_err(|_| {
            ObjectStorageError::InternalError {
                message: "Failed to create directory".to_string(),
            }
        })?;
        fs::write(&part_path, data)
            .await
            .map_err(|_| ObjectStorageError::InternalError {
                message: "Failed to write part file".to_string(),
            })?;

        Ok(PartInfo {
            part_number,
            etag: format!("{:x}", md5::compute(data)),
            size: data.len() as u64,
        })
    }

    async fn complete_multipart_upload(
        &self,
        key: &str,
        upload_id: &str,
        parts: &[PartInfo],
    ) -> ObjectStorageResult<()> {
        let final_path = self.get_object_path(key);
        self.ensure_dir_exists(&final_path).await.map_err(|_| {
            ObjectStorageError::InternalError {
                message: "Failed to create directory".to_string(),
            }
        })?;

        let mut final_file =
            fs::File::create(&final_path)
                .await
                .map_err(|_| ObjectStorageError::InternalError {
                    message: "Failed to create file".to_string(),
                })?;

        // 按分片号排序
        let mut sorted_parts = parts.to_vec();
        sorted_parts.sort_by_key(|p| p.part_number);

        // 合并所有分片
        for part in sorted_parts {
            let part_path = self
                .root_path
                .join(&self.bucket)
                .join("_multipart")
                .join(upload_id)
                .join(format!("part-{}", part.part_number));

            if part_path.exists() {
                let part_data =
                    fs::read(&part_path)
                        .await
                        .map_err(|_| ObjectStorageError::InternalError {
                            message: "Failed to read part file".to_string(),
                        })?;
                final_file.write_all(&part_data).await.map_err(|_| {
                    ObjectStorageError::InternalError {
                        message: "Failed to write to final file".to_string(),
                    }
                })?;
            }
        }

        final_file.flush().await?;

        // 清理临时文件
        let multipart_dir = self
            .root_path
            .join(&self.bucket)
            .join("_multipart")
            .join(upload_id);

        if multipart_dir.exists() {
            let _ = fs::remove_dir_all(&multipart_dir).await;
        }
        Ok(())
    }

    async fn abort_multipart_upload(&self, _key: &str, upload_id: &str) -> ObjectStorageResult<()> {
        // 清理临时文件
        let multipart_dir = self
            .root_path
            .join(&self.bucket)
            .join("_multipart")
            .join(upload_id);

        if multipart_dir.exists() {
            fs::remove_dir_all(&multipart_dir)
                .await
                .map_err(|e| ObjectStorageError::IoError {
                    message: e.to_string(),
                })?;
        }
        Ok(())
    }

    async fn get_storage_stats(&self) -> ObjectStorageResult<StorageStats> {
        let bucket_path = self.root_path.join(&self.bucket);
        let mut object_count = 0u64;
        let mut total_size = 0u64;

        if bucket_path.exists() {
            self.calculate_stats(&bucket_path, &mut object_count, &mut total_size)
                .await?;
        }
        Ok(StorageStats {
            bucket: self.bucket.clone(),
            object_count,
            total_size,
            last_access: Some(Utc::now()),
        })
    }

    async fn health_check(&self) -> ObjectStorageResult<()> {
        let bucket_path = self.root_path.join(&self.bucket);

        // 确保存储桶目录存在
        fs::create_dir_all(&bucket_path)
            .await
            .map_err(|e| ObjectStorageError::IoError {
                message: e.to_string(),
            })?;

        // 尝试写入一个测试文件
        let test_path = bucket_path.join(".health_check");
        fs::write(&test_path, b"ok")
            .await
            .map_err(|e| ObjectStorageError::IoError {
                message: e.to_string(),
            })?;
        fs::remove_file(&test_path)
            .await
            .map_err(|e| ObjectStorageError::IoError {
                message: e.to_string(),
            })?;

        Ok(())
    }
}

impl FilesystemStorage {
    /// 递归收集对象信息
    async fn collect_objects(
        &self,
        dir: &Path,
        prefix: &str,
        delimiter: Option<&str>,
        objects: &mut Vec<ObjectInfo>,
        common_prefixes: &mut Vec<String>,
    ) -> ObjectStorageResult<()> {
        let mut entries = match fs::read_dir(dir).await {
            Ok(entries) => entries,
            Err(e) => {
                return Err(ObjectStorageError::InternalError {
                    message: format!("Read dir failed: {:?}", e),
                })
            }
        };

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let relative_path = path
                .strip_prefix(self.root_path.join(&self.bucket))
                .map_err(|_| ObjectStorageError::InternalError {
                    message: "Operation failed".to_string(),
                })?
                .to_string_lossy()
                .replace('\\', "/");

            if !relative_path.starts_with(prefix) {
                continue;
            }
            if path.is_file() {
                let metadata =
                    entry
                        .metadata()
                        .await
                        .map_err(|_| ObjectStorageError::InternalError {
                            message: "Failed to get metadata".to_string(),
                        })?;
                let last_modified = metadata
                    .modified()
                    .map(DateTime::<Utc>::from)
                    .unwrap_or_else(|_| Utc::now());

                objects.push(ObjectInfo {
                    key: relative_path,
                    size: metadata.len(),
                    last_modified,
                    etag: Some(self.calculate_etag(&metadata)),
                })
            } else if path.is_dir() {
                if let Some(_delim) = delimiter {
                    let dir_prefix = format!("{}/", relative_path);
                    if dir_prefix.len() > prefix.len() {
                        common_prefixes.push(dir_prefix);
                    }
                } else {
                    // 递归遍历子目录
                    match Box::pin(self.collect_objects(
                        &path,
                        prefix,
                        delimiter,
                        objects,
                        common_prefixes,
                    ))
                    .await
                    {
                        Ok(_) => {}
                        Err(e) => return Err(e),
                    };
                }
            }
        }

        Ok(())
    }

    /// 计算存储统计信息
    async fn calculate_stats(
        &self,
        dir: &Path,
        object_count: &mut u64,
        total_size: &mut u64,
    ) -> ObjectStorageResult<()> {
        let mut entries = match fs::read_dir(dir).await {
            Ok(entries) => entries,
            Err(e) => {
                return Err(ObjectStorageError::InternalError {
                    message: format!("Read dir failed: {:?}", e),
                })
            }
        };

        while let Some(entry) = match entries.next_entry().await {
            Ok(Some(entry)) => Some(entry),
            Ok(None) => None,
            Err(e) => {
                return Err(ObjectStorageError::InternalError {
                    message: format!("Entry read failed: {:?}", e),
                })
            }
        } {
            let path = entry.path();

            if path.is_file() {
                let metadata = match entry.metadata().await {
                    Ok(meta) => meta,
                    Err(e) => {
                        return Err(ObjectStorageError::InternalError {
                            message: format!("Metadata read failed: {:?}", e),
                        })
                    }
                };
                *object_count += 1;
                *total_size += metadata.len();
            } else if path.is_dir() {
                if let Some(file_name) = path.file_name() {
                    if !file_name.to_string_lossy().starts_with('_') {
                        // 递归计算子目录，但跳过以_开头的目录（如_multipart）
                        match Box::pin(self.calculate_stats(&path, object_count, total_size)).await
                        {
                            Ok(_) => {}
                            Err(e) => return Err(e),
                        };
                    }
                }
            }
        }

        Ok(())
    }
}
