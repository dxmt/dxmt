use async_trait::async_trait;
use aws_config::Region;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client;
use bytes::Bytes;
use chrono::Utc;
use futures::stream::BoxStream;
use std::collections::HashMap;

use super::errors::ObjectStorageError;
use super::traits::{ObjectStorage, ObjectStorageResult, PutCondition, StorageStats};
use super::types::{
    ListObjectsOptions, ListObjectsResult, ObjectInfo, ObjectMetadata, ObjectStorageConfig,
    PartInfo,
};

/// AWS S3 存储实现（简化版本）
pub struct S3Storage {
    client: Client,
    bucket: String,
    prefix: Option<String>,
}

impl S3Storage {
    /// 创建新的 S3 存储实例
    pub async fn new(config: ObjectStorageConfig) -> ObjectStorageResult<Self> {
        let region_name = config.region.unwrap_or_else(|| "us-east-1".to_string());
        let region = Region::new(region_name);

        // Configure AWS SDK with Tokio sleep implementation for timeouts/retries
        let aws_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(region)
            .sleep_impl(aws_smithy_async::rt::sleep::TokioSleep::new())
            .load()
            .await;

        let mut client_config = aws_sdk_s3::config::Builder::from(&aws_config);

        // The configured keys, when given, and nothing else. They were read into the config and
        // dropped here, so the SDK's default chain answered instead -- `~/.aws/credentials` or
        // the instance role -- and a MinIO deployment was signed with the machine's AWS key.
        match (&config.access_key_id, &config.secret_access_key) {
            (Some(id), Some(secret)) => {
                client_config =
                    client_config.credentials_provider(aws_sdk_s3::config::Credentials::new(
                        id.clone(),
                        secret.clone(),
                        config.session_token.clone(),
                        None,
                        "dxmt-config",
                    ));
            }
            (None, None) => {}
            _ => {
                return Err(ObjectStorageError::InvalidConfiguration(
                    "set both DXMT_AWS_ACCESS_KEY_ID and DXMT_AWS_SECRET_ACCESS_KEY, or neither \
                     (to use the AWS SDK's default credential chain)"
                        .to_string(),
                ));
            }
        }

        // A custom endpoint (MinIO and other S3-compatible services) is addressed by path:
        // `bucket.127.0.0.1` does not resolve.
        if let Some(endpoint) = &config.endpoint {
            client_config = client_config.endpoint_url(endpoint).force_path_style(true);
        }
        let client = Client::from_conf(client_config.build());

        Ok(Self {
            client,
            bucket: config.bucket,
            prefix: config.prefix,
        })
    }

    /// 构建完整的对象键（包含前缀）
    fn build_key(&self, key: &str) -> String {
        match &self.prefix {
            Some(prefix) => format!("{}/{}", prefix, key),
            None => key.to_string(),
        }
    }
}

#[async_trait]
impl ObjectStorage for S3Storage {
    async fn put_object(
        &self,
        key: &str,
        data: Bytes,
        content_type: Option<&str>,
    ) -> ObjectStorageResult<()> {
        let s3_key = self.build_key(key);
        let body = ByteStream::from(data.as_ref().to_vec());

        let mut request = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(&s3_key)
            .body(body);

        if let Some(ct) = content_type {
            request = request.content_type(ct);
        }
        request
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
            })?;

        Ok(())
    }

    async fn get_object_versioned(
        &self,
        key: &str,
    ) -> ObjectStorageResult<(Vec<u8>, Option<String>)> {
        let response = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(self.build_key(key))
            .send()
            .await
            .map_err(|e| {
                let message = aws_sdk_s3::error::DisplayErrorContext(&e).to_string();
                if message.contains("NoSuchKey") {
                    ObjectStorageError::ObjectNotFound {
                        key: key.to_string(),
                    }
                } else {
                    ObjectStorageError::AwsError { message }
                }
            })?;
        let version = response.e_tag().map(str::to_string);
        let data = response
            .body
            .collect()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
            })?;
        Ok((data.into_bytes().to_vec(), version))
    }

    async fn put_object_if(
        &self,
        key: &str,
        data: Bytes,
        content_type: Option<&str>,
        condition: PutCondition,
    ) -> ObjectStorageResult<String> {
        let mut request = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(self.build_key(key))
            .body(ByteStream::from(data.to_vec()));
        if let Some(ct) = content_type {
            request = request.content_type(ct);
        }
        // GCS's S3 API accepts these and ignores them (measured: every form succeeded against
        // a version the object did not have), which `probe_conditional_writes` detects.
        let sent = match condition {
            PutCondition::IfMatch(version) => request.if_match(version),
            PutCondition::IfAbsent => request.if_none_match("*"),
        }
        .send()
        .await;
        let response = sent.map_err(|e| {
            // 412 is the condition failing; S3 answers 409 to a conditional write racing
            // another one on the same key, which for this caller means the same thing.
            let status = e.raw_response().map(|r| r.status().as_u16());
            if matches!(status, Some(412) | Some(409)) {
                ObjectStorageError::PreconditionFailed {
                    key: key.to_string(),
                }
            } else {
                ObjectStorageError::AwsError {
                    message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
                }
            }
        })?;
        response
            .e_tag()
            .map(str::to_string)
            .ok_or_else(|| ObjectStorageError::AwsError {
                message: format!("{key}: a conditional write answered without an ETag"),
            })
    }

    async fn get_object(&self, key: &str) -> ObjectStorageResult<Vec<u8>> {
        let s3_key = self.build_key(key);

        let response = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&s3_key)
            .send()
            .await
            .map_err(|e| {
                if aws_sdk_s3::error::DisplayErrorContext(&e)
                    .to_string()
                    .contains("NoSuchKey")
                {
                    ObjectStorageError::ObjectNotFound {
                        key: key.to_string(),
                    }
                } else {
                    ObjectStorageError::AwsError {
                        message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
                    }
                }
            })?;

        let data = response
            .body
            .collect()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
            })?;

        Ok(data.into_bytes().to_vec())
    }

    async fn get_object_range(
        &self,
        key: &str,
        start: u64,
        end: u64,
    ) -> ObjectStorageResult<Vec<u8>> {
        let s3_key = self.build_key(key);
        let range_header = format!("bytes={}-{}", start, end - 1);

        let response = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&s3_key)
            .range(range_header)
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
            })?;

        let data = response
            .body
            .collect()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
            })?;

        Ok(data.into_bytes().to_vec())
    }

    async fn delete_object(&self, key: &str) -> ObjectStorageResult<()> {
        let s3_key = self.build_key(key);

        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(&s3_key)
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
            })?;

        Ok(())
    }

    async fn delete_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<String>> {
        let mut deleted = Vec::new();

        // 简化实现：逐个删除
        for key in keys {
            if self.delete_object(key).await.is_ok() {
                deleted.push(key.clone())
            }
        }

        Ok(deleted)
    }

    async fn object_exists(&self, key: &str) -> ObjectStorageResult<bool> {
        let s3_key = self.build_key(key);

        match self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(&s3_key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(_) => Ok(false), // 简化：任何错误都视为不存在
        }
    }

    async fn head_object(&self, key: &str) -> ObjectStorageResult<ObjectMetadata> {
        let s3_key = self.build_key(key);

        let response = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(&s3_key)
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
            })?;

        let last_modified = response
            .last_modified
            .map(|_dt| Utc::now()) // 简化：直接使用当前时间
            .unwrap_or_else(Utc::now);

        let mut metadata = HashMap::new();
        if let Some(meta) = response.metadata {
            for (k, v) in meta {
                metadata.insert(k, v);
            }
        }

        Ok(ObjectMetadata {
            key: key.to_string(),
            size: response.content_length.unwrap_or(0) as u64,
            last_modified,
            etag: response.e_tag,
            content_type: response.content_type,
            payload: metadata,
        })
    }

    async fn list_objects(
        &self,
        options: &ListObjectsOptions,
    ) -> ObjectStorageResult<ListObjectsResult> {
        let prefix = match &self.prefix {
            Some(base_prefix) => {
                if let Some(opt_prefix) = &options.prefix {
                    format!("{}/{}", base_prefix, opt_prefix)
                } else {
                    format!("{}/", base_prefix)
                }
            }
            None => options.prefix.clone().unwrap_or_default(),
        };

        let mut request = self
            .client
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(&prefix);

        if let Some(max_keys) = options.max_keys {
            request = request.max_keys(max_keys);
        }
        if let Some(token) = &options.continuation_token {
            request = request.continuation_token(token);
        }
        if let Some(delimiter) = &options.delimiter {
            request = request.delimiter(delimiter);
        }
        let response = request
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
            })?;

        let mut objects = Vec::new();
        if let Some(contents) = response.contents {
            for obj in contents {
                if let Some(obj_key) = obj.key {
                    // 移除前缀
                    let original_key = if let Some(base_prefix) = &self.prefix {
                        obj_key
                            .strip_prefix(&format!("{}/", base_prefix))
                            .unwrap_or(&obj_key)
                            .to_string()
                    } else {
                        obj_key
                    };

                    let last_modified = obj
                        .last_modified
                        .map(|_dt| Utc::now()) // 简化：直接使用当前时间
                        .unwrap_or_else(Utc::now);

                    objects.push(ObjectInfo {
                        key: original_key,
                        size: obj.size.unwrap_or(0) as u64,
                        last_modified,
                        etag: obj.e_tag,
                    });
                }
            }
        }

        let mut common_prefixes = Vec::new();
        if let Some(prefixes) = response.common_prefixes {
            for prefix_obj in prefixes {
                if let Some(prefix_key) = prefix_obj.prefix {
                    // 移除基础前缀
                    let clean_prefix = if let Some(base_prefix) = &self.prefix {
                        prefix_key
                            .strip_prefix(&format!("{}/", base_prefix))
                            .unwrap_or(&prefix_key)
                            .to_string()
                    } else {
                        prefix_key
                    };
                    common_prefixes.push(clean_prefix);
                }
            }
        }

        Ok(ListObjectsResult {
            objects,
            common_prefixes,
            is_truncated: response.is_truncated.unwrap_or(false),
            next_continuation_token: response.next_continuation_token,
        })
    }

    async fn copy_object(&self, source_key: &str, dest_key: &str) -> ObjectStorageResult<()> {
        let source_s3_key = self.build_key(source_key);
        let dest_s3_key = self.build_key(dest_key);
        let copy_source = format!("{}/{}", self.bucket, source_s3_key);

        self.client
            .copy_object()
            .bucket(&self.bucket)
            .key(&dest_s3_key)
            .copy_source(&copy_source)
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
            })?;

        Ok(())
    }

    async fn put_objects(
        &self,
        objects: Vec<(String, Vec<u8>)>,
    ) -> ObjectStorageResult<Vec<String>> {
        let mut uploaded = Vec::new();

        // S3 没有原生的批量上传API，只能逐个上传
        for (key, data) in objects {
            if self.put_object(&key, Bytes::from(data), None).await.is_ok() {
                uploaded.push(key);
            }
        }

        Ok(uploaded)
    }

    async fn get_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<(String, Vec<u8>)>> {
        let mut results = Vec::new();

        // S3 没有原生的批量下载API，只能逐个下载
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
        content_type: Option<&str>,
    ) -> ObjectStorageResult<()> {
        // 简化实现：收集所有数据然后上传
        use futures::StreamExt;

        let mut data = Vec::new();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ObjectStorageError::IoError {
                message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
            })?;
            data.extend_from_slice(&chunk);
        }

        self.put_object(key, Bytes::from(data), content_type).await
    }

    async fn get_stream(
        &self,
        key: &str,
    ) -> ObjectStorageResult<BoxStream<'static, Result<Bytes, ObjectStorageError>>> {
        // 简化实现：下载全部数据然后转为流
        let data = self.get_object(key).await?;
        use futures::stream::StreamExt;
        let stream = futures::stream::iter(vec![Bytes::from(data)]);
        Ok(Box::pin(stream.map(Ok)))
    }

    async fn create_multipart_upload(
        &self,
        _key: &str,
        _content_type: Option<&str>,
    ) -> ObjectStorageResult<String> {
        // 简化实现：返回一个假的上传ID
        Ok("simplified-upload-id".to_string())
    }

    async fn upload_part(
        &self,
        _key: &str,
        _upload_id: &str,
        part_number: u32,
        data: &[u8],
    ) -> ObjectStorageResult<PartInfo> {
        // 简化实现：返回基础的分片信息
        Ok(PartInfo {
            part_number,
            etag: format!("{:x}", md5::compute(data)),
            size: data.len() as u64,
        })
    }

    async fn complete_multipart_upload(
        &self,
        _key: &str,
        _upload_id: &str,
        _parts: &[PartInfo],
    ) -> ObjectStorageResult<()> {
        // 简化实现：直接成功
        Ok(())
    }

    async fn abort_multipart_upload(
        &self,
        _key: &str,
        _upload_id: &str,
    ) -> ObjectStorageResult<()> {
        // 简化实现：直接成功
        Ok(())
    }

    async fn get_storage_stats(&self) -> ObjectStorageResult<StorageStats> {
        // S3 不提供直接的统计API，这里返回基础信息
        Ok(StorageStats {
            bucket: self.bucket.clone(),
            object_count: 0, // 需要通过列表操作计算
            total_size: 0,   // 需要通过列表操作计算
            last_access: Some(Utc::now()),
        })
    }

    async fn health_check(&self) -> ObjectStorageResult<()> {
        // 尝试列出存储桶来检查连接性
        self.client
            .list_objects_v2()
            .bucket(&self.bucket)
            .max_keys(1)
            .send()
            .await
            .map_err(|e| {
                if aws_sdk_s3::error::DisplayErrorContext(&e)
                    .to_string()
                    .contains("NoSuchBucket")
                {
                    ObjectStorageError::BucketNotFound {
                        bucket: self.bucket.clone(),
                    }
                } else {
                    ObjectStorageError::AwsError {
                        message: aws_sdk_s3::error::DisplayErrorContext(&e).to_string(),
                    }
                }
            })?;

        Ok(())
    }
}
