use async_trait::async_trait;
use aws_sdk_s3::config::{Credentials, Region};
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::{Delete, ObjectIdentifier};
use aws_sdk_s3::{Client, Config};
use bytes::Bytes;
use futures::future::join_all;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

use crate::storage::object::{
    traits::{ObjectStorage, ObjectStorageResult, StorageStats},
    types::{ListObjectsOptions, ListObjectsResult, ObjectMetadata, PartInfo},
    ObjectInfo, ObjectStorageError,
};
use futures::stream::BoxStream;

/// Enhanced S3 storage with batch operations and parallelization
pub struct BatchS3Storage {
    client: Client,
    bucket: String,
    prefix: String,
    /// Semaphore to limit concurrent operations
    concurrency_limiter: Arc<Semaphore>,
    /// Maximum number of concurrent operations
    max_concurrent_ops: usize,
    /// Batch size for operations
    batchsize: usize,
    /// Prefetch cache
    prefetch_cache: Arc<RwLock<lru::LruCache<String, Bytes>>>,
}

impl BatchS3Storage {
    // Nine S3 connection knobs, each independently optional. Bundling them into a config
    // struct is worth doing, but it is a change to every caller, not a lint fix.
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        bucket: String,
        prefix: String,
        region: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        endpoint: Option<String>,
        max_concurrent_ops: Option<usize>,
        batchsize: Option<usize>,
        cache_size: Option<usize>,
    ) -> ObjectStorageResult<Self> {
        let region = Region::new(region.unwrap_or_else(|| "us-east-1".to_string()));

        let mut config_builder = Config::builder().region(region).behavior_version_latest();

        if let (Some(access_key), Some(secret_key)) = (access_key_id, secret_access_key) {
            let creds = Credentials::new(access_key, secret_key, None, None, "batch_s3");
            config_builder = config_builder.credentials_provider(creds);
        }
        if let Some(endpoint_url) = endpoint {
            config_builder = config_builder
                .endpoint_url(endpoint_url)
                .force_path_style(true);
        }
        let config = config_builder.build();
        let client = Client::from_conf(config);

        let max_concurrent_ops = max_concurrent_ops.unwrap_or(100);
        let batch_size = batchsize.unwrap_or(1000);
        let cache_size = cache_size.unwrap_or(10000);

        Ok(Self {
            client,
            bucket,
            prefix,
            concurrency_limiter: Arc::new(Semaphore::new(max_concurrent_ops)),
            max_concurrent_ops,
            batchsize: batch_size,
            prefetch_cache: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(cache_size).ok_or_else(|| {
                    ObjectStorageError::InvalidConfiguration(
                        "Cache size cannot be zero".to_string(),
                    )
                })?,
            ))),
        })
    }

    fn full_key(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{}", self.prefix, key)
        }
    }

    /// Batch get objects with parallel fetching
    pub async fn batch_get_objects(
        &self,
        keys: &[String],
    ) -> ObjectStorageResult<Vec<(String, Bytes)>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(keys.len());
        let mut cache_hits = Vec::new();
        let mut keys_to_fetch = Vec::new();

        // Check cache first
        {
            let mut cache = self.prefetch_cache.write().await;
            for key in keys {
                if let Some(data) = cache.get(key) {
                    cache_hits.push((key.clone(), data.clone()))
                } else {
                    keys_to_fetch.push(key.clone())
                }
            }
        }

        results.extend(cache_hits);

        // Fetch missing keys in parallel batches
        if !keys_to_fetch.is_empty() {
            let fetched = match self.parallel_fetch(&keys_to_fetch).await {
                Ok(data) => data,
                Err(e) => {
                    return Err(ObjectStorageError::InternalError {
                        message: format!("Parallel fetch failed: {:?}", e),
                    })
                }
            };

            // Update cache
            {
                let mut cache = self.prefetch_cache.write().await;
                for (key, data) in &fetched {
                    cache.put(key.clone(), data.clone());
                }
            }

            results.extend(fetched);
        }

        Ok(results)
    }

    /// Parallel fetch implementation
    async fn parallel_fetch(&self, keys: &[String]) -> ObjectStorageResult<Vec<(String, Bytes)>> {
        let _chunks: Vec<_> = keys.chunks(self.batchsize).collect();
        let mut all_results = Vec::new();

        for chunk in keys.chunks(self.batchsize) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|key| {
                    let client = self.client.clone();
                    let bucket = self.bucket.clone();
                    let full_key = self.full_key(key);
                    let key_clone = key.clone();
                    let permit = self.concurrency_limiter.clone().acquire_owned();

                    async move {
                        let _permit = match permit.await {
                            Ok(p) => p,
                            Err(_) => return None,
                        };

                        match client
                            .get_object()
                            .bucket(&bucket)
                            .key(&full_key)
                            .send()
                            .await
                        {
                            Ok(output) => match output.body.collect().await {
                                Ok(data) => Some((key_clone, data.into_bytes())),
                                Err(_) => None,
                            },
                            Err(_) => None,
                        }
                    }
                })
                .collect();

            let chunk_results: Vec<_> = join_all(futures).await.into_iter().flatten().collect();

            all_results.extend(chunk_results);
        }

        Ok(all_results)
    }

    /// Batch put objects with parallel uploading
    pub async fn batch_put_objects(
        &self,
        objects: Vec<(String, Bytes)>,
    ) -> ObjectStorageResult<Vec<String>> {
        if objects.is_empty() {
            return Ok(Vec::new());
        }

        let chunks: Vec<_> = objects.chunks(self.batchsize).collect();
        let mut all_uploaded = Vec::new();

        for chunk in chunks {
            let futures: Vec<_> = chunk
                .iter()
                .map(|(key, data)| {
                    let client = self.client.clone();
                    let bucket = self.bucket.clone();
                    let full_key = self.full_key(key);
                    let key_clone = key.clone();
                    let data_clone = data.clone();
                    let permit = self.concurrency_limiter.clone().acquire_owned();

                    async move {
                        let _permit = match permit.await {
                            Ok(p) => p,
                            Err(_) => return None,
                        };

                        match client
                            .put_object()
                            .bucket(&bucket)
                            .key(&full_key)
                            .body(data_clone.into())
                            .send()
                            .await
                        {
                            Ok(_) => Some(key_clone),
                            Err(_) => None,
                        }
                    }
                })
                .collect();

            let chunk_results: Vec<_> = join_all(futures).await.into_iter().flatten().collect();

            all_uploaded.extend(chunk_results);
        }

        // Update cache with newly uploaded objects
        let mut cache = self.prefetch_cache.write().await;
        for (key, data) in &objects {
            if all_uploaded.contains(key) {
                cache.put(key.clone(), data.clone());
            }
        }

        Ok(all_uploaded)
    }

    /// Batch delete objects
    pub async fn batch_delete_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<String>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_deleted = Vec::new();

        // S3 allows up to 1000 objects per delete request
        for chunk in keys.chunks(1000) {
            let objects: Vec<ObjectIdentifier> = chunk
                .iter()
                .filter_map(|key| {
                    ObjectIdentifier::builder()
                        .key(self.full_key(key))
                        .build()
                        .ok()
                })
                .collect();

            let delete = Delete::builder()
                .set_objects(Some(objects))
                .build()
                .expect("Failed to build delete request");

            match self
                .client
                .delete_objects()
                .bucket(&self.bucket)
                .delete(delete)
                .send()
                .await
            {
                Ok(output) => {
                    if let Some(deleted) = output.deleted {
                        for obj in deleted {
                            if let Some(key) = obj.key {
                                // Remove prefix to get original key
                                let original_key = if self.prefix.is_empty() {
                                    key
                                } else {
                                    key.strip_prefix(&format!("{}/", self.prefix))
                                        .unwrap_or(&key)
                                        .to_string()
                                };
                                all_deleted.push(original_key);
                            }
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        // Remove deleted objects from cache
        let mut cache = self.prefetch_cache.write().await;
        for key in &all_deleted {
            cache.pop(key);
        }

        Ok(all_deleted)
    }

    /// Prefetch objects based on access patterns
    pub async fn prefetch_objects(&self, keys: &[String]) -> ObjectStorageResult<()> {
        // Check which keys are not in cache
        let keys_to_prefetch = {
            let cache = self.prefetch_cache.read().await;
            keys.iter()
                .filter(|key| !cache.contains(*key))
                .cloned()
                .collect::<Vec<_>>()
        };

        if !keys_to_prefetch.is_empty() {
            // Fetch in background
            let self_clone = self.clone();
            tokio::spawn(async move {
                let _ = self_clone.batch_get_objects(&keys_to_prefetch).await;
            });
        }

        Ok(())
    }
}

#[async_trait]
impl ObjectStorage for BatchS3Storage {
    async fn put_object(
        &self,
        key: &str,
        data: Bytes,
        _content_type: Option<&str>,
    ) -> ObjectStorageResult<()> {
        let _full_key = self.full_key(key);

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(data.clone().into())
            .send()
            .await
            .map_err(|e| ObjectStorageError::NetworkError {
                message: format!("S3 put failed: {}", e),
            })?;

        // Update cache
        {
            let mut cache = self.prefetch_cache.write().await;
            cache.put(key.to_string(), data);
        }

        Ok(())
    }

    async fn get_object(&self, key: &str) -> ObjectStorageResult<Vec<u8>> {
        // Check cache first
        {
            let mut cache = self.prefetch_cache.write().await;
            if let Some(data) = cache.get(key) {
                return Ok(data.to_vec());
            }
        }

        let full_key = self.full_key(key);

        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .send()
            .await
            .map_err(|e| ObjectStorageError::NetworkError {
                message: format!("Failed to get object {}: {}", key, e),
            })?;

        let data = output
            .body
            .collect()
            .await
            .map_err(|e| ObjectStorageError::NetworkError {
                message: format!("Failed to get object {}: {}", key, e),
            })?
            .into_bytes();

        // Update cache
        {
            let mut cache = self.prefetch_cache.write().await;
            cache.put(key.to_string(), data.clone());
        }

        Ok(data.to_vec())
    }

    async fn delete_object(&self, key: &str) -> ObjectStorageResult<()> {
        let full_key = self.full_key(key);

        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: format!("Failed to delete object {}: {}", key, e),
            })?;

        // Remove from cache
        {
            let mut cache = self.prefetch_cache.write().await;
            cache.pop(key);
        }

        Ok(())
    }

    async fn list_objects(
        &self,
        options: &ListObjectsOptions,
    ) -> ObjectStorageResult<ListObjectsResult> {
        let prefix = options.prefix.as_deref().unwrap_or("");
        let full_prefix = self.full_key(prefix);

        let mut request = self
            .client
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(&full_prefix);

        if let Some(max_keys) = options.max_keys {
            request = request.max_keys(max_keys);
        }
        // start_after not supported in ListObjectsOptions

        let output = request
            .send()
            .await
            .map_err(|e| ObjectStorageError::InternalError {
                message: format!("Failed to list objects with prefix {}: {}", prefix, e),
            })?;

        let objects = output
            .contents()
            .iter()
            .filter_map(|obj| {
                obj.key.as_ref().map(|key| {
                    let original_key = if self.prefix.is_empty() {
                        key.clone()
                    } else {
                        key.strip_prefix(&format!("{}/", self.prefix))
                            .unwrap_or(key)
                            .to_string()
                    };

                    ObjectInfo {
                        key: original_key,
                        size: obj.size.unwrap_or(0) as u64,
                        last_modified: obj
                            .last_modified
                            .and_then(|t| t.to_millis().ok())
                            .and_then(chrono::DateTime::from_timestamp_millis)
                            .unwrap_or_else(chrono::Utc::now),
                        etag: obj.e_tag.clone(),
                    }
                })
            })
            .collect();

        Ok(ListObjectsResult {
            objects,
            common_prefixes: vec![],
            is_truncated: output.is_truncated().unwrap_or(false),
            next_continuation_token: output.next_continuation_token().map(|s| s.to_string()),
        })
    }

    async fn object_exists(&self, key: &str) -> ObjectStorageResult<bool> {
        let full_key = self.full_key(key);

        let result = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .send()
            .await;
        match result {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn put_objects(
        &self,
        objects: Vec<(String, Vec<u8>)>,
    ) -> ObjectStorageResult<Vec<String>> {
        let objectsbytes: Vec<_> = objects
            .into_iter()
            .map(|(k, v)| (k, Bytes::from(v)))
            .collect();

        Ok(self.batch_put_objects(objectsbytes).await?)
    }

    async fn get_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<(String, Vec<u8>)>> {
        let results = self.batch_get_objects(keys).await?;
        Ok(results.into_iter().map(|(k, v)| (k, v.to_vec())).collect())
    }

    async fn delete_objects(&self, keys: &[String]) -> ObjectStorageResult<Vec<String>> {
        Ok(self.batch_delete_objects(keys).await?)
    }

    async fn create_multipart_upload(
        &self,
        _key: &str,
        _content_type: Option<&str>,
    ) -> ObjectStorageResult<String> {
        Err(ObjectStorageError::InternalError {
            message: format!("Operation not implemented: {}", "create_multipart_upload"),
        })
    }

    async fn get_object_range(
        &self,
        key: &str,
        start: u64,
        end: u64,
    ) -> ObjectStorageResult<Vec<u8>> {
        let full_key = self.full_key(key);
        let range = format!("bytes={}-{}", start, end - 1);

        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .range(range)
            .send()
            .await
            .map_err(|e| ObjectStorageError::NetworkError {
                message: format!("Failed to get object {}: {}", key, e),
            })?;

        let data = output
            .body
            .collect()
            .await
            .map_err(|e| ObjectStorageError::NetworkError {
                message: format!("Failed to get object {}: {}", key, e),
            })?
            .into_bytes();

        Ok(data.to_vec())
    }

    async fn head_object(&self, key: &str) -> ObjectStorageResult<ObjectMetadata> {
        let full_key = self.full_key(key);

        let output = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .send()
            .await
            .map_err(|_e| ObjectStorageError::ObjectNotFound {
                key: full_key.clone(),
            })?;

        Ok(ObjectMetadata {
            key: key.to_string(),
            size: output.content_length().unwrap_or(0) as u64,
            content_type: output.content_type().map(|s| s.to_string()),
            last_modified: output
                .last_modified()
                .and_then(|dt| dt.to_millis().ok())
                .and_then(chrono::DateTime::from_timestamp_millis)
                .unwrap_or_else(chrono::Utc::now),
            etag: output.e_tag().map(|s| s.to_string()),
            payload: HashMap::new(),
        })
    }

    async fn copy_object(&self, source_key: &str, dest_key: &str) -> ObjectStorageResult<()> {
        let source_full = self.full_key(source_key);
        let dest_full = self.full_key(dest_key);
        let copy_source = format!("{}/{}", self.bucket, source_full);

        self.client
            .copy_object()
            .copy_source(copy_source)
            .bucket(&self.bucket)
            .key(&dest_full)
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: format!(
                    "Failed to copy object from {} to {}: {}",
                    source_key, dest_key, e
                ),
            })?;

        Ok(())
    }

    async fn put_stream(
        &self,
        key: &str,
        stream: BoxStream<'static, Result<Bytes, std::io::Error>>,
        content_length: Option<u64>,
        _content_type: Option<&str>,
    ) -> ObjectStorageResult<()> {
        use aws_sdk_s3::primitives::ByteStream;

        let full_key = self.full_key(key);

        // Convert the stream to ByteStream
        // AWS SDK v2 需要特殊处理stream
        // 暂时使用简化实现，收集所有数据
        use futures::StreamExt;
        let mut data = Vec::new();
        let mut stream = stream;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => data.extend_from_slice(&bytes),
                Err(e) => {
                    return Err(ObjectStorageError::IoError {
                        message: format!("Failed to read stream: {}", e),
                    });
                }
            }
        }
        let body = ByteStream::from(data);

        let mut request = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .body(body);

        if let Some(len) = content_length {
            request = request.content_length(len as i64);
        }
        request
            .send()
            .await
            .map_err(|e| ObjectStorageError::NetworkError {
                message: format!("Failed to put object {}: {}", key, e),
            })?;

        Ok(())
    }

    async fn get_stream(
        &self,
        key: &str,
    ) -> ObjectStorageResult<BoxStream<'static, Result<Bytes, ObjectStorageError>>> {
        let full_key = self.full_key(key);

        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .send()
            .await
            .map_err(|e| ObjectStorageError::NetworkError {
                message: format!("Failed to get object {}: {}", key, e),
            })?;

        // Convert ByteStream to our stream type
        use futures::stream::{self, StreamExt};
        let key_owned = key.to_string();
        let stream =
            stream::once(async move {
                let aggregated = output.body.collect().await.map_err(move |e| {
                    ObjectStorageError::NetworkError {
                        message: format!("Failed to get object {}: {}", key_owned, e),
                    }
                })?;
                Ok::<Bytes, ObjectStorageError>(aggregated.into_bytes())
            });

        Ok(Box::pin(stream.map(|result| result)))
    }

    async fn upload_part(
        &self,
        key: &str,
        upload_id: &str,
        part_number: u32,
        data: &[u8],
    ) -> ObjectStorageResult<PartInfo> {
        let full_key = self.full_key(key);

        let output = self
            .client
            .upload_part()
            .bucket(&self.bucket)
            .key(&full_key)
            .upload_id(upload_id)
            .part_number(part_number as i32)
            .body(ByteStream::from(data.to_vec()))
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: format!("Failed to perform operation for {}: {}", key, e),
            })?;

        Ok(PartInfo {
            part_number,
            etag: output.e_tag().unwrap_or_default().to_string(),
            size: data.len() as u64,
        })
    }

    async fn complete_multipart_upload(
        &self,
        key: &str,
        upload_id: &str,
        parts: &[PartInfo],
    ) -> ObjectStorageResult<()> {
        use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};

        let full_key = self.full_key(key);

        let completed_parts: Vec<_> = parts
            .iter()
            .map(|part| {
                CompletedPart::builder()
                    .part_number(part.part_number as i32)
                    .e_tag(&part.etag)
                    .build()
            })
            .collect();

        let multipart_upload = CompletedMultipartUpload::builder()
            .set_parts(Some(completed_parts))
            .build();

        self.client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(&full_key)
            .upload_id(upload_id)
            .multipart_upload(multipart_upload)
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: format!("Failed to perform operation for {}: {}", key, e),
            })?;

        Ok(())
    }

    async fn abort_multipart_upload(&self, key: &str, upload_id: &str) -> ObjectStorageResult<()> {
        let full_key = self.full_key(key);

        self.client
            .abort_multipart_upload()
            .bucket(&self.bucket)
            .key(&full_key)
            .upload_id(upload_id)
            .send()
            .await
            .map_err(|e| ObjectStorageError::AwsError {
                message: format!("Failed to perform operation for {}: {}", key, e),
            })?;

        Ok(())
    }

    async fn get_storage_stats(&self) -> ObjectStorageResult<StorageStats> {
        let mut total_size = 0u64;
        let mut object_count = 0u64;

        let mut continuation_token = None;
        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&self.prefix)
                .max_keys(1000);

            if let Some(token) = continuation_token {
                request = request.continuation_token(token);
            }
            let output = request
                .send()
                .await
                .map_err(|e| ObjectStorageError::AwsError {
                    message: format!("Failed to list objects with prefix {}: {}", self.prefix, e),
                })?;

            let contents = output.contents();
            for obj in contents {
                object_count += 1;
                total_size += obj.size().unwrap_or(0) as u64
            }

            if output.is_truncated().unwrap_or(false) {
                continuation_token = output.next_continuation_token().map(|s| s.to_string())
            } else {
                break;
            }
        }

        Ok(StorageStats {
            bucket: self.bucket.clone(),
            object_count,
            total_size,
            last_access: Some(chrono::Utc::now()),
        })
    }

    async fn health_check(&self) -> ObjectStorageResult<()> {
        self.client
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(&self.prefix)
            .max_keys(1)
            .send()
            .await
            .map_err(|e| ObjectStorageError::NetworkError {
                message: format!("Failed to check bucket {}: {}", self.bucket, e),
            })?;

        Ok(())
    }
}

impl Clone for BatchS3Storage {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            bucket: self.bucket.clone(),
            prefix: self.prefix.clone(),
            concurrency_limiter: self.concurrency_limiter.clone(),
            max_concurrent_ops: self.max_concurrent_ops,
            batchsize: self.batchsize,
            prefetch_cache: self.prefetch_cache.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_operations() -> Result<(), Box<dyn std::error::Error>> {
        // This would require a mock S3 or test container
        // For now, just verify compilation
        let _storage = BatchS3Storage::new(
            "test-bucket".to_string(),
            "test-prefix".to_string(),
            None,
            None,
            None,
            None,
            Some(50),
            Some(100),
            Some(1000),
        )
        .await?;
        Ok(())
    }
}
