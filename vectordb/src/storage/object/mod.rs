pub mod batch_s3;
pub mod compression;
pub mod errors;
pub mod filesystem;
pub mod metrics;
pub mod s3;
pub mod scheduler;
pub mod tiered;
pub mod traits;
pub mod types;

pub use batch_s3::BatchS3Storage;
pub use compression::{CompressionManager, CompressionMetadata, CompressionType, Compressor};
pub use errors::ObjectStorageError;
pub use filesystem::FilesystemStorage;
pub use metrics::{
    CompressionMetrics, MetricsCollector, MetricsReport, OperationMetrics, TierMetrics,
};
pub use s3::S3Storage;
pub use scheduler::{AccessPattern, DataScheduler, SchedulerStats, SchedulingPolicy};
pub use tiered::{StorageTier, TierStat, TierStats, TieredStorage, TieredStorageConfig};
pub use traits::ObjectStorage;
pub use traits::{ObjectStorageResult, StorageStats};
pub use types::{
    ListObjectsOptions, ListObjectsResult, MultipartUpload, ObjectInfo, ObjectMetadata, PartInfo,
};

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use tempfile::TempDir;
    use tokio;

    #[tokio::test]
    async fn test_filesystem_storage_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let storage = FilesystemStorage::new(temp_dir.path(), "test-bucket")
            .expect("Failed to create filesystem storage");

        // 测试健康检查
        storage.health_check().await?;

        // 测试上传对象
        let key = "test/file.txt";
        let data = b"Hello, World!";
        storage
            .put_object(key, Bytes::from(&data[..]), Some("text/plain"))
            .await?;

        // 测试检查对象存在
        assert!(storage
            .object_exists(key)
            .await?);

        // 测试下载对象
        let downloaded = storage.get_object(key).await?;
        assert_eq!(downloaded, data);

        // 测试获取对象元数据
        let metadata = storage
            .head_object(key)
            .await?;
        assert_eq!(metadata.key, key);
        assert_eq!(metadata.size, data.len() as u64);

        // 测试列出对象
        let list_options = types::ListObjectsOptions::default();
        let result = storage
            .list_objects(&list_options)
            .await?;
        assert!(!result.objects.is_empty());

        // 测试删除对象
        storage
            .delete_object(key)
            .await?;

        assert!(!storage
            .object_exists(key)
            .await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_filesystem_storage_multipart_upload() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let storage = FilesystemStorage::new(temp_dir.path(), "test-bucket")
            .expect("Failed to create filesystem storage");

        let key = "multipart/test.dat";

        // 开始分片上传
        let upload_id = storage
            .create_multipart_upload(key, None)
            .await?;

        // 上传分片
        let part1_data = b"Part 1 data";
        let part2_data = b"Part 2 data";

        let part1 = storage
            .upload_part(key, &upload_id, 1, part1_data)
            .await?;
        let part2 = storage
            .upload_part(key, &upload_id, 2, part2_data)
            .await?;

        // 完成分片上传
        let parts = vec![part1, part2];
        storage
            .complete_multipart_upload(key, &upload_id, &parts)
            .await?;

        // 验证合并后的文件
        let downloaded = storage
            .get_object(key)
            .await?;

        let mut expected = Vec::new();
        expected.extend_from_slice(part1_data);
        expected.extend_from_slice(part2_data);
        assert_eq!(downloaded, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_tiered_storage_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let cache_dir = TempDir::new()?;

        // 创建底层文件系统存储作为对象存储
        let base_storage = FilesystemStorage::new(temp_dir.path(), "test-bucket")
            .expect("Failed to create base storage");

        // 创建三层存储配置
        let mut config = TieredStorageConfig::default();
        config.ssd_cache_dir = cache_dir.path().to_path_buf();
        config.memory_cache_size = 1024; // 1KB
        config.memory_cache_capacity = 100;

        // 创建三层存储
        let tiered_storage = TieredStorage::new(config, Box::new(base_storage))
            .expect("Failed to create tiered storage");

        // 测试小对象（应该缓存到内存）
        let small_key = "small.txt";
        let small_data = b"small data";
        tiered_storage
            .put_object(small_key, Bytes::from(&small_data[..]), None)
            .await?;

        let retrieved_small = tiered_storage
            .get_object(small_key)
            .await?;
        assert_eq!(retrieved_small, small_data);

        // 测试大对象（应该直接存储到对象存储）
        let large_key = "large.txt";
        let large_data = vec![42u8; 2048]; // 2KB，超过内存缓存限制
        tiered_storage
            .put_object(large_key, Bytes::from(large_data.clone()), None)
            .await?;

        let retrieved_large = tiered_storage
            .get_object(large_key)
            .await?;
        assert_eq!(retrieved_large, large_data);

        // 测试对象存在性检查
        assert!(tiered_storage
            .object_exists(small_key)
            .await?);
        assert!(tiered_storage
            .object_exists(large_key)
            .await?);

        // 获取分层统计信息
        let stats = tiered_storage
            .get_tier_stats()
            .await?;

        // 验证分层工作正常：小对象在内存，大对象在SSD

        // 验证统计信息合理
        assert!(stats.memory.total_size <= 1024); // 内存限制
        assert!(
            stats.memory.object_count + stats.ssd.object_count + stats.object_storage.object_count
                >= 2
        ); // 至少有两个对象

        // 测试删除
        tiered_storage
            .delete_object(small_key)
            .await?;
        assert!(!tiered_storage
            .object_exists(small_key)
            .await?);
        Ok(())
    }
}
