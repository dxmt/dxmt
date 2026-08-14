use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use zstd;

use crate::storage::object::traits::ObjectStorage;
use crate::types::table::Table;
use crate::wal::WalManager;

/// 备份恢复管理器
pub struct BackupManager {
    /// 主存储
    primary_storage: Arc<dyn ObjectStorage>,
    /// 备份存储列表
    backup_storages: Vec<Arc<dyn ObjectStorage>>,
    /// 备份配置
    config: BackupConfig,
    /// 备份状态
    backup_state: Arc<RwLock<BackupState>>,
    /// WAL引用
    wal: Arc<WalManager>,
}

/// 备份配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// 全量备份间隔（小时）
    pub full_backup_interval_hours: u64,
    /// 增量备份间隔（分钟）
    pub incremental_backup_interval_minutes: u64,
    /// 备份保留天数
    pub retention_days: u64,
    /// 并行备份线程数
    pub backup_threads: usize,
    /// 启用压缩
    pub enable_compression: bool,
    /// 启用加密
    pub enable_encryption: bool,
    /// 备份验证
    pub enable_verification: bool,
    /// 最大重试次数
    pub max_retries: u32,
    /// 重试间隔（秒）
    pub retry_interval_seconds: u64,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            full_backup_interval_hours: 24,          // 每天全量备份
            incremental_backup_interval_minutes: 30, // 每30分钟增量备份
            retention_days: 30,                      // 保留30天
            backup_threads: 4,                       // 4个并行线程
            enable_compression: true,
            enable_encryption: false, // 默认不加密
            enable_verification: true,
            max_retries: 3,
            retry_interval_seconds: 60,
        }
    }
}

/// 备份状态
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackupState {
    /// 最后全量备份时间
    pub last_full_backup: Option<DateTime<Utc>>,
    /// 最后增量备份时间
    pub last_incremental_backup: Option<DateTime<Utc>>,
    /// 备份任务列表
    pub active_backups: HashMap<Uuid, BackupTask>,
    /// 备份历史记录
    pub backup_history: Vec<BackupRecord>,
}

/// 备份任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupTask {
    /// 任务ID
    pub id: Uuid,
    /// 备份类型
    pub backup_type: BackupType,
    /// 任务状态
    pub status: BackupStatus,
    /// 开始时间
    pub started_at: DateTime<Utc>,
    /// 完成时间
    pub completed_at: Option<DateTime<Utc>>,
    /// 备份的表列表
    pub tables: Vec<String>,
    /// 进度百分比
    pub progress_percentage: f64,
    /// 错误信息
    pub error_message: Option<String>,
}

/// 备份类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BackupType {
    /// 全量备份
    Full,
    /// 增量备份
    Incremental,
    /// 手动备份
    Manual,
}

/// 备份状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BackupStatus {
    /// 进行中
    InProgress,
    /// 已完成
    Completed,
    /// 失败
    Failed,
    /// 已取消
    Cancelled,
}

/// 备份记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRecord {
    /// 备份ID
    pub id: Uuid,
    /// 备份类型
    pub backup_type: BackupType,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 备份大小（字节）
    pub sizebytes: u64,
    /// 包含的表数量
    pub table_count: usize,
    /// 记录数量
    pub record_count: u64,
    /// 校验和
    pub checksum: String,
    /// 备份路径
    pub backup_path: String,
    /// 压缩率
    pub compression_ratio: Option<f64>,
}

/// 恢复选项
#[derive(Debug, Clone)]
pub struct RestoreOptions {
    /// 目标表名（为空则恢复所有表）
    pub target_tables: Option<Vec<String>>,
    /// 恢复到指定时间点
    pub point_in_time: Option<DateTime<Utc>>,
    /// 验证数据完整性
    pub verify_integrity: bool,
    /// 并行恢复线程数
    pub parallel_threads: usize,
    /// 覆盖现有数据
    pub overwrite_existing: bool,
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self {
            target_tables: None,
            point_in_time: None,
            verify_integrity: true,
            parallel_threads: 4,
            overwrite_existing: false,
        }
    }
}

impl BackupManager {
    /// 创建备份管理器
    pub fn new(
        primary_storage: Arc<dyn ObjectStorage>,
        backup_storages: Vec<Arc<dyn ObjectStorage>>,
        config: BackupConfig,
        wal: Arc<WalManager>,
    ) -> Self {
        Self {
            primary_storage,
            backup_storages,
            config,
            backup_state: Arc::new(RwLock::new(BackupState::default())),
            wal,
        }
    }

    /// 启动定期备份任务
    pub async fn start_periodic_backup(&self) -> crate::Result<()> {
        let manager = self.clone();

        // 启动全量备份任务
        tokio::spawn(async move {
            let mut full_backup_interval = interval(Duration::from_secs(
                manager.config.full_backup_interval_hours * 3600,
            ));

            loop {
                full_backup_interval.tick().await;

                if let Err(e) = manager.create_full_backup().await {
                    error!("Scheduled full backup failed: {}", e);
                }
            }
        });

        let manager = self.clone();

        // 启动增量备份任务
        tokio::spawn(async move {
            let mut incremental_backup_interval = interval(Duration::from_secs(
                manager.config.incremental_backup_interval_minutes * 60,
            ));

            loop {
                incremental_backup_interval.tick().await;

                if let Err(e) = manager.create_incremental_backup().await {
                    error!("Scheduled incremental backup failed: {}", e);
                }
            }
        });

        info!("Started periodic backup tasks");
        Ok(())
    }

    /// 创建全量备份
    pub async fn create_full_backup(&self) -> crate::Result<Uuid> {
        self.create_backup(BackupType::Full, None).await
    }

    /// 创建增量备份
    pub async fn create_incremental_backup(&self) -> crate::Result<Uuid> {
        let since_time = {
            let state = self
                .backup_state
                .read()
                .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            state.last_incremental_backup.or(state.last_full_backup)
        };

        self.create_backup(BackupType::Incremental, since_time)
            .await
    }

    /// 创建手动备份
    pub async fn create_manual_backup(&self, tables: Option<Vec<String>>) -> crate::Result<Uuid> {
        self.create_backup_for_tables(BackupType::Manual, tables)
            .await
    }

    /// 创建备份
    async fn create_backup(
        &self,
        backup_type: BackupType,
        since_time: Option<DateTime<Utc>>,
    ) -> crate::Result<Uuid> {
        let task_id = Uuid::new_v4();
        let backup_task = BackupTask {
            id: task_id,
            backup_type: backup_type.clone(),
            status: BackupStatus::InProgress,
            started_at: Utc::now(),
            completed_at: None,
            tables: vec![], // 将在备份过程中填充
            progress_percentage: 0.0,
            error_message: None,
        };

        // 注册备份任务
        {
            let mut state = self
                .backup_state
                .write()
                .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            state.active_backups.insert(task_id, backup_task);
        }

        let backup_type_clone = backup_type.clone();

        // 启动异步备份任务
        let manager_clone = self.clone();
        tokio::spawn(async move {
            let result = manager_clone
                .execute_backup(task_id, backup_type, since_time)
                .await;

            match result {
                Ok(_) => {
                    info!("Backup task {} completed successfully", task_id);
                }
                Err(e) => {
                    error!("Backup task {} failed: {}", task_id, e);

                    // 更新任务状态为失败
                    let mut state = manager_clone.backup_state.write().unwrap();
                    if let Some(task) = state.active_backups.get_mut(&task_id) {
                        task.error_message = Some(e.to_string());
                        task.completed_at = Some(Utc::now());
                        task.completed_at = Some(Utc::now())
                    }
                }
            }
        });

        info!(
            "Started backup task: {} (type: {:?})",
            task_id, backup_type_clone
        );
        Ok(task_id)
    }

    /// 为特定表创建备份
    async fn create_backup_for_tables(
        &self,
        backup_type: BackupType,
        tables: Option<Vec<String>>,
    ) -> crate::Result<Uuid> {
        // If no table filter, fall through to full backup
        let Some(table_filter) = tables else {
            return self.create_backup(backup_type, None).await;
        };

        let task_id = Uuid::new_v4();
        let backup_task = BackupTask {
            id: task_id,
            backup_type: backup_type.clone(),
            status: BackupStatus::InProgress,
            started_at: Utc::now(),
            completed_at: None,
            tables: table_filter.clone(),
            progress_percentage: 0.0,
            error_message: None,
        };

        {
            let mut state = self
                .backup_state
                .write()
                .map_err(|_| crate::Error::Internal("Backup state lock poisoned".to_string()))?;
            state.active_backups.insert(task_id, backup_task);
        }

        let manager_clone = self.clone();
        tokio::spawn(async move {
            let result = manager_clone
                .execute_backup_filtered(task_id, backup_type, None, table_filter)
                .await;

            match result {
                Ok(_) => info!("Filtered backup task {} completed", task_id),
                Err(e) => {
                    error!("Filtered backup task {} failed: {}", task_id, e);
                    let mut state = manager_clone.backup_state.write().unwrap();
                    if let Some(task) = state.active_backups.get_mut(&task_id) {
                        task.status = BackupStatus::Failed;
                        task.error_message = Some(e.to_string());
                        task.completed_at = Some(Utc::now());
                    }
                }
            }
        });

        Ok(task_id)
    }

    /// 执行限定表范围的备份
    async fn execute_backup_filtered(
        &self,
        task_id: Uuid,
        backup_type: BackupType,
        since_time: Option<DateTime<Utc>>,
        table_filter: Vec<String>,
    ) -> crate::Result<BackupRecord> {
        // Collect all data then filter to the requested tables
        let mut backup_data = self.collect_backup_data(since_time, task_id).await?;
        backup_data
            .tables
            .retain(|t| table_filter.contains(&t.table.name));
        backup_data.total_records = backup_data
            .tables
            .iter()
            .map(|t| t.records.len() as u64)
            .sum();

        let checksum = self.calculate_checksum(&backup_data)?;
        let backup_path = format!("backups/filtered/{}", Utc::now().format("%Y%m%d_%H%M%S"));
        let serialized_data = if self.config.enable_compression {
            self.compress_backup_data(&backup_data)?
        } else {
            bincode::serialize(&backup_data)
                .map_err(|e| crate::error::Error::Serialization(e.to_string()))?
        };

        let raw_size = bincode::serialize(&backup_data)
            .map(|b| b.len())
            .unwrap_or(serialized_data.len());
        let compression_ratio = if self.config.enable_compression && raw_size > 0 {
            Some(raw_size as f64 / serialized_data.len() as f64)
        } else {
            None
        };

        let backup_key = format!("{}/data.bin", backup_path);
        self.primary_storage
            .put_object(
                &backup_key,
                bytes::Bytes::from(serialized_data.clone()),
                Some("application/octet-stream"),
            )
            .await
            .map_err(|e| crate::Error::Storage(e.to_string()))?;

        let record = BackupRecord {
            id: task_id,
            backup_type,
            created_at: Utc::now(),
            sizebytes: serialized_data.len() as u64,
            table_count: backup_data.tables.len(),
            record_count: backup_data.total_records,
            checksum,
            backup_path,
            compression_ratio,
        };

        {
            let mut state = self
                .backup_state
                .write()
                .map_err(|_| crate::Error::Internal("Backup state lock poisoned".to_string()))?;
            state.backup_history.push(record.clone());
            if let Some(task) = state.active_backups.get_mut(&task_id) {
                task.status = BackupStatus::Completed;
                task.completed_at = Some(Utc::now());
                task.progress_percentage = 100.0;
            }
        }

        Ok(record)
    }

    /// 执行备份操作
    async fn execute_backup(
        &self,
        task_id: Uuid,
        backup_type: BackupType,
        since_time: Option<DateTime<Utc>>,
    ) -> crate::Result<BackupRecord> {
        let backup_start = Utc::now();
        let backup_path = format!(
            "backups/{}/{}",
            backup_type.to_string().to_lowercase(),
            backup_start.format("%Y%m%d_%H%M%S")
        );

        info!(
            "Executing {} backup to path: {}",
            backup_type.to_string().to_lowercase(),
            backup_path
        );

        // 获取需要备份的数据
        let backup_data = match self.collect_backup_data(since_time, task_id).await {
            Ok(data) => data,
            Err(e) => {
                return Err(crate::Error::Internal(format!(
                    "Backup data collection failed: {:?}",
                    e
                )))
            }
        };

        // 计算校验和
        let checksum = match self.calculate_checksum(&backup_data) {
            Ok(sum) => sum,
            Err(e) => {
                return Err(crate::Error::Internal(format!(
                    "Checksum calculation failed: {:?}",
                    e
                )))
            }
        };

        // 备份到所有存储后端
        let mut total_size = 0u64;
        let mut backup_succeeded = false;

        for (i, storage) in self.backup_storages.iter().enumerate() {
            match self
                .write_backup_to_storage(storage.as_ref(), &backup_path, &backup_data, i + 1)
                .await
            {
                Ok(size) => {
                    total_size = size;
                    backup_succeeded = true;
                    info!(
                        "Backup written to storage backend {} ({} bytes)",
                        i + 1,
                        size
                    );
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    error!(
                        "Failed to write backup to storage backend {}: {}",
                        i + 1,
                        error_msg
                    );
                }
            }
        }

        if !backup_succeeded {
            return Err(crate::error::Error::Internal(
                "All backup storage backends failed".to_string(),
            ));
        }

        // 更新备份状态
        {
            let mut state = self
                .backup_state
                .write()
                .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            match backup_type {
                BackupType::Full => state.last_full_backup = Some(backup_start),
                BackupType::Incremental => state.last_incremental_backup = Some(backup_start),
                BackupType::Manual => {} // 手动备份不更新状态
            }
        }

        let backup_record = BackupRecord {
            id: task_id,
            backup_type,
            created_at: backup_start,
            sizebytes: total_size,
            table_count: backup_data.tables.len(),
            record_count: backup_data.total_records,
            checksum,
            backup_path,
            compression_ratio: if self.config.enable_compression {
                Some(0.7)
            } else {
                None
            },
        };

        info!(
            "Backup completed: {} ({} bytes, {} tables, {} records)",
            task_id,
            total_size,
            backup_data.tables.len(),
            backup_data.total_records
        );

        Ok(backup_record)
    }

    /// 收集备份数据
    async fn collect_backup_data(
        &self,
        since_time: Option<DateTime<Utc>>,
        task_id: Uuid,
    ) -> crate::Result<BackupData> {
        info!("Collecting backup data since: {:?}", since_time);

        let mut backup_data = BackupData {
            tables: vec![],
            total_records: 0,
            payload: BackupMetadata {
                version: "1.0".to_string(),
                created_at: Utc::now(),
                backup_type: if since_time.is_some() {
                    BackupType::Incremental
                } else {
                    BackupType::Full
                },
                since_time,
            },
        };

        // 从主存储获取所有对象
        let list_options = crate::storage::object::types::ListObjectsOptions {
            prefix: Some("tables/".to_string()),
            ..Default::default()
        };

        let objects = match self.primary_storage.list_objects(&list_options).await {
            Ok(objs) => objs,
            Err(e) => return Err(crate::Error::Storage(e.to_string())),
        };

        info!("Found {} objects for backup", objects.objects.len());
        self.update_backup_progress(task_id, 20.0).await;

        // 按表分组收集数据
        let mut table_data_map: HashMap<String, Vec<(String, Vec<u8>)>> = HashMap::new();

        for (i, obj) in objects.objects.iter().enumerate() {
            // 检查时间过滤
            if let Some(since) = since_time {
                if obj.last_modified < since {
                    continue; // 跳过旧数据（增量备份）
                }
            }

            // 解析对象路径以确定表名
            if let Some(table_name) = self.extract_table_name_from_path(&obj.key) {
                // 获取对象数据
                match self.primary_storage.get_object(&obj.key).await {
                    Ok(data) => table_data_map
                        .entry(table_name)
                        .or_default()
                        .push((obj.key.clone(), data)),
                    Err(e) => {
                        warn!("Failed to get object {}: {}", obj.key, e);
                    }
                }
            }

            // 更新进度
            let progress = 20.0 + (i as f64 / objects.objects.len() as f64) * 60.0;
            self.update_backup_progress(task_id, progress).await;
        }

        // 将收集的数据转换为表备份格式
        for (table_name, table_objects) in table_data_map {
            match self
                .reconstruct_table_backup(&table_name, table_objects)
                .await
            {
                Ok(table_backup) => {
                    backup_data.total_records += table_backup.records.len() as u64;
                    backup_data.tables.push(table_backup);
                }
                Err(e) => {
                    error!("Failed to reconstruct table {}: {}", table_name, e);
                }
            }
        }

        self.update_backup_progress(task_id, 90.0).await;

        info!(
            "Collected backup data: {} tables, {} records",
            backup_data.tables.len(),
            backup_data.total_records
        );

        Ok(backup_data)
    }

    /// 从对象路径提取表名
    fn extract_table_name_from_path(&self, path: &str) -> Option<String> {
        // _预期路径格式: tables/{table_name}/records/{record_id} 或 tables/{table_name}/index
        if path.starts_with("tables/") {
            let parts: Vec<&str> = path.split('/').collect();
            if parts.len() >= 2 {
                return Some(parts[1].to_string());
            }
        }
        None
    }

    /// 重构表备份数据
    async fn reconstruct_table_backup(
        &self,
        tablename: &str,
        objects: Vec<(String, Vec<u8>)>,
    ) -> crate::Result<TableBackup> {
        info!("Reconstructing table backup for: {}", tablename);

        let mut records = Vec::new();
        let mut index_data = None;

        // 创建默认表结构（实际应该从元数据获取）
        let table = Table {
            id: Uuid::new_v4(),
            name: tablename.to_string(),
            dimension: 128, // 默认维度，实际应该从元数据读取
            distance_metric: crate::types::DistanceMetric::Cosine,
            description: None,
            index_type: crate::types::IndexType::Hnsw,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            record_count: records.len() as u64,
            text_fields: None,
            enable_full_text_search: false,
            schema_version: 1,
            payload_schema: None,
            memory_optimization: None,
            named_vectors: None,
        };

        for (path, data) in objects {
            if path.contains("/records/") {
                // 这是记录数据
                match bincode::deserialize::<crate::types::Record>(&data) {
                    Ok(record) => records.push(record),
                    Err(e) => {
                        warn!("Failed to deserialize record from {}: {}", path, e);
                    }
                }
            } else if path.contains("/index") {
                // 这是索引数据
                index_data = Some(data)
            }
        }

        info!(
            "Reconstructed table {} with {} records",
            tablename,
            records.len()
        );

        Ok(TableBackup {
            table,
            records,
            index_data,
        })
    }

    /// 写入备份到存储
    async fn write_backup_to_storage(
        &self,
        storage: &dyn ObjectStorage,
        path: &str,
        data: &BackupData,
        storage_index: usize,
    ) -> crate::Result<u64> {
        let serialized_data = if self.config.enable_compression {
            match self.compress_backup_data(data) {
                Ok(data) => data,
                Err(e) => {
                    return Err(crate::Error::Internal(format!(
                        "Compression failed: {:?}",
                        e
                    )))
                }
            }
        } else {
            bincode::serialize(data)
                .map_err(|e| crate::error::Error::Serialization(e.to_string()))?
        };

        let checksum = self.calculate_checksum(data)?;
        let checksum_data = checksum.as_bytes().to_vec();

        let backup_key = format!("{}/backup_{}.dat", path, storage_index);
        let checksum_key = format!("{}/checksum_{}.txt", path, storage_index);

        // 写入主备份文件
        match storage
            .put_object(
                &backup_key,
                Bytes::from(serialized_data.clone()),
                Some("application/octet-stream"),
            )
            .await
        {
            Ok(_) => {}
            Err(e) => return Err(crate::Error::Storage(e.to_string())),
        };

        // 写入校验和文件
        match storage
            .put_object(
                &checksum_key,
                Bytes::from(checksum_data),
                Some("text/plain"),
            )
            .await
        {
            Ok(_) => {}
            Err(e) => return Err(crate::Error::Storage(e.to_string())),
        };

        // 写入元数据文件
        let metadata_key = format!("{}/metadata_{}.json", path, storage_index);
        let metadata_json = serde_json::to_string(&data.payload)
            .map_err(|e| crate::error::Error::Serialization(e.to_string()))?;
        match storage
            .put_object(
                &metadata_key,
                Bytes::from(metadata_json.into_bytes()),
                Some("application/json"),
            )
            .await
        {
            Ok(_) => {}
            Err(e) => return Err(crate::Error::Storage(e.to_string())),
        };

        Ok(serialized_data.len() as u64)
    }

    /// 压缩备份数据（zstd level 3）
    fn compress_backup_data(&self, data: &BackupData) -> crate::Result<Vec<u8>> {
        let serialized = bincode::serialize(data)
            .map_err(|e| crate::error::Error::Serialization(e.to_string()))?;
        zstd::encode_all(serialized.as_slice(), 3)
            .map_err(|e| crate::error::Error::Internal(format!("Compression failed: {}", e)))
    }

    /// 计算校验和
    fn calculate_checksum(&self, data: &BackupData) -> crate::Result<String> {
        let serialized = bincode::serialize(data)
            .map_err(|e| crate::error::Error::Serialization(e.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(&serialized);
        let checksum = format!("{:x}", hasher.finalize());

        Ok(checksum)
    }

    /// 更新备份进度
    async fn update_backup_progress(&self, task_id: Uuid, progress: f64) {
        let mut state = self.backup_state.write().unwrap();
        if let Some(task) = state.active_backups.get_mut(&task_id) {
            task.progress_percentage = progress
        }
    }

    /// 恢复备份
    pub async fn restore_backup(
        &self,
        backup_id: Uuid,
        options: RestoreOptions,
    ) -> crate::Result<()> {
        info!(
            "Starting restore for backup: {} with _options: {:?}",
            backup_id, options
        );

        // 如果指定了时间点恢复，需要找到最佳的备份组合
        if let Some(point_in_time) = options.point_in_time {
            return self.restore_point_in_time(point_in_time, options).await;
        }

        // 查找备份记录
        let backup_record = {
            let state = self
                .backup_state
                .read()
                .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            state
                .backup_history
                .iter()
                .find(|record| record.id == backup_id)
                .cloned()
        };

        let backup_record = backup_record.ok_or_else(|| {
            crate::error::Error::InvalidInput(format!("Backup {} not found", backup_id))
        })?;

        // 从备份存储读取数据
        // Load backup data from storage
        let backup_data = match self
            .primary_storage
            .get_object(&backup_record.backup_path)
            .await
        {
            Ok(data) => data,
            Err(_) => {
                return Err(crate::error::Error::InvalidInput(
                    "Backup data not found".to_string(),
                ))
            }
        };

        // 验证数据完整性
        if options.verify_integrity {
            let parsed_backup_data: BackupData =
                serde_json::from_slice(&backup_data).map_err(|e| {
                    crate::Error::Internal(format!("Failed to parse backup data: {}", e))
                })?;
            self.verify_backup_integrity(&parsed_backup_data, &backup_record.checksum)?;
            // Implementation placeholder
        }
        // 执行恢复
        let parsed_backup_data: BackupData = serde_json::from_slice(&backup_data)
            .map_err(|e| crate::Error::Internal(format!("Failed to parse backup data: {}", e)))?;
        self.execute_restore(&parsed_backup_data, &options).await?;

        info!("Restore completed for backup: {}", backup_id);
        Ok(())
    }

    /// 时间点恢复
    pub async fn restore_point_in_time(
        &self,
        target_time: DateTime<Utc>,
        options: RestoreOptions,
    ) -> crate::Result<()> {
        info!("Starting point-in-time restore to: {}", target_time);

        // 从锁中提取所需数据
        let (base_full_backup, incremental_backups) = {
            let state = self
                .backup_state
                .read()
                .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;

            // 找到目标时间点之前的最新全量备份
            let base_full_backup = state
                .backup_history
                .iter()
                .filter(|record| {
                    record.backup_type == BackupType::Full && record.created_at <= target_time
                })
                .max_by_key(|record| record.created_at)
                .ok_or_else(|| {
                    crate::error::Error::InvalidInput(format!(
                        "No full backup found before target time: {}",
                        target_time
                    ))
                })?
                .clone();

            // 找到全量备份之后、目标时间之前的所有增量备份
            let mut incremental_backups: Vec<_> = state
                .backup_history
                .iter()
                .filter(|record| {
                    record.backup_type == BackupType::Incremental
                        && record.created_at > base_full_backup.created_at
                        && record.created_at <= target_time
                })
                .cloned()
                .collect();

            // 按时间排序增量备份
            incremental_backups.sort_by_key(|record| record.created_at);

            (base_full_backup, incremental_backups)
        };

        info!(
            "Using base full backup: {} created at {}",
            base_full_backup.id, base_full_backup.created_at
        );
        info!(
            "Found {} incremental backups to apply",
            incremental_backups.len()
        );

        // 应用基础全量备份
        let full_backup_data = match self
            .read_backup_from_storage(&base_full_backup.backup_path)
            .await
        {
            Ok(data) => data,
            Err(e) => {
                return Err(crate::Error::Internal(format!(
                    "Backup read failed: {:?}",
                    e
                )))
            }
        };

        if options.verify_integrity {
            match self.verify_backup_integrity(&full_backup_data, &base_full_backup.checksum) {
                Ok(_) => {}
                Err(e) => {
                    return Err(crate::Error::Internal(format!(
                        "Integrity verification failed: {:?}",
                        e
                    )))
                }
            };
        }
        // 先清理目标表（如果允许覆盖）
        if options.overwrite_existing {
            info!("Clearing existing data before point-in-time restore");
            match self.clear_target_tables(&options.target_tables).await {
                Ok(_) => {}
                Err(e) => {
                    return Err(crate::Error::Internal(format!(
                        "Table clear failed: {:?}",
                        e
                    )))
                }
            };
        }
        // 恢复全量备份
        match self.execute_restore(&full_backup_data, &options).await {
            Ok(_) => {}
            Err(e) => {
                return Err(crate::Error::Internal(format!(
                    "Restore execution failed: {:?}",
                    e
                )))
            }
        };

        // 依次应用增量备份
        for (i, incremental_record) in incremental_backups.iter().enumerate() {
            info!(
                "Applying incremental backup {} of {} (created: {})",
                i + 1,
                incremental_backups.len(),
                incremental_record.created_at
            );

            let incremental_data = match self
                .read_backup_from_storage(&incremental_record.backup_path)
                .await
            {
                Ok(data) => data,
                Err(e) => {
                    return Err(crate::Error::Internal(format!(
                        "Incremental backup read failed: {:?}",
                        e
                    )))
                }
            };

            if options.verify_integrity {
                match self.verify_backup_integrity(&incremental_data, &incremental_record.checksum)
                {
                    Ok(_) => {}
                    Err(e) => {
                        return Err(crate::Error::Internal(format!(
                            "Incremental integrity verification failed: {:?}",
                            e
                        )))
                    }
                };
            }
            // 应用增量变更（使用WAL重放）
            match self
                .apply_incremental_changes(&incremental_data, &options)
                .await
            {
                Ok(_) => {}
                Err(e) => {
                    return Err(crate::Error::Internal(format!(
                        "Incremental changes application failed: {:?}",
                        e
                    )))
                }
            };
        }

        info!("Point-in-time restore completed to: {}", target_time);
        Ok(())
    }

    /// 清理目标表
    async fn clear_target_tables(&self, target_tables: &Option<Vec<String>>) -> crate::Result<()> {
        let list_options = crate::storage::object::types::ListObjectsOptions {
            prefix: Some("tables/".to_string()),
            ..Default::default()
        };

        let objects = match self.primary_storage.list_objects(&list_options).await {
            Ok(objs) => objs,
            Err(e) => return Err(crate::Error::Storage(e.to_string())),
        };

        for obj in objects.objects {
            if let Some(table_name) = self.extract_table_name_from_path(&obj.key) {
                // 检查是否为目标表
                if let Some(ref target_list) = target_tables {
                    if !target_list.contains(&table_name) {
                        continue; // 跳过非目标表
                    }
                }

                // 删除对象
                match self.primary_storage.delete_object(&obj.key).await {
                    Ok(_) => {}
                    Err(e) => {
                        warn!("Failed to delete object {}: {}", obj.key, e);
                    }
                }
            }
        }
        Ok(())
    }

    /// 应用增量变更
    async fn apply_incremental_changes(
        &self,
        incremental_data: &BackupData,
        options: &RestoreOptions,
    ) -> crate::Result<()> {
        info!(
            "Applying incremental changes with {} tables",
            incremental_data.tables.len()
        );

        // 过滤目标表
        let tables_to_apply: Vec<&TableBackup> =
            if let Some(ref target_tables) = options.target_tables {
                incremental_data
                    .tables
                    .iter()
                    .filter(|table_backup| target_tables.contains(&table_backup.table.name))
                    .collect()
            } else {
                incremental_data.tables.iter().collect()
            };

        // 应用每个表的增量变更
        for table_backup in tables_to_apply {
            match self
                .apply_table_incremental_changes(table_backup, options)
                .await
            {
                Ok(_) => {}
                Err(e) => {
                    return Err(crate::Error::Internal(format!(
                        "Table incremental changes failed: {:?}",
                        e
                    )))
                }
            };
        }
        Ok(())
    }

    /// 应用表的增量变更
    async fn apply_table_incremental_changes(
        &self,
        table_backup: &TableBackup,
        _options: &RestoreOptions,
    ) -> crate::Result<()> {
        let table_name = &table_backup.table.name;
        info!("Applying incremental changes for table: {}", table_name);

        // 更新表元数据（如果有变更）
        let table_path = format!("tables/{}/metadata", table_name);
        let table_metadata = bincode::serialize(&table_backup.table)
            .map_err(|e| crate::error::Error::Serialization(e.to_string()))?;
        match self
            .primary_storage
            .put_object(
                &table_path,
                Bytes::from(table_metadata),
                Some("application/octet-stream"),
            )
            .await
        {
            Ok(_) => {}
            Err(e) => return Err(crate::Error::Storage(e.to_string())),
        };

        // 应用记录变更（增量备份中的记录是新增或修改的）
        for record in &table_backup.records {
            let record_path = format!("tables/{}/records/{}", table_name, record.id);
            let record_data = bincode::serialize(record)
                .map_err(|e| crate::Error::Serialization(e.to_string()))?;

            match self
                .primary_storage
                .put_object(
                    &record_path,
                    Bytes::from(record_data),
                    Some("application/octet-stream"),
                )
                .await
            {
                Ok(_) => {}
                Err(e) => return Err(crate::Error::Storage(e.to_string())),
            };
        }

        // 更新索引（如果有）
        if let Some(ref index_data) = table_backup.index_data {
            let index_path = format!("tables/{}/index", table_name);
            match self
                .primary_storage
                .put_object(
                    &index_path,
                    Bytes::from(index_data.clone()),
                    Some("application/octet-stream"),
                )
                .await
            {
                Ok(_) => {}
                Err(e) => return Err(crate::Error::Storage(e.to_string())),
            };
        }

        info!(
            "Applied incremental changes for table {} ({} records)",
            table_name,
            table_backup.records.len()
        );

        Ok(())
    }

    /// 从存储读取备份数据
    async fn read_backup_from_storage(&self, backup_path: &str) -> crate::Result<BackupData> {
        let mut last_error = None;

        // 尝试从所有备份存储读取
        for (i, storage) in self.backup_storages.iter().enumerate() {
            let backup_key = format!("{}/backup_{}.dat", backup_path, i + 1);

            match storage.get_object(&backup_key).await {
                Ok(data) => {
                    return self.deserialize_backup_data(&data);
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    last_error = Some(e);
                    warn!(
                        "Failed to read backup from storage {}: {}",
                        i + 1,
                        error_msg
                    );
                }
            }
        }

        Err(crate::Error::Storage(format!(
            "Failed to read backup from all storage backends: {:?}",
            last_error
        )))
    }

    /// 反序列化备份数据
    fn deserialize_backup_data(&self, data: &[u8]) -> crate::Result<BackupData> {
        if self.config.enable_compression {
            self.decompress_backup_data(data)
        } else {
            bincode::deserialize(data)
                .map_err(|e| crate::error::Error::Serialization(e.to_string()))
        }
    }

    /// 解压缩备份数据（zstd）
    fn decompress_backup_data(&self, data: &[u8]) -> crate::Result<BackupData> {
        let decompressed = zstd::decode_all(data)
            .map_err(|e| crate::error::Error::Internal(format!("Decompression failed: {}", e)))?;
        bincode::deserialize(&decompressed)
            .map_err(|e| crate::error::Error::Serialization(e.to_string()))
    }

    /// 验证备份完整性
    fn verify_backup_integrity(
        &self,
        data: &BackupData,
        expected_checksum: &str,
    ) -> crate::Result<()> {
        let actual_checksum = self.calculate_checksum(data)?;

        if actual_checksum != expected_checksum {
            return Err(crate::error::Error::InvalidInput(format!(
                "Backup checksum mismatch: expected {}, got {}",
                expected_checksum, actual_checksum
            )));
        }

        info!("Backup integrity verification passed");
        Ok(())
    }

    /// 执行恢复操作
    async fn execute_restore(
        &self,
        backup_data: &BackupData,
        options: &RestoreOptions,
    ) -> crate::Result<()> {
        info!("Executing restore with {} tables", backup_data.tables.len());

        // 并行恢复每个表
        let tables_to_restore: Vec<TableBackup> =
            if let Some(ref target_tables) = options.target_tables {
                backup_data
                    .tables
                    .iter()
                    .filter(|table| target_tables.contains(&table.table.name))
                    .cloned()
                    .collect()
            } else {
                backup_data.tables.clone()
            };
        let total_tables = tables_to_restore.len();

        // 并发限制
        let semaphore = Arc::new(tokio::sync::Semaphore::new(4)); // 最多4个并发恢复任务
        let mut tasks = Vec::new();

        for (i, table_backup) in tables_to_restore.into_iter().enumerate() {
            let table_backup = table_backup.clone();
            let primary_storage = self.primary_storage.clone();
            let semaphore = semaphore.clone();
            let options = options.clone();
            let task = tokio::spawn(async move {
                let _permit = match semaphore.acquire().await {
                    Ok(permit) => permit,
                    Err(e) => {
                        return Err(crate::Error::Internal(format!(
                            "Semaphore acquire failed: {:?}",
                            e
                        )))
                    }
                };
                match Self::restore_single_table(
                    &primary_storage,
                    &table_backup,
                    &options,
                    i + 1,
                    total_tables,
                )
                .await
                {
                    Ok(_) => Ok(()),
                    Err(e) => Err(crate::Error::Internal(format!(
                        "Table restore failed: {:?}",
                        e
                    ))),
                }
            });
            tasks.push(task);
        }
        // 等待所有恢复任务完成
        for task in tasks {
            match task.await {
                Ok(result) => {
                    if let Err(join_err) = result {
                        error!("Table restore task join failed: {}", join_err);
                        return Err(crate::error::Error::Internal(join_err.to_string()));
                    }
                }
                Err(e) => {
                    error!("Task join failed: {}", e);
                    return Err(crate::error::Error::Internal(e.to_string()));
                }
            }
        }
        info!("Restore completed for all {} tables", total_tables);
        Ok(())
    }

    /// 恢复单个表
    async fn restore_single_table(
        storage: &Arc<dyn ObjectStorage>,
        table_backup: &TableBackup,
        options: &RestoreOptions,
        table_index: usize,
        total_tables: usize,
    ) -> crate::Result<()> {
        let table_name = &table_backup.table.name;
        info!(
            "Restoring table {} ({}/{})",
            table_name, table_index, total_tables
        );

        // 检查表是否已存在
        let table_path = format!("tables/{}/metadata", table_name);
        let table_exists = match storage.object_exists(&table_path).await {
            Ok(exists) => exists,
            Err(e) => return Err(crate::Error::Storage(e.to_string())),
        };

        if table_exists && !options.overwrite_existing {
            warn!(
                "Table {} already exists, skipping (use overwrite_existing=true to replace)",
                table_name
            );
            return Ok(());
        }

        // 恢复表元数据
        let table_metadata = bincode::serialize(&table_backup.table)
            .map_err(|e| crate::error::Error::Serialization(e.to_string()))?;
        match storage
            .put_object(
                &table_path,
                Bytes::from(table_metadata),
                Some("application/octet-stream"),
            )
            .await
        {
            Ok(_) => {}
            Err(e) => return Err(crate::Error::Storage(e.to_string())),
        };

        // 恢复记录数据（批量上传优化）
        let batch_size = 100; // 每批处理100条记录
        for (batch_idx, batch) in table_backup.records.chunks(batch_size).enumerate() {
            let mut batch_tasks = Vec::new();
            for record in batch.iter() {
                let record_path = format!("tables/{}/records/{}", table_name, record.id);
                let record_data = match bincode::serialize(record) {
                    Ok(data) => data,
                    Err(e) => return Err(crate::error::Error::Serialization(e.to_string())),
                };
                let storage_clone = storage.clone();
                let task = async move {
                    match storage_clone
                        .put_object(
                            &record_path,
                            Bytes::from(record_data),
                            Some("application/octet-stream"),
                        )
                        .await
                    {
                        Ok(_) => Ok(()),
                        Err(e) => Err(crate::Error::Storage(e.to_string())),
                    }
                };
                batch_tasks.push(task);
            }

            // 并行执行批次内的记录恢复
            match futures::future::try_join_all(batch_tasks).await {
                Ok(_) => {}
                Err(e) => {
                    return Err(crate::Error::Internal(format!(
                        "Batch task failed: {:?}",
                        e
                    )))
                }
            };

            debug!(
                "Restored batch {} for table {} ({} records)",
                batch_idx + 1,
                table_name,
                batch.len()
            );
        }

        // 恢复索引数据
        if let Some(ref index_data) = table_backup.index_data {
            let index_path = format!("tables/{}/index", table_name);
            match storage
                .put_object(
                    &index_path,
                    Bytes::from(index_data.clone()),
                    Some("application/octet-stream"),
                )
                .await
            {
                Ok(_) => {}
                Err(e) => return Err(crate::Error::Storage(e.to_string())),
            };

            info!(
                "Restored index for table {} ({} bytes)",
                table_name,
                index_data.len()
            );
        }

        info!(
            "Successfully restored table {} with {} records",
            table_name,
            table_backup.records.len()
        );

        Ok(())
    }

    /// 获取备份状态
    pub fn get_backup_state(&self) -> BackupState {
        let state = self.backup_state.read().unwrap();
        state.clone()
    }

    /// 列出所有备份
    pub fn list_backups(&self) -> Vec<BackupRecord> {
        let state = self.backup_state.read().unwrap();
        state.backup_history.clone()
    }

    /// 删除备份
    pub async fn delete_backup(&self, backup_id: Uuid) -> crate::Result<()> {
        let backup_record = {
            let mut state = self
                .backup_state
                .write()
                .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?;
            state
                .backup_history
                .iter()
                .position(|record| record.id == backup_id)
                .map(|index| state.backup_history.remove(index))
        };

        let backup_record = backup_record.ok_or_else(|| {
            crate::error::Error::InvalidInput(format!("Backup {} not found", backup_id))
        })?;

        // 从所有存储后端删除备份文件
        for (i, storage) in self.backup_storages.iter().enumerate() {
            let backup_key = format!("{}/backup_{}.dat", backup_record.backup_path, i + 1);
            let metadata_key = format!("{}/metadata_{}.json", backup_record.backup_path, i + 1);

            if let Err(e) = storage.delete_object(&backup_key).await {
                warn!("Failed to delete backup file from storage {}: {}", i + 1, e);
            }

            if let Err(e) = storage.delete_object(&metadata_key).await {
                warn!(
                    "Failed to delete metadata file from storage {}: {}",
                    i + 1,
                    e
                );
            }
        }

        info!("Deleted backup: {}", backup_id);
        Ok(())
    }
}

impl Clone for BackupManager {
    fn clone(&self) -> Self {
        Self {
            primary_storage: Arc::clone(&self.primary_storage),
            backup_storages: self.backup_storages.clone(),
            config: self.config.clone(),
            backup_state: Arc::clone(&self.backup_state),
            wal: Arc::clone(&self.wal),
        }
    }
}

/// 备份数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupData {
    /// 表数据
    pub tables: Vec<TableBackup>,
    /// 总记录数
    pub total_records: u64,
    /// 元数据
    pub payload: BackupMetadata,
}

/// 表备份数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableBackup {
    /// 表信息
    pub table: Table,
    /// 记录数据
    pub records: Vec<crate::types::Record>,
    /// 索引数据
    pub index_data: Option<Vec<u8>>,
}

/// 备份元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// 备份格式版本
    pub version: String,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 备份类型
    pub backup_type: BackupType,
    /// 增量备份起始时间
    pub since_time: Option<DateTime<Utc>>,
}

impl std::fmt::Display for BackupType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BackupType::Full => "full",
            BackupType::Incremental => "incremental",
            BackupType::Manual => "manual",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {

    // Commented out - WalConfig no longer exists, needs rewrite
    // #[tokio::test]
    // async fn test_backup_manager_creation() {
    //     let temp_dir = TempDir::new()?;
    //     let primary_storage = Arc::new(FilesystemStorage::new(temp_dir.path(), "primary")?);
    //     let backup_storage = Arc::new(FilesystemStorage::new(temp_dir.path(), "backup")?);
    //
    //     let config = BackupConfig::default();
    //     let wal_config = crate::wal::WalConfig::default();
    //     let wal = Arc::new(WalManager::new(wal_config)??);
    //
    //     let manager = BackupManager::new(primary_storage, vec![backup_storage], config, wal);
    //
    //     assert_eq!(manager.backup_storages.len(), 1)
    // }
}
