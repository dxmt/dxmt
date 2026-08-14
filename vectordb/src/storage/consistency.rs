#![allow(unused_variables)]
#![allow(unused_imports)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::types::vector::RecordId;
use crate::wal::WalManager;

/// 数据一致性验证器
pub struct ConsistencyChecker {
    storage: Arc<super::IntelligentIndexedStorage>,
    #[allow(dead_code)]
    wal: Arc<WalManager>,
    config: ConsistencyConfig,
}

/// 一致性检查配置
#[derive(Debug, Clone)]
pub struct ConsistencyConfig {
    /// 是否启用实时一致性检查
    pub enable_realtime_check: bool,
    /// 定期检查间隔（秒）
    pub periodic_check_interval: u64,
    /// 检查批次大小
    pub check_batch_size: usize,
    /// 检查超时（秒）
    pub check_timeout: u64,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            enable_realtime_check: true,
            periodic_check_interval: 3600, // 1小时
            check_batch_size: 1000,
            check_timeout: 300, // 5分钟
        }
    }
}

/// 一致性检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyReport {
    /// 检查时间
    pub check_time: DateTime<Utc>,
    /// 检查的表
    pub tablename: String,
    /// 总记录数
    pub total_records: u64,
    /// 检查的记录数
    pub checked_records: u64,
    /// 发现的不一致问题
    pub inconsistencies: Vec<InconsistencyIssue>,
    /// 检查耗时（毫秒）
    pub check_duration_ms: u64,
    /// 检查状态
    pub status: CheckStatus,
}

/// 检查状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckStatus {
    /// 成功完成
    Success,
    /// 部分完成
    Partial,
    /// 失败
    Failed,
    /// 超时
    Timeout,
}

/// 不一致问题类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InconsistencyIssue {
    /// 记录在存储中存在但索引中不存在
    MissingInIndex {
        record_id: RecordId,
        tablename: String,
    },
    /// 记录在索引中存在但存储中不存在
    MissingInStorage {
        record_id: RecordId,
        tablename: String,
    },
    /// 向量数据不匹配
    VectorMismatch {
        record_id: RecordId,
        tablename: String,
        storage_vector: Option<Vec<f32>>,
        index_vector: Option<Vec<f32>>,
    },
    /// 记录状态不一致
    StatusMismatch {
        record_id: RecordId,
        tablename: String,
        expected_status: String,
        actual_status: String,
    },
    /// 版本不一致
    VersionMismatch {
        record_id: RecordId,
        tablename: String,
        storage_version: u64,
        expected_version: u64,
    },
    /// 元数据不一致
    MetadataMismatch {
        tablename: String,
        storage_count: u64,
        index_count: usize,
    },
}

impl ConsistencyChecker {
    /// 创建新的一致性检查器
    pub fn new(
        storage: Arc<super::IntelligentIndexedStorage>,
        wal: Arc<WalManager>,
        config: ConsistencyConfig,
    ) -> Self {
        Self {
            storage,
            wal,
            config,
        }
    }

    /// 执行全面的一致性检查
    pub async fn check_consistency(&self, tablename: &str) -> crate::Result<ConsistencyReport> {
        let start_time = std::time::Instant::now();
        let check_time = Utc::now();
        let mut inconsistencies = Vec::new();

        info!("Starting consistency check for table: {}", tablename);

        // 获取表信息
        let table = self.storage.get_table(tablename).await?;
        let total_records = table.record_count;

        let mut checked_records = 0;
        let mut status = CheckStatus::Success;

        // 检查1：验证索引和存储的记录数一致性
        if let Err(e) = self
            .check_record_count_consistency(tablename, &mut inconsistencies)
            .await
        {
            warn!("Record count consistency check failed: {}", e);
            status = CheckStatus::Partial;
        }

        // 检查2：逐个验证记录的一致性
        match self
            .check_individual_records(tablename, &mut inconsistencies, &mut checked_records)
            .await
        {
            Ok(_) => {
                info!("Individual record check completed for table: {}", tablename);
            }
            Err(e) => {
                error!("Individual record check failed: {}", e);
                status = CheckStatus::Failed;
            }
        }

        // 检查3：验证WAL和实际状态的一致性
        if self.config.enable_realtime_check {
            if let Err(e) = self
                .check_wal_consistency(tablename, &mut inconsistencies)
                .await
            {
                warn!("WAL consistency check failed: {}", e);
                status = CheckStatus::Partial;
            }
        }

        let check_duration_ms = start_time.elapsed().as_millis() as u64;

        let report = ConsistencyReport {
            check_time,
            tablename: tablename.to_string(),
            total_records,
            checked_records,
            inconsistencies: inconsistencies.clone(),
            check_duration_ms,
            status,
        };

        // 记录检查结果
        if !inconsistencies.is_empty() {
            warn!(
                "Found {} inconsistencies in table {}",
                inconsistencies.len(),
                tablename
            );
            for issue in &inconsistencies {
                warn!("Inconsistency: {:?}", issue);
            }
        } else {
            info!("No inconsistencies found in table {}", tablename);
        }

        Ok(report)
    }

    /// 检查记录数一致性
    async fn check_record_count_consistency(
        &self,
        tablename: &str,
        issues: &mut Vec<InconsistencyIssue>,
    ) -> crate::Result<()> {
        let table = self.storage.get_table(tablename).await?;
        let storage_count = table.record_count;

        let index_count = self.storage.get_index_record_count(tablename).await?;

        if storage_count != index_count as u64 {
            issues.push(InconsistencyIssue::MetadataMismatch {
                tablename: tablename.to_string(),
                storage_count,
                index_count,
            });
        }

        Ok(())
    }

    /// 检查单个记录的一致性
    async fn check_individual_records(
        &self,
        tablename: &str,
        issues: &mut Vec<InconsistencyIssue>,
        checked_count: &mut u64,
    ) -> crate::Result<()> {
        // 获取表中的所有记录ID
        let record_ids = self.get_all_record_ids(tablename).await?;

        for record_id in record_ids.iter().take(self.config.check_batch_size) {
            *checked_count += 1;

            // 从存储获取记录
            let storage_record = match self.storage.get_record(tablename, record_id).await {
                Ok(record) => Some(record),
                Err(crate::Error::RecordNotFound { .. }) => None,
                Err(e) => return Err(e),
            };

            // 检查索引中是否存在该记录
            let index_exists = self.check_record_in_index(tablename, record_id).await?;

            // 验证一致性
            match (storage_record, index_exists) {
                (Some(record), false) => {
                    // 存储中有，索引中没有
                    issues.push(InconsistencyIssue::MissingInIndex {
                        record_id: record_id.clone(),
                        tablename: tablename.to_string(),
                    });
                }
                (None, true) => {
                    // 索引中有，存储中没有
                    issues.push(InconsistencyIssue::MissingInStorage {
                        record_id: record_id.clone(),
                        tablename: tablename.to_string(),
                    });
                }
                (Some(record), true) => {
                    // 都存在，检查向量是否一致
                    if let Err(issue) = self.verify_vector_consistency(tablename, &record).await {
                        issues.push(issue);
                    }
                }
                (None, false) => {
                    // 都不存在，正常情况
                }
            }
        }

        Ok(())
    }

    /// 检查WAL一致性
    async fn check_wal_consistency(
        &self,
        tablename: &str,
        _issues: &mut Vec<InconsistencyIssue>,
    ) -> crate::Result<()> {
        // Simplified WAL consistency check
        // The current WalManager doesn't support transaction tracking
        // This is a placeholder for future implementation

        warn!(
            "WAL consistency check not fully implemented for table: {}",
            tablename
        );

        Ok(())
    }

    /// 获取表中所有记录ID
    async fn get_all_record_ids(&self, tablename: &str) -> crate::Result<Vec<RecordId>> {
        self.storage.list_all_record_ids(tablename).await
    }

    /// 检查记录是否在索引中存在
    async fn check_record_in_index(
        &self,
        tablename: &str,
        record_id: &RecordId,
    ) -> crate::Result<bool> {
        // 尝试搜索该记录的向量
        // 如果找到相同ID的记录，说明在索引中存在

        // 首先获取记录的向量
        match self.storage.get_record(tablename, record_id).await {
            Ok(record) => {
                // 使用向量搜索来验证索引中是否存在
                let search_query = crate::types::SearchQuery {
                    vector: record.vector,
                    top_k: 1,
                    filter: None,
                    with_payload: false,
                    with_vector: false,
                    limit: 1,
                    include_metadata: false,
                    ..Default::default()
                };

                match self.storage.search(tablename, search_query).await {
                    Ok(result) => {
                        // 检查搜索结果中是否包含目标记录ID
                        Ok(result.hits.iter().any(|hit| hit.id == *record_id))
                    }
                    Err(_) => Ok(false),
                }
            }
            Err(_) => Ok(false),
        }
    }

    /// 验证向量一致性
    async fn verify_vector_consistency(
        &self,
        tablename: &str,
        record: &crate::Record,
    ) -> std::result::Result<(), InconsistencyIssue> {
        // 从索引中搜索相同向量，验证是否返回相同记录
        let search_query = crate::types::SearchQuery {
            vector: record.vector.clone(),
            top_k: 1,
            filter: None,
            with_payload: false,
            with_vector: true,
            limit: 1,
            include_metadata: false,
            ..Default::default()
        };

        match self.storage.search(tablename, search_query).await {
            Ok(result) => {
                if let Some(hit) = result.hits.first() {
                    if hit.id == record.id {
                        if let Some(ref index_vector) = hit.vector {
                            // 比较向量值
                            if *index_vector != record.vector {
                                return Err(InconsistencyIssue::VectorMismatch {
                                    record_id: record.id.clone(),
                                    tablename: tablename.to_string(),
                                    storage_vector: Some(record.vector.clone()),
                                    index_vector: Some(index_vector.clone()),
                                });
                            }
                        }
                    }
                }
            }
            Err(_) => {
                // 搜索失败，可能索引中不存在
                return Err(InconsistencyIssue::MissingInIndex {
                    record_id: record.id.clone(),
                    tablename: tablename.to_string(),
                });
            }
        }

        Ok(())
    }

    /// 修复发现的不一致问题
    pub async fn repair_inconsistencies(
        &self,
        tablename: &str,
        issues: &[InconsistencyIssue],
    ) -> crate::Result<RepairReport> {
        let mut repaired = 0;
        let mut failed = 0;
        let repair_start = std::time::Instant::now();

        for issue in issues {
            match self.repair_single_issue(tablename, issue).await {
                Ok(_) => {
                    repaired += 1;
                    info!("Repaired inconsistency: {:?}", issue);
                }
                Err(e) => {
                    failed += 1;
                    error!("Failed to repair inconsistency: {:?}, error: {}", issue, e);
                }
            }
        }

        let repair_duration_ms = repair_start.elapsed().as_millis() as u64;

        Ok(RepairReport {
            total_issues: issues.len(),
            repaired_count: repaired,
            failed_count: failed,
            repair_duration_ms,
        })
    }

    /// 修复单个不一致问题
    async fn repair_single_issue(
        &self,
        tablename: &str,
        issue: &InconsistencyIssue,
    ) -> crate::Result<()> {
        match issue {
            InconsistencyIssue::MissingInIndex { record_id, .. } => {
                // 从存储获取记录，通过 update_record 触发 remove+insert 重建索引
                let record = self.storage.get_record(tablename, record_id).await?;
                let req = crate::types::record::CreateRecordRequest {
                    id: record.id.clone(),
                    vector: record.vector.clone(),
                    sparse_vector: record.sparse_vector.clone(),
                    payload: Some(record.payload.clone()),
                    named_vectors: None,
                };
                self.storage
                    .update_record(tablename, record_id, req)
                    .await?;
                info!(
                    "Repaired: re-indexed record {} in table {}",
                    record_id, tablename
                );
            }
            InconsistencyIssue::MissingInStorage { record_id, .. } => {
                // 存储中不存在该记录，从索引中删除
                self.storage
                    .delete_records(tablename, vec![record_id.clone()])
                    .await?;
                info!(
                    "Repaired: removed orphan index entry for {} in table {}",
                    record_id, tablename
                );
            }
            InconsistencyIssue::VectorMismatch { record_id, .. } => {
                // 向量不一致：以存储中的向量为准，强制更新索引
                let record = self.storage.get_record(tablename, record_id).await?;
                let req = crate::types::record::CreateRecordRequest {
                    id: record.id.clone(),
                    vector: record.vector.clone(),
                    sparse_vector: record.sparse_vector.clone(),
                    payload: Some(record.payload.clone()),
                    named_vectors: None,
                };
                self.storage
                    .update_record(tablename, record_id, req)
                    .await?;
                info!(
                    "Repaired: re-synced vector for record {} in table {}",
                    record_id, tablename
                );
            }
            _ => {
                warn!("Unsupported repair for issue: {:?}", issue);
            }
        }

        Ok(())
    }
}

/// 修复报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairReport {
    pub total_issues: usize,
    pub repaired_count: usize,
    pub failed_count: usize,
    pub repair_duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageConfig;
    use crate::storage::IntelligentIndexedStorage;
    use crate::types::record::CreateRecordRequest;
    use crate::types::table::TableConfig;
    use tempfile::TempDir;

    async fn make_storage() -> std::sync::Arc<IntelligentIndexedStorage> {
        let config = StorageConfig {
            r#type: "memory".to_string(),
            memory: Some(crate::config::MemoryConfig {
                max_size_mb: 64,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        std::sync::Arc::new(IntelligentIndexedStorage::new(&config).await.unwrap())
    }

    fn make_wal(dir: &TempDir) -> std::sync::Arc<crate::wal::WalManager> {
        std::sync::Arc::new(crate::wal::WalManager::new(dir.path()).unwrap())
    }

    #[tokio::test]
    async fn test_consistency_checker_clean_table() {
        let tmp = TempDir::new().unwrap();
        let storage = make_storage().await;
        let wal = make_wal(&tmp);
        let checker = ConsistencyChecker::new(storage.clone(), wal, ConsistencyConfig::default());

        let table_cfg = TableConfig {
            name: "test_consistency".to_string(),
            dimension: 4,
            distance_metric: crate::types::vector::DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
        };
        storage.create_table(table_cfg).await.unwrap();

        // Insert a record
        storage
            .insert_records(
                "test_consistency",
                vec![CreateRecordRequest {
                    id: "r1".to_string(),
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    sparse_vector: None,
                    payload: None,
                    named_vectors: None,
                }],
            )
            .await
            .unwrap();

        let report = checker.check_consistency("test_consistency").await.unwrap();
        // A freshly inserted record should not have inconsistencies
        assert_eq!(
            report.inconsistencies.len(),
            0,
            "Expected no inconsistencies, got: {:?}",
            report.inconsistencies
        );
    }

    #[tokio::test]
    async fn test_consistency_get_all_record_ids() {
        let tmp = TempDir::new().unwrap();
        let storage = make_storage().await;
        let wal = make_wal(&tmp);
        let checker = ConsistencyChecker::new(storage.clone(), wal, ConsistencyConfig::default());

        let table_cfg = TableConfig {
            name: "test_ids".to_string(),
            dimension: 2,
            distance_metric: crate::types::vector::DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
        };
        storage.create_table(table_cfg).await.unwrap();
        storage
            .insert_records(
                "test_ids",
                vec![
                    CreateRecordRequest {
                        id: "a".to_string(),
                        vector: vec![1.0, 0.0],
                        sparse_vector: None,
                        named_vectors: None,
                        payload: None,
                    },
                    CreateRecordRequest {
                        id: "b".to_string(),
                        vector: vec![0.0, 1.0],
                        sparse_vector: None,
                        named_vectors: None,
                        payload: None,
                    },
                ],
            )
            .await
            .unwrap();

        let ids = checker.get_all_record_ids("test_ids").await.unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"b".to_string()));
    }
}
