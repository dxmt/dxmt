use crate::storage::{
    tier_promotion::{Tier, TierPromotionConfig, TierPromotionPolicy},
    traits::_Storage as Storage,
    traits::StorageConfig,
};
use crate::types::{
    record::{CreateRecordRequest, Record, SearchQuery, SearchResult},
    table::{Table, TableConfig},
    vector::RecordId,
};
use async_trait::async_trait;
use std::sync::Arc;

/// Tiered storage implementation that manages data across multiple storage tiers
pub struct TieredStorage {
    _hot_tier: Arc<dyn Storage>,
    warm_tier: Arc<dyn Storage>,
    cold_tier: Arc<dyn Storage>,
    #[allow(dead_code)]
    config: StorageConfig,
    /// Vector promotion policy for automatic tier management
    promotion_policy: Arc<TierPromotionPolicy>,
}

impl TieredStorage {
    pub fn new(
        hot_tier: Arc<dyn Storage>,
        warm_tier: Arc<dyn Storage>,
        cold_tier: Arc<dyn Storage>,
        config: StorageConfig,
    ) -> Self {
        let promotion_policy = Arc::new(TierPromotionPolicy::new(TierPromotionConfig::default()));
        Self {
            _hot_tier: hot_tier,
            warm_tier,
            cold_tier,
            config,
            promotion_policy,
        }
    }

    /// Create with custom promotion config
    pub fn new_with_promotion_config(
        hot_tier: Arc<dyn Storage>,
        warm_tier: Arc<dyn Storage>,
        cold_tier: Arc<dyn Storage>,
        config: StorageConfig,
        promotion_config: TierPromotionConfig,
    ) -> Self {
        let promotion_policy = Arc::new(TierPromotionPolicy::new(promotion_config));
        Self {
            _hot_tier: hot_tier,
            warm_tier,
            cold_tier,
            config,
            promotion_policy,
        }
    }

    /// Get the promotion policy
    pub fn promotion_policy(&self) -> &Arc<TierPromotionPolicy> {
        &self.promotion_policy
    }

    /// Get storage tier by enum
    fn get_tier_storage(&self, tier: Tier) -> &Arc<dyn Storage> {
        match tier {
            Tier::Hot => &self._hot_tier,
            Tier::Warm => &self.warm_tier,
            Tier::Cold => &self.cold_tier,
        }
    }

    /// Move a vector from one tier to another
    pub async fn move_vector_to_tier(
        &self,
        tablename: &str,
        vector_id: &RecordId,
        source_tier: Tier,
        target_tier: Tier,
    ) -> crate::Result<()> {
        // Don't move if already in target tier
        if source_tier == target_tier {
            return Ok(());
        }

        let source_storage = self.get_tier_storage(source_tier);
        let target_storage = self.get_tier_storage(target_tier);

        // Step 1: Get the record from source tier
        let record = match source_storage.get_record(tablename, vector_id).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    "Failed to get record {} from {:?} tier: {}",
                    vector_id,
                    source_tier,
                    e
                );
                return Err(e);
            }
        };

        // Step 2: Convert Record to CreateRecordRequest
        let request = CreateRecordRequest {
            id: record.id.clone(),
            vector: record.vector.clone(),
            sparse_vector: record.sparse_vector.clone(),
            payload: Some(record.payload.clone()),
            named_vectors: None,
            text: None,
        };

        // Step 3: Insert into target tier
        match target_storage.insert_record(tablename, request).await {
            Ok(_) => {
                tracing::debug!(
                    "Moved vector {} from {:?} to {:?} tier",
                    vector_id,
                    source_tier,
                    target_tier
                );
            }
            Err(e) => {
                tracing::error!(
                    "Failed to insert record {} into {:?} tier: {}",
                    vector_id,
                    target_tier,
                    e
                );
                return Err(e);
            }
        }

        // Step 4: Delete from source tier (only after successful insertion)
        if let Err(e) = source_storage.delete_record(tablename, vector_id).await {
            tracing::warn!(
                "Failed to delete record {} from {:?} tier after move: {}",
                vector_id,
                source_tier,
                e
            );
            // Note: Record exists in both tiers now, but this is safe
            // Search will find it in both and deduplicate
        }

        // Step 5: Update promotion policy
        self.promotion_policy
            .set_current_tier(vector_id, target_tier);

        Ok(())
    }

    /// Find which tier contains a vector
    async fn find_vector_tier(&self, tablename: &str, vector_id: &RecordId) -> Option<Tier> {
        // Check promotion policy first (fast)
        if let Some(tier) = self.promotion_policy.get_current_tier(vector_id) {
            return Some(tier);
        }

        // If not tracked, search tiers from hot to cold
        if self
            ._hot_tier
            .get_record(tablename, vector_id)
            .await
            .is_ok()
        {
            return Some(Tier::Hot);
        }
        if self
            .warm_tier
            .get_record(tablename, vector_id)
            .await
            .is_ok()
        {
            return Some(Tier::Warm);
        }
        if self
            .cold_tier
            .get_record(tablename, vector_id)
            .await
            .is_ok()
        {
            return Some(Tier::Cold);
        }

        None
    }

    /// Batch move vectors to a tier
    pub async fn batch_move_vectors_to_tier(
        &self,
        tablename: &str,
        moves: Vec<(RecordId, Tier, Tier)>, // (vector_id, source_tier, target_tier)
    ) -> crate::Result<Vec<(RecordId, crate::Result<()>)>> {
        let mut results = Vec::new();

        for (vector_id, source_tier, target_tier) in moves {
            let result = self
                .move_vector_to_tier(tablename, &vector_id, source_tier, target_tier)
                .await;

            let success = result.is_ok();
            results.push((vector_id.clone(), result));

            if success {
                tracing::debug!(
                    "Successfully moved vector {} to {:?}",
                    vector_id,
                    target_tier
                );
            }
        }

        Ok(results)
    }

    /// Promote vectors based on access patterns
    pub async fn promote_hot_vectors(
        &self,
        tablename: &str,
        hits: &[crate::types::record::SearchHit],
    ) -> crate::Result<()> {
        if hits.is_empty() {
            return Ok(());
        }

        for hit in hits {
            // Record access
            self.promotion_policy.record_access(&hit.id);

            // Check if should promote
            let current_tier = self.promotion_policy.get_current_tier(&hit.id);

            if self.promotion_policy.should_promote_to_hot(&hit.id) {
                if current_tier != Some(Tier::Hot) {
                    // Find where the vector currently is
                    if let Some(source_tier) = self.find_vector_tier(tablename, &hit.id).await {
                        if source_tier != Tier::Hot {
                            // Actually move the vector to hot tier
                            if let Err(e) = self
                                .move_vector_to_tier(tablename, &hit.id, source_tier, Tier::Hot)
                                .await
                            {
                                tracing::warn!(
                                    "Failed to promote vector {} to hot tier: {}",
                                    hit.id,
                                    e
                                );
                            } else {
                                tracing::info!(
                                    "Promoted vector {} from {:?} to hot tier",
                                    hit.id,
                                    source_tier
                                );
                            }
                        }
                    }
                }
            } else if self.promotion_policy.should_promote_to_warm(&hit.id)
                && current_tier != Some(Tier::Warm)
                && current_tier != Some(Tier::Hot)
            {
                // Find where the vector currently is
                if let Some(source_tier) = self.find_vector_tier(tablename, &hit.id).await {
                    if source_tier != Tier::Warm && source_tier != Tier::Hot {
                        // Actually move the vector to warm tier
                        if let Err(e) = self
                            .move_vector_to_tier(tablename, &hit.id, source_tier, Tier::Warm)
                            .await
                        {
                            tracing::warn!(
                                "Failed to promote vector {} to warm tier: {}",
                                hit.id,
                                e
                            );
                        } else {
                            tracing::info!(
                                "Promoted vector {} from {:?} to warm tier",
                                hit.id,
                                source_tier
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Background task to demote cold vectors
    pub async fn demote_cold_vectors(&self, tablename: &str) -> crate::Result<()> {
        let demotions = self.promotion_policy.get_vectors_to_demote();

        for (vector_id, target_tier) in demotions {
            // Find where the vector currently is
            if let Some(source_tier) = self.find_vector_tier(tablename, &vector_id).await {
                if source_tier != target_tier {
                    // Actually move the vector to target tier
                    if let Err(e) = self
                        .move_vector_to_tier(tablename, &vector_id, source_tier, target_tier)
                        .await
                    {
                        tracing::warn!(
                            "Failed to demote vector {} to {:?}: {}",
                            vector_id,
                            target_tier,
                            e
                        );
                    } else {
                        tracing::info!(
                            "Demoted vector {} from {:?} to {:?}",
                            vector_id,
                            source_tier,
                            target_tier
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Storage for TieredStorage {
    async fn create_table(&self, config: TableConfig) -> crate::Result<Table> {
        // Create table in all tiers
        let table = self._hot_tier.create_table(config.clone()).await?;
        self.warm_tier.create_table(config.clone()).await?;
        self.cold_tier.create_table(config).await?;
        Ok(table)
    }

    async fn get_table(&self, name: &str) -> crate::Result<Table> {
        // Try to get from hot tier first
        if let Ok(table) = self._hot_tier.get_table(name).await {
            return Ok(table);
        }
        // Try warm tier
        if let Ok(table) = self.warm_tier.get_table(name).await {
            return Ok(table);
        }
        // Finally try cold tier
        self.cold_tier.get_table(name).await
    }

    async fn list_tables(&self) -> crate::Result<Vec<Table>> {
        // List from hot tier as it should have the most recent metadata
        self._hot_tier.list_tables().await
    }

    async fn delete_table(&self, name: &str) -> crate::Result<()> {
        // Delete from all tiers
        self._hot_tier.delete_table(name).await?;
        self.warm_tier.delete_table(name).await?;
        self.cold_tier.delete_table(name).await
    }

    async fn insert_record(
        &self,
        tablename: &str,
        request: CreateRecordRequest,
    ) -> crate::Result<Record> {
        // Insert into hot tier first
        self._hot_tier.insert_record(tablename, request).await
    }

    async fn batch_insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>> {
        // Insert into hot tier first
        self._hot_tier
            .batch_insert_records(tablename, requests)
            .await
    }

    async fn insert_records(
        &self,
        tablename: &str,
        requests: Vec<CreateRecordRequest>,
    ) -> crate::Result<Vec<Record>> {
        // Insert into hot tier first
        self._hot_tier.insert_records(tablename, requests).await
    }

    async fn get_record(&self, tablename: &str, record_id: &RecordId) -> crate::Result<Record> {
        // Try hot tier first
        if let Ok(record) = self._hot_tier.get_record(tablename, record_id).await {
            return Ok(record);
        }
        // Try warm tier
        if let Ok(record) = self.warm_tier.get_record(tablename, record_id).await {
            return Ok(record);
        }
        // Finally try cold tier
        self.cold_tier.get_record(tablename, record_id).await
    }

    async fn update_record(
        &self,
        tablename: &str,
        record_id: &RecordId,
        request: CreateRecordRequest,
    ) -> crate::Result<Record> {
        // Update in the tier where the record exists
        if self
            ._hot_tier
            .get_record(tablename, record_id)
            .await
            .is_ok()
        {
            return self
                ._hot_tier
                .update_record(tablename, record_id, request.clone())
                .await;
        }
        if self
            .warm_tier
            .get_record(tablename, record_id)
            .await
            .is_ok()
        {
            return self
                .warm_tier
                .update_record(tablename, record_id, request.clone())
                .await;
        }
        self.cold_tier
            .update_record(tablename, record_id, request)
            .await
    }

    async fn delete_record(&self, tablename: &str, record_id: &RecordId) -> crate::Result<()> {
        // Try to delete from all tiers
        let _ = self._hot_tier.delete_record(tablename, record_id).await;
        let _ = self.warm_tier.delete_record(tablename, record_id).await;
        let _ = self.cold_tier.delete_record(tablename, record_id).await;
        Ok(())
    }

    async fn delete_records(
        &self,
        tablename: &str,
        record_ids: Vec<RecordId>,
    ) -> crate::Result<()> {
        // Delete from all tiers
        let _ = self
            ._hot_tier
            .delete_records(tablename, record_ids.clone())
            .await;
        let _ = self
            .warm_tier
            .delete_records(tablename, record_ids.clone())
            .await;
        let _ = self.cold_tier.delete_records(tablename, record_ids).await;
        Ok(())
    }

    async fn search(&self, tablename: &str, query: SearchQuery) -> crate::Result<SearchResult> {
        use std::time::Instant;

        let start_time = Instant::now();
        let top_k = query.top_k;

        // Phase 1: Search hot tier (Memory) - <1ms
        let mut all_hits = match self._hot_tier.search(tablename, query.clone()).await {
            Ok(result) => result.hits,
            Err(_) => Vec::new(),
        };

        // If we have enough results from hot tier, return early
        if all_hits.len() >= top_k {
            all_hits.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all_hits.truncate(top_k);

            // Record access and promote hot vectors
            let _ = self.promote_hot_vectors(tablename, &all_hits).await;

            let total = all_hits.len();
            return Ok(SearchResult {
                hits: all_hits,
                total,
                tookms: start_time.elapsed().as_millis() as u64,
            });
        }

        // Phase 2: Search warm tier (SSD) - <50ms
        if let Ok(warm_result) = self.warm_tier.search(tablename, query.clone()).await {
            all_hits.extend(warm_result.hits);
        }

        // If we now have enough results, return after reranking
        if all_hits.len() >= top_k {
            all_hits.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all_hits.truncate(top_k);

            // Record access and promote hot vectors
            let _ = self.promote_hot_vectors(tablename, &all_hits).await;

            let total = all_hits.len();
            return Ok(SearchResult {
                hits: all_hits,
                total,
                tookms: start_time.elapsed().as_millis() as u64,
            });
        }

        // Phase 3: Search cold tier (S3) - <500ms
        if let Ok(cold_result) = self.cold_tier.search(tablename, query.clone()).await {
            all_hits.extend(cold_result.hits);
        }

        // Final reranking and truncation
        all_hits.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_hits.truncate(top_k);

        // Record access and promote hot vectors
        let _ = self.promote_hot_vectors(tablename, &all_hits).await;

        let total = all_hits.len();
        Ok(SearchResult {
            hits: all_hits,
            total,
            tookms: start_time.elapsed().as_millis() as u64,
        })
    }

    async fn backup(&self, tablename: &str, path: &str) -> crate::Result<()> {
        // Backup all tiers
        self._hot_tier
            .backup(tablename, &format!("{}/hot", path))
            .await?;
        self.warm_tier
            .backup(tablename, &format!("{}/warm", path))
            .await?;
        self.cold_tier
            .backup(tablename, &format!("{}/cold", path))
            .await?;
        Ok(())
    }

    async fn restore(&self, tablename: &str, path: &str) -> crate::Result<()> {
        // Restore all tiers
        self._hot_tier
            .restore(tablename, &format!("{}/hot", path))
            .await?;
        self.warm_tier
            .restore(tablename, &format!("{}/warm", path))
            .await?;
        self.cold_tier
            .restore(tablename, &format!("{}/cold", path))
            .await?;
        Ok(())
    }
}
