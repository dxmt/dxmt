use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use bytes::{BufMut, Bytes};
use dashmap::DashMap;
use lru::LruCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::types::{vector::DenseVector, record::{SearchHit, SearchQuery}};

/// Cache key for vector searches
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheKey {
    /// Table name
    pub tablename: String,
    /// Query vector hash
    pub vector_hash: u64,
    /// Top K
    pub top_k: usize,
    /// Search parameters hash
    pub params_hash: u64,
}

impl CacheKey {
    pub fn from_query(tablename: &str, query: &SearchQuery) -> Self {
        let vector_hash = compute_vector_hash(&query.vector);
        let params_hash = compute_params_hash(query);

        Self {
            tablename: tablename.to_string(),
            vector_hash,
            top_k: query.top_k,
            params_hash,
        }
    }
}

/// Cached search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    /// Search hits
    pub hits: Vec<SearchHit>,
    /// Cache timestamp (milliseconds since epoch)
    pub cached_at_ms: u64,
    /// Query execution time (ms)
    pub query_time_ms: u64,
}

/// Multi-level query cache
pub struct QueryCache {
    /// L1 cache: Hot results (in-memory)
    l1_cache: Arc<RwLock<LruCache<CacheKey, CachedResult>>>,
    /// L2 cache: Warm results (compressed in-memory)
    l2_cache: Arc<DashMap<CacheKey, Bytes>>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// L1 cache size (number of entries)
    pub l1_size: usize,
    /// L2 cache size (number of entries)
    pub l2_size: usize,
    /// TTL for cached results
    pub ttl: Duration,
    /// Enable compression for L2 cache
    pub enable_compression: bool,
    /// Minimum query time to cache (ms)
    pub min_query_time_ms: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size: 10_000,
            l2_size: 100_000,
            ttl: Duration::from_secs(300), // 5 minutes
            enable_compression: true,
            min_query_time_ms: 5, // Only cache queries taking > 5ms
        }
    }
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub l1hits: u64,
    pub l1_misses: u64,
    pub l2hits: u64,
    pub l2_misses: u64,
    pub evictions: u64,
    pub total_queries: u64,
}

impl QueryCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            l1_cache: Arc::new(RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(config.l1_size).unwrap()))),
            l2_cache: Arc::new(DashMap::with_capacity(config.l2_size)),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Get cached result
    pub fn get(&self, key: &CacheKey) -> Option<CachedResult> {
        let mut stats = self.stats.write();
        stats.total_queries += 1;

        // Check L1 cache first
        {
            let mut l1 = self.l1_cache.write();
            if let Some(result) = l1.get(key) {
                // Check TTL
                let now_ms = chrono::Utc::now().timestamp_millis() as u64;
                if now_ms.saturating_sub(result.cached_at_ms) < self.config.ttl.as_millis() as u64 {
                    stats.l1hits += 1;
                    return Some(result.clone());
                } else {
                    // Expired, remove it
                    l1.pop(key);
                }
            }
        }

        stats.l1_misses += 1;

        // Check L2 cache
        if let Some(compressed) = self.l2_cache.get(key) {
            if let Ok(result) = self.decompress_result(&compressed) {
                let now_ms = chrono::Utc::now().timestamp_millis() as u64;
                if now_ms.saturating_sub(result.cached_at_ms) < self.config.ttl.as_millis() as u64 {
                    stats.l2hits += 1;

                    // Promote to L1
                    self.promote_to_l1(key.clone(), result.clone());

                    return Some(result);
                } else {
                    // Expired, remove it
                    self.l2_cache.remove(key);
                }
            }
        }

        stats.l2_misses += 1;
        None
    }

    /// Put result in cache
    pub fn put(&self, key: CacheKey, hits: Vec<SearchHit>, query_time_ms: u64) -> crate::Result<()> {
        // Only cache if query time exceeds threshold
        if query_time_ms < self.config.min_query_time_ms {
            return Ok(());
        }

        let result = CachedResult {
            hits,
            cached_at_ms: chrono::Utc::now().timestamp_millis() as u64,
            query_time_ms,
        };

        // Add to L1 cache
        let evicted = {
            let mut l1 = self.l1_cache.write();
            l1.push(key.clone(), result.clone())
        };

        // If evicted from L1, move to L2
        if let Some((evicted_key, evicted_result)) = evicted {
            self.demote_to_l2(evicted_key, evicted_result);
        }
        Ok(())
    }

    /// Clear expired entries
    pub fn evict_expired(&self) {
        let now_ms = chrono::Utc::now().timestamp_millis() as u64;
        let ttl_ms = self.config.ttl.as_millis() as u64;

        // Clear L1 expired entries
        {
            let mut l1 = self.l1_cache.write();
            let expired_keys: Vec<_> = l1
                .iter()
                .filter(|(_, result)| now_ms.saturating_sub(result.cached_at_ms) >= ttl_ms)
                .map(|(key, _)| key.clone())
                .collect();

            for key in expired_keys {
                l1.pop(&key);
                self.stats.write().evictions += 1
            }
        }

        // Clear L2 expired entries
        self.l2_cache.retain(|_, compressed| {
            if let Ok(result) = self.decompress_result(compressed) {
                if now_ms.saturating_sub(result.cached_at_ms) < ttl_ms {
                    true
                } else {
                    self.stats.write().evictions += 1;
                    false
                }
            } else {
                false
            }
        });
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        (*self.stats.read()).clone()
    }

    /// Clear all cache
    pub fn clear(&self) {
        self.l1_cache.write().clear();
        self.l2_cache.clear();
        *self.stats.write() = CacheStats::default()
    }

    /// Promote result from L2 to L1
    fn promote_to_l1(&self, key: CacheKey, result: CachedResult) {
        let evicted = {
            let mut l1 = self.l1_cache.write();
            l1.push(key, result)
        };

        if let Some((evicted_key, evicted_result)) = evicted {
            self.demote_to_l2(evicted_key, evicted_result);
        }
    }

    /// Demote result from L1 to L2
    fn demote_to_l2(&self, key: CacheKey, result: CachedResult) {
        if let Ok(compressed) = self.compress_result(&result) {
            // Check L2 size limit
            if self.l2_cache.len() >= self.config.l2_size {
                // Remove oldest entries (approximate LRU)
                let to_remove = self.config.l2_size / 10; // Remove 10%
                let mut removed = 0;

                self.l2_cache.retain(|_, _| {
                    if removed < to_remove {
                        removed += 1;
                        false
                    } else {
                        true
                    }
                });

                self.stats.write().evictions += removed as u64;
            }

            self.l2_cache.insert(key, compressed);
        }
    }

    /// Compress result for L2 cache
    fn compress_result(&self, result: &CachedResult) -> crate::Result<Bytes> {
        if self.config.enable_compression {
            // Use bincode for serialization
            let serialized = bincode::serialize(result).map_err(|e| crate::Error::Internal(format!("Failed to serialize cache result: {}", e)))?;

            // Use zstd for compression
            let compressed =
                zstd::encode_all(&serialized[..], 3).map_err(|e| crate::Error::Internal(format!("Failed to compress cache result: {}", e)))?;

            Ok(Bytes::from(compressed))
        } else {
            // Just serialize without compression
            let serialized = bincode::serialize(result).map_err(|e| crate::Error::Internal(format!("Failed to serialize cache result: {}", e)))?;
            Ok(Bytes::from(serialized))
        }
    }

    /// Decompress result from L2 cache
    fn decompress_result(&self, compressed: &Bytes) -> crate::Result<CachedResult> {
        if self.config.enable_compression {
            // Decompress with zstd
            let decompressed =
                zstd::decode_all(&compressed[..]).map_err(|e| crate::Error::Internal(format!("Failed to decompress cache result: {}", e)))?;

            // Deserialize with bincode
            bincode::deserialize(&decompressed).map_err(|e| crate::Error::Internal(format!("Failed to deserialize cache result: {}", e)))
        } else {
            // Just deserialize
            bincode::deserialize(compressed).map_err(|e| crate::Error::Internal(format!("Failed to deserialize cache result: {}", e)))
        }
    }
}

/// Compute hash for vector
fn compute_vector_hash(vector: &DenseVector) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Hash vector values with reduced precision for better cache hits
    for &val in vector {
        // Round to 4 decimal places
        let rounded = (val * 10000.0).round() as i32;
        rounded.hash(&mut hasher);
    }

    hasher.finish()
}

/// Compute hash for search parameters
fn compute_params_hash(query: &SearchQuery) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Hash search parameters
    query.top_k.hash(&mut hasher);

    if let Some(ref filter) = query.filter {
        format!("{:?}", filter).hash(&mut hasher);
    }

    hasher.finish()
}

/// Distributed cache for multi-node deployments
pub struct DistributedQueryCache {
    local_cache: Arc<QueryCache>,
    // distributed backend (Redis/Valkey) not yet integrated
}

impl DistributedQueryCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            local_cache: Arc::new(QueryCache::new(config)),
        }
    }

    pub fn get(&self, key: &CacheKey) -> Option<CachedResult> {
        // First check local cache
        if let Some(result) = self.local_cache.get(key) {
            return Some(result);
        }

        None
    }

    pub fn put(&self, key: CacheKey, hits: Vec<SearchHit>, query_time_ms: u64) {
        let _ = self.local_cache.put(key.clone(), hits, query_time_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
use crate::SearchQuery;


    #[test]
    fn test_query_cache() {
        let config = CacheConfig {
            l1_size: 10,
            l2_size: 20,
            ttl: _Duration::from_secs(60),
            enable_compression: true,
            min_query_time_ms: 0,
        };

        let cache = QueryCache::new(config);

        // Create test query
        let query = SearchQuery {
            vector: vec![1.0, 2.0, 3.0],
            top_k: 10,
            hnsw_ef: Some(100),
            filter: None,
            ..Default::default()
        };

        let key = CacheKey::from_query("test_table", &query);

        // Test cache miss
        assert!(cache.get(&key).is_none());

        // Put result in cache
        let hits = vec![SearchHit {
            id: "1".to_string(),
            distance: 0.1,
            score: 0.9,
        }];

        cache.put(key.clone(), hits.clone(), 10);

        // Test cache hit
        let cached = cache.get(&key)?;
        assert_eq!(cached.hits.len(), 1);
        assert_eq!(cached.hits[0].id, "1");

        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.l1_hits, 1);
        assert_eq!(stats.l1_misses, 1);
    }
}
