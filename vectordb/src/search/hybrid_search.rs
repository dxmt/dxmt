use crate::search::VectorIndex;
use crate::types::{vector::{RecordId, DenseVector}, Payload};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tantivy::{schema::*, Index, TantivyDocument};

// ---------------------------------------------------------------------------
// Distribution-Based Score Fusion (DBSF)
//
// Normalises each result list independently using (score - mean) / std, then
// clips to [-3σ, +3σ] and rescales to [0, 1] before summing. This removes the
// unit mismatch between vector cosine/distance scores and BM25 TF-IDF scores,
// making the fusion parameter-free (no alpha tuning required).
//
// Reference: Qdrant "Distribution-Based Score Fusion" blog post / implementation.
// ---------------------------------------------------------------------------

/// Normalise a list of (id, raw_score) to the [0, 1] range using mean/std.
/// Higher raw score ⟹ higher normalised score (assumes similarity, not distance).
fn dbsf_normalize(scores: &[(RecordId, f32)]) -> HashMap<RecordId, f32> {
    if scores.is_empty() {
        return HashMap::new();
    }

    let n = scores.len() as f32;
    let mean = scores.iter().map(|(_, s)| s).sum::<f32>() / n;
    let variance = scores.iter().map(|(_, s)| (s - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt().max(1e-8); // avoid division by zero

    // Clip to [-3σ, +3σ] then rescale to [0, 1]
    let low = mean - 3.0 * std;
    let high = mean + 3.0 * std;
    let range = (high - low).max(1e-8);

    scores
        .iter()
        .map(|(id, s)| {
            let s = *s;
            let clipped = s.max(low).min(high);
            let normalised = (clipped - low) / range;
            (id.clone(), normalised)
        })
        .collect()
}

/// Simple search result for hybrid search
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: RecordId,
    pub score: f32,
    pub payload: Option<Payload>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Eq(String, serde_json::Value),
    Neq(String, serde_json::Value),
    Gt(String, serde_json::Value),
    Gte(String, serde_json::Value),
    Lt(String, serde_json::Value),
    Lte(String, serde_json::Value),
    In(String, Vec<serde_json::Value>),
    Nin(String, Vec<serde_json::Value>),
    Contains(String, String),
    Regex(String, String),
    And(Vec<Box<FilterOperator>>),
    Or(Vec<Box<FilterOperator>>),
}

pub struct HybridSearchEngine {
    vector_index: Box<dyn VectorIndex>,
    text_index: Index,
    #[allow(dead_code)]
    metadata_engine: HashMap<String, serde_json::Value>, // Simplified metadata engine
}

impl HybridSearchEngine {
    /// Create a new hybrid search engine
    /// Note: This is a simplified constructor for basic functionality
    pub fn new(
        vector_index: Box<dyn VectorIndex>,
        text_index: Index,
    ) -> Self {
        Self {
            vector_index,
            text_index,
            metadata_engine: HashMap::new(),
        }
    }

    pub async fn search(
        &self,
        query_vector: Option<&DenseVector>,
        text_query: Option<&str>,
        filters: Option<Vec<FilterOperator>>,
        k: usize,
        alpha: f32, // weight for vector search (0-1)
    ) -> crate::Result<Vec<SearchResult>> {
        let mut scores: HashMap<RecordId, f32> = HashMap::new();

        // 1. Vector search
        if let Some(vector) = query_vector {

            let vector_results = self.vector_index.search(vector, k * 3).await?;
            for (id, score) in vector_results {
                scores.insert(id.clone(), score * alpha);

        }
        }

        // 2. Text search
        if let Some(text) = text_query {

            let text_results = self.search_text(text, k * 3).await?;
            for result in text_results {
                let score = scores.get(&result.id).unwrap_or(&0.0) + result.score * (1.0 - alpha);
                scores.insert(result.id, score);

        }
        }

        // 3. Apply filters
        let filtered_ids = if let Some(filters) = filters {
            self.apply_filters(&filters, scores.keys().cloned().collect()).await?
        } else {
            scores.keys().cloned().collect()
        };

        // 4. Sort and return top-k
        let mut results: Vec<SearchResult> = filtered_ids
            .into_iter()
            .filter_map(|id| {
                scores.get(&id).map(|&score| SearchResult {
                    id,
                    score,
                    payload: None,
                })
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        // 5. Load payloads
        for result in &mut results {
            result.payload = self.load_payload(&result.id).await?;
        }

        Ok(results)
    }

    async fn search_text(
        &self,
        query: &str,
        limit: usize) -> crate::Result<Vec<SearchResult>> {
        let reader = self.text_index.reader()
            .map_err(|e| crate::Error::Internal(format!("Failed to get reader: {}", e) ))?;
        let searcher = reader.searcher();

        let query_parser = tantivy::query::QueryParser::for_index(&self.text_index, vec![]);
        let query = query_parser.parse_query(query)
            .map_err(|e| crate::Error::Internal(format!("Failed to parse query: {}", e) ))?;

        let top_docs = searcher.search(&query, &tantivy::collector::TopDocs::with_limit(limit))
            .map_err(|e| crate::Error::Internal(format!("Search failed: {}", e) ))?;

        let mut results = Vec::new();
        for (_score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)
                .map_err(|e| crate::Error::Internal(format!("Failed to get doc: {}", e) ))?;
            if let Some(id_field) = doc.get_first(self.text_index.schema().get_field("id").map_err(|_| crate::Error::Internal("Operation failed".to_string()))?)
            {
                if let Some(id) = id_field.as_str() {
                    results.push(SearchResult {
                        id: RecordId::from(id),
                        score: _score,
                        payload: None,
                    })
                }
            }
        }

        Ok(results)
    }

    async fn apply_filters(
        &self,
        _filters: &[FilterOperator],
        candidates: Vec<RecordId>) -> crate::Result<Vec<RecordId>> {
        // This is a placeholder for the actual metadata engine.
        // In a real scenario, this would call a separate metadata engine.
        // For now, it just returns all candidates.
        Ok(candidates)
    }

    async fn load_payload(
        &self,
        _id: &RecordId) -> crate::Result<Option<serde_json::Value>> {
        // This is a placeholder for loading payloads.
        // In a real scenario, this would fetch the payload from a storage backend.
        Ok(None)
    }
}

/// Fusion strategy for combining vector and text result lists
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Linear weighted sum: `alpha * vector_score + (1-alpha) * text_score`.
    /// Requires careful alpha tuning because scores are on different scales.
    Linear { alpha: f32 },

    /// Reciprocal Rank Fusion: score = Σ 1/(k + rank).
    /// Ignores raw score magnitude; robust but loses signal from score gaps.
    Rrf { k: i32 },

    /// Distribution-Based Score Fusion: normalises each list to N(0,1) then
    /// clips to [-3σ,+3σ] and rescales to [0,1] before summing.
    /// Parameter-free — no alpha tuning needed.
    #[default]
    Dbsf,
}

impl HybridSearchEngine {
    /// Hybrid search with configurable fusion strategy.
    ///
    /// Unlike `search()` (which is linear-only), this method supports DBSF and RRF
    /// and handles the normalisation internally.
    pub async fn search_hybrid(
        &self,
        query_vector: Option<&DenseVector>,
        text_query: Option<&str>,
        filters: Option<Vec<FilterOperator>>,
        k: usize,
        fusion: FusionStrategy,
    ) -> crate::Result<Vec<SearchResult>> {
        let fetch_k = k * 3; // over-fetch for fusion quality

        // --- collect raw vector results ---
        let vector_raw: Vec<(RecordId, f32)> = if let Some(v) = query_vector {
            self.vector_index.search(v, fetch_k).await?
        } else {
            vec![]
        };

        // --- collect raw text results ---
        let text_raw: Vec<(RecordId, f32)> = if let Some(t) = text_query {
            self.search_text(t, fetch_k)
                .await?
                .into_iter()
                .map(|r| (r.id, r.score))
                .collect()
        } else {
            vec![]
        };

        let merged_scores: HashMap<RecordId, f32> = match fusion {
            FusionStrategy::Linear { alpha } => {
                let mut scores: HashMap<RecordId, f32> = HashMap::new();
                for (id, s) in &vector_raw {
                    *scores.entry(id.clone()).or_insert(0.0) += alpha * s;
                }
                for (id, s) in &text_raw {
                    *scores.entry(id.clone()).or_insert(0.0) += (1.0 - alpha) * s;
                }
                scores
            }

            FusionStrategy::Rrf { k: rrf_k } => {
                let mut scores: HashMap<RecordId, f32> = HashMap::new();
                for (rank, (id, _)) in vector_raw.iter().enumerate() {
                    *scores.entry(id.clone()).or_insert(0.0) +=
                        1.0 / (rrf_k as f32 + rank as f32 + 1.0);
                }
                for (rank, (id, _)) in text_raw.iter().enumerate() {
                    *scores.entry(id.clone()).or_insert(0.0) +=
                        1.0 / (rrf_k as f32 + rank as f32 + 1.0);
                }
                scores
            }

            FusionStrategy::Dbsf => {
                let vec_norm = dbsf_normalize(&vector_raw);
                let txt_norm = dbsf_normalize(&text_raw);

                let mut scores: HashMap<RecordId, f32> = HashMap::new();
                for (id, s) in vec_norm {
                    *scores.entry(id).or_insert(0.0) += s;
                }
                for (id, s) in txt_norm {
                    *scores.entry(id).or_insert(0.0) += s;
                }
                scores
            }
        };

        // --- apply filters ---
        let ids: Vec<RecordId> = if let Some(filters) = filters {
            self.apply_filters(&filters, merged_scores.keys().cloned().collect())
                .await?
        } else {
            merged_scores.keys().cloned().collect()
        };

        // --- sort & truncate ---
        let mut results: Vec<SearchResult> = ids
            .into_iter()
            .filter_map(|id| {
                merged_scores
                    .get(&id)
                    .map(|&score| SearchResult { id, score, payload: None })
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        for result in &mut results {
            result.payload = self.load_payload(&result.id).await?;
        }

        Ok(results)
    }
}

// Reranking strategies
pub enum RerankStrategy {
    RRF { k: i32 }, // Reciprocal Rank Fusion
    Linear { vector_weight: f32 },
    CrossEncoder { model: String },
}

impl HybridSearchEngine {
    pub async fn rerank(
        &self,
        results: Vec<SearchResult>,
        strategy: RerankStrategy) -> crate::Result<Vec<SearchResult>> {
        Ok(match strategy {
            RerankStrategy::RRF { k } => self.rerank_rrf(results, k)?,
            RerankStrategy::Linear { vector_weight } => {
                self.rerank_linear(results, vector_weight).await?
            }
            RerankStrategy::CrossEncoder { model } => {
                self.rerank_cross_encoder(results, &model).await?
            }
        })
    }

    fn rerank_rrf(
        &self,
        results: Vec<SearchResult>,
        k: i32) -> crate::Result<Vec<SearchResult>> {
        // Implement Reciprocal Rank Fusion
        let mut rrf_scores: HashMap<RecordId, f32> = HashMap::new();

        for (rank, result) in results.iter().enumerate() {
            let score = 1.0 / (k as f32 + rank as f32 + 1.0);
            rrf_scores.insert(result.id.clone(), score);
        }

        let mut reranked: Vec<_> = rrf_scores
            .into_iter()
            .map(|(id, score)| SearchResult {
                id,
                score,
                payload: None,
            })
            .collect();

        reranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(reranked)
    }

    async fn rerank_linear(
        &self,
        results: Vec<SearchResult>,
        vector_weight: f32) -> crate::Result<Vec<SearchResult>> {
        let mut reranked: Vec<_> = results
            .into_iter()
            .map(|result| SearchResult {
                id: result.id,
                score: result.score * vector_weight,
                payload: result.payload,
            })
            .collect();
        reranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(reranked)
    }

    async fn rerank_cross_encoder(
        &self,
        results: Vec<SearchResult>,
        _model_name: &str) -> crate::Result<Vec<SearchResult>> {
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbsf_normalize_empty() {
        let result = dbsf_normalize(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dbsf_normalize_range() {
        let scores: Vec<(RecordId, f32)> = vec![
            ("a".to_string(), 10.0),
            ("b".to_string(), 20.0),
            ("c".to_string(), 30.0),
            ("d".to_string(), 40.0),
            ("e".to_string(), 50.0),
        ];
        let norm = dbsf_normalize(&scores);

        // All normalised values must be in [0, 1]
        for (_, v) in &norm {
            assert!(*v >= 0.0 && *v <= 1.0, "value {v} out of [0,1]");
        }

        // Higher raw score → higher normalised score
        assert!(norm["e"] > norm["a"]);
        assert!(norm["d"] > norm["b"]);
    }

    #[test]
    fn test_dbsf_normalize_uniform_scores() {
        // All same score — std ≈ 0, should not panic
        let scores: Vec<(RecordId, f32)> = vec![
            ("a".to_string(), 5.0),
            ("b".to_string(), 5.0),
            ("c".to_string(), 5.0),
        ];
        let norm = dbsf_normalize(&scores);
        for (_, v) in &norm {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dbsf_fusion_dominance() {
        // A result that tops BOTH vector and text lists should win the fused ranking
        let vector_raw: Vec<(RecordId, f32)> = vec![
            ("best".to_string(), 1.0),
            ("mid".to_string(), 0.6),
            ("low".to_string(), 0.3),
        ];
        let text_raw: Vec<(RecordId, f32)> = vec![
            ("best".to_string(), 10.0),
            ("only_text".to_string(), 8.0),
            ("mid".to_string(), 5.0),
        ];

        let vec_norm = dbsf_normalize(&vector_raw);
        let txt_norm = dbsf_normalize(&text_raw);

        let mut scores: HashMap<RecordId, f32> = HashMap::new();
        for (id, s) in vec_norm { *scores.entry(id).or_insert(0.0) += s; }
        for (id, s) in txt_norm { *scores.entry(id).or_insert(0.0) += s; }

        let best_score = scores["best"];
        for (id, &s) in &scores {
            if id != "best" {
                assert!(best_score >= s, "'best' should have the highest fused score");
            }
        }
    }

    #[test]
    fn test_fusion_strategy_default_is_dbsf() {
        assert!(matches!(FusionStrategy::default(), FusionStrategy::Dbsf));
    }
}
