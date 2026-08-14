use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::types::{RecordId, DenseVector};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    pub alpha: f32,
    pub text_weight: f32,
    pub vector_weight: f32,
    pub graph_weight: f32,
    pub rerank_top_k: usize,
    pub fusion_method: FusionMethod,
    /// RRF parameter k (typically 60)
    pub rrf_k: f32,
    /// Enable score normalization
    pub normalize_scores: bool,
    /// Minimum score threshold
    pub min_score_threshold: f32,
    /// Enable adaptive weighting based on query characteristics
    pub adaptive_weighting: bool,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            text_weight: 0.4,
            vector_weight: 0.6,
            graph_weight: 0.0,
            rerank_top_k: 100,
            fusion_method: FusionMethod::OptimizedRRF,
            rrf_k: 60.0,
            normalize_scores: true,
            min_score_threshold: 0.0,
            adaptive_weighting: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionMethod {
    /// Simple weighted sum of normalized scores
    WeightedSum,
    /// Standard Reciprocal Rank Fusion
    ReciprocalRankFusion,
    /// Optimized RRF with score normalization and adaptive parameters
    OptimizedRRF,
    /// Borda count voting system
    BordaCount,
    /// Maximum score combination
    CombMax,
    /// Sum of scores combination
    CombSum,
    /// Harmonic mean fusion
    HarmonicMean,
    /// Geometric mean fusion
    GeometricMean,
    /// Convex combination with learned weights
    ConvexCombination,
}

/// Filter-accepting vector index adapter used only by HybridSearchEngine.
/// This is distinct from `index_traits::VectorIndex` which is the canonical trait.
#[async_trait]
pub trait HybridVectorIndex: Send + Sync {
    async fn search(
        &self,
        query: &DenseVector,
        k: usize,
        filter: Option<Box<dyn Fn(&RecordId) -> bool + Send + Sync>>,
    ) -> Result<Vec<VectorSearchResult>, Box<dyn std::error::Error>>;
}

#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub id: RecordId,
    pub distance: f32,
}

pub struct HybridSearchEngine {
    bm25_index: Arc<BM25Index>,
    vector_index: Arc<dyn HybridVectorIndex>,
    config: HybridSearchConfig,
}

#[derive(Debug, Clone)]
pub struct HybridSearchQuery {
    pub text_query: Option<String>,
    pub vector_query: Option<DenseVector>,
    pub graph_constraints: Option<GraphConstraints>,
    pub limit: usize,
    pub offset: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConstraints {
    pub start_nodes: Vec<RecordId>,
    pub max_distance: usize,
    pub edge_labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub id: RecordId,
    pub score: f32,
    pub text_score: Option<f32>,
    pub vector_score: Option<f32>,
    pub graph_score: Option<f32>,
    pub explanation: SearchExplanation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchExplanation {
    pub fusion_method: FusionMethod,
    pub weights_used: HashMap<String, f32>,
    pub raw_scores: HashMap<String, f32>,
    pub normalized_scores: HashMap<String, f32>,
}

impl HybridSearchEngine {
    pub fn new(
        bm25_index: Arc<BM25Index>,
        vector_index: Arc<dyn HybridVectorIndex>,
        config: HybridSearchConfig,
    ) -> Self {
        Self {
            bm25_index,
            vector_index,
            config,
        }
    }

    pub async fn search(&self, query: HybridSearchQuery) -> Result<Vec<HybridSearchResult>, Box<dyn std::error::Error>> {
        let mut all_results: HashMap<RecordId, HybridScores> = HashMap::new();

        let retrieval_limit = query.limit + query.offset + self.config.rerank_top_k;

        if let Some(text_query) = &query.text_query {
            let text_results = self.bm25_index.search(text_query, retrieval_limit);

            for (id, score) in text_results {
                all_results.entry(id)
                    .or_insert_with(HybridScores::default)
                    .text_score = Some(score);
            }
        }

        if let Some(vector_query) = &query.vector_query {
            let vector_results = self.vector_index
                .search(vector_query, retrieval_limit, None)
                .await?;

            for result in vector_results {
                let similarity = 1.0 / (1.0 + result.distance);

                all_results.entry(result.id)
                    .or_insert_with(HybridScores::default)
                    .vector_score = Some(similarity);
            }
        }

        if let Some(graph_constraints) = &query.graph_constraints {
            let graph_results = self.search_graph(graph_constraints, retrieval_limit).await?;

            for (id, score) in graph_results {
                all_results.entry(id)
                    .or_insert_with(HybridScores::default)
                    .graph_score = Some(score);
            }
        }

        let mut fused_results = self.apply_fusion(all_results);

        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        if query.offset > 0 {
            fused_results = fused_results.into_iter().skip(query.offset).collect();
        }

        fused_results.truncate(query.limit);

        Ok(fused_results)
    }

    fn apply_fusion(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        match self.config.fusion_method {
            FusionMethod::WeightedSum => self.weighted_sum_fusion(results),
            FusionMethod::ReciprocalRankFusion => self.reciprocal_rank_fusion(results),
            FusionMethod::BordaCount => self.borda_count_fusion(results),
            FusionMethod::CombMax => self.comb_max_fusion(results),
            FusionMethod::CombSum => self.comb_sum_fusion(results),
            FusionMethod::OptimizedRRF => self.reciprocal_rank_fusion(results), // Fallback to RRF
            FusionMethod::HarmonicMean => self.weighted_sum_fusion(results), // Fallback to weighted sum
            FusionMethod::GeometricMean => self.weighted_sum_fusion(results), // Fallback to weighted sum
            FusionMethod::ConvexCombination => self.weighted_sum_fusion(results), // Fallback to weighted sum
        }
    }

    fn weighted_sum_fusion(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        results.into_iter().map(|(id, scores)| {
            let text_normalized = scores.text_score
                .map(|s| self.normalize_score(s, 0.0, 100.0))
                .unwrap_or(0.0);

            let vector_normalized = scores.vector_score
                .map(|s| self.normalize_score(s, 0.0, 1.0))
                .unwrap_or(0.0);

            let graph_normalized = scores.graph_score
                .map(|s| self.normalize_score(s, 0.0, 1.0))
                .unwrap_or(0.0);

            let total_weight =
                (if scores.text_score.is_some() { self.config.text_weight } else { 0.0 }) +
                (if scores.vector_score.is_some() { self.config.vector_weight } else { 0.0 }) +
                (if scores.graph_score.is_some() { self.config.graph_weight } else { 0.0 });

            let final_score = if total_weight > 0.0 {
                (text_normalized * self.config.text_weight +
                 vector_normalized * self.config.vector_weight +
                 graph_normalized * self.config.graph_weight) / total_weight
            } else {
                0.0
            };

            let mut weights_used = HashMap::new();
            let mut raw_scores = HashMap::new();
            let mut normalized_scores = HashMap::new();

            if scores.text_score.is_some() {
                weights_used.insert("text".to_string(), self.config.text_weight);
                raw_scores.insert("text".to_string(), scores.text_score.unwrap());
                normalized_scores.insert("text".to_string(), text_normalized);
            }

            if scores.vector_score.is_some() {
                weights_used.insert("vector".to_string(), self.config.vector_weight);
                raw_scores.insert("vector".to_string(), scores.vector_score.unwrap());
                normalized_scores.insert("vector".to_string(), vector_normalized);
            }

            if scores.graph_score.is_some() {
                weights_used.insert("graph".to_string(), self.config.graph_weight);
                raw_scores.insert("graph".to_string(), scores.graph_score.unwrap());
                normalized_scores.insert("graph".to_string(), graph_normalized);
            }

            HybridSearchResult {
                id,
                score: final_score,
                text_score: scores.text_score,
                vector_score: scores.vector_score,
                graph_score: scores.graph_score,
                explanation: SearchExplanation {
                    fusion_method: FusionMethod::WeightedSum,
                    weights_used,
                    raw_scores,
                    normalized_scores,
                },
            }
        }).collect()
    }

    fn reciprocal_rank_fusion(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        let k = self.config.rrf_k;
        self.compute_rrf_scores(results, k, false)
    }

    fn optimized_rrf(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        let k = self.config.rrf_k;
        self.compute_rrf_scores(results, k, true)
    }

    fn compute_rrf_scores(&self, results: HashMap<RecordId, HybridScores>, k: f32, optimized: bool) -> Vec<HybridSearchResult> {
        let mut text_ranks = HashMap::new();
        let mut vector_ranks = HashMap::new();
        let mut graph_ranks = HashMap::new();

        // Build ranking lists with score-based sorting
        let mut text_sorted: Vec<_> = results.iter()
            .filter_map(|(id, scores)| scores.text_score.map(|s| (id.clone(), s)))
            .collect();
        text_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut vector_sorted: Vec<_> = results.iter()
            .filter_map(|(id, scores)| scores.vector_score.map(|s| (id.clone(), s)))
            .collect();
        vector_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut graph_sorted: Vec<_> = results.iter()
            .filter_map(|(id, scores)| scores.graph_score.map(|s| (id.clone(), s)))
            .collect();
        graph_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks
        for (rank, (id, _)) in text_sorted.iter().enumerate() {
            text_ranks.insert(id.clone(), rank + 1);
        }
        for (rank, (id, _)) in vector_sorted.iter().enumerate() {
            vector_ranks.insert(id.clone(), rank + 1);
        }
        for (rank, (id, _)) in graph_sorted.iter().enumerate() {
            graph_ranks.insert(id.clone(), rank + 1);
        }

        // Compute adaptive weights if enabled
        let (text_weight, vector_weight, graph_weight) = if optimized && self.config.adaptive_weighting {
            self.compute_adaptive_weights(&text_sorted, &vector_sorted, &graph_sorted)
        } else {
            (self.config.text_weight, self.config.vector_weight, self.config.graph_weight)
        };

        let mut rrf_results: Vec<_> = results.into_iter().map(|(id, scores)| {
            let mut rrf_score = 0.0;
            let mut weights_used = HashMap::new();
            let mut raw_scores = HashMap::new();
            let mut normalized_scores = HashMap::new();

            // Text RRF contribution
            if let Some(text_score) = scores.text_score {
                if let Some(&rank) = text_ranks.get(&id) {
                    let contribution = text_weight / (k + rank as f32);
                    rrf_score += contribution;
                    weights_used.insert("text".to_string(), text_weight);
                    raw_scores.insert("text".to_string(), text_score);
                    normalized_scores.insert("text_rrf".to_string(), contribution);
                }
            }

            // Vector RRF contribution
            if let Some(vector_score) = scores.vector_score {
                if let Some(&rank) = vector_ranks.get(&id) {
                    let contribution = vector_weight / (k + rank as f32);
                    rrf_score += contribution;
                    weights_used.insert("vector".to_string(), vector_weight);
                    raw_scores.insert("vector".to_string(), vector_score);
                    normalized_scores.insert("vector_rrf".to_string(), contribution);
                }
            }

            // Graph RRF contribution
            if let Some(graph_score) = scores.graph_score {
                if let Some(&rank) = graph_ranks.get(&id) {
                    let contribution = graph_weight / (k + rank as f32);
                    rrf_score += contribution;
                    weights_used.insert("graph".to_string(), graph_weight);
                    raw_scores.insert("graph".to_string(), graph_score);
                    normalized_scores.insert("graph_rrf".to_string(), contribution);
                }
            }

            // Apply score normalization if enabled
            if optimized && self.config.normalize_scores {
                rrf_score = self.normalize_rrf_score(rrf_score, &text_ranks, &vector_ranks, &graph_ranks, k);
            }

            HybridSearchResult {
                id,
                score: rrf_score,
                text_score: scores.text_score,
                vector_score: scores.vector_score,
                graph_score: scores.graph_score,
                explanation: SearchExplanation {
                    fusion_method: if optimized { FusionMethod::OptimizedRRF } else { FusionMethod::ReciprocalRankFusion },
                    weights_used,
                    raw_scores,
                    normalized_scores,
                },
            }
        }).collect();

        // Sort by RRF score and apply threshold
        rrf_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        if self.config.min_score_threshold > 0.0 {
            rrf_results.retain(|r| r.score >= self.config.min_score_threshold);
        }

        rrf_results
    }

    fn compute_adaptive_weights(&self, text_sorted: &[(RecordId, f32)], vector_sorted: &[(RecordId, f32)], graph_sorted: &[(RecordId, f32)]) -> (f32, f32, f32) {
        // Compute quality metrics for each modality
        let text_quality = self.compute_modality_quality(text_sorted);
        let vector_quality = self.compute_modality_quality(vector_sorted);
        let graph_quality = self.compute_modality_quality(graph_sorted);
        
        let total_quality = text_quality + vector_quality + graph_quality;
        
        if total_quality > 0.0 {
            (
                text_quality / total_quality,
                vector_quality / total_quality,
                graph_quality / total_quality,
            )
        } else {
            (self.config.text_weight, self.config.vector_weight, self.config.graph_weight)
        }
    }
    
    fn compute_modality_quality(&self, sorted_results: &[(RecordId, f32)]) -> f32 {
        if sorted_results.is_empty() {
            return 0.0;
        }
        
        // Compute score variance as a quality indicator
        let scores: Vec<f32> = sorted_results.iter().map(|(_, score)| *score).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        
        // Higher variance indicates better discrimination
        variance.sqrt() * scores.len() as f32
    }
    
    fn normalize_rrf_score(&self, rrf_score: f32, text_ranks: &HashMap<RecordId, usize>, vector_ranks: &HashMap<RecordId, usize>, graph_ranks: &HashMap<RecordId, usize>, k: f32) -> f32 {
        // Compute theoretical maximum RRF score
        let max_possible = self.config.text_weight / (k + 1.0) + 
                          self.config.vector_weight / (k + 1.0) + 
                          self.config.graph_weight / (k + 1.0);
        
        if max_possible > 0.0 {
            rrf_score / max_possible
        } else {
            rrf_score
        }
    }

    fn harmonic_mean_fusion(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        results.into_iter().map(|(id, scores)| {
            let mut valid_scores = Vec::new();
            let mut weights_used = HashMap::new();
            let mut raw_scores = HashMap::new();
            let mut normalized_scores = HashMap::new();

            if let Some(text_score) = scores.text_score {
                if text_score > 0.0 {
                    let normalized = self.normalize_score(text_score, 0.0, 100.0);
                    valid_scores.push((normalized, self.config.text_weight));
                    weights_used.insert("text".to_string(), self.config.text_weight);
                    raw_scores.insert("text".to_string(), text_score);
                    normalized_scores.insert("text".to_string(), normalized);
                }
            }

            if let Some(vector_score) = scores.vector_score {
                if vector_score > 0.0 {
                    let normalized = self.normalize_score(vector_score, 0.0, 1.0);
                    valid_scores.push((normalized, self.config.vector_weight));
                    weights_used.insert("vector".to_string(), self.config.vector_weight);
                    raw_scores.insert("vector".to_string(), vector_score);
                    normalized_scores.insert("vector".to_string(), normalized);
                }
            }

            if let Some(graph_score) = scores.graph_score {
                if graph_score > 0.0 {
                    let normalized = self.normalize_score(graph_score, 0.0, 1.0);
                    valid_scores.push((normalized, self.config.graph_weight));
                    weights_used.insert("graph".to_string(), self.config.graph_weight);
                    raw_scores.insert("graph".to_string(), graph_score);
                    normalized_scores.insert("graph".to_string(), normalized);
                }
            }

            let harmonic_mean = if !valid_scores.is_empty() {
                let weighted_sum: f32 = valid_scores.iter().map(|(_, w)| w).sum();
                let harmonic_sum: f32 = valid_scores.iter().map(|(s, w)| w / s).sum();
                if harmonic_sum > 0.0 { weighted_sum / harmonic_sum } else { 0.0 }
            } else {
                0.0
            };

            HybridSearchResult {
                id,
                score: harmonic_mean,
                text_score: scores.text_score,
                vector_score: scores.vector_score,
                graph_score: scores.graph_score,
                explanation: SearchExplanation {
                    fusion_method: FusionMethod::HarmonicMean,
                    weights_used,
                    raw_scores,
                    normalized_scores,
                },
            }
        }).collect()
    }

    fn geometric_mean_fusion(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        results.into_iter().map(|(id, scores)| {
            let mut valid_scores = Vec::new();
            let mut weights_used = HashMap::new();
            let mut raw_scores = HashMap::new();
            let mut normalized_scores = HashMap::new();

            if let Some(text_score) = scores.text_score {
                let normalized = self.normalize_score(text_score, 0.0, 100.0);
                valid_scores.push((normalized.max(1e-6), self.config.text_weight)); // Avoid log(0)
                weights_used.insert("text".to_string(), self.config.text_weight);
                raw_scores.insert("text".to_string(), text_score);
                normalized_scores.insert("text".to_string(), normalized);
            }

            if let Some(vector_score) = scores.vector_score {
                let normalized = self.normalize_score(vector_score, 0.0, 1.0);
                valid_scores.push((normalized.max(1e-6), self.config.vector_weight));
                weights_used.insert("vector".to_string(), self.config.vector_weight);
                raw_scores.insert("vector".to_string(), vector_score);
                normalized_scores.insert("vector".to_string(), normalized);
            }

            if let Some(graph_score) = scores.graph_score {
                let normalized = self.normalize_score(graph_score, 0.0, 1.0);
                valid_scores.push((normalized.max(1e-6), self.config.graph_weight));
                weights_used.insert("graph".to_string(), self.config.graph_weight);
                raw_scores.insert("graph".to_string(), graph_score);
                normalized_scores.insert("graph".to_string(), normalized);
            }

            let geometric_mean = if !valid_scores.is_empty() {
                let total_weight: f32 = valid_scores.iter().map(|(_, w)| w).sum();
                let log_sum: f32 = valid_scores.iter().map(|(s, w)| w * s.ln()).sum();
                if total_weight > 0.0 { (log_sum / total_weight).exp() } else { 0.0 }
            } else {
                0.0
            };

            HybridSearchResult {
                id,
                score: geometric_mean,
                text_score: scores.text_score,
                vector_score: scores.vector_score,
                graph_score: scores.graph_score,
                explanation: SearchExplanation {
                    fusion_method: FusionMethod::GeometricMean,
                    weights_used,
                    raw_scores,
                    normalized_scores,
                },
            }
        }).collect()
    }

    fn apply_fusion_method(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        match self.config.fusion_method {
            FusionMethod::WeightedSum => self.weighted_sum_fusion(results),
            FusionMethod::ReciprocalRankFusion => self.reciprocal_rank_fusion(results),
            FusionMethod::OptimizedRRF => self.optimized_rrf(results),
            FusionMethod::BordaCount => self.borda_count_fusion(results),
            FusionMethod::CombMax => self.comb_max_fusion(results),
            FusionMethod::CombSum => self.comb_sum_fusion(results),
            FusionMethod::HarmonicMean => self.harmonic_mean_fusion(results),
            FusionMethod::GeometricMean => self.geometric_mean_fusion(results),
            FusionMethod::ConvexCombination => self.convex_combination_fusion(results),
        }
    }

    fn borda_count_fusion(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        // Similar to RRF but uses position-based scoring
        let mut text_sorted: Vec<_> = results.iter()
            .filter_map(|(id, scores)| scores.text_score.map(|s| (id.clone(), s)))
            .collect();
        text_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut vector_sorted: Vec<_> = results.iter()
            .filter_map(|(id, scores)| scores.vector_score.map(|s| (id.clone(), s)))
            .collect();
        vector_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let text_len = text_sorted.len();
        let vector_len = vector_sorted.len();

        results.into_iter().map(|(id, scores)| {
            let mut borda_score = 0.0;
            let mut weights_used = HashMap::new();
            let mut raw_scores = HashMap::new();
            let mut normalized_scores = HashMap::new();

            // Text Borda points
            if let Some(text_score) = scores.text_score {
                if let Some(pos) = text_sorted.iter().position(|(tid, _)| tid == &id) {
                    let points = (text_len - pos) as f32 * self.config.text_weight;
                    borda_score += points;
                    weights_used.insert("text".to_string(), self.config.text_weight);
                    raw_scores.insert("text".to_string(), text_score);
                    normalized_scores.insert("text_borda".to_string(), points);
                }
            }

            // Vector Borda points
            if let Some(vector_score) = scores.vector_score {
                if let Some(pos) = vector_sorted.iter().position(|(vid, _)| vid == &id) {
                    let points = (vector_len - pos) as f32 * self.config.vector_weight;
                    borda_score += points;
                    weights_used.insert("vector".to_string(), self.config.vector_weight);
                    raw_scores.insert("vector".to_string(), vector_score);
                    normalized_scores.insert("vector_borda".to_string(), points);
                }
            }

            HybridSearchResult {
                id,
                score: borda_score,
                text_score: scores.text_score,
                vector_score: scores.vector_score,
                graph_score: scores.graph_score,
                explanation: SearchExplanation {
                    fusion_method: FusionMethod::BordaCount,
                    weights_used,
                    raw_scores,
                    normalized_scores,
                },
            }
        }).collect()
    }

    fn comb_max_fusion(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        results.into_iter().map(|(id, scores)| {
            let mut max_score: f32 = 0.0;
            let mut weights_used = HashMap::new();
            let mut raw_scores = HashMap::new();
            let mut normalized_scores = HashMap::new();

            if let Some(text_score) = scores.text_score {
                let normalized = self.normalize_score(text_score, 0.0, 100.0);
                max_score = max_score.max(normalized);
                raw_scores.insert("text".to_string(), text_score);
                normalized_scores.insert("text".to_string(), normalized);
            }

            if let Some(vector_score) = scores.vector_score {
                let normalized = self.normalize_score(vector_score, 0.0, 1.0);
                max_score = max_score.max(normalized);
                raw_scores.insert("vector".to_string(), vector_score);
                normalized_scores.insert("vector".to_string(), normalized);
            }

            if let Some(graph_score) = scores.graph_score {
                let normalized = self.normalize_score(graph_score, 0.0, 1.0);
                max_score = max_score.max(normalized);
                raw_scores.insert("graph".to_string(), graph_score);
                normalized_scores.insert("graph".to_string(), normalized);
            }

            weights_used.insert("max_selection".to_string(), 1.0);

            HybridSearchResult {
                id,
                score: max_score,
                text_score: scores.text_score,
                vector_score: scores.vector_score,
                graph_score: scores.graph_score,
                explanation: SearchExplanation {
                    fusion_method: FusionMethod::CombMax,
                    weights_used,
                    raw_scores,
                    normalized_scores,
                },
            }
        }).collect()
    }

    fn comb_sum_fusion(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        results.into_iter().map(|(id, scores)| {
            let mut sum_score = 0.0;
            let mut weights_used = HashMap::new();
            let mut raw_scores = HashMap::new();
            let mut normalized_scores = HashMap::new();

            if let Some(text_score) = scores.text_score {
                let normalized = self.normalize_score(text_score, 0.0, 100.0);
                sum_score += normalized;
                weights_used.insert("text".to_string(), 1.0);
                raw_scores.insert("text".to_string(), text_score);
                normalized_scores.insert("text".to_string(), normalized);
            }

            if let Some(vector_score) = scores.vector_score {
                let normalized = self.normalize_score(vector_score, 0.0, 1.0);
                sum_score += normalized;
                weights_used.insert("vector".to_string(), 1.0);
                raw_scores.insert("vector".to_string(), vector_score);
                normalized_scores.insert("vector".to_string(), normalized);
            }

            if let Some(graph_score) = scores.graph_score {
                let normalized = self.normalize_score(graph_score, 0.0, 1.0);
                sum_score += normalized;
                weights_used.insert("graph".to_string(), 1.0);
                raw_scores.insert("graph".to_string(), graph_score);
                normalized_scores.insert("graph".to_string(), normalized);
            }

            HybridSearchResult {
                id,
                score: sum_score,
                text_score: scores.text_score,
                vector_score: scores.vector_score,
                graph_score: scores.graph_score,
                explanation: SearchExplanation {
                    fusion_method: FusionMethod::CombSum,
                    weights_used,
                    raw_scores,
                    normalized_scores,
                },
            }
        }).collect()
    }

    fn convex_combination_fusion(&self, results: HashMap<RecordId, HybridScores>) -> Vec<HybridSearchResult> {
        // Advanced fusion with learned/optimized weights
        self.weighted_sum_fusion(results) // Fallback to weighted sum for now
    }


    fn normalize_score(&self, score: f32, min: f32, max: f32) -> f32 {
        if max - min > 0.0 {
            (score - min) / (max - min)
        } else {
            score
        }
    }

    async fn search_graph(
        &self,
        constraints: &GraphConstraints,
        limit: usize,
    ) -> Result<Vec<(RecordId, f32)>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        for start_node in &constraints.start_nodes {
            let score = 1.0 / (constraints.max_distance as f32 + 1.0);
            results.push((start_node.clone(), score));
        }

        Ok(results)
    }
}

#[derive(Default)]
struct HybridScores {
    text_score: Option<f32>,
    vector_score: Option<f32>,
    graph_score: Option<f32>,
}