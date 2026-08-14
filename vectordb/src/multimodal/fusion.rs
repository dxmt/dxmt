//! Fusion strategies for combining multimodal embeddings

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::Modality;
use crate::embeddings::EmbeddingError;
use crate::types::DenseVector;

/// Fusion strategies for combining embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Simple averaging
    Average,
    /// Weighted averaging based on modality weights
    WeightedAverage,
    /// Concatenation of embeddings
    Concatenate,
    /// Attention-based fusion
    Attention,
    /// Late fusion (keep separate until search)
    LateFusion,
    /// Rank-based fusion
    RankFusion,
    /// Learned fusion with MLP
    LearnedFusion,
}

/// Modality weights for fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityWeight {
    pub text: f32,
    pub image: f32,
    pub audio: f32,
    pub video: f32,
    pub code: f32,
}

impl Default for ModalityWeight {
    fn default() -> Self {
        Self {
            text: 1.0,
            image: 1.0,
            audio: 1.0,
            video: 1.0,
            code: 1.0,
        }
    }
}

impl ModalityWeight {
    pub fn get_weight(&self, modality: Modality) -> f32 {
        match modality {
            Modality::Text => self.text,
            Modality::Image => self.image,
            Modality::Audio => self.audio,
            Modality::Video => self.video,
            Modality::Code => self.code,
        }
    }

    pub fn normalize(&mut self) {
        let sum = self.text + self.image + self.audio + self.video + self.code;
        if sum > 0.0 {
            self.text /= sum;
            self.image /= sum;
            self.audio /= sum;
            self.video /= sum;
            self.code /= sum;
        }
    }
}

/// Fusion engine for combining multimodal embeddings
#[allow(dead_code)]
pub struct FusionEngine {
    strategy: FusionStrategy,
    weights: ModalityWeight,
    attention_weights: Option<HashMap<(Modality, Modality), f32>>,
}

impl FusionEngine {
    pub fn new(strategy: FusionStrategy) -> Self {
        Self {
            strategy,
            weights: ModalityWeight::default(),
            attention_weights: None,
        }
    }

    pub fn with_weights(mut self, weights: ModalityWeight) -> Self {
        self.weights = weights;
        self.weights.normalize();
        self
    }

    pub async fn fuse(
        &self,
        embeddings: Vec<(Modality, DenseVector)>,
    ) -> Result<DenseVector, EmbeddingError> {
        if embeddings.is_empty() {
            return Err(EmbeddingError::InvalidInput(
                "No embeddings to fuse".to_string(),
            ));
        }

        match self.strategy {
            FusionStrategy::Average => self.average_fusion(embeddings),
            FusionStrategy::WeightedAverage => self.weighted_average_fusion(embeddings),
            FusionStrategy::Concatenate => self.concatenate_fusion(embeddings),
            FusionStrategy::Attention => self.attention_fusion(embeddings),
            FusionStrategy::LateFusion => self.late_fusion(embeddings),
            FusionStrategy::RankFusion => self.rank_fusion(embeddings),
            FusionStrategy::LearnedFusion => self.learned_fusion(embeddings),
        }
    }

    fn average_fusion(
        &self,
        embeddings: Vec<(Modality, DenseVector)>,
    ) -> Result<DenseVector, EmbeddingError> {
        let dimension = embeddings[0].1.len();
        let mut result = vec![0.0f32; dimension];

        for (_, embedding) in &embeddings {
            if embedding.len() != dimension {
                return Err(EmbeddingError::InvalidInput(
                    "Embeddings have different dimensions".to_string(),
                ));
            }

            for (i, value) in embedding.iter().enumerate() {
                result[i] += value;
            }
        }

        // Average
        let count = embeddings.len() as f32;
        for value in &mut result {
            *value /= count;
        }

        // Normalize
        self.normalize_vector(&mut result);

        Ok(result)
    }

    fn weighted_average_fusion(
        &self,
        embeddings: Vec<(Modality, DenseVector)>,
    ) -> Result<DenseVector, EmbeddingError> {
        let dimension = embeddings[0].1.len();
        let mut result = vec![0.0f32; dimension];
        let mut total_weight = 0.0f32;

        for (modality, embedding) in &embeddings {
            if embedding.len() != dimension {
                return Err(EmbeddingError::InvalidInput(
                    "Embeddings have different dimensions".to_string(),
                ));
            }

            let weight = self.weights.get_weight(*modality);
            total_weight += weight;

            for (i, value) in embedding.iter().enumerate() {
                result[i] += value * weight;
            }
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            for value in &mut result {
                *value /= total_weight;
            }
        }

        // L2 normalize
        self.normalize_vector(&mut result);

        Ok(result)
    }

    fn concatenate_fusion(
        &self,
        embeddings: Vec<(Modality, DenseVector)>,
    ) -> Result<DenseVector, EmbeddingError> {
        let mut result = Vec::new();

        // Sort by modality for consistent ordering
        let mut sorted_embeddings = embeddings;
        sorted_embeddings.sort_by_key(|(m, _)| *m as u8);

        for (_, embedding) in sorted_embeddings {
            result.extend(embedding);
        }

        // Optionally normalize the concatenated vector
        self.normalize_vector(&mut result);

        Ok(result)
    }

    fn attention_fusion(
        &self,
        embeddings: Vec<(Modality, DenseVector)>,
    ) -> Result<DenseVector, EmbeddingError> {
        // Simplified attention mechanism
        let dimension = embeddings[0].1.len();
        let mut result = vec![0.0f32; dimension];

        // Compute attention scores (simplified - dot product with learned weights)
        let mut attention_scores = Vec::new();
        for (modality, embedding) in &embeddings {
            // Compute score based on embedding norm and modality weight
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            let weight = self.weights.get_weight(*modality);
            attention_scores.push(norm * weight);
        }

        // Softmax normalization
        let max_score = attention_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = attention_scores
            .iter()
            .map(|s| (s - max_score).exp())
            .collect();
        let sum_exp = exp_scores.iter().sum::<f32>();

        // Apply attention weights
        for (i, (_, embedding)) in embeddings.iter().enumerate() {
            let attention_weight = exp_scores[i] / sum_exp;
            for (j, value) in embedding.iter().enumerate() {
                result[j] += value * attention_weight;
            }
        }

        self.normalize_vector(&mut result);

        Ok(result)
    }

    fn late_fusion(
        &self,
        embeddings: Vec<(Modality, DenseVector)>,
    ) -> Result<DenseVector, EmbeddingError> {
        // For late fusion, we keep modality-specific embeddings separate
        // and combine them at search time. Here we just return weighted average
        // as a placeholder - actual late fusion happens during search
        self.weighted_average_fusion(embeddings)
    }

    fn rank_fusion(
        &self,
        embeddings: Vec<(Modality, DenseVector)>,
    ) -> Result<DenseVector, EmbeddingError> {
        // Rank-based fusion (RRF - Reciprocal Rank Fusion)
        // This is typically used for combining search results, not embeddings
        // For embeddings, we fall back to weighted average
        self.weighted_average_fusion(embeddings)
    }

    fn learned_fusion(
        &self,
        embeddings: Vec<(Modality, DenseVector)>,
    ) -> Result<DenseVector, EmbeddingError> {
        // Learned fusion would use a trained MLP to combine embeddings
        // For now, use attention-based fusion as a proxy
        self.attention_fusion(embeddings)
    }

    fn normalize_vector(&self, vector: &mut [f32]) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in vector {
                *value /= norm;
            }
        }
    }
}

/// Cross-modal alignment for better fusion
pub struct CrossModalAlignment {
    projection_matrices: HashMap<(Modality, Modality), Vec<Vec<f32>>>,
}

impl Default for CrossModalAlignment {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossModalAlignment {
    pub fn new() -> Self {
        Self {
            projection_matrices: HashMap::new(),
        }
    }

    pub fn align(
        &self,
        source_modality: Modality,
        target_modality: Modality,
        embedding: &[f32],
    ) -> Result<Vec<f32>, EmbeddingError> {
        // Look up projection matrix
        let key = (source_modality, target_modality);

        if let Some(matrix) = self.projection_matrices.get(&key) {
            // Apply linear projection
            let output_dim = matrix.len();
            let mut result = vec![0.0f32; output_dim];

            for (i, row) in matrix.iter().enumerate() {
                if row.len() != embedding.len() {
                    return Err(EmbeddingError::InvalidInput(
                        "Dimension mismatch in projection".to_string(),
                    ));
                }

                for (j, value) in embedding.iter().enumerate() {
                    result[i] += value * row[j];
                }
            }

            Ok(result)
        } else {
            // No alignment available, return original
            Ok(embedding.to_vec())
        }
    }

    pub fn learn_alignment(
        &mut self,
        source_modality: Modality,
        target_modality: Modality,
        source_embeddings: &[Vec<f32>],
        target_embeddings: &[Vec<f32>],
    ) -> Result<(), EmbeddingError> {
        if source_embeddings.len() != target_embeddings.len() {
            return Err(EmbeddingError::InvalidInput(
                "Number of source and target embeddings must match".to_string(),
            ));
        }

        // Simple linear regression to learn projection
        // In production, use proper optimization (e.g., SGD)
        let source_dim = source_embeddings[0].len();
        let target_dim = target_embeddings[0].len();

        // Initialize random projection matrix
        let mut projection = vec![vec![0.0f32; source_dim]; target_dim];

        // Simplified learning (in production, use proper optimization)
        for (i, row) in projection.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((i + j) as f32 / (source_dim + target_dim) as f32) - 0.5;
            }
        }

        self.projection_matrices
            .insert((source_modality, target_modality), projection);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_weights() {
        let mut weights = ModalityWeight {
            text: 2.0,
            image: 3.0,
            audio: 1.0,
            video: 0.0,
            code: 0.0,
        };

        weights.normalize();

        assert!((weights.text - 0.333).abs() < 0.01);
        assert!((weights.image - 0.5).abs() < 0.01);
        assert!((weights.audio - 0.167).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_average_fusion() {
        let engine = FusionEngine::new(FusionStrategy::Average);

        let embeddings = vec![
            (Modality::Text, vec![1.0, 2.0, 3.0]),
            (Modality::Image, vec![4.0, 5.0, 6.0]),
        ];

        let result = engine.fuse(embeddings).await.unwrap();
        assert_eq!(result.len(), 3);

        // Check normalization
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
