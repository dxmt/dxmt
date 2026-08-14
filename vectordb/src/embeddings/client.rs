use crate::types::DenseVector;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// OpenAI Embedding Client
pub struct OpenAIEmbeddingClient {
    api_key: String,
    model: String,
    client: Client,
}

impl OpenAIEmbeddingClient {
    pub fn new(api_key: String, model: Option<String>) -> Self {
        Self {
            api_key,
            model: model.unwrap_or_else(|| "text-embedding-ada-002".to_string()),
            client: Client::new(),
        }
    }

    pub async fn embed(&self, text: &str) -> crate::Result<DenseVector> {
        #[derive(Serialize)]
        struct Request {
            input: String,
            model: String,
        }

        #[derive(Deserialize)]
        struct Response {
            data: Vec<EmbeddingData>,
        }

        #[derive(Deserialize)]
        struct EmbeddingData {
            embedding: Vec<f32>,
        }

        let request = Request {
            input: text.to_string(),
            model: self.model.clone(),
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::Error::Internal(format!("OpenAI API request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(crate::Error::Internal(format!(
                "OpenAI API error: {}",
                response.status()
            )));
        }

        let resp: Response = response
            .json()
            .await
            .map_err(|e| crate::Error::Internal(format!("Failed to parse response: {}", e)))?;

        resp.data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| crate::Error::Internal("No embedding returned".to_string()))
    }
}

/// HuggingFace Embedding Client
pub struct HuggingFaceEmbeddingClient {
    api_key: String,
    model: String,
    client: Client,
}

impl HuggingFaceEmbeddingClient {
    pub fn new(api_key: String, model: Option<String>) -> Self {
        Self {
            api_key,
            model: model.unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string()),
            client: Client::new(),
        }
    }

    pub async fn embed(&self, text: &str) -> crate::Result<DenseVector> {
        #[derive(Serialize)]
        struct Request {
            inputs: String,
        }

        let request = Request {
            inputs: text.to_string(),
        };

        let response = self
            .client
            .post(format!(
                "https://api-inference.huggingface.co/models/{}",
                self.model
            ))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                crate::Error::Internal(format!("HuggingFace API request failed: {}", e))
            })?;

        if !response.status().is_success() {
            return Err(crate::Error::Internal(format!(
                "HuggingFace API error: {}",
                response.status()
            )));
        }

        let embedding: Vec<f32> = response
            .json()
            .await
            .map_err(|e| crate::Error::Internal(format!("Failed to parse response: {}", e)))?;

        Ok(embedding)
    }
}

/// Local ONNX Embedding Client using ort (ONNX Runtime)
pub struct LocalEmbeddingClient {
    #[allow(dead_code)]
    model_path: String,
}

impl LocalEmbeddingClient {
    pub fn new(model_path: String) -> Self {
        Self { model_path }
    }

    pub async fn embed(&self, text: &str) -> crate::Result<DenseVector> {
        // For now, return a mock embedding
        // In production, this would use ONNX Runtime to run the model
        let dimension = 384; // Common dimension for sentence transformers
        let mut embedding = vec![0.0f32; dimension];

        // Simple hash-based mock embedding
        for (i, c) in text.chars().enumerate() {
            embedding[i % dimension] += (c as u32) as f32 / 1000.0;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }
}

/// Unified embedding client that can switch between providers
pub enum EmbeddingClient {
    OpenAI(OpenAIEmbeddingClient),
    HuggingFace(HuggingFaceEmbeddingClient),
    Local(LocalEmbeddingClient),
}

impl EmbeddingClient {
    pub async fn embed(&self, text: &str) -> crate::Result<DenseVector> {
        match self {
            Self::OpenAI(client) => client.embed(text).await,
            Self::HuggingFace(client) => client.embed(text).await,
            Self::Local(client) => client.embed(text).await,
        }
    }

    pub fn from_config(
        provider: &str,
        api_key: Option<String>,
        model: Option<String>,
    ) -> crate::Result<Self> {
        match provider.to_lowercase().as_str() {
            "openai" => {
                let api_key = api_key.ok_or_else(|| {
                    crate::Error::InvalidInput("OpenAI requires API key".to_string())
                })?;
                Ok(Self::OpenAI(OpenAIEmbeddingClient::new(api_key, model)))
            }
            "huggingface" => {
                let api_key = api_key.ok_or_else(|| {
                    crate::Error::InvalidInput("HuggingFace requires API key".to_string())
                })?;
                Ok(Self::HuggingFace(HuggingFaceEmbeddingClient::new(
                    api_key, model,
                )))
            }
            "local" => {
                let model_path = model.unwrap_or_else(|| "models/default.onnx".to_string());
                Ok(Self::Local(LocalEmbeddingClient::new(model_path)))
            }
            _ => Err(crate::Error::InvalidInput(format!(
                "Unknown provider: {}",
                provider
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_local_embedding() {
        let client = LocalEmbeddingClient::new("test.onnx".to_string());
        let embedding = client.embed("test text").await.unwrap();
        assert_eq!(embedding.len(), 384);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
