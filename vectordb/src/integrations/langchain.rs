//! LangChain integration for DXMT VectorDB

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use uuid::Uuid;

use crate::types::Record;
use crate::types::record::SearchQuery;

use crate::storage::Storage;
use crate::embeddings::{EmbeddingFunction, AutoEmbedding, AutoEmbeddingConfig};


/// LangChain Document structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub page_content: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Document {
    pub fn new(content: String) -> Self {
        Self {
            page_content: content,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// LangChain VectorStore trait implementation for DXMT
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Add documents to the vector store
    async fn add_documents(&self, documents: Vec<Document>) -> Result<Vec<String>, VectorStoreError>;
    
    /// Add texts to the vector store
    async fn add_texts(&self, texts: Vec<String>, metadatas: Option<Vec<HashMap<String, serde_json::Value>>>) -> Result<Vec<String>, VectorStoreError>;
    
    /// Similarity search
    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<Document>, VectorStoreError>;
    
    /// Similarity search with score
    async fn similarity_search_with_score(&self, query: &str, k: usize) -> Result<Vec<(Document, f32)>, VectorStoreError>;
    
    /// Similarity search by vector
    async fn similarity_search_by_vector(&self, embedding: Vec<f32>, k: usize) -> Result<Vec<Document>, VectorStoreError>;
    
    /// Max marginal relevance search
    async fn max_marginal_relevance_search(&self, query: &str, k: usize, fetch_k: usize, lambda_mult: f32) -> Result<Vec<Document>, VectorStoreError>;
    
    /// Delete documents
    async fn delete(&self, ids: Vec<String>) -> Result<bool, VectorStoreError>;
}

/// Vector store errors
#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
    
    #[error("Search error: {0}")]
    SearchError(String),
    
    #[error("Invalid document: {0}")]
    InvalidDocument(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
}

/// DXMT VectorStore implementation
pub struct DXMTVectorStore {
    storage: Arc<dyn Storage + Send + Sync>,
    embedding_function: Arc<dyn EmbeddingFunction>,
    table_name: String,
    text_key: String,
    embedding_key: String,
    namespace: Option<String>,
}

impl DXMTVectorStore {
    pub async fn new(
        storage: Arc<dyn Storage + Send + Sync>,
        table_name: String,
        embedding_function: Option<Arc<dyn EmbeddingFunction>>,
    ) -> Result<Self, VectorStoreError> {
        // Use auto-embedding if no function provided
        let embedding_function = match embedding_function {
            Some(ef) => ef,
            None => {
                let config = AutoEmbeddingConfig::default();
                let auto_embedding = AutoEmbedding::new(config).await
                    .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;
                Arc::new(auto_embedding) as Arc<dyn EmbeddingFunction>
            }
        };
        
        Ok(Self {
            storage,
            embedding_function,
            table_name,
            text_key: "text".to_string(),
            embedding_key: "embedding".to_string(),
            namespace: None,
        })
    }
    
    pub fn with_namespace(mut self, namespace: String) -> Self {
        self.namespace = Some(namespace);
        self
    }
    
    pub fn with_text_key(mut self, key: String) -> Self {
        self.text_key = key;
        self
    }
    
    fn document_to_record(&self, doc: &Document, embedding: Vec<f32>, id: Option<String>) -> Record {
        let record_id = id.unwrap_or_else(|| Uuid::new_v4().to_string());
        
        let mut payload_map = serde_json::Map::new();
        payload_map.insert(self.text_key.clone(), serde_json::Value::String(doc.page_content.clone()));
        
        // Add metadata
        for (key, value) in &doc.metadata {
            payload_map.insert(key.clone(), value.clone());
        }
        
        // Add namespace if set
        if let Some(ns) = &self.namespace {
            payload_map.insert("namespace".to_string(), serde_json::Value::String(ns.clone()));
        }
        
        let payload = serde_json::Value::Object(payload_map);
        
        Record {
            id: record_id,
            vector: embedding,
            sparse_vector: None,
            payload,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
            status: crate::types::record::RecordStatus::Active,
            named_vectors: None,
        }
    }
    
    fn record_to_document(&self, record: &Record) -> Document {
        let mut metadata = HashMap::new();
        
        if let Some(obj) = record.payload.as_object() {
            for (key, value) in obj.iter() {
                if key != &self.text_key {
                    metadata.insert(key.clone(), value.clone());
                }
            }
        }
        
        let page_content = record.payload
            .get(&self.text_key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
            
        Document {
            page_content,
            metadata,
        }
    }
    
    async fn mmr_rerank(&self, query_embedding: Vec<f32>, results: Vec<Record>, k: usize, lambda_mult: f32) -> Vec<Record> {
        if results.is_empty() || k == 0 {
            return vec![];
        }
        
        let mut selected = Vec::new();
        let mut candidates = results;
        
        // Select first document (most similar to query)
        if let Some(best_idx) = Self::find_most_similar(&query_embedding, &candidates) {
            selected.push(candidates.remove(best_idx));
        }
        
        // Iteratively select documents
        while selected.len() < k && !candidates.is_empty() {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = 0;
            
            for (idx, candidate) in candidates.iter().enumerate() {
                // Similarity to query
                let query_sim = Self::cosine_similarity(&query_embedding, &candidate.vector);
                
                // Max similarity to already selected documents
                let max_sim = selected.iter()
                    .map(|doc| Self::cosine_similarity(&candidate.vector, &doc.vector))
                    .fold(f32::NEG_INFINITY, f32::max);
                    
                // MMR score
                let mmr_score = lambda_mult * query_sim - (1.0 - lambda_mult) * max_sim;
                
                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = idx;
                }
            }
            
            selected.push(candidates.remove(best_idx));
        }
        
        selected
    }
    
    fn find_most_similar(query: &[f32], candidates: &[Record]) -> Option<usize> {
        candidates.iter()
            .enumerate()
            .map(|(idx, record)| (idx, Self::cosine_similarity(query, &record.vector)))
            .max_by(|a, b| {
                match a.1.partial_cmp(&b.1) {
                    Some(ord) => ord,
                    None => std::cmp::Ordering::Equal,
                }
            })
            .map(|(idx, _)| idx)
    }
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

#[async_trait]
impl VectorStore for DXMTVectorStore {
    async fn add_documents(&self, documents: Vec<Document>) -> Result<Vec<String>, VectorStoreError> {
        let texts: Vec<String> = documents.iter()
            .map(|doc| doc.page_content.clone())
            .collect();
            
        let metadatas: Vec<HashMap<String, serde_json::Value>> = documents.iter()
            .map(|doc| doc.metadata.clone())
            .collect();
            
        self.add_texts(texts, Some(metadatas)).await
    }
    
    async fn add_texts(&self, texts: Vec<String>, metadatas: Option<Vec<HashMap<String, serde_json::Value>>>) -> Result<Vec<String>, VectorStoreError> {
        // Generate embeddings
        let embeddings = self.embedding_function.embed_batch(&texts).await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;
            
        // Create documents and records
        let mut ids = Vec::new();
        let mut records = Vec::new();
        
        for (i, (text, embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
            let mut doc = Document::new(text.clone());
            
            if let Some(ref metas) = metadatas {
                if let Some(meta) = metas.get(i) {
                    doc.metadata = meta.clone();
                }
            }
            
            let record = self.document_to_record(&doc, embedding.clone(), None);
            ids.push(record.id.to_string());
            records.push(record);
        }
        
        for record in records {
            let request = crate::types::record::CreateRecordRequest {
                id: record.id.clone(),
                vector: record.vector,
                sparse_vector: record.sparse_vector,
                payload: Some(record.payload),
                named_vectors: None,
            };
            self.storage.insert_record(&self.table_name, request).await
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        }
        
        Ok(ids)
    }
    
    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<Document>, VectorStoreError> {
        let results = self.similarity_search_with_score(query, k).await?;
        Ok(results.into_iter().map(|(doc, _)| doc).collect())
    }
    
    async fn similarity_search_with_score(&self, query: &str, k: usize) -> Result<Vec<(Document, f32)>, VectorStoreError> {
        // Generate query embedding
        let query_embedding = self.embedding_function.embed(query).await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;
            
        self.similarity_search_by_vector(query_embedding, k).await
            .map(|docs| docs.into_iter().map(|doc| (doc, 0.0)).collect())
    }
    
    async fn similarity_search_by_vector(&self, embedding: Vec<f32>, k: usize) -> Result<Vec<Document>, VectorStoreError> {
        let search_query = SearchQuery {
            vector: embedding.clone(),
            top_k: k,
            filter: None, // Simplified for compilation
            with_payload: true,
            with_vector: true,  // Need vectors for similarity calculation
                    limit: 10,
                    include_metadata: true,
            ..Default::default()
        };
        
        // Use search to get records
        let search_results = self.storage.search(&self.table_name, search_query).await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        // Convert search results to documents (search already returns sorted by similarity)
        let mut results: Vec<_> = search_results.hits.into_iter()
            .filter_map(|hit| {
                if let Some(ref vec) = hit.vector {
                    let similarity = Self::cosine_similarity(&embedding, vec);
                    Some((hit, similarity))
                } else {
                    None
                }
            })
            .collect();
        
        results.sort_by(|a, b| {
            match b.1.partial_cmp(&a.1) {
                Some(ord) => ord,
                None => std::cmp::Ordering::Equal,
            }
        });
        results.truncate(k);
        
        let documents: Vec<Document> = results.into_iter()
            .map(|(hit, _)| {
                let record = Record {
                    id: hit.id,
                    vector: hit.vector.unwrap_or_default(),
                    sparse_vector: None,
                    payload: hit.payload.unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new())),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    version: 0,
                    status: crate::types::record::RecordStatus::Active,
                    named_vectors: None,
                };
                self.record_to_document(&record)
            })
            .collect();
        
        Ok(documents)
    }
    
    async fn max_marginal_relevance_search(&self, query: &str, k: usize, fetch_k: usize, lambda_mult: f32) -> Result<Vec<Document>, VectorStoreError> {
        // Generate query embedding
        let query_embedding = self.embedding_function.embed(query).await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;
            
        let search_query = SearchQuery {
            vector: query_embedding.clone(),
            top_k: fetch_k,
            filter: None, // Simplified for compilation
            with_payload: true,
            with_vector: true,
                    limit: 10,
                    include_metadata: true,
            ..Default::default()
        };
        
        // Use search to get records
        let search_results = self.storage.search(&self.table_name, search_query).await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        // Convert search results to documents
        let mut results: Vec<_> = search_results.hits.into_iter()
            .filter_map(|hit| {
                if let Some(ref vec) = hit.vector {
                    let similarity = Self::cosine_similarity(&query_embedding, vec);
                    Some((hit, similarity))
                } else {
                    None
                }
            })
            .collect();
        
        results.sort_by(|a, b| {
            match b.1.partial_cmp(&a.1) {
                Some(ord) => ord,
                None => std::cmp::Ordering::Equal,
            }
        });
        results.truncate(fetch_k);
        
        // Convert SearchHits to Records
        let records: Vec<Record> = results.into_iter().map(|(hit, _)| Record {
            id: hit.id,
            vector: hit.vector.unwrap_or_default(),
            sparse_vector: None,
            payload: hit.payload.unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new())),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
            status: crate::types::RecordStatus::Active,
            named_vectors: None,
        }).collect();
        
        // Apply MMR reranking
        let reranked = self.mmr_rerank(query_embedding, records, k, lambda_mult).await;
        
        Ok(reranked.iter().map(|r| self.record_to_document(r)).collect())
    }
    
    async fn delete(&self, ids: Vec<String>) -> Result<bool, VectorStoreError> {
        for id in ids {
            let record_id = id;
            self.storage.delete_record(&self.table_name, &record_id).await
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        }
        
        Ok(true)
    }
}

/// Document loader for various file formats
pub struct DocumentLoader {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl Default for DocumentLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader {
    pub fn new() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
        }
    }
    
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }
    
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }
    
    pub async fn load_text(&self, text: String, source: Option<String>) -> Vec<Document> {
        self.split_text(text, source)
    }
    
    pub async fn load_file(&self, path: &str) -> Result<Vec<Document>, VectorStoreError> {
        let content = tokio::fs::read_to_string(path).await
            .map_err(|e| VectorStoreError::InvalidDocument(e.to_string()))?;
            
        Ok(self.split_text(content, Some(path.to_string())))
    }
    
    pub async fn load_directory(&self, path: &str, pattern: Option<&str>) -> Result<Vec<Document>, VectorStoreError> {
        let pattern = pattern.unwrap_or("*.txt");
        let mut documents = Vec::new();
        
        let mut entries = tokio::fs::read_dir(path).await
            .map_err(|e| VectorStoreError::InvalidDocument(e.to_string()))?;
            
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| VectorStoreError::InvalidDocument(e.to_string()))? {
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name() {
                    if name.to_string_lossy().contains(pattern.trim_start_matches('*')) {
                        let docs = self.load_file(&path.to_string_lossy()).await?;
                        documents.extend(docs);
                    }
                }
            }
        }
        
        Ok(documents)
    }
    
    fn split_text(&self, text: String, source: Option<String>) -> Vec<Document> {
        let mut documents = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let total_chars = chars.len();
        
        let mut start = 0;
        let mut chunk_index = 0;
        
        while start < total_chars {
            let end = (start + self.chunk_size).min(total_chars);
            let chunk: String = chars[start..end].iter().collect();
            
            let mut metadata = HashMap::new();
            if let Some(ref src) = source {
                metadata.insert("source".to_string(), serde_json::Value::String(src.clone()));
            }
            metadata.insert("chunk_index".to_string(), serde_json::Value::Number(chunk_index.into()));
            metadata.insert("start_char".to_string(), serde_json::Value::Number(start.into()));
            metadata.insert("end_char".to_string(), serde_json::Value::Number(end.into()));
            
            documents.push(Document {
                page_content: chunk,
                metadata,
            });
            
            start = if start + self.chunk_size >= total_chars {
                total_chars
            } else {
                start + self.chunk_size - self.chunk_overlap
            };
            
            chunk_index += 1;
        }
        
        documents
    }
}

/// Retriever chain for question answering
pub struct RetrieverChain {
    vector_store: Arc<dyn VectorStore>,
    k: usize,
}

impl RetrieverChain {
    pub fn new(vector_store: Arc<dyn VectorStore>) -> Self {
        Self {
            vector_store,
            k: 4,
        }
    }
    
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
    
    pub async fn retrieve(&self, query: &str) -> Result<Vec<Document>, VectorStoreError> {
        self.vector_store.similarity_search(query, self.k).await
    }
    
    pub async fn retrieve_with_mmr(&self, query: &str, lambda_mult: f32) -> Result<Vec<Document>, VectorStoreError> {
        self.vector_store.max_marginal_relevance_search(query, self.k, self.k * 2, lambda_mult).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_document_creation() {
        let doc = Document::new("test content".to_string())
            .with_metadata("key".to_string(), serde_json::Value::String("value".to_string()));
            
        assert_eq!(doc.page_content, "test content");
        assert_eq!(doc.metadata.get("key").unwrap(), &serde_json::Value::String("value".to_string()));
    }
    
    #[test]
    fn test_document_loader() {
        let loader = DocumentLoader::new()
            .with_chunk_size(10)
            .with_chunk_overlap(2);
            
        let text = "This is a test document with multiple words".to_string();
        let docs = loader.split_text(text, None);
        
        assert!(docs.len() > 1);
        assert!(docs[0].page_content.len() <= 10);
    }
}
// Missing VectorSearchResult struct
#[derive(Debug, Clone)]
struct VectorSearchResult {
    id: String,
    distance: f32,
}
