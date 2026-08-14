// Example: Semantic Search Application using VectorDB
// This demonstrates a real-world use case for document search

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio;
use vectordb::{
    client::VectorDBClient,
    types::{CreateRecordRequest, CreateTableRequest, DistanceMetric, SearchQuery},
};

// ==================== Data Models ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Document {
    id: String,
    title: String,
    content: String,
    author: String,
    category: String,
    published_date: String,
    tags: Vec<String>,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchResult {
    document: Document,
    score: f32,
    highlights: Vec<String>,
}

// ==================== Embedding Service ====================

struct EmbeddingService {
    model_name: String,
    dimension: usize,
}

impl EmbeddingService {
    fn new() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimension: 384, // MiniLM dimension
        }
    }

    /// Generate embeddings for text (mock implementation)
    /// In production, this would call a real embedding model
    async fn embed(&self, text: &str) -> Vec<f32> {
        // Simulate embedding generation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        // Generate deterministic "embeddings" based on text
        (0..self.dimension)
            .map(|i| {
                let val = ((seed + i as u64) % 1000) as f32 / 1000.0;
                val * 2.0 - 1.0 // Normalize to [-1, 1]
            })
            .collect()
    }

    /// Batch embed multiple texts
    async fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.embed(text).await);
        }
        embeddings
    }
}

// ==================== Document Store ====================

struct DocumentStore {
    client: VectorDBClient,
    embedding_service: EmbeddingService,
    table_name: String,
}

impl DocumentStore {
    async fn new(vectordb_url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let client = VectorDBClient::new(vectordb_url)?;
        let embedding_service = EmbeddingService::new();

        Ok(Self {
            client,
            embedding_service,
            table_name: "documents".to_string(),
        })
    }

    /// Initialize the document table
    async fn initialize(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔧 Initializing document store...");

        // Create table for document embeddings
        let request = CreateTableRequest {
            name: self.table_name.clone(),
            dimension: self.embedding_service.dimension,
            distance_metric: Some("cosine".to_string()),
            description: Some("Document embeddings for semantic search".to_string()),
            index_type: Some("hnsw".to_string()),
        };

        match self.client.create_table(request).await {
            Ok(_) => println!("✅ Table '{}' created successfully", self.table_name),
            Err(e) if e.to_string().contains("already exists") => {
                println!("ℹ️ Table '{}' already exists", self.table_name)
            }
            Err(e) => return Err(e.into()),
        }

        Ok(())
    }

    /// Index a document
    async fn index_document(&self, doc: &Document) -> Result<(), Box<dyn std::error::Error>> {
        // Generate embedding for document
        let content_for_embedding = format!("{} {}", doc.title, doc.content);
        let embedding = self.embedding_service.embed(&content_for_embedding).await;

        // Create record
        let record = CreateRecordRequest {
            id: doc.id.clone(),
            vector: embedding,
            payload: Some(serde_json::to_value(doc)?),
        };

        // Insert into VectorDB
        self.client
            .insert_records(&self.table_name, vec![record])
            .await?;

        Ok(())
    }

    /// Index multiple documents in batch
    async fn index_documents(&self, docs: &[Document]) -> Result<(), Box<dyn std::error::Error>> {
        println!("📝 Indexing {} documents...", docs.len());

        let mut records = Vec::new();

        for doc in docs {
            let content_for_embedding = format!("{} {}", doc.title, doc.content);
            let embedding = self.embedding_service.embed(&content_for_embedding).await;

            records.push(CreateRecordRequest {
                id: doc.id.clone(),
                vector: embedding,
                payload: Some(serde_json::to_value(doc)?),
            });
        }

        self.client
            .insert_records(&self.table_name, records)
            .await?;
        println!("✅ Successfully indexed {} documents", docs.len());

        Ok(())
    }

    /// Search for similar documents
    async fn search(
        &self,
        query: &str,
        limit: usize,
        filters: Option<HashMap<String, String>>,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        // Generate query embedding
        let query_embedding = self.embedding_service.embed(query).await;

        // Build search query
        let search_query = SearchQuery {
            vector: query_embedding,
            limit: Some(limit),
            filter: filters.map(|f| serde_json::to_value(f).unwrap()),
            distance_metric: Some(DistanceMetric::Cosine),
        };

        // Execute search
        let results = self.client.search(&self.table_name, search_query).await?;

        // Process results
        let mut search_results = Vec::new();
        for match_result in results.matches {
            if let Some(payload) = match_result.payload {
                let doc: Document = serde_json::from_value(payload)?;

                // Generate highlights (simplified)
                let highlights = self.generate_highlights(&doc.content, query);

                search_results.push(SearchResult {
                    document: doc,
                    score: 1.0 - match_result.distance, // Convert distance to similarity score
                    highlights,
                });
            }
        }

        Ok(search_results)
    }

    /// Update a document
    async fn update_document(&self, doc: &Document) -> Result<(), Box<dyn std::error::Error>> {
        let content_for_embedding = format!("{} {}", doc.title, doc.content);
        let embedding = self.embedding_service.embed(&content_for_embedding).await;

        self.client
            .update_record(
                &self.table_name,
                &doc.id,
                Some(embedding),
                Some(serde_json::to_value(doc)?),
            )
            .await?;

        Ok(())
    }

    /// Delete a document
    async fn delete_document(&self, doc_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.client
            .delete_records(&self.table_name, &[doc_id.to_string()])
            .await?;
        Ok(())
    }

    /// Generate text highlights for search results
    fn generate_highlights(&self, content: &str, query: &str) -> Vec<String> {
        let query_words: Vec<&str> = query.split_whitespace().collect();
        let sentences: Vec<&str> = content.split(". ").collect();

        let mut highlights = Vec::new();
        for sentence in sentences.iter().take(3) {
            for word in &query_words {
                if sentence.to_lowercase().contains(&word.to_lowercase()) {
                    highlights.push(format!("...{}...", sentence));
                    break;
                }
            }
        }

        highlights
    }
}

// ==================== Sample Data ====================

fn create_sample_documents() -> Vec<Document> {
    vec![
        Document {
            id: "doc1".to_string(),
            title: "Introduction to Vector Databases".to_string(),
            content: "Vector databases are specialized systems designed to store and search high-dimensional vectors. They enable similarity search, which is crucial for AI applications like recommendation systems and semantic search.".to_string(),
            author: "Alice Smith".to_string(),
            category: "Technology".to_string(),
            published_date: "2024-01-15".to_string(),
            tags: vec!["database".to_string(), "vectors".to_string(), "AI".to_string()],
            embedding: vec![],
        },
        Document {
            id: "doc2".to_string(),
            title: "Machine Learning in Production".to_string(),
            content: "Deploying machine learning models in production requires careful consideration of scalability, monitoring, and maintenance. This guide covers best practices for ML deployment.".to_string(),
            author: "Bob Johnson".to_string(),
            category: "Technology".to_string(),
            published_date: "2024-01-14".to_string(),
            tags: vec!["ML".to_string(), "production".to_string(), "deployment".to_string()],
            embedding: vec![],
        },
        Document {
            id: "doc3".to_string(),
            title: "The Future of AI and Society".to_string(),
            content: "Artificial intelligence is reshaping society in profound ways. From healthcare to transportation, AI technologies are creating new possibilities and challenges.".to_string(),
            author: "Carol White".to_string(),
            category: "AI".to_string(),
            published_date: "2024-01-13".to_string(),
            tags: vec!["AI".to_string(), "society".to_string(), "future".to_string()],
            embedding: vec![],
        },
        Document {
            id: "doc4".to_string(),
            title: "Understanding Neural Networks".to_string(),
            content: "Neural networks are the foundation of deep learning. This article explains how neural networks work, from basic perceptrons to complex architectures like transformers.".to_string(),
            author: "David Brown".to_string(),
            category: "AI".to_string(),
            published_date: "2024-01-12".to_string(),
            tags: vec!["neural networks".to_string(), "deep learning".to_string(), "AI".to_string()],
            embedding: vec![],
        },
        Document {
            id: "doc5".to_string(),
            title: "Cloud Computing Best Practices".to_string(),
            content: "Cloud computing offers scalability and flexibility for modern applications. Learn about best practices for cloud architecture, security, and cost optimization.".to_string(),
            author: "Eve Davis".to_string(),
            category: "Cloud".to_string(),
            published_date: "2024-01-11".to_string(),
            tags: vec!["cloud".to_string(), "architecture".to_string(), "DevOps".to_string()],
            embedding: vec![],
        },
    ]
}

// ==================== Main Application ====================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VectorDB Semantic Search Example");
    println!("====================================\n");

    // Initialize document store
    let store = DocumentStore::new("http://localhost:8080").await?;
    store.initialize().await?;

    // Index sample documents
    let documents = create_sample_documents();
    store.index_documents(&documents).await?;

    // Demonstrate different search queries
    println!("\n🔍 Performing semantic searches...\n");

    // Search 1: General query
    println!("Query: 'How do vector databases work?'");
    let results = store
        .search("How do vector databases work?", 3, None)
        .await?;
    print_results(&results);

    // Search 2: Specific technology query
    println!("\nQuery: 'machine learning deployment production'");
    let results = store
        .search("machine learning deployment production", 3, None)
        .await?;
    print_results(&results);

    // Search 3: Filtered search
    println!("\nQuery: 'artificial intelligence' (filtered by category: AI)");
    let mut filters = HashMap::new();
    filters.insert("category".to_string(), "AI".to_string());
    let results = store
        .search("artificial intelligence", 3, Some(filters))
        .await?;
    print_results(&results);

    // Demonstrate update
    println!("\n📝 Updating document...");
    let mut updated_doc = documents[0].clone();
    updated_doc.content = "Vector databases are revolutionary systems for AI applications. They excel at similarity search and are essential for modern machine learning systems.".to_string();
    store.update_document(&updated_doc).await?;
    println!("✅ Document updated successfully");

    // Search again to see updated content
    println!("\nQuery after update: 'revolutionary AI systems'");
    let results = store.search("revolutionary AI systems", 2, None).await?;
    print_results(&results);

    // Demonstrate deletion
    println!("\n🗑️ Deleting document doc5...");
    store.delete_document("doc5").await?;
    println!("✅ Document deleted successfully");

    // Statistics
    println!("\n📊 Search Statistics:");
    println!("- Documents indexed: {}", documents.len());
    println!(
        "- Embedding dimension: {}",
        store.embedding_service.dimension
    );
    println!("- Distance metric: Cosine similarity");

    Ok(())
}

/// Print search results in a formatted way
fn print_results(results: &[SearchResult]) {
    for (i, result) in results.iter().enumerate() {
        println!(
            "  {}. {} (score: {:.3})",
            i + 1,
            result.document.title,
            result.score
        );
        println!("     Author: {}", result.document.author);
        println!("     Category: {}", result.document.category);

        if !result.highlights.is_empty() {
            println!("     Highlights:");
            for highlight in &result.highlights {
                println!("       {}", highlight);
            }
        }
    }
}

// ==================== VectorDB Client Implementation ====================

pub mod client {
    use super::*;
    use reqwest;
    use serde_json;

    pub struct VectorDBClient {
        base_url: String,
        client: reqwest::Client,
    }

    impl VectorDBClient {
        pub fn new(base_url: &str) -> Result<Self, Box<dyn std::error::Error>> {
            Ok(Self {
                base_url: base_url.to_string(),
                client: reqwest::Client::new(),
            })
        }

        pub async fn create_table(
            &self,
            request: CreateTableRequest,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let url = format!("{}/tables", self.base_url);
            let response = self.client.post(&url).json(&request).send().await?;

            if !response.status().is_success() {
                return Err(format!("Failed to create table: {}", response.text().await?).into());
            }

            Ok(())
        }

        pub async fn insert_records(
            &self,
            table: &str,
            records: Vec<CreateRecordRequest>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let url = format!("{}/tables/{}/records", self.base_url, table);
            let body = serde_json::json!({
                "records": records,
                "options": {
                    "parallel": true,
                    "batch_size": 100,
                }
            });

            let response = self.client.post(&url).json(&body).send().await?;

            if !response.status().is_success() {
                return Err(format!("Failed to insert records: {}", response.text().await?).into());
            }

            Ok(())
        }

        pub async fn search(
            &self,
            table: &str,
            query: SearchQuery,
        ) -> Result<super::SearchResult, Box<dyn std::error::Error>> {
            let url = format!("{}/tables/{}/search", self.base_url, table);
            let response = self.client.post(&url).json(&query).send().await?;

            if !response.status().is_success() {
                return Err(format!("Search failed: {}", response.text().await?).into());
            }

            let result = response.json().await?;
            Ok(result)
        }

        pub async fn update_record(
            &self,
            table: &str,
            id: &str,
            vector: Option<Vec<f32>>,
            payload: Option<serde_json::Value>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let url = format!("{}/tables/{}/records/{}", self.base_url, table, id);
            let body = serde_json::json!({
                "vector": vector,
                "payload": payload,
            });

            let response = self.client.put(&url).json(&body).send().await?;

            if !response.status().is_success() {
                return Err(format!("Failed to update record: {}", response.text().await?).into());
            }

            Ok(())
        }

        pub async fn delete_records(
            &self,
            table: &str,
            ids: &[String],
        ) -> Result<(), Box<dyn std::error::Error>> {
            let url = format!("{}/tables/{}/records", self.base_url, table);
            let body = serde_json::json!({
                "ids": ids,
            });

            let response = self.client.delete(&url).json(&body).send().await?;

            if !response.status().is_success() {
                return Err(format!("Failed to delete records: {}", response.text().await?).into());
            }

            Ok(())
        }
    }
}
