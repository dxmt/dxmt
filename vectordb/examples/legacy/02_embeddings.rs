/// Embeddings Example - Working with Different Embedding Providers
///
/// This example demonstrates:
/// - Using OpenAI embeddings
/// - Using Cohere embeddings  
/// - Auto-embedding with fallback
/// - Multimodal embeddings (text, image)

use vectordb::{
    embeddings::{
        AutoEmbedding, AutoEmbeddingConfig,
        openai::OpenAIEmbedding,
        cohere::CohereEmbedding,
        EmbeddingFunction,
    },
    storage::{MemoryStorage, Storage},
    types::{
        table::{TableConfig, TableSchema},
        record::CreateRecordRequest,
        DistanceMetric,
    },
    Result,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎯 DXMT VectorDB - Embeddings Example\n");
    
    // Initialize storage
    let storage = Arc::new(MemoryStorage::new());
    
    // Example 1: OpenAI Embeddings
    println!("1️⃣ OpenAI Embeddings Example");
    println!("   Note: Set OPENAI_API_KEY environment variable");
    
    if std::env::var("OPENAI_API_KEY").is_ok() {
        let openai = OpenAIEmbedding::new(
            std::env::var("OPENAI_API_KEY").unwrap(),
            "text-embedding-ada-002".to_string(),
        );
        
        // Create table for OpenAI embeddings (1536 dimensions)
        let openai_table = TableConfig {
            name: "openai_embeddings".to_string(),
            dimension: 1536,
            distance_metric: DistanceMetric::Cosine,
            schema: TableSchema::Exploratory,
            description: Some("OpenAI text embeddings".to_string()),
            metadata: None,
        };
        
        storage.create_table(openai_table.clone()).await?;
        
        // Generate embeddings for sample texts
        let texts = vec![
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming industries",
            "Vector databases enable semantic search",
        ];
        
        for (i, text) in texts.iter().enumerate() {
            match openai.embed(text).await {
                Ok(embedding) => {
                    let record = CreateRecordRequest {
                        id: format!("openai_{}", i),
                        vector: embedding,
                        sparse_vector: None,
                        payload: Some(json!({
                            "text": text,
                            "model": "text-embedding-ada-002"
                        })),
                    };
                    storage.insert_record(&openai_table.name, record).await?;
                    println!("   ✅ Embedded: {}", text);
                }
                Err(e) => {
                    println!("   ⚠️ Error embedding text: {}", e);
                }
            }
        }
    } else {
        println!("   ⏭️ Skipping - OPENAI_API_KEY not set");
    }
    
    // Example 2: Cohere Embeddings
    println!("\n2️⃣ Cohere Embeddings Example");
    println!("   Note: Set COHERE_API_KEY environment variable");
    
    if std::env::var("COHERE_API_KEY").is_ok() {
        let cohere = CohereEmbedding::new(
            std::env::var("COHERE_API_KEY").unwrap(),
            "embed-english-v3.0".to_string(),
        );
        
        // Create table for Cohere embeddings (1024 dimensions)
        let cohere_table = TableConfig {
            name: "cohere_embeddings".to_string(),
            dimension: 1024,
            distance_metric: DistanceMetric::Cosine,
            schema: TableSchema::Exploratory,
            description: Some("Cohere text embeddings".to_string()),
            metadata: None,
        };
        
        storage.create_table(cohere_table.clone()).await?;
        
        // Generate embeddings
        let text = "Natural language processing with transformers";
        match cohere.embed(text).await {
            Ok(embedding) => {
                let record = CreateRecordRequest {
                    id: "cohere_001".to_string(),
                    vector: embedding,
                    sparse_vector: None,
                    payload: Some(json!({
                        "text": text,
                        "model": "embed-english-v3.0"
                    })),
                };
                storage.insert_record(&cohere_table.name, record).await?;
                println!("   ✅ Embedded text with Cohere");
            }
            Err(e) => {
                println!("   ⚠️ Error: {}", e);
            }
        }
    } else {
        println!("   ⏭️ Skipping - COHERE_API_KEY not set");
    }
    
    // Example 3: Auto-Embedding with Fallback
    println!("\n3️⃣ Auto-Embedding with Fallback");
    
    let auto_config = AutoEmbeddingConfig::default();
    let auto_embedder = AutoEmbedding::new(auto_config).await?;
    
    // Create table for auto embeddings
    let auto_table = TableConfig {
        name: "auto_embeddings".to_string(),
        dimension: 384,  // Default dimension for auto-embedder
        distance_metric: DistanceMetric::Cosine,
        schema: TableSchema::Exploratory,
        description: Some("Auto-selected embeddings".to_string()),
        metadata: None,
    };
    
    storage.create_table(auto_table.clone()).await?;
    
    // Generate embeddings with automatic provider selection
    let documents = vec![
        ("doc1", "Artificial intelligence is the future"),
        ("doc2", "Deep learning models require large datasets"),
        ("doc3", "Neural networks mimic human brain function"),
    ];
    
    for (id, text) in documents {
        match auto_embedder.embed(text).await {
            Ok(embedding) => {
                let record = CreateRecordRequest {
                    id: id.to_string(),
                    vector: embedding,
                    sparse_vector: None,
                    payload: Some(json!({
                        "text": text,
                        "provider": auto_embedder.name()
                    })),
                };
                storage.insert_record(&auto_table.name, record).await?;
                println!("   ✅ Auto-embedded: {}", text);
            }
            Err(e) => {
                println!("   ⚠️ Error: {}", e);
            }
        }
    }
    
    // Example 4: Batch Embeddings
    println!("\n4️⃣ Batch Embeddings Example");
    
    let batch_texts = vec![
        "First document about technology",
        "Second document about science",
        "Third document about innovation",
        "Fourth document about research",
        "Fifth document about development",
    ];
    
    println!("   Processing {} documents in batch...", batch_texts.len());
    
    // Simulate batch processing (would use actual batch API in production)
    let mut embeddings = Vec::new();
    for text in &batch_texts {
        // In production, use batch_embed method
        match auto_embedder.embed(text).await {
            Ok(emb) => embeddings.push(emb),
            Err(e) => println!("   ⚠️ Failed to embed: {}", e),
        }
    }
    
    println!("   ✅ Generated {} embeddings", embeddings.len());
    
    // Example 5: Embedding Metadata and Configuration
    println!("\n5️⃣ Embedding Configuration");
    
    println!("   Available embedding providers:");
    println!("   • OpenAI (text-embedding-ada-002): 1536 dimensions");
    println!("   • Cohere (embed-english-v3.0): 1024 dimensions");
    println!("   • SentenceTransformers (all-MiniLM-L6-v2): 384 dimensions");
    println!("   • Auto-Embedding: Automatic selection with fallback");
    
    // Display embedding statistics
    println!("\n📊 Embedding Statistics:");
    let tables = storage.list_tables().await?;
    for table in tables {
        println!("   Table: {} ({}D)", table.name, table.dimension);
    }
    
    println!("\n✨ Embeddings example completed!");
    
    Ok(())
}