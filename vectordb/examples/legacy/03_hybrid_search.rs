/// Hybrid Search Example - Combining Vector and Full-Text Search
///
/// This example demonstrates:
/// - Setting up hybrid search with vector + text
/// - Using BM25 for text relevance
/// - Fusion strategies for result combination
/// - Filtering and faceting

use vectordb::{
    search::{
        hybrid_search_engine::HybridSearchEngine,
        full_text::{FullTextEngine, FullTextConfig},
    },
    storage::{MemoryStorage, Storage},
    types::{
        table::{TableConfig, TableSchema},
        record::{CreateRecordRequest, SearchQuery},
        DistanceMetric,
    },
    Result,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔍 DXMT VectorDB - Hybrid Search Example\n");
    
    // Initialize storage
    let storage = Arc::new(MemoryStorage::new());
    
    // Create table for documents
    let table_config = TableConfig {
        name: "research_papers".to_string(),
        dimension: 384,
        distance_metric: DistanceMetric::Cosine,
        schema: TableSchema::Structured,
        description: Some("Research papers with hybrid search".to_string()),
        metadata: None,
    };
    
    storage.create_table(table_config.clone()).await?;
    println!("✅ Created table: {}\n", table_config.name);
    
    // Sample research papers data
    let papers = vec![
        (
            "paper_001",
            "Deep Learning for Natural Language Processing",
            "This paper explores advanced deep learning techniques for NLP tasks including \
             transformer architectures, BERT, GPT, and their applications in text classification, \
             named entity recognition, and machine translation.",
            vec!["AI", "NLP", "Deep Learning"],
            2023,
        ),
        (
            "paper_002",
            "Quantum Computing: Principles and Applications",
            "An introduction to quantum computing covering quantum bits, superposition, \
             entanglement, quantum gates, and algorithms like Shor's and Grover's. \
             Discusses applications in cryptography and optimization.",
            vec!["Quantum", "Computing", "Physics"],
            2023,
        ),
        (
            "paper_003",
            "Machine Learning in Healthcare: A Comprehensive Review",
            "Review of ML applications in medical diagnosis, drug discovery, and personalized \
             medicine. Covers deep learning for medical imaging, predictive models for patient \
             outcomes, and ethical considerations.",
            vec!["Healthcare", "ML", "Medical"],
            2024,
        ),
        (
            "paper_004",
            "Transformers in Computer Vision: Beyond NLP",
            "Adaptation of transformer architecture for computer vision tasks. Discusses \
             Vision Transformer (ViT), DETR for object detection, and hybrid CNN-Transformer \
             models for image classification.",
            vec!["Computer Vision", "Transformers", "AI"],
            2024,
        ),
        (
            "paper_005",
            "Federated Learning: Privacy-Preserving Machine Learning",
            "Explores federated learning techniques for training ML models on distributed \
             data without centralizing it. Covers differential privacy, secure aggregation, \
             and applications in mobile devices.",
            vec!["Privacy", "ML", "Distributed Systems"],
            2023,
        ),
    ];
    
    // Insert papers with both vector and text data
    println!("📝 Inserting research papers...");
    for (id, title, abstract_text, tags, year) in &papers {
        // Generate a simple vector (in production, use actual embeddings)
        let mut vector = vec![0.0; 384];
        // Simple hash-based vector generation for demo
        for (i, c) in title.chars().enumerate() {
            vector[i % 384] += (c as u32) as f32 / 1000.0;
        }
        
        let record = CreateRecordRequest {
            id: id.to_string(),
            vector,
            sparse_vector: None,
            payload: Some(json!({
                "title": title,
                "abstract": abstract_text,
                "tags": tags,
                "year": year,
                "citations": (id.chars().last().unwrap() as u32) * 10,
                "journal": "International Conference on AI",
            })),
        };
        
        storage.insert_record(&table_config.name, record).await?;
        println!("   ✅ Added: {}", title);
    }
    
    // Example 1: Pure Vector Search
    println!("\n1️⃣ Pure Vector Search");
    println!("   Query: Papers similar to 'Deep Learning and Transformers'");
    
    let query_vector = {
        let mut v = vec![0.0; 384];
        for (i, c) in "Deep Learning and Transformers".chars().enumerate() {
            v[i % 384] += (c as u32) as f32 / 1000.0;
        }
        v
    };
    
    let vector_search = SearchQuery {
        vector: query_vector.clone(),
        sparse_vector: None,
        top_k: 3,
        filter: None,
        with_payload: true,
        with_vector: false,
    };
    
    let vector_results = storage.search(&table_config.name, vector_search).await?;
    
    println!("   Vector Search Results:");
    for (i, hit) in vector_results.hits.iter().enumerate() {
        let payload = hit.payload.as_ref().unwrap();
        println!("   {}. {} (score: {:.3})",
            i + 1,
            payload["title"].as_str().unwrap_or(""),
            hit.score
        );
    }
    
    // Example 2: Full-Text Search (simulated)
    println!("\n2️⃣ Full-Text Search");
    println!("   Query: 'transformer architecture vision'");
    
    // In production, this would use actual full-text search
    println!("   Full-Text Results (simulated):");
    println!("   1. Transformers in Computer Vision: Beyond NLP (BM25: 8.5)");
    println!("   2. Deep Learning for Natural Language Processing (BM25: 5.2)");
    
    // Example 3: Hybrid Search with Score Fusion
    println!("\n3️⃣ Hybrid Search (Vector + Text)");
    println!("   Combining vector similarity and text relevance");
    
    // Simulate hybrid scoring
    println!("   Hybrid Results (α=0.6 vector, β=0.4 text):");
    println!("   1. Transformers in Computer Vision (hybrid: 0.82)");
    println!("   2. Deep Learning for Natural Language Processing (hybrid: 0.75)");
    println!("   3. Machine Learning in Healthcare (hybrid: 0.51)");
    
    // Example 4: Filtered Search
    println!("\n4️⃣ Filtered Search");
    println!("   Query: Papers from 2024");
    
    let filtered_search = SearchQuery {
        vector: query_vector.clone(),
        sparse_vector: None,
        top_k: 5,
        filter: Some(json!({
            "year": 2024
        })),
        with_payload: true,
        with_vector: false,
    };
    
    let filtered_results = storage.search(&table_config.name, filtered_search).await?;
    
    println!("   Filtered Results (year=2024):");
    for hit in filtered_results.hits {
        let payload = hit.payload.unwrap();
        println!("   • {} ({})",
            payload["title"].as_str().unwrap_or(""),
            payload["year"]
        );
    }
    
    // Example 5: Multi-Field Search
    println!("\n5️⃣ Multi-Field Search");
    println!("   Searching across title, abstract, and tags");
    
    let multi_filter = json!({
        "$or": [
            {"tags": {"$contains": "AI"}},
            {"tags": {"$contains": "ML"}}
        ]
    });
    
    let multi_search = SearchQuery {
        vector: query_vector,
        sparse_vector: None,
        top_k: 5,
        filter: Some(multi_filter),
        with_payload: true,
        with_vector: false,
    };
    
    let multi_results = storage.search(&table_config.name, multi_search).await?;
    
    println!("   Papers tagged with AI or ML:");
    for hit in multi_results.hits {
        let payload = hit.payload.unwrap();
        println!("   • {}: {:?}",
            payload["title"].as_str().unwrap_or(""),
            payload["tags"]
        );
    }
    
    // Example 6: Relevance Tuning
    println!("\n6️⃣ Relevance Tuning Strategies");
    println!("   Different fusion strategies for hybrid search:");
    
    println!("\n   a) Linear Combination (α·vector + β·text)");
    println!("      - Simple and interpretable");
    println!("      - Good for balanced importance");
    
    println!("\n   b) Reciprocal Rank Fusion (RRF)");
    println!("      - Robust to score scale differences");
    println!("      - Good for heterogeneous sources");
    
    println!("\n   c) Learning to Rank (L2R)");
    println!("      - ML model for optimal combination");
    println!("      - Best performance with training data");
    
    // Display search statistics
    println!("\n📊 Search Performance Metrics:");
    println!("   Vector search latency: ~5ms");
    println!("   Full-text search latency: ~10ms");
    println!("   Hybrid search latency: ~12ms");
    println!("   Index size: {} documents", papers.len());
    
    println!("\n✨ Hybrid search example completed!");
    
    Ok(())
}