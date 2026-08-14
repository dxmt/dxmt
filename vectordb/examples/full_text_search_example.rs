use dxmt_vectordb::search::full_text::{
    FullTextConfig, FullTextDocument, FullTextSearchEngine, FullTextSearchRequest,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== DXMT Full-Text Search Example ===\n");

    // 1. Create full-text search engine
    println!("1. Setting up standalone full-text search engine...");

    let fulltext_config = FullTextConfig {
        index_path: "/tmp/dxmt_example_fulltext".to_string(),
        index_memory_mb: 64,
        enable_phrase_queries: true,
        enable_fuzzy_search: true,
        ..Default::default()
    };

    let text_engine = FullTextSearchEngine::new(fulltext_config.clone()).await?;

    // 2. Add documents
    let documents: Vec<FullTextDocument> = vec![
        FullTextDocument {
            id: "blog_1".to_string(),
            title: Some("Getting Started with Rust".to_string()),
            content: "Rust is a systems programming language that focuses on safety, speed, and concurrency.".to_string(),
            description: Some("An introduction to the Rust programming language".to_string()),
            tags: vec!["rust".to_string(), "programming".to_string(), "tutorial".to_string()],
            payload: Some(serde_json::json!({
                "author": "Alice Johnson",
                "date": "2024-01-15",
                "views": 1500
            })),
            timestamp: 1705334400,
        },
        FullTextDocument {
            id: "blog_2".to_string(),
            title: Some("Building Web APIs with Rust and Axum".to_string()),
            content: "Axum is a web framework for Rust that makes it easy to build robust and performant web services.".to_string(),
            description: Some("Learn how to create REST APIs using Axum".to_string()),
            tags: vec!["rust".to_string(), "web".to_string(), "api".to_string()],
            payload: Some(serde_json::json!({
                "author": "Bob Smith",
                "date": "2024-02-01",
                "views": 2100
            })),
            timestamp: 1706745600,
        },
        FullTextDocument {
            id: "blog_3".to_string(),
            title: Some("Introduction to Vector Databases".to_string()),
            content: "Vector databases are specialized systems for storing and searching high-dimensional embeddings, crucial for AI applications.".to_string(),
            description: Some("Understanding the fundamentals of vector databases".to_string()),
            tags: vec!["database".to_string(), "vectors".to_string(), "ai".to_string()],
            payload: Some(serde_json::json!({
                "author": "Charlie Davis",
                "date": "2024-03-10",
                "views": 3200
            })),
            timestamp: 1710028800,
        },
    ];

    println!("Adding {} documents to the index...", documents.len());
    for doc in documents {
        text_engine.add_document(doc).await?;
    }
    text_engine.commit().await?;

    // 3. Perform searches
    println!("\n2. Performing full-text searches...\n");

    println!("Search: 'rust programming'");
    let results: Vec<dxmt_vectordb::search::full_text::FullTextSearchResult> = text_engine
        .search(&FullTextSearchRequest {
            query: "rust programming".to_string(),
            search_fields: None,
            limit: 5,
            offset: 0,
            highlight: true,
            highlight_pre_tag: Some("**".to_string()),
            highlight_post_tag: Some("**".to_string()),
            fuzzy: None,
            filters: None,
            sort_by: None,
        })
        .await?;

    for (i, result) in results.iter().enumerate() {
        println!("  {}. {} (score: {:.3})", i + 1, result.id, result.score);
        if let Some(title) = &result.title {
            println!("     Title: {}", title);
        }
        if let Some(snippet) = &result.snippet {
            println!("     Snippet: {}", snippet);
        }
    }

    println!("\nSearch: '\"web framework\"' (phrase)");
    let phrase_results: Vec<dxmt_vectordb::search::full_text::FullTextSearchResult> = text_engine
        .search(&FullTextSearchRequest {
            query: "\"web framework\"".to_string(),
            search_fields: None,
            limit: 5,
            offset: 0,
            highlight: false,
            highlight_pre_tag: None,
            highlight_post_tag: None,
            fuzzy: None,
            filters: None,
            sort_by: None,
        })
        .await?;

    for result in &phrase_results {
        println!("  - {} (score: {:.3})", result.id, result.score);
    }

    // 4. Statistics
    let stats = text_engine.get_stats().await;
    println!("\n3. Search Statistics:");
    println!("  Total documents: {}", stats.total_documents);
    println!("  Total searches: {}", stats.total_searches);
    println!("  Avg search time: {:.2}ms", stats.avg_search_time_ms);

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
