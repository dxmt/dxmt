//! MCP Protocol Demo - AI Assistant Integration
//!
//! This example shows how AI assistants like Claude can interact with DXMT VectorDB
//! through the Model Context Protocol (MCP).

use axum::{routing::get, Router};
use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    mcp::server::{mcp_websocket_handler, McpServer},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use std::net::SocketAddr;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    println!("Starting MCP Demo Server");

    // Initialize storage
    let config = StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 256,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };
    let storage = Arc::new(IntelligentIndexedStorage::new(&config).await?);

    // Create sample table
    storage
        .create_table(TableConfig {
            name: "products".to_string(),
            dimension: 4,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
        })
        .await?;

    // Insert sample products
    let records = vec![
        CreateRecordRequest {
            id: "laptop_1".to_string(),
            vector: vec![0.1, 0.2, 0.3, 0.4],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "name": "MacBook Pro 16",
                "category": "Laptops",
                "price": 2499
            })),
            named_vectors: None,
        },
        CreateRecordRequest {
            id: "laptop_2".to_string(),
            vector: vec![0.15, 0.25, 0.35, 0.45],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "name": "ThinkPad X1 Carbon",
                "category": "Laptops",
                "price": 1899
            })),
            named_vectors: None,
        },
        CreateRecordRequest {
            id: "phone_1".to_string(),
            vector: vec![0.8, 0.1, 0.1, 0.2],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "name": "iPhone 15 Pro",
                "category": "Phones",
                "price": 1199
            })),
            named_vectors: None,
        },
    ];
    storage.insert_records("products", records).await?;
    println!("Sample data created: 3 products");

    // Demo: perform a vector search
    let query = SearchQuery {
        vector: vec![0.12, 0.22, 0.32, 0.42],
        top_k: 2,
        filter: None,
        with_payload: true,
        with_vector: false,
        limit: 2,
        include_metadata: true,
        search_mode: SearchMode::Normal,
        vector_name: None,
    };
    let results = storage.search("products", query).await?;
    println!("\nVector search results (top 2):");
    for hit in &results.hits {
        println!("  - {} (score: {:.3})", hit.id, hit.score);
    }

    // Start MCP server
    let mcp_server = McpServer::new(storage);
    let app = Router::new()
        .route("/mcp", get(mcp_websocket_handler))
        .with_state(Arc::new(mcp_server));

    let bind_addr: SocketAddr = "0.0.0.0:9998".parse()?;
    println!("\nMCP server listening at ws://localhost:9998/mcp");
    println!("(Press Ctrl+C to stop)\n");
    print_example_interactions();

    axum::Server::bind(&bind_addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

fn print_example_interactions() {
    println!("Example MCP interactions:");
    println!(r#"  1. {{ "method": "initialize", "params": {{ "protocol_version": "1.0.0" }} }}"#);
    println!(r#"  2. {{ "method": "tools/list" }}"#);
    println!(
        r#"  3. {{ "method": "tools/call", "params": {{ "name": "vector_search", "arguments": {{ "table": "products", "vector": [0.12, 0.22, 0.32, 0.42], "limit": 5 }} }} }}"#
    );
}
