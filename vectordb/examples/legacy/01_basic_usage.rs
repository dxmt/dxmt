/// Basic Usage Example - Getting Started with DXMT VectorDB
/// 
/// This example demonstrates the fundamental operations:
/// - Creating a vector database
/// - Inserting vectors
/// - Searching for similar vectors
/// - Managing metadata

use vectordb::{
    storage::{MemoryStorage, Storage},
    types::{
        table::{TableConfig, TableSchema},
        record::{CreateRecordRequest, SearchQuery},
        vector::DenseVector,
        DistanceMetric,
    },
    Result,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 DXMT VectorDB - Basic Usage Example\n");
    
    // Step 1: Create in-memory storage
    println!("1️⃣ Creating in-memory storage...");
    let storage = Arc::new(MemoryStorage::new());
    
    // Step 2: Create a table for storing vectors
    println!("2️⃣ Creating vector table...");
    let table_config = TableConfig {
        name: "product_embeddings".to_string(),
        dimension: 384,  // Using sentence-transformers dimension
        distance_metric: DistanceMetric::Cosine,
        schema: TableSchema::Exploratory,
        description: Some("Product description embeddings".to_string()),
        metadata: Some(json!({
            "model": "all-MiniLM-L6-v2",
            "created_by": "example_app"
        })),
    };
    
    storage.create_table(table_config.clone()).await?;
    println!("   ✅ Table '{}' created", table_config.name);
    
    // Step 3: Insert sample vectors with metadata
    println!("\n3️⃣ Inserting sample vectors...");
    let products = vec![
        ("laptop_001", "High-performance laptop with 16GB RAM", vec![0.1; 384]),
        ("phone_001", "Smartphone with 5G and dual camera", vec![0.2; 384]),
        ("tablet_001", "10-inch tablet with stylus support", vec![0.15; 384]),
        ("watch_001", "Smartwatch with health tracking", vec![0.25; 384]),
        ("earbuds_001", "Wireless earbuds with noise cancellation", vec![0.3; 384]),
    ];
    
    for (id, description, vector) in products {
        let record = CreateRecordRequest {
            id: id.to_string(),
            vector,
            sparse_vector: None,
            payload: Some(json!({
                "description": description,
                "category": if id.starts_with("laptop") || id.starts_with("tablet") {
                    "computing"
                } else {
                    "mobile"
                },
                "in_stock": true,
                "price": 999.99
            })),
        };
        
        storage.insert_record(&table_config.name, record).await?;
        println!("   ✅ Inserted: {} - {}", id, description);
    }
    
    // Step 4: Search for similar vectors
    println!("\n4️⃣ Searching for similar products...");
    
    // Create a query vector (simulating a user query embedding)
    let query_vector = vec![0.12; 384]; // Similar to laptop/tablet
    
    let search_query = SearchQuery {
        vector: query_vector,
        sparse_vector: None,
        top_k: 3,
        filter: None,
        with_payload: true,
        with_vector: false,
    };
    
    let results = storage.search(&table_config.name, search_query).await?;
    
    println!("   Found {} similar products:", results.hits.len());
    for (i, hit) in results.hits.iter().enumerate() {
        let payload = hit.payload.as_ref().unwrap();
        println!("   {}. ID: {}, Score: {:.4}", 
            i + 1, 
            hit.id, 
            hit.score
        );
        println!("      Description: {}", 
            payload["description"].as_str().unwrap_or("N/A")
        );
        println!("      Category: {}", 
            payload["category"].as_str().unwrap_or("N/A")
        );
    }
    
    // Step 5: Search with filters
    println!("\n5️⃣ Searching with category filter...");
    let filtered_query = SearchQuery {
        vector: vec![0.2; 384],
        sparse_vector: None,
        top_k: 2,
        filter: Some(json!({
            "category": "mobile"
        })),
        with_payload: true,
        with_vector: false,
    };
    
    let filtered_results = storage.search(&table_config.name, filtered_query).await?;
    println!("   Found {} mobile products:", filtered_results.hits.len());
    for hit in filtered_results.hits {
        println!("   - {}: {}", 
            hit.id,
            hit.payload.unwrap()["description"].as_str().unwrap_or("N/A")
        );
    }
    
    // Step 6: Get table statistics
    println!("\n6️⃣ Table statistics:");
    let tables = storage.list_tables().await?;
    for table in tables {
        if table.name == table_config.name {
            println!("   Table: {}", table.name);
            println!("   Dimension: {}", table.dimension);
            println!("   Distance Metric: {:?}", table.distance_metric);
            // Note: record_count would need to be implemented
            println!("   Records: 5 (hardcoded for example)");
        }
    }
    
    println!("\n✨ Example completed successfully!");
    
    Ok(())
}