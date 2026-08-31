//! Quick start example for DXMT VectorDB
//!
//! Demonstrates basic usage: create table, insert vectors, search, delete.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DXMT VectorDB Quick Start Example\n");

    // 1. Create in-memory storage
    println!("1. Creating storage...");
    let config = StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 256,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };
    let storage = IntelligentIndexedStorage::new(&config).await?;
    println!("   Storage ready\n");

    // 2. Create table
    println!("2. Creating table 'quickstart_embeddings'...");
    storage
        .create_table(TableConfig {
            name: "quickstart_embeddings".to_string(),
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
    println!("   Table created\n");

    // 3. Insert sample vectors
    println!("3. Inserting sample vectors...");
    let samples = vec![
        (
            "doc1",
            vec![0.1_f32, 0.2, 0.3, 0.4],
            "Introduction to machine learning",
        ),
        (
            "doc2",
            vec![0.2, 0.1, 0.4, 0.3],
            "Deep learning fundamentals",
        ),
        (
            "doc3",
            vec![0.3, 0.3, 0.1, 0.5],
            "Natural language processing",
        ),
        ("doc4", vec![0.1, 0.4, 0.2, 0.6], "Computer vision basics"),
        ("doc5", vec![0.4, 0.1, 0.3, 0.2], "Reinforcement learning"),
    ];

    let records: Vec<CreateRecordRequest> = samples
        .iter()
        .map(|(id, vec, text)| CreateRecordRequest {
            id: id.to_string(),
            vector: normalize(vec),
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "text": text,
                "source": "documentation"
            })),
            named_vectors: None,
        })
        .collect();

    storage
        .insert_records("quickstart_embeddings", records)
        .await?;
    for (id, _, text) in &samples {
        println!("   Inserted: {} - {}", id, text);
    }
    println!();

    // 4. Similarity search
    println!("4. Performing similarity search...");
    let query_vec = normalize(&[0.15, 0.25, 0.35, 0.45]);
    let results = storage
        .search(
            "quickstart_embeddings",
            SearchQuery {
                vector: query_vec,
                top_k: 3,
                filter: None,
                with_payload: true,
                with_vector: false,
                limit: 3,
                include_metadata: true,
                search_mode: SearchMode::Normal,
                vector_name: None,
            },
        )
        .await?;

    println!("   Top 3 results:");
    for (i, hit) in results.hits.iter().enumerate() {
        let text = hit
            .payload
            .as_ref()
            .and_then(|p| p.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        println!("   {}. Score: {:.4} - {}", i + 1, hit.score, text);
    }
    println!();

    // 5. Unfiltered search (filters require Filter type from record module)
    println!("5. Performing another search...");
    let query_vec2 = normalize(&[0.35, 0.15, 0.45, 0.25]);
    let filtered = storage
        .search(
            "quickstart_embeddings",
            SearchQuery {
                vector: query_vec2,
                top_k: 2,
                filter: None,
                with_payload: true,
                with_vector: false,
                limit: 2,
                include_metadata: true,
                search_mode: SearchMode::Normal,
                vector_name: None,
            },
        )
        .await?;

    println!("   Filtered results:");
    for (i, hit) in filtered.hits.iter().enumerate() {
        let text = hit
            .payload
            .as_ref()
            .and_then(|p| p.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        println!("   {}. Score: {:.4} - {}", i + 1, hit.score, text);
    }
    println!();

    // 6. List tables
    println!("6. Listing tables...");
    let tables = storage.list_tables().await?;
    for table in &tables {
        println!(
            "   - {} (dim={}, records={})",
            table.name, table.dimension, table.record_count
        );
    }
    println!();

    // 7. Delete record
    println!("7. Deleting record 'doc5'...");
    storage
        .delete_records("quickstart_embeddings", vec!["doc5".to_string()])
        .await?;
    println!("   Deleted doc5\n");

    // 8. Drop table
    println!("8. Dropping table...");
    storage.delete_table("quickstart_embeddings").await?;
    println!("   Table dropped\n");

    println!("Quick start example completed successfully!");
    Ok(())
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}
