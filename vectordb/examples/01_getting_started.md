# Getting Started with DXMT VectorDB

Welcome to DXMT VectorDB! This tutorial will guide you through your first steps with our high-performance vector database.

## Table of Contents
1. [Installation](#installation)
2. [Your First Vector Database](#your-first-vector-database)
3. [Understanding Core Concepts](#understanding-core-concepts)
4. [Basic Operations](#basic-operations)
5. [Next Steps](#next-steps)

## Installation

### Prerequisites
- Rust 1.70+ (install from [rustup.rs](https://rustup.rs/))
- 4GB RAM minimum (8GB recommended)
- 10GB free disk space

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/dxmt-vectordb.git
cd dxmt-vectordb

# Build the project
cargo build --release

# Run tests to verify installation
cargo test
```

### Add to Your Project

Add DXMT VectorDB to your `Cargo.toml`:

```toml
[dependencies]
dxmt-vectordb = "0.1.0"
tokio = { version = "1", features = ["full"] }
serde_json = "1.0"
```

## Your First Vector Database

Let's create a simple application that stores and searches product embeddings:

```rust
use dxmt_vectordb::{
    storage::{MemoryStorage, Storage},
    types::{
        table::{TableConfig, TableSchema},
        record::{CreateRecordRequest, SearchQuery},
        DistanceMetric,
    },
    Result,
};
use std::sync::Arc;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Create storage backend
    let storage = Arc::new(MemoryStorage::new());
    
    // 2. Define your table
    let config = TableConfig {
        name: "products".to_string(),
        dimension: 384,  // Embedding dimension
        distance_metric: DistanceMetric::Cosine,
        schema: TableSchema::Exploratory,
        description: Some("Product catalog embeddings".to_string()),
        metadata: None,
    };
    
    // 3. Create the table
    storage.create_table(config.clone()).await?;
    
    // 4. Insert a vector
    let product = CreateRecordRequest {
        id: "product_001".to_string(),
        vector: vec![0.1; 384],  // Your embedding here
        sparse_vector: None,
        payload: Some(json!({
            "name": "Laptop",
            "price": 999.99,
            "category": "Electronics"
        })),
    };
    
    storage.insert_record(&config.name, product).await?;
    
    // 5. Search for similar vectors
    let query = SearchQuery {
        vector: vec![0.12; 384],  // Query embedding
        sparse_vector: None,
        top_k: 5,
        filter: None,
        with_payload: true,
        with_vector: false,
    };
    
    let results = storage.search(&config.name, query).await?;
    
    // 6. Process results
    for hit in results.hits {
        println!("Found: {} with score {}", hit.id, hit.score);
        if let Some(payload) = hit.payload {
            println!("  Product: {}", payload["name"]);
        }
    }
    
    Ok(())
}
```

## Understanding Core Concepts

### Vectors
Vectors are numerical representations of data (text, images, audio) that capture semantic meaning:
- **Dense Vectors**: Regular arrays of floating-point numbers
- **Sparse Vectors**: Vectors with many zero values (efficient storage)
- **Dimension**: Number of elements in the vector (e.g., 384, 768, 1536)

### Distance Metrics
How we measure similarity between vectors:
- **Cosine Similarity**: Measures angle between vectors (good for text)
- **Euclidean Distance**: Straight-line distance (good for dense features)
- **Dot Product**: Fast similarity for normalized vectors

### Tables
Collections that store vectors with the same dimension:
```rust
TableSchema::Exploratory  // Flexible, schema-less
TableSchema::Structured   // Defined schema, validated
TableSchema::Enterprise   // Full features, audit trail
```

### Storage Tiers
DXMT uses a three-tier architecture:
1. **Hot (Memory)**: Frequently accessed data, <5ms latency
2. **Warm (SSD)**: Medium frequency, <50ms latency
3. **Cold (Object Storage)**: Archive data, <500ms latency

## Basic Operations

### Creating Tables

```rust
// Simple table
let simple_table = TableConfig {
    name: "embeddings".to_string(),
    dimension: 768,
    distance_metric: DistanceMetric::Cosine,
    schema: TableSchema::Exploratory,
    description: None,
    metadata: None,
};

// Advanced table with metadata
let advanced_table = TableConfig {
    name: "documents".to_string(),
    dimension: 1536,
    distance_metric: DistanceMetric::Euclidean,
    schema: TableSchema::Structured,
    description: Some("Document embeddings with metadata".to_string()),
    metadata: Some(json!({
        "model": "text-embedding-ada-002",
        "language": "en",
        "index_type": "hnsw"
    })),
};
```

### Inserting Data

```rust
// Single record
let record = CreateRecordRequest {
    id: "doc_001".to_string(),
    vector: embedding_vector,
    sparse_vector: None,
    payload: Some(json!({
        "title": "Introduction to AI",
        "author": "John Doe",
        "date": "2024-01-15"
    })),
};
storage.insert_record("documents", record).await?;

// Batch insert
let records = vec![record1, record2, record3];
for record in records {
    storage.insert_record("documents", record).await?;
}
```

### Searching

```rust
// Basic search
let basic_search = SearchQuery {
    vector: query_embedding,
    sparse_vector: None,
    top_k: 10,
    filter: None,
    with_payload: true,
    with_vector: false,
};

// Filtered search
let filtered_search = SearchQuery {
    vector: query_embedding,
    sparse_vector: None,
    top_k: 5,
    filter: Some(json!({
        "category": "Electronics",
        "price": {"$lt": 500}
    })),
    with_payload: true,
    with_vector: false,
};

// Hybrid search (vector + sparse)
let hybrid_search = SearchQuery {
    vector: dense_embedding,
    sparse_vector: Some(sparse_embedding),
    top_k: 10,
    filter: None,
    with_payload: true,
    with_vector: true,
};
```

### Updating Records

```rust
// Update vector
storage.update_record(
    "documents",
    "doc_001",
    UpdateRecordRequest {
        vector: Some(new_embedding),
        payload: None,
    }
).await?;

// Update metadata
storage.update_record(
    "documents",
    "doc_001",
    UpdateRecordRequest {
        vector: None,
        payload: Some(json!({
            "status": "reviewed",
            "last_modified": "2024-01-20"
        })),
    }
).await?;
```

### Deleting Records

```rust
// Delete single record
storage.delete_record("documents", "doc_001").await?;

// Delete multiple records
let ids = vec!["doc_001", "doc_002", "doc_003"];
storage.delete_records("documents", ids).await?;

// Delete with filter
storage.delete_by_filter("documents", json!({
    "status": "archived"
})).await?;
```

## Best Practices

### 1. Choose the Right Dimension
- Smaller dimensions (384) = faster, less accurate
- Larger dimensions (1536) = slower, more accurate
- Match your embedding model's output

### 2. Select Appropriate Distance Metric
- Text similarity: Cosine
- Image similarity: Euclidean
- Normalized vectors: Dot Product

### 3. Optimize for Your Use Case
```rust
// High accuracy, slower
let accurate = SearchQuery {
    top_k: 100,  // Search more candidates
    ..Default::default()
};

// Fast, good enough
let fast = SearchQuery {
    top_k: 10,  // Fewer candidates
    ..Default::default()
};
```

### 4. Use Filters Effectively
```rust
// Pre-filter for performance
filter: Some(json!({
    "timestamp": {"$gte": "2024-01-01"},
    "status": "active"
}))
```

### 5. Monitor Performance
```rust
// Enable metrics
let config = ServerConfig {
    metrics_enabled: true,
    metrics_port: 9090,
    ..Default::default()
};
```

## Common Patterns

### Semantic Search
```rust
// Store documents with embeddings
let doc = CreateRecordRequest {
    id: content_hash,
    vector: text_embedding,
    payload: Some(json!({
        "content": document_text,
        "url": source_url,
        "timestamp": created_at
    })),
};

// Search by meaning
let results = storage.search("documents", SearchQuery {
    vector: query_embedding("What is machine learning?"),
    top_k: 10,
    ..Default::default()
}).await?;
```

### Recommendation System
```rust
// Store user preferences
let user_embedding = compute_user_embedding(user_history);

// Find similar items
let recommendations = storage.search("products", SearchQuery {
    vector: user_embedding,
    top_k: 20,
    filter: Some(json!({
        "in_stock": true,
        "category": {"$in": user_interests}
    })),
    ..Default::default()
}).await?;
```

### Duplicate Detection
```rust
// Check for similar content
let similar = storage.search("content", SearchQuery {
    vector: new_content_embedding,
    top_k: 1,
    ..Default::default()
}).await?;

if similar.hits[0].score > 0.95 {
    println!("Duplicate detected!");
}
```

## Troubleshooting

### Common Issues

**Issue**: Out of memory errors
```bash
# Increase memory limit
export DXMT_MEMORY_LIMIT=8GB
```

**Issue**: Slow searches
```rust
// Use indexing
metadata: Some(json!({
    "index_type": "hnsw",
    "index_params": {
        "m": 16,
        "ef_construction": 200
    }
}))
```

**Issue**: Connection errors
```rust
// Retry logic
use backoff::{retry, ExponentialBackoff};

retry(ExponentialBackoff::default(), || {
    storage.search(table, query).await
}).await?;
```

## Next Steps

Now that you understand the basics:

1. **[Tutorial 2: Working with Embeddings](./02_embeddings.md)** - Learn about different embedding models
2. **[Filtering](./filter_example.md)** - Metadata filters and the operators behind them

## Resources

- [Community Discord](https://discord.gg/dxmt)
- [GitHub Issues](https://github.com/your-org/dxmt-vectordb/issues)

## Getting Help

If you run into issues:
1. Search [existing issues](https://github.com/your-org/dxmt-vectordb/issues)
2. Join our [Discord community](https://discord.gg/dxmt)
3. Create a [new issue](https://github.com/your-org/dxmt-vectordb/issues/new)

Happy vector searching! 🚀