//! What DXQL parses, compiles and executes -- and what it refuses.
//!
//! Every statement below runs against a real in-memory table, through the same three stages the
//! server's `POST /query` uses: parse, compile, execute. That matters, because `DxqlEngine::new`
//! wires the *storage-free* executor, which returns an empty result for every plan; a demo built
//! on it printed "Results: 0 records found" for a SELECT and looked like an empty table rather
//! than like nothing having happened.
//!
//! The refusals are the point as much as the successes. DXQL says no to vector search, text
//! search and graph traversal in a WHERE clause, and names the route to use instead, rather than
//! accepting the syntax and quietly doing something else.

use dxmt_vectordb::config::{MemoryConfig, StorageConfig};
use dxmt_vectordb::dxql::{compiler::Compiler, executor::StorageAwareExecutor, parser::Parser};
use dxmt_vectordb::dxql::{DxqlError, QueryResult};
use dxmt_vectordb::storage::IntelligentIndexedStorage;
use std::sync::Arc;

/// Parse, compile, execute: the three stages, wired the way `POST /query` wires them.
async fn run(executor: &StorageAwareExecutor, query: &str) -> Result<QueryResult, DxqlError> {
    let statement = Parser::new().parse(query)?;
    let plan = Compiler::new().compile(statement)?;
    executor.execute(plan).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DXQL Query Language Demo");
    println!("========================\n");

    let storage = Arc::new(
        IntelligentIndexedStorage::new(&StorageConfig {
            r#type: "memory".to_string(),
            memory: Some(MemoryConfig {
                max_size_mb: 512,
                eviction_policy: "lru".to_string(),
            }),
            ..Default::default()
        })
        .await?,
    );
    let executor = StorageAwareExecutor::new(
        storage.clone() as Arc<dyn dxmt_vectordb::storage::traits::_Storage>
    );

    // Two tables for the statements below to act on. Created through DXQL itself, which is
    // also section 9's subject.
    // Only `products`: section 9 creates `documents` itself, and the WHERE-clause sections
    // that name it are refused before storage is ever consulted.
    run(
        &executor,
        "CREATE TABLE products (id TEXT, vector VECTOR(3), price FLOAT, category TEXT)",
    )
    .await?;
    run(
        &executor,
        "INSERT INTO products (id, vector, price, category) \
         VALUES ('prod_123', [0.9, 0.1, 0.0], 149.99, 'wearables')",
    )
    .await?;
    println!("(set up: table 'products' with one row)\n");

    // Example 1: Simple SELECT query
    println!("1. Simple SELECT Query:");
    let query = "SELECT * FROM products WHERE price > 100 LIMIT 10";
    println!("Query: {}", query);
    match run(&executor, query).await {
        Ok(result) => print_result(&result),
        Err(e) => println!("Error: {}", e),
    }

    // Example 2: Vector similarity search
    println!("\n2. Vector Similarity Search:");
    let query = r#"
        SELECT id, name, score
        FROM documents
        WHERE vector_similarity(embedding, [0.1, 0.2, 0.3, 0.4]) > 0.8
        ORDER BY score DESC
        LIMIT 5
    "#;
    println!("Query: {}", query);
    match run(&executor, query).await {
        Ok(result) => print_result(&result),
        Err(e) => println!("Error: {}", e),
    }

    // Example 3: Text search with BM25
    println!("\n3. Text Search:");
    let query = r#"
        SELECT id, title, content
        FROM articles
        WHERE text_match(content, "machine learning")
        ORDER BY score DESC
        LIMIT 10
    "#;
    println!("Query: {}", query);
    match run(&executor, query).await {
        Ok(result) => print_result(&result),
        Err(e) => println!("Error: {}", e),
    }

    // Example 4: Hybrid search combining vector and text
    println!("\n4. Hybrid Search:");
    let query = r#"
        SELECT id, title, score
        FROM papers
        WHERE vector_similarity(embedding, $query_vec) > 0.7
          AND text_match(abstract, "neural networks")
        ORDER BY hybrid_score(0.3, 0.7, 0.0)
        LIMIT 20
    "#;
    println!("Query: {}", query);
    match run(&executor, query).await {
        Ok(result) => print_result(&result),
        Err(e) => println!("Error: {}", e),
    }

    // Example 5: Graph distance query
    println!("\n5. Graph Distance Query:");
    let query = r#"
        SELECT id, name
        FROM users
        WHERE graph_distance(connections, "user_123") < 3
        ORDER BY distance ASC
        LIMIT 50
    "#;
    println!("Query: {}", query);
    match run(&executor, query).await {
        Ok(result) => print_result(&result),
        Err(e) => println!("Error: {}", e),
    }

    // Example 6: Complex filtering with multiple conditions
    println!("\n6. Complex Filtering:");
    let query = r#"
        SELECT id, title, category, price
        FROM products
        WHERE (category = "electronics" OR category = "computers")
          AND price BETWEEN 500 AND 2000
          AND vector_similarity(features, $search_vec) > 0.6
          AND rating > 4.0
        ORDER BY price ASC
        LIMIT 15
    "#;
    println!("Query: {}", query);
    match run(&executor, query).await {
        Ok(result) => print_result(&result),
        Err(e) => println!("Error: {}", e),
    }

    // Example 7: INSERT statement
    println!("\n7. INSERT Statement:");
    let query = r#"
        INSERT INTO products (id, vector, name, price, category)
        VALUES ('prod_456', [0.1, 0.2, 0.3], 'Smart Watch', 299.99, 'wearables')
    "#;
    println!("Query: {}", query);
    match run(&executor, query).await {
        Ok(result) => print_result(&result),
        Err(e) => println!("Error: {}", e),
    }

    // Example 8: UPDATE statement
    println!("\n8. UPDATE Statement:");
    let query = r#"
        UPDATE products
        SET price = 249.99, updated_at = NOW()
        WHERE id = 'prod_456'
    "#;
    println!("Query: {}", query);
    match run(&executor, query).await {
        Ok(result) => print_result(&result),
        Err(e) => println!("Error: {}", e),
    }

    // Example 9: CREATE TABLE with vector column
    println!("\n9. CREATE TABLE:");
    let query = r#"
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT,
            embedding VECTOR(384) NOT NULL,
            metadata JSON,
            created_at TIMESTAMP DEFAULT NOW()
        )
    "#;
    println!("Query: {}", query);
    match run(&executor, query).await {
        Ok(result) => print_result(&result),
        Err(e) => println!("Error: {}", e),
    }

    // Example 10: CREATE INDEX
    println!("\n10. CREATE INDEX:");
    let query = r#"
        CREATE HNSW INDEX embedding_idx
        ON documents(embedding)
        WITH (m = 16, ef_construct = 200)
    "#;
    println!("Query: {}", query);
    match run(&executor, query).await {
        Ok(result) => print_result(&result),
        Err(e) => println!("Error: {}", e),
    }

    Ok(())
}

fn print_result(result: &QueryResult) {
    // SELECT returns records; INSERT, UPDATE, DELETE and CREATE move rows and return none. The
    // first version of this printed the record count for all of them, so a successful INSERT
    // announced "0 records found" and read like a failure.
    match result.metadata.plan_type.as_str() {
        "SELECT" => println!("Results: {} records found", result.records.len()),
        _ => println!("Affected rows: {}", result.metadata.affected_rows),
    }
    println!("Execution time: {}ms", result.metadata.execution_time_ms);
    println!("Plan type: {}", result.metadata.plan_type);

    if !result.records.is_empty() {
        println!("Sample records:");
        for (i, record) in result.records.iter().take(3).enumerate() {
            println!("  Record {}: {:?}", i + 1, record.id);
            if let Some(score) = record.score {
                println!("    Score: {:.4}", score);
            }
            for (key, value) in &record.fields {
                println!("    {}: {:?}", key, value);
            }
        }
    }
}
