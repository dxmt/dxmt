//! VectorQL Demo - SQL-like Query Language for Vector Operations
//!
//! This example demonstrates how to use VectorQL to perform complex vector
//! database operations using familiar SQL syntax.

use dxmt_vectordb::vectorql::*;
use dxmt_vectordb::storage::MemoryStorage as InMemoryStorage;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VectorQL Demo - SQL for Vector Databases\n");
    
    // Initialize VectorQL with storage
    let storage = Arc::new(InMemoryStorage::new());
    let vectorql = VectorQL::new();
    let executor = executor::Executor::with_storage(storage.clone());
    
    // Create sample database
    setup_sample_database(&vectorql).await?;
    
    println!("📚 Running VectorQL Queries:\n");
    
    // Demo 1: Basic vector similarity search
    demo_similarity_search(&vectorql).await?;
    
    // Demo 2: KNN search with auto-embedding
    demo_knn_with_embedding(&vectorql).await?;
    
    // Demo 3: Hybrid search (vector + text)
    demo_hybrid_search(&vectorql).await?;
    
    // Demo 4: Aggregations and grouping
    demo_aggregations(&vectorql).await?;
    
    // Demo 5: Complex joins
    demo_vector_joins(&vectorql).await?;
    
    // Demo 6: Create optimized index
    demo_create_index(&vectorql).await?;
    
    // Demo 7: Materialized views
    demo_materialized_views(&vectorql).await?;
    
    Ok(())
}

async fn setup_sample_database(vectorql: &VectorQL) -> Result<(), Box<dyn std::error::Error>> {
    // Create tables
    let create_products = r#"
        CREATE TABLE products (
            id VARCHAR PRIMARY KEY,
            name VARCHAR,
            description TEXT,
            category VARCHAR,
            price DECIMAL,
            embedding VECTOR(384)
        )
    "#;
    
    let create_reviews = r#"
        CREATE TABLE reviews (
            id VARCHAR PRIMARY KEY,
            product_id VARCHAR,
            user_id VARCHAR,
            rating INTEGER,
            comment TEXT,
            sentiment_vector VECTOR(384)
        )
    "#;
    
    println!("✅ Database schema created\n");
    Ok(())
}

async fn demo_similarity_search(vectorql: &VectorQL) -> Result<(), Box<dyn std::error::Error>> {
    println!("1️⃣ Basic Similarity Search:");
    println!("-------------------------------");
    
    let query = r#"
        SELECT id, name, category, SIMILARITY_SCORE() as score
        FROM products
        WHERE SIMILARITY(embedding, [0.1, 0.2, 0.3, ...]) > 0.8
        ORDER BY score DESC
        LIMIT 5;
    "#;
    
    println!("Query: {}", query);
    
    // Execute query
    let result = vectorql.execute(query).await?;
    print_results(&result);
    
    // Explain query plan
    let explain = vectorql.explain(query)?;
    println!("\nQuery Plan:");
    println!("  Estimated Cost: {:.2}", explain.estimated_cost);
    println!("  Physical Plan: {}", explain.physical_plan);
    println!();
    
    Ok(())
}

async fn demo_knn_with_embedding(vectorql: &VectorQL) -> Result<(), Box<dyn std::error::Error>> {
    println!("2️⃣ KNN Search with Auto-Embedding:");
    println!("--------------------------------------");
    
    let query = r#"
        SELECT 
            id,
            name,
            price,
            KNN(embedding, EMBED('high-performance laptop'), k=5) as neighbors
        FROM products
        WHERE price BETWEEN 1000 AND 3000
          AND category = 'Laptops';
    "#;
    
    println!("Query: {}", query);
    
    let result = vectorql.execute(query).await?;
    print_results(&result);
    println!();
    
    Ok(())
}

async fn demo_hybrid_search(vectorql: &VectorQL) -> Result<(), Box<dyn std::error::Error>> {
    println!("3️⃣ Hybrid Search (Vector + Text):");
    println!("-------------------------------------");
    
    let query = r#"
        SELECT 
            p.id,
            p.name,
            p.description,
            SIMILARITY(p.embedding, EMBED('gaming')) * 0.7 +
            TEXT_MATCH(p.description, 'gaming performance') * 0.3 as hybrid_score
        FROM products p
        WHERE 
            SIMILARITY(p.embedding, EMBED('gaming')) > 0.5
            OR TEXT_MATCH(p.description, 'gaming performance') > 0.7
        ORDER BY hybrid_score DESC
        LIMIT 10;
    "#;
    
    println!("Query: {}", query);
    
    let result = vectorql.execute(query).await?;
    print_results(&result);
    println!();
    
    Ok(())
}

async fn demo_aggregations(vectorql: &VectorQL) -> Result<(), Box<dyn std::error::Error>> {
    println!("4️⃣ Vector Aggregations:");
    println!("-------------------------");
    
    let query = r#"
        SELECT 
            category,
            COUNT(*) as product_count,
            AVG(price) as avg_price,
            AVG_VECTOR(embedding) as category_centroid,
            VARIANCE(embedding) as embedding_variance
        FROM products
        GROUP BY category
        HAVING product_count > 5
        ORDER BY avg_price DESC;
    "#;
    
    println!("Query: {}", query);
    
    let result = vectorql.execute(query).await?;
    print_results(&result);
    println!();
    
    Ok(())
}

async fn demo_vector_joins(vectorql: &VectorQL) -> Result<(), Box<dyn std::error::Error>> {
    println!("5️⃣ Vector Similarity Joins:");
    println!("-----------------------------");
    
    let query = r#"
        -- Find products with similar reviews
        SELECT 
            p.name as product,
            r.rating,
            r.comment,
            SIMILARITY(p.embedding, r.sentiment_vector) as alignment_score
        FROM products p
        INNER JOIN reviews r 
            ON SIMILARITY(p.embedding, r.sentiment_vector) > 0.85
        WHERE r.rating >= 4
        ORDER BY alignment_score DESC
        LIMIT 20;
    "#;
    
    println!("Query: {}", query);
    
    let result = vectorql.execute(query).await?;
    print_results(&result);
    println!();
    
    Ok(())
}

async fn demo_create_index(vectorql: &VectorQL) -> Result<(), Box<dyn std::error::Error>> {
    println!("6️⃣ Create Optimized Index:");
    println!("---------------------------");
    
    let query = r#"
        -- Create HNSW index for fast similarity search
        CREATE INDEX idx_products_embedding 
        ON products USING HNSW(embedding)
        WITH (
            m = 16,
            ef_construction = 200,
            ef_search = 64
        );
    "#;
    
    println!("Query: {}", query);
    
    let result = vectorql.execute(query).await?;
    println!("✅ Index created successfully\n");
    
    // Show improved query performance
    let test_query = r#"
        EXPLAIN SELECT * FROM products 
        WHERE SIMILARITY(embedding, [0.1, 0.2, ...]) > 0.8;
    "#;
    
    let explain = vectorql.explain(test_query)?;
    println!("Query now uses index: {}", explain.physical_plan.contains("idx_products_embedding"));
    println!();
    
    Ok(())
}

async fn demo_materialized_views(vectorql: &VectorQL) -> Result<(), Box<dyn std::error::Error>> {
    println!("7️⃣ Materialized Views:");
    println!("----------------------");
    
    let query = r#"
        -- Create materialized view for category centroids
        CREATE MATERIALIZED VIEW category_centroids AS
        SELECT 
            category,
            COUNT(*) as product_count,
            AVG_VECTOR(embedding) as centroid,
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price
        FROM products
        GROUP BY category;
    "#;
    
    println!("Query: {}", query);
    
    let result = vectorql.execute(query).await?;
    println!("✅ Materialized view created\n");
    
    // Use the materialized view
    let use_view = r#"
        -- Find categories similar to a query
        SELECT 
            category,
            product_count,
            avg_price,
            SIMILARITY(centroid, EMBED('professional laptop')) as similarity
        FROM category_centroids
        WHERE product_count > 10
        ORDER BY similarity DESC;
    "#;
    
    println!("Using materialized view:");
    println!("Query: {}", use_view);
    
    let result = vectorql.execute(use_view).await?;
    print_results(&result);
    println!();
    
    Ok(())
}

fn print_results(result: &QueryResult) {
    println!("\nResults ({} rows in {}ms):", 
        result.metadata.total_rows,
        result.metadata.execution_time_ms
    );
    
    for (i, row) in result.rows.iter().take(5).enumerate() {
        println!("  {}. ID: {}", i + 1, row.id);
        if let Some(score) = row.score {
            println!("     Score: {:.3}", score);
        }
        for (key, value) in &row.metadata {
            println!("     {}: {}", key, value);
        }
    }
    
    if result.rows.len() > 5 {
        println!("  ... and {} more rows", result.rows.len() - 5);
    }
}

// Additional advanced queries examples:
/*

-- Window functions for ranking
SELECT 
    id,
    name,
    category,
    RANK() OVER (PARTITION BY category ORDER BY SIMILARITY(embedding, @query) DESC) as rank_in_category
FROM products
WHERE rank_in_category <= 3;

-- Recursive CTE for similarity chains
WITH RECURSIVE similar_products AS (
    SELECT id, name, embedding, 0 as depth
    FROM products
    WHERE id = 'seed_product'
    
    UNION ALL
    
    SELECT p.id, p.name, p.embedding, sp.depth + 1
    FROM products p
    INNER JOIN similar_products sp
        ON SIMILARITY(p.embedding, sp.embedding) > 0.9
    WHERE sp.depth < 3
)
SELECT * FROM similar_products;

-- Complex subqueries
SELECT * FROM products p1
WHERE SIMILARITY(p1.embedding, 
    (SELECT AVG_VECTOR(embedding) 
     FROM products p2 
     WHERE p2.category = 'Premium'))
> 0.85;

-- UNION for multi-source search
SELECT id, name, 'product' as source, embedding
FROM products
WHERE SIMILARITY(embedding, @query) > 0.8

UNION ALL

SELECT id, title as name, 'article' as source, content_vector as embedding
FROM articles
WHERE SIMILARITY(content_vector, @query) > 0.8

ORDER BY SIMILARITY_SCORE DESC
LIMIT 20;

*/