/// Concept Demonstration - Shows VectorDB concepts without async complexity
/// 
/// This example demonstrates the core concepts of vector databases
/// using simple, synchronous code that actually compiles and runs.

use std::collections::HashMap;
use serde_json::{json, Value as JsonValue};

type Vector = Vec<f32>;
type RecordId = String;

/// Simple in-memory vector storage
struct SimpleVectorDB {
    name: String,
    dimension: usize,
    records: HashMap<RecordId, (Vector, JsonValue)>,
}

impl SimpleVectorDB {
    fn new(name: &str, dimension: usize) -> Self {
        println!("📦 Creating table '{}' with dimension {}", name, dimension);
        Self {
            name: name.to_string(),
            dimension,
            records: HashMap::new(),
        }
    }
    
    fn insert(&mut self, id: &str, vector: Vector, metadata: JsonValue) -> Result<(), String> {
        if vector.len() != self.dimension {
            return Err(format!("Vector dimension {} doesn't match table dimension {}", 
                vector.len(), self.dimension));
        }
        
        self.records.insert(id.to_string(), (vector, metadata));
        println!("   ✅ Inserted record: {}", id);
        Ok(())
    }
    
    fn search(&self, query: &Vector, top_k: usize) -> Vec<(String, f32, JsonValue)> {
        if query.len() != self.dimension {
            println!("⚠️  Query dimension mismatch");
            return vec![];
        }
        
        // Calculate similarities
        let mut results: Vec<(String, f32, JsonValue)> = self.records
            .iter()
            .map(|(id, (vector, metadata))| {
                let score = cosine_similarity(query, vector);
                (id.clone(), score, metadata.clone())
            })
            .collect();
        
        // Sort by score (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top K
        results.truncate(top_k);
        results
    }
    
    fn search_with_filter<F>(&self, query: &Vector, top_k: usize, filter: F) -> Vec<(String, f32, JsonValue)> 
    where
        F: Fn(&JsonValue) -> bool,
    {
        let mut results = self.search(query, self.records.len());
        results.retain(|(_, _, metadata)| filter(metadata));
        results.truncate(top_k);
        results
    }
    
    fn stats(&self) {
        println!("\n📊 Table Statistics:");
        println!("   • Table name: {}", self.name);
        println!("   • Dimension: {}", self.dimension);
        println!("   • Total records: {}", self.records.len());
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Generate a simple embedding for text (mock)
fn embed_text(text: &str, dim: usize) -> Vector {
    // Simple hash-based embedding for demonstration
    let mut embedding = vec![0.0; dim];
    for (i, ch) in text.chars().enumerate() {
        let idx = i % dim;
        embedding[idx] += (ch as u32) as f32 / 1000.0;
    }
    
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    
    embedding
}

fn main() {
    println!("🚀 DXMT VectorDB - Concept Demonstration");
    println!("=========================================\n");
    
    // Create a vector database
    let mut db = SimpleVectorDB::new("products", 128);
    
    // Sample product data
    let products = vec![
        ("laptop_001", "High-performance gaming laptop with RTX 4090", "Electronics", 2999.99),
        ("phone_001", "Latest smartphone with 5G and AI camera", "Electronics", 1299.99),
        ("book_001", "Introduction to Machine Learning", "Books", 49.99),
        ("laptop_002", "Ultrabook for professionals with long battery", "Electronics", 1899.99),
        ("book_002", "Deep Learning with Python", "Books", 59.99),
        ("tablet_001", "10-inch tablet perfect for reading and browsing", "Electronics", 599.99),
    ];
    
    // Insert products
    println!("\n1️⃣ Inserting products...");
    for (id, description, category, price) in &products {
        let embedding = embed_text(description, 128);
        let metadata = json!({
            "description": description,
            "category": category,
            "price": price,
            "in_stock": true
        });
        
        db.insert(id, embedding, metadata).unwrap();
    }
    
    // Semantic search
    println!("\n2️⃣ Semantic Search: 'portable computer for work'");
    let query = embed_text("portable computer for work", 128);
    let results = db.search(&query, 3);
    
    for (i, (id, score, metadata)) in results.iter().enumerate() {
        println!("   {}. {} (similarity: {:.3})", i + 1, id, score);
        println!("      {}", metadata["description"]);
        println!("      Price: ${}", metadata["price"]);
    }
    
    // Filtered search
    println!("\n3️⃣ Filtered Search: Books only");
    let book_query = embed_text("learning artificial intelligence", 128);
    let book_results = db.search_with_filter(&book_query, 5, |metadata| {
        metadata["category"] == "Books"
    });
    
    for (i, (id, score, metadata)) in book_results.iter().enumerate() {
        println!("   {}. {} (similarity: {:.3})", i + 1, id, score);
        println!("      {}", metadata["description"]);
    }
    
    // Price range search
    println!("\n4️⃣ Price Range Search: Products under $1000");
    let budget_query = embed_text("affordable device", 128);
    let budget_results = db.search_with_filter(&budget_query, 5, |metadata| {
        metadata["price"].as_f64().unwrap_or(0.0) < 1000.0
    });
    
    for (i, (id, _, metadata)) in budget_results.iter().enumerate() {
        println!("   {}. {} - ${}", i + 1, id, metadata["price"]);
        println!("      {}", metadata["description"]);
    }
    
    // Show statistics
    db.stats();
    
    println!("\n✨ Demonstration completed!");
    println!("\n📝 Key Concepts Demonstrated:");
    println!("   • Vector embeddings for semantic meaning");
    println!("   • Cosine similarity for finding related items");
    println!("   • Metadata filtering for refined searches");
    println!("   • Combining semantic and attribute-based search");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }
    
    #[test]
    fn test_vector_db() {
        let mut db = SimpleVectorDB::new("test", 3);
        let vec1 = vec![1.0, 0.0, 0.0];
        let metadata = json!({"test": true});
        
        assert!(db.insert("test1", vec1.clone(), metadata.clone()).is_ok());
        
        let results = db.search(&vec1, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "test1");
        assert!((results[0].1 - 1.0).abs() < 0.001);
    }
}