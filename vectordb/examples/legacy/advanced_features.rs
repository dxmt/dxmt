//! Advanced features example for DXMT VectorDB
//! 
//! This example demonstrates advanced features including:
//! - Multimodal embeddings (text, image, audio)
//! - GPU acceleration
//! - Batch operations
//! - Hybrid search (dense + sparse)
//! - LLM framework integration

use dxmt_vectordb::{
    Client, MultimodalContent, MultimodalQuery, FusionStrategy,
    GpuConfig, SimdLevel, HybridSearchQuery, SparseVector,
};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 DXMT VectorDB Advanced Features Example\n");
    
    let client = Client::connect("http://localhost:3030").await?;
    
    // ===== 1. Multimodal Embeddings =====
    println!("1. Multimodal Embeddings Demo");
    println!("   Creating multimodal table...");
    
    let mm_table = client.create_multimodal_table(
        "multimodal_demo",
        768,
        vec!["text", "image", "audio"],
    ).await?;
    
    // Index text content
    let text_content = MultimodalContent::text("A beautiful sunset over the ocean".to_string());
    let text_id = client.index_multimodal(&mm_table.name, text_content).await?;
    println!("   ✅ Indexed text: {}", text_id);
    
    // Index image content (would be actual image bytes in production)
    let image_content = MultimodalContent::image(vec![0u8; 1024]);
    let image_id = client.index_multimodal(&mm_table.name, image_content).await?;
    println!("   ✅ Indexed image: {}", image_id);
    
    // Multimodal search with fusion
    let query = MultimodalQuery::new()
        .with_text("ocean sunset".to_string())
        .with_fusion(FusionStrategy::WeightedAverage)
        .with_k(5);
    
    let mm_results = client.search_multimodal(&mm_table.name, query).await?;
    println!("   Found {} multimodal results\n", mm_results.len());
    
    // ===== 2. GPU Acceleration =====
    println!("2. GPU Acceleration Demo");
    
    // Check GPU availability
    let gpu_info = client.get_gpu_info().await?;
    if let Some(gpu) = gpu_info {
        println!("   ✅ GPU available: {}", gpu.name);
        println!("   Memory: {} GB", gpu.memory_gb);
        println!("   Compute capability: {:?}", gpu.compute_capability);
        
        // Enable GPU acceleration
        let gpu_config = GpuConfig {
            device_id: 0,
            memory_fraction: 0.8,
            batch_size: 1024,
            enable_fp16: true,
            enable_tensor_cores: true,
            stream_count: 4,
        };
        
        client.enable_gpu_acceleration(gpu_config).await?;
        println!("   ✅ GPU acceleration enabled\n");
    } else {
        println!("   ℹ️ No GPU available, using CPU\n");
    }
    
    // ===== 3. SIMD Optimization =====
    println!("3. SIMD Optimization Demo");
    
    let simd_level = client.detect_simd_level().await?;
    println!("   Detected SIMD level: {:?}", simd_level);
    
    match simd_level {
        SimdLevel::Avx512 => println!("   ✅ AVX-512 enabled - maximum performance"),
        SimdLevel::Avx2 => println!("   ✅ AVX2 enabled - excellent performance"),
        SimdLevel::Avx => println!("   ✅ AVX enabled - good performance"),
        _ => println!("   ℹ️ Basic SIMD support"),
    }
    println!();
    
    // ===== 4. Batch Operations =====
    println!("4. Batch Operations Demo");
    
    let batch_table = client.create_table(
        "batch_demo",
        512,
        DistanceMetric::Euclidean,
    ).await?;
    
    // Prepare batch of records
    let mut batch_records = Vec::new();
    for i in 0..1000 {
        let vector = generate_random_vector(512);
        let mut payload = Payload::new();
        payload.insert("batch_id", i);
        payload.insert("timestamp", chrono::Utc::now().timestamp());
        
        batch_records.push(Record::new(
            format!("batch_{}", i),
            vector,
            Some(payload),
        ));
    }
    
    // Batch insert with progress tracking
    println!("   Inserting 1000 records in batch...");
    let start = std::time::Instant::now();
    
    let batch_result = client.batch_insert(
        &batch_table.name,
        batch_records,
        100, // Batch size
    ).await?;
    
    let duration = start.elapsed();
    println!("   ✅ Inserted {} records in {:.2}s", batch_result.success_count, duration.as_secs_f64());
    println!("   Throughput: {:.0} records/sec\n", 
             batch_result.success_count as f64 / duration.as_secs_f64());
    
    // ===== 5. Hybrid Search (Dense + Sparse) =====
    println!("5. Hybrid Search Demo");
    
    let hybrid_table = client.create_hybrid_table(
        "hybrid_demo",
        384,  // Dense vector dimension
        10000, // Sparse vector dimension
    ).await?;
    
    // Insert hybrid record
    let dense_vector = generate_random_vector(384);
    let sparse_vector = SparseVector::new(vec![
        (10, 0.5),
        (25, 0.8),
        (100, 0.3),
        (500, 0.9),
    ]);
    
    let hybrid_record = Record::new_hybrid(
        "hybrid_1",
        dense_vector.clone(),
        sparse_vector.clone(),
        None,
    );
    
    client.insert_record(&hybrid_table.name, hybrid_record).await?;
    println!("   ✅ Inserted hybrid record");
    
    // Perform hybrid search
    let hybrid_query = HybridSearchQuery {
        dense_vector: Some(dense_vector),
        sparse_vector: Some(sparse_vector),
        k: 10,
        alpha: 0.7, // Weight for dense vector (0.7 dense, 0.3 sparse)
        filter: None,
    };
    
    let hybrid_results = client.hybrid_search(&hybrid_table.name, hybrid_query).await?;
    println!("   Found {} hybrid results\n", hybrid_results.len());
    
    // ===== 6. LLM Framework Integration =====
    println!("6. LLM Framework Integration Demo");
    
    // LangChain integration
    println!("   LangChain VectorStore:");
    let langchain_store = client.as_langchain_vectorstore("langchain_demo").await?;
    
    // Add documents
    let documents = vec![
        ("The history of artificial intelligence", vec!["AI", "history"]),
        ("Machine learning algorithms explained", vec!["ML", "algorithms"]),
        ("Deep learning revolutionizes computer vision", vec!["DL", "vision"]),
    ];
    
    for (text, tags) in documents {
        langchain_store.add_document(text, tags).await?;
        println!("      ✅ Added: {}", text);
    }
    
    // Similarity search
    let langchain_results = langchain_store
        .similarity_search("AI and machine learning", 2)
        .await?;
    println!("   Found {} similar documents", langchain_results.len());
    
    // LlamaIndex integration
    println!("\n   LlamaIndex Integration:");
    let llama_index = client.as_llamaindex_storage("llama_demo").await?;
    
    // Create index
    let index = llama_index.create_index(
        "knowledge_base",
        512,
        "gpt-3.5-turbo",
    ).await?;
    println!("      ✅ Created index: {}", index.name);
    
    // Query
    let answer = llama_index.query(
        &index.name,
        "What is deep learning?",
        5, // Top k
    ).await?;
    println!("      Answer: {}", answer);
    println!();
    
    // ===== 7. Performance Metrics =====
    println!("7. Performance Metrics");
    
    let metrics = client.get_performance_metrics().await?;
    println!("   Operations/sec: {:.0}", metrics.operations_per_second);
    println!("   Latency P50: {:.2}ms", metrics.latency_p50_ms);
    println!("   Latency P95: {:.2}ms", metrics.latency_p95_ms);
    println!("   Latency P99: {:.2}ms", metrics.latency_p99_ms);
    println!("   Memory usage: {:.2}MB", metrics.memory_usage_mb);
    
    if metrics.gpu_utilization > 0.0 {
        println!("   GPU utilization: {:.1}%", metrics.gpu_utilization);
    }
    println!("   Cache hit rate: {:.1}%", metrics.cache_hit_rate * 100.0);
    println!();
    
    // ===== 8. Cleanup =====
    println!("8. Cleaning up...");
    client.delete_table(&mm_table.name).await?;
    client.delete_table(&batch_table.name).await?;
    client.delete_table(&hybrid_table.name).await?;
    println!("   ✅ Cleanup complete\n");
    
    println!("🎉 Advanced features demo completed successfully!");
    
    Ok(())
}

fn generate_random_vector(dimension: usize) -> Vec<f32> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    
    let mut vector: Vec<f32> = (0..dimension)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    // Normalize
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    
    vector
}