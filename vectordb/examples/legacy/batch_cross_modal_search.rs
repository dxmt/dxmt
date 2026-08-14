use std::collections::HashMap;
use std::sync::Arc;
use tokio;
use tracing::{info, Level};
use tracing_subscriber;

use vectordb::{
    embeddings::{ClipPythonEmbedding, EmbeddingManager, OpenAIEmbedding, WhisperPythonEmbedding},
    multimodal::{
        create_production_monitoring_integration, FusionStrategy, ModalityType,
        MultimodalMonitoringIntegration, MultimodalQuery,
    },
    search::{
        BatchSearchBuilder, BatchSearchConfig, BatchSearchRequest, CrossModalBatchSearch,
        HnswConfig, HnswIndex, VectorIndex,
    },
    storage::memory::InMemoryStorage,
    types::{multimodal::MultimodalContent, vector::DenseVector, Distance, SearchResult},
};

/// Example demonstrating batch cross-modal search optimization
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("Starting batch cross-modal search example");

    // Create storage and monitoring
    let storage = Arc::new(InMemoryStorage::new());
    let monitoring = Arc::new(create_production_monitoring_integration());
    monitoring.start().await?;

    // Create batch search engine with optimized configuration
    let config = BatchSearchConfig {
        max_batch_size: 50,
        max_concurrent_searches: 10,
        parallel_index_search: true,
        num_threads: Some(4),
        prefetch_embeddings: true,
        cache_results: true,
        cache_ttl_seconds: 300,
        adaptive_batch_sizing: true,
        min_batch_efficiency: 0.7,
    };

    let batch_search =
        CrossModalBatchSearch::new(storage.clone(), config, Some(monitoring.clone()));

    // Initialize embedding functions
    let mut embedding_manager = EmbeddingManager::new();

    // Register text embeddings (OpenAI)
    let text_embedding =
        OpenAIEmbedding::new(std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"));
    embedding_manager.register_function(
        "openai_text",
        monitoring.instrument_embedding_function(
            Box::new(text_embedding),
            "openai_text".to_string(),
            "openai".to_string(),
        ),
    );

    // Register image embeddings (CLIP)
    let clip_embedding = ClipPythonEmbedding::new(Default::default())?;
    embedding_manager.register_function(
        "clip_image",
        monitoring.instrument_embedding_function(
            Box::new(clip_embedding),
            "clip_image".to_string(),
            "clip".to_string(),
        ),
    );

    // Register audio embeddings (Whisper)
    let whisper_embedding = WhisperPythonEmbedding::new(Default::default())?;
    embedding_manager.register_function(
        "whisper_audio",
        monitoring.instrument_embedding_function(
            Box::new(whisper_embedding),
            "whisper_audio".to_string(),
            "whisper".to_string(),
        ),
    );

    // Create indices for each modality
    let text_index = Arc::new(HnswIndex::new(HnswConfig {
        dimensions: 1536, // OpenAI embedding dimension
        max_neighbors: 16,
        ef_construction: 200,
        max_layers: 16,
        seed: 42,
    }));

    let image_index = Arc::new(HnswIndex::new(HnswConfig {
        dimensions: 512, // CLIP embedding dimension
        max_neighbors: 16,
        ef_construction: 200,
        max_layers: 16,
        seed: 42,
    }));

    let audio_index = Arc::new(HnswIndex::new(HnswConfig {
        dimensions: 512, // Whisper embedding dimension
        max_neighbors: 16,
        ef_construction: 200,
        max_layers: 16,
        seed: 42,
    }));

    // Register indices and embedding functions with batch search
    batch_search
        .register_index(ModalityType::Text, text_index as Arc<dyn VectorIndex>)
        .await?;
    batch_search
        .register_index(ModalityType::Image, image_index as Arc<dyn VectorIndex>)
        .await?;
    batch_search
        .register_index(ModalityType::Audio, audio_index as Arc<dyn VectorIndex>)
        .await?;

    batch_search
        .register_embedding_function(
            ModalityType::Text,
            embedding_manager.get_function("openai_text").unwrap(),
        )
        .await?;
    batch_search
        .register_embedding_function(
            ModalityType::Image,
            embedding_manager.get_function("clip_image").unwrap(),
        )
        .await?;
    batch_search
        .register_embedding_function(
            ModalityType::Audio,
            embedding_manager.get_function("whisper_audio").unwrap(),
        )
        .await?;

    // Populate indices with sample data
    populate_sample_data(&batch_search).await?;

    // Example 1: Simple batch search with text queries
    info!("Example 1: Batch text search");
    let text_queries = vec![
        "Find documents about machine learning",
        "Search for articles on climate change",
        "Look for papers on quantum computing",
        "Find tutorials on rust programming",
        "Search for guides on distributed systems",
    ];

    let batch_request = BatchSearchBuilder::new()
        .add_queries(
            text_queries
                .into_iter()
                .map(|text| MultimodalQuery {
                    modalities: vec![(MultimodalContent::Text(text.to_string()), 1.0)],
                    filters: None,
                })
                .collect(),
        )
        .k(5)
        .fusion_strategy(FusionStrategy::WeightedAverage)
        .build();

    let response = batch_search.batch_search(batch_request).await?;
    print_batch_results("Text Batch Search", &response);

    // Example 2: Multi-modal batch search
    info!("Example 2: Multi-modal batch search");
    let multimodal_queries = vec![
        // Text + Image query
        MultimodalQuery {
            modalities: vec![
                (MultimodalContent::Text("sunset beach".to_string()), 0.7),
                (
                    MultimodalContent::ImagePath("/path/to/beach.jpg".to_string()),
                    0.3,
                ),
            ],
            filters: None,
        },
        // Text + Audio query
        MultimodalQuery {
            modalities: vec![
                (MultimodalContent::Text("classical music".to_string()), 0.6),
                (
                    MultimodalContent::AudioPath("/path/to/music.mp3".to_string()),
                    0.4,
                ),
            ],
            filters: None,
        },
        // All three modalities
        MultimodalQuery {
            modalities: vec![
                (
                    MultimodalContent::Text("nature documentary".to_string()),
                    0.5,
                ),
                (
                    MultimodalContent::ImagePath("/path/to/nature.jpg".to_string()),
                    0.3,
                ),
                (
                    MultimodalContent::AudioPath("/path/to/nature.wav".to_string()),
                    0.2,
                ),
            ],
            filters: None,
        },
    ];

    let batch_request = BatchSearchBuilder::new()
        .add_queries(multimodal_queries)
        .k(10)
        .fusion_strategy(FusionStrategy::RankFusion)
        .build();

    let response = batch_search.batch_search(batch_request).await?;
    print_batch_results("Multi-modal Batch Search", &response);

    // Example 3: Filtered batch search
    info!("Example 3: Filtered batch search");
    let mut filters = HashMap::new();
    filters.insert("category".to_string(), serde_json::json!("science"));
    filters.insert("year".to_string(), serde_json::json!({"$gte": 2020}));

    let filtered_queries = vec![
        MultimodalQuery {
            modalities: vec![(
                MultimodalContent::Text("artificial intelligence".to_string()),
                1.0,
            )],
            filters: Some(filters.clone()),
        },
        MultimodalQuery {
            modalities: vec![(MultimodalContent::Text("neural networks".to_string()), 1.0)],
            filters: Some(filters.clone()),
        },
    ];

    let batch_request = BatchSearchBuilder::new()
        .add_queries(filtered_queries)
        .k(5)
        .fusion_strategy(FusionStrategy::MaxScore)
        .filters(filters)
        .build();

    let response = batch_search.batch_search(batch_request).await?;
    print_batch_results("Filtered Batch Search", &response);

    // Example 4: Large batch with adaptive sizing
    info!("Example 4: Large batch with adaptive sizing");
    let large_batch_queries: Vec<_> = (0..100)
        .map(|i| {
            let modalities = match i % 3 {
                0 => vec![(MultimodalContent::Text(format!("Query text {}", i)), 1.0)],
                1 => vec![
                    (MultimodalContent::Text(format!("Query {}", i)), 0.6),
                    (
                        MultimodalContent::ImagePath(format!("/image_{}.jpg", i)),
                        0.4,
                    ),
                ],
                _ => vec![
                    (MultimodalContent::Text(format!("Complex query {}", i)), 0.5),
                    (MultimodalContent::ImagePath(format!("/img_{}.jpg", i)), 0.3),
                    (
                        MultimodalContent::AudioPath(format!("/audio_{}.wav", i)),
                        0.2,
                    ),
                ],
            };

            MultimodalQuery {
                modalities,
                filters: if i % 5 == 0 {
                    let mut f = HashMap::new();
                    f.insert("type".to_string(), serde_json::json!("premium"));
                    Some(f)
                } else {
                    None
                },
            }
        })
        .collect();

    let batch_request = BatchSearchBuilder::new()
        .add_queries(large_batch_queries)
        .k(3)
        .fusion_strategy(FusionStrategy::WeightedAverage)
        .build();

    let start_time = std::time::Instant::now();
    let response = batch_search.batch_search(batch_request).await?;
    let elapsed = start_time.elapsed();

    info!("Large batch completed in {:?}", elapsed);
    print_batch_metrics(&response);

    // Get monitoring metrics
    let metrics = monitoring.get_metrics_snapshot().await;
    info!(
        "Total embedding requests: {}",
        metrics.multimodal.total_requests
    );
    info!(
        "Average latency: {:.2}ms",
        metrics.multimodal.average_latency_ms
    );
    info!(
        "Cache hit rate: {:.2}%",
        metrics.cache.overall_hit_rate * 100.0
    );

    // Check system health
    let health = monitoring.get_system_health().await;
    info!("System health: {:?}", health.health_level);

    Ok(())
}

/// Populate sample data for testing
async fn populate_sample_data(
    batch_search: &CrossModalBatchSearch<InMemoryStorage>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Populating sample data...");

    // This is a simplified example - in reality, you would:
    // 1. Load actual content (text, images, audio)
    // 2. Generate embeddings using the registered functions
    // 3. Insert into the appropriate indices

    // For demonstration, we'll create some random vectors
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Add text embeddings
    for i in 0..1000 {
        let embedding: Vec<f32> = (0..1536).map(|_| rng.gen_range(-1.0..1.0)).collect();
        // In real implementation, you would insert into the index
    }

    // Add image embeddings
    for i in 0..500 {
        let embedding: Vec<f32> = (0..512).map(|_| rng.gen_range(-1.0..1.0)).collect();
        // In real implementation, you would insert into the index
    }

    // Add audio embeddings
    for i in 0..300 {
        let embedding: Vec<f32> = (0..512).map(|_| rng.gen_range(-1.0..1.0)).collect();
        // In real implementation, you would insert into the index
    }

    info!("Sample data populated");
    Ok(())
}

/// Print batch search results
fn print_batch_results(title: &str, response: &BatchSearchResponse) {
    println!("\n=== {} ===", title);
    println!("Total queries: {}", response.batch_metrics.total_queries);
    println!("Successful: {}", response.batch_metrics.successful_queries);
    println!("Failed: {}", response.batch_metrics.failed_queries);
    println!("Total time: {}ms", response.batch_metrics.total_time_ms);
    println!(
        "Avg query time: {:.2}ms",
        response.batch_metrics.avg_query_time_ms
    );

    // Print first few results
    for (idx, result) in response.results.iter().take(3).enumerate() {
        println!("\nQuery {}: {} results", idx, result.results.len());
        for (rank, item) in result.results.iter().take(3).enumerate() {
            println!("  #{}: {} (score: {:.4})", rank + 1, item.id, item.score);
        }
    }
}

/// Print detailed batch metrics
fn print_batch_metrics(response: &BatchSearchResponse) {
    println!("\n=== Batch Processing Metrics ===");
    println!(
        "Queries processed: {}",
        response.batch_metrics.total_queries
    );
    println!(
        "Success rate: {:.2}%",
        (response.batch_metrics.successful_queries as f64
            / response.batch_metrics.total_queries as f64)
            * 100.0
    );
    println!("Total time: {}ms", response.batch_metrics.total_time_ms);
    println!(
        "  - Embedding: {}ms ({:.1}%)",
        response.batch_metrics.embedding_time_ms,
        (response.batch_metrics.embedding_time_ms as f64
            / response.batch_metrics.total_time_ms as f64)
            * 100.0
    );
    println!(
        "  - Search: {}ms ({:.1}%)",
        response.batch_metrics.search_time_ms,
        (response.batch_metrics.search_time_ms as f64
            / response.batch_metrics.total_time_ms as f64)
            * 100.0
    );
    println!(
        "  - Fusion: {}ms ({:.1}%)",
        response.batch_metrics.fusion_time_ms,
        (response.batch_metrics.fusion_time_ms as f64
            / response.batch_metrics.total_time_ms as f64)
            * 100.0
    );
    println!(
        "Batch efficiency: {:.2}%",
        response.batch_metrics.batch_efficiency * 100.0
    );

    let queries_per_second = (response.batch_metrics.total_queries as f64 * 1000.0)
        / response.batch_metrics.total_time_ms as f64;
    println!("Throughput: {:.2} queries/second", queries_per_second);
}
