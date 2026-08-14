use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio;
use tracing::{info, Level};
use tracing_subscriber;

use vectordb::{
    cache::{CacheConfig, CachedEmbeddingFunction, EmbeddingCache},
    config::multimodal::{MonitoringConfig, MultimodalConfig},
    embeddings::{
        EmbeddingFunction, EmbeddingManager, GpuAcceleratedBuilder, GpuBackend, QuantizationType,
        QuantizedEmbeddingFunction, QuantizedEmbeddingStorage, QuantizedStorageConfig,
    },
    monitoring::{HealthLevel, MultimodalHealthMonitor, MultimodalMetricsCollector},
    multimodal::{
        create_production_monitoring_integration, FusionStrategy, ModalityType,
        MultimodalMonitoringIntegration, MultimodalQuery,
    },
    search::{
        BatchSearchBuilder, BatchSearchConfig, CrossModalBatchSearch, HnswConfig, HnswIndex,
        VectorIndex,
    },
    storage::memory::InMemoryStorage,
    types::{
        multimodal::{EmbeddingResult, MultimodalContent},
        Result,
    },
};

/// Comprehensive showcase of all multi-modal features
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🚀 DXMT VectorDB Multi-Modal Showcase");
    info!("=====================================");

    // 1. Initialize Core Components
    let storage = Arc::new(InMemoryStorage::new());
    let monitoring = Arc::new(create_production_monitoring_integration());
    monitoring.start().await?;

    info!("✅ Core components initialized");

    // 2. Set up 3-Tier Caching
    let cache_config = CacheConfig {
        l1_capacity: 10000,
        l1_ttl_seconds: 3600,
        l2_redis_url: None, // Would use Redis in production
        l3_cache_dir: Some(std::path::PathBuf::from("./showcase_cache")),
        l3_max_size_mb: Some(100),
        promotion_threshold: 3,
        demotion_threshold: 86400,
    };

    let cache = Arc::new(EmbeddingCache::new(cache_config).await?);
    info!("✅ 3-tier caching system configured");

    // 3. Set up Quantized Storage
    let quantized_storage = Arc::new(QuantizedEmbeddingStorage::new(QuantizedStorageConfig {
        quantization_type: QuantizationType::Product,
        num_subvectors: Some(32),
        codebook_size: Some(256),
        target_compression_ratio: 8.0,
        quality_threshold: 0.95,
        ..Default::default()
    }));
    info!("✅ Quantized storage configured (8x compression target)");

    // 4. Create Embedding Pipeline
    let mut embedding_manager = EmbeddingManager::new();

    // Text embeddings with full pipeline
    let text_pipeline = create_embedding_pipeline(
        "text",
        create_mock_embedding("text_model", 1536),
        cache.clone(),
        quantized_storage.clone(),
        monitoring.clone(),
        true, // Enable GPU
    )
    .await?;

    embedding_manager.register_function("text_embedding", text_pipeline);

    // Image embeddings with pipeline
    let image_pipeline = create_embedding_pipeline(
        "image",
        create_mock_embedding("image_model", 512),
        cache.clone(),
        quantized_storage.clone(),
        monitoring.clone(),
        true,
    )
    .await?;

    embedding_manager.register_function("image_embedding", image_pipeline);

    // Audio embeddings with pipeline
    let audio_pipeline = create_embedding_pipeline(
        "audio",
        create_mock_embedding("audio_model", 512),
        cache.clone(),
        quantized_storage.clone(),
        monitoring.clone(),
        false, // CPU for audio
    )
    .await?;

    embedding_manager.register_function("audio_embedding", audio_pipeline);

    info!("✅ Multi-stage embedding pipelines created");

    // 5. Set up Batch Cross-Modal Search
    let batch_config = BatchSearchConfig {
        max_batch_size: 100,
        max_concurrent_searches: 10,
        parallel_index_search: true,
        num_threads: Some(8),
        prefetch_embeddings: true,
        cache_results: true,
        cache_ttl_seconds: 300,
        adaptive_batch_sizing: true,
        min_batch_efficiency: 0.7,
    };

    let batch_search = Arc::new(CrossModalBatchSearch::new(
        storage.clone(),
        batch_config,
        Some(monitoring.clone()),
    ));

    // Create and register indices
    let text_index = Arc::new(create_hnsw_index(1536));
    let image_index = Arc::new(create_hnsw_index(512));
    let audio_index = Arc::new(create_hnsw_index(512));

    batch_search
        .register_index(ModalityType::Text, text_index)
        .await?;
    batch_search
        .register_index(ModalityType::Image, image_index)
        .await?;
    batch_search
        .register_index(ModalityType::Audio, audio_index)
        .await?;

    // Register embedding functions
    batch_search
        .register_embedding_function(
            ModalityType::Text,
            embedding_manager.get_function("text_embedding").unwrap(),
        )
        .await?;
    batch_search
        .register_embedding_function(
            ModalityType::Image,
            embedding_manager.get_function("image_embedding").unwrap(),
        )
        .await?;
    batch_search
        .register_embedding_function(
            ModalityType::Audio,
            embedding_manager.get_function("audio_embedding").unwrap(),
        )
        .await?;

    info!("✅ Batch cross-modal search configured");

    // 6. Demonstrate Features
    info!("\n📊 Running feature demonstrations...\n");

    // Demo 1: Cache Performance
    demonstrate_cache_performance(&embedding_manager, &monitoring).await?;

    // Demo 2: Quantization Efficiency
    demonstrate_quantization(&quantized_storage).await?;

    // Demo 3: Batch Processing
    demonstrate_batch_processing(&batch_search).await?;

    // Demo 4: Monitoring & Health
    demonstrate_monitoring(&monitoring).await?;

    // Demo 5: End-to-End Performance
    demonstrate_e2e_performance(&batch_search, &monitoring).await?;

    // 7. Final System Report
    print_system_report(&monitoring, &cache, &quantized_storage).await?;

    info!("\n✨ Showcase completed successfully!");

    Ok(())
}

/// Create a complete embedding pipeline
async fn create_embedding_pipeline(
    modality: &str,
    base_fn: Box<dyn EmbeddingFunction>,
    cache: Arc<EmbeddingCache>,
    quantized_storage: Arc<QuantizedEmbeddingStorage>,
    monitoring: Arc<MultimodalMonitoringIntegration>,
    use_gpu: bool,
) -> Result<Box<dyn EmbeddingFunction>> {
    // Stage 1: Caching
    let cached_fn = Box::new(CachedEmbeddingFunction::new(base_fn, cache));

    // Stage 2: GPU Acceleration (if enabled)
    let accelerated_fn = if use_gpu {
        match GpuAcceleratedBuilder::new()
            .backend(GpuBackend::Cuda)
            .max_batch_size(256)
            .mixed_precision(true)
            .cpu_fallback(true)
            .build(cached_fn)
            .await
        {
            Ok(gpu_fn) => Box::new(gpu_fn) as Box<dyn EmbeddingFunction>,
            Err(_) => {
                info!("GPU not available for {}, using CPU", modality);
                cached_fn
            }
        }
    } else {
        cached_fn
    };

    // Stage 3: Quantization
    let quantized_fn = Box::new(
        QuantizedEmbeddingFunction::new(accelerated_fn, quantized_storage, modality.to_string())
            .await?,
    );

    // Stage 4: Monitoring
    let monitored_fn = monitoring.instrument_embedding_function(
        quantized_fn,
        format!("{}_pipeline", modality),
        modality.to_string(),
    );

    Ok(monitored_fn)
}

/// Demonstrate cache performance
async fn demonstrate_cache_performance(
    embedding_manager: &EmbeddingManager,
    monitoring: &MultimodalMonitoringIntegration,
) -> Result<()> {
    info!("Demo 1: Cache Performance");
    info!("-------------------------");

    let text_fn = embedding_manager.get_function("text_embedding").unwrap();
    let content = MultimodalContent::Text("Cache performance test".to_string());

    // Cold cache
    let start = Instant::now();
    let _ = text_fn.embed(&content).await?;
    let cold_time = start.elapsed();

    // Warm cache
    let start = Instant::now();
    let _ = text_fn.embed(&content).await?;
    let warm_time = start.elapsed();

    let speedup = cold_time.as_secs_f64() / warm_time.as_secs_f64();

    info!("  Cold cache: {:?}", cold_time);
    info!("  Warm cache: {:?}", warm_time);
    info!("  Speedup: {:.2}x", speedup);

    // Record cache event
    monitoring.record_cache_event("l1", true, warm_time).await;

    Ok(())
}

/// Demonstrate quantization efficiency
async fn demonstrate_quantization(
    quantized_storage: &Arc<QuantizedEmbeddingStorage>,
) -> Result<()> {
    info!("\nDemo 2: Quantization Efficiency");
    info!("-------------------------------");

    // Generate test embeddings
    let embeddings = generate_test_embeddings(100, 512);
    let ids: Vec<String> = (0..100).map(|i| format!("demo_{}", i)).collect();

    // Store with quantization
    let start = Instant::now();
    let _ = quantized_storage
        .store_embeddings("demo", embeddings, ids)
        .await?;
    let quantize_time = start.elapsed();

    let stats = quantized_storage.get_stats().await;

    info!("  Embeddings: 100 x 512 dimensions");
    info!(
        "  Original size: {:.2} MB",
        stats.original_size_bytes as f64 / 1_048_576.0
    );
    info!(
        "  Compressed size: {:.2} MB",
        stats.compressed_size_bytes as f64 / 1_048_576.0
    );
    info!("  Compression ratio: {:.2}x", stats.compression_ratio);
    info!("  Quantization time: {:?}", quantize_time);
    info!(
        "  Quality loss: {:.2}%",
        stats.avg_quantization_error * 100.0
    );

    Ok(())
}

/// Demonstrate batch processing
async fn demonstrate_batch_processing(
    batch_search: &Arc<CrossModalBatchSearch<InMemoryStorage>>,
) -> Result<()> {
    info!("\nDemo 3: Batch Processing");
    info!("------------------------");

    // Create varied queries
    let queries = vec![
        // Single modality
        MultimodalQuery {
            modalities: vec![(MultimodalContent::Text("sunset beach".to_string()), 1.0)],
            filters: None,
        },
        // Dual modality
        MultimodalQuery {
            modalities: vec![
                (MultimodalContent::Text("classical music".to_string()), 0.6),
                (
                    MultimodalContent::AudioPath("/demo/music.mp3".to_string()),
                    0.4,
                ),
            ],
            filters: None,
        },
        // Triple modality
        MultimodalQuery {
            modalities: vec![
                (
                    MultimodalContent::Text("nature documentary".to_string()),
                    0.5,
                ),
                (
                    MultimodalContent::ImagePath("/demo/nature.jpg".to_string()),
                    0.3,
                ),
                (
                    MultimodalContent::AudioPath("/demo/nature.wav".to_string()),
                    0.2,
                ),
            ],
            filters: None,
        },
    ];

    // Add more queries for batch effect
    let mut batch_queries = queries.clone();
    for i in 0..7 {
        batch_queries.push(MultimodalQuery {
            modalities: vec![(MultimodalContent::Text(format!("batch query {}", i)), 1.0)],
            filters: None,
        });
    }

    let request = BatchSearchBuilder::new()
        .add_queries(batch_queries)
        .k(5)
        .fusion_strategy(FusionStrategy::WeightedAverage)
        .build();

    let start = Instant::now();
    let response = batch_search.batch_search(request).await?;
    let batch_time = start.elapsed();

    info!("  Total queries: {}", response.batch_metrics.total_queries);
    info!(
        "  Successful: {}",
        response.batch_metrics.successful_queries
    );
    info!("  Total time: {:?}", batch_time);
    info!(
        "  Throughput: {:.2} queries/sec",
        response.batch_metrics.total_queries as f64 / batch_time.as_secs_f64()
    );
    info!(
        "  Batch efficiency: {:.2}%",
        response.batch_metrics.batch_efficiency * 100.0
    );

    Ok(())
}

/// Demonstrate monitoring capabilities
async fn demonstrate_monitoring(monitoring: &MultimodalMonitoringIntegration) -> Result<()> {
    info!("\nDemo 4: Monitoring & Health");
    info!("---------------------------");

    // Get current metrics
    let metrics = monitoring.get_metrics_snapshot().await;
    let health = monitoring.get_system_health().await;

    info!("  System Health: {:?}", health.health_level);
    info!("  Total Requests: {}", metrics.multimodal.total_requests);
    info!(
        "  Success Rate: {:.2}%",
        (metrics.multimodal.successful_requests as f64
            / metrics.multimodal.total_requests.max(1) as f64)
            * 100.0
    );
    info!(
        "  Average Latency: {:.2}ms",
        metrics.multimodal.average_latency_ms
    );
    info!(
        "  Cache Hit Rate: {:.2}%",
        metrics.cache.overall_hit_rate * 100.0
    );

    // Check alerts
    let alerts = monitoring.get_metrics_collector().get_alerts(Some(5)).await;
    info!("  Active Alerts: {}", alerts.len());

    // Get Prometheus metrics
    let prometheus = monitoring.get_prometheus_metrics().await;
    info!("  Prometheus metrics: {} lines", prometheus.lines().count());

    Ok(())
}

/// Demonstrate end-to-end performance
async fn demonstrate_e2e_performance(
    batch_search: &Arc<CrossModalBatchSearch<InMemoryStorage>>,
    monitoring: &MultimodalMonitoringIntegration,
) -> Result<()> {
    info!("\nDemo 5: End-to-End Performance");
    info!("------------------------------");

    // Large batch test
    let large_batch: Vec<_> = (0..50)
        .map(|i| {
            let modalities = match i % 3 {
                0 => vec![(MultimodalContent::Text(format!("e2e test {}", i)), 1.0)],
                1 => vec![
                    (MultimodalContent::Text(format!("e2e {}", i)), 0.7),
                    (
                        MultimodalContent::ImagePath(format!("/e2e/img_{}.jpg", i)),
                        0.3,
                    ),
                ],
                _ => vec![
                    (MultimodalContent::Text(format!("e2e multi {}", i)), 0.5),
                    (
                        MultimodalContent::ImagePath(format!("/e2e/img_{}.jpg", i)),
                        0.3,
                    ),
                    (
                        MultimodalContent::AudioPath(format!("/e2e/audio_{}.wav", i)),
                        0.2,
                    ),
                ],
            };

            MultimodalQuery {
                modalities,
                filters: None,
            }
        })
        .collect();

    let request = BatchSearchBuilder::new()
        .add_queries(large_batch)
        .k(10)
        .fusion_strategy(FusionStrategy::RankFusion)
        .build();

    let start = Instant::now();
    let response = batch_search.batch_search(request).await?;
    let total_time = start.elapsed();

    info!("  Batch size: 50 queries");
    info!("  Total time: {:?}", total_time);
    info!(
        "  Embedding time: {:?}",
        Duration::from_millis(response.batch_metrics.embedding_time_ms)
    );
    info!(
        "  Search time: {:?}",
        Duration::from_millis(response.batch_metrics.search_time_ms)
    );
    info!(
        "  Fusion time: {:?}",
        Duration::from_millis(response.batch_metrics.fusion_time_ms)
    );
    info!(
        "  Throughput: {:.2} queries/sec",
        50.0 / total_time.as_secs_f64()
    );

    Ok(())
}

/// Print comprehensive system report
async fn print_system_report(
    monitoring: &MultimodalMonitoringIntegration,
    cache: &Arc<EmbeddingCache>,
    quantized_storage: &Arc<QuantizedEmbeddingStorage>,
) -> Result<()> {
    info!("\n📈 System Report");
    info!("================");

    // Overall metrics
    let metrics = monitoring.get_metrics_snapshot().await;
    info!("\nOverall Performance:");
    info!(
        "  Total embeddings generated: {}",
        metrics
            .models
            .values()
            .map(|m| m.total_embeddings_generated)
            .sum::<u64>()
    );
    info!(
        "  Average latency: {:.2}ms",
        metrics.multimodal.average_latency_ms
    );
    info!("  P95 latency: {:.2}ms", metrics.multimodal.p95_latency_ms);
    info!("  P99 latency: {:.2}ms", metrics.multimodal.p99_latency_ms);

    // Cache performance
    let cache_stats = cache.get_stats().await;
    info!("\nCache Performance:");
    info!("  L1 hit rate: {:.2}%", cache_stats.l1_hit_rate * 100.0);
    info!("  L2 hit rate: {:.2}%", cache_stats.l2_hit_rate * 100.0);
    info!("  L3 hit rate: {:.2}%", cache_stats.l3_hit_rate * 100.0);
    info!(
        "  Overall hit rate: {:.2}%",
        cache_stats.overall_hit_rate * 100.0
    );

    // Quantization efficiency
    let quant_stats = quantized_storage.get_stats().await;
    info!("\nQuantization Efficiency:");
    info!("  Compression ratio: {:.2}x", quant_stats.compression_ratio);
    info!(
        "  Storage saved: {:.2} MB",
        (quant_stats.original_size_bytes - quant_stats.compressed_size_bytes) as f64 / 1_048_576.0
    );
    info!(
        "  Quality retained: {:.2}%",
        (1.0 - quant_stats.avg_quantization_error) * 100.0
    );

    // Model-specific metrics
    info!("\nModel Performance:");
    for (name, model_metrics) in &metrics.models {
        info!("  {}:", name);
        info!("    Requests: {}", model_metrics.requests_processed);
        info!(
            "    Avg latency: {:.2}ms",
            model_metrics.avg_processing_time_ms
        );
        info!(
            "    Success rate: {:.2}%",
            (model_metrics.successful_requests as f64
                / model_metrics.requests_processed.max(1) as f64)
                * 100.0
        );
    }

    Ok(())
}

/// Create mock embedding function for demonstration
fn create_mock_embedding(name: &str, dimension: usize) -> Box<dyn EmbeddingFunction> {
    struct MockEmbedding {
        name: String,
        dimension: usize,
    }

    #[async_trait::async_trait]
    impl EmbeddingFunction for MockEmbedding {
        fn name(&self) -> &str {
            &self.name
        }
        fn supported_modalities(&self) -> Vec<&str> {
            vec!["text", "image", "audio"]
        }
        fn dimension(&self) -> usize {
            self.dimension
        }

        async fn embed(&self, _input: &MultimodalContent) -> Result<EmbeddingResult> {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            // Simulate processing delay
            tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

            let embedding: Vec<f32> = (0..self.dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();

            Ok(EmbeddingResult {
                embeddings: vec![embedding],
                model: self.name.clone(),
                usage: Some(HashMap::new()),
            })
        }

        async fn embed_batch(&self, inputs: &[MultimodalContent]) -> Result<EmbeddingResult> {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            // Simulate batch processing
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

            let embeddings: Vec<Vec<f32>> = inputs
                .iter()
                .map(|_| {
                    (0..self.dimension)
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect()
                })
                .collect();

            Ok(EmbeddingResult {
                embeddings,
                model: self.name.clone(),
                usage: Some(HashMap::from([(
                    "batch_size".to_string(),
                    inputs.len() as f64,
                )])),
            })
        }

        fn metadata(&self) -> HashMap<String, String> {
            HashMap::from([
                ("type".to_string(), "mock".to_string()),
                ("dimension".to_string(), self.dimension.to_string()),
            ])
        }
    }

    Box::new(MockEmbedding {
        name: name.to_string(),
        dimension,
    })
}

/// Create HNSW index
fn create_hnsw_index(dimension: usize) -> HnswIndex {
    HnswIndex::new(HnswConfig {
        dimensions: dimension,
        max_neighbors: 16,
        ef_construction: 200,
        max_layers: 16,
        seed: 42,
    })
}

/// Generate test embeddings
fn generate_test_embeddings(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}
