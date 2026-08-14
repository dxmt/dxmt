//! Adaptive Indexing Demo - Automatic Index Optimization
//!
//! This example demonstrates how DXMT tracks workload patterns and recommends
//! optimal index types via AdaptiveIndex.

use dxmt_vectordb::adaptive_index::{AdaptiveConfig, AdaptiveIndex};
use dxmt_vectordb::types::{table::IndexType, vector::DistanceMetric};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Adaptive Indexing Demo\n");
    println!("Demonstrates how DXMT recommends optimal index types based on workload.\n");

    // Phase 1: Small dataset — BruteForce is optimal
    println!("=== Phase 1: Small Dataset (<5K vectors) ===");
    let config = AdaptiveConfig {
        initial_type: IndexType::BruteForce,
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        max_elements: 1_000_000,
        switch_threshold: 0.2,
        evaluation_interval_ms: 5000,
        memory_limit_mb: 2048,
        ..Default::default()
    };
    let index = AdaptiveIndex::new(config).await;

    let current = index.current_index_type().await;
    println!("  Current index: {:?}", current);

    let result = index.adapt().await?;
    println!("  Adapt result:");
    println!("    switched: {}", result.switched);
    println!("    from: {:?}", result.from_type);
    println!("    to: {:?}", result.to_type);
    println!("    reason: {}", result.reason);
    println!("    expected improvement: {:.1}%\n", result.expected_improvement * 100.0);

    // Phase 2: Medium dataset — HNSW recommended
    println!("=== Phase 2: Medium Dataset (10K-100K vectors) ===");
    let config2 = AdaptiveConfig {
        initial_type: IndexType::Hnsw,
        dimension: 768,
        distance_metric: DistanceMetric::Cosine,
        max_elements: 50_000,
        switch_threshold: 0.15,
        evaluation_interval_ms: 10000,
        memory_limit_mb: 4096,
        ..Default::default()
    };
    let index2 = AdaptiveIndex::new(config2).await;

    let current2 = index2.current_index_type().await;
    println!("  Current index: {:?}", current2);

    let result2 = index2.adapt().await?;
    println!("  Adapt result:");
    println!("    switched: {}", result2.switched);
    println!("    from: {:?}", result2.from_type);
    println!("    to: {:?}", result2.to_type);
    println!("    reason: {}\n", result2.reason);

    // Phase 3: Large dataset — IvfPq for memory efficiency
    println!("=== Phase 3: Large Dataset (>1M vectors) ===");
    let config3 = AdaptiveConfig {
        initial_type: IndexType::IvfPq,
        dimension: 1024,
        distance_metric: DistanceMetric::Euclidean,
        max_elements: 2_000_000,
        switch_threshold: 0.1,
        evaluation_interval_ms: 30000,
        memory_limit_mb: 8192,
        ..Default::default()
    };
    let index3 = AdaptiveIndex::new(config3).await;

    let recommendation = index3.get_recommendation().await?;
    println!("  Recommendation:");
    println!("    target: {:?}", recommendation.target_type);
    println!("    should switch: {}", recommendation.should_switch());
    println!("    expected improvement: {:.1}%", recommendation.expected_improvement * 100.0);
    println!("    reason: {}\n", recommendation.reason);

    println!("Adaptive indexing demo completed successfully!");
    Ok(())
}
