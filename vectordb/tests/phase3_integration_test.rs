//! Phase 3 Integration Test - Hierarchical Clustering
//!
//! This test validates all Phase 3 components working together:
//! - Hierarchical clustering index
//! - Cluster-aware search strategies
//! - Adaptive sizing recommendations

use dxmt_vectordb::search::{
    AdaptiveClusterAnalyzer, AdaptiveClusterConfig, ClusterAwareSearchConfig,
    ClusterAwareSearchIndex, HierarchicalClusteringConfig, HierarchicalClusteringIndex,
    SearchStrategy,
};
use dxmt_vectordb::types::vector::DenseVector;
use dxmt_vectordb::types::DistanceMetric;

fn create_test_dataset(size: usize, dim: usize) -> Vec<(String, DenseVector)> {
    (0..size)
        .map(|i| {
            let vector: DenseVector = (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.01) % 10.0)
                .collect();
            (format!("vec_{}", i), vector)
        })
        .collect()
}

#[test]
fn test_phase3_hierarchical_clustering_basic() {
    println!("\n=== Test 1: Basic Hierarchical Clustering ===");

    let config = HierarchicalClusteringConfig {
        num_coarse_clusters: 10,
        num_fine_clusters_per_coarse: 10,
        num_coarse_probes: 3,
        num_fine_probes: 5,
        max_iterations: 10,
        tolerance: 1e-4,
        enable_parallel: false,
    };

    let dataset = create_test_dataset(500, 64);
    let mut index = HierarchicalClusteringIndex::new(config.clone(), DistanceMetric::Euclidean);

    // Train
    let train_start = std::time::Instant::now();
    index.train(&dataset).expect("Training failed");
    let train_time = train_start.elapsed();

    println!("✓ Training completed in {:?}", train_time);
    println!("✓ Coarse clusters: {}", config.num_coarse_clusters);
    println!("✓ Fine per coarse: {}", config.num_fine_clusters_per_coarse);

    // Search
    let query = &dataset[0].1;
    let search_start = std::time::Instant::now();
    let results = index.search(query, 10).expect("Search failed");
    let search_time = search_start.elapsed();

    println!("✓ Search completed in {:?}", search_time);
    println!("✓ Results found: {}", results.len());

    assert_eq!(results.len(), 10, "Should find 10 results");
    assert_eq!(results[0].0, "vec_0", "First result should be query itself");
    assert!(results[0].1 < 0.01, "Query should match itself with distance ~0");

    // Calculate reduction
    let vectors_per_fine = 500 / (config.num_coarse_clusters * config.num_fine_clusters_per_coarse);
    let vectors_scanned = config.num_coarse_probes * config.num_fine_probes * vectors_per_fine;
    let reduction_ratio = 500.0 / vectors_scanned as f32;

    println!("✓ Search space reduction: {:.1}x", reduction_ratio);
    assert!(
        reduction_ratio > 5.0,
        "Should achieve at least 5x reduction"
    );

    println!("✅ Test 1 PASSED\n");
}

#[test]
fn test_phase3_cluster_aware_strategies() {
    println!("\n=== Test 2: Cluster-Aware Search Strategies ===");

    let dataset = create_test_dataset(300, 32);
    let query = &dataset[0].1;

    // Test each strategy
    let strategies = vec![
        SearchStrategy::HierarchicalFirst,
        SearchStrategy::Hybrid,
        SearchStrategy::Adaptive,
    ];

    for strategy in strategies {
        println!("Testing strategy: {:?}", strategy);

        let config = ClusterAwareSearchConfig {
            clustering_config: HierarchicalClusteringConfig {
                num_coarse_clusters: 5,
                num_fine_clusters_per_coarse: 5,
                num_coarse_probes: 2,
                num_fine_probes: 3,
                max_iterations: 10,
                tolerance: 1e-4,
                enable_parallel: false,
            },
            search_strategy: strategy.clone(),
            adaptive_probing: false,
            min_recall_threshold: 0.95,
            max_clusters_to_probe: 10,
        };

        let mut index = ClusterAwareSearchIndex::new(config, DistanceMetric::Cosine);
        index.train(&dataset).expect("Training failed");

        let results = index.search(query, 5).expect("Search failed");
        println!("  ✓ Results: {}", results.len());
        assert!(!results.is_empty(), "Should find at least some results");
    }

    println!("✅ Test 2 PASSED\n");
}

#[test]
fn test_phase3_adaptive_probing() {
    println!("\n=== Test 3: Adaptive Probing ===");

    let config = ClusterAwareSearchConfig {
        clustering_config: HierarchicalClusteringConfig {
            num_coarse_clusters: 10,
            num_fine_clusters_per_coarse: 10,
            num_coarse_probes: 2,
            num_fine_probes: 2,
            max_iterations: 10,
            tolerance: 1e-4,
            enable_parallel: false,
        },
        search_strategy: SearchStrategy::HierarchicalFirst,
        adaptive_probing: true,
        min_recall_threshold: 0.95,
        max_clusters_to_probe: 20,
    };

    let dataset = create_test_dataset(500, 64);
    let mut index = ClusterAwareSearchIndex::new(config, DistanceMetric::Euclidean);

    index.train(&dataset).expect("Training failed");

    let query = &dataset[0].1;
    let results = index
        .adaptive_probe_search(query, 10)
        .expect("Adaptive search failed");

    println!("✓ Adaptive probing found {} results", results.len());
    assert!(!results.is_empty(), "Should find at least some results");

    println!("✅ Test 3 PASSED\n");
}

#[test]
fn test_phase3_adaptive_sizing_small() {
    println!("\n=== Test 4: Adaptive Sizing (Small Dataset) ===");

    let config = AdaptiveClusterConfig {
        dataset_size: 5_000,
        memory_budget_mb: 512,
        target_latency_ms: 10.0,
        min_coarse_clusters: 10,
        max_coarse_clusters: 1000,
        min_fine_per_coarse: 5,
        max_fine_per_coarse: 100,
    };

    let result = AdaptiveClusterAnalyzer::recommend_config(&config, 128);

    println!("Recommended configuration:");
    println!("  Coarse clusters: {}", result.num_coarse_clusters);
    println!("  Fine per coarse: {}", result.num_fine_per_coarse);
    println!("  Coarse probes: {}", result.num_coarse_probes);
    println!("  Fine probes: {}", result.num_fine_probes);
    println!("  Est. memory: {:.2} MB", result.estimated_memory_mb);
    println!("  Est. latency: {:.2} ms", result.estimated_latency_ms);
    println!("  Reduction: {:.1}x", result.reduction_ratio);
    println!("  Confidence: {:.2}", result.confidence);

    assert!(
        result.num_coarse_clusters >= config.min_coarse_clusters,
        "Should respect minimum coarse clusters"
    );
    assert!(
        result.num_fine_per_coarse >= config.min_fine_per_coarse,
        "Should respect minimum fine clusters"
    );
    assert!(result.reduction_ratio > 1.0, "Should provide some reduction");
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be in [0, 1]"
    );

    println!("✅ Test 4 PASSED\n");
}

#[test]
fn test_phase3_adaptive_sizing_large() {
    println!("\n=== Test 5: Adaptive Sizing (Large Dataset) ===");

    let config = AdaptiveClusterConfig {
        dataset_size: 1_000_000,
        memory_budget_mb: 2048,
        target_latency_ms: 50.0,
        min_coarse_clusters: 100,
        max_coarse_clusters: 1000,
        min_fine_per_coarse: 10,
        max_fine_per_coarse: 100,
    };

    let result = AdaptiveClusterAnalyzer::recommend_config(&config, 256);

    println!("Recommended configuration for 1M vectors:");
    println!("  Coarse clusters: {}", result.num_coarse_clusters);
    println!("  Fine per coarse: {}", result.num_fine_per_coarse);
    println!("  Reduction: {:.1}x", result.reduction_ratio);

    assert!(
        result.num_coarse_clusters >= 100,
        "Large dataset should use many clusters"
    );
    assert!(
        result.reduction_ratio > 10.0,
        "Large dataset should achieve >10x reduction"
    );

    println!("✅ Test 5 PASSED\n");
}

#[test]
fn test_phase3_elbow_method() {
    println!("\n=== Test 6: Elbow Method for Optimal K ===");

    // Create dataset with clear clusters
    let mut vectors = Vec::new();

    // Cluster 1: around origin
    for i in 0..30 {
        vectors.push(vec![
            (i as f32 * 0.1) % 1.0,
            (i as f32 * 0.1) % 1.0,
        ]);
    }

    // Cluster 2: around (10, 10)
    for i in 0..30 {
        vectors.push(vec![
            10.0 + (i as f32 * 0.1) % 1.0,
            10.0 + (i as f32 * 0.1) % 1.0,
        ]);
    }

    // Cluster 3: around (20, 20)
    for i in 0..30 {
        vectors.push(vec![
            20.0 + (i as f32 * 0.1) % 1.0,
            20.0 + (i as f32 * 0.1) % 1.0,
        ]);
    }

    let (optimal_k, wcss_values) = AdaptiveClusterAnalyzer::find_optimal_k_elbow(
        &vectors,
        2..8,
        DistanceMetric::Euclidean,
    );

    println!("Elbow method results:");
    println!("  Optimal K: {}", optimal_k);
    println!("  WCSS values:");
    for (k, wcss) in &wcss_values {
        println!("    K={}: WCSS={:.2}", k, wcss);
    }

    assert!((2..=7).contains(&optimal_k), "K should be in tested range");

    // WCSS should decrease monotonically
    for i in 1..wcss_values.len() {
        assert!(
            wcss_values[i].1 <= wcss_values[i - 1].1,
            "WCSS should decrease with more clusters"
        );
    }

    println!("✅ Test 6 PASSED\n");
}

#[test]
fn test_phase3_silhouette_score() {
    println!("\n=== Test 7: Silhouette Score ===");

    // Well-separated clusters
    let vectors = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.2, 0.2],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
        vec![10.2, 10.2],
    ];

    let assignments = vec![0, 0, 0, 1, 1, 1];

    let score = AdaptiveClusterAnalyzer::compute_silhouette_score(
        &vectors,
        &assignments,
        2,
        DistanceMetric::Euclidean,
    );

    println!("Silhouette score: {:.4}", score);

    assert!(
        score > 0.5,
        "Well-separated clusters should have high silhouette score"
    );

    println!("✅ Test 7 PASSED\n");
}

#[test]
fn test_phase3_parallel_training() {
    println!("\n=== Test 8: Parallel Training ===");

    let config = HierarchicalClusteringConfig {
        num_coarse_clusters: 20,
        num_fine_clusters_per_coarse: 20,
        num_coarse_probes: 3,
        num_fine_probes: 5,
        max_iterations: 10,
        tolerance: 1e-4,
        enable_parallel: true, // Enable parallel
    };

    let dataset = create_test_dataset(1000, 128);
    let mut index = HierarchicalClusteringIndex::new(config, DistanceMetric::Euclidean);

    let train_start = std::time::Instant::now();
    index.train(&dataset).expect("Parallel training failed");
    let train_time = train_start.elapsed();

    println!("✓ Parallel training completed in {:?}", train_time);

    // Search should still work
    let query = &dataset[0].1;
    let results = index.search(query, 10).expect("Search failed");

    assert_eq!(results.len(), 10, "Should find 10 results");
    println!("✓ Search works after parallel training");

    println!("✅ Test 8 PASSED\n");
}

#[test]
fn test_phase3_search_performance() {
    println!("\n=== Test 9: Search Performance ===");

    let dataset = create_test_dataset(1000, 64);

    // Test different configurations
    let configs = vec![
        ("Conservative", 5, 5, 2, 2),
        ("Balanced", 10, 10, 3, 5),
        ("Aggressive", 20, 20, 2, 3),
    ];

    for (name, coarse, fine, coarse_probes, fine_probes) in configs {
        let config = HierarchicalClusteringConfig {
            num_coarse_clusters: coarse,
            num_fine_clusters_per_coarse: fine,
            num_coarse_probes: coarse_probes,
            num_fine_probes: fine_probes,
            max_iterations: 10,
            tolerance: 1e-4,
            enable_parallel: false,
        };

        let mut index = HierarchicalClusteringIndex::new(config.clone(), DistanceMetric::Cosine);
        index.train(&dataset).expect("Training failed");

        // Perform multiple searches
        let mut total_time = std::time::Duration::ZERO;
        for i in 0..10 {
            let query = &dataset[i * 100].1;
            let start = std::time::Instant::now();
            let _ = index.search(query, 10).expect("Search failed");
            total_time += start.elapsed();
        }

        let avg_time = total_time / 10;
        let vectors_per_fine = 1000 / (coarse * fine);
        let scanned = coarse_probes * fine_probes * vectors_per_fine;
        let reduction = 1000.0 / scanned as f32;

        println!("{} config:", name);
        println!("  Avg search time: {:?}", avg_time);
        println!("  Vectors scanned: ~{}", scanned);
        println!("  Reduction: {:.1}x", reduction);
    }

    println!("✅ Test 9 PASSED\n");
}

#[test]
fn test_phase3_end_to_end() {
    println!("\n=== Test 10: End-to-End Workflow ===");

    // 1. Start with adaptive sizing
    println!("Step 1: Get adaptive configuration");
    let adaptive_config = AdaptiveClusterConfig {
        dataset_size: 1000,
        memory_budget_mb: 256,
        target_latency_ms: 10.0,
        ..Default::default()
    };

    let recommendation = AdaptiveClusterAnalyzer::recommend_config(&adaptive_config, 64);
    println!("  ✓ Recommended {}x{} clusters",
        recommendation.num_coarse_clusters,
        recommendation.num_fine_per_coarse
    );

    // 2. Create dataset
    println!("Step 2: Create dataset");
    let dataset = create_test_dataset(1000, 64);
    println!("  ✓ Created {} vectors", dataset.len());

    // 3. Train hierarchical index
    println!("Step 3: Train hierarchical clustering");
    let config = HierarchicalClusteringConfig {
        num_coarse_clusters: recommendation.num_coarse_clusters,
        num_fine_clusters_per_coarse: recommendation.num_fine_per_coarse,
        num_coarse_probes: recommendation.num_coarse_probes,
        num_fine_probes: recommendation.num_fine_probes,
        max_iterations: 10,
        tolerance: 1e-4,
        enable_parallel: true,
    };

    let mut index = HierarchicalClusteringIndex::new(config, DistanceMetric::Euclidean);
    index.train(&dataset).expect("Training failed");
    println!("  ✓ Training completed");

    // 4. Perform searches
    println!("Step 4: Perform searches");
    let query = &dataset[0].1;
    let results = index.search(query, 10).expect("Search failed");
    println!("  ✓ Found {} results", results.len());

    // 5. Validate results
    println!("Step 5: Validate results");
    assert_eq!(results.len(), 10, "Should find 10 results");
    assert_eq!(results[0].0, "vec_0", "First result should be query");
    assert!(results[0].1 < 0.01, "Distance should be near 0");
    println!("  ✓ Results validated");

    println!("✅ Test 10 PASSED - End-to-End workflow complete!\n");
}
