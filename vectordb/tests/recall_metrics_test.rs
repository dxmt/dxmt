#![cfg(feature = "benchmarks")]

// Standalone recall metrics test
//
// Tests the RecallEvaluator and RecallStatistics functionality

mod fixtures;

use dxmt_vectordb::{
    benchmarks::metrics::RecallEvaluator, search::hnsw::HnswIndex, types::vector::DistanceMetric,
};
use fixtures::*;

#[test]
fn test_recall_evaluator_basic() {
    let evaluator = RecallEvaluator::new(10);

    // Perfect recall
    let retrieved = vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"];
    let ground_truth = vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"];

    let recall = evaluator.calculate_recall(&retrieved, &ground_truth);
    assert!((recall - 1.0).abs() < 1e-6, "Perfect recall should be 1.0");

    // Partial recall
    let retrieved = vec![
        "v1", "v2", "v5", "v6", "v11", "v12", "v13", "v14", "v15", "v16",
    ];
    let ground_truth = vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"];

    let recall = evaluator.calculate_recall(&retrieved, &ground_truth);
    assert_eq!(recall, 0.4); // 4 matches out of 10
}

#[test]
fn test_recall_statistics() {
    let evaluator = RecallEvaluator::new(10);

    let results = vec![
        vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"],
        vec![
            "v1", "v2", "v3", "v4", "v5", "v11", "v12", "v13", "v14", "v15",
        ],
        vec![
            "v1", "v2", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
        ],
        vec![
            "v1", "v2", "v3", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        ],
    ];

    let ground_truths = vec![
        vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"],
        vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"],
        vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"],
        vec!["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"],
    ];

    let stats = evaluator.evaluate_batch(&results, &ground_truths);

    println!("{}", stats);
    println!("{}", stats.format_percent());

    assert_eq!(stats.k, 10);
    assert_eq!(stats.num_queries, 4);
    assert!(stats.mean > 0.0);
    assert!(stats.mean <= 1.0);
}

#[test]
fn test_hnsw_recall() {
    // Create HNSW index with small dataset
    let dimension = 64;
    let mut index = HnswIndex::new(DistanceMetric::Cosine, None);

    // Generate and insert vectors
    let vectors = generate_normalized_test_vectors(100, dimension);
    for (i, vec) in vectors.iter().enumerate() {
        index.insert(format!("v{}", i), vec.clone()).unwrap();
    }

    // Use first vectors as queries and compute ground truth
    let num_queries = 10;
    let k = 10;
    let evaluator = RecallEvaluator::new(k);

    let mut results = Vec::new();
    let mut ground_truths = Vec::new();

    for i in 0..num_queries {
        let query = &vectors[i];

        // Get HNSW results
        let hnsw_results = index.search(query, k).unwrap();
        let hnsw_ids: Vec<String> = hnsw_results.iter().map(|(id, _dist)| id.clone()).collect();

        // Compute ground truth (brute force)
        let mut distances: Vec<(String, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let dist = DistanceMetric::Cosine.calculate(query, vec).unwrap();
                (format!("v{}", idx), dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth_ids: Vec<String> =
            distances.iter().take(k).map(|(id, _)| id.clone()).collect();

        results.push(hnsw_ids);
        ground_truths.push(ground_truth_ids);
    }

    // Calculate recall
    let stats = evaluator.evaluate_batch(&results, &ground_truths);

    println!("\n=== HNSW Recall Benchmark ===");
    println!("Dataset: 100 vectors, {}D", dimension);
    println!("Index: HNSW (M=16, efConstruction=200)");
    println!("{}", stats.format_percent());
    println!("Detailed: {}", stats);

    // HNSW should have very high recall on small datasets
    assert!(
        stats.mean > 0.95,
        "HNSW recall should be > 95% on small dataset, got {}",
        stats.mean
    );
}

#[test]
fn test_recall_with_sift_sample() {
    let sample = SiftSample::new_small();
    let dimension = sample.dimension;

    // Create index
    let mut index = HnswIndex::new(DistanceMetric::Cosine, None);

    // Insert training vectors
    for (i, vec) in sample.train_vectors.iter().enumerate() {
        index.insert(format!("train_{}", i), vec.clone()).unwrap();
    }

    let k = 10;
    let evaluator = RecallEvaluator::new(k);

    let mut results = Vec::new();
    let mut ground_truths = Vec::new();

    // For each test query
    for query in &sample.test_queries {
        // Get HNSW results
        let hnsw_results = index.search(query, k).unwrap();
        let hnsw_ids: Vec<String> = hnsw_results.iter().map(|(id, _dist)| id.clone()).collect();

        // Compute ground truth
        let mut distances: Vec<(String, f32)> = sample
            .train_vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let dist = DistanceMetric::Cosine.calculate(query, vec).unwrap();
                (format!("train_{}", idx), dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth_ids: Vec<String> =
            distances.iter().take(k).map(|(id, _)| id.clone()).collect();

        results.push(hnsw_ids);
        ground_truths.push(ground_truth_ids);
    }

    let stats = evaluator.evaluate_batch(&results, &ground_truths);

    println!("\n=== Synthetic SIFT-shaped Recall Benchmark ===");
    println!(
        "Dataset: {} training vectors, {}D",
        sample.train_vectors.len(),
        dimension
    );
    println!("Queries: {}", sample.test_queries.len());
    println!("{}", stats.format_percent());

    // Should have good recall
    assert!(
        stats.mean > 0.9,
        "Recall should be > 90% on SIFT sample, got {}",
        stats.mean
    );
}
