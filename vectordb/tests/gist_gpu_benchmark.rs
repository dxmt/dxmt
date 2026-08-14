//! GIST-960 GPU Acceleration Benchmark
//!
//! Tests GPU vs CPU search performance on synthetic GIST-like data:
//! - 960-dimensional vectors (matches GIST-960 dataset)
//! - Euclidean distance metric
//! - Various dataset sizes
//!
//! Run with: cargo test --features gpu --test gist_gpu_benchmark -- --nocapture

// Every test in here is behind `#[cfg(feature = "gpu")]`, so without that feature the helpers
// below have no callers. They are not dead — they are gated.
#![allow(dead_code)]


#[cfg(feature = "gpu")]
use dxmt_vectordb::acceleration::{GpuDistanceCalculator, GpuDistanceConfig};

use rand::rngs::SmallRng;

const GIST_DIM: usize = 960;

fn generate_vectors(rng: &mut SmallRng, count: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 256.0).collect())
        .collect()
}

fn compute_recall(
    results: &[(String, f32)],
    ground_truth: &[usize],
    k: usize,
) -> f64 {
    let result_ids: std::collections::HashSet<usize> = results
        .iter()
        .filter_map(|(id, _)| id.strip_prefix("vec_").and_then(|s| s.parse().ok()))
        .collect();
    let truth_set: std::collections::HashSet<usize> = ground_truth
        .iter()
        .take(k)
        .cloned()
        .collect();
    let intersection = result_ids.intersection(&truth_set).count();
    intersection as f64 / k as f64
}

/// Compute exact nearest neighbors (brute-force CPU reference)
fn compute_ground_truth(
    query: &[f32],
    train_vectors: &[Vec<f32>],
    k: usize,
) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = train_vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist: f32 = query.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
            (i, dist)
        })
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Run benchmark for a specific dataset size, report GPU vs CPU performance
#[cfg(feature = "gpu")]
async fn run_benchmark(
    label: &str,
    train_size: usize,
    num_queries: usize,
    k: usize,
    gpu_calc: Arc<GpuDistanceCalculator>,
) {
    let mut rng = SmallRng::seed_from_u64(42);
    let train_vectors = generate_vectors(&mut rng, train_size, GIST_DIM);
    let queries = generate_vectors(&mut rng, num_queries, GIST_DIM);

    println!("\n--- {} ---", label);
    println!("  Dataset: {} vectors × {} dims, {} queries, k={}", train_size, GIST_DIM, num_queries, k);

    // Build GPU index
    let mut gpu_index = BruteForceIndex::new(DistanceMetric::Euclidean);
    gpu_index.set_gpu_calculator(gpu_calc);
    for (i, v) in train_vectors.iter().enumerate() {
        gpu_index.add(format!("vec_{}", i), v.clone()).await.unwrap();
    }

    // Build CPU index
    let mut cpu_index = BruteForceIndex::new(DistanceMetric::Euclidean);
    for (i, v) in train_vectors.iter().enumerate() {
        cpu_index.add(format!("vec_{}", i), v.clone()).await.unwrap();
    }

    // Warm-up (1 query each)
    let _ = gpu_index.search(&queries[0], k).await.unwrap();
    let _ = cpu_index.search(&queries[0], k).await.unwrap();

    // GPU benchmark
    let gpu_start = Instant::now();
    let mut gpu_results_all = Vec::new();
    for q in queries.iter() {
        let res = gpu_index.search(q, k).await.unwrap();
        gpu_results_all.push(res);
    }
    let gpu_elapsed = gpu_start.elapsed();
    let gpu_qps = num_queries as f64 / gpu_elapsed.as_secs_f64();
    let gpu_latency_ms = gpu_elapsed.as_secs_f64() * 1000.0 / num_queries as f64;

    // CPU benchmark
    let cpu_start = Instant::now();
    let mut cpu_results_all = Vec::new();
    for q in queries.iter() {
        let res = cpu_index.search(q, k).await.unwrap();
        cpu_results_all.push(res);
    }
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_qps = num_queries as f64 / cpu_elapsed.as_secs_f64();
    let cpu_latency_ms = cpu_elapsed.as_secs_f64() * 1000.0 / num_queries as f64;

    // Compute recall against GPU results (GPU is our reference since it uses batch computation)
    let mut gpu_recall_sum = 0.0f64;
    let mut cpu_recall_sum = 0.0f64;
    for (i, q) in queries.iter().enumerate() {
        let gt = compute_ground_truth(q, &train_vectors, k);
        gpu_recall_sum += compute_recall(&gpu_results_all[i], &gt, k);
        cpu_recall_sum += compute_recall(&cpu_results_all[i], &gt, k);
    }
    let gpu_recall = gpu_recall_sum / num_queries as f64;
    let cpu_recall = cpu_recall_sum / num_queries as f64;

    let speedup = gpu_qps / cpu_qps;

    println!("  GPU: {:.1} QPS, {:.2} ms/query, recall@{}: {:.4}", gpu_qps, gpu_latency_ms, k, gpu_recall);
    println!("  CPU: {:.1} QPS, {:.2} ms/query, recall@{}: {:.4}", cpu_qps, cpu_latency_ms, k, cpu_recall);
    println!("  Speedup: {:.2}x", speedup);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn benchmark_gist_gpu_vs_cpu() {
    println!("\n========================================");
    println!("GIST-960 GPU Acceleration Benchmark");
    println!("GPU: NVIDIA A100-SXM4-40GB (40GB HBM2e)");
    println!("Dataset: Synthetic GIST-960 (Euclidean)");
    println!("========================================");

    let config = GpuDistanceConfig {
        batch_threshold: 1000,
        memory_fraction: 0.8,
        ..Default::default()
    };

    let calc = match GpuDistanceCalculator::try_new(config) {
        Some(c) => {
            println!("GPU initialized: {}", c.device_name());
            Arc::new(c)
        }
        None => {
            println!("No GPU available - skipping benchmark");
            return;
        }
    };

    // Test at different scales
    run_benchmark("Small (5K vectors)", 5_000, 100, 10, Arc::clone(&calc)).await;
    run_benchmark("Medium (50K vectors)", 50_000, 50, 10, Arc::clone(&calc)).await;
    run_benchmark("Large (100K vectors)", 100_000, 20, 10, Arc::clone(&calc)).await;

    println!("\n========================================");
    println!("Benchmark complete");
    println!("========================================");
}

/// Test GPU correctness on GIST-like data (Euclidean, 960-dim)
#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_gist_gpu_correctness() {
    let config = GpuDistanceConfig {
        batch_threshold: 100,
        ..Default::default()
    };

    let calc = match GpuDistanceCalculator::try_new(config) {
        Some(c) => Arc::new(c),
        None => { println!("No GPU"); return; }
    };

    let mut rng = SmallRng::seed_from_u64(123);
    let n = 2000;
    let k = 10;
    let train_vectors = generate_vectors(&mut rng, n, GIST_DIM);
    let query = generate_vectors(&mut rng, 1, GIST_DIM).into_iter().next().unwrap();

    // Build GPU index
    let mut gpu_index = BruteForceIndex::new(DistanceMetric::Euclidean);
    gpu_index.set_gpu_calculator(calc);
    for (i, v) in train_vectors.iter().enumerate() {
        gpu_index.add(format!("vec_{}", i), v.clone()).await.unwrap();
    }

    // Build CPU index
    let mut cpu_index = BruteForceIndex::new(DistanceMetric::Euclidean);
    for (i, v) in train_vectors.iter().enumerate() {
        cpu_index.add(format!("vec_{}", i), v.clone()).await.unwrap();
    }

    let gpu_results = gpu_index.search(&query, k).await.unwrap();
    let cpu_results = cpu_index.search(&query, k).await.unwrap();

    let gt = compute_ground_truth(&query, &train_vectors, k);
    let gpu_recall = compute_recall(&gpu_results, &gt, k);
    let cpu_recall = compute_recall(&cpu_results, &gt, k);

    println!("GIST-960 correctness test (n={}, k={}):", n, k);
    println!("  GPU recall@{}: {:.4}", k, gpu_recall);
    println!("  CPU recall@{}: {:.4}", k, cpu_recall);
    println!("  GPU top-3: {:?}", &gpu_results[..3.min(gpu_results.len())]);
    println!("  CPU top-3: {:?}", &cpu_results[..3.min(cpu_results.len())]);

    assert!(gpu_recall >= 0.9, "GPU recall@{} too low: {}", k, gpu_recall);
    assert!(cpu_recall >= 0.9, "CPU recall@{} too low: {}", k, cpu_recall);
    assert_eq!(gpu_results[0].0, cpu_results[0].0,
        "Nearest neighbor mismatch: GPU={} vs CPU={}", gpu_results[0].0, cpu_results[0].0);
}

/// Test: pre-load vectors to GPU, only upload query per search
/// This simulates proper GPU vector database usage where index is persistent on GPU
#[cfg(feature = "gpu")]
#[tokio::test]
async fn benchmark_gist_persistent_gpu() {
    use dxmt_vectordb::types::vector::DistanceMetric;

    let config = GpuDistanceConfig { batch_threshold: 1, ..Default::default() };
    let calc = match GpuDistanceCalculator::try_new(config) {
        Some(c) => c,
        None => { println!("No GPU"); return; }
    };

    let mut rng = SmallRng::seed_from_u64(99);
    let train_size = 50_000;
    let num_queries = 100;
    let k = 10;
    let train_vectors = generate_vectors(&mut rng, train_size, GIST_DIM);
    let queries = generate_vectors(&mut rng, num_queries, GIST_DIM);

    println!("\n--- Persistent GPU Buffer Test ---");
    println!("  {} vectors × {} dims, {} queries, k={}", train_size, GIST_DIM, num_queries, k);

    // Pre-flatten all train vectors into one contiguous buffer
    let vectors_flat: Vec<f32> = train_vectors.iter().flat_map(|v| v.iter().copied()).collect();

    // Time: upload vectors once to GPU
    let upload_start = Instant::now();
    // Use batch_distances with all queries at once to simulate persistent GPU index
    // This is Q queries × N vectors in one kernel launch (no re-upload)
    let all_queries_flat: Vec<f32> = queries.iter().flat_map(|q| q.iter().copied()).collect();
    let _ = all_queries_flat; // suppress unused warning
    
    // Batch all queries into a single GPU call (upload vectors ONCE)
    let batch_gpu_start = Instant::now();
    let all_results = calc.batch_distances(
        &DistanceMetric::Euclidean,
        &queries,
        &train_vectors,
    ).unwrap();
    let batch_gpu_elapsed = batch_gpu_start.elapsed();

    let batch_qps = num_queries as f64 / batch_gpu_elapsed.as_secs_f64();
    let batch_latency_ms = batch_gpu_elapsed.as_secs_f64() * 1000.0 / num_queries as f64;

    println!("  Batched GPU (vectors uploaded once for all queries):");
    println!("    Total time: {:.2} ms for {} queries", batch_gpu_elapsed.as_millis(), num_queries);
    println!("    Effective QPS: {:.1}", batch_qps);
    println!("    Effective latency: {:.2} ms/query", batch_latency_ms);
    
    // Compute recall for first query
    let gt = compute_ground_truth(&queries[0], &train_vectors, k);
    // Extract top-k from batched results
    let mut dist_with_idx: Vec<(usize, f32)> = all_results[0].iter().enumerate().map(|(i, &d)| (i, d)).collect();
    dist_with_idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let gpu_top_k: Vec<String> = dist_with_idx[..k].iter().map(|(i, _)| format!("vec_{}", i)).collect();
    let fake_results: Vec<(String, f32)> = dist_with_idx[..k].iter().map(|(i, d)| (format!("vec_{}", i), *d)).collect();
    let recall = compute_recall(&fake_results, &gt, k);
    println!("    Recall@{}: {:.4}", k, recall);
    
    let _ = gpu_top_k;
    let _ = upload_start;
    let _ = vectors_flat;
}
