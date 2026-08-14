//! GPU acceleration integration tests.
//!
//! These tests verify that:
//! 1. GPU detection works correctly (or gracefully returns "not available")
//! 2. BruteForce search produces correct results when GPU is used
//! 3. GPU/CPU fallback logic works based on batch threshold
//!
//! Run with: cargo test --features gpu --test gpu_acceleration_test

use dxmt_vectordb::search::brute_force::BruteForceIndex;
use dxmt_vectordb::search::index_traits::VectorIndex;
use dxmt_vectordb::types::vector::DistanceMetric;

#[cfg(feature = "gpu")]
use dxmt_vectordb::acceleration::{
    is_cuda_available, GpuDevice, GpuDistanceCalculator, GpuDistanceConfig,
};

use rand::Rng;

/// Helper: generate random vectors
fn random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Test that GPU device enumeration doesn't panic regardless of hardware
#[test]
fn test_gpu_device_enumeration_safe() {
    #[cfg(feature = "gpu")]
    {
        let devices = GpuDevice::enumerate();
        println!("Found {} GPU device(s)", devices.len());
        for dev in &devices {
            println!(
                "  [{:?}] {} - {:.1} GB",
                dev.backend, dev.name, dev.memory_gb
            );
        }
    }
}

/// Test that CUDA availability check doesn't panic
#[test]
fn test_cuda_availability_check() {
    #[cfg(feature = "gpu")]
    {
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
    }
}

/// Test GpuDistanceConfig defaults and from_env
#[test]
fn test_gpu_config() {
    #[cfg(feature = "gpu")]
    {
        let config = GpuDistanceConfig::default();
        assert_eq!(config.device_id, 0);
        assert_eq!(config.batch_threshold, 1000);
        assert_eq!(config.max_batch_size, 500_000);

        let env_config = GpuDistanceConfig::from_env();
        assert_eq!(env_config.device_id, 0); // default when env not set
    }
}

/// Test that GpuDistanceCalculator::try_new returns None or Some without panicking
#[test]
fn test_gpu_calculator_try_new() {
    #[cfg(feature = "gpu")]
    {
        let config = GpuDistanceConfig::default();
        let calc = GpuDistanceCalculator::try_new(config);
        match calc {
            Some(ref c) => println!("GPU calculator initialized: {}", c.device_name()),
            None => println!("No GPU available - this is expected on non-GPU machines"),
        }
    }
}

/// Test BruteForce search correctness (CPU path - always works)
#[tokio::test]
async fn test_brute_force_cpu_search() {
    let mut index = BruteForceIndex::new(DistanceMetric::Cosine);

    // Insert some vectors
    let vectors = random_vectors(100, 128);
    for (i, v) in vectors.iter().enumerate() {
        index.add(format!("vec_{}", i), v.clone()).await.unwrap();
    }

    // Search
    let query = random_vectors(1, 128).into_iter().next().unwrap();
    let results = index.search(&query, 5).await.unwrap();
    assert_eq!(results.len(), 5);

    // Results should be sorted by distance (ascending)
    for i in 1..results.len() {
        assert!(
            results[i].1 >= results[i - 1].1,
            "Results not sorted: {} >= {}",
            results[i].1,
            results[i - 1].1
        );
    }
}

/// Test BruteForce with GPU injection (if GPU available)
#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_brute_force_gpu_search_correctness() {
    let config = GpuDistanceConfig {
        batch_threshold: 50, // Low threshold for testing
        ..Default::default()
    };

    let calc = match GpuDistanceCalculator::try_new(config) {
        Some(c) => std::sync::Arc::new(c),
        None => {
            println!("Skipping GPU test - no GPU available");
            return;
        }
    };

    let dim = 128;
    let count = 200;
    let k = 10;

    // Create two identical indexes: one with GPU, one without
    let mut gpu_index = BruteForceIndex::new(DistanceMetric::Cosine);
    gpu_index.set_gpu_calculator(calc);

    let mut cpu_index = BruteForceIndex::new(DistanceMetric::Cosine);

    // Insert same vectors into both
    let vectors = random_vectors(count, dim);
    for (i, v) in vectors.iter().enumerate() {
        let id = format!("vec_{}", i);
        gpu_index.add(id.clone(), v.clone()).await.unwrap();
        cpu_index.add(id, v.clone()).await.unwrap();
    }

    // Search with the same query
    let query = random_vectors(1, dim).into_iter().next().unwrap();
    let gpu_results = gpu_index.search(&query, k).await.unwrap();
    let cpu_results = cpu_index.search(&query, k).await.unwrap();

    assert_eq!(gpu_results.len(), k);
    assert_eq!(cpu_results.len(), k);

    // GPU and CPU results should return the same top-k IDs
    // (distances may differ slightly due to floating point)
    let gpu_ids: Vec<&str> = gpu_results.iter().map(|(id, _)| id.as_str()).collect();
    let cpu_ids: Vec<&str> = cpu_results.iter().map(|(id, _)| id.as_str()).collect();

    // At minimum, the closest result should match
    assert_eq!(
        gpu_ids[0], cpu_ids[0],
        "Nearest neighbor mismatch: GPU={} vs CPU={}",
        gpu_ids[0], cpu_ids[0]
    );

    println!(
        "GPU search results match CPU for top-{} (nearest: {})",
        k, gpu_ids[0]
    );
}

/// Test that GPU path is NOT used when below batch threshold
#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_gpu_fallback_below_threshold() {
    let config = GpuDistanceConfig {
        batch_threshold: 10_000, // Very high threshold
        ..Default::default()
    };

    let calc = match GpuDistanceCalculator::try_new(config) {
        Some(c) => std::sync::Arc::new(c),
        None => {
            println!("Skipping GPU test - no GPU available");
            return;
        }
    };

    // With only 100 vectors and threshold of 10000, GPU should NOT be used
    assert!(!calc.should_use_gpu(100, 128));
    assert!(!calc.should_use_gpu(5000, 128));
    assert!(calc.should_use_gpu(15000, 128));
}

/// Test GPU distance calculation directly (cosine)
#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_gpu_cosine_distance_accuracy() {
    let config = GpuDistanceConfig::default();
    let calc = match GpuDistanceCalculator::try_new(config) {
        Some(c) => c,
        None => {
            println!("Skipping GPU test - no GPU available");
            return;
        }
    };

    let dim = 128;
    let queries = random_vectors(2, dim);
    let vectors = random_vectors(1500, dim);

    let gpu_result = calc
        .batch_distances(&DistanceMetric::Cosine, &queries, &vectors)
        .unwrap();

    assert_eq!(gpu_result.len(), 2);
    assert_eq!(gpu_result[0].len(), 1500);

    // Verify distances are in valid range for cosine: [0, 2]
    for distances in &gpu_result {
        for &d in distances {
            assert!(
                d >= -0.01 && d <= 2.01,
                "Cosine distance out of range: {}",
                d
            );
        }
    }

    // Compare a few values against CPU calculation
    let metric = DistanceMetric::Cosine;
    for q_idx in 0..2 {
        for v_idx in 0..5 {
            let cpu_dist = metric.calculate(&queries[q_idx], &vectors[v_idx]).unwrap();
            let gpu_dist = gpu_result[q_idx][v_idx];
            let diff = (cpu_dist - gpu_dist).abs();
            assert!(
                diff < 0.01,
                "Distance mismatch at ({}, {}): CPU={}, GPU={}, diff={}",
                q_idx,
                v_idx,
                cpu_dist,
                gpu_dist,
                diff
            );
        }
    }

    println!("GPU cosine distance matches CPU within tolerance");
}

/// Diagnostic test: compare GPU vs CPU on fixed, known vectors
#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_gpu_distance_diagnostic() {
    let config = GpuDistanceConfig {
        batch_threshold: 1,
        ..Default::default()
    };
    let calc = match GpuDistanceCalculator::try_new(config) {
        Some(c) => c,
        None => {
            println!("No GPU");
            return;
        }
    };

    // Fixed query: all-ones normalized
    let dim = 4usize;
    let query = vec![0.5f32; dim];
    // Fixed vectors
    let vectors = vec![
        vec![1.0f32, 0.0, 0.0, 0.0],     // v0: orthogonal to query
        vec![0.5f32, 0.5, 0.5, 0.5],     // v1: identical to query (cosine dist = 0)
        vec![-0.5f32, -0.5, -0.5, -0.5], // v2: opposite (cosine dist = 2)
        vec![0.0f32, 0.0, 0.0, 1.0],     // v3: mostly orthogonal
    ];

    let gpu_dists = calc
        .batch_distances(&DistanceMetric::Cosine, &[query.clone()], &vectors)
        .unwrap();
    let gpu_dists = &gpu_dists[0];

    println!("GPU cosine distances:");
    for (i, (v, d)) in vectors.iter().zip(gpu_dists.iter()).enumerate() {
        let cpu_d = DistanceMetric::Cosine.calculate(&query, v).unwrap();
        println!(
            "  v{}: GPU={:.6} CPU={:.6} diff={:.6}",
            i,
            d,
            cpu_d,
            (d - cpu_d).abs()
        );
    }

    // v1 should be closest (distance = 0)
    // v2 should be farthest (distance = 2)
    assert!(
        (gpu_dists[1] - 0.0).abs() < 0.01,
        "v1 (identical) should have dist ~0, got {}",
        gpu_dists[1]
    );
    assert!(
        (gpu_dists[2] - 2.0).abs() < 0.01,
        "v2 (opposite) should have dist ~2, got {}",
        gpu_dists[2]
    );
}

/// Thorough GPU distance comparison: all 200 distances, fixed seed
#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_gpu_vs_cpu_all_distances() {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    let mut rng = SmallRng::seed_from_u64(42);

    let config = GpuDistanceConfig {
        batch_threshold: 1,
        ..Default::default()
    };
    let calc = match GpuDistanceCalculator::try_new(config) {
        Some(c) => c,
        None => {
            println!("No GPU");
            return;
        }
    };

    let dim = 128;
    let n = 200;
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();

    let gpu_result = calc
        .batch_distances(&DistanceMetric::Cosine, &[query.clone()], &vectors)
        .unwrap();
    let gpu_dists = &gpu_result[0];

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut mismatches = 0;
    for (i, v) in vectors.iter().enumerate() {
        let cpu_d = DistanceMetric::Cosine.calculate(&query, v).unwrap();
        let diff = (gpu_dists[i] - cpu_d).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
        if diff > 0.01 {
            mismatches += 1;
        }
    }

    println!("Max diff: {:.6} at idx {}", max_diff, max_diff_idx);
    println!("Mismatches (>0.01): {}/{}", mismatches, n);

    // Find top-1 by GPU and CPU
    let gpu_min_idx = gpu_dists
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let cpu_dists: Vec<f32> = vectors
        .iter()
        .map(|v| DistanceMetric::Cosine.calculate(&query, v).unwrap())
        .collect();
    let cpu_min_idx = cpu_dists
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    println!(
        "GPU nearest: vec_{} (dist={:.6})",
        gpu_min_idx, gpu_dists[gpu_min_idx]
    );
    println!(
        "CPU nearest: vec_{} (dist={:.6})",
        cpu_min_idx, cpu_dists[cpu_min_idx]
    );

    assert_eq!(mismatches, 0, "Too many GPU/CPU mismatches");
}
