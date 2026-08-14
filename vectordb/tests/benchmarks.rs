// Benchmark tests harness
//
// This file allows running all benchmark tests with: cargo test --test benchmarks
//
// Benchmark tests measure performance and validate against thresholds.

mod common;
mod fixtures;

// Include benchmark test modules
mod wal_performance_test {
    include!("benchmarks/wal_performance_test.rs");
}

mod recall_benchmark {
    include!("benchmarks/recall_benchmark.rs");
}

mod hnsw_performance_benchmark {
    use dxmt_vectordb::{
        search::brute_force::BruteForceIndex,
        search::hnsw::{HnswConfig, HnswIndex},
        types::vector::DistanceMetric,
    };
    use rand::Rng;
    use std::time::Instant;

    fn generate_random_vectors(num_vectors: usize, dimension: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();

        (0..num_vectors)
            .map(|_| {
                let vec: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
                // Normalize for cosine similarity
                let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                vec.iter().map(|x| x / norm).collect()
            })
            .collect()
    }

    #[test]
    #[ignore = "performance benchmark — run with: cargo test --test benchmarks test_hnsw_performance_10k_vectors -- --ignored"]
    fn test_hnsw_performance_10k_vectors() {
        println!("\n{}", "=".repeat(70));
        println!("HNSW Performance Benchmark - 10K Vectors");
        println!("{}\n", "=".repeat(70));

        let num_vectors = 10_000;
        let dimension = 128;
        let num_queries = 1_000;
        let k = 10;

        println!("Configuration:");
        println!("  Vectors:    {}", num_vectors);
        println!("  Dimension:  {}", dimension);
        println!("  Queries:    {}", num_queries);
        println!("  Top-K:      {}", k);
        println!();

        // Generate dataset
        println!("[1/5] Generating test vectors...");
        let start = Instant::now();
        let vectors = generate_random_vectors(num_vectors, dimension);
        let queries = vectors[..num_queries].to_vec();
        println!("  Generated in {:.2}s", start.elapsed().as_secs_f64());

        // Build HNSW index
        println!("\n[2/5] Building HNSW index...");
        let config = HnswConfig::from_params(16, 200, 100);
        let mut hnsw_index = HnswIndex::new(DistanceMetric::Cosine, Some(config));

        let start = Instant::now();
        for (i, vec) in vectors.iter().enumerate() {
            hnsw_index.insert(format!("v{}", i), vec.clone()).unwrap();
        }
        let build_time = start.elapsed();
        println!(
            "  Built in {:.2}s ({:.0} vectors/sec)",
            build_time.as_secs_f64(),
            num_vectors as f64 / build_time.as_secs_f64()
        );

        // Benchmark HNSW search
        println!("\n[3/5] Benchmarking HNSW search...");
        let start = Instant::now();
        let mut total_results = 0;

        for query in &queries {
            let results = hnsw_index.search(query, k).unwrap();
            total_results += results.len();
        }

        let search_time = start.elapsed();
        let qps = num_queries as f64 / search_time.as_secs_f64();
        let avg_latency_ms = search_time.as_millis() as f64 / num_queries as f64;

        println!("  HNSW Performance:");
        println!("    Search time:    {:.2}s", search_time.as_secs_f64());
        println!("    QPS:            {:.0} queries/sec", qps);
        println!("    Avg latency:    {:.3}ms", avg_latency_ms);
        println!("    Total results:  {}", total_results);

        // Benchmark brute force for comparison
        println!("\n[4/5] Benchmarking brute force search (baseline)...");
        let mut brute_force_index = BruteForceIndex::new(DistanceMetric::Cosine);
        for (i, vec) in vectors.iter().enumerate() {
            brute_force_index
                .insert(format!("v{}", i), vec.clone())
                .unwrap();
        }

        let start = Instant::now();
        for query in &queries {
            let _ = brute_force_index.search_sync(query, k).unwrap();
        }

        let brute_force_time = start.elapsed();
        let brute_force_qps = num_queries as f64 / brute_force_time.as_secs_f64();
        let brute_force_latency_ms = brute_force_time.as_millis() as f64 / num_queries as f64;

        println!("  Brute Force Performance:");
        println!("    Search time:    {:.2}s", brute_force_time.as_secs_f64());
        println!("    QPS:            {:.0} queries/sec", brute_force_qps);
        println!("    Avg latency:    {:.3}ms", brute_force_latency_ms);

        // Calculate speedup
        let speedup = qps / brute_force_qps;
        println!("\n[5/5] Performance Analysis:");
        println!("  Speedup:        {:.1}x faster than brute force", speedup);
        println!(
            "  Efficiency:     {:.1}% of brute force latency",
            (brute_force_latency_ms / avg_latency_ms) * 100.0
        );

        // Performance assertions
        println!("\n{}", "=".repeat(70));
        println!("Performance Validation:");
        println!("{}", "=".repeat(70));

        assert!(
            qps > 1_000.0,
            "HNSW QPS should be > 1,000 for 10K vectors, got {:.0}",
            qps
        );
        println!("  ✓ QPS > 1,000: {:.0}", qps);

        assert!(
            avg_latency_ms < 10.0,
            "HNSW latency should be < 10ms, got {:.3}ms",
            avg_latency_ms
        );
        println!("  ✓ Latency < 10ms: {:.3}ms", avg_latency_ms);

        assert!(
            speedup > 5.0,
            "HNSW should be > 5x faster than brute force, got {:.1}x",
            speedup
        );
        println!("  ✓ Speedup > 5x: {:.1}x", speedup);

        println!("\n{}", "=".repeat(70));
        println!("BENCHMARK PASSED ✓");
        println!("{}\n", "=".repeat(70));
    }
}
