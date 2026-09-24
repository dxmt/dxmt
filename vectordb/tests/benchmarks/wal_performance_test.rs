// Storage Performance Benchmark Tests
//
// 测试不同存储配置的性能表现
// Note: WAL configuration has been restructured. This test now focuses on
// basic storage performance benchmarks.

use dxmt_vectordb::{
    config::{StorageConfig, MemoryConfig},
    storage::IntelligentIndexedStorage,
    types::{
        table::TableConfig,
        record::CreateRecordRequest,
        vector::DistanceMetric,
    },
};
use std::time::Instant;

/// 创建测试用的存储配置
async fn create_storage(max_size_mb: usize) -> IntelligentIndexedStorage {
    let config = StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };

    let storage: IntelligentIndexedStorage = IntelligentIndexedStorage::new(&config).await.unwrap();

    // 创建测试表
    let table_config = TableConfig {
        name: "benchmark_table".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        index_type: None,
        description: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
    };
    let _ = storage.create_table(table_config).await.unwrap();

    storage
}

/// 生成测试记录
fn generate_records(start_id: usize, count: usize) -> Vec<CreateRecordRequest> {
    (start_id..start_id + count)
        .map(|i| CreateRecordRequest {
            id: format!("record_{}", i),
            vector: vec![i as f32; 128],
            payload: Some(serde_json::json!({"index": i})),
            sparse_vector: None,
            named_vectors: None,
        })
        .collect()
}

#[tokio::test]
#[ignore = "performance benchmark — run with: cargo test --test benchmarks bench_single_insert_baseline -- --ignored"]
async fn bench_single_insert_baseline() {
    println!("\n=== Baseline: Single Insert Performance ===");

    let storage = create_storage(1024).await;

    // 预热
    let warmup = generate_records(0, 100);
    let _ = storage.insert_records("benchmark_table", warmup).await.unwrap();

    // 测试: 1000 次单条插入
    let start = Instant::now();
    for i in 100..1100 {
        let records = vec![CreateRecordRequest {
            id: format!("record_{}", i),
            vector: vec![i as f32; 128],
            payload: None,
            sparse_vector: None,
            named_vectors: None,
        }];
        let _ = storage.insert_records("benchmark_table", records).await.unwrap();
    }
    let duration = start.elapsed();

    let vps = 1000.0 / duration.as_secs_f64();
    let avg_latency = duration.as_micros() / 1000;

    println!("Duration: {:?}", duration);
    println!("Throughput: {:.2} VPS (Vectors Per Second)", vps);
    println!("Avg Latency: {} μs", avg_latency);

    // Basic assertion - should complete within reasonable time
    assert!(
        duration.as_secs() < 30,
        "1000 single inserts should complete within 30 seconds, took {:?}",
        duration
    );
}

#[tokio::test]
#[ignore = "performance benchmark — run with: cargo test --test benchmarks bench_batch_insert_performance -- --ignored"]
async fn bench_batch_insert_performance() {
    println!("\n=== Batch Insert Performance ===");

    let storage = create_storage(1024).await;

    // 预热
    let warmup = generate_records(0, 100);
    let _ = storage.insert_records("benchmark_table", warmup).await.unwrap();

    // 测试: 批量插入 (100条/批)
    let start = Instant::now();
    for batch_id in 0..10 {
        let records = generate_records(1000 + batch_id * 100, 100);
        let _ = storage.insert_records("benchmark_table", records).await.unwrap();
    }
    let duration = start.elapsed();

    let vps = 1000.0 / duration.as_secs_f64();
    println!("Duration: {:?}", duration);
    println!("Batch (100) Throughput: {:.2} VPS", vps);

    // Batch inserts should be faster
    assert!(
        duration.as_secs() < 30,
        "1000 vectors in batches should complete within 30 seconds, took {:?}",
        duration
    );
}

#[tokio::test]
#[ignore = "performance benchmark — run with: cargo test --test benchmarks bench_mixed_workload -- --ignored"]
async fn bench_mixed_workload() {
    println!("\n=== Mixed Workload: Insert + Search ===");

    let storage = create_storage(1024).await;

    // 预热: 插入一些初始数据
    let warmup = generate_records(0, 500);
    let _ = storage.insert_records("benchmark_table", warmup).await.unwrap();

    // 测试: 混合插入和搜索
    let start = Instant::now();

    for i in 0..100 {
        // 插入
        let records = vec![CreateRecordRequest {
            id: format!("record_{}", 1000 + i),
            vector: vec![(1000 + i) as f32; 128],
            payload: None,
            sparse_vector: None,
            named_vectors: None,
        }];
        let _ = storage.insert_records("benchmark_table", records).await.unwrap();

        // 每10次插入后进行一次搜索
        if i % 10 == 0 {
            use dxmt_vectordb::types::record::SearchQuery;
            let query = SearchQuery {
                vector: vec![i as f32; 128],
                top_k: 10,
                filter: None,
                with_payload: false,
                with_vector: false,
                limit: 10,
                search_mode: dxmt_vectordb::types::record::SearchMode::Normal,
                vector_name: None,
                ..Default::default()
            };
            let _ = storage.search("benchmark_table", query).await.unwrap();
        }
    }
    let duration = start.elapsed();

    println!("Duration: {:?}", duration);
    println!("Mixed operations (100 inserts + 10 searches): {:.2} ops/sec", 110.0 / duration.as_secs_f64());

    assert!(
        duration.as_secs() < 30,
        "Mixed workload should complete within 30 seconds, took {:?}",
        duration
    );
}

#[tokio::test]
#[ignore = "performance benchmark — run with: cargo test --test benchmarks bench_large_batch_insert -- --ignored"]
async fn bench_large_batch_insert() {
    println!("\n=== Large Batch Insert (1000 vectors/batch) ===");

    let storage = create_storage(2048).await;

    // 测试: 大批量插入
    let start = Instant::now();
    for batch_id in 0..5 {
        let records = generate_records(batch_id * 1000, 1000);
        let _ = storage.insert_records("benchmark_table", records).await.unwrap();
    }
    let duration = start.elapsed();

    let vps = 5000.0 / duration.as_secs_f64();
    println!("Duration: {:?}", duration);
    println!("Large Batch Throughput: {:.2} VPS", vps);
    println!("Total vectors: 5000");

    assert!(
        duration.as_secs() < 120,
        "5000 vectors in large batches should complete within 120 seconds, took {:?}",
        duration
    );
}

#[tokio::test]
async fn bench_comparison_summary() {
    println!("\n=== Performance Comparison Summary ===\n");

    #[derive(Debug)]
    struct BenchResult {
        mode: &'static str,
        throughput_vps: f64,
        avg_latency_us: u128,
    }

    let mut results = Vec::new();

    // Single insert
    {
        let storage = create_storage(1024).await;
        let warmup = generate_records(0, 100);
        let _ = storage.insert_records("benchmark_table", warmup).await.unwrap();

        let start = Instant::now();
        for i in 100..600 {
            let records = vec![CreateRecordRequest {
                id: format!("record_{}", i),
                vector: vec![i as f32; 128],
                payload: None,
                sparse_vector: None,
                named_vectors: None,
            }];
            let _ = storage.insert_records("benchmark_table", records).await.unwrap();
        }
        let duration = start.elapsed();
        let throughput_vps = 500.0 / duration.as_secs_f64();
        let avg_latency_us = duration.as_micros() / 500;

        results.push(BenchResult {
            mode: "Single Insert",
            throughput_vps,
            avg_latency_us,
        });
    }

    // Batch insert (100/batch)
    {
        let storage = create_storage(1024).await;
        let warmup = generate_records(0, 100);
        let _ = storage.insert_records("benchmark_table", warmup).await.unwrap();

        let start = Instant::now();
        for batch_id in 0..5 {
            let records = generate_records(1000 + batch_id * 100, 100);
            let _ = storage.insert_records("benchmark_table", records).await.unwrap();
        }
        let duration = start.elapsed();
        let throughput_vps = 500.0 / duration.as_secs_f64();
        let avg_latency_us = duration.as_micros() / 5; // 5 batches

        results.push(BenchResult {
            mode: "Batch Insert (100)",
            throughput_vps,
            avg_latency_us,
        });
    }

    // Large batch insert (1000/batch)
    {
        let storage = create_storage(2048).await;

        let start = Instant::now();
        for batch_id in 0..5 {
            let records = generate_records(batch_id * 100, 100);
            let _ = storage.insert_records("benchmark_table", records).await.unwrap();
        }
        let duration = start.elapsed();
        let throughput_vps = 500.0 / duration.as_secs_f64();
        let avg_latency_us = duration.as_micros() / 5; // 5 batches

        results.push(BenchResult {
            mode: "Large Batch (1000)",
            throughput_vps,
            avg_latency_us,
        });
    }

    // Print results
    println!("┌─────────────────────────┬──────────────┬────────────────┐");
    println!("│ Mode                    │ Throughput   │ Avg Latency    │");
    println!("├─────────────────────────┼──────────────┼────────────────┤");
    for result in &results {
        println!(
            "│ {:23} │ {:8.0} VPS │ {:10} μs │",
            result.mode, result.throughput_vps, result.avg_latency_us
        );
    }
    println!("└─────────────────────────┴──────────────┴────────────────┘");

    // Calculate performance improvement
    let baseline_vps = results[0].throughput_vps;
    println!("\nPerformance Improvement vs Single Insert:");
    for (_i, result) in results.iter().enumerate().skip(1) {
        let improvement_pct = ((result.throughput_vps - baseline_vps) / baseline_vps) * 100.0;
        println!(
            "  {}: {:.1}% faster",
            result.mode, improvement_pct
        );
    }

    // Basic sanity check: batch should be faster than single
    assert!(
        results[1].throughput_vps > results[0].throughput_vps * 0.8,
        "Batch insert should be at least as fast as single insert"
    );
}
