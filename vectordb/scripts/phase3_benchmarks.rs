//! Phase 3 Performance Benchmarks
//!
//! Comprehensive benchmarking suite for measuring Phase 3 component performance
//! This validates that implementations meet production targets

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use dxmt::adaptive_index::AdaptiveConfig;
use dxmt::index_management::{
    IndexFactory, DataMigrator, TransactionLog, OperationProfiler,
    RecordIteratorBuilder,
};
use dxmt::types::table::IndexType;
use std::time::Duration;
use std::path::PathBuf;

// ============================================================================
// FACTORY BENCHMARKS
// ============================================================================

fn bench_factory_creation(c: &mut Criterion) {
    let config = AdaptiveConfig::default();
    let factory = IndexFactory::new(config);

    let mut group = c.benchmark_group("factory_creation");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    group.bench_function("hnsw_creation", |b| {
        b.iter(|| {
            factory.create_index(black_box(IndexType::Hnsw))
        })
    });

    group.bench_function("ivf_creation", |b| {
        b.iter(|| {
            factory.create_index(black_box(IndexType::Ivf))
        })
    });

    group.bench_function("bruteforce_creation", |b| {
        b.iter(|| {
            factory.create_index(black_box(IndexType::BruteForce))
        })
    });

    group.bench_function("diskann_creation", |b| {
        b.iter(|| {
            factory.create_index(black_box(IndexType::DiskAnn))
        })
    });

    group.finish();
}

fn bench_factory_validation(c: &mut Criterion) {
    let config = AdaptiveConfig::default();
    let factory = IndexFactory::new(config);

    c.bench_function("config_validation", |b| {
        b.iter(|| {
            factory.validate_config()
        })
    });
}

// ============================================================================
// MIGRATOR BENCHMARKS
// ============================================================================

#[tokio::main]
async fn bench_migrator_estimation() {
    let migrator = DataMigrator::new(1000, Duration::from_secs(300));
    let config = AdaptiveConfig::default();
    let factory = IndexFactory::new(config);
    let index = factory.create_index(IndexType::BruteForce).unwrap();

    for record_count in [1000, 10000, 100000, 1000000].iter() {
        let start = std::time::Instant::now();
        let _ = migrator.estimate_duration(&index, *record_count).await;
        let elapsed = start.elapsed();

        println!("Estimate for {} records: {:.2}µs",
            record_count, elapsed.as_micros());
    }
}

// ============================================================================
// PROFILER BENCHMARKS
// ============================================================================

#[tokio::main]
async fn bench_profiler_overhead() {
    let profiler = OperationProfiler::new();

    // Measure profiler overhead
    let iterations = 10000;

    // Start/finish overhead
    let start = std::time::Instant::now();
    for i in 0..iterations {
        let handle = profiler.start_operation(&format!("op_{}", i % 100));
        handle.finish_success(1);
    }
    let elapsed = start.elapsed();

    let overhead_us = elapsed.as_micros() as f64 / iterations as f64;
    println!("Profiler overhead: {:.3}µs per operation", overhead_us);

    // Statistics retrieval
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _profiles = profiler.get_all_profiles();
    }
    let elapsed = start.elapsed();

    println!("Stats retrieval: {:.3}µs per call", elapsed.as_micros() as f64 / 1000.0);
}

// ============================================================================
// TRANSACTION LOG BENCHMARKS
// ============================================================================

#[tokio::main]
async fn bench_transaction_log() {
    let log = TransactionLog::new(PathBuf::from("/tmp/bench_wal.log"), false);

    println!("\n=== Transaction Log Benchmarks ===\n");

    // In-memory logging (no disk I/O)
    let start = std::time::Instant::now();
    for i in 0..10000 {
        let from = IndexType::Hnsw;
        let to = IndexType::Ivf;
        let _ = log.log_begin_switch(&from, &to).await;
    }
    let elapsed = start.elapsed();
    println!("In-memory log_begin_switch: {:.3}µs per entry",
        elapsed.as_micros() as f64 / 10000.0);

    // Read entries
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = log.read_entries().await;
    }
    let elapsed = start.elapsed();
    println!("read_entries() call: {:.3}µs per call",
        elapsed.as_micros() as f64 / 1000.0);

    // Recovery detection
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = log.recover().await;
    }
    let elapsed = start.elapsed();
    println!("recover() call: {:.3}µs per call",
        elapsed.as_micros() as f64 / 1000.0);
}

// ============================================================================
// RECORD ITERATOR BENCHMARKS
// ============================================================================

#[tokio::main]
async fn bench_record_iterator() {
    println!("\n=== Record Iterator Benchmarks ===\n");

    let mut iterator = RecordIteratorBuilder::new()
        .batch_size(1000)
        .build();

    // Add test records
    for i in 0..100000 {
        use dxmt::index_management::IndexRecord;
        use dxmt::types::{RecordId, DenseVector};

        let record = IndexRecord {
            id: RecordId::new(i as u64),
            vector: DenseVector::from(vec![1.0; 128]),
        };
        iterator.add_record(record);
    }

    // Benchmark iteration
    let start = std::time::Instant::now();
    let mut batch_count = 0;
    let mut total_records = 0;

    loop {
        match iterator.next_batch(1000).await {
            Ok(batch) => {
                if batch.is_empty() {
                    break;
                }
                batch_count += 1;
                total_records += batch.len();
            }
            Err(_) => break,
        }
    }
    let elapsed = start.elapsed();

    println!("Iteration stats:");
    println!("  Total records: {}", total_records);
    println!("  Batches: {}", batch_count);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Per record: {:.3}µs",
        elapsed.as_micros() as f64 / total_records as f64);
}

// ============================================================================
// CRITERION BENCHMARKS
// ============================================================================

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets =
        bench_factory_creation,
        bench_factory_validation
);

criterion_main!(benches);

// ============================================================================
// ASYNC BENCHMARKS (Run with: cargo bench --bench phase3_benchmarks)
// ============================================================================

#[cfg(test)]
mod async_benches {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn bench_all_async() {
        println!("\n=== PHASE 3 PERFORMANCE BENCHMARKS ===\n");

        bench_migrator_estimation().await;
        bench_profiler_overhead().await;
        bench_transaction_log().await;
        bench_record_iterator().await;

        println!("\n=== BENCHMARK COMPLETE ===\n");
    }
}
