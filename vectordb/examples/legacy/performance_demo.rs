use tracing::{info, Level};
use tracing_subscriber;
use vectordb::{
    config::performance_optimization::PerformanceConfig,
    performance_init::{get_performance_env, PerformanceInitializer},
    search::zen5_optimized::VectorBenchmark,
    types::Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("VectorDB Performance Optimization Demo");
    info!("Hardware: AMD Turin 8-core, 63GB RAM, 375GB NVMe");

    // 获取优化的运行时环境
    let perf_env = get_performance_env()?;
    info!("Performance environment initialized");

    // 运行向量计算基准测试
    info!("\nRunning vector computation benchmarks...");
    let benchmark = VectorBenchmark::new(1536, 10000);
    let results = benchmark.run();

    info!("Benchmark Results:");
    info!(
        "  Dot Product: {:.2} ms for {} operations",
        results.dot_product_time_ms, 10000
    );
    info!(
        "  Throughput: {:.0} ops/sec",
        results.dot_product_ops_per_sec
    );

    // 演示查询线程池使用
    info!("\nDemonstrating optimized query processing...");
    let query_pool = &perf_env.query_pool;

    let start = std::time::Instant::now();
    query_pool.install(|| {
        // 模拟并行查询处理
        (0..100).into_par_iter().for_each(|i| {
            // 每个查询被分配到特定的CPU核心
            let vector = vec![i as f32; 1536];
            let sum: f32 = vector.iter().sum();
            if i == 0 {
                info!("Query {} processed on CPU core (sum: {})", i, sum);
            }
        });
    });
    let elapsed = start.elapsed();
    info!("Processed 100 queries in {:.2} ms", elapsed.as_millis());

    // 演示内存管理
    info!("\nDemonstrating optimized memory management...");
    let memory_manager = &perf_env.memory_manager;

    // 缓存一些向量
    for i in 0..10 {
        let vector = vec![i as f32; 1536];
        memory_manager.cache_vector(format!("vector_{}", i), vector)?;
    }

    // 访问向量并检查缓存命中率
    for i in 0..20 {
        let id = format!("vector_{}", i % 10);
        let _ = memory_manager.get_vector(&id);
    }

    let stats = memory_manager.get_memory_stats();
    info!("Memory Statistics:");
    info!(
        "  Vector cache: {:.2}/{:.0} GB used",
        stats.vector_cache_used_gb, stats.vector_cache_total_gb
    );
    info!("  Cache hit rate: {:.1}%", stats.cache_hit_rate * 100.0);
    info!("  Cached vectors: {}", stats.vector_count);

    // 演示SIMD优化的向量运算
    info!("\nDemonstrating SIMD-optimized vector operations...");
    let vector_ops = &perf_env.vector_ops;

    let a = vec![1.0; 1536];
    let b = vec![2.0; 1536];

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        unsafe {
            let _ = vectordb::search::zen5_optimized::Zen5VectorOps::dot_product_avx2_fma(&a, &b);
        }
    }
    let simd_time = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    }
    let scalar_time = start.elapsed();

    info!(
        "SIMD dot product: {:.2} µs",
        simd_time.as_micros() as f64 / 1000.0
    );
    info!(
        "Scalar dot product: {:.2} µs",
        scalar_time.as_micros() as f64 / 1000.0
    );
    info!(
        "Speedup: {:.2}x",
        scalar_time.as_micros() as f64 / simd_time.as_micros() as f64
    );

    // 显示CPU特性
    let cpu_features = vectordb::search::zen5_optimized::Zen5VectorOps::check_cpu_features();
    info!("\nDetected CPU Features:");
    info!("  AVX2: {}", cpu_features.avx2);
    info!("  AVX-512: {}", cpu_features.avx512f);
    info!("  FMA: {}", cpu_features.fma);
    info!("  VNNI: {}", cpu_features.avx512vnni);

    info!("\nPerformance optimization demo completed!");

    Ok(())
}

// 辅助导入
use rayon::prelude::*;
