use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use dxmt_vectordb::benchmarks::{
    AnnBenchmark, BenchmarkConfig, HnswParams, IndexType,
};
use dxmt_vectordb::types::DistanceMetric;

/// DXMT VectorDB Benchmark CLI
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run ANN-Benchmarks on a dataset
    Run {
        /// Path to HDF5 dataset file
        #[arg(short, long)]
        dataset: PathBuf,
        
        /// Index type to benchmark
        #[arg(short = 'i', long, default_value = "hnsw")]
        index_type: String,
        
        /// Number of neighbors to retrieve
        #[arg(short = 'k', long, default_value = "10")]
        k: usize,
        
        /// Number of queries to run
        #[arg(short = 'n', long, default_value = "1000")]
        num_queries: usize,
        
        /// Output file for results (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// HNSW M parameter
        #[arg(long, default_value = "16")]
        hnsw_m: usize,
        
        /// HNSW ef_construction parameter
        #[arg(long, default_value = "200")]
        hnsw_ef_construction: usize,
        
        /// HNSW ef_search parameter
        #[arg(long, default_value = "100")]
        hnsw_ef_search: usize,
        
        /// Distance metric (euclidean, cosine, dot_product)
        #[arg(long)]
        metric: Option<String>,
        
        /// Number of threads for parallel queries
        #[arg(short = 't', long, default_value = "4")]
        threads: usize,
        
        /// Measure index build time
        #[arg(long)]
        measure_build_time: bool,
    },
    
    /// List available datasets
    List,
    
    /// Download ANN-Benchmarks datasets
    Download {
        /// Dataset to download (sift, gist, glove-25, glove-50, glove-100, glove-200, all)
        dataset: String,
        
        /// Output directory for datasets
        #[arg(short, long, default_value = "./datasets")]
        output_dir: PathBuf,
    },
    
    /// Analyze benchmark results
    Analyze {
        /// Directory containing benchmark results
        #[arg(short, long, default_value = "./benchmark_results")]
        results_dir: PathBuf,
        
        /// Generate plots
        #[arg(short, long)]
        plot: bool,
        
        /// Output format (json, csv)
        #[arg(short = 'f', long)]
        format: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info"))
        )
        .init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Run {
            dataset,
            index_type,
            k,
            num_queries,
            output,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_ef_search,
            metric,
            threads,
            measure_build_time,
        } => {
            info!("Running ANN-Benchmarks on {:?}", dataset);
            
            // Parse index type
            let idx_type = match index_type.as_str() {
                "brute_force" | "bruteforce" => IndexType::BruteForce,
                "hnsw" | "optimal_hnsw" => IndexType::Hnsw,  // All HNSW variants map to OptimalHnsw
                _ => {
                    warn!("Unknown index type: {}, using HNSW", index_type);
                    IndexType::Hnsw
                }
            };
            
            // Determine distance metric
            let distance_metric = if let Some(m) = metric {
                match m.as_str() {
                    "euclidean" | "l2" => DistanceMetric::Euclidean,
                    "cosine" | "angular" => DistanceMetric::Cosine,
                    "dot" | "dot_product" | "inner_product" => DistanceMetric::Dot,
                    "manhattan" => DistanceMetric::Manhattan,
                    _ => {
                        warn!("Unknown metric: {}, using Euclidean", m);
                        DistanceMetric::Euclidean
                    }
                }
            } else {
                // Try to infer from dataset name
                let filename = dataset.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                
                if filename.contains("angular") || filename.contains("glove") {
                    DistanceMetric::Cosine
                } else {
                    DistanceMetric::Euclidean
                }
            };
            
            // Create benchmark configuration
            let config = BenchmarkConfig {
                distance_metric,
                index_type: idx_type.clone(),
                k,
                num_queries,
                hnsw_params: match idx_type {
                    IndexType::Hnsw => Some(HnswParams {
                        m: hnsw_m,
                        ef_construction: hnsw_ef_construction,
                        ef_search: hnsw_ef_search,
                    }),
                    _ => None,
                },
                measure_build_time,
                num_threads: threads,
            };
            
            // Run benchmark
            let benchmark = AnnBenchmark::new(config);
            let result = benchmark.run(&dataset).await?;
            
            // Save results
            let output_path = output.unwrap_or_else(|| {
                let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
                PathBuf::from(format!("benchmark_result_{}.json", timestamp))
            });
            
            let json = serde_json::to_string_pretty(&result)?;
            std::fs::write(&output_path, json)?;
            
            info!("Results saved to {:?}", output_path);
            
            // Print summary
            println!("\n=== Benchmark Results ===");
            println!("Dataset: {}", result.dataset);
            println!("Index Type: {}", result.index_type);
            println!("Vectors: {}", result.dataset_metrics.num_vectors);
            println!("Dimensions: {}", result.dataset_metrics.dimensions);
            println!("\nPerformance:");
            println!("  Recall@{}: {:.4}", k, result.query_metrics.recall_at_k);
            println!("  QPS: {:.2}", result.query_metrics.qps);
            println!("  Mean Latency: {:.2}ms", result.query_metrics.mean_latency_ms);
            println!("  P99 Latency: {:.2}ms", result.query_metrics.p99_latency_ms);
            
            if let Some(build_time_ms) = result.dataset_metrics.index_build_time_ms {
                println!("  Build Time: {:.2}s", build_time_ms as f64 / 1000.0);
            }
        }
        
        Commands::List => {
            println!("Available ANN-Benchmarks datasets:");
            println!();
            println!("  sift      - SIFT-1M (128 dims, 1M vectors, Euclidean)");
            println!("  gist      - GIST-1M (960 dims, 1M vectors, Euclidean)");
            println!("  glove-25  - GloVe-25 (25 dims, 1.18M vectors, Cosine)");
            println!("  glove-50  - GloVe-50 (50 dims, 1.18M vectors, Cosine)");
            println!("  glove-100 - GloVe-100 (100 dims, 1.18M vectors, Cosine)");
            println!("  glove-200 - GloVe-200 (200 dims, 1.18M vectors, Cosine)");
            println!();
            println!("Download datasets using:");
            println!("  ./scripts/download_datasets.sh <dataset>");
        }
        
        Commands::Download { dataset, output_dir } => {
            println!("Downloading dataset: {}", dataset);
            println!("Output directory: {:?}", output_dir);
            println!();
            println!("Please use the download script:");
            println!("  ./scripts/download_datasets.sh {}", dataset);
        }
        
        Commands::Analyze { results_dir, plot, format } => {
            println!("Analyzing results in: {:?}", results_dir);
            
            if plot {
                println!("Generating plots...");
            }
            
            if let Some(ref fmt) = format {
                println!("Export format: {}", fmt);
            }
            
            println!();
            println!("Please use the Python analysis script:");
            println!("  python3 scripts/benchmark_analysis.py \\");
            println!("    --results-dir {:?} \\", results_dir);
            if plot {
                println!("    --plot \\");
            }
            if let Some(ref fmt) = format {
                println!("    --export {} \\", fmt);
            }
            println!("    --summary");
        }
    }
    
    Ok(())
}