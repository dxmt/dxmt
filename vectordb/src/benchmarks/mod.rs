// ANN-Benchmarks module for standardized vector database testing

pub mod ann_benchmark;
pub mod metrics;
pub mod datasets;
pub mod thresholds;

pub use ann_benchmark::{
    AnnBenchmark, BenchmarkConfig, BenchmarkDataset, BenchmarkResult,
    DatasetMetrics, QueryMetrics, HnswParams, IndexType,
};

pub use metrics::{
    RecallEvaluator, RecallStatistics,
    PrecisionEvaluator, NdcgEvaluator, MapEvaluator,
    MetricsEvaluator, MetricsReport, MetricStatistics,
};

pub use datasets::{
    Dataset, DatasetRegistry, DatasetMetadata, DatasetCache,
    SizeCategory, DimensionCategory,
};

pub use thresholds::{
    PerformanceThreshold, ThresholdValidator, ThresholdValidation,
    ValidationMetrics,
};