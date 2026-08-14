//! Hardware acceleration module for DXMT VectorDB.
#![allow(clippy::arc_with_non_send_sync)]
//!
//! Provides GPU, SIMD, parallel, and memory optimizations.
//! GPU acceleration is enabled at compile time with `--features gpu`
//! and detected at runtime -- if no GPU hardware is present,
//! the system silently falls back to CPU+SIMD.

pub mod gpu;
pub mod gpu_distance;
pub mod gpu_kernels;
pub mod simd;
pub mod parallel;
pub mod memory;

use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Re-export main types
pub use gpu::{GpuAccelerator, GpuConfig, GpuDevice, GpuMetrics, GpuBackendType, is_cuda_available};
pub use gpu_distance::{GpuDistanceCalculator, GpuDistanceConfig};
pub use simd::{SimdAccelerator, SimdLevel, VectorOps};
pub use parallel::{ParallelExecutor, ThreadPool, WorkStealing};
pub use memory::{MemoryOptimizer, CacheLineAligned, MemoryPool};

/// Hardware capabilities detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub simd_level: SimdLevel,
    pub gpu_available: bool,
    pub gpu_devices: Vec<GpuDevice>,
    pub total_memory_gb: f64,
    pub cache_sizes: CacheSizes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSizes {
    pub l1_kb: usize,
    pub l2_kb: usize,
    pub l3_mb: usize,
}

impl HardwareCapabilities {
    pub fn detect() -> Self {
        let cpu_cores = num_cpus::get_physical();
        let cpu_threads = num_cpus::get();
        let simd_level = SimdLevel::detect();
        let gpu_devices = GpuDevice::enumerate();

        Self {
            cpu_cores,
            cpu_threads,
            simd_level,
            gpu_available: !gpu_devices.is_empty(),
            gpu_devices,
            total_memory_gb: Self::get_total_memory_gb(),
            cache_sizes: Self::detect_cache_sizes(),
        }
    }

    fn get_total_memory_gb() -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb as f64 / 1_048_576.0;
                            }
                        }
                    }
                }
            }
        }
        16.0 // Default fallback
    }

    fn detect_cache_sizes() -> CacheSizes {
        CacheSizes {
            l1_kb: 32,
            l2_kb: 256,
            l3_mb: 8,
        }
    }
}

/// Performance optimizer that combines all acceleration techniques.
///
/// Selects the best backend (GPU / SIMD-CPU / scalar CPU) for each operation
/// based on hardware capabilities and workload characteristics.
#[allow(dead_code)]
pub struct PerformanceOptimizer {
    capabilities: HardwareCapabilities,
    gpu_accelerator: Option<Arc<GpuAccelerator>>,
    gpu_distance: Option<Arc<GpuDistanceCalculator>>,
    simd_accelerator: Arc<SimdAccelerator>,
    parallel_executor: Arc<ParallelExecutor>,
    memory_optimizer: Arc<MemoryOptimizer>,
}

impl PerformanceOptimizer {
    pub fn new() -> Self {
        let capabilities = HardwareCapabilities::detect();

        // Try to initialize GPU
        let (gpu_accelerator, gpu_distance) = if capabilities.gpu_available {
            let config = GpuDistanceConfig::from_env();
            match GpuDistanceCalculator::try_new(config) {
                Some(calc) => {
                    let calc = Arc::new(calc);
                    let accel = GpuAccelerator::new(GpuConfig::default(), calc.clone());
                    (Some(Arc::new(accel)), Some(calc))
                }
                None => (None, None),
            }
        } else {
            (None, None)
        };

        let simd_accelerator = Arc::new(SimdAccelerator::new(capabilities.simd_level));
        let parallel_executor = Arc::new(ParallelExecutor::new(capabilities.cpu_threads));
        let memory_optimizer = Arc::new(MemoryOptimizer::new(capabilities.cache_sizes.clone()));

        Self {
            capabilities,
            gpu_accelerator,
            gpu_distance,
            simd_accelerator,
            parallel_executor,
            memory_optimizer,
        }
    }

    pub fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    /// Get the shared GPU distance calculator if available.
    pub fn gpu_distance(&self) -> Option<&Arc<GpuDistanceCalculator>> {
        self.gpu_distance.as_ref()
    }

    pub fn gpu_accelerator(&self) -> Option<&Arc<GpuAccelerator>> {
        self.gpu_accelerator.as_ref()
    }

    pub fn select_best_backend(&self, operation: Operation) -> Backend {
        match operation {
            Operation::VectorSearch { dimension, count } => {
                if self.gpu_distance.is_some() && count > 1000 && dimension >= 64 {
                    Backend::Gpu
                } else if count > 500 {
                    Backend::SimdCpu
                } else {
                    Backend::Cpu
                }
            },
            Operation::MatrixMultiply { rows, cols } => {
                if self.gpu_distance.is_some() && rows * cols > 500_000 {
                    Backend::Gpu
                } else {
                    Backend::SimdCpu
                }
            },
            Operation::Embedding { batch_size, .. } => {
                if self.gpu_distance.is_some() && batch_size > 32 {
                    Backend::Gpu
                } else {
                    Backend::SimdCpu
                }
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum Operation {
    VectorSearch { dimension: usize, count: usize },
    MatrixMultiply { rows: usize, cols: usize },
    Embedding { batch_size: usize, dimension: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    SimdCpu,
    Gpu,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub operations_per_second: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization: f64,
    pub cache_hit_rate: f64,
}

impl PerformanceMetrics {
    pub fn throughput_mbps(&self, data_size_bytes: usize) -> f64 {
        (self.operations_per_second * data_size_bytes as f64) / 1_048_576.0
    }
}
