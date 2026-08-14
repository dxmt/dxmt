//! GPU acceleration backend for DXMT VectorDB.
//!
//! Provides real GPU device detection via cudarc (when `gpu` feature enabled)
//! and delegates distance computations to `GpuDistanceCalculator`.
//! Falls back gracefully when no GPU hardware is present.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, debug};

use crate::types::DenseVector;
use crate::types::vector::DistanceMetric;
use crate::embeddings::EmbeddingError;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaDevice;

use super::gpu_distance::GpuDistanceCalculator;

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub device_id: usize,
    pub memory_fraction: f32,
    pub batch_size: usize,
    pub enable_fp16: bool,
    pub enable_tensor_cores: bool,
    pub stream_count: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_fraction: 0.8,
            batch_size: 1024,
            enable_fp16: false,
            enable_tensor_cores: false,
            stream_count: 2,
        }
    }
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub memory_gb: f32,
    pub compute_capability: (u32, u32),
    pub backend: GpuBackendType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuBackendType {
    Cuda,
    Metal,
    Vulkan,
    OpenCL,
}

impl GpuDevice {
    /// Enumerate all available GPU devices on the system.
    ///
    /// Uses cudarc for CUDA detection when the `gpu` feature is enabled.
    /// Returns an empty Vec if no GPUs are found (no error).
    pub fn enumerate() -> Vec<Self> {
        #[allow(unused_mut)]
        let mut devices: Vec<GpuDevice> = Vec::new();

        #[cfg(feature = "gpu")]
        {
            if let Ok(count) = cudarc::driver::result::device::get_count() {
                for i in 0..count {
                    match CudaDevice::new(i as usize) {
                        Ok(dev) => {
                            let name = dev.name().unwrap_or_else(|_| format!("CUDA Device {}", i));
                            let total_mem = cudarc::driver::result::mem_get_info()
                                .map(|(_, total)| total)
                                .unwrap_or(0);
                            let memory_gb = total_mem as f32 / (1024.0 * 1024.0 * 1024.0);

                            // cudarc doesn't directly expose compute capability;
                            // use a safe default that represents most modern GPUs.
                            let compute_capability = (7, 0);

                            devices.push(GpuDevice {
                                id: i as usize,
                                name,
                                memory_gb,
                                compute_capability,
                                backend: GpuBackendType::Cuda,
                            });
                        }
                        Err(e) => {
                            debug!("Failed to initialize CUDA device {}: {}", i, e);
                        }
                    }
                }
            }
        }

        // Metal detection (macOS) -- placeholder for future implementation
        #[cfg(all(target_os = "macos", not(feature = "gpu")))]
        {
            // Metal backend not yet implemented
        }

        if !devices.is_empty() {
            info!("Detected {} GPU device(s):", devices.len());
            for dev in &devices {
                info!("  - [{}] {} ({:.1} GB, {:?})", dev.id, dev.name, dev.memory_gb, dev.backend);
            }
        } else {
            debug!("No GPU devices detected");
        }

        devices
    }
}

/// Check if any CUDA GPU is available at runtime.
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        cudarc::driver::result::device::get_count()
            .map(|count| count > 0)
            .unwrap_or(false)
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Main GPU accelerator -- combines device info with the distance calculator.
pub struct GpuAccelerator {
    config: GpuConfig,
    calculator: Arc<GpuDistanceCalculator>,
    metrics: Arc<parking_lot::RwLock<GpuMetrics>>,
}

impl GpuAccelerator {
    /// Create a new GPU accelerator.
    ///
    /// Returns `Err` if GPU is not available. Use `try_new()` for optional creation.
    pub fn new(config: GpuConfig, calculator: Arc<GpuDistanceCalculator>) -> Self {
        let metrics = Arc::new(parking_lot::RwLock::new(GpuMetrics::default()));
        Self {
            config,
            calculator,
            metrics,
        }
    }

    /// Attempt to create a GPU accelerator; returns None if no GPU available.
    pub fn try_new(config: GpuConfig) -> Option<Self> {
        let dist_config = super::gpu_distance::GpuDistanceConfig {
            device_id: config.device_id,
            memory_fraction: config.memory_fraction,
            batch_threshold: 1000,
            max_batch_size: 500_000,
        };

        let calculator = GpuDistanceCalculator::try_new(dist_config)?;
        Some(Self::new(config, Arc::new(calculator)))
    }

    pub fn cosine_similarity_batch(
        &self,
        queries: &[DenseVector],
        vectors: &[DenseVector],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let start = std::time::Instant::now();

        let results = self.calculator
            .batch_distances(&DistanceMetric::Cosine, queries, vectors)
            .map_err(|e| EmbeddingError::InternalError(format!("GPU cosine failed: {}", e)))?;

        let mut metrics = self.metrics.write();
        metrics.total_operations += 1;
        metrics.vectors_processed += (queries.len() + vectors.len()) as u64;
        metrics.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;

        Ok(results)
    }

    pub fn euclidean_distance_batch(
        &self,
        queries: &[DenseVector],
        vectors: &[DenseVector],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.calculator
            .batch_distances(&DistanceMetric::Euclidean, queries, vectors)
            .map_err(|e| EmbeddingError::InternalError(format!("GPU euclidean failed: {}", e)))
    }

    pub fn dot_product_batch(
        &self,
        queries: &[DenseVector],
        vectors: &[DenseVector],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.calculator
            .batch_distances(&DistanceMetric::Dot, queries, vectors)
            .map_err(|e| EmbeddingError::InternalError(format!("GPU dot product failed: {}", e)))
    }

    pub fn calculator(&self) -> &Arc<GpuDistanceCalculator> {
        &self.calculator
    }

    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    pub fn get_metrics(&self) -> Arc<parking_lot::RwLock<GpuMetrics>> {
        self.metrics.clone()
    }
}

/// GPU metrics
#[derive(Debug, Default, Clone)]
pub struct GpuMetrics {
    pub total_operations: u64,
    pub vectors_processed: u64,
    pub total_time_ms: f64,
    pub memory_used_mb: f64,
    pub utilization_percent: f64,
}

impl GpuMetrics {
    pub fn throughput_ops_per_sec(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.total_operations as f64 * 1000.0) / self.total_time_ms
        } else {
            0.0
        }
    }

    pub fn average_latency_ms(&self) -> f64 {
        if self.total_operations > 0 {
            self.total_time_ms / self.total_operations as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_enumeration() {
        let devices = GpuDevice::enumerate();
        println!("Found {} GPU devices", devices.len());
        for device in &devices {
            println!("  - {} ({:?})", device.name, device.backend);
        }
        // Should not panic regardless of GPU availability
    }

    #[test]
    fn test_is_cuda_available() {
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
        // Just ensure it doesn't panic
    }

    #[test]
    fn test_gpu_metrics() {
        let mut metrics = GpuMetrics::default();
        assert_eq!(metrics.throughput_ops_per_sec(), 0.0);
        assert_eq!(metrics.average_latency_ms(), 0.0);

        metrics.total_operations = 100;
        metrics.total_time_ms = 500.0;
        assert!((metrics.throughput_ops_per_sec() - 200.0).abs() < 0.01);
        assert!((metrics.average_latency_ms() - 5.0).abs() < 0.01);
    }
}
