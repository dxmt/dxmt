//! GPU-accelerated distance calculation engine.
//!
//! Provides `GpuDistanceCalculator` -- the central abstraction for GPU-based
//! batch distance computation. Used by BruteForce and other indexes.
//!
//! If no GPU is available, `try_new()` returns `None` and callers
//! silently fall back to CPU+SIMD paths.

use crate::types::vector::{DenseVector, DistanceMetric};
use tracing::debug;

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use tracing::{info, warn};

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchConfig, LaunchAsync};

#[cfg(feature = "gpu")]
use super::gpu_kernels;

/// Configuration for GPU distance calculations.
#[derive(Debug, Clone)]
pub struct GpuDistanceConfig {
    /// CUDA device ID (0-indexed)
    pub device_id: usize,
    /// Fraction of GPU memory available for use (0.0-1.0)
    pub memory_fraction: f32,
    /// Minimum number of vectors to justify GPU transfer overhead.
    /// Below this threshold, CPU+SIMD is faster due to PCIe latency.
    pub batch_threshold: usize,
    /// Maximum vectors per GPU batch (bounded by GPU memory)
    pub max_batch_size: usize,
}

impl Default for GpuDistanceConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_fraction: 0.8,
            batch_threshold: 1000,
            max_batch_size: 500_000,
        }
    }
}

impl GpuDistanceConfig {
    /// Load configuration from environment variables with sensible defaults.
    pub fn from_env() -> Self {
        Self {
            device_id: std::env::var("DB_GPU_DEVICE_ID")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            memory_fraction: std::env::var("DB_GPU_MEMORY_FRACTION")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.8),
            batch_threshold: std::env::var("DB_GPU_BATCH_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000),
            max_batch_size: std::env::var("DB_GPU_MAX_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(500_000),
        }
    }
}

/// GPU-accelerated batch distance calculator.
///
/// Wraps a CUDA device and pre-loaded PTX kernels for computing
/// distance matrices between query vectors and database vectors.
#[derive(Debug)]
pub struct GpuDistanceCalculator {
    config: GpuDistanceConfig,
    #[cfg(feature = "gpu")]
    device: Arc<CudaDevice>,
    /// Total GPU memory in bytes (for budget checks)
    #[cfg(feature = "gpu")]
    total_memory: usize,
}

impl GpuDistanceCalculator {
    /// Attempt to initialize the GPU distance calculator.
    ///
    /// Returns `None` if:
    /// - The `gpu` feature is not enabled at compile time
    /// - No CUDA driver is available on the system
    /// - The specified device ID does not exist
    /// - PTX kernel loading fails
    ///
    /// This is the primary entry point -- callers should treat `None`
    /// as "GPU not available, use CPU fallback".
    pub fn try_new(config: GpuDistanceConfig) -> Option<Self> {
        #[cfg(feature = "gpu")]
        {
            Self::try_new_cuda(config)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = config;
            debug!("GPU feature not enabled at compile time");
            None
        }
    }

    #[cfg(feature = "gpu")]
    fn try_new_cuda(config: GpuDistanceConfig) -> Option<Self> {
        // Attempt to create CUDA device
        let device = match CudaDevice::new(config.device_id) {
            Ok(dev) => dev,
            Err(e) => {
                info!("CUDA device {} not available: {}. GPU acceleration disabled.", config.device_id, e);
                return None;
            }
        };

        // Load PTX kernels
        if let Err(e) = device.load_ptx(
            cudarc::nvrtc::compile_ptx(gpu_kernels::DISTANCE_KERNELS_PTX).unwrap_or_else(|_| {
                // Fallback: try loading raw PTX directly
                cudarc::nvrtc::Ptx::from_src(gpu_kernels::DISTANCE_KERNELS_PTX)
            }),
            gpu_kernels::MODULE_NAME,
            &[
                gpu_kernels::COSINE_KERNEL,
                gpu_kernels::EUCLIDEAN_KERNEL,
                gpu_kernels::DOT_PRODUCT_KERNEL,
            ],
        ) {
            warn!("Failed to load CUDA PTX kernels: {}. GPU acceleration disabled.", e);
            return None;
        }

        // Query device memory (free, total)
        let total_memory = cudarc::driver::result::mem_get_info()
            .map(|(_, total)| total)
            .unwrap_or(4 * 1024 * 1024 * 1024);

        let device_name = device.name().unwrap_or_else(|_| "Unknown CUDA Device".to_string());
        info!(
            "GPU acceleration enabled: {} (device {}), {:.1} GB memory, batch threshold: {}",
            device_name,
            config.device_id,
            total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            config.batch_threshold,
        );

        Some(Self {
            config,
            device,
            total_memory,
        })
    }

    /// Determine whether GPU should be used for the given workload.
    ///
    /// Returns false when:
    /// - Vector count is below the batch threshold (PCIe overhead > compute gain)
    /// - The required GPU memory exceeds the configured budget
    #[allow(unreachable_code)]
    pub fn should_use_gpu(&self, num_vectors: usize, dimension: usize) -> bool {
        if num_vectors < self.config.batch_threshold {
            return false;
        }

        // Estimate GPU memory needed: vectors + queries + output
        // vectors: num_vectors * dimension * 4 bytes
        // We check just the vector buffer; query and output are small by comparison
        let vector_bytes = num_vectors * dimension * std::mem::size_of::<f32>();
        #[cfg(feature = "gpu")]
        {
            let budget = (self.total_memory as f64 * self.config.memory_fraction as f64) as usize;
            if vector_bytes > budget {
                debug!(
                    "GPU memory budget exceeded: need {} MB, budget {} MB",
                    vector_bytes / (1024 * 1024),
                    budget / (1024 * 1024),
                );
                return false;
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = vector_bytes;
            return false;
        }

        true
    }

    /// Compute distances between multiple queries and multiple vectors on GPU.
    ///
    /// Returns a Q x N matrix where result[q][n] is the distance between
    /// query q and vector n, using the specified distance metric.
    #[cfg(feature = "gpu")]
    pub fn batch_distances(
        &self,
        metric: &DistanceMetric,
        queries: &[DenseVector],
        vectors: &[DenseVector],
    ) -> crate::Result<Vec<Vec<f32>>> {
        if queries.is_empty() || vectors.is_empty() {
            return Ok(vec![vec![]; queries.len()]);
        }

        let dimension = queries[0].len();
        let num_queries = queries.len();
        let num_vectors = vectors.len();

        // Flatten to contiguous buffers
        let queries_flat: Vec<f32> = queries.iter().flat_map(|q| q.iter().copied()).collect();
        let vectors_flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        // Process in chunks if num_vectors > max_batch_size
        if num_vectors > self.config.max_batch_size {
            return self.batch_distances_chunked(metric, &queries_flat, &vectors_flat, num_queries, num_vectors, dimension);
        }

        self.compute_distances_gpu(metric, &queries_flat, &vectors_flat, num_queries, num_vectors, dimension)
    }

    /// Compute distances for a single query against multiple vectors.
    #[cfg(feature = "gpu")]
    pub fn single_query_distances(
        &self,
        metric: &DistanceMetric,
        query: &DenseVector,
        vectors: &[DenseVector],
    ) -> crate::Result<Vec<f32>> {
        let result = self.batch_distances(metric, &[query.clone()], vectors)?;
        Ok(result.into_iter().next().unwrap_or_default())
    }

    #[cfg(feature = "gpu")]
    fn compute_distances_gpu(
        &self,
        metric: &DistanceMetric,
        queries_flat: &[f32],
        vectors_flat: &[f32],
        num_queries: usize,
        num_vectors: usize,
        dimension: usize,
    ) -> crate::Result<Vec<Vec<f32>>> {
        // Upload to GPU
        let d_queries: CudaSlice<f32> = self.device.htod_sync_copy(queries_flat)
            .map_err(|e| crate::Error::Internal(format!("GPU upload queries failed: {}", e)))?;
        let d_vectors: CudaSlice<f32> = self.device.htod_sync_copy(vectors_flat)
            .map_err(|e| crate::Error::Internal(format!("GPU upload vectors failed: {}", e)))?;

        // Allocate output on GPU
        let output_size = num_queries * num_vectors;
        let d_output: CudaSlice<f32> = self.device.alloc_zeros(output_size)
            .map_err(|e| crate::Error::Internal(format!("GPU alloc output failed: {}", e)))?;

        // Select kernel based on distance metric
        let kernel_name = match metric {
            DistanceMetric::Cosine => gpu_kernels::COSINE_KERNEL,
            DistanceMetric::Euclidean => gpu_kernels::EUCLIDEAN_KERNEL,
            DistanceMetric::Dot => gpu_kernels::DOT_PRODUCT_KERNEL,
            // For other metrics, fall back to euclidean
            _ => gpu_kernels::EUCLIDEAN_KERNEL,
        };

        let func = self.device.get_func(gpu_kernels::MODULE_NAME, kernel_name)
            .ok_or_else(|| crate::Error::Internal(format!("GPU kernel '{}' not found", kernel_name)))?;

        // Launch configuration
        // Each thread handles one (query, vector) pair
        let block_size = 256u32;
        let grid_x = ((num_vectors as u32) + block_size - 1) / block_size;
        let grid_y = num_queries as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            func.launch(cfg, (
                &d_queries,
                &d_vectors,
                &d_output,
                num_queries as u32,
                num_vectors as u32,
                dimension as u32,
            ))
        }
        .map_err(|e| crate::Error::Internal(format!("GPU kernel launch failed: {}", e)))?;

        // Copy results back to host
        let output_flat: Vec<f32> = self.device.dtoh_sync_copy(&d_output)
            .map_err(|e| crate::Error::Internal(format!("GPU download output failed: {}", e)))?;

        // Reshape flat output into Q x N matrix
        let result: Vec<Vec<f32>> = output_flat
            .chunks(num_vectors)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(result)
    }

    /// Handle large vector sets by processing in chunks.
    #[cfg(feature = "gpu")]
    fn batch_distances_chunked(
        &self,
        metric: &DistanceMetric,
        queries_flat: &[f32],
        vectors_flat: &[f32],
        num_queries: usize,
        num_vectors: usize,
        dimension: usize,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let chunk_size = self.config.max_batch_size;
        let mut all_results: Vec<Vec<f32>> = vec![Vec::with_capacity(num_vectors); num_queries];

        let mut offset = 0;
        while offset < num_vectors {
            let end = std::cmp::min(offset + chunk_size, num_vectors);
            let chunk_count = end - offset;
            let chunk_start = offset * dimension;
            let chunk_end = end * dimension;
            let chunk_flat = &vectors_flat[chunk_start..chunk_end];

            let chunk_result = self.compute_distances_gpu(
                metric,
                queries_flat,
                chunk_flat,
                num_queries,
                chunk_count,
                dimension,
            )?;

            for (q, distances) in chunk_result.into_iter().enumerate() {
                all_results[q].extend(distances);
            }

            offset = end;
        }

        Ok(all_results)
    }

    /// Get the device name for logging/diagnostics.
    pub fn device_name(&self) -> String {
        #[cfg(feature = "gpu")]
        {
            self.device.name().unwrap_or_else(|_| "Unknown GPU".to_string())
        }
        #[cfg(not(feature = "gpu"))]
        {
            "No GPU".to_string()
        }
    }

    /// Get the batch threshold.
    pub fn batch_threshold(&self) -> usize {
        self.config.batch_threshold
    }

    /// Get the GPU device id.
    pub fn device_id(&self) -> usize {
        self.config.device_id
    }
}

// Not compiled without gpu feature, but provide a stub type for non-gpu builds
#[cfg(not(feature = "gpu"))]
impl GpuDistanceCalculator {
    pub fn batch_distances(
        &self,
        _metric: &DistanceMetric,
        _queries: &[DenseVector],
        _vectors: &[DenseVector],
    ) -> crate::Result<Vec<Vec<f32>>> {
        Err(crate::Error::Internal("GPU feature not enabled".to_string()))
    }

    pub fn single_query_distances(
        &self,
        _metric: &DistanceMetric,
        _query: &DenseVector,
        _vectors: &[DenseVector],
    ) -> crate::Result<Vec<f32>> {
        Err(crate::Error::Internal("GPU feature not enabled".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = GpuDistanceConfig::default();
        assert_eq!(config.device_id, 0);
        assert_eq!(config.batch_threshold, 1000);
        assert_eq!(config.max_batch_size, 500_000);
        assert!((config.memory_fraction - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_from_env() {
        // Without env vars set, should return defaults
        let config = GpuDistanceConfig::from_env();
        assert_eq!(config.device_id, 0);
        assert_eq!(config.batch_threshold, 1000);
    }

    #[test]
    fn test_try_new_without_gpu_feature() {
        // When gpu feature is not enabled, try_new should return None
        // When gpu feature IS enabled but no GPU hardware, also returns None
        let config = GpuDistanceConfig::default();
        let calc = GpuDistanceCalculator::try_new(config);
        // Result depends on hardware; just check it doesn't panic
        if calc.is_none() {
            // Expected on machines without GPU
            assert!(true);
        }
    }
}
