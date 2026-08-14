use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression as FlateCompression;
use std::io::{Read, Write};

/// Compression utilities for storage optimization
pub struct CompressionManager {
    compression_level: FlateCompression,
}

impl CompressionManager {
    pub fn new(level: CompressionLevel) -> Self {
        let compression_level = match level {
            CompressionLevel::None => FlateCompression::none(),
            CompressionLevel::Fast => FlateCompression::fast(),
            CompressionLevel::Default => FlateCompression::default(),
            CompressionLevel::Best => FlateCompression::best(),
        };

        Self { compression_level }
    }

    pub fn compress(&self, data: &[u8]) -> crate::Result<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), self.compression_level);
        encoder
            .write_all(data)
            .map_err(|e| crate::Error::Storage(format!("Compression failed: {}", e)))?;
        encoder.finish().map_err(|e| crate::Error::Storage(format!("Compression finish failed: {}", e)))
    }

    pub fn decompress(&self, data: &[u8]) -> crate::Result<Vec<u8>> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| crate::Error::Storage(format!("Decompression failed: {}", e))
            )?;
        Ok(decompressed)
    }

    pub fn estimate_compression_ratio(&self, sample_data: &[u8]) -> crate::Result<f64> {
        let compressed = self.compress(sample_data)?;
        Ok(compressed.len() as f64 / sample_data.len() as f64)
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[derive(Default)]
pub enum CompressionLevel {
    None,
    Fast,
    #[default]
    Default,
    Best,
}


/// Vector-specific compression algorithms
pub mod vector_compression {
    
    use crate::types::vector::DenseVector;

    /// Scalar quantization for float vectors
    pub struct ScalarQuantizer {
        _min: f32,
        max: f32,
        bits: u8,
    }

    impl ScalarQuantizer {
        pub fn new(bits: u8) -> Self {
            Self {
                _min: 0.0,
                max: 0.0,
                bits,
            }
        }

        pub fn fit(&mut self, vectors: &[DenseVector]) {
            if vectors.is_empty() {
                return;
            }

            self._min = f32::INFINITY;
            self.max = f32::NEG_INFINITY;

            for vector in vectors {
                for &value in vector.iter() {
                    self._min = self._min.min(value);
                    self.max = self.max.max(value)
                }
            }
        }

        pub fn compress(&self, vector: &DenseVector) -> crate::Result<Vec<u8>> {
            let scale = (1 << self.bits) as f32 - 1.0;
            let range = self.max - self._min;

            if range == 0.0 {
                return Ok(vec![0; vector.len()]);
            }

            let compressed: Vec<u8> = vector
                .iter()
                .map(|&value| {
                    let normalized = (value - self._min) / range;
                    
                    (normalized * scale).round() as u8
                })
                .collect();

            Ok(compressed)
        }

        pub fn decompress(&self, compressed: &[u8], dimension: usize) -> crate::Result<DenseVector> {
            if compressed.len() != dimension {
                return Err(crate::Error::Storage("Compressed vector has wrong dimension".to_string()));
            }

            let scale = (1 << self.bits) as f32 - 1.0;
            let range = self.max - self._min;

            let data: Vec<f32> = compressed
                .iter()
                .map(|&quantized| {
                    let normalized = quantized as f32 / scale;
                    self._min + normalized * range
                })
                .collect();

            Ok(data)
        }
    }

    /// Product quantization for high-dimensional vectors
    pub struct ProductQuantizer {
        subquantizers: Vec<ScalarQuantizer>,
        subvector_size: usize,
    }

    impl ProductQuantizer {
        pub fn new(dimension: usize, num_subquantizers: usize, bits: u8) -> Self {
            let subvector_size = dimension / num_subquantizers;
            let subquantizers = (0..num_subquantizers)
                .map(|_| ScalarQuantizer::new(bits))
                .collect();

            Self {
                subquantizers,
                subvector_size,
            }
        }

        pub fn fit(&mut self, vectors: &[DenseVector]) {
            for (i, sq) in self.subquantizers.iter_mut().enumerate() {
                let subvectors: Vec<DenseVector> = vectors
                    .iter()
                    .map(|v| {
                        let start = i * self.subvector_size;
                        let end = start + self.subvector_size;
                        v[start..end].to_vec()
                    })
                    .collect();

                sq.fit(&subvectors);
            }
        }

        pub fn compress(&self, vector: &DenseVector) -> crate::Result<Vec<u8>> {
            let mut compressed = Vec::new();

            for (i, sq) in self.subquantizers.iter().enumerate() {
                let start = i * self.subvector_size;
                let end = start + self.subvector_size;
                let subvector = vector[start..end].to_vec();
                compressed.extend(sq.compress(&subvector)?)
            }

            Ok(compressed)
        }

        pub fn decompress(&self, compressed: &[u8]) -> crate::Result<DenseVector> {
            let mut data = Vec::new();
            let bytes_per_subvector = self.subvector_size;

            for (i, sq) in self.subquantizers.iter().enumerate() {
                let start = i * bytes_per_subvector;
                let end = start + bytes_per_subvector;
                let subvector = sq.decompress(&compressed[start..end], self.subvector_size)?;
                data.extend(subvector);
            }

            Ok(data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_manager() -> Result<(), Box<dyn std::error::Error>> {
        let manager = CompressionManager::new(CompressionLevel::Default);
        // Use larger, repetitive data that compresses well
        let data = "Hello, world! This is a test of compression. ".repeat(100);
        let data = data.as_bytes();

        let compressed = manager.compress(data)?;
        // Compression should work well on repetitive data
        assert!(compressed.len() < data.len(),
            "Compressed size {} should be less than original size {}",
            compressed.len(), data.len());

        let decompressed = manager.decompress(&compressed)?;
        assert_eq!(decompressed, data);
        Ok(())
    }

    #[test]
    fn test_scalar_quantizer() -> Result<(), Box<dyn std::error::Error>> {
        use crate::types::vector::DenseVector;

        let mut quantizer = vector_compression::ScalarQuantizer::new(8);
        let vectors: Vec<DenseVector> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        quantizer.fit(&vectors);

        let compressed = quantizer.compress(&vectors[0])?;
        assert_eq!(compressed.len(), 4);

        let decompressed = quantizer.decompress(&compressed, 4)?;
        // Check that values are close (quantization introduces some error)
        for (original, decompressed) in vectors[0].iter().zip(decompressed.iter()) {
            assert!((original - decompressed).abs() < 0.1);
        }
        Ok(())
    }
}
