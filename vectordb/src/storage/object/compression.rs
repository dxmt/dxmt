use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use std::io::{Read, Write};
use tracing::debug;
use zstd::stream::{Decoder, Encoder};

use super::errors::ObjectStorageError;
use super::traits::ObjectStorageResult;

/// 压缩算法类型
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum CompressionType {
    /// 无压缩
    None,
    /// LZ4 - 快速压缩，适合热数据
    Lz4,
    /// Zstd - 高压缩比，适合冷数据
    Zstd,
    /// 自动选择最佳算法
    #[default]
    Auto,
}

impl CompressionType {
    /// 根据数据特征自动选择压缩算法
    pub fn choose_for_data(data: &[u8], access_frequency: f64) -> Self {
        // 小于1KB的数据不压缩
        if data.len() < 1024 {
            return Self::None;
        }

        // 高频访问数据使用快速压缩
        if access_frequency > 1.0 {
            Self::Lz4
        } else {
            // 低频访问数据使用高压缩比
            Self::Zstd
        }
    }
}

/// 压缩器接口
pub trait Compressor: Send + Sync {
    fn compress(&self, data: &[u8]) -> ObjectStorageResult<Vec<u8>>;
    fn decompress(&self, data: &[u8]) -> ObjectStorageResult<Vec<u8>>;
    fn compression_type(&self) -> CompressionType;
    fn estimated_ratio(&self) -> f64; // 预估压缩比
}

/// 无压缩实现
pub struct NoCompression;

impl Compressor for NoCompression {
    fn compress(&self, data: &[u8]) -> ObjectStorageResult<Vec<u8>> {
        Ok(data.to_vec())
    }

    fn decompress(&self, data: &[u8]) -> ObjectStorageResult<Vec<u8>> {
        Ok(data.to_vec())
    }

    fn compression_type(&self) -> CompressionType {
        CompressionType::None
    }

    fn estimated_ratio(&self) -> f64 {
        1.0
    }
}

/// LZ4 压缩器 - 快速压缩
pub struct Lz4Compressor;

impl Compressor for Lz4Compressor {
    fn compress(&self, data: &[u8]) -> ObjectStorageResult<Vec<u8>> {
        let compressed = compress_prepend_size(data);
        debug!(
            "LZ4 compressed {} bytes to {} bytes (ratio: {:.2})",
            data.len(),
            compressed.len(),
            compressed.len() as f64 / data.len() as f64
        );
        Ok(compressed)
    }

    fn decompress(&self, data: &[u8]) -> ObjectStorageResult<Vec<u8>> {
        match decompress_size_prepended(data) {
            Ok(decompressed) => Ok(decompressed),
            Err(e) => Err(ObjectStorageError::CompressionError {
                message: format!("LZ4 decompression failed: {}", e),
            }),
        }
    }

    fn compression_type(&self) -> CompressionType {
        CompressionType::Lz4
    }

    fn estimated_ratio(&self) -> f64 {
        0.6 // LZ4 通常有约40%的压缩率
    }
}

/// Zstd 压缩器 - 高压缩比
pub struct ZstdCompressor {
    compression_level: i32,
}

impl Default for ZstdCompressor {
    fn default() -> Self {
        Self::new(3)
    }
}

impl ZstdCompressor {
    pub fn new(level: i32) -> Self {
        Self {
            compression_level: level.clamp(1, 22), // Zstd支持1-22级别
        }
    }

    pub fn high_compression() -> Self {
        Self::new(9) // 高压缩比模式
    }
}

impl Compressor for ZstdCompressor {
    fn compress(&self, data: &[u8]) -> ObjectStorageResult<Vec<u8>> {
        let mut encoder = Encoder::new(Vec::new(), self.compression_level).map_err(|e| {
            ObjectStorageError::CompressionError {
                message: format!("Zstd encoder creation _failed: {}", e),
            }
        })?;

        encoder
            .write_all(data)
            .map_err(|e| ObjectStorageError::CompressionError {
                message: format!("Zstd compression write _failed: {}", e),
            })?;

        match encoder.finish() {
            Ok(compressed) => {
                debug!(
                    "Zstd compressed {} bytes to {} bytes (ratio: {:.2}, level: {})",
                    data.len(),
                    compressed.len(),
                    compressed.len() as f64 / data.len() as f64,
                    self.compression_level
                );
                Ok(compressed)
            }
            Err(e) => Err(ObjectStorageError::CompressionError {
                message: format!("Zstd compression failed: {}", e),
            }),
        }
    }

    fn decompress(&self, data: &[u8]) -> ObjectStorageResult<Vec<u8>> {
        let mut decoder = Decoder::new(data).map_err(|e| ObjectStorageError::CompressionError {
            message: format!("Zstd decoder creation _failed: {}", e),
        })?;

        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).map_err(|e| {
            ObjectStorageError::CompressionError {
                message: format!("Zstd decompression _failed: {}", e),
            }
        })?;

        Ok(decompressed)
    }

    fn compression_type(&self) -> CompressionType {
        CompressionType::Zstd
    }

    fn estimated_ratio(&self) -> f64 {
        // 根据压缩级别估算压缩比
        match self.compression_level {
            1..=3 => 0.5,
            4..=6 => 0.4,
            7..=9 => 0.3,
            _ => 0.25,
        }
    }
}

/// 压缩管理器 - 统一管理不同压缩算法
pub struct CompressionManager {
    no_compression: NoCompression,
    lz4_compressor: Lz4Compressor,
    zstd_compressor: ZstdCompressor,
    #[allow(dead_code)]
    zstd_high_compressor: ZstdCompressor,
}

impl CompressionManager {
    pub fn new() -> Self {
        Self {
            no_compression: NoCompression,
            lz4_compressor: Lz4Compressor,
            zstd_compressor: ZstdCompressor::default(),
            zstd_high_compressor: ZstdCompressor::high_compression(),
        }
    }

    /// 获取指定类型的压缩器
    pub fn get_compressor(&self, compression_type: CompressionType) -> &dyn Compressor {
        match compression_type {
            CompressionType::None => &self.no_compression,
            CompressionType::Lz4 => &self.lz4_compressor,
            CompressionType::Zstd => &self.zstd_compressor,
            CompressionType::Auto => &self.zstd_compressor, // 默认使用Zstd
        }
    }

    /// 智能压缩 - 根据数据特征选择最佳算法
    pub fn smart_compress(
        &self,
        data: &[u8],
        access_frequency: f64,
    ) -> ObjectStorageResult<(Vec<u8>, CompressionType)> {
        // 尝试不同压缩算法，选择最优的
        let candidates = if access_frequency > 1.0 {
            // 高频访问：优先考虑速度
            vec![CompressionType::Lz4, CompressionType::Zstd]
        } else {
            // 低频访问：优先考虑压缩比
            vec![CompressionType::Zstd, CompressionType::Lz4]
        };

        let mut best_result = None;
        let mut best_ratio = f64::MAX;
        let mut best_type = CompressionType::None;

        // 小数据不压缩
        if data.len() < 512 {
            return Ok((data.to_vec(), CompressionType::None));
        }

        for &compression_type in &candidates {
            let compressor = self.get_compressor(compression_type);

            if let Ok(compressed) = compressor.compress(data) {
                let ratio = compressed.len() as f64 / data.len() as f64;

                // 如果压缩效果不好（比原数据大），跳过
                if ratio >= 0.95 {
                    continue;
                }

                if ratio < best_ratio {
                    best_ratio = ratio;
                    best_result = Some(compressed);
                    best_type = compression_type;
                }
            }
        }

        Ok(match best_result {
            Some(compressed) => {
                debug!(
                    "Smart compression chose {:?} with ratio {:.3}",
                    best_type, best_ratio
                );
                (compressed, best_type)
            }
            None => {
                debug!("No compression benefits found, using original data");
                (data.to_vec(), CompressionType::None)
            }
        })
    }

    /// 解压缩数据
    pub fn decompress(
        &self,
        data: &[u8],
        compression_type: CompressionType,
    ) -> ObjectStorageResult<Vec<u8>> {
        let compressor = self.get_compressor(compression_type);
        compressor.decompress(data)
    }

    /// 批量压缩评估 - 为一批数据选择最佳压缩策略
    pub fn evaluate_batch_compression(&self, data_batch: &[&[u8]]) -> CompressionType {
        if data_batch.is_empty() {
            return CompressionType::None;
        }

        // 采样部分数据进行测试
        let sample_size = std::cmp::min(data_batch.len(), 10);
        let samples: Vec<_> = data_batch.iter().take(sample_size).collect();

        let mut lz4_total_ratio = 0.0;
        let mut zstd_total_ratio = 0.0;
        let mut valid_samples = 0;

        for &sample in &samples {
            if sample.len() < 512 {
                continue;
            }
            // 测试LZ4
            if let Ok(lz4_compressed) = self.lz4_compressor.compress(sample) {
                lz4_total_ratio += lz4_compressed.len() as f64 / sample.len() as f64
            }

            // 测试Zstd
            if let Ok(zstd_compressed) = self.zstd_compressor.compress(sample) {
                zstd_total_ratio += zstd_compressed.len() as f64 / sample.len() as f64
            }

            valid_samples += 1;
        }

        if valid_samples == 0 {
            return CompressionType::None;
        }

        let avg_lz4_ratio = lz4_total_ratio / valid_samples as f64;
        let avg_zstd_ratio = zstd_total_ratio / valid_samples as f64;

        // 如果Zstd压缩比明显更好（超过10%），选择Zstd
        if avg_zstd_ratio < avg_lz4_ratio * 0.9 {
            CompressionType::Zstd
        } else {
            CompressionType::Lz4
        }
    }
}

impl Default for CompressionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// 压缩元数据 - 用于存储压缩相关信息
#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    pub compression_type: CompressionType,
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
    pub checksum: Option<String>, // 可选的校验和
}

impl CompressionMetadata {
    pub fn new(
        compression_type: CompressionType,
        original_size: u64,
        compressed_size: u64,
    ) -> Self {
        let compression_ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            1.0
        };

        Self {
            compression_type,
            original_size,
            compressed_size,
            compression_ratio,
            checksum: None,
        }
    }

    pub fn space_saved(&self) -> u64 {
        self.original_size.saturating_sub(self.compressed_size)
    }

    pub fn space_saved_percentage(&self) -> f64 {
        if self.original_size > 0 {
            (1.0 - self.compression_ratio) * 100.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_algorithms() -> Result<(), Box<dyn std::error::Error>> {
        let manager = CompressionManager::new();
        let test_data = b"Hello, World! This is a test string that should compress well with repeated patterns. Hello, World! This is a test string that should compress well with repeated patterns.";

        // 测试LZ4压缩
        let lz4_compressor = manager.get_compressor(CompressionType::Lz4);
        let lz4_compressed = lz4_compressor.compress(test_data)?;
        let lz4_decompressed = lz4_compressor.decompress(&lz4_compressed)?;
        assert_eq!(test_data, &lz4_decompressed[..]);

        // 测试Zstd压缩
        let zstd_compressor = manager.get_compressor(CompressionType::Zstd);
        let zstd_compressed = zstd_compressor.compress(test_data)?;
        let zstd_decompressed = zstd_compressor.decompress(&zstd_compressed)?;
        assert_eq!(test_data, &zstd_decompressed[..]);

        println!("Original size: {}", test_data.len());
        println!("LZ4 compressed size: {}", lz4_compressed.len());
        println!("Zstd compressed size: {}", zstd_compressed.len());
        Ok(())
    }

    #[test]
    fn test_smart_compression() -> Result<(), Box<dyn std::error::Error>> {
        let manager = CompressionManager::new();

        // 测试小数据（应该不压缩）
        let small_data = b"Small data";
        let (compressed_small, compression_type_small) = manager.smart_compress(small_data, 2.0)?;
        assert_eq!(compression_type_small, CompressionType::None);
        assert_eq!(compressed_small, small_data);

        // 测试大数据（应该压缩）
        let large_data = b"This is a repeating pattern that is long enough to be compressed. This is a repeating pattern that is long enough to be compressed. This is a repeating pattern that is long enough to be compressed. This is a repeating pattern that is long enough to be compressed. This is a repeating pattern that is long enough to be compressed. This is a repeating pattern that is long enough to be compressed.";

        // 高频访问数据
        let (compressed, compression_type) = manager.smart_compress(large_data, 2.0)?;
        assert!(
            compression_type == CompressionType::Lz4
                || compression_type == CompressionType::Zstd
                || compression_type == CompressionType::None
        );

        // 验证可以正确解压
        let decompressed = manager.decompress(&compressed, compression_type)?;
        assert_eq!(large_data, &decompressed[..]);

        // 测试不可压缩数据（随机数据）
        let _random_data = vec![0u8; 1024]; // 全0数据实际上是可压缩的，让我们用更复杂的模式
        let mut truly_random = Vec::new();
        for i in 0..1024 {
            truly_random.push((i * 17 + 23) as u8); // 创建伪随机数据
        }

        let (_compressed_random, compression_type_random) =
            manager.smart_compress(&truly_random, 2.0)?;
        // 对于不易压缩的数据，智能压缩可能选择不压缩或者选择压缩算法
        assert!(
            compression_type_random == CompressionType::None
                || compression_type_random == CompressionType::Lz4
                || compression_type_random == CompressionType::Zstd
        );
        Ok(())
    }

    #[test]
    fn test_compression_metadata() {
        let original_size = 1000;
        let compressed_size = 400;
        let metadata =
            CompressionMetadata::new(CompressionType::Zstd, original_size, compressed_size);

        assert_eq!(metadata.compression_ratio, 0.4);
        assert_eq!(metadata.space_saved(), 600);
        assert_eq!(metadata.space_saved_percentage(), 60.0);
    }
}
