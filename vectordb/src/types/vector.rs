use crate::types::error::{Error};
use serde::{Deserialize, Serialize};
#[cfg(feature = "api-swagger")]
use utoipa::ToSchema;
use std::fmt;

/// 密集向量类型
pub type DenseVector = Vec<f32>;

/// 记录ID类型
pub type RecordId = String;

/// 负载元数据类型
pub type Payload = serde_json::Value;

/// 向量类型的枚举
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(ToSchema))]
#[serde(tag = "type", content = "data")]
pub enum Vector {
    Dense(Vec<f32>),
    Sparse { indices: Vec<usize>, values: Vec<f32> },
}

/// 距离度量枚举
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    /// 余弦相似度
    Cosine,
    /// 欧几里得距离
    Euclidean,
    /// 点积
    Dot,
    /// 曼哈顿距离
    Manhattan,
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceMetric::Cosine => write!(f, "cosine"),
            DistanceMetric::Euclidean => write!(f, "euclidean"),
            DistanceMetric::Dot => write!(f, "dot"),
            DistanceMetric::Manhattan => write!(f, "manhattan"),
        }
    }
}

impl DistanceMetric {
    /// 计算两个向量之间的距离
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> crate::Result<f32> {
        if a.len() != b.len() {
            return Err(crate::Error::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            })
        }

        match self {
            DistanceMetric::Cosine => {
                let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    return Ok(0.0);
                }

                // 返回余弦距离 (1 - 余弦相似度)
                Ok(1.0 - (dot_product / (norm_a * norm_b)))
            }
            DistanceMetric::Euclidean => {
                let sum_squared_diff: f32 =
                    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
                Ok(sum_squared_diff.sqrt())
            }
            DistanceMetric::Dot => {
                // 点积相似度，返回负值作为距离（越大越相似）
                let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                Ok(-dot_product)
            }
            DistanceMetric::Manhattan => {
                let sum_abs_diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
                Ok(sum_abs_diff)
            }
        }
    }
}

impl std::str::FromStr for DistanceMetric {
    type Err = Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(DistanceMetric::Cosine),
            "euclidean" => Ok(DistanceMetric::Euclidean),
            "dot" => Ok(DistanceMetric::Dot),
            "manhattan" => Ok(DistanceMetric::Manhattan),
            _ => Err(crate::Error::InvalidConfig(format!("Invalid distance metric: {}", s))),
        }
    }
}

/// Configuration for vector validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum allowed dimension
    pub max_dimension: usize,
    /// Minimum allowed dimension
    pub min_dimension: usize,
    /// Whether to allow zero vectors (all zeros)
    pub allow_zero_vectors: bool,
    /// Optional value range validation (min, max)
    pub validate_range: Option<(f32, f32)>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_dimension: 4096,
            min_dimension: 1,
            allow_zero_vectors: false,
            validate_range: None,
        }
    }
}

/// 向量验证工具
pub struct VectorValidator;

impl VectorValidator {
    /// 验证向量是否有效
    pub fn validate(vector: &DenseVector, expected_dimension: usize) -> crate::Result<()> {
        let config = ValidationConfig::default();
        Self::validate_with_config(vector, expected_dimension, &config)
    }

    /// Validate vector with custom configuration
    pub fn validate_with_config(
        vector: &DenseVector,
        expected_dimension: usize,
        config: &ValidationConfig,
    ) -> crate::Result<()> {
        // 1. Empty check
        if vector.is_empty() {
            return Err(crate::Error::InvalidVector(
                "Vector cannot be empty".to_string()
            ));
        }

        // 2. Dimension validation with bounds
        if expected_dimension < config.min_dimension {
            return Err(crate::Error::InvalidVector(format!(
                "Expected dimension {} is below minimum {}",
                expected_dimension, config.min_dimension
            )));
        }

        if expected_dimension > config.max_dimension {
            return Err(crate::Error::InvalidVector(format!(
                "Expected dimension {} exceeds maximum {}",
                expected_dimension, config.max_dimension
            )));
        }

        if vector.len() != expected_dimension {
            return Err(crate::Error::DimensionMismatch {
                expected: expected_dimension,
                actual: vector.len(),
            })
        }

        // 3. NaN/Infinity/Range validation
        let mut is_zero_vector = true;
        for (i, &value) in vector.iter().enumerate() {
            if value.is_nan() {
                return Err(crate::Error::InvalidVector(format!(
                    "NaN value at index {}", i
                )));
            }

            if value.is_infinite() {
                return Err(crate::Error::InvalidVector(format!(
                    "Infinite value at index {}: {}", i, value
                )));
            }

            if let Some((min, max)) = config.validate_range {
                if value < min || value > max {
                    return Err(crate::Error::InvalidVector(format!(
                        "Value at index {} ({}) outside allowed range [{}, {}]",
                        i, value, min, max
                    )));
                }
            }

            if value != 0.0 {
                is_zero_vector = false;
            }
        }

        // 4. Zero vector check
        if is_zero_vector && !config.allow_zero_vectors {
            return Err(crate::Error::InvalidVector(
                "Zero vectors are not allowed".to_string()
            ));
        }

        Ok(())
    }

    /// 规范化向量（单位化）
    pub fn normalize(vector: &mut DenseVector) -> crate::Result<()> {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 {
            return Err(crate::Error::InvalidVector("Cannot normalize zero vector".to_string()))
        }

        for value in vector.iter_mut() {
            *value /= norm
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance() -> Result<(), Box<dyn std::error::Error>> {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let distance = DistanceMetric::Cosine.calculate(&a, &b)?;
        assert!((distance - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_euclidean_distance() -> Result<(), Box<dyn std::error::Error>> {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let distance = DistanceMetric::Euclidean.calculate(&a, &b)?;
        assert!((distance - 5.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_vector_validation() {
        let valid_vector = vec![1.0, 2.0, 3.0];
        assert!(VectorValidator::validate(&valid_vector, 3).is_ok());

        let invalid_dimension = vec![1.0, 2.0];
        assert!(VectorValidator::validate(&invalid_dimension, 3).is_err());

        let invalid_values = vec![1.0, f32::NAN, 3.0];
        assert!(VectorValidator::validate(&invalid_values, 3).is_err())
    }

    #[test]
    fn test_vector_normalization() -> Result<(), Box<dyn std::error::Error>> {
        let mut vector = vec![3.0, 4.0];
        VectorValidator::normalize(&mut vector)?;
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        Ok(())
    }
}
