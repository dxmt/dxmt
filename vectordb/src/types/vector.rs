use crate::types::error::Error;
use serde::{Deserialize, Serialize};
use std::fmt;
#[cfg(feature = "api-swagger")]
use utoipa::ToSchema;

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
    Sparse {
        indices: Vec<usize>,
        values: Vec<f32>,
    },
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
            });
        }
        Ok(ExactScorer::new(*self, a).score(b))
    }
}

/// A query prepared once, then scored against many stored vectors.
///
/// Two things this exists for, both from a profile of the 1M benchmark under load, where
/// `DistanceMetric::calculate` was 11.5% of the server's CPU:
///
/// 1. **The query's norm is a constant.** Cosine recomputed it for every candidate, so a
///    500-candidate rerank took 500 passes over the query to produce the same number.
/// 2. **`.sum()` over `f32` does not vectorize.** Float addition is not associative, so the
///    compiler must keep the source order and emits scalar adds. The kernels below use four
///    accumulators of eight lanes and say so explicitly; the result differs from a strictly
///    sequential sum in the last bits, which is what any parallel reduction does.
///
/// Cosine and Euclidean have AVX2 kernels because they are the metrics the benchmarks use.
pub struct ExactScorer<'q> {
    metric: DistanceMetric,
    query: &'q [f32],
    /// `|query|` for cosine, so it is computed once rather than once per candidate.
    query_norm: f32,
}

impl<'q> ExactScorer<'q> {
    pub fn new(metric: DistanceMetric, query: &'q [f32]) -> Self {
        let query_norm = match metric {
            DistanceMetric::Cosine => dot_and_normsq(query, query).1.sqrt(),
            _ => 0.0,
        };
        Self {
            metric,
            query,
            query_norm,
        }
    }

    /// The distance from the query to `v`, which must have the query's length.
    pub fn score(&self, v: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => {
                let (dot, norm_sq) = dot_and_normsq(self.query, v);
                let norm_v = norm_sq.sqrt();
                if self.query_norm == 0.0 || norm_v == 0.0 {
                    return 0.0;
                }
                1.0 - dot / (self.query_norm * norm_v)
            }
            DistanceMetric::Euclidean => l2_squared(self.query, v).sqrt(),
            DistanceMetric::Dot => -dot_and_normsq(self.query, v).0,
            DistanceMetric::Manhattan => self
                .query
                .iter()
                .zip(v.iter())
                .map(|(x, y)| (x - y).abs())
                .sum(),
        }
    }

    /// `None` rather than a wrong answer when the lengths disagree.
    pub fn try_score(&self, v: &[f32]) -> Option<f32> {
        (v.len() == self.query.len()).then(|| self.score(v))
    }
}

/// `(a·b, b·b)` in one pass.
fn dot_and_normsq(a: &[f32], b: &[f32]) -> (f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if simd_available() {
            // SAFETY: the features were detected and both slices hold `n` floats.
            return unsafe { dot_and_normsq_avx2(&a[..n], &b[..n]) };
        }
    }
    let mut dot = 0.0f32;
    let mut nrm = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        nrm += b[i] * b[i];
    }
    (dot, nrm)
}

/// `|a - b|²`.
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if simd_available() {
            // SAFETY: the features were detected and both slices hold `n` floats.
            return unsafe { l2_squared_avx2(&a[..n], &b[..n]) };
        }
    }
    let mut sum = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
fn simd_available() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    _mm_cvtss_f32(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_and_normsq_avx2(a: &[f32], b: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut d0 = _mm256_setzero_ps();
    let mut d1 = _mm256_setzero_ps();
    let mut n0 = _mm256_setzero_ps();
    let mut n1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let a0 = _mm256_loadu_ps(pa.add(i));
        let b0 = _mm256_loadu_ps(pb.add(i));
        let a1 = _mm256_loadu_ps(pa.add(i + 8));
        let b1 = _mm256_loadu_ps(pb.add(i + 8));
        d0 = _mm256_fmadd_ps(a0, b0, d0);
        n0 = _mm256_fmadd_ps(b0, b0, n0);
        d1 = _mm256_fmadd_ps(a1, b1, d1);
        n1 = _mm256_fmadd_ps(b1, b1, n1);
        i += 16;
    }
    while i + 8 <= n {
        let a0 = _mm256_loadu_ps(pa.add(i));
        let b0 = _mm256_loadu_ps(pb.add(i));
        d0 = _mm256_fmadd_ps(a0, b0, d0);
        n0 = _mm256_fmadd_ps(b0, b0, n0);
        i += 8;
    }
    let mut dot = hsum256(_mm256_add_ps(d0, d1));
    let mut nrm = hsum256(_mm256_add_ps(n0, n1));
    while i < n {
        dot += *pa.add(i) * *pb.add(i);
        nrm += *pb.add(i) * *pb.add(i);
        i += 1;
    }
    (dot, nrm)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        let d1 = _mm256_sub_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
        );
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        i += 16;
    }
    while i + 8 <= n {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        i += 8;
    }
    let mut sum = hsum256(_mm256_add_ps(s0, s1));
    while i < n {
        let d = *pa.add(i) - *pb.add(i);
        sum += d * d;
        i += 1;
    }
    sum
}

impl std::str::FromStr for DistanceMetric {
    type Err = Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(DistanceMetric::Cosine),
            "euclidean" => Ok(DistanceMetric::Euclidean),
            "dot" => Ok(DistanceMetric::Dot),
            "manhattan" => Ok(DistanceMetric::Manhattan),
            _ => Err(crate::Error::InvalidConfig(format!(
                "Invalid distance metric: {}",
                s
            ))),
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

    /// Validate against the metric the table actually uses.
    ///
    /// **Whether an all-zero vector is a mistake depends entirely on the metric.** Under
    /// cosine it has no direction and every similarity to it is `0/0`; under dot it scores
    /// zero against everything, which is a ranking but not a meaningful one. Under euclidean
    /// or manhattan it is the origin: an ordinary point, at an ordinary distance from every
    /// other point, and nothing about it is degenerate.
    ///
    /// The unconditional refusal that used to be here made a standard benchmark impossible to
    /// load. GIST-960-euclidean — the dataset `ann-benchmarks` publishes ground truth for, and
    /// the one Epsilla's own example script loads — contains **ten all-zero descriptors** among
    /// its million training vectors, at indices 7030, 113718, 141476 and so on. The load
    /// stopped at the first of them with "Zero vectors are not allowed", on a euclidean table,
    /// where the vector is perfectly valid. `allow_zero_vectors` existed for this and no
    /// caller ever set it.
    pub fn validate_for_metric(
        vector: &DenseVector,
        expected_dimension: usize,
        metric: DistanceMetric,
    ) -> crate::Result<()> {
        let config = ValidationConfig {
            allow_zero_vectors: matches!(
                metric,
                DistanceMetric::Euclidean | DistanceMetric::Manhattan
            ),
            ..ValidationConfig::default()
        };
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
                "Vector cannot be empty".to_string(),
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
            });
        }

        // 3. NaN/Infinity/Range validation
        let mut is_zero_vector = true;
        for (i, &value) in vector.iter().enumerate() {
            if value.is_nan() {
                return Err(crate::Error::InvalidVector(format!(
                    "NaN value at index {}",
                    i
                )));
            }

            if value.is_infinite() {
                return Err(crate::Error::InvalidVector(format!(
                    "Infinite value at index {}: {}",
                    i, value
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
                "a zero vector has no direction, so this table's metric cannot rank it. \
                 Euclidean and manhattan tables accept it -- it is the origin, an ordinary \
                 point -- and cosine and dot tables do not."
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// 规范化向量（单位化）
    pub fn normalize(vector: &mut DenseVector) -> crate::Result<()> {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 {
            return Err(crate::Error::InvalidVector(
                "Cannot normalize zero vector".to_string(),
            ));
        }

        for value in vector.iter_mut() {
            *value /= norm
        }

        Ok(())
    }
}

#[cfg(test)]
mod scorer_tests {
    use super::*;

    fn vecs(dim: usize) -> (Vec<f32>, Vec<f32>) {
        let mut st = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 40) as f32 / 16_777_216.0 - 0.5
        };
        (
            (0..dim).map(|_| next()).collect(),
            (0..dim).map(|_| next()).collect(),
        )
    }

    /// The vectorized kernels must agree with a plain sequential sum to within the error a
    /// reordered float reduction can introduce, at every length around the 8- and 16-lane
    /// boundaries. Exact equality is the wrong assertion: four accumulators add in a
    /// different order on purpose.
    #[test]
    fn the_kernels_agree_with_a_sequential_sum() {
        for dim in [1usize, 3, 7, 8, 9, 15, 16, 17, 31, 64, 127, 768, 1536] {
            let (a, b) = vecs(dim);
            let dot_seq: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let nrm_seq: f64 = b.iter().map(|y| *y as f64 * *y as f64).sum();
            let (dot, nrm) = dot_and_normsq(&a, &b);
            let tol = (dim as f32) * 1e-6;
            assert!(
                (dot - dot_seq as f32).abs() <= tol.max(dot_seq.abs() as f32 * 1e-5),
                "dim {dim}: dot {dot} vs {dot_seq}"
            );
            assert!(
                (nrm - nrm_seq as f32).abs() <= tol.max(nrm_seq.abs() as f32 * 1e-5),
                "dim {dim}: normsq {nrm} vs {nrm_seq}"
            );

            let l2_seq: f64 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| {
                    let d = *x as f64 - *y as f64;
                    d * d
                })
                .sum();
            let l2 = l2_squared(&a, &b);
            assert!(
                (l2 - l2_seq as f32).abs() <= tol.max(l2_seq.abs() as f32 * 1e-5),
                "dim {dim}: l2 {l2} vs {l2_seq}"
            );
        }
    }

    /// Scoring many candidates with one scorer must give what `calculate` gives one at a
    /// time, for every metric -- that is the contract the rerank relies on.
    #[test]
    fn a_hoisted_query_scores_the_same_as_a_pairwise_call() {
        let dim = 768;
        let (query, _) = vecs(dim);
        let candidates: Vec<Vec<f32>> = (0..8).map(|_| vecs(dim).1).collect();
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::Dot,
            DistanceMetric::Manhattan,
        ] {
            let scorer = ExactScorer::new(metric, &query);
            for c in &candidates {
                let hoisted = scorer.score(c);
                let pairwise = metric.calculate(&query, c).unwrap();
                assert!(
                    (hoisted - pairwise).abs() <= 1e-5,
                    "{metric:?}: {hoisted} vs {pairwise}"
                );
            }
        }
    }

    /// A zero vector has no direction, and a length mismatch is not a distance.
    #[test]
    fn degenerate_inputs_do_not_produce_garbage() {
        let zero = vec![0.0f32; 8];
        let (v, _) = vecs(8);
        assert_eq!(
            ExactScorer::new(DistanceMetric::Cosine, &zero).score(&v),
            0.0
        );
        assert_eq!(
            ExactScorer::new(DistanceMetric::Cosine, &v).score(&zero),
            0.0
        );
        assert_eq!(
            ExactScorer::new(DistanceMetric::Cosine, &v).try_score(&zero[..4]),
            None
        );
        assert!(DistanceMetric::Cosine.calculate(&v, &zero[..4]).is_err());
        // A vector against itself is distance zero under cosine and euclidean.
        assert!(ExactScorer::new(DistanceMetric::Cosine, &v).score(&v).abs() <= 1e-6);
        assert!(ExactScorer::new(DistanceMetric::Euclidean, &v).score(&v) <= 1e-6);
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
