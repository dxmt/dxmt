// SIMD优化的距离计算
// 使用硬件指令集加速向量运算

use crate::types::vector::DenseVector;
use crate::types::vector::DistanceMetric;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD距离计算特性
pub trait SimdDistance {
    /// SIMD加速的距离计算
    fn calculate_simd(&self, a: &[f32], b: &[f32]) -> crate::Result<f32>;

    /// 检查是否支持SIMD
    fn supports_simd(&self) -> bool;
}

/// Dot product, dispatched once to the widest SIMD this CPU has.
///
/// Two things this does that the `DistanceMetric::calculate_simd` path does not. It is a bare dot
/// product, which is all a cosine comparison needs once both sides are unit length — the general
/// path recomputes both norms and two square roots on every call. And it resolves the CPU feature
/// check **once** rather than per call: `is_x86_feature_detected!` caches its answer, but the
/// branch and the atomic load are still on the hot path of every distance evaluation, and this is
/// the hot path of both search and index construction. Resolving the metric function once, at
/// construction, keeps it off every comparison.
type DotProductFn = fn(&[f32], &[f32]) -> f32;

/// The kernels behind these functions read `a.len()` floats from *both* slices through raw
/// pointers, so unequal lengths were an out-of-bounds read -- guarded only by a
/// `debug_assert`, i.e. not at all in a release build. A named-vector search with a short
/// query reached them that way. The callers check widths at the API; this makes a missed check
/// a panic instead of a read of whatever follows the shorter vector. One comparison per call.
#[inline]
fn check_lengths(a: &[f32], b: &[f32]) {
    assert_eq!(
        a.len(),
        b.len(),
        "distance between vectors of different lengths"
    );
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    check_lengths(a, b);
    static DISPATCH: std::sync::OnceLock<DotProductFn> = std::sync::OnceLock::new();
    DISPATCH.get_or_init(resolve_dot_product)(a, b)
}

fn resolve_dot_product() -> DotProductFn {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return |a, b| unsafe { dot_product_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |a, b| unsafe { dot_product_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return |a, b| unsafe { dot_product_sse(a, b) };
        }
    }
    |a, b| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Squared euclidean distance, resolved once like [`dot_product`]. The square root the
/// metric's own path takes is a cost and nothing else when the caller only ranks -- the
/// quantized graph compares decoded codes with this, thousands of times per insert.
pub fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    check_lengths(a, b);
    static DISPATCH: std::sync::OnceLock<DotProductFn> = std::sync::OnceLock::new();
    DISPATCH.get_or_init(resolve_squared_euclidean)(a, b)
}

fn resolve_squared_euclidean() -> DotProductFn {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |a, b| unsafe { squared_euclidean_avx2(a, b) };
        }
    }
    |a, b| a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Euclidean distance, resolved once like [`dot_product`]: the metric's `calculate_simd`
/// path re-runs the feature check and two levels of dispatch on every call, which at 25
/// dimensions was a third as much time as the arithmetic itself.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    check_lengths(a, b);
    static DISPATCH: std::sync::OnceLock<DotProductFn> = std::sync::OnceLock::new();
    DISPATCH.get_or_init(resolve_euclidean_distance)(a, b)
}

fn resolve_euclidean_distance() -> DotProductFn {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return |a, b| unsafe { euclidean_distance_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |a, b| unsafe { euclidean_distance_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return |a, b| unsafe { euclidean_distance_sse(a, b) };
        }
    }
    |a, b| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }
}

impl SimdDistance for DistanceMetric {
    fn calculate_simd(&self, a: &[f32], b: &[f32]) -> crate::Result<f32> {
        if a.len() != b.len() {
            return Err(crate::Error::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            // ✅ AVX-512 优先级最高 (16 floats at a time)
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return self.calculate_avx512(a, b);
                }
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    return self.calculate_avx2(a, b);
                }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe {
                    return self.calculate_sse(a, b);
                }
            }
        }

        // 回退到标准实现
        self.calculate(a, b)
    }

    fn supports_simd(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx512f")
                || is_x86_feature_detected!("avx2")
                || is_x86_feature_detected!("sse4.1")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl DistanceMetric {
    /// AVX-512 优化的距离计算 (16 floats at a time)
    #[target_feature(enable = "avx512f")]
    unsafe fn calculate_avx512(&self, a: &[f32], b: &[f32]) -> crate::Result<f32> {
        Ok(match self {
            DistanceMetric::Cosine => 1.0 - cosine_similarity_avx512(a, b),
            DistanceMetric::Euclidean => euclidean_distance_avx512(a, b),
            DistanceMetric::Dot => -dot_product_avx512(a, b),
            DistanceMetric::Manhattan => manhattan_distance_avx512(a, b),
        })
    }

    /// AVX2优化的距离计算
    #[target_feature(enable = "avx2")]
    unsafe fn calculate_avx2(&self, a: &[f32], b: &[f32]) -> crate::Result<f32> {
        Ok(match self {
            DistanceMetric::Cosine => 1.0 - cosine_similarity_avx2(a, b),
            DistanceMetric::Euclidean => euclidean_distance_avx2(a, b),
            DistanceMetric::Dot => -dot_product_avx2(a, b),
            DistanceMetric::Manhattan => manhattan_distance_avx2(a, b),
        })
    }

    /// SSE优化的距离计算
    #[target_feature(enable = "sse4.1")]
    unsafe fn calculate_sse(&self, a: &[f32], b: &[f32]) -> crate::Result<f32> {
        Ok(match self {
            DistanceMetric::Cosine => 1.0 - cosine_similarity_sse(a, b),
            DistanceMetric::Euclidean => euclidean_distance_sse(a, b),
            DistanceMetric::Dot => -dot_product_sse(a, b),
            DistanceMetric::Manhattan => manhattan_distance_sse(a, b),
        })
    }
}

// ============================================================================
// AVX-512 Implementation (16 floats at a time)
// ============================================================================

#[cfg(target_arch = "x86_64")]
/// AVX-512 优化的点积计算 (16 floats at a time)
///
/// # Safety
/// - Requires AVX-512F CPU feature to be available
/// - Input slices must have the same length
/// - Pointers must be valid for the entire slice length
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have equal length");

    if a.is_empty() {
        return 0.0;
    }

    let mut sum = _mm512_setzero_ps();
    let mut i = 0;
    let len = a.len();

    // Process 16 elements at a time (AVX-512)
    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));

        // Fused multiply-add: sum = sum + va * vb
        sum = _mm512_fmadd_ps(va, vb, sum);
        i += 16;
    }

    // Horizontal sum (AVX-512 provides _mm512_reduce_add_ps for simpler reduction)
    let mut result = _mm512_reduce_add_ps(sum);

    // Process remaining elements
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
/// AVX-512 优化的向量范数平方计算
///
/// # Safety
/// - Requires AVX-512F CPU feature to be available
/// - Input slice must be valid for its entire length
#[target_feature(enable = "avx512f")]
unsafe fn norm_squared_avx512(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }

    let mut sum = _mm512_setzero_ps();
    let mut i = 0;
    let len = v.len();

    // Process 16 elements at a time
    while i + 16 <= len {
        let vec = _mm512_loadu_ps(v.as_ptr().add(i));
        // Fused multiply-add: sum = sum + vec * vec
        sum = _mm512_fmadd_ps(vec, vec, sum);
        i += 16;
    }

    // Horizontal sum
    let mut result = _mm512_reduce_add_ps(sum);

    // Process remaining elements
    while i < len {
        result += v[i] * v[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
/// AVX-512 优化的余弦相似度
#[target_feature(enable = "avx512f")]
unsafe fn cosine_similarity_avx512(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = dot_product_avx512(a, b);
    let norm_a = norm_squared_avx512(a).sqrt();
    let norm_b = norm_squared_avx512(b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(target_arch = "x86_64")]
/// AVX-512 优化的欧几里得距离
///
/// # Safety
/// - Requires AVX-512F CPU feature to be available
/// - Input slices must have the same length
#[target_feature(enable = "avx512f")]
unsafe fn euclidean_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }

    let mut sum = _mm512_setzero_ps();
    let mut i = 0;
    let len = a.len();

    // Process 16 elements at a time
    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);

        // Fused multiply-add: sum = sum + diff * diff
        sum = _mm512_fmadd_ps(diff, diff, sum);
        i += 16;
    }

    // Horizontal sum
    let mut result = _mm512_reduce_add_ps(sum);

    // Process remaining elements
    while i < len {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result.sqrt()
}

#[cfg(target_arch = "x86_64")]
/// AVX-512 优化的曼哈顿距离
#[target_feature(enable = "avx512f")]
unsafe fn manhattan_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }

    let mut sum = _mm512_setzero_ps();
    let mut i = 0;
    let len = a.len();

    // Process 16 elements at a time
    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);

        // Compute absolute value using bit manipulation (clear sign bit)
        // Create a mask with all bits set except the sign bit
        let abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
        let abs_diff = _mm512_and_ps(diff, abs_mask);
        sum = _mm512_add_ps(sum, abs_diff);
        i += 16;
    }

    // Horizontal sum
    let mut result = _mm512_reduce_add_ps(sum);

    // Process remaining elements
    while i < len {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

// ============================================================================
// AVX2 Implementation (8 floats at a time)
// ============================================================================

#[cfg(target_arch = "x86_64")]
/// AVX2优化的点积计算
///
/// # Safety
/// - Requires AVX2 CPU feature to be available
/// - Input slices must have the same length
/// - Pointers must be valid for the entire slice length
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have equal length");
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // Four independent accumulators, fused multiply-add. One accumulator makes every
    // iteration wait on the previous add (four cycles of latency per eight floats, so two
    // floats a cycle); the core retires two FMAs a cycle, which four chains let it do. This
    // is the difference between the graph build spending 63% of its CPU in this function
    // and spending it on the walk -- measured with the in-process profiler on 100K x 100,
    // where the build went from 3,678 to the figure in the commit that introduced this.
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 32 <= len {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        s1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
            s1,
        );
        s2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
            s2,
        );
        s3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
            s3,
        );
        i += 32;
    }
    while i + 8 <= len {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        i += 8;
    }
    let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    let mut result = hsum256(sum);
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }
    result
}

/// Horizontal sum of eight lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(low, high);
    let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 1));
    _mm_cvtss_f32(s32)
}

#[cfg(target_arch = "x86_64")]
/// AVX2优化的向量范数计算
///
/// # Safety
/// - Requires AVX2 CPU feature to be available
/// - Input slice must be valid for its entire length
#[target_feature(enable = "avx2")]
unsafe fn norm_squared_avx2(v: &[f32]) -> f32 {
    // Safety check: handle empty slice
    if v.is_empty() {
        return 0.0;
    }

    // Safety check: ensure valid pointer
    debug_assert!(v.as_ptr() as usize != 0, "Null pointer detected");

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    let len = v.len();

    // Process 8 elements at a time with bounds checking
    while i + 8 <= len {
        // Safety: we've checked that i + 8 <= len
        debug_assert!(i + 8 <= len, "Index out of bounds in norm_squared_avx2");

        let vec = _mm256_loadu_ps(v.as_ptr().add(i));
        let squared = _mm256_mul_ps(vec, vec);
        sum = _mm256_add_ps(sum, squared);
        i += 8
    }

    // 水平求和
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_low, sum_high);

    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    // 处理剩余元素
    while i < v.len() {
        result += v[i] * v[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
/// AVX2优化的余弦相似度
#[target_feature(enable = "avx2")]
unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = dot_product_avx2(a, b);
    let norm_a = norm_squared_avx2(a).sqrt();
    let norm_b = norm_squared_avx2(b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(target_arch = "x86_64")]
/// AVX2优化的欧几里得距离
///
/// # Safety
/// - Requires AVX2 CPU feature to be available
/// - Input slices must have the same length
/// - Pointers must be valid for the entire slice length
#[target_feature(enable = "avx2,fma")]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    squared_euclidean_avx2(a, b).sqrt()
}

/// The sum of squared differences, before the root `euclidean_distance_avx2` takes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn squared_euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have equal length");
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // Four accumulators and FMA, for the reason given on `dot_product_avx2`.
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 32 <= len {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        let d1 = _mm256_sub_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
        );
        let d2 = _mm256_sub_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
        );
        let d3 = _mm256_sub_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
        );
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
        i += 32;
    }
    while i + 8 <= len {
        let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        s0 = _mm256_fmadd_ps(d, d, s0);
        i += 8;
    }
    let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    let mut result = hsum256(sum);
    while i < len {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
/// AVX2优化的曼哈顿距离
#[target_feature(enable = "avx2")]
unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= a.len() {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);

        // 计算绝对值 - AVX2没有直接的abs指令，使用符号掩码
        let sign_mask = _mm256_set1_ps(-0.0);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);

        sum = _mm256_add_ps(sum, abs_diff);
        i += 8
    }

    // 水平求和
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_low, sum_high);

    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    // 处理剩余元素
    while i < a.len() {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

// SSE实现 (类似但使用128位寄存器)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // 处理4个元素一组
    while i + 4 <= a.len() {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let prod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, prod);
        i += 4
    }

    // 水平求和
    let sum2 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    let mut result = _mm_cvtss_f32(sum1);

    // 处理剩余元素
    while i < a.len() {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn cosine_similarity_sse(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = dot_product_sse(a, b);

    // 计算范数（简化版，实际应该也用SSE优化）
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn euclidean_distance_sse(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= a.len() {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(va, vb);
        let squared = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, squared);
        i += 4
    }

    // 水平求和
    let sum2 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    let mut result = _mm_cvtss_f32(sum1);

    // 处理剩余元素
    while i < a.len() {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn manhattan_distance_sse(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= a.len() {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(va, vb);

        // 计算绝对值
        let sign_mask = _mm_set1_ps(-0.0);
        let abs_diff = _mm_andnot_ps(sign_mask, diff);

        sum = _mm_add_ps(sum, abs_diff);
        i += 4
    }

    // 水平求和
    let sum2 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    let mut result = _mm_cvtss_f32(sum1);

    // 处理剩余元素
    while i < a.len() {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

/// 批量距离计算（SIMD优化）
pub fn batch_distance_calculation(
    queries: &[DenseVector],
    targets: &[DenseVector],
    metric: &DistanceMetric,
) -> crate::Result<Vec<Vec<f32>>> {
    let mut results = Vec::with_capacity(queries.len());

    for query in queries {
        let mut query_results = Vec::with_capacity(targets.len());

        for target in targets {
            let distance = if metric.supports_simd() {
                metric.calculate_simd(query, target)?
            } else {
                metric.calculate(query, target)?
            };
            query_results.push(distance);
        }

        results.push(query_results);
    }

    Ok(results)
}

/// 向量归一化（SIMD优化）
pub fn normalize_vector_simd(vector: &mut [f32]) -> crate::Result<()> {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            let norm_sq = norm_squared_avx2(vector);
            let norm = norm_sq.sqrt();

            if norm == 0.0 {
                return Err(crate::Error::InvalidInput(
                    "Cannot normalize zero vector".to_string(),
                ));
            }

            let norm_vec = _mm256_set1_ps(norm);
            let mut i = 0;

            while i + 8 <= vector.len() {
                let vec = _mm256_loadu_ps(vector.as_ptr().add(i));
                let normalized = _mm256_div_ps(vec, norm_vec);
                _mm256_storeu_ps(vector.as_mut_ptr().add(i), normalized);
                i += 8;
            }

            // 处理剩余元素
            while i < vector.len() {
                vector[i] /= norm;
                i += 1;
            }

            return Ok(());
        }
    }

    // 回退到标准实现
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm == 0.0 {
        return Err(crate::Error::InvalidInput(
            "Cannot normalize zero vector".to_string(),
        ));
    }

    for value in vector.iter_mut() {
        *value /= norm;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The unrolled kernels against the scalar definition, at lengths that exercise every
    /// branch: under one 8-lane block, one block, a 32-lane block plus an 8-lane remainder
    /// plus a scalar tail, and the widths tables are built at.
    #[test]
    fn euclidean_dispatch_agrees_with_the_metric_path() {
        for len in [1usize, 3, 25, 100, 768] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).cos()).collect();
            let via_metric = DistanceMetric::Euclidean.calculate_simd(&a, &b).unwrap();
            let got = euclidean_distance(&a, &b);
            assert!(
                (got - via_metric).abs() <= 1e-5 * via_metric.max(1.0),
                "len {len}"
            );
        }
    }

    #[test]
    fn squared_euclidean_agrees_with_the_scalar_definition() {
        for len in [1usize, 3, 7, 8, 25, 31, 32, 65, 100, 768] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).cos()).collect();
            let plain: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            let got = squared_euclidean(&a, &b);
            assert!(
                (got - plain).abs() <= 1e-4 * plain.max(1.0),
                "len {len}: {got} vs {plain}"
            );
        }
    }

    #[test]
    fn unrolled_kernels_agree_with_the_scalar_definition() {
        for len in [
            1usize, 3, 7, 8, 9, 15, 16, 25, 31, 32, 33, 40, 47, 64, 100, 128, 200, 768, 960, 1536,
        ] {
            let a: Vec<f32> = (0..len)
                .map(|i| ((i * 7919) % 101) as f32 / 101.0 - 0.5)
                .collect();
            let b: Vec<f32> = (0..len)
                .map(|i| ((i * 104729) % 97) as f32 / 97.0 - 0.5)
                .collect();
            let dot_ref: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let l2_ref: f32 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt();
            let dot = dot_product(&a, &b);
            let l2 = DistanceMetric::Euclidean.calculate_simd(&a, &b).unwrap();
            let tol = 1e-4 * (1.0 + len as f32 / 64.0);
            assert!(
                (dot - dot_ref).abs() <= tol,
                "dot at len {len}: {dot} vs {dot_ref}"
            );
            assert!(
                (l2 - l2_ref).abs() <= tol,
                "euclidean at len {len}: {l2} vs {l2_ref}"
            );
        }
    }

    #[test]
    fn test_simd_support_detection() {
        let metric = DistanceMetric::Cosine;
        println!("SIMD support: {}", metric.supports_simd());

        // 这个测试主要用于验证编译和功能检测
        #[cfg(target_arch = "x86_64")]
        {
            println!("AVX-512F: {}", is_x86_feature_detected!("avx512f"));
            println!("AVX2: {}", is_x86_feature_detected!("avx2"));
            println!("SSE4.1: {}", is_x86_feature_detected!("sse4.1"));
        }
    }

    #[test]
    fn test_simd_vs_standard() -> Result<(), Box<dyn std::error::Error>> {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let metric = DistanceMetric::Cosine;

        let standard_result = metric.calculate(&a, &b)?;
        let simd_result = metric.calculate_simd(&a, &b)?;

        // 结果应该非常接近（允许浮点误差）
        assert!((standard_result - simd_result).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_batch_distance_calculation() -> Result<(), Box<dyn std::error::Error>> {
        let queries = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let targets = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let metric = DistanceMetric::Cosine;
        let results = batch_distance_calculation(&queries, &targets, &metric)?;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 3);
        assert_eq!(results[1].len(), 3);

        // 验证对角线上的距离最小（相同向量）
        assert!(results[0][0] < results[0][1]);
        assert!(results[1][1] < results[1][0]);
        Ok(())
    }

    #[test]
    fn test_vector_normalization_simd() -> Result<(), Box<dyn std::error::Error>> {
        let mut vector = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        normalize_vector_simd(&mut vector)?;

        // 验证归一化后的向量长度为1
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);

        // 验证比例正确
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
        Ok(())
    }
}
