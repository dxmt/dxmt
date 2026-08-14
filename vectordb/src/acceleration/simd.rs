//! SIMD acceleration for vector operations

use serde::{Deserialize, Serialize};
use std::arch::x86_64::*;
use std::sync::Arc;

use crate::types::DenseVector;

/// SIMD capability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimdLevel {
    None,
    Sse,
    Sse2,
    Sse3,
    Ssse3,
    Sse41,
    Sse42,
    Avx,
    Avx2,
    Avx512,
    Neon, // ARM
}

impl SimdLevel {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            if is_x86_feature_detected!("avx") {
                return Self::Avx;
            }
            if is_x86_feature_detected!("sse4.2") {
                return Self::Sse42;
            }
            if is_x86_feature_detected!("sse4.1") {
                return Self::Sse41;
            }
            if is_x86_feature_detected!("ssse3") {
                return Self::Ssse3;
            }
            if is_x86_feature_detected!("sse3") {
                return Self::Sse3;
            }
            if is_x86_feature_detected!("sse2") {
                return Self::Sse2;
            }
            if is_x86_feature_detected!("sse") {
                return Self::Sse;
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self::Neon;
            }
        }
        
        Self::None
    }
    
    pub fn vector_width(&self) -> usize {
        match self {
            Self::None => 1,
            Self::Sse | Self::Sse2 | Self::Sse3 | Self::Ssse3 | Self::Sse41 | Self::Sse42 => 4,
            Self::Avx | Self::Avx2 => 8,
            Self::Avx512 => 16,
            Self::Neon => 4,
        }
    }
}

/// SIMD accelerator for vector operations
pub struct SimdAccelerator {
    level: SimdLevel,
    ops: Arc<dyn VectorOps + Send + Sync>,
}

impl SimdAccelerator {
    pub fn new(level: SimdLevel) -> Self {
        let ops: Arc<dyn VectorOps + Send + Sync> = match level {
            SimdLevel::Avx512 => Arc::new(Avx512Ops),
            SimdLevel::Avx2 => Arc::new(Avx2Ops),
            SimdLevel::Avx => Arc::new(AvxOps),
            _ => Arc::new(ScalarOps),
        };
        
        Self { level, ops }
    }
    
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        self.ops.cosine_similarity(a, b)
    }
    
    pub fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.ops.euclidean_distance(a, b)
    }
    
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        self.ops.dot_product(a, b)
    }
    
    pub fn vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        self.ops.vector_add(a, b, result)
    }
    
    pub fn vector_sub(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        self.ops.vector_sub(a, b, result)
    }
    
    pub fn vector_scale(&self, a: &[f32], scalar: f32, result: &mut [f32]) {
        self.ops.vector_scale(a, scalar, result)
    }
    
    pub fn vector_normalize(&self, a: &[f32], result: &mut [f32]) {
        self.ops.vector_normalize(a, result)
    }
    
    pub fn batch_cosine_similarity(&self, query: &[f32], vectors: &[DenseVector]) -> Vec<f32> {
        vectors.iter()
            .map(|v| self.cosine_similarity(query, v))
            .collect()
    }
    
    pub fn level(&self) -> SimdLevel {
        self.level
    }
}

/// Vector operations trait
pub trait VectorOps: Send + Sync {
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32;
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32;
    fn vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]);
    fn vector_sub(&self, a: &[f32], b: &[f32], result: &mut [f32]);
    fn vector_scale(&self, a: &[f32], scalar: f32, result: &mut [f32]);
    fn vector_normalize(&self, a: &[f32], result: &mut [f32]);
}

/// Scalar (non-SIMD) implementation
struct ScalarOps;

impl VectorOps for ScalarOps {
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot = self.dot_product(a, b);
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
    
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }
    
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    
    fn vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }
    
    fn vector_sub(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] - b[i];
        }
    }
    
    fn vector_scale(&self, a: &[f32], scalar: f32, result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * scalar;
        }
    }
    
    fn vector_normalize(&self, a: &[f32], result: &mut [f32]) {
        let norm = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            self.vector_scale(a, 1.0 / norm, result);
        } else {
            result.copy_from_slice(a);
        }
    }
}

/// AVX implementation
struct AvxOps;

impl VectorOps for AvxOps {
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe { self.cosine_similarity_avx(a, b) }
    }
    
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe { self.euclidean_distance_impl(a, b) }
    }
    
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe { self.dot_product_impl(a, b) }
    }
    
    fn vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe { self.vector_add_impl(a, b, result) }
    }
    
    fn vector_sub(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe { self.vector_sub_impl(a, b, result) }
    }
    
    fn vector_scale(&self, a: &[f32], scalar: f32, result: &mut [f32]) {
        unsafe { self.vector_scale_impl(a, scalar, result) }
    }
    
    fn vector_normalize(&self, a: &[f32], result: &mut [f32]) {
        let norm = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            self.vector_scale(a, 1.0 / norm, result);
        }
    }
}

impl AvxOps {
    #[target_feature(enable = "avx")]
    unsafe fn cosine_similarity_avx(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();

        let chunks = a.len() / 8;
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));

            dot_sum = _mm256_add_ps(dot_sum, _mm256_mul_ps(a_vec, b_vec));
            norm_a_sum = _mm256_add_ps(norm_a_sum, _mm256_mul_ps(a_vec, a_vec));
            norm_b_sum = _mm256_add_ps(norm_b_sum, _mm256_mul_ps(b_vec, b_vec));
        }

        let dot = horizontal_sum_ps256(dot_sum);
        let norm_a = horizontal_sum_ps256(norm_a_sum).sqrt();
        let norm_b = horizontal_sum_ps256(norm_b_sum).sqrt();

        let remainder = a.len() % 8;
        if remainder > 0 {
            let start = chunks * 8;
            let scalar_dot: f32 = a[start..].iter()
                .zip(b[start..].iter())
                .map(|(x, y)| x * y)
                .sum();
            let scalar_norm_a: f32 = a[start..].iter().map(|x| x * x).sum::<f32>().sqrt();
            let scalar_norm_b: f32 = b[start..].iter().map(|x| x * x).sum::<f32>().sqrt();

            (dot + scalar_dot) / ((norm_a + scalar_norm_a) * (norm_b + scalar_norm_b))
        } else {
            dot / (norm_a * norm_b)
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn euclidean_distance_impl(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }
        
        let mut result = horizontal_sum_ps256(sum);
        
        // Handle remainder
        let remainder = a.len() % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..a.len() {
                let diff = a[i] - b[i];
                result += diff * diff;
            }
        }
        
        result.sqrt()
    }
    
    #[target_feature(enable = "avx")]
    unsafe fn dot_product_impl(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
        }
        
        let mut result = horizontal_sum_ps256(sum);
        
        // Handle remainder
        let remainder = a.len() % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..a.len() {
                result += a[i] * b[i];
            }
        }
        
        result
    }
    
    #[target_feature(enable = "avx")]
    unsafe fn vector_add_impl(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let sum = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i * 8), sum);
        }
        
        // Handle remainder
        let remainder = a.len() % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..a.len() {
                result[i] = a[i] + b[i];
            }
        }
    }
    
    #[target_feature(enable = "avx")]
    unsafe fn vector_sub_impl(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(a_vec, b_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i * 8), diff);
        }
        
        // Handle remainder
        let remainder = a.len() % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..a.len() {
                result[i] = a[i] - b[i];
            }
        }
    }
    
    #[target_feature(enable = "avx")]
    unsafe fn vector_scale_impl(&self, a: &[f32], scalar: f32, result: &mut [f32]) {
        let scalar_vec = _mm256_set1_ps(scalar);
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let scaled = _mm256_mul_ps(a_vec, scalar_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i * 8), scaled);
        }
        
        // Handle remainder
        let remainder = a.len() % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..a.len() {
                result[i] = a[i] * scalar;
            }
        }
    }
}

/// AVX2 implementation
struct Avx2Ops;

impl VectorOps for Avx2Ops {
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        // AVX2 adds FMA instructions
        unsafe { self.cosine_similarity_fma(a, b) }
    }
    
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe { self.euclidean_distance_fma(a, b) }
    }
    
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe { self.dot_product_fma(a, b) }
    }
    
    fn vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        AvxOps.vector_add(a, b, result)
    }
    
    fn vector_sub(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        AvxOps.vector_sub(a, b, result)
    }
    
    fn vector_scale(&self, a: &[f32], scalar: f32, result: &mut [f32]) {
        AvxOps.vector_scale(a, scalar, result)
    }
    
    fn vector_normalize(&self, a: &[f32], result: &mut [f32]) {
        AvxOps.vector_normalize(a, result)
    }
}

impl Avx2Ops {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn cosine_similarity_fma(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();
        
        let chunks = a.len() / 8;
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            
            dot_sum = _mm256_fmadd_ps(a_vec, b_vec, dot_sum);
            norm_a_sum = _mm256_fmadd_ps(a_vec, a_vec, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(b_vec, b_vec, norm_b_sum);
        }
        
        let dot = horizontal_sum_ps256(dot_sum);
        let norm_a = horizontal_sum_ps256(norm_a_sum).sqrt();
        let norm_b = horizontal_sum_ps256(norm_b_sum).sqrt();
        
        dot / (norm_a * norm_b)
    }
    
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn euclidean_distance_fma(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        
        horizontal_sum_ps256(sum).sqrt()
    }
    
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_product_fma(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }
        
        horizontal_sum_ps256(sum)
    }
}

/// AVX-512 implementation
struct Avx512Ops;

impl VectorOps for Avx512Ops {
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe { self.cosine_similarity_avx512(a, b) }
    }
    
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe { self.euclidean_distance_avx512(a, b) }
    }
    
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe { self.dot_product_avx512(a, b) }
    }
    
    fn vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe { self.vector_add_avx512(a, b, result) }
    }
    
    fn vector_sub(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe { self.vector_sub_avx512(a, b, result) }
    }
    
    fn vector_scale(&self, a: &[f32], scalar: f32, result: &mut [f32]) {
        unsafe { self.vector_scale_avx512(a, scalar, result) }
    }
    
    fn vector_normalize(&self, a: &[f32], result: &mut [f32]) {
        let norm = self.dot_product(a, a).sqrt();
        if norm > 0.0 {
            self.vector_scale(a, 1.0 / norm, result);
        }
    }
}

impl Avx512Ops {
    #[target_feature(enable = "avx512f")]
    unsafe fn cosine_similarity_avx512(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot_sum = _mm512_setzero_ps();
        let mut norm_a_sum = _mm512_setzero_ps();
        let mut norm_b_sum = _mm512_setzero_ps();
        
        let chunks = a.len() / 16;
        for i in 0..chunks {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            
            dot_sum = _mm512_fmadd_ps(a_vec, b_vec, dot_sum);
            norm_a_sum = _mm512_fmadd_ps(a_vec, a_vec, norm_a_sum);
            norm_b_sum = _mm512_fmadd_ps(b_vec, b_vec, norm_b_sum);
        }
        
        let dot = _mm512_reduce_add_ps(dot_sum);
        let norm_a = _mm512_reduce_add_ps(norm_a_sum).sqrt();
        let norm_b = _mm512_reduce_add_ps(norm_b_sum).sqrt();
        
        dot / (norm_a * norm_b)
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn euclidean_distance_avx512(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm512_setzero_ps();
        let chunks = a.len() / 16;
        
        for i in 0..chunks {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            let diff = _mm512_sub_ps(a_vec, b_vec);
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }
        
        _mm512_reduce_add_ps(sum).sqrt()
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_product_avx512(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm512_setzero_ps();
        let chunks = a.len() / 16;
        
        for i in 0..chunks {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
        }
        
        _mm512_reduce_add_ps(sum)
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn vector_add_avx512(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let chunks = a.len() / 16;
        
        for i in 0..chunks {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            let sum = _mm512_add_ps(a_vec, b_vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i * 16), sum);
        }
        
        // Handle remainder
        let remainder = a.len() % 16;
        if remainder > 0 {
            let start = chunks * 16;
            for i in start..a.len() {
                result[i] = a[i] + b[i];
            }
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn vector_sub_avx512(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let chunks = a.len() / 16;
        
        for i in 0..chunks {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            let diff = _mm512_sub_ps(a_vec, b_vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i * 16), diff);
        }
        
        // Handle remainder
        let remainder = a.len() % 16;
        if remainder > 0 {
            let start = chunks * 16;
            for i in start..a.len() {
                result[i] = a[i] - b[i];
            }
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn vector_scale_avx512(&self, a: &[f32], scalar: f32, result: &mut [f32]) {
        let scalar_vec = _mm512_set1_ps(scalar);
        let chunks = a.len() / 16;
        
        for i in 0..chunks {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let scaled = _mm512_mul_ps(a_vec, scalar_vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i * 16), scaled);
        }
        
        // Handle remainder
        let remainder = a.len() % 16;
        if remainder > 0 {
            let start = chunks * 16;
            for i in start..a.len() {
                result[i] = a[i] * scalar;
            }
        }
    }
}

// Helper function for horizontal sum of AVX register
#[target_feature(enable = "avx")]
unsafe fn horizontal_sum_ps256(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x1);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_detection() {
        let level = SimdLevel::detect();
        println!("Detected SIMD level: {:?}", level);
        println!("Vector width: {}", level.vector_width());
    }
    
    #[test]
    fn test_simd_operations() {
        let accelerator = SimdAccelerator::new(SimdLevel::detect());
        
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        
        let dot = accelerator.dot_product(&a, &b);
        println!("Dot product: {}", dot);
        
        let cosine = accelerator.cosine_similarity(&a, &b);
        println!("Cosine similarity: {}", cosine);
        
        let euclidean = accelerator.euclidean_distance(&a, &b);
        println!("Euclidean distance: {}", euclidean);
        
        let mut result = vec![0.0; 8];
        accelerator.vector_add(&a, &b, &mut result);
        println!("Vector add: {:?}", result);
    }
}