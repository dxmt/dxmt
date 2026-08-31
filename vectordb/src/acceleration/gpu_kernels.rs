//! Embedded PTX kernels for GPU-accelerated distance calculations.
//!
//! These kernels compute batch distance matrices between query vectors and database vectors.
//! Layout: queries (Q x D), vectors (N x D), output (Q x N).
//! Each thread computes one (query, vector) distance pair.

/// PTX source for distance calculation kernels.
///
/// Contains three kernels:
/// - `cosine_distance_batch`: cosine distance = 1.0 - dot(q,v) / (norm(q) * norm(v))
/// - `euclidean_distance_batch`: euclidean distance = sqrt(sum((q_i - v_i)^2))
/// - `dot_product_batch`: negative dot product = -dot(q,v) (for distance semantics)
///
/// Parameters for all kernels:
/// - queries: pointer to Q*D floats (row-major)
/// - vectors: pointer to N*D floats (row-major)
/// - output: pointer to Q*N floats (row-major)
/// - num_queries: Q
/// - num_vectors: N
/// - dimension: D
#[cfg(feature = "gpu")]
pub const DISTANCE_KERNELS_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// cosine_distance_batch(queries, vectors, output, num_queries, num_vectors, dimension)
.visible .entry cosine_distance_batch(
    .param .u64 queries,
    .param .u64 vectors,
    .param .u64 output,
    .param .u32 num_queries,
    .param .u32 num_vectors,
    .param .u32 dimension
)
{
    .reg .u32 %r<20>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<20>;
    .reg .pred %p<5>;

    // Thread indices: blockIdx.x * blockDim.x + threadIdx.x = vector_idx
    //                 blockIdx.y = query_idx
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;   // r3 = vector_idx
    mov.u32 %r4, %ctaid.y;             // r4 = query_idx

    // Load params
    ld.param.u64 %rd0, [queries];
    ld.param.u64 %rd1, [vectors];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r5, [num_queries];
    ld.param.u32 %r6, [num_vectors];
    ld.param.u32 %r7, [dimension];

    // Bounds check
    setp.ge.u32 %p0, %r3, %r6;
    setp.ge.u32 %p1, %r4, %r5;
    or.pred %p2, %p0, %p1;
    @%p2 bra COSINE_EXIT;

    // Compute base pointers
    // query_base = queries + query_idx * dimension
    mul.wide.u32 %rd3, %r4, %r7;
    shl.b64 %rd3, %rd3, 2;           // * sizeof(f32)
    add.u64 %rd4, %rd0, %rd3;        // rd4 = &queries[query_idx * dim]

    // vector_base = vectors + vector_idx * dimension
    mul.wide.u32 %rd5, %r3, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.u64 %rd6, %rd1, %rd5;        // rd6 = &vectors[vector_idx * dim]

    // Accumulate dot product, query norm^2, vector norm^2
    mov.f32 %f0, 0f00000000;          // dot = 0
    mov.f32 %f1, 0f00000000;          // q_norm_sq = 0
    mov.f32 %f2, 0f00000000;          // v_norm_sq = 0
    mov.u32 %r8, 0;                   // i = 0

COSINE_LOOP:
    setp.ge.u32 %p3, %r8, %r7;
    @%p3 bra COSINE_DONE;

    // Load q[i] and v[i]
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd8, %rd4, %rd7;
    add.u64 %rd9, %rd6, %rd7;
    ld.global.f32 %f3, [%rd8];       // q_val
    ld.global.f32 %f4, [%rd9];       // v_val

    fma.rn.f32 %f0, %f3, %f4, %f0;  // dot += q * v
    fma.rn.f32 %f1, %f3, %f3, %f1;  // q_norm_sq += q * q
    fma.rn.f32 %f2, %f4, %f4, %f2;  // v_norm_sq += v * v

    add.u32 %r8, %r8, 1;
    bra COSINE_LOOP;

COSINE_DONE:
    // distance = 1.0 - dot / (sqrt(q_norm_sq) * sqrt(v_norm_sq))
    sqrt.rn.f32 %f5, %f1;            // sqrt(q_norm_sq)
    sqrt.rn.f32 %f6, %f2;            // sqrt(v_norm_sq)
    mul.rn.f32 %f7, %f5, %f6;        // norm_product

    // Guard against division by zero
    mov.f32 %f8, 0f3727C5AC;         // epsilon ~= 1e-5
    max.f32 %f7, %f7, %f8;
    div.rn.f32 %f9, %f0, %f7;        // cosine_sim = dot / norm_product

    // Clamp to [-1, 1]
    mov.f32 %f10, 0fBF800000;        // -1.0
    mov.f32 %f11, 0f3F800000;        // 1.0
    max.f32 %f9, %f9, %f10;
    min.f32 %f9, %f9, %f11;

    // distance = 1.0 - cosine_sim
    sub.f32 %f12, %f11, %f9;

    // Store output[query_idx * num_vectors + vector_idx]
    mul.wide.u32 %rd10, %r4, %r6;
    cvt.u64.u32 %rd11, %r3;
    add.u64 %rd10, %rd10, %rd11;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd12, %rd2, %rd10;
    st.global.f32 [%rd12], %f12;

COSINE_EXIT:
    ret;
}

// euclidean_distance_batch(queries, vectors, output, num_queries, num_vectors, dimension)
.visible .entry euclidean_distance_batch(
    .param .u64 queries,
    .param .u64 vectors,
    .param .u64 output,
    .param .u32 num_queries,
    .param .u32 num_vectors,
    .param .u32 dimension
)
{
    .reg .u32 %r<20>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<15>;
    .reg .pred %p<5>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;   // vector_idx
    mov.u32 %r4, %ctaid.y;             // query_idx

    ld.param.u64 %rd0, [queries];
    ld.param.u64 %rd1, [vectors];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r5, [num_queries];
    ld.param.u32 %r6, [num_vectors];
    ld.param.u32 %r7, [dimension];

    setp.ge.u32 %p0, %r3, %r6;
    setp.ge.u32 %p1, %r4, %r5;
    or.pred %p2, %p0, %p1;
    @%p2 bra EUCLID_EXIT;

    mul.wide.u32 %rd3, %r4, %r7;
    shl.b64 %rd3, %rd3, 2;
    add.u64 %rd4, %rd0, %rd3;

    mul.wide.u32 %rd5, %r3, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.u64 %rd6, %rd1, %rd5;

    mov.f32 %f0, 0f00000000;          // sum_sq = 0
    mov.u32 %r8, 0;

EUCLID_LOOP:
    setp.ge.u32 %p3, %r8, %r7;
    @%p3 bra EUCLID_DONE;

    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd8, %rd4, %rd7;
    add.u64 %rd9, %rd6, %rd7;
    ld.global.f32 %f1, [%rd8];
    ld.global.f32 %f2, [%rd9];

    sub.f32 %f3, %f1, %f2;            // diff = q - v
    fma.rn.f32 %f0, %f3, %f3, %f0;    // sum_sq += diff * diff

    add.u32 %r8, %r8, 1;
    bra EUCLID_LOOP;

EUCLID_DONE:
    sqrt.rn.f32 %f4, %f0;             // sqrt(sum_sq)

    mul.wide.u32 %rd10, %r4, %r6;
    cvt.u64.u32 %rd11, %r3;
    add.u64 %rd10, %rd10, %rd11;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd12, %rd2, %rd10;
    st.global.f32 [%rd12], %f4;

EUCLID_EXIT:
    ret;
}

// dot_product_batch(queries, vectors, output, num_queries, num_vectors, dimension)
// Output: negative dot product (distance semantics: smaller = more similar)
.visible .entry dot_product_batch(
    .param .u64 queries,
    .param .u64 vectors,
    .param .u64 output,
    .param .u32 num_queries,
    .param .u32 num_vectors,
    .param .u32 dimension
)
{
    .reg .u32 %r<20>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<15>;
    .reg .pred %p<5>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;   // vector_idx
    mov.u32 %r4, %ctaid.y;             // query_idx

    ld.param.u64 %rd0, [queries];
    ld.param.u64 %rd1, [vectors];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r5, [num_queries];
    ld.param.u32 %r6, [num_vectors];
    ld.param.u32 %r7, [dimension];

    setp.ge.u32 %p0, %r3, %r6;
    setp.ge.u32 %p1, %r4, %r5;
    or.pred %p2, %p0, %p1;
    @%p2 bra DOT_EXIT;

    mul.wide.u32 %rd3, %r4, %r7;
    shl.b64 %rd3, %rd3, 2;
    add.u64 %rd4, %rd0, %rd3;

    mul.wide.u32 %rd5, %r3, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.u64 %rd6, %rd1, %rd5;

    mov.f32 %f0, 0f00000000;          // dot = 0
    mov.u32 %r8, 0;

DOT_LOOP:
    setp.ge.u32 %p3, %r8, %r7;
    @%p3 bra DOT_DONE;

    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd8, %rd4, %rd7;
    add.u64 %rd9, %rd6, %rd7;
    ld.global.f32 %f1, [%rd8];
    ld.global.f32 %f2, [%rd9];

    fma.rn.f32 %f0, %f1, %f2, %f0;   // dot += q * v

    add.u32 %r8, %r8, 1;
    bra DOT_LOOP;

DOT_DONE:
    // Negate for distance semantics
    neg.f32 %f3, %f0;

    mul.wide.u32 %rd10, %r4, %r6;
    cvt.u64.u32 %rd11, %r3;
    add.u64 %rd10, %rd10, %rd11;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd12, %rd2, %rd10;
    st.global.f32 [%rd12], %f3;

DOT_EXIT:
    ret;
}
"#;

/// Kernel function names
#[cfg(feature = "gpu")]
pub const COSINE_KERNEL: &str = "cosine_distance_batch";
#[cfg(feature = "gpu")]
pub const EUCLIDEAN_KERNEL: &str = "euclidean_distance_batch";
#[cfg(feature = "gpu")]
pub const DOT_PRODUCT_KERNEL: &str = "dot_product_batch";

/// PTX module name used when loading into CUDA
#[cfg(feature = "gpu")]
pub const MODULE_NAME: &str = "dxmt_distance_kernels";
