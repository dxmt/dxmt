// Safety validation and checking utilities
// Provides runtime and debug-time checks for unsafe operations

use std::slice;

/// Safety check configuration
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    /// Enable bounds checking in release mode
    pub release_bounds_check: bool,
    /// Maximum allowed buffer size (prevent OOM)
    pub max_buffer_size: usize,
    /// Enable alignment verification
    pub verify_alignment: bool,
    /// Enable null pointer checks
    pub check_null_pointers: bool,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            release_bounds_check: cfg!(debug_assertions),
            max_buffer_size: 1 << 30, // 1GB default max
            verify_alignment: true,
            check_null_pointers: true,
        }
    }
}

/// Safety validator for pointer operations
pub struct PointerValidator;

impl PointerValidator {
    /// Validate a pointer is not null
    #[inline]
    pub fn check_not_null<T>(ptr: *const T, context: &str) -> crate::Result<()> {
        if ptr.is_null() {
            return Err(crate::error::Error::Internal(format!("Null pointer detected in {}", context)));
        }
        Ok(())
    }

    /// Validate pointer alignment
    #[inline]
    pub fn check_alignment<T>(ptr: *const T, context: &str) -> crate::Result<()> {
        let alignment = std::mem::align_of::<T>();
        if !(ptr as usize).is_multiple_of(alignment) {
            return Err(crate::error::Error::Internal(format!(
                "Misaligned pointer in {}: address {:p} not aligned to {}",
                context, ptr, alignment)));
        }
        Ok(())
    }

    /// Validate pointer range for slice access
    #[inline]
    pub fn check_slice_bounds<T>(
        ptr: *const T,
        len: usize,
        buffer_size: usize,
        context: &str) -> crate::Result<()> {
        // Check for null
        Self::check_not_null(ptr, context)?;

        // Check for overflow
        let element_size = std::mem::size_of::<T>();
        let total_size = len
            .checked_mul(element_size)
            .ok_or_else(|| crate::error::Error::Internal(format!("Size overflow in {}: {} * {}", context, len, element_size)))?;

        // Check bounds
        if total_size > buffer_size {
            return Err(crate::error::Error::Internal(format!(
                "Buffer overflow in {}: {} bytes requested, {} available",
                context, total_size, buffer_size)));
        }

        Ok(())
    }

    /// Create a validated slice from raw pointer
    ///
    /// # Safety
    /// The caller must ensure: /// - The pointer is valid for reads of `len * _size_of::<T>()` bytes
    /// - The pointer remains valid for the lifetime 'a
    /// - The memory is properly initialized
    #[inline]
    pub unsafe fn create_slice_checked<'a, T>(
        ptr: *const T,
        len: usize,
        context: &str) -> crate::Result<&'a [T]> {
        Self::check_not_null(ptr, context)?;
        Self::check_alignment(ptr, context)?;

        // Additional check for reasonable size
        const MAX_SLICE_ELEMENTS: usize = 1 << 28; // 256M elements max
        if len > MAX_SLICE_ELEMENTS {
            return Err(crate::error::Error::Internal(format!(
                "Slice too large in {}: {} elements exceeds limit {}",
                context, len, MAX_SLICE_ELEMENTS)));
        }

        Ok(slice::from_raw_parts(ptr, len))
    }
}

/// SIMD operation safety validator
pub struct SimdValidator;

impl SimdValidator {
    /// Validate vectors for SIMD operations
    #[inline]
    pub fn validate_vectors(a: &[f32], b: &[f32], context: &str) -> crate::Result<()> {
        // Check equal length
        if a.len() != b.len() {
            return Err(crate::error::Error::Internal(format!(
                "Vector dimension mismatch in {}: a={}, b={}",
                context,
                a.len(),
                b.len()
            )));
        }

        // Check non-empty
        if a.is_empty() {
            return Err(crate::error::Error::Internal(format!("Empty vectors in {}", context)));
        }

        // Check for NaN/Inf values that could cause issues
        #[cfg(debug_assertions)]
        {
            for (i, &val) in a.iter().enumerate() {
                if !val.is_finite() {
                    return Err(crate::error::Error::Internal(format!(
                        "Non-finite value in vector a[{}]: {} in {}",
                        i, val, context)));
                }
            }
            for (i, &val) in b.iter().enumerate() {
                if !val.is_finite() {
                    return Err(crate::error::Error::Internal(format!(
                        "Non-finite value in vector b[{}]: {} in {}",
                        i, val, context)));
                }
            }
        }

        Ok(())
    }

    /// Check SIMD alignment requirements
    #[inline]
    pub fn check_simd_alignment(
        ptr: *const f32,
        alignment: usize,
        context: &str) -> crate::Result<()> {
        if !(ptr as usize).is_multiple_of(alignment) {
            return Err(crate::error::Error::Internal(format!(
                "SIMD misaligned pointer in {}: address {:p} not aligned to {}",
                context, ptr, alignment)));
        }
        Ok(())
    }
}

/// Memory operation safety validator
pub struct MemoryValidator;

impl MemoryValidator {
    /// Validate memory copy operation
    #[inline]
    pub fn validate_copy(
        src: *const u8,
        dst: *mut u8,
        len: usize,
        context: &str) -> crate::Result<()> {
        // Check for null pointers
        PointerValidator::check_not_null(src, context)?;
        PointerValidator::check_not_null(dst, context)?;

        // Check for overlap (for non-overlapping copy)
        let src_range = src as usize..((src as usize) + len);
        let dst_range = dst as usize..((dst as usize) + len);

        if src_range.start < dst_range.end && dst_range.start < src_range.end {
            return Err(crate::error::Error::Internal(format!(
                "Memory ranges overlap in {}: src[{}..{}] overlaps dst[{}..{}]",
                context,
                src_range.start,
                src_range.end,
                dst_range.start,
                dst_range.end
            )));
        }

        Ok(())
    }

    /// Validate allocation size
    #[inline]
    pub fn validate_allocation_size(
        size: usize,
        max_size: usize,
        context: &str) -> crate::Result<()> {
        if size == 0 {
            return Err(crate::error::Error::Internal(format!("Zero-size allocation requested in {}", context)));
        }

        if size > max_size {
            return Err(crate::error::Error::Internal(format!(
                "Allocation too large in {}: {} bytes exceeds limit {}",
                context, size, max_size)));
        }

        Ok(())
    }
}

/// Macro for debug-only safety assertions
#[macro_export]
macro_rules! debug_assert_safety {
    ($cond:expr, $msg:expr) => {
        #[cfg(debug_assertions)]
        {
            if !$cond {
                panic!("Safety assertion failed: {}", $msg)
            }
        }
    };
}

/// Macro for bounds checking
#[macro_export]
macro_rules! check_bounds {
    ($index:expr, $len:expr, $context:expr) => {
        if $index >= $len {
            return Err($crate::error::Error::Internal(format!(
                "Index out of bounds in {}: {} >= {}",
                $context, $index, $len
            )));
        }
    };
}

/// Macro for safe unsafe block with documentation
#[macro_export]
macro_rules! unsafe_block {
    (
        // Safety requirements documentation
        safety: $safety:expr,
        // The unsafe code block
        $($code:tt)*
    ) => {
        unsafe {
            // Safety: $safety
            $($code)*
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointer_validation() {
        let data = [1.0f32, 2.0, 3.0];
        let ptr = data.as_ptr();

        // Should pass
        assert!(PointerValidator::check_not_null(ptr, "test").is_ok());
        assert!(PointerValidator::check_alignment(ptr, "test").is_ok());

        // Null pointer should fail
        let null_ptr: *const f32 = std::ptr::null();
        assert!(PointerValidator::check_not_null(null_ptr, "test").is_err())
    }

    #[test]
    fn test_simd_validation() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // Should pass
        assert!(SimdValidator::validate_vectors(&a, &b, "test").is_ok());

        // Different lengths should fail
        let c = vec![7.0, 8.0];
        assert!(SimdValidator::validate_vectors(&a, &c, "test").is_err());

        // Empty vectors should fail
        let empty: Vec<f32> = vec![];
        assert!(SimdValidator::validate_vectors(&empty, &empty, "test").is_err())
    }

    #[test]
    fn test_memory_validation() {
        let src = [1u8, 2, 3, 4];
        let mut dst = vec![0u8; 4];

        // Non-overlapping should pass
        assert!(MemoryValidator::validate_copy(src.as_ptr(), dst.as_mut_ptr(), 4, "test").is_ok());

        // Overlapping should fail
        assert!(
            MemoryValidator::validate_copy(src.as_ptr(), src.as_ptr() as *mut u8, 4, "test")
                .is_err()
        )
    }
}
