use serde::{Deserialize, Serialize};

/// CPU架构类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuArchitecture {
    AmdMilan, // Zen 3
    AmdTurin, // Zen 5
    AmdGenoa, // Zen 4
    IntelCascadeLake,
    IntelSapphireRapids,
    IntelEmeraldRapids,
    Unknown,
}

/// CPU信息
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub architecture: CpuArchitecture,
    pub vendor: String,
    pub family: u32,
    pub model: u32,
    pub stepping: u32,
    pub core_count: usize,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_fma: bool,
}

// 使用 OnceLock 代替 static mut 以避免 unsafe
use std::sync::OnceLock;
static CPU_INFO: OnceLock<CpuInfo> = OnceLock::new();

/// 获取CPU信息（单例）
pub fn get_cpu_info() -> &'static CpuInfo {
    CPU_INFO.get_or_init(detect_cpu)
}

/// 检测CPU类型
fn detect_cpu() -> CpuInfo {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;

        // 获取CPU厂商
        let vendor = get_cpu_vendor();

        // 获取CPU家族、型号、步进
        let cpuid_result = __cpuid(1);
        let family = ((cpuid_result.eax >> 8) & 0xf) + ((cpuid_result.eax >> 20) & 0xff);
        let model = ((cpuid_result.eax >> 4) & 0xf) + ((cpuid_result.eax >> 12) & 0xf0);
        let stepping = cpuid_result.eax & 0xf;

        // 检测特性
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_avx512 = is_x86_feature_detected!("avx512f");
        let has_fma = is_x86_feature_detected!("fma");

        // 确定架构
        let architecture = match vendor.as_str() {
            "AuthenticAMD" => detect_amd_architecture(family, model),
            "GenuineIntel" => detect_intel_architecture(family, model),
            _ => CpuArchitecture::Unknown,
        };

        // 获取核心数
        let core_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        CpuInfo {
            architecture,
            vendor,
            family,
            model,
            stepping,
            core_count,
            has_avx2,
            has_avx512,
            has_fma,
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        CpuInfo {
            architecture: CpuArchitecture::Unknown,
            vendor: "Unknown".to_string(),
            family: 0,
            model: 0,
            stepping: 0,
            core_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            has_avx2: false,
            has_avx512: false,
            has_fma: false,
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn get_cpu_vendor() -> String {
    use std::arch::x86_64::__cpuid;

    let result = __cpuid(0);
    let mut vendor = [0u8; 12];

    vendor[0..4].copy_from_slice(&result.ebx.to_le_bytes());
    vendor[4..8].copy_from_slice(&result.edx.to_le_bytes());
    vendor[8..12].copy_from_slice(&result.ecx.to_le_bytes());

    String::from_utf8_lossy(&vendor).to_string()
}

fn detect_amd_architecture(family: u32, model: u32) -> CpuArchitecture {
    match family {
        0x19 => {
            match model {
                0x00..=0x0F | 0x20..=0x2F => CpuArchitecture::AmdMilan, // Zen 3
                0x10..=0x1F | 0x60..=0x6F => CpuArchitecture::AmdGenoa, // Zen 4
                0x70..=0x7F | 0xA0..=0xAF => CpuArchitecture::AmdTurin, // Zen 5 (预估)
                _ => CpuArchitecture::Unknown,
            }
        }
        0x1A => CpuArchitecture::AmdTurin, // Zen 5 新家族号
        _ => CpuArchitecture::Unknown,
    }
}

fn detect_intel_architecture(family: u32, model: u32) -> CpuArchitecture {
    match family {
        0x06 => match model {
            0x55 => CpuArchitecture::IntelCascadeLake,
            0x8F => CpuArchitecture::IntelSapphireRapids,
            0xCF => CpuArchitecture::IntelEmeraldRapids,
            _ => CpuArchitecture::Unknown,
        },
        _ => CpuArchitecture::Unknown,
    }
}

/// 根据CPU类型获取优化配置
pub fn get_cpu_optimized_config(_cpuinfo: &CpuInfo) -> CpuOptimizedConfig {
    match CPU_INFO.get().unwrap().architecture {
        CpuArchitecture::AmdMilan => CpuOptimizedConfig {
            vector_batch_size: 128,
            prefetch_distance: 4,
            unroll_factor: 2,
            cache_line_size: 64,
            l1d_cache_size: 32 * 1024,
            l2_cache_size: 512 * 1024,
            l3_cache_per_core: 4 * 1024 * 1024,
            use_avx512: false,
            memory_channels: 8,
            optimal_thread_count: CPU_INFO.get().unwrap().core_count,
        },

        CpuArchitecture::AmdTurin => CpuOptimizedConfig {
            vector_batch_size: 256,
            prefetch_distance: 8,
            unroll_factor: 4,
            cache_line_size: 64,
            l1d_cache_size: 48 * 1024,
            l2_cache_size: 1024 * 1024,
            l3_cache_per_core: 8 * 1024 * 1024,
            use_avx512: true,
            memory_channels: 12,
            optimal_thread_count: CPU_INFO.get().unwrap().core_count,
        },

        CpuArchitecture::AmdGenoa => CpuOptimizedConfig {
            vector_batch_size: 192,
            prefetch_distance: 6,
            unroll_factor: 3,
            cache_line_size: 64,
            l1d_cache_size: 32 * 1024,
            l2_cache_size: 1024 * 1024,
            l3_cache_per_core: 6 * 1024 * 1024,
            use_avx512: false,
            memory_channels: 12,
            optimal_thread_count: CPU_INFO.get().unwrap().core_count,
        },

        _ => CpuOptimizedConfig::default(),
    }
}

/// CPU优化配置
#[derive(Debug, Clone)]
pub struct CpuOptimizedConfig {
    pub vector_batch_size: usize,
    pub prefetch_distance: usize,
    pub unroll_factor: usize,
    pub cache_line_size: usize,
    pub l1d_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_per_core: usize,
    pub use_avx512: bool,
    pub memory_channels: usize,
    pub optimal_thread_count: usize,
}

impl Default for CpuOptimizedConfig {
    fn default() -> Self {
        Self {
            vector_batch_size: 64,
            prefetch_distance: 2,
            unroll_factor: 2,
            cache_line_size: 64,
            l1d_cache_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            l3_cache_per_core: 2 * 1024 * 1024,
            use_avx512: false,
            memory_channels: 2,
            optimal_thread_count: 4,
        }
    }
}

/// 打印CPU信息
pub fn print_cpu_info() {
    let info = get_cpu_info();
    println!("CPU Information: ");
    println!("  Architecture: {:?}", info.architecture);
    println!("  Vendor: {}", info.vendor);
    println!(
        "  Family: {:#x}, Model: {:#x}, Stepping: {}",
        info.family, info.model, info.stepping
    );
    println!("  Core Count: {}", info.core_count);
    println!("  Features: ");
    println!("    AVX2: {}", info.has_avx2);
    println!("    AVX-512: {}", info.has_avx512);
    println!("    FMA: {}", info.has_fma);

    let config = get_cpu_optimized_config(info);
    println!("\nOptimized Configuration: ");
    println!("  Vector Batch Size: {}", config.vector_batch_size);
    println!("  Prefetch Distance: {}", config.prefetch_distance);
    println!(
        "  Cache Sizes: L1D={}KB, L2={}KB, L3={}MB/core",
        config.l1d_cache_size / 1024,
        config.l2_cache_size / 1024,
        config.l3_cache_per_core / 1024 / 1024
    );
}

/// Check if CPU supports AVX2
pub fn has_avx2() -> bool {
    get_cpu_info().has_avx2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detection() {
        let info = get_cpu_info();
        println!("Detected CPU: {:?}", info);

        // 验证基本信息
        assert!(!info.vendor.is_empty());
        assert!(info.core_count > 0);
    }

    #[test]
    fn test_cpu_config() {
        let info = get_cpu_info();
        let config = get_cpu_optimized_config(info);

        println!("CPU Config: {:?}", config);

        // 验证配置合理性
        assert!(config.vector_batch_size > 0);
        assert!(config.cache_line_size == 64); // x86_64标准
    }
}
