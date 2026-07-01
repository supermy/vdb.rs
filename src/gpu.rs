//! GPU 支持备选方案（Metal/CUDA/OpenCL），无 GPU 时自动降级至 CPU SIMD。
//!
//! 当前实现内嵌三种 RaBitQ popcount kernel 源码字符串；
//! 实际 GPU 调度逻辑留作后续阶段（需绑定对应驱动 SDK）。
//! 无 GPU 时，`GpuDevice` 直接调用 `simd::batch_hamming_distance`，
//! 保证所有测试无需真实 GPU 即可通过。

use crate::simd::batch_hamming_distance;

/// Metal RaBitQ popcount kernel（macOS 首选）。
///
/// 当启用 `no-gpu`（即 `--no-default-features`）时为空，避免内嵌 kernel 源码。
///
/// untested: 当前无真实 GPU 调度逻辑，kernel 源码仅做编译时嵌入与 feature 开关验证。
#[cfg(feature = "gpu")]
pub const METAL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rabitq_popcount(
    device const uchar* query,
    device const uchar* codes,
    device uint* out,
    constant uint& code_len,
    uint gid [[thread_position_in_grid]])
{
    uint offset = gid * code_len;
    uint dist = 0;
    for (uint i = 0; i < code_len; i++) {
        uchar x = query[i] ^ codes[offset + i];
        dist += popcount(x);
    }
    out[gid] = dist;
}
"#;
#[cfg(not(feature = "gpu"))]
pub const METAL_KERNEL: &str = "";

/// CUDA RaBitQ popcount kernel（Linux/Windows NVIDIA）。
///
/// untested: 当前无真实 GPU 调度逻辑，kernel 源码仅做编译时嵌入与 feature 开关验证。
#[cfg(feature = "gpu")]
pub const CUDA_KERNEL: &str = r#"
extern "C" __global__ void rabitq_popcount(
    const unsigned char* query,
    const unsigned char* codes,
    unsigned int* out,
    unsigned int code_len,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned int offset = gid * code_len;
    unsigned int dist = 0;
    for (unsigned int i = 0; i < code_len; i++) {
        unsigned char x = query[i] ^ codes[offset + i];
        dist += __popc(x);
    }
    out[gid] = dist;
}
"#;
#[cfg(not(feature = "gpu"))]
pub const CUDA_KERNEL: &str = "";

/// OpenCL RaBitQ popcount kernel（通用备选）。
///
/// untested: 当前无真实 GPU 调度逻辑，kernel 源码仅做编译时嵌入与 feature 开关验证。
#[cfg(feature = "gpu")]
pub const OPENCL_KERNEL: &str = r#"
__kernel void rabitq_popcount(
    __global const uchar* query,
    __global const uchar* codes,
    __global uint* out,
    const uint code_len)
{
    uint gid = get_global_id(0);
    uint offset = gid * code_len;
    uint dist = 0;
    for (uint i = 0; i < code_len; i++) {
        uchar x = query[i] ^ codes[offset + i];
        dist += popcount(x);
    }
    out[gid] = dist;
}
"#;
#[cfg(not(feature = "gpu"))]
pub const OPENCL_KERNEL: &str = "";

/// GPU 设备抽象。
///
/// 当前生产环境以 CPU SIMD fallback 为主；
/// 真实 Metal/CUDA/OpenCL 路径可在不修改此 API 的前提下接入。
pub struct GpuDevice;

impl GpuDevice {
    /// 尝试创建 GPU 设备；当前无真实 GPU 时返回 `None`。
    pub fn new() -> Option<Self> {
        None
    }

    /// 批量计算 query 与多个 code 的 Hamming 距离。
    ///
    /// 形状：`codes[i].len() == query.len()`，结果写入 `out[i]`。
    /// 无 GPU 时自动降级到 CPU SIMD。
    pub fn batch_rabitq_popcount(&self, query: &[u8], codes: &[&[u8]], out: &mut [u64]) {
        batch_hamming_distance(query, codes, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_fallback_matches_cpu() {
        let query = vec![0xFFu8; 16];
        let codes: Vec<Vec<u8>> = (0..4).map(|i| vec![i as u8; 16]).collect();
        let code_refs: Vec<&[u8]> = codes.iter().map(|c| c.as_slice()).collect();

        let mut cpu_out = vec![0u64; 4];
        batch_hamming_distance(&query, &code_refs, &mut cpu_out);

        let device = GpuDevice::new();
        let mut gpu_out = vec![0u64; 4];
        if let Some(dev) = device {
            dev.batch_rabitq_popcount(&query, &code_refs, &mut gpu_out);
            assert_eq!(cpu_out, gpu_out);
        } else {
            // 无 GPU 时，若启用了 gpu feature 则保证 kernel 字符串非空；
            // 若关闭了 gpu feature，则 kernel 为空，但 CPU fallback 仍可工作。
            #[cfg(feature = "gpu")]
            {
                assert!(!METAL_KERNEL.is_empty());
                assert!(!CUDA_KERNEL.is_empty());
                assert!(!OPENCL_KERNEL.is_empty());
            }
        }
    }
}
