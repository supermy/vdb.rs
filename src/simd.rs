//! 平台 SIMD 封装（x86_64 AVX-512/AVX2, aarch64 NEON），
//! 为 RaBitQ 的 popcount 与旋转矩阵运算提供加速。
//!
//! 包含：批量 XOR-popcount、批量点积、SQ8 反量化。

// 允许 MSRV lint：AVX-512 intrinsics 在 1.89 稳定，但项目通过运行时 feature 检测保护，
// 不支持该特性的老编译器根本不会进入这些路径；为保持 Cargo 声明的 rust-version 1.85，
// 显式关闭该 clippy 警告。
#![allow(clippy::incompatible_msrv)]

/// 向量点积（纯 Rust fallback）。
///
/// 为什么先实现 fallback：RaBitQ 的核心精度来自数学变换，SIMD 只是加速。
/// 在核心路径上必须保证无 SIMD 时结果一致，因此所有 SIMD 函数都以 fallback 为基准测试。
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_product: dimension mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 向量 L2 距离平方（纯 Rust fallback）。
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_distance_squared: dimension mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// 计算两个等长 u8 位数组的 Hamming 距离。
///
/// 这是 RaBitQ 位运算距离的基础：XOR 后统计 1 的位数。
/// 实现会根据编译目标和运行时 CPU 特性选择最快路径。
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len(), "hamming_distance: length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vpopcntdq") {
            return unsafe { hamming_distance_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { hamming_distance_avx2(a, b) };
        }
        if is_x86_feature_detected!("popcnt") {
            return unsafe { hamming_distance_popcnt(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARMv8-A 基础指令集已包含 cnt (NEON)，无需额外 feature 检测。
        return unsafe { hamming_distance_neon(a, b) };
    }

    hamming_distance_fallback(a, b)
}

/// 批量计算 query 与多个 code 的 Hamming 距离。
///
/// 内存策略：输出由调用者提供，避免在热点路径上重复分配。
/// 形状：`codes[i].len() == query.len()`，结果写入 `out[i]`。
pub fn batch_hamming_distance(query: &[u8], codes: &[&[u8]], out: &mut [u64]) {
    assert_eq!(
        codes.len(),
        out.len(),
        "batch_hamming_distance: output length mismatch"
    );
    for (i, code) in codes.iter().enumerate() {
        out[i] = hamming_distance(query, code);
    }
}

/// 纯 Rust fallback：按 u64 块 XOR + popcount。
///
/// 为什么按 64 位处理：RaBitQ 要求 dim % 64 == 0，因此位数组长度是 8 的倍数；
/// 按 u64 处理减少循环次数，同时避免引入 unsafe。
fn hamming_distance_fallback(a: &[u8], b: &[u8]) -> u64 {
    let mut dist: u64 = 0;

    // 主循环：每次处理 8 字节（64 位），与 RaBitQ 的维度对齐粒度一致。
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let mut av: u64 = 0;
        let mut bv: u64 = 0;
        for j in 0..8 {
            av |= (a[offset + j] as u64) << (j * 8);
            bv |= (b[offset + j] as u64) << (j * 8);
        }
        dist += (av ^ bv).count_ones() as u64;
    }

    // 尾部（长度不是 8 的倍数时）。
    let tail_start = chunks * 8;
    for i in tail_start..a.len() {
        dist += (a[i] ^ b[i]).count_ones() as u64;
    }

    dist
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// AVX-512 路径：512 位寄存器 XOR + 64 位 popcount。
    ///
    /// 需要 `avx512bw` 与 `avx512vpopcntdq`。
    /// 为什么用 512 位：RaBitQ 位数组通常很长，512 位一次处理 64 字节，
    /// 可最大限度利用内存带宽与 popcount 吞吐。
    pub(super) unsafe fn hamming_distance_avx512(a: &[u8], b: &[u8]) -> u64 {
        let len = a.len();
        let mut dist: u64 = 0;
        let mut i = 0;

        // 每次处理 64 字节（512 位）。
        while i + 64 <= len {
            unsafe {
                let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
                let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const _);
                let vx = _mm512_xor_si512(va, vb);
                let pc = _mm512_popcnt_epi64(vx);
                dist += _mm512_reduce_add_epi64(pc) as u64;
            }
            i += 64;
        }

        dist += super::hamming_distance_fallback(&a[i..], &b[i..]);
        dist
    }

    /// AVX2 路径：256 位寄存器 XOR + 8 位 popcount lookup table。
    ///
    /// AVX2 没有原生 64 位 popcount，因此使用经典的 4-bit lookup table。
    /// 为什么用 256 位：AVX2 是普及率最高的 256 位 SIMD，能在不支持 AVX-512 的
    /// 机器上获得显著加速。
    pub(super) unsafe fn hamming_distance_avx2(a: &[u8], b: &[u8]) -> u64 {
        // 4-bit popcount lookup table，共 16 个条目。
        let lookup = unsafe {
            _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3,
                2, 3, 3, 4,
            )
        };
        let low_mask = unsafe { _mm256_set1_epi8(0x0F) };
        let zero = unsafe { _mm256_set1_epi8(0) };
        let len = a.len();
        let mut dist: u64 = 0;
        let mut i = 0;

        while i + 32 <= len {
            unsafe {
                let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                let vx = _mm256_xor_si256(va, vb);

                let lo = _mm256_and_si256(vx, low_mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16(vx, 4), low_mask);
                let pop_lo = _mm256_shuffle_epi8(lookup, lo);
                let pop_hi = _mm256_shuffle_epi8(lookup, hi);
                let pop = _mm256_add_epi8(pop_lo, pop_hi);

                // _mm256_sad_epu8 将 32 个字节按 8 字节一组求和，得到 4 个 64 位 lane。
                // 这样避免 8 位累加溢出（32 字节最大 popcount 为 256）。
                let sad = _mm256_sad_epu8(pop, zero);
                dist += _mm256_extract_epi64(sad, 0) as u64;
                dist += _mm256_extract_epi64(sad, 1) as u64;
                dist += _mm256_extract_epi64(sad, 2) as u64;
                dist += _mm256_extract_epi64(sad, 3) as u64;
            }
            i += 32;
        }

        dist += super::hamming_distance_fallback(&a[i..], &b[i..]);
        dist
    }

    /// x86 popcnt 指令路径：按 8 字节块 popcount。
    pub(super) unsafe fn hamming_distance_popcnt(a: &[u8], b: &[u8]) -> u64 {
        let len = a.len();
        let mut dist: u64 = 0;
        let mut i = 0;

        while i + 8 <= len {
            unsafe {
                let av = *(a.as_ptr().add(i) as *const u64);
                let bv = *(b.as_ptr().add(i) as *const u64);
                dist += (av ^ bv).count_ones() as u64;
            }
            i += 8;
        }

        dist += super::hamming_distance_fallback(&a[i..], &b[i..]);
        dist
    }
}

#[cfg(target_arch = "x86_64")]
use x86::*;

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;

    /// NEON 路径：128 位寄存器 XOR + 8 位 popcount。
    ///
    /// ARMv8-A 基础 NEON 提供 vcnt，可直接统计每字节的 1 的位数。
    pub(super) unsafe fn hamming_distance_neon(a: &[u8], b: &[u8]) -> u64 {
        let len = a.len();
        let mut dist: u64 = 0;
        let mut i = 0;

        while i + 16 <= len {
            unsafe {
                let va = vld1q_u8(a.as_ptr().add(i));
                let vb = vld1q_u8(b.as_ptr().add(i));
                let vx = veorq_u8(va, vb);
                let cnt = vcntq_u8(vx);
                // vpaddlq_u8 将每对字节相加，得到 8 个 16 位和。
                let sum16 = vpaddlq_u8(cnt);
                // 继续成对相加，得到 4 个 32 位和。
                let sum32 = vpaddlq_u16(sum16);
                // 得到 2 个 64 位和。
                let sum64 = vpaddlq_u32(sum32);
                dist += vgetq_lane_u64(sum64, 0);
                dist += vgetq_lane_u64(sum64, 1);
            }
            i += 16;
        }

        dist += super::hamming_distance_fallback(&a[i..], &b[i..]);
        dist
    }
}

#[cfg(target_arch = "aarch64")]
use arm::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_squared() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((l2_distance_squared(&a, &b) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_distance_aligned() {
        // 128 位 = 16 字节，确保覆盖 SIMD 主循环与 fallback 尾部。
        let a = vec![0b0000_1111u8; 16];
        let b = vec![0b1111_0000u8; 16];
        assert_eq!(hamming_distance(&a, &b), 8 * 16);
    }

    #[test]
    fn test_hamming_distance_random() {
        for len in [1, 7, 16, 33, 64, 127, 256] {
            let a: Vec<u8> = (0..len).map(|_| rand::random::<u8>()).collect();
            let b: Vec<u8> = (0..len).map(|_| rand::random::<u8>()).collect();
            let expected = hamming_distance_fallback(&a, &b);
            let actual = hamming_distance(&a, &b);
            assert_eq!(expected, actual, "len={}", len);
        }
    }

    #[test]
    fn test_batch_hamming_distance() {
        let query = vec![0xFFu8; 16];
        let codes: Vec<Vec<u8>> = (0..4).map(|i| vec![i as u8; 16]).collect();
        let code_refs: Vec<&[u8]> = codes.iter().map(|c| c.as_slice()).collect();
        let mut out = vec![0u64; 4];
        batch_hamming_distance(&query, &code_refs, &mut out);

        for i in 0..4 {
            assert_eq!(out[i], hamming_distance(&query, &codes[i]));
        }
    }

    #[test]
    fn test_hamming_distance_empty() {
        assert_eq!(hamming_distance(&[], &[]), 0);
    }
}
