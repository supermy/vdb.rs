//! 系统资源探测与检索参数推荐。
//!
//! 用途：为 CLI / Server 提供基于 CPU、内存、数据规模的默认参数建议，
//! 避免在不同机器上重复手动调参。

use std::num::NonZeroUsize;

/// 内存阈值：低于此值按低内存策略限制 mmap 预算（字节）。
pub const MEMORY_THRESHOLD_BYTES: u64 = 16u64 * 1024 * 1024 * 1024;

/// 低内存场景 mmap 缓存占物理内存比例（%）。
pub const SMALL_MEMORY_CACHE_PERCENT: u64 = 50;

/// 大内存场景 mmap 缓存占物理内存比例（%）。
pub const LARGE_MEMORY_CACHE_PERCENT: u64 = 85;

/// IVF 分区数下限。
pub const MIN_PARTITIONS: usize = 4;

/// IVF 分区数上限（RaBitQ 索引内核同步保持此值）。
pub const MAX_PARTITIONS: usize = 65_536;

/// 小数据集阈值（向量数）。
pub const SMALL_DATASET_THRESHOLD: usize = 10_000;

/// 中等数据集阈值（向量数）。
pub const MEDIUM_DATASET_THRESHOLD: usize = 1_000_000;

/// 大规模数据集默认 nprobe 下限。
pub const LARGE_NPROBE_MIN: usize = 100;

/// 大规模数据集默认 nprobe 上限。
pub const LARGE_NPROBE_MAX: usize = 300;

/// 极速模式默认 nprobe。
pub const LATENCY_NPROBE: usize = 16;

/// 获取可用并行度（逻辑 CPU 核心数），至少返回 1。
pub fn available_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(1)
        .max(1)
}

/// 获取物理内存字节数。
///
/// 实现：Unix/macOS 通过 `sysconf(_SC_PHYS_PAGES)` 获取；
/// Windows 路径当前返回 `None`，调用方可使用固定预算。
pub fn physical_memory_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        unsafe {
            let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
            let page_size = libc::sysconf(libc::_SC_PAGE_SIZE);
            if pages > 0 && page_size > 0 {
                return Some((pages as u64) * (page_size as u64));
            }
        }
    }
    None
}

/// 根据物理内存计算 mmap 缓存预算。
///
/// 规则（与 AGENTS.md 一致）：
/// - 内存 < 16 GB：不超过物理内存 50%；
/// - 内存 ≥ 16 GB：不超过物理内存 85%。
pub fn recommend_mmap_cache_bytes(memory_bytes: u64) -> u64 {
    if memory_bytes < MEMORY_THRESHOLD_BYTES {
        (memory_bytes * SMALL_MEMORY_CACHE_PERCENT) / 100
    } else {
        (memory_bytes * LARGE_MEMORY_CACHE_PERCENT) / 100
    }
}

/// 推荐 IVF 分区数。
///
/// 公式：`min(max(MIN_PARTITIONS, sqrt(N)), MAX_PARTITIONS)`，与索引内核保持一致。
pub fn recommend_num_partitions(n: usize) -> usize {
    let sqrt = (n as f64).sqrt() as usize;
    sqrt.clamp(MIN_PARTITIONS, MAX_PARTITIONS)
}

/// 根据系统资源与数据规模推荐搜索参数。
///
/// 规则：
/// - 极速：小 nprobe + query_bits=8，牺牲部分召回换取低延迟；
/// - 平衡：中等 nprobe + 原始向量精排，召回 ≥ 0.95；
/// - 高召回：大 nprobe + 大 refine_k。
pub fn recommend_search_options(n: usize, k: usize) -> RecommendedOptions {
    let partitions = recommend_num_partitions(n);
    let (nprobe, refine_k, query_bits, recall_target) = if n < SMALL_DATASET_THRESHOLD {
        // 数据量较小，直接扫描更多分区，精排候选数不需要太大。
        (partitions.min(50), k * 20, 0, "≥ 0.99")
    } else if n < MEDIUM_DATASET_THRESHOLD {
        // 中等规模：平衡配置。
        (50usize, 1_000usize, 0, "≥ 0.95")
    } else {
        // 大规模：nprobe 取分区数的 1% 左右，但限制在 [LARGE_NPROBE_MIN, LARGE_NPROBE_MAX]；精排候选数放大。
        let nprobe = (partitions / 100).clamp(LARGE_NPROBE_MIN, LARGE_NPROBE_MAX);
        (nprobe, 5_000, 0, "≥ 0.96")
    };

    RecommendedOptions {
        partitions,
        latency: LatencyOptions {
            nprobe: LATENCY_NPROBE,
            refine_k: k * 10,
            query_bits: 8,
            fastscan: true,
            recall_target: "中等",
        },
        balanced: LatencyOptions {
            nprobe,
            refine_k,
            query_bits,
            fastscan: true,
            recall_target,
        },
        recall: LatencyOptions {
            nprobe: (partitions / 10).clamp(LARGE_NPROBE_MIN, LARGE_NPROBE_MAX),
            refine_k: 5_000,
            query_bits: 0,
            fastscan: true,
            recall_target: "≥ 0.99",
        },
        mmap_cache_bytes: physical_memory_bytes().map(recommend_mmap_cache_bytes),
    }
}

#[derive(Debug, Clone)]
pub struct LatencyOptions {
    pub nprobe: usize,
    pub refine_k: usize,
    pub query_bits: u8,
    pub fastscan: bool,
    pub recall_target: &'static str,
}

#[derive(Debug, Clone)]
pub struct RecommendedOptions {
    pub partitions: usize,
    pub latency: LatencyOptions,
    pub balanced: LatencyOptions,
    pub recall: LatencyOptions,
    pub mmap_cache_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommend_num_partitions() {
        assert_eq!(recommend_num_partitions(1), 4);
        assert_eq!(recommend_num_partitions(100), 10);
        assert_eq!(recommend_num_partitions(1_000_000), 1000);
    }

    #[test]
    fn test_recommend_mmap_cache_bytes() {
        let mem_8g = 8u64 * 1024 * 1024 * 1024;
        assert_eq!(recommend_mmap_cache_bytes(mem_8g), mem_8g / 2);

        let mem_32g = 32u64 * 1024 * 1024 * 1024;
        assert_eq!(recommend_mmap_cache_bytes(mem_32g), (mem_32g * 85) / 100);
    }
}
