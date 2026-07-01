//! 系统资源探测与检索参数推荐。
//!
//! 用途：为 CLI / Server 提供基于 CPU、内存、数据规模的默认参数建议，
//! 避免在不同机器上重复手动调参。

use std::num::NonZeroUsize;

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
    let threshold = 16u64 * 1024 * 1024 * 1024;
    if memory_bytes < threshold {
        memory_bytes / 2
    } else {
        (memory_bytes * 85) / 100
    }
}

/// 推荐 IVF 分区数。
///
/// 公式：`min(max(4, sqrt(N)), 65536)`，与索引内核保持一致。
pub fn recommend_num_partitions(n: usize) -> usize {
    let sqrt = (n as f64).sqrt() as usize;
    sqrt.clamp(4, 65536)
}

/// 根据系统资源与数据规模推荐搜索参数。
///
/// 规则：
/// - 极速：小 nprobe + query_bits=8，牺牲部分召回换取低延迟；
/// - 平衡：中等 nprobe + 原始向量精排，召回 ≥ 0.95；
/// - 高召回：大 nprobe + 大 refine_k。
pub fn recommend_search_options(n: usize, k: usize) -> RecommendedOptions {
    let partitions = recommend_num_partitions(n);
    let (nprobe, refine_k, query_bits, recall_target) = if n < 10_000 {
        // 数据量较小，直接扫描更多分区，精排候选数不需要太大。
        (partitions.min(50), k * 20, 0, "≥ 0.99")
    } else if n < 1_000_000 {
        // 中等规模：平衡配置。
        (50usize, 1_000usize, 0, "≥ 0.95")
    } else {
        // 大规模：nprobe 取分区数的 1% 左右，但不超过 300；精排候选数放大。
        let nprobe = (partitions / 100).clamp(100, 300);
        (nprobe, 5_000, 0, "≥ 0.96")
    };

    RecommendedOptions {
        partitions,
        latency: LatencyOptions {
            nprobe: 16,
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
            nprobe: (partitions / 10).clamp(100, 300),
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
