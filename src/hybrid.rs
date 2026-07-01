//! 向量 + 全文 + SQL 三路混合查询的调度、RRF/加权融合重排、结果去重与截断。
//!
//! 当前为骨架阶段：提供 RRF（Reciprocal Rank Fusion）最小实现，
//! 真实三路混合查询待 fulltext.rs 的 Tantivy FFI 完成后替换为完整版本。

use std::collections::HashMap;

/// 对多路召回结果做 Reciprocal Rank Fusion。
///
/// 输入：每一路的 (id, score) 列表，按该路内部排序（越靠前越相关）。
/// 输出：融合后的 (id, 融合分) 列表，按分数降序排列。
///
/// RRF 公式：score = sum(1 / (k + rank))，k 为常数，rank 从 1 开始。
/// 为什么用 RRF：无需对不同路的原始分数做归一化，仅依赖排序位置即可融合。
pub fn reciprocal_rank_fusion(k: usize, lists: &[Vec<(u64, f32)>]) -> Vec<(u64, f32)> {
    let mut scores: HashMap<u64, f32> = HashMap::new();
    for list in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            let r = (rank + 1) as f32;
            *scores.entry(*id).or_insert(0.0) += 1.0 / (k as f32 + r);
        }
    }

    let mut result: Vec<(u64, f32)> = scores.into_iter().collect();
    // 分数高的排前面；分数相同时 id 小的稳定。
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    result
}

/// 加权分数融合。
///
/// 输入：每一路的 (id, score) 列表与对应权重。
/// 输出：融合后的 (id, 融合分) 列表，按分数降序排列。
/// 注意：要求各路 score 已经归一化到可比范围，否则结果无意义。
pub fn weighted_fusion(lists: &[(f32, Vec<(u64, f32)>)]) -> Vec<(u64, f32)> {
    let mut scores: HashMap<u64, f32> = HashMap::new();
    for (weight, list) in lists {
        for (id, score) in list {
            *scores.entry(*id).or_insert(0.0) += weight * score;
        }
    }

    let mut result: Vec<(u64, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    result
}
