//! 混合查询测试：向量 + 全文 + SQL 三路融合的融合正确性（占位阶段）。
//!
//! 当前 fulltext.rs 为占位实现，因此本文件主要验证 RRF/加权融合的数学正确性。

use vdb_rs::hybrid::{reciprocal_rank_fusion, weighted_fusion};

#[test]
fn hybrid_rrf_basic() {
    let list_a = vec![(1, 0.0), (2, 0.0), (3, 0.0)];
    let list_b = vec![(2, 0.0), (3, 0.0), (4, 0.0)];

    let merged = reciprocal_rank_fusion(60, &[list_a, list_b]);
    let ids: Vec<u64> = merged.iter().map(|(id, _)| *id).collect();

    // id=2 和 id=3 在两路中都出现，应排在最前。
    assert!(ids[..2].contains(&2));
    assert!(ids[..2].contains(&3));
    assert!(ids.contains(&1));
    assert!(ids.contains(&4));
}

#[test]
fn hybrid_weighted_fusion_basic() {
    let list_a = vec![(1, 1.0), (2, 0.5)];
    let list_b = vec![(2, 1.0), (3, 0.5)];

    let merged = weighted_fusion(&[(0.5, list_a), (0.5, list_b)]);
    let ids: Vec<u64> = merged.iter().map(|(id, _)| *id).collect();

    // id=2 出现在两路中，融合分最高。
    assert_eq!(ids[0], 2);
}
