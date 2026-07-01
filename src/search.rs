//! 查询调度、SQL 谓词下推、nprobe 剪枝、分区级并行位运算检索、
//! 精排层（Refine/SQ8）、多路召回归并。

use crate::index_ivf_rq::{IvfRabitqIndex, QueryQuantizedCode, RabitqCode};
use crate::simd::{batch_hamming_distance, dot_product, l2_distance_squared};
use crate::sql::{SqlPredicate, parse_sql_filter};
use crate::thread_pool::ThreadPool;
use std::sync::Mutex;

/// 搜索选项。
///
/// 为什么默认开启 refine：RaBitQ 的位运算距离是估计值，
/// 对 TopK 结果用原始向量重新计算真实 L2，可在极低额外开销下把召回率拉到生产水平。
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub k: usize,
    /// nprobe = 0 表示扫描所有分区。
    pub nprobe: usize,
    pub refine: bool,
    pub refine_k: usize,
    pub fastscan: bool,
    /// Query Quantization 位数，0 表示禁用。
    pub query_bits: u8,
    /// 使用 SQ8 码进行精排（在没有原始向量或希望减少带宽时）。
    pub sq8_refine: bool,
    pub sql_filter: Option<String>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            k: 10,
            nprobe: 0,
            refine: true,
            refine_k: 100,
            fastscan: true,
            query_bits: 0,
            sq8_refine: false,
            sql_filter: None,
        }
    }
}

/// 对指定索引执行一次搜索。
///
/// 执行顺序：
/// 1. 解析 SQL 谓词（如有）。
/// 2. 量化查询向量。
/// 3. 按质心距离选 nprobe 个分区。
/// 4. 在候选分区上并行/顺序扫描 RaBitQ 码，计算估计距离。
/// 5. 对粗排 TopK 做 refine（如有原始向量）。
/// 6. 返回最终 TopK。
pub fn search(
    index: &IvfRabitqIndex,
    query: &[f32],
    options: &SearchOptions,
    pool: Option<&ThreadPool>,
) -> Vec<(u64, f32)> {
    assert_eq!(
        query.len(),
        index.config().dim,
        "search: dimension mismatch"
    );
    if index.is_empty() || options.k == 0 {
        return Vec::new();
    }

    let nprobe = if options.nprobe == 0 {
        index.num_partitions()
    } else {
        options.nprobe.min(index.num_partitions())
    };

    let predicate = options
        .sql_filter
        .as_ref()
        .and_then(|s| parse_sql_filter(s).ok());

    let query_code = index.encode_query(query);
    let query_quantized = if options.query_bits > 0 {
        Some(
            index
                .quantizer()
                .encode_query_quantized(query, options.query_bits),
        )
    } else {
        None
    };

    // 分区级 SQL 谓词下推：先按统计信息排除不可能包含结果的分区，
    // 避免 IVF 路由把 nprobe 浪费在无关分区上。
    let eligible_pids: Vec<usize> = (0..index.num_partitions())
        .filter(|&pid| {
            predicate
                .as_ref()
                .map(|p| crate::sql::can_partition_match(p, index.partition_stats(pid)))
                .unwrap_or(true)
        })
        .collect();

    // 按查询到质心的 L2 距离排序，取 nprobe 个分区。
    // 使用 R*centroid 预计算：查询向量只旋转一次，然后与各分区 R*centroid 做点积，
    // 将 IVF 路由从 O(dim^2) 每分区降到 O(dim) 每分区。
    let rotated_query = index.quantizer().rotate_vector(query);
    let q_norm_sq: f32 = rotated_query.iter().map(|v| v * v).sum();
    let mut partition_dists: Vec<(usize, f32)> = eligible_pids
        .iter()
        .map(|&i| {
            let rc = &index.rotated_centroids()[i];
            let dot = dot_product(&rotated_query, rc);
            let dist = q_norm_sq + index.rotated_centroid_norms_sq()[i] - 2.0 * dot;
            (i, dist)
        })
        .collect();
    partition_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let selected_pids: Vec<usize> = partition_dists
        .into_iter()
        .take(nprobe)
        .map(|(p, _)| p)
        .collect();

    // 扫描候选分区。
    // 使用 std::thread::scope 实现并行，避免捕获非 'static 引用；
    // ThreadPool 由上层 batchSearch 复用。
    // untested: thread_pool 路径由上层 batchSearch 复用，当前单元测试使用单线程路径。
    let candidates: Vec<(u64, f32)> = if pool.is_some() && selected_pids.len() > 1 {
        let results: Mutex<Vec<(u64, f32)>> = Mutex::new(Vec::new());
        let query_quantized_ref = query_quantized.as_ref();
        std::thread::scope(|s| {
            for chunk in selected_pids.chunks(1) {
                let pid = chunk[0];
                let results = &results;
                let query_code = &query_code;
                let predicate = predicate.as_ref();
                s.spawn(move || {
                    let part = if let Some(qq) = query_quantized_ref {
                        scan_partition_quantized(index, qq, pid, predicate)
                    } else {
                        scan_partition(index, query_code, pid, predicate, options.fastscan)
                    };
                    results.lock().unwrap().extend(part);
                });
            }
        });
        results.into_inner().unwrap()
    } else {
        let mut results = Vec::new();
        for &pid in &selected_pids {
            if let Some(ref qq) = query_quantized {
                results.extend(scan_partition_quantized(index, qq, pid, predicate.as_ref()));
            } else {
                results.extend(scan_partition(
                    index,
                    &query_code,
                    pid,
                    predicate.as_ref(),
                    options.fastscan,
                ));
            }
        }
        results
    };

    if candidates.is_empty() {
        return Vec::new();
    }

    // 精排层：优先用原始向量重算真实 L2；若启用 sq8_refine 且 SQ8 数据新鲜，
    // 则用 SQ8 反量化近似向量重排，节省带宽与计算。
    // 当 nprobe 已覆盖全部分区时，直接精排所有候选，避免估计误差导致真实近邻被截断。
    let use_sq8 = options.sq8_refine && !index.sq8_dirty();
    let use_raw = options.refine && !index.raw_vectors().is_empty() && !use_sq8;

    let mut refined = if options.refine && options.refine_k > 0 {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let refine_n = if nprobe == index.num_partitions() {
            sorted.len()
        } else {
            options.refine_k.min(sorted.len())
        };

        let mut reranked: Vec<(u64, f32)> = if use_raw {
            sorted[..refine_n]
                .iter()
                .map(|(id, _)| {
                    let exact = index
                        .raw_vector(*id)
                        .map(|v| l2_distance_squared(query, v))
                        .unwrap_or(f32::MAX);
                    (*id, exact)
                })
                .collect()
        } else if use_sq8 {
            sorted[..refine_n]
                .iter()
                .map(|(id, _)| {
                    let approx = index.sq8_distance(query, *id).unwrap_or(f32::MAX);
                    (*id, approx)
                })
                .collect()
        } else {
            sorted[..refine_n].to_vec()
        };
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        reranked
    } else {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted
    };

    refined.truncate(options.k);
    refined
}

fn scan_partition(
    index: &IvfRabitqIndex,
    query_code: &RabitqCode,
    pid: usize,
    predicate: Option<&SqlPredicate>,
    fastscan: bool,
) -> Vec<(u64, f32)> {
    let dim = index.config().dim;
    let entries = index.partition_entries(pid);
    let mut results = Vec::with_capacity(entries.len());

    if fastscan {
        // FastScan：批量 XOR-popcount，减少 SIMD 调用次数。
        let batch_size = 64;
        let mut out = vec![0u64; batch_size];
        let mut i = 0;
        while i < entries.len() {
            let end = (i + batch_size).min(entries.len());
            let codes: Vec<&[u8]> = entries[i..end]
                .iter()
                .map(|(_, code)| code.bits.as_slice())
                .collect();
            batch_hamming_distance(&query_code.bits, &codes, &mut out[..end - i]);

            for (j, (id, code)) in entries[i..end].iter().enumerate() {
                if !passes_predicate(index, *id, predicate) {
                    continue;
                }
                let dist = estimate_distance_from_hamming(dim, query_code, code, out[j]);
                results.push((*id, dist));
            }
            i = end;
        }
    } else {
        for (id, code) in entries {
            if !passes_predicate(index, *id, predicate) {
                continue;
            }
            let dist = index.quantizer().estimate_distance_sq(query_code, code);
            results.push((*id, dist));
        }
    }

    results
}

fn passes_predicate(index: &IvfRabitqIndex, id: u64, predicate: Option<&SqlPredicate>) -> bool {
    match predicate {
        Some(pred) => index.payload(id).map(|p| pred.eval(p)).unwrap_or(false),
        None => true,
    }
}

fn scan_partition_quantized(
    index: &IvfRabitqIndex,
    query_code: &QueryQuantizedCode,
    pid: usize,
    predicate: Option<&SqlPredicate>,
) -> Vec<(u64, f32)> {
    let dim = index.config().dim;
    let entries = index.partition_entries(pid);
    let mut results = Vec::with_capacity(entries.len());
    for (id, code) in entries {
        if !passes_predicate(index, *id, predicate) {
            continue;
        }
        let dist = estimate_distance_quantized(dim, query_code, code);
        results.push((*id, dist));
    }
    results
}

fn estimate_distance_quantized(
    dim: usize,
    query_code: &QueryQuantizedCode,
    code: &RabitqCode,
) -> f32 {
    // 计算 sum_{code.bit=1} query.value(d)。
    // 利用 code.bits 按位存储，避免逐维度调用 QueryQuantizedCode::value 的除法/取模。
    let mut sum_pos: f32 = 0.0;
    for d in 0..dim {
        let byte_idx = d / 8;
        let bit = (code.bits[byte_idx] >> (d % 8)) & 1;
        if bit == 1 {
            sum_pos += query_code.value(d) as f32;
        }
    }
    // sum_i q_i * sign_i = 2 * sum_pos - sum_all。
    let sum = 2.0 * sum_pos - query_code.total_sum;
    let estimated_dot = code.beta * query_code.scale * sum / dim as f32;
    query_code.alpha + code.alpha - 2.0 * estimated_dot
}

fn estimate_distance_from_hamming(
    dim: usize,
    query_code: &RabitqCode,
    code: &RabitqCode,
    hamming: u64,
) -> f32 {
    let dim_f = dim as f32;
    let hamming_f = hamming as f32;
    let s_xy = dim_f - 2.0 * hamming_f;
    let estimated_dot = query_code.beta * code.beta * s_xy / dim_f;
    query_code.alpha + code.alpha - 2.0 * estimated_dot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_ivf_rq::{IvfRabitqIndex, Payload};
    use serde_json::json;

    fn gaussian_random() -> f32 {
        let u1 = rand::random::<f32>().max(1e-7);
        let u2 = rand::random::<f32>();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
    }

    #[test]
    fn test_search_recall_with_refine() {
        let dim = 128;
        let n = 1000;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_random()).collect())
            .collect();
        let index = IvfRabitqIndex::build(&vectors);
        let query: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();

        let truth: std::collections::HashSet<u64> = index
            .flat_search(&query, 10)
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        let options = SearchOptions {
            k: 10,
            nprobe: 0,
            refine: true,
            refine_k: 100,
            fastscan: true,
            query_bits: 0,
            sq8_refine: false,
            sql_filter: None,
        };
        let results = search(&index, &query, &options, None);
        let recall = results.iter().filter(|(id, _)| truth.contains(id)).count();
        assert!(recall >= 5, "search recall too low: {}/10", recall);
    }

    #[test]
    fn test_sql_filter_search() {
        let dim = 64;
        let mut index = IvfRabitqIndex::new(dim);
        for i in 0..20 {
            let v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            let mut payload = Payload::new();
            payload.insert("score".to_string(), json!(i as f64));
            index.add_with_payload(&v, payload);
        }
        let query: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let options = SearchOptions {
            k: 10,
            nprobe: 0,
            refine: false,
            refine_k: 0,
            fastscan: false,
            query_bits: 0,
            sq8_refine: false,
            sql_filter: Some("score >= 15".to_string()),
        };
        let results = search(&index, &query, &options, None);
        assert!(!results.is_empty());
        for (id, _) in &results {
            let p = index.payload(*id).unwrap();
            let score = p["score"].as_f64().unwrap();
            assert!(score >= 15.0, "score {} should be >= 15", score);
        }
    }

    #[test]
    fn test_sq8_refine_recall() {
        let dim = 128;
        let n = 1000;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_random()).collect())
            .collect();
        let mut index = IvfRabitqIndex::build(&vectors);
        let query: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();

        let truth: std::collections::HashSet<u64> = index
            .flat_search(&query, 10)
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        let options = SearchOptions {
            k: 10,
            nprobe: 0,
            refine: true,
            refine_k: 100,
            fastscan: true,
            query_bits: 0,
            sq8_refine: true,
            sql_filter: None,
        };
        let results = search(&index, &query, &options, None);
        let recall = results.iter().filter(|(id, _)| truth.contains(id)).count();
        assert!(recall >= 5, "SQ8 refine recall too low: {}/10", recall);

        // 增量插入后 SQ8 变脏；重建后应再次可用。
        index.add(&(0..dim).map(|_| gaussian_random()).collect::<Vec<_>>());
        assert!(index.sq8_dirty());
        index.rebuild_sq8();
        assert!(!index.sq8_dirty());
    }

    #[test]
    fn test_query_quantization_recall() {
        let dim = 128;
        let n = 1000;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_random()).collect())
            .collect();
        let index = IvfRabitqIndex::build(&vectors);
        let query: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();

        let truth: std::collections::HashSet<u64> = index
            .flat_search(&query, 10)
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        for bits in [1u8, 2, 4, 8] {
            let options = SearchOptions {
                k: 10,
                nprobe: 0,
                refine: true,
                refine_k: 100,
                fastscan: false,
                query_bits: bits,
                sq8_refine: false,
                sql_filter: None,
            };
            let results = search(&index, &query, &options, None);
            let recall = results.iter().filter(|(id, _)| truth.contains(id)).count();
            assert!(
                recall >= 5,
                "query quantization {}-bit recall too low: {}/10",
                bits,
                recall
            );
        }
    }
}
