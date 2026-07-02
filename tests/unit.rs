//! 单元测试：模块级函数正确性。

use std::collections::HashSet;
use vdb_rs::gpu::GpuDevice;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;
use vdb_rs::search::{SearchOptions, search};
use vdb_rs::simd::{batch_hamming_distance, dot_product, hamming_distance, l2_distance_squared};
use vdb_rs::sql::parse_sql_filter;
use vdb_rs::sys_info::{
    available_parallelism, physical_memory_bytes, recommend_mmap_cache_bytes,
    recommend_num_partitions, recommend_search_options,
};

fn gaussian_random() -> f32 {
    let u1 = rand::random::<f32>().max(1e-7);
    let u2 = rand::random::<f32>();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
}

#[test]
fn simd_basic_operations() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    assert!((l2_distance_squared(&a, &b) - 27.0).abs() < 1e-6);
}

#[test]
fn simd_hamming_distance() {
    let x = vec![0b0000_1111u8; 16];
    let y = vec![0b1111_0000u8; 16];
    assert_eq!(hamming_distance(&x, &y), 8 * 16);
}

#[test]
fn simd_batch_hamming_distance() {
    let query = vec![0xFFu8; 16];
    let codes: Vec<Vec<u8>> = (0..4).map(|i| vec![i as u8; 16]).collect();
    let refs: Vec<&[u8]> = codes.iter().map(|c| c.as_slice()).collect();
    let mut out = vec![0u64; 4];
    batch_hamming_distance(&query, &refs, &mut out);
    for i in 0..4 {
        assert_eq!(out[i], hamming_distance(&query, &codes[i]));
    }
}

#[test]
fn index_build_and_recall() {
    let dim = 128;
    let n = 1000;
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| gaussian_random()).collect())
        .collect();
    let index = IvfRabitqIndex::build(&vectors);
    assert_eq!(index.len(), n);

    let query: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();
    let truth: HashSet<u64> = index
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
    assert!(recall >= 5, "unit search recall too low: {}/10", recall);
}

#[test]
fn sql_parser_basic() {
    let pred = parse_sql_filter("age >= 18 AND name = 'alice'").unwrap();
    let mut payload = serde_json::Map::new();
    payload.insert("age".to_string(), serde_json::json!(20));
    payload.insert("name".to_string(), serde_json::json!("alice"));
    assert!(pred.eval(&payload));

    payload.insert("age".to_string(), serde_json::json!(16));
    assert!(!pred.eval(&payload));
}

#[test]
fn gpu_fallback() {
    if let Some(dev) = GpuDevice::new() {
        let query = vec![0xFFu8; 16];
        let codes: Vec<Vec<u8>> = (0..4).map(|i| vec![i as u8; 16]).collect();
        let refs: Vec<&[u8]> = codes.iter().map(|c| c.as_slice()).collect();
        let mut out = vec![0u64; 4];
        dev.batch_rabitq_popcount(&query, &refs, &mut out);
        let mut expected = vec![0u64; 4];
        batch_hamming_distance(&query, &refs, &mut expected);
        assert_eq!(out, expected);
    }
}

#[test]
fn sys_info_available_parallelism_nonzero() {
    let cpus = available_parallelism();
    assert!(cpus >= 1, "available_parallelism should be at least 1");
}

#[test]
#[cfg(unix)]
fn sys_info_physical_memory_nonzero_on_unix() {
    let mem = physical_memory_bytes().expect("physical_memory_bytes should succeed on unix");
    assert!(mem > 0, "physical memory should be positive");
}

#[test]
fn sys_info_recommend_mmap_cache_bytes_boundary() {
    let mem_16g = 16u64 * 1024 * 1024 * 1024;
    // 恰好在边界上按 16GB 的 85% 计算。
    assert_eq!(recommend_mmap_cache_bytes(mem_16g), (mem_16g * 85) / 100);
}

#[test]
fn sys_info_recommend_num_partitions_clamped() {
    assert_eq!(recommend_num_partitions(0), 4);
    assert_eq!(recommend_num_partitions(3), 4);
    assert_eq!(recommend_num_partitions(100), 10);
    assert_eq!(recommend_num_partitions(65536 * 65536), 65536);
}

#[test]
fn sys_info_recommend_search_options_scales() {
    let small = recommend_search_options(1_000, 10);
    assert_eq!(small.partitions, recommend_num_partitions(1_000));
    assert!(small.balanced.nprobe <= 50);

    let medium = recommend_search_options(100_000, 10);
    assert_eq!(medium.partitions, 316);
    assert_eq!(medium.balanced.nprobe, 50);

    let large = recommend_search_options(10_000_000, 10);
    assert!(large.partitions > 1000);
    assert!(large.balanced.nprobe >= 100 && large.balanced.nprobe <= 300);

    // latency 模式启用 query_bits=8，recall 模式关闭量化。
    assert_eq!(small.latency.query_bits, 8);
    assert_eq!(small.recall.query_bits, 0);
}

#[test]
fn sql_predicate_all_operators() {
    let pred = parse_sql_filter(
        "a = 1 AND b != 2 AND c < 10 AND d <= 10 AND e > 0 AND f >= 5 AND g IN (7, 8, 9)",
    )
    .unwrap();

    let mut payload = serde_json::Map::new();
    payload.insert("a".to_string(), serde_json::json!(1));
    payload.insert("b".to_string(), serde_json::json!(1));
    payload.insert("c".to_string(), serde_json::json!(9));
    payload.insert("d".to_string(), serde_json::json!(10));
    payload.insert("e".to_string(), serde_json::json!(1));
    payload.insert("f".to_string(), serde_json::json!(5));
    payload.insert("g".to_string(), serde_json::json!(8));
    assert!(pred.eval(&payload));

    // 逐一破坏条件，验证每个操作符都能正确返回 false。
    let mut p = payload.clone();
    p.insert("a".to_string(), serde_json::json!(2));
    assert!(!pred.eval(&p));

    let mut p = payload.clone();
    p.insert("b".to_string(), serde_json::json!(2));
    assert!(!pred.eval(&p));

    let mut p = payload.clone();
    p.insert("c".to_string(), serde_json::json!(10));
    assert!(!pred.eval(&p));

    let mut p = payload.clone();
    p.insert("d".to_string(), serde_json::json!(11));
    assert!(!pred.eval(&p));

    let mut p = payload.clone();
    p.insert("e".to_string(), serde_json::json!(0));
    assert!(!pred.eval(&p));

    let mut p = payload.clone();
    p.insert("f".to_string(), serde_json::json!(4));
    assert!(!pred.eval(&p));

    let mut p = payload.clone();
    p.insert("g".to_string(), serde_json::json!(6));
    assert!(!pred.eval(&p));
}

/// SQL 解析器错误路径：TDD 保证非法输入被拒绝，而非静默返回错误结果。
#[test]
fn sql_parser_error_paths() {
    // 未闭合字符串。
    assert!(parse_sql_filter("name = 'alice").is_err());
    // 非法字符。
    assert!(parse_sql_filter("a @ 1").is_err());
    // 尾部多余 token。
    assert!(parse_sql_filter("a = 1 )").is_err());
    // 缺少操作符。
    assert!(parse_sql_filter("a 1").is_err());
    // 缺少标量值。
    assert!(parse_sql_filter("a = ").is_err());
    // 缺少标识符。
    assert!(parse_sql_filter("= 1").is_err());
    // IN 缺少右括号。
    assert!(parse_sql_filter("a IN (1, 2").is_err());
    // IN 缺少左括号。
    assert!(parse_sql_filter("a IN 1, 2)").is_err());
    // 空输入。
    assert!(parse_sql_filter("").is_err());
    // 括号不匹配。
    assert!(parse_sql_filter("(a = 1").is_err());
}

/// `<>` 作为 `!=` 的别名，以及 Bool 值谓词求值。
#[test]
fn sql_ne_alias_and_bool() {
    // <> 等价于 !=。
    let pred = parse_sql_filter("status <> 'active'").unwrap();
    let mut p = serde_json::Map::new();
    p.insert("status".to_string(), serde_json::json!("inactive"));
    assert!(pred.eval(&p));
    p.insert("status".to_string(), serde_json::json!("active"));
    assert!(!pred.eval(&p));

    // Bool 值相等判断（TRUE -> 1.0, FALSE -> 0.0）。
    let pred = parse_sql_filter("flag = TRUE").unwrap();
    let mut p = serde_json::Map::new();
    p.insert("flag".to_string(), serde_json::json!(true));
    // 注意：TRUE 被词法化为 Number(1.0)，而 payload 存的是 Bool(true)，
    // 类型不匹配时返回 false——这是当前实现的预期行为。
    assert!(!pred.eval(&p));
    p.insert("flag".to_string(), serde_json::json!(1));
    assert!(pred.eval(&p));

    // FALSE 词法化为 0.0。
    let pred = parse_sql_filter("flag = FALSE").unwrap();
    let mut p = serde_json::Map::new();
    p.insert("flag".to_string(), serde_json::json!(0));
    assert!(pred.eval(&p));
}

/// OR 与括号优先级：`(a = 1 OR b = 2) AND c = 3`。
#[test]
fn sql_or_with_parentheses_precedence() {
    let pred = parse_sql_filter("(a = 1 OR b = 2) AND c = 3").unwrap();

    let mut p = serde_json::Map::new();
    p.insert("a".to_string(), serde_json::json!(1));
    p.insert("c".to_string(), serde_json::json!(3));
    assert!(pred.eval(&p));

    let mut p = serde_json::Map::new();
    p.insert("b".to_string(), serde_json::json!(2));
    p.insert("c".to_string(), serde_json::json!(3));
    assert!(pred.eval(&p));

    // OR 不满足但 AND 满足 -> false。
    let mut p = serde_json::Map::new();
    p.insert("a".to_string(), serde_json::json!(0));
    p.insert("c".to_string(), serde_json::json!(3));
    assert!(!pred.eval(&p));

    // OR 满足但 AND 不满足 -> false。
    let mut p = serde_json::Map::new();
    p.insert("a".to_string(), serde_json::json!(1));
    p.insert("c".to_string(), serde_json::json!(0));
    assert!(!pred.eval(&p));
}

/// 字段不存在时所有比较操作符应返回 false（不 panic）。
#[test]
fn sql_missing_field_returns_false() {
    let pred = parse_sql_filter("missing = 1 AND missing > 0 AND missing IN (1, 2)").unwrap();
    let p = serde_json::Map::new();
    assert!(!pred.eval(&p));
}
