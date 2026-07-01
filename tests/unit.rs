//! 单元测试：模块级函数正确性。

use std::collections::HashSet;
use vdb_rs::gpu::GpuDevice;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;
use vdb_rs::search::{SearchOptions, search};
use vdb_rs::simd::{batch_hamming_distance, dot_product, hamming_distance, l2_distance_squared};
use vdb_rs::sql::parse_sql_filter;

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
