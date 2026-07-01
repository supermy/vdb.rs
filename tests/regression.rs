//! 回归测试：防止已修复 bug 复发。
//!
//! 覆盖点：维度校验、空索引、manifest 持久化、负值点积、checksum 损坏检测。

use std::io::{Read, Seek, SeekFrom, Write};
use tempfile::TempDir;
use vdb_rs::index_ivf_rq::{IvfRabitqIndex, RabitqConfig, RabitqQuantizer};
use vdb_rs::search::{SearchOptions, search};
use vdb_rs::simd::{dot_product, l2_distance_squared};
use vdb_rs::storage::{load_index, save_index};

fn gaussian_random() -> f32 {
    let u1 = rand::random::<f32>().max(1e-7);
    let u2 = rand::random::<f32>();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
}

#[test]
#[should_panic(expected = "RaBitQ requires dim % 64 == 0")]
fn regression_dim_must_be_multiple_of_64() {
    RabitqConfig::new(100);
}

#[test]
fn regression_empty_index_search_returns_empty() {
    let index = IvfRabitqIndex::new(64);
    let query: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();

    let options = SearchOptions {
        k: 10,
        nprobe: 2,
        refine: false,
        refine_k: 0,
        fastscan: false,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: None,
    };
    let results = search(&index, &query, &options, None);
    assert!(results.is_empty());

    let results = index.search(&query, 10, 2);
    assert!(results.is_empty());
}

#[test]
fn regression_negative_dot_product_and_distance() {
    let a = vec![-1.0f32, -2.0, -3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    assert!((dot_product(&a, &b) - (-32.0)).abs() < 1e-6);
    assert!((l2_distance_squared(&a, &b) - 155.0).abs() < 1e-5);
}

#[test]
fn regression_checksum_detects_file_corruption() {
    let dim = 64;
    let mut index = IvfRabitqIndex::new(dim);
    for _ in 0..20 {
        let v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        index.add(&v);
    }

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("index.vdb");
    save_index(&path, &index).unwrap();

    let mut file = std::fs::File::options()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    file.seek(SeekFrom::Start(150 + 10)).unwrap();
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte).unwrap();
    byte[0] = !byte[0];
    file.seek(SeekFrom::Start(150 + 10)).unwrap();
    file.write_all(&byte).unwrap();
    drop(file);

    assert!(
        load_index(&path).is_err(),
        "corrupted file should fail checksum"
    );
}

#[test]
fn regression_manifest_persists_after_reopen() {
    use vdb_rs::vdb::Database;

    let dir = TempDir::new().unwrap();
    let db = Database::create(dir.path(), 64).unwrap();
    for _ in 0..5 {
        let v: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
        db.insert(&v).unwrap();
    }
    let stats_before = db.stats();
    assert!(stats_before.version > 0);

    let db2 = Database::open(dir.path()).unwrap();
    let stats_after = db2.stats();
    assert_eq!(stats_after.version, stats_before.version);
    assert_eq!(stats_after.num_vectors, stats_before.num_vectors);
}

/// 验收级召回率契约：在 1K 高斯随机向量上，IVF_RaBitQ + refine 的 Recall@10
/// 必须不低于 0.95。若此测试失败，说明质心初始化或距离估计存在回归。
#[test]
fn regression_recall_at_10_contract_1k() {
    let dim = 128;
    let n = 1_000;
    let k = 10;

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| gaussian_random()).collect())
        .collect();
    let index = IvfRabitqIndex::build(&vectors);

    let query: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();
    let truth: std::collections::HashSet<u64> = index
        .flat_search(&query, k)
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    let options = vdb_rs::search::SearchOptions {
        k,
        nprobe: 0,
        refine: true,
        refine_k: 100,
        fastscan: true,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: None,
    };
    let results = vdb_rs::search::search(&index, &query, &options, None);
    let recall = results.iter().filter(|(id, _)| truth.contains(id)).count();
    assert!(
        recall as f32 / k as f32 >= 0.95,
        "Recall@10 contract failed: {}/{} = {:.2}",
        recall,
        k,
        recall as f32 / k as f32
    );
}

/// 验证分区级 SQL 谓词下推不会丢失结果：
/// 带 SQL 过滤的搜索结果与“暴力 Flat + 同一谓词”的结果集合一致。
#[test]
fn regression_sql_pushdown_no_false_negative() {
    let dim = 64;
    let mut index = IvfRabitqIndex::new(dim);
    for i in 0..100 {
        let v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let mut payload = serde_json::Map::new();
        payload.insert("score".to_string(), serde_json::json!(i as f64));
        payload.insert(
            "tag".to_string(),
            serde_json::json!(if i % 3 == 0 { "a" } else { "b" }),
        );
        index.add_with_payload(&v, payload);
    }

    let query: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
    let sql = "score >= 50 AND tag = 'a'";

    // 暴力 Flat + SQL 结果。
    let mut truth: Vec<(u64, f32)> = index
        .raw_vectors()
        .iter()
        .enumerate()
        .filter(|(id, _)| {
            let p = index.payload(*id as u64).unwrap();
            vdb_rs::sql::parse_sql_filter(sql).unwrap().eval(p)
        })
        .map(|(id, v)| (id as u64, vdb_rs::simd::l2_distance_squared(&query, v)))
        .collect();
    truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let truth_set: std::collections::HashSet<u64> =
        truth.iter().take(10).map(|(id, _)| *id).collect();

    // IVF + SQL 下推结果。
    let options = SearchOptions {
        k: 10,
        nprobe: 0,
        refine: true,
        refine_k: 100,
        fastscan: false,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: Some(sql.to_string()),
    };
    let results = search(&index, &query, &options, None);
    let result_set: std::collections::HashSet<u64> = results.iter().map(|(id, _)| *id).collect();

    assert_eq!(
        result_set, truth_set,
        "SQL pushdown produced different top-10 than flat+SQL"
    );
}

#[test]
fn regression_rabitq_distance_is_nonnegative() {
    let dim = 64;
    let config = RabitqConfig::new(dim);
    let quantizer = RabitqQuantizer::new(config);

    let a: Vec<f32> = (0..dim).map(|_| rand::random::<f32>() - 0.5).collect();
    let b: Vec<f32> = (0..dim).map(|_| rand::random::<f32>() - 0.5).collect();

    let code_a = quantizer.encode(&a);
    let code_b = quantizer.encode(&b);

    let dist = quantizer.estimate_distance_sq(&code_a, &code_b);
    assert!(
        dist.is_finite() && dist >= 0.0,
        "estimated distance should be finite and non-negative, got {}",
        dist
    );
}
