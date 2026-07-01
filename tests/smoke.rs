//! 冒烟测试：快速构建验证，带 `[SMOKE]` 前缀日志。

use std::collections::HashSet;
use tempfile::TempDir;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;
use vdb_rs::search::{SearchOptions, search};
use vdb_rs::sql::parse_sql_filter;
use vdb_rs::storage::{load_index, save_index};

fn gaussian_random() -> f32 {
    let u1 = rand::random::<f32>().max(1e-7);
    let u2 = rand::random::<f32>();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
}

#[test]
fn smoke_index_build_and_search() {
    eprintln!("[SMOKE] index build + search");
    let dim = 64;
    let n = 200;
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| gaussian_random()).collect())
        .collect();
    let index = IvfRabitqIndex::build(&vectors);
    assert_eq!(index.len(), n);

    let query: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();
    let truth: HashSet<u64> = index
        .flat_search(&query, 5)
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    let options = SearchOptions {
        k: 5,
        nprobe: 0,
        refine: true,
        refine_k: 50,
        fastscan: true,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: None,
    };
    let results = search(&index, &query, &options, None);
    let recall = results.iter().filter(|(id, _)| truth.contains(id)).count();
    eprintln!("[SMOKE] recall@5 = {}/{}", recall, 5);
    assert!(recall >= 1, "smoke recall too low: {}/5", recall);
}

#[test]
fn smoke_sql_parser() {
    eprintln!("[SMOKE] sql parser");
    let pred = parse_sql_filter("a = 1 AND b IN ('x', 'y')").unwrap();
    let mut payload = serde_json::Map::new();
    payload.insert("a".to_string(), serde_json::json!(1));
    payload.insert("b".to_string(), serde_json::json!("y"));
    assert!(pred.eval(&payload));
}

#[test]
fn smoke_storage_roundtrip() {
    eprintln!("[SMOKE] storage roundtrip");
    let dim = 64;
    let mut index = IvfRabitqIndex::new(dim);
    for _ in 0..50 {
        let v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        index.add(&v);
    }

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("smoke.vdb");
    save_index(&path, &index).unwrap();

    let loaded = load_index(&path).unwrap();
    assert_eq!(loaded.len(), index.len());
    assert_eq!(loaded.num_partitions(), index.num_partitions());
    eprintln!(
        "[SMOKE] loaded {} vectors in {} partitions",
        loaded.len(),
        loaded.num_partitions()
    );
}
