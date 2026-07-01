//! 验收测试：用户可见功能验证。
//!
//! 覆盖：Database 创建-插入-搜索、批量导入、payload 过滤、reopen 后一致性。

use std::collections::HashSet;
use tempfile::TempDir;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;
use vdb_rs::search::{SearchOptions, search};
use vdb_rs::storage::{load_index, save_index};
use vdb_rs::vdb::Database;

fn gaussian_random() -> f32 {
    let u1 = rand::random::<f32>().max(1e-7);
    let u2 = rand::random::<f32>();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
}

#[test]
fn acceptance_database_create_insert_search() {
    let dir = TempDir::new().unwrap();
    let db = Database::create(dir.path(), 64).unwrap();

    let v: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
    let id = db.insert(&v).unwrap();
    assert_eq!(id, 0);

    let results = db.search(&v, &SearchOptions::default());
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 0);
}

#[test]
fn acceptance_database_persist_and_reopen() {
    let dir = TempDir::new().unwrap();
    let db = Database::create(dir.path(), 64).unwrap();
    for _ in 0..20 {
        let v: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
        db.insert(&v).unwrap();
    }
    let stats = db.stats();

    let db2 = Database::open(dir.path()).unwrap();
    let stats2 = db2.stats();
    assert_eq!(stats2.num_vectors, stats.num_vectors);
    assert_eq!(stats2.num_partitions, stats.num_partitions);
    assert_eq!(stats2.version, stats.version);

    let query: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
    let results = db2.search(&query, &SearchOptions::default());
    assert_eq!(
        results.len(),
        SearchOptions::default().k.min(stats.num_vectors)
    );
}

#[test]
fn acceptance_payload_filter_search() {
    let dim = 64;
    let mut index = IvfRabitqIndex::new(dim);
    for i in 0..30 {
        let v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let mut payload = serde_json::Map::new();
        payload.insert(
            "tag".to_string(),
            serde_json::json!(if i % 2 == 0 { "even" } else { "odd" }),
        );
        payload.insert("score".to_string(), serde_json::json!(i as f64));
        index.add_with_payload(&v, payload);
    }

    let query: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
    let options = SearchOptions {
        k: 100,
        nprobe: 0,
        refine: false,
        refine_k: 0,
        fastscan: false,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: Some("tag = 'even' AND score >= 10".to_string()),
    };
    let results = search(&index, &query, &options, None);
    assert!(!results.is_empty());
    for (id, _) in &results {
        let p = index.payload(*id).unwrap();
        assert_eq!(p["tag"].as_str().unwrap(), "even");
        assert!(p["score"].as_f64().unwrap() >= 10.0);
    }
}

#[test]
fn acceptance_save_load_recall() {
    let dim = 128;
    let n = 500;
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| gaussian_random()).collect())
        .collect();
    let index = IvfRabitqIndex::build(&vectors);

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("acceptance.vdb");
    save_index(&path, &index).unwrap();

    let loaded = load_index(&path).unwrap();
    let query: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();
    let truth: HashSet<u64> = loaded
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
    let results = search(&loaded, &query, &options, None);
    let recall = results.iter().filter(|(id, _)| truth.contains(id)).count();
    assert!(recall >= 5, "acceptance recall too low: {}/10", recall);
}
