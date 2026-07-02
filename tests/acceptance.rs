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

/// 批量插入 + compact：验证 batch_insert_with_payload 只生成一个版本快照，
/// 且 compact 能安全清理旧版本。
#[test]
fn acceptance_database_batch_insert_and_compact() {
    let dir = TempDir::new().unwrap();
    let db = Database::create(dir.path(), 64).unwrap();

    // 初始创建会写入 index-0.vdb。
    let mut vectors = Vec::new();
    let mut payloads = Vec::new();
    for i in 0..5 {
        let v: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
        let mut payload = serde_json::Map::new();
        payload.insert("idx".to_string(), serde_json::json!(i));
        vectors.push(v);
        payloads.push(payload);
    }
    let first_id = db.batch_insert_with_payload(&vectors, payloads).unwrap();
    assert_eq!(first_id, 0);

    let stats = db.stats();
    assert_eq!(stats.num_vectors, 5);

    // 再逐条插入一次，产生另一个全量快照，用于验证 compact 清理旧版本。
    let extra: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
    db.insert(&extra).unwrap();
    assert_eq!(db.stats().num_vectors, 6);

    let removed = db.compact().unwrap();
    assert!(
        removed >= 1,
        "compact should remove at least one old snapshot"
    );

    // 验证只保留 manifest 指向的最新 index 文件。
    let manifest_index = std::fs::read_to_string(dir.path().join("manifest.json")).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&manifest_index).unwrap();
    let index_file = manifest["index_file"].as_str().unwrap();
    let mut index_files: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().to_str()?.to_string();
            if name.starts_with("index-") && name.ends_with(".vdb") {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    index_files.sort();
    assert_eq!(index_files, vec![index_file.to_string()]);

    // reopen 后数据应一致。
    let db2 = Database::open(dir.path()).unwrap();
    assert_eq!(db2.stats().num_vectors, 6);

    // 通过 payload 搜索验证批量插入的向量可检索。
    let query = vectors[0].clone();
    let options = SearchOptions {
        k: 6,
        nprobe: 0,
        refine: true,
        refine_k: 50,
        fastscan: true,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: Some("idx = 0".to_string()),
    };
    let results = db2.search(&query, &options);
    assert!(!results.is_empty());
    assert!(results.iter().any(|(id, _)| *id == 0));
}
