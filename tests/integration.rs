//! 集成测试：模块间交互。

use std::collections::HashSet;
use tempfile::TempDir;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;
use vdb_rs::search::{SearchOptions, search};
use vdb_rs::storage::{load_index, save_index};

fn gaussian_random() -> f32 {
    let u1 = rand::random::<f32>().max(1e-7);
    let u2 = rand::random::<f32>();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
}

#[test]
fn save_load_search_recall() {
    let dim = 128;
    let n = 1000;
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| gaussian_random()).collect())
        .collect();
    let index = IvfRabitqIndex::build(&vectors);

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("index.vdb");
    save_index(&path, &index).unwrap();

    let loaded = load_index(&path).unwrap();
    assert_eq!(loaded.len(), n);
    assert_eq!(loaded.num_partitions(), index.num_partitions());

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
    assert!(recall >= 5, "loaded index recall too low: {}/10", recall);
}

#[test]
fn sql_filter_matches_flat_filter() {
    let dim = 64;
    let mut index = IvfRabitqIndex::new(dim);
    for i in 0..50 {
        let v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let mut payload = serde_json::Map::new();
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
        sql_filter: Some("score >= 40".to_string()),
    };
    let results = search(&index, &query, &options, None);
    for (id, _) in &results {
        let score = index.payload(*id).unwrap()["score"].as_f64().unwrap();
        assert!(score >= 40.0);
    }
}
