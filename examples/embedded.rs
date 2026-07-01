//! 嵌入式模式完整示例：创建数据库、插入、搜索、SQL 过滤、统计。
//!
//! 运行方式：
//!   cargo run --release --example embedded

use std::fs;
use vdb_rs::search::SearchOptions;
use vdb_rs::vdb::Database;

fn random_vec(dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rand::random::<f32>()).collect()
}

fn main() {
    let dim = 128;
    let dir = "./examples/data/embedded";
    let _ = fs::remove_dir_all(dir);

    // 1. 创建数据库。
    let db = Database::create(dir, dim).expect("create db failed");

    // 2. 批量插入带 payload 的向量。
    for i in 0..1000 {
        let mut payload = vdb_rs::index_ivf_rq::Payload::new();
        payload.insert("id".to_string(), serde_json::json!(i));
        payload.insert("score".to_string(), serde_json::json!((i % 100) as f64));
        let v = random_vec(dim);
        db.insert_with_payload(&v, payload).expect("insert failed");
    }

    let stats = db.stats();
    println!(
        "[embedded] version={} vectors={} partitions={}",
        stats.version, stats.num_vectors, stats.num_partitions
    );

    // 3. 普通向量搜索（推荐生产配置）。
    let query = random_vec(dim);
    let opts = SearchOptions {
        k: 10,
        nprobe: 50,
        refine: true,
        refine_k: 1000,
        fastscan: true,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: None,
    };
    let results = db.search(&query, &opts);
    println!(
        "[embedded] top-10 results: {:?}",
        results.iter().take(3).collect::<Vec<_>>()
    );

    // 4. SQL WHERE + 向量搜索联合查询。
    let opts_sql = SearchOptions {
        k: 5,
        nprobe: 50,
        refine: true,
        refine_k: 500,
        fastscan: true,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: Some("score >= 90".to_string()),
    };
    let results = db.search(&query, &opts_sql);
    println!("[embedded] SQL filtered results: {} items", results.len());

    // 5. 高召回配置示例。
    let opts_high_recall = SearchOptions {
        k: 10,
        nprobe: 100,
        refine: true,
        refine_k: 5000,
        fastscan: true,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: None,
    };
    let results = db.search(&query, &opts_high_recall);
    println!("[embedded] high-recall results: {} items", results.len());
}
