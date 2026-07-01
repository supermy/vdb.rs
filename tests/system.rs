//! 系统测试：真实负载与内存 bounded 校验。
//!
//! 门控：10 万向量插入 release < 60s（debug < 120s），100 次查询 < 10s，
//! 插入后分区总向量数等于插入数。

use std::time::Instant;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;
use vdb_rs::search::{SearchOptions, search};

fn gaussian_random() -> f32 {
    let u1 = rand::random::<f32>().max(1e-7);
    let u2 = rand::random::<f32>();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
}

#[test]
fn system_100k_insert_time_gate() {
    let dim = 64;
    let n = 100_000;

    let start = Instant::now();
    let mut index = IvfRabitqIndex::new(dim);
    for _ in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();
        index.add(&v);
    }
    let elapsed = start.elapsed();

    assert_eq!(index.len(), n, "total vectors after insert should equal n");

    let total_partitioned: usize = (0..index.num_partitions())
        .map(|pid| index.partition_entries(pid).len())
        .sum();
    assert_eq!(
        total_partitioned, n,
        "sum of partition entries should equal n"
    );

    eprintln!("[SYSTEM] 100K insert elapsed: {:?}", elapsed);
    // release 目标 60s；debug 构建允许放宽到 120s，避免 CI/本地调试时因未优化代码超时。
    let limit_secs = if cfg!(debug_assertions) { 120 } else { 60 };
    assert!(
        elapsed.as_secs() < limit_secs,
        "100K insert took too long: {:?}",
        elapsed
    );
}

#[test]
fn system_100_queries_time_gate() {
    let dim = 64;
    let n = 10_000;
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| gaussian_random()).collect())
        .collect();
    let index = IvfRabitqIndex::build(&vectors);

    let queries: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..dim).map(|_| gaussian_random()).collect())
        .collect();

    let options = SearchOptions {
        k: 10,
        nprobe: 8,
        refine: true,
        refine_k: 100,
        fastscan: true,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: None,
    };

    let start = Instant::now();
    for q in &queries {
        let _ = search(&index, q, &options, None);
    }
    let elapsed = start.elapsed();

    eprintln!("[SYSTEM] 100 queries elapsed: {:?}", elapsed);
    assert!(
        elapsed.as_secs() < 10,
        "100 queries took too long: {:?}",
        elapsed
    );
}
