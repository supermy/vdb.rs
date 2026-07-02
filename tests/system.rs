//! 系统测试：真实负载与内存 bounded 校验。
//!
//! 门控：release 下 10 万向量插入 < 60s、100 次查询 < 10s；
//! debug 下数据量减半，避免覆盖率插桩/未优化代码导致 CI 过慢。
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
fn system_bulk_insert_time_gate() {
    let dim = 64;
    // debug 构建数据量减半，既保留回归意义又避免 CI 超时。
    let (n, limit_secs) = if cfg!(debug_assertions) {
        (50_000, 60)
    } else {
        (100_000, 60)
    };

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

    eprintln!("[SYSTEM] {} insert elapsed: {:?}", n, elapsed);
    assert!(
        elapsed.as_secs() < limit_secs,
        "{} insert took too long: {:?}",
        n,
        elapsed
    );
}

#[test]
fn system_queries_time_gate() {
    let dim = 64;
    // debug 下同样减半，保持 100 次查询不变以验证延迟稳定性。
    let n = if cfg!(debug_assertions) {
        5_000
    } else {
        10_000
    };
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
