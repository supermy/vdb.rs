//! 最佳性能配置示例：在随机数据上对比不同搜索配置的延迟、QPS 与召回率。
//!
//! 运行方式：
//!   cargo run --release --example best_performance
//!
//! 说明：
//! - 使用 `IvfRabitqIndex::build` 批量构建索引（k-means 训练质心）。
//! - 以暴力 Flat 结果为 ground truth 计算 recall@k。
//! - 输出若干典型配置的组合建议。

use std::time::Instant;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;
use vdb_rs::search::{search, SearchOptions};

fn random_vec(dim: usize) -> Vec<f32> {
    // 近似高斯分布，使 k-means 分区更有意义。
    (0..dim)
        .map(|_| {
            let u1 = rand::random::<f32>().max(1e-7);
            let u2 = rand::random::<f32>();
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
        })
        .collect()
}

fn recall_at_k(results: &[(u64, f32)], truth: &[(u64, f32)], k: usize) -> f32 {
    let truth_ids: std::collections::HashSet<u64> = truth.iter().take(k).map(|(id, _)| *id).collect();
    let hit = results.iter().take(k).filter(|(id, _)| truth_ids.contains(id)).count();
    hit as f32 / k.min(truth_ids.len()).max(1) as f32
}

fn bench_config(
    name: &str,
    index: &IvfRabitqIndex,
    queries: &[Vec<f32>],
    truth: &[Vec<(u64, f32)>],
    options: &SearchOptions,
) {
    let k = options.k;
    let start = Instant::now();
    let mut results = Vec::with_capacity(queries.len());
    for query in queries {
        results.push(search(index, query, options, None));
    }
    let elapsed = start.elapsed();
    let total_qps = queries.len() as f64 / elapsed.as_secs_f64();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / queries.len() as f64;

    let avg_recall: f32 = results
        .iter()
        .enumerate()
        .map(|(i, r)| recall_at_k(r, &truth[i], k))
        .sum::<f32>()
        / queries.len() as f32;

    println!(
        "[perf] {:22} recall@{:2}={:.3}  QPS={:>7.1}  p50(ms)={:.3}",
        name, k, avg_recall, total_qps, avg_ms
    );
}

fn main() {
    let dim = 128;
    let n = 10_000;
    let n_queries = 100;
    let k = 10;

    println!("[perf] building index: dim={} n={}", dim, n);
    let build_start = Instant::now();
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vec(dim)).collect();
    let index = IvfRabitqIndex::build(&vectors);
    println!(
        "[perf] build completed in {:.2}s, partitions={}",
        build_start.elapsed().as_secs_f64(),
        index.num_partitions()
    );

    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| random_vec(dim)).collect();
    let truth: Vec<Vec<(u64, f32)>> = queries
        .iter()
        .map(|q| index.flat_search(q, k))
        .collect();

    println!("\n[perf] latency-optimized configurations");
    bench_config(
        "fastscan nprobe=16",
        &index,
        &queries,
        &truth,
        &SearchOptions {
            k,
            nprobe: 16,
            refine: true,
            refine_k: k * 10,
            fastscan: true,
            query_bits: 0,
            sq8_refine: false,
            sql_filter: None,
        },
    );
    bench_config(
        "fastscan+qq4 nprobe=16",
        &index,
        &queries,
        &truth,
        &SearchOptions {
            k,
            nprobe: 16,
            refine: true,
            refine_k: k * 10,
            fastscan: true,
            query_bits: 4,
            sq8_refine: false,
            sql_filter: None,
        },
    );
    bench_config(
        "fastscan+qq8 nprobe=16",
        &index,
        &queries,
        &truth,
        &SearchOptions {
            k,
            nprobe: 16,
            refine: true,
            refine_k: k * 10,
            fastscan: true,
            query_bits: 8,
            sq8_refine: false,
            sql_filter: None,
        },
    );

    println!("\n[perf] balanced configurations");
    bench_config(
        "fastscan nprobe=50 refine_k=1000",
        &index,
        &queries,
        &truth,
        &SearchOptions {
            k,
            nprobe: 50,
            refine: true,
            refine_k: 1000,
            fastscan: true,
            query_bits: 0,
            sq8_refine: false,
            sql_filter: None,
        },
    );
    bench_config(
        "fastscan+qq8 nprobe=50 refine_k=1000",
        &index,
        &queries,
        &truth,
        &SearchOptions {
            k,
            nprobe: 50,
            refine: true,
            refine_k: 1000,
            fastscan: true,
            query_bits: 8,
            sq8_refine: false,
            sql_filter: None,
        },
    );

    println!("\n[perf] high-recall configurations");
    bench_config(
        "nprobe=100 refine_k=5000",
        &index,
        &queries,
        &truth,
        &SearchOptions {
            k,
            nprobe: 100,
            refine: true,
            refine_k: 5000,
            fastscan: true,
            query_bits: 0,
            sq8_refine: false,
            sql_filter: None,
        },
    );
    bench_config(
        "all-partitions",
        &index,
        &queries,
        &truth,
        &SearchOptions {
            k,
            nprobe: 0,
            refine: true,
            refine_k: k * 10,
            fastscan: true,
            query_bits: 0,
            sq8_refine: false,
            sql_filter: None,
        },
    );

    println!("\n[perf] recommendations");
    println!("  - 速度优先：nprobe=16, fastscan=true, query_bits=8, refine_k=k*10");
    println!("  - 平衡配置：nprobe=50, fastscan=true, query_bits=0/8, refine_k=1000");
    println!("  - 高召回配置：nprobe=100, refine_k=5000, fastscan=true");
    println!("  - 精确检索：nprobe=0（扫描全部分区），适合离线校验");
}
