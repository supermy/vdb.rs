//! 性能矩阵测试示例：自动调参 + 多配置对比。
//!
//! 运行方式：
//!   cargo run --release --example performance_matrix
//!   cargo run --release --example performance_matrix -- --n 50000 --dim 128 --k 10
//!
//! 说明：
//! - 先通过 sys_info 输出当前机器资源与推荐参数；
//! - 再批量构建索引，对比不同 nprobe / query_bits / refine_k 组合；
//! - 输出 CSV 风格结果，便于复制到表格做延迟-召回权衡。

use std::time::Instant;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;
use vdb_rs::search::{SearchOptions, search};
use vdb_rs::sys_info::recommend_search_options;

fn random_vec(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| {
            let u1 = rand::random::<f32>().max(1e-7);
            let u2 = rand::random::<f32>();
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
        })
        .collect()
}

fn recall_at_k(results: &[(u64, f32)], truth: &[(u64, f32)], k: usize) -> f32 {
    let truth_ids: std::collections::HashSet<u64> =
        truth.iter().take(k).map(|(id, _)| *id).collect();
    let hit = results
        .iter()
        .take(k)
        .filter(|(id, _)| truth_ids.contains(id))
        .count();
    hit as f32 / k.min(truth_ids.len()).max(1) as f32
}

fn run_matrix(index: &IvfRabitqIndex, queries: &[Vec<f32>], truth: &[Vec<(u64, f32)>], k: usize) {
    let configs: Vec<(&str, SearchOptions)> = vec![
        (
            "latency",
            SearchOptions {
                k,
                nprobe: 16,
                refine: true,
                refine_k: k * 10,
                fastscan: true,
                query_bits: 8,
                sq8_refine: false,
                sql_filter: None,
            },
        ),
        (
            "balanced",
            SearchOptions {
                k,
                nprobe: 50,
                refine: true,
                refine_k: 1000,
                fastscan: true,
                query_bits: 0,
                sq8_refine: false,
                sql_filter: None,
            },
        ),
        (
            "balanced+qq8",
            SearchOptions {
                k,
                nprobe: 50,
                refine: true,
                refine_k: 1000,
                fastscan: true,
                query_bits: 8,
                sq8_refine: false,
                sql_filter: None,
            },
        ),
        (
            "high-recall",
            SearchOptions {
                k,
                nprobe: 100,
                refine: true,
                refine_k: 5000,
                fastscan: true,
                query_bits: 0,
                sq8_refine: false,
                sql_filter: None,
            },
        ),
        (
            "exact",
            SearchOptions {
                k,
                nprobe: 0,
                refine: true,
                refine_k: k * 10,
                fastscan: true,
                query_bits: 0,
                sq8_refine: false,
                sql_filter: None,
            },
        ),
    ];

    println!("\nname,nprobe,refine_k,query_bits,fastscan,recall@k,qps,p50_ms");
    for (name, opts) in configs {
        let start = Instant::now();
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            results.push(search(index, query, &opts, None));
        }
        let elapsed = start.elapsed();
        let qps = queries.len() as f64 / elapsed.as_secs_f64();
        let p50_ms = elapsed.as_secs_f64() * 1000.0 / queries.len() as f64;
        let avg_recall: f32 = results
            .iter()
            .enumerate()
            .map(|(i, r)| recall_at_k(r, &truth[i], opts.k))
            .sum::<f32>()
            / queries.len() as f32;

        println!(
            "{},{},{},{},{},{:.4},{:.1},{:.3}",
            name,
            opts.nprobe,
            opts.refine_k,
            opts.query_bits,
            opts.fastscan,
            avg_recall,
            qps,
            p50_ms
        );
    }
}

#[derive(Debug)]
struct Args {
    n: usize,
    dim: usize,
    k: usize,
    queries: usize,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut n = 10_000usize;
    let mut dim = 128usize;
    let mut k = 10usize;
    let mut queries = 100usize;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--n" => n = args.next().unwrap().parse().unwrap(),
            "--dim" => dim = args.next().unwrap().parse().unwrap(),
            "--k" => k = args.next().unwrap().parse().unwrap(),
            "--queries" => queries = args.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }
    Args { n, dim, k, queries }
}

fn main() {
    let args = parse_args();
    println!(
        "[matrix] n={} dim={} k={} queries={}",
        args.n, args.dim, args.k, args.queries
    );

    let rec = recommend_search_options(args.n, args.k);
    println!("[matrix] recommended partitions: {}", rec.partitions);
    println!(
        "[matrix] latency:  nprobe={} refine_k={} query_bits={} fastscan={}",
        rec.latency.nprobe, rec.latency.refine_k, rec.latency.query_bits, rec.latency.fastscan
    );
    println!(
        "[matrix] balanced: nprobe={} refine_k={} query_bits={} fastscan={}",
        rec.balanced.nprobe, rec.balanced.refine_k, rec.balanced.query_bits, rec.balanced.fastscan
    );
    println!(
        "[matrix] recall:   nprobe={} refine_k={} query_bits={} fastscan={}",
        rec.recall.nprobe, rec.recall.refine_k, rec.recall.query_bits, rec.recall.fastscan
    );

    let build_start = Instant::now();
    let vectors: Vec<Vec<f32>> = (0..args.n).map(|_| random_vec(args.dim)).collect();
    let index = IvfRabitqIndex::build(&vectors);
    println!(
        "[matrix] build: {:.2}s partitions={}",
        build_start.elapsed().as_secs_f64(),
        index.num_partitions()
    );

    let queries: Vec<Vec<f32>> = (0..args.queries).map(|_| random_vec(args.dim)).collect();
    let truth: Vec<Vec<(u64, f32)>> = queries
        .iter()
        .map(|q| index.flat_search(q, args.k))
        .collect();

    run_matrix(&index, &queries, &truth, args.k);
}
