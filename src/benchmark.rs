//! 内置对比测试框架。
//!
//! 测量 QPS、latency(p50,p99)、recall@k、build time 等指标。
//!
//! untested: benchmark 为独立二进制入口，通过 `cargo run --bin vdb-benchmark` 手动执行；
//! 单元测试层不测主函数、I/O 与真实数据集加载路径。

use clap::Parser;
use serde::Serialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

use vdb_rs::index_ivf_rq::IvfRabitqIndex;
use vdb_rs::search::{SearchOptions, search};

#[derive(Parser)]
#[command(name = "vdb-benchmark")]
#[command(about = "vdb.rs built-in benchmark")]
struct Args {
    /// Vector dimension
    #[arg(long, default_value_t = 128)]
    dim: usize,
    /// Number of vectors to index
    #[arg(long, default_value_t = 10000)]
    n: usize,
    /// TopK
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// Number of probe partitions (0 = all)
    #[arg(long, default_value_t = 0)]
    nprobe: usize,
    /// Number of query iterations
    #[arg(long, default_value_t = 100)]
    queries: usize,
    /// Disable refine
    #[arg(long)]
    no_refine: bool,
    /// Refine top-N candidates (default k*10)
    #[arg(long, default_value_t = 0)]
    refine_k: usize,
    /// Disable fastscan
    #[arg(long)]
    no_fastscan: bool,
    /// 输出 JSON 报告文件路径
    #[arg(long)]
    output: Option<String>,
    /// 跑测试矩阵（dim/n 组合）
    #[arg(long)]
    matrix: bool,
    /// 真实数据集路径前缀（如 ../models/data/siftsmall/siftsmall），
    /// 将自动加载 <prefix>_base.fvecs、<prefix>_query.fvecs、<prefix>_groundtruth.ivecs
    #[arg(long)]
    dataset: Option<String>,
}

fn gaussian_random() -> f32 {
    let u1 = rand::random::<f32>().max(1e-7);
    let u2 = rand::random::<f32>();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
}

/// 读取 fvecs 格式文件（每个向量前 4 字节小端 dim，随后 dim 个 float）。
fn read_fvecs<P: AsRef<Path>>(path: P) -> Vec<Vec<f32>> {
    let mut file = File::open(path).expect("open fvecs failed");
    let mut vectors = Vec::new();
    let mut dim_buf = [0u8; 4];
    while file.read_exact(&mut dim_buf).is_ok() {
        let dim = u32::from_le_bytes(dim_buf) as usize;
        assert!(dim % 64 == 0, "dataset dim {} not multiple of 64", dim);
        let mut bytes = vec![0u8; dim * 4];
        file.read_exact(&mut bytes)
            .expect("read vector bytes failed");
        let vec = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        vectors.push(vec);
    }
    vectors
}

/// 读取 ivecs 格式文件（每组前 4 字节小端 k，随后 k 个 int32）。
fn read_ivecs<P: AsRef<Path>>(path: P) -> Vec<Vec<i32>> {
    let mut file = File::open(path).expect("open ivecs failed");
    let mut lists = Vec::new();
    let mut k_buf = [0u8; 4];
    while file.read_exact(&mut k_buf).is_ok() {
        let k = u32::from_le_bytes(k_buf) as usize;
        let mut bytes = vec![0u8; k * 4];
        file.read_exact(&mut bytes)
            .expect("read groundtruth bytes failed");
        let list = bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        lists.push(list);
    }
    lists
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    dim: usize,
    n: usize,
    k: usize,
    nprobe: usize,
    build_time_ms: u128,
    qps: f64,
    p50_ms: f64,
    p99_ms: f64,
    recall_at_k: f64,
}

fn run_single(args: &Args) -> BenchmarkReport {
    // 加载真实数据集或生成随机数据。
    let (vectors, queries, groundtruth) = if let Some(prefix) = &args.dataset {
        println!("[BENCH] dataset={}", prefix);
        let base = read_fvecs(format!("{}_base.fvecs", prefix));
        let query = read_fvecs(format!("{}_query.fvecs", prefix));
        let truth = read_ivecs(format!("{}_groundtruth.ivecs", prefix));
        println!(
            "[BENCH] loaded base={} query={} truth={}",
            base.len(),
            query.len(),
            truth.len()
        );
        (base, query, Some(truth))
    } else {
        assert!(args.dim % 64 == 0, "dim must be a multiple of 64");
        let vectors: Vec<Vec<f32>> = (0..args.n)
            .map(|_| (0..args.dim).map(|_| gaussian_random()).collect())
            .collect();
        let queries: Vec<Vec<f32>> = (0..args.queries)
            .map(|_| (0..args.dim).map(|_| gaussian_random()).collect())
            .collect();
        (vectors, queries, None)
    };

    let dim = vectors.first().map(|v| v.len()).unwrap_or(args.dim);
    let n = vectors.len();
    let query_count = queries.len();
    let k = if let Some(gt) = &groundtruth {
        // 真实 groundtruth 的 k 可能和命令行不一致，取二者较小值。
        args.k
            .min(gt.first().map(|row| row.len()).unwrap_or(args.k))
    } else {
        args.k
    };

    println!(
        "[BENCH] dim={} n={} queries={} k={} nprobe={}",
        dim, n, query_count, k, args.nprobe
    );

    let build_start = Instant::now();
    let index = IvfRabitqIndex::build(&vectors);
    let build_time = build_start.elapsed();
    println!("[BENCH] build_time_ms={}", build_time.as_millis());

    let refine_k = if args.refine_k == 0 {
        k * 10
    } else {
        args.refine_k
    };
    let options = SearchOptions {
        k,
        nprobe: args.nprobe,
        refine: !args.no_refine,
        refine_k,
        fastscan: !args.no_fastscan,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: None,
    };

    // Warmup
    for q in queries.iter().take(5) {
        let _ = search(&index, q, &options, None);
    }

    let mut latencies: Vec<Duration> = Vec::with_capacity(query_count);
    let mut total_recall = 0usize;

    for (i, q) in queries.iter().enumerate() {
        let truth: HashSet<u64> = if let Some(gt) = &groundtruth {
            gt.get(i)
                .map(|row| row.iter().take(k).map(|&id| id as u64).collect())
                .unwrap_or_default()
        } else {
            index
                .flat_search(q, k)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        };

        let start = Instant::now();
        let results = search(&index, q, &options, None);
        latencies.push(start.elapsed());

        let recall = results.iter().filter(|(id, _)| truth.contains(id)).count();
        total_recall += recall;
    }

    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[(latencies.len() * 99) / 100];
    let total_lat: Duration = latencies.iter().sum();
    let qps = query_count as f64 / total_lat.as_secs_f64().max(1e-9);
    let avg_recall = total_recall as f64 / (query_count * k).max(1) as f64;

    println!(
        "[BENCH] qps={:.1} p50_ms={:.3} p99_ms={:.3} recall@{:.2}",
        qps,
        p50.as_secs_f64() * 1000.0,
        p99.as_secs_f64() * 1000.0,
        avg_recall
    );

    BenchmarkReport {
        dim,
        n,
        k,
        nprobe: args.nprobe,
        build_time_ms: build_time.as_millis(),
        qps,
        p50_ms: p50.as_secs_f64() * 1000.0,
        p99_ms: p99.as_secs_f64() * 1000.0,
        recall_at_k: avg_recall,
    }
}

fn main() {
    let args = Args::parse();

    let reports = if args.dataset.is_some() {
        // 真实数据集模式：只跑一次，不进入随机测试矩阵。
        vec![run_single(&args)]
    } else if args.matrix {
        // 测试矩阵：覆盖常见维度与数据量，大组合在快速基准中跳过。
        let dims = vec![64, 128, 256, 512, 768, 1024];
        let ns = vec![1_000, 10_000, 100_000];
        let mut reports = Vec::new();
        for dim in dims {
            for n in &ns {
                // 跳过可能超时的组合。
                if *n > 10_000 && dim > 512 {
                    continue;
                }
                let single_args = Args {
                    dim,
                    n: *n,
                    k: args.k,
                    nprobe: args.nprobe,
                    queries: args.queries,
                    no_refine: args.no_refine,
                    refine_k: args.refine_k,
                    no_fastscan: args.no_fastscan,
                    output: None,
                    matrix: false,
                    dataset: None,
                };
                reports.push(run_single(&single_args));
            }
        }
        reports
    } else {
        vec![run_single(&args)]
    };

    let json = serde_json::to_string_pretty(&reports).unwrap();
    println!("{}", json);

    if let Some(path) = &args.output {
        std::fs::write(path, json).expect("write report failed");
        println!("[BENCH] report written to {}", path);
    }
}
