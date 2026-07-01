//! 命令行入口。
//!
//! 功能：创建数据库、插入/搜索、基于系统资源与数据规模的参数推荐。
//!
//! 为什么需要 `tune` 子命令：
//! IVF_RaBitQ 的最佳参数（nprobe / refine_k / query_bits）与数据规模、机器内存、CPU 核心数相关；
//! 通过 `tune` 可以在启动服务前快速获得推荐配置，避免写死参数。

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use vdb_rs::index_ivf_rq::Payload;
use vdb_rs::search::SearchOptions;
use vdb_rs::sys_info::{available_parallelism, physical_memory_bytes, recommend_search_options};
use vdb_rs::vdb::Database;

#[derive(Parser)]
#[command(name = "vdb")]
#[command(about = "vdb.rs command line interface")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 创建新的空数据库
    Create {
        /// 数据库目录
        #[arg(long)]
        dir: PathBuf,
        /// 向量维度（必须被 64 整除）
        #[arg(long)]
        dim: usize,
    },
    /// 插入单条向量
    Insert {
        /// 数据库目录
        #[arg(long)]
        dir: PathBuf,
        /// 向量 JSON 数组，如 "[0.1, 0.2, ...]"
        #[arg(long)]
        vector: String,
        /// 可选标量 payload JSON 对象
        #[arg(long)]
        payload: Option<String>,
    },
    /// 搜索最近邻
    Search {
        /// 数据库目录
        #[arg(long)]
        dir: PathBuf,
        /// 查询向量 JSON 数组
        #[arg(long)]
        query: String,
        /// 返回 TopK
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// 扫描分区数（0 表示全部）
        #[arg(long, default_value_t = 50)]
        nprobe: usize,
        /// 禁用原始向量精排
        #[arg(long)]
        no_refine: bool,
        /// 精排候选数
        #[arg(long, default_value_t = 1000)]
        refine_k: usize,
        /// 禁用 FastScan
        #[arg(long)]
        no_fastscan: bool,
        /// Query Quantization 位数（0 禁用）
        #[arg(long, default_value_t = 0)]
        query_bits: u8,
        /// 使用 SQ8 精排
        #[arg(long, default_value_t = false)]
        sq8_refine: bool,
        /// SQL WHERE 过滤条件
        #[arg(long)]
        sql_filter: Option<String>,
    },
    /// 根据系统资源与数据规模推荐参数
    Tune {
        /// 数据向量总数
        #[arg(long)]
        n: usize,
        /// 返回 TopK
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
}

fn parse_vector(s: &str) -> Result<Vec<f32>, String> {
    serde_json::from_str(s).map_err(|e| format!("invalid vector JSON: {e}"))
}

fn parse_payload(s: &str) -> Result<Payload, String> {
    let value: serde_json::Value =
        serde_json::from_str(s).map_err(|e| format!("invalid payload JSON: {e}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "payload must be a JSON object".to_string())?;
    let mut payload = Payload::new();
    for (k, v) in object {
        payload.insert(k.clone(), v.clone());
    }
    Ok(payload)
}

fn main() {
    let cli = Cli::parse();
    match run(cli.command) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("[vdb] error: {e}");
            std::process::exit(1);
        }
    }
}

fn run(command: Commands) -> Result<(), String> {
    match command {
        Commands::Create { dir, dim } => {
            if dim % 64 != 0 {
                return Err(format!("dim={dim} 不满足 dim % 64 == 0"));
            }
            let db = Database::create(&dir, dim).map_err(|e| e.to_string())?;
            let stats = db.stats();
            println!(
                "[create] dir={} dim={} version={} partitions={}",
                dir.display(),
                dim,
                stats.version,
                stats.num_partitions
            );
        }
        Commands::Insert {
            dir,
            vector,
            payload,
        } => {
            let vector = parse_vector(&vector)?;
            let payload = match payload {
                Some(s) => parse_payload(&s)?,
                None => Payload::new(),
            };
            let db = Database::open(&dir).map_err(|e| e.to_string())?;
            let id = db
                .insert_with_payload(&vector, payload)
                .map_err(|e| e.to_string())?;
            println!("[insert] id={}", id);
        }
        Commands::Search {
            dir,
            query,
            k,
            nprobe,
            no_refine,
            refine_k,
            no_fastscan,
            query_bits,
            sq8_refine,
            sql_filter,
        } => {
            let query = parse_vector(&query)?;
            let db = Database::open(&dir).map_err(|e| e.to_string())?;
            let options = SearchOptions {
                k,
                nprobe,
                refine: !no_refine,
                refine_k,
                fastscan: !no_fastscan,
                query_bits,
                sq8_refine,
                sql_filter,
            };
            let results = db.search(&query, &options);
            println!("[search] {} results", results.len());
            for (id, dist) in results {
                println!("  id={} distance={}", id, dist);
            }
        }
        Commands::Tune { n, k } => {
            let opts = recommend_search_options(n, k);
            println!("[tune] data size N={} k={}", n, k);
            println!("[tune] logical cpus={}", available_parallelism());
            if let Some(mem) = physical_memory_bytes() {
                println!(
                    "[tune] physical memory={:.2} GiB",
                    mem as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                println!(
                    "[tune] recommended mmap cache={:.2} GiB",
                    opts.mmap_cache_bytes.unwrap_or(0) as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            }
            println!("[tune] recommended partitions={}", opts.partitions);
            println!(
                "[tune] latency-optimized: nprobe={} refine_k={} query_bits={} fastscan={} recall={}",
                opts.latency.nprobe,
                opts.latency.refine_k,
                opts.latency.query_bits,
                opts.latency.fastscan,
                opts.latency.recall_target
            );
            println!(
                "[tune] balanced:          nprobe={} refine_k={} query_bits={} fastscan={} recall={}",
                opts.balanced.nprobe,
                opts.balanced.refine_k,
                opts.balanced.query_bits,
                opts.balanced.fastscan,
                opts.balanced.recall_target
            );
            println!(
                "[tune] high-recall:       nprobe={} refine_k={} query_bits={} fastscan={} recall={}",
                opts.recall.nprobe,
                opts.recall.refine_k,
                opts.recall.query_bits,
                opts.recall.fastscan,
                opts.recall.recall_target
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vector() {
        let v = parse_vector("[1.0, 2.0, 3.0]").unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_payload() {
        let p = parse_payload(r#"{"tag": "news", "score": 0.95}"#).unwrap();
        assert_eq!(p["tag"], serde_json::json!("news"));
        assert_eq!(p["score"], serde_json::json!(0.95));
    }
}
