//! OpenAI/Anthropic 兼容的 HTTP API 服务入口，基于 libevent evhttp。
//!
//! 特性：事件驱动 HTTP 服务、`include_str!` 编译时嵌入静态资源、
//! CORS 跨域支持、请求日志、`std::sync::Mutex` 并发保护、k≤256 保护、
//! 批量导入、search/insert/stats API、服务器级默认搜索参数。
//!
//! 通过 libevent C FFI 实现事件循环，替代 `std::net` 多线程模型。
//!
//! 启动时支持通过命令行覆盖默认搜索参数，避免请求体或代码中写死最佳配置。

use clap::Parser;
use std::net::SocketAddr;
use vdb_rs::http_server::{HttpServer, ServerOptions};
use vdb_rs::index_ivf_rq::IvfRabitqIndex;
use vdb_rs::search::SearchOptions;
use vdb_rs::sys_info::{available_parallelism, physical_memory_bytes, recommend_mmap_cache_bytes};

#[derive(Parser, Debug)]
#[command(name = "vdb-server", version, about = "vdb.rs HTTP server")]
struct Args {
    /// 监听地址
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,

    /// 向量维度
    #[arg(long, default_value_t = 64)]
    dim: usize,

    /// 默认 TopK
    #[arg(long, default_value_t = 10)]
    default_k: usize,

    /// 默认扫描分区数（0 表示全部）
    #[arg(long, default_value_t = 50)]
    default_nprobe: usize,

    /// 默认禁用原始向量精排
    #[arg(long)]
    default_no_refine: bool,

    /// 默认精排候选数
    #[arg(long, default_value_t = 1000)]
    default_refine_k: usize,

    /// 默认禁用 FastScan
    #[arg(long)]
    default_no_fastscan: bool,

    /// 默认 Query Quantization 位数
    #[arg(long, default_value_t = 0)]
    default_query_bits: u8,

    /// 默认使用 SQ8 精排
    #[arg(long, default_value_t = false)]
    default_sq8_refine: bool,
}

fn main() {
    // untested: server 为独立二进制入口，生命周期由 HTTP 事件循环持有，单元测试不覆盖启动路径。
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let index = IvfRabitqIndex::new(args.dim);

    let server_options = ServerOptions {
        search: SearchOptions {
            k: args.default_k,
            nprobe: args.default_nprobe,
            refine: !args.default_no_refine,
            refine_k: args.default_refine_k,
            fastscan: !args.default_no_fastscan,
            query_bits: args.default_query_bits,
            sq8_refine: args.default_sq8_refine,
            sql_filter: None,
        },
    };

    log::info!(
        "vdb-server starting with dim={} cpus={}",
        args.dim,
        available_parallelism()
    );
    if let Some(mem) = physical_memory_bytes() {
        log::info!(
            "physical memory={:.2} GiB recommended mmap cache={:.2} GiB",
            mem as f64 / (1024.0 * 1024.0 * 1024.0),
            recommend_mmap_cache_bytes(mem) as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    log::info!("default search options: {:?}", server_options.search);

    let mut server = HttpServer::new_with_options(index, server_options);

    server.bind(args.listen).expect("bind failed");
    server.run().expect("server failed");
}
