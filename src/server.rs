//! OpenAI/Anthropic 兼容的 HTTP API 服务入口，基于 libevent evhttp。
//!
//! 特性：事件驱动 HTTP 服务、`include_str!` 编译时嵌入静态资源、
//! CORS 跨域支持、请求日志、`std::sync::Mutex` 并发保护、k≤256 保护、
//! 批量导入、search/insert/stats API。
//!
//! 通过 libevent C FFI 实现事件循环，替代 `std::net` 多线程模型。

use clap::Parser;
use std::net::SocketAddr;
use vdb_rs::http_server::HttpServer;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;

#[derive(Parser, Debug)]
#[command(name = "vdb-server", version, about = "vdb.rs HTTP server")]
struct Args {
    /// 监听地址
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,

    /// 向量维度
    #[arg(long, default_value_t = 64)]
    dim: usize,
}

fn main() {
    // untested: server 为独立二进制入口，生命周期由 HTTP 事件循环持有，单元测试不覆盖启动路径。
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let index = IvfRabitqIndex::new(args.dim);
    let mut server = HttpServer::new(index);

    server.bind(args.listen).expect("bind failed");
    server.run().expect("server failed");
}
