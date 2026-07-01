//! 命令行入口。
//!
//! 功能：索引构建（create-index --type IVF_RQ）、分区维护、本地 REPL 查询、
//! 嵌入式模式直接查询。

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "vdb")]
#[command(about = "vdb.rs command line interface")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create an index
    CreateIndex {
        /// Index type
        #[arg(long, default_value = "IVF_RQ")]
        #[allow(dead_code)]
        r#type: String,
    },
    /// Query in embedded mode
    Query,
}

fn main() {
    // untested: CLI 当前为占位实现，完整子命令逻辑待后续阶段补充测试。
    let _cli = Cli::parse();
    println!("vdb CLI placeholder");
}
