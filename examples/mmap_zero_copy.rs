//! 分块 mmap + 用户态 LRU 零拷贝启动示例。
//!
//! 运行方式：
//!   cargo run --release --example mmap_zero_copy
//!
//! 说明：
//! - 先用普通 `Database` 写入数据；
//! - 再用 `MmapDatabase` 以零拷贝方式重新打开，查询时不把全量数据载入内存。

use std::fs;
use vdb_rs::vdb::{Database, MmapDatabase};

fn random_vec(dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rand::random::<f32>()).collect()
}

#[cfg(unix)]
fn main() {
    let dim = 128;
    let dir = "./examples/data/mmap_zero_copy";
    let _ = fs::remove_dir_all(dir);

    // 1. 写入阶段：使用普通 Database。
    let db = Database::create(dir, dim).expect("create db failed");
    for _ in 0..1000 {
        db.insert(&random_vec(dim)).expect("insert failed");
    }
    println!("[mmap] wrote {} vectors", db.stats().num_vectors);

    // 2. 零拷贝打开：只加载元数据，查询按需 fault 64MB chunk。
    let mmap_db = MmapDatabase::open(dir).expect("mmap open failed");
    println!(
        "[mmap] zero-copy opened: dim={} vectors={} partitions={}",
        mmap_db.dim(),
        mmap_db.stats().num_vectors,
        mmap_db.stats().num_partitions
    );

    let query = random_vec(dim);
    let results = mmap_db.search(&query, 10, 100);
    println!("[mmap] top-10 results: {:?}", results.iter().take(3).collect::<Vec<_>>());
}

#[cfg(not(unix))]
fn main() {
    println!("MmapDatabase is only available on Unix-like systems.");
}
