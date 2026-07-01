//! vdb.rs 核心库
//!
//! 面向单机十亿级、内存/磁盘受限场景的 IVF_RaBitQ 向量检索引擎。

pub mod fulltext;
pub mod gpu;
pub mod http_server;
pub mod hybrid;
pub mod index_ivf_rq;
#[cfg(unix)]
pub mod mmap_index;
pub mod search;
pub mod simd;
pub mod sql;
pub mod storage;
pub mod thread_pool;
pub mod vdb;
