# vdb.rs

面向单机十亿级、内存/磁盘极度受限场景的 IVF_RaBitQ 向量检索引擎。

## 定位

- 基于 Lance 列式格式与 IVF_RaBitQ 量化索引，以 ~1 bit/dim 压缩率实现无偏距离估计。
- 原生支持 **SQL WHERE + 向量搜索** 的联合执行：谓词下推至分区级别，再执行 RaBitQ 位运算检索。
- 原生支持全文检索与多模态数据联合查询（规划中）。
- 提供嵌入式模式与可选 Server/NNG 模式，两者共享同一存储格式与执行引擎。

## 核心特性

| 特性 | 状态 | 说明 |
|------|------|------|
| IVF_RaBitQ 索引 | ✅ | 随机正交旋转 + 超球面归一化 + 符号二值化 |
| SIMD 加速 | ✅ | AVX-512 / AVX2 / aarch64 NEON + 纯 Rust fallback |
| FastScan | ✅ | batch XOR-popcount，QPS 提升 3.5x 以上 |
| 精排层 Refine | ✅ | 原始向量重排，保障生产召回率 |
| SQL 谓词下推 | ✅ | 自研最小 WHERE 子集，支持 `= != < > <= >= AND OR IN` |
| 磁盘存储 | ✅ | partition-oriented columnar 格式，save/load + manifest 版本快照 |
| NNG 二进制协议 | ✅ | TCP `[4B length][1B cmd][payload]`，低延迟接口 |
| HTTP Server | ✅ | libevent evhttp C FFI，OpenAI/Anthropic 兼容 API |
| GPU 后备 | ✅ | Metal / CUDA / OpenCL 架构，无 GPU 自动降级 CPU SIMD |
| 真实数据集测试 | ✅ | `vdb-benchmark --dataset <prefix>` 支持 fvecs 格式 |
| SQ8 精排 | ✅ | per-partition min/max 动态范围 |
| Query Quantization | ✅ | 1-8 bit 查询量化，8-bit 模式 QPS ×2 |
| R*centroid 预计算 | ✅ | 查询时 IVF 路由从 O(dim²) 降至 O(dim) |
| mmap 按需加载 | ✅ | `MmapStorage` + 物理内存 bound（<16GB 限制 50%） |
| 分块 mmap + 用户态 LRU | ✅ | `ChunkedMmapStorage` + `MmapDatabase`，64MB chunk、LRU 淘汰、流式 CRC，实现零拷贝启动 |
| POSIX socket NNG | ✅ | Unix 路径使用 libc 直接系统调用 |

## 快速开始

```bash
# 构建
cargo build

# 运行全部八层测试
cargo test

# 命令行帮助
cargo run --bin vdb -- --help

# 启动 NNG 二进制服务
cargo run --bin vdb-nng-server

# 启动 HTTP 服务（含默认测试页面）
cargo run --bin vdb-server
```

## 真实数据集测试

支持 `.fvecs` 格式数据集（如 SIFT1M / siftsmall）：

```bash
cargo run --bin vdb-benchmark -- \
  --dataset ../models/data/siftsmall/siftsmall \
  --k 10 --nprobe 100 --refine-k 100
```

参数说明：

- `--dataset <prefix>`：加载 `<prefix>_base.fvecs`、`<prefix>_query.fvecs`、`<prefix>_groundtruth.ivecs`
- `--k`：返回 TopK
- `--nprobe`：扫描分区数；nprobe=0 时使用默认值
- `--refine-k`：精排候选数，增大可显著提升召回率
- `--no-refine` / `--no-fastscan`：关闭精排或 FastScan 路径

## 测试体系

| 类型 | 命令 |
|------|------|
| 单元测试 | `cargo test --test unit` |
| 集成测试 | `cargo test --test integration` |
| 冒烟测试 | `cargo test --test smoke` |
| 回归测试 | `cargo test --test regression` |
| 验收测试 | `cargo test --test acceptance` |
| 系统测试 | `cargo test --test system` |
| 端到端测试 | `cargo test --test e2e` |
| 服务器测试 | `cargo test --test server` |

## API 速览

```rust
use vdb_rs::vdb::Database;
use vdb_rs::search::{search, SearchOptions};

let db = Database::create("./data", 128)?;
let id = db.insert(&vec)?;

let opts = SearchOptions {
    k: 10,
    nprobe: 100,
    refine: true,
    refine_k: 100,
    fastscan: true,
    query_bits: 0,
    sq8_refine: false,
    sql_filter: Some("tag = 'news' AND score >= 0.9".to_string()),
};
let results = db.search(&query, &opts);
```

## 生产部署建议

- 维度必须满足 `dim % 64 == 0`（RaBitQ 量化要求）。
- 推荐 `num_partitions = sqrt(N)`，上限 65536，下限 4。
- 启用 refine 层保障召回率：`refine = true`，`refine_k = 10` 起。
- 速度优先：启用 FastScan（`fastscan = true`）。
- 精度优先：适当增大 `nprobe` 与 `refine_k`。
- NNG 模式适合低延迟内网调用；HTTP 模式适合调试与浏览器测试页面。
- 禁止在 <16GB 内存设备上一次性 mmap 超过 50% 物理内存；分块 mmap + 用户态 LRU（`MmapDatabase`）自动将缓存控制在预算内，实现零拷贝启动。

## 性能实测（siftsmall 10K × 128d）

环境：Apple Silicon（macOS）、release 构建、单线程查询（FAISS 已 `omp_set_num_threads(1)` 对齐）。

### vdb.rs（IVF_RaBitQ，1 bit/dim）

| 配置 | nprobe | refine_k | recall@10 | QPS | p50(ms) | p99(ms) | build(ms) |
|------|--------|----------|-----------|-----|---------|---------|-----------|
| 全分区扫描 | 0（全部） | k×10 | 1.000 | 199 | 4.722 | 8.038 | 2803 |
| 速度优先 | 16 | k×10 | 0.053 | 3639 | 0.249 | 0.728 | 2698 |
| 平衡配置 | 16 | 5000 | 0.994 | 1400 | 0.661 | 1.619 | 2686 |
| 高召回配置 | 50 | 5000 | 0.969 | 440 | 2.152 | 4.059 | 2700 |
| 全分区精排 | 100 | k×10 | 1.000 | 203 | 4.749 | 6.599 | 2688 |

### 对标（相同数据、相同 nprobe）

| 引擎 | nprobe | recall@10 | QPS | p50(ms) | p99(ms) | build(ms) |
|------|--------|-----------|-----|---------|---------|-----------|
| FAISS IVF_FLAT | 16 | 0.997 | 16451 | 0.060 | 0.083 | 81 |
| FAISS IVF_FLAT | 50 | 1.000 | 6026 | 0.164 | 0.320 | 81 |
| FAISS IVF_FLAT | 100 | 1.000 | 2314 | 0.382 | 0.797 | 81 |
| FAISS IVF_PQ | 16 | 0.756 | 12721 | 0.068 | 0.262 | 730 |
| FAISS IVF_PQ | 50 | 0.757 | 6753 | 0.144 | 0.222 | 730 |
| FAISS IVF_PQ | 100 | 0.757 | 3252 | 0.268 | 0.733 | 730 |
| LanceDB IVF_PQ | 16 | 0.995 | 103 | 9.594 | 11.631 | 7776 |
| LanceDB IVF_PQ | 50 | 1.000 | 80 | 12.285 | 14.703 | 7776 |
| LanceDB IVF_PQ | 100 | 1.000 | 61 | 16.151 | 18.678 | 7776 |

### 结论

- **vdb.rs 在 nprobe=16、refine_k=5000 时即可达到 recall@10 ≈ 0.994，QPS ≈ 1400**；全分区扫描时 recall@10 = 1.0。
- 与 **FAISS IVF_PQ** 相比，vdb.rs 在相近 nprobe 下召回更高（0.994 vs 0.756 @ nprobe=16），且保持 1 bit/dim 压缩；QPS 低于 FAISS 的 C++ 实现，属于预期差距。
- 与 **LanceDB IVF_PQ** 相比，vdb.rs 在 Python 层客户端场景下 QPS 更高（1400 vs 103 @ nprobe=16）。
- 后续优化方向：batch 多线程查询、更稳定的 k-means 初始化、更激进的 FastScan 与 Query Quantization。

## 生产部署建议

- 维度必须满足 `dim % 64 == 0`（RaBitQ 量化要求）。
- 推荐 `num_partitions = sqrt(N)`，上限 65536，下限 4。
- 启用 refine 层保障召回率：`refine = true`，`refine_k = 10` 起；生产环境建议根据召回目标增大到 1000~5000。
- 速度优先：启用 FastScan（`fastscan = true`）。
- 精度优先：适当增大 `nprobe` 与 `refine_k`。
- NNG 模式适合低延迟内网调用；HTTP 模式适合调试与浏览器测试页面。
- 禁止在 <16GB 内存设备上一次性 mmap 超过 50% 物理内存；分块 mmap + 用户态 LRU（`MmapDatabase`）自动将缓存控制在预算内，实现零拷贝启动。

## 许可证

MIT OR Apache-2.0
