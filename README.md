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

# CLI：创建数据库 → 插入 → 搜索
cargo run --bin vdb -- create --dir ./data/mydb --dim 128
cargo run --bin vdb -- insert --dir ./data/mydb --vector "[0.1, 0.2, ...]" --payload '{"score":0.95}'
cargo run --bin vdb -- search --dir ./data/mydb --query "[0.1, 0.2, ...]" --k 10 --nprobe 50

# 自动获取基于系统资源与数据规模的推荐参数
cargo run --bin vdb -- tune --n 100000 --k 10

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

## 常见使用场景

下面给出五种典型使用方式的完整步骤、配置建议与命令。所有示例均为可运行代码，位于 `examples/` 目录。

### 1. 嵌入式模式（业务进程内直接调用）

适用于需要在业务服务内部直接嵌入向量检索能力的场景，无需独立进程。

```bash
cargo run --release --example embedded
```

该示例展示：
- 使用 `Database::create` 创建数据库；
- 使用 `insert_with_payload` 插入带标量 payload 的向量；
- 使用 `SearchOptions` 进行普通搜索、SQL 过滤搜索与高召回搜索；
- 通过 `db.stats()` 查看版本、向量数与分区数。

关键 API：

```rust
use vdb_rs::vdb::Database;
use vdb_rs::search::SearchOptions;
use vdb_rs::index_ivf_rq::Payload;

let db = Database::create("./data/mydb", 128)?;

let mut payload = Payload::new();
payload.insert("category".to_string(), serde_json::json!("news"));
payload.insert("score".to_string(), serde_json::json!(0.95));
let id = db.insert_with_payload(&vector, payload)?;

let opts = SearchOptions {
    k: 10,
    nprobe: 50,
    refine: true,
    refine_k: 1000,
    fastscan: true,
    query_bits: 0,
    sq8_refine: false,
    sql_filter: Some("score >= 0.9".to_string()),
};
let results = db.search(&query, &opts);
```

### 2. 分块 mmap 零拷贝启动

适用于内存受限、启动速度敏感、以只读为主的场景。`MmapDatabase` 仅加载元数据，查询时按需 fault 64MB 数据块，并通过用户态 LRU 控制总映射量。

```bash
cargo run --release --example mmap_zero_copy
```

要点：
- 写入阶段使用普通 `Database`；
- 读取阶段使用 `MmapDatabase::open` 零拷贝打开；
- 搜索接口为 `mmap_db.search(query, k, nprobe)`；
- 当前仅支持 Unix-like 系统。

```rust
use vdb_rs::vdb::{Database, MmapDatabase};

let db = Database::create("./data/mmap_db", 128)?;
for _ in 0..1000 { db.insert(&vector)?; }

let mmap_db = MmapDatabase::open("./data/mmap_db")?;
let results = mmap_db.search(&query, 10, 100);
```

### 3. HTTP Server 模式

提供 OpenAI/Anthropic 兼容的 JSON API，并内置 llama-server 风格的浏览器测试页面。

启动服务：

```bash
# 默认监听 127.0.0.1:8080
cargo run --release --bin vdb-server

# 自定义地址与维度
cargo run --release --bin vdb-server -- --listen 0.0.0.0:8080 --dim 128

# 设置服务器级默认搜索参数（请求体可继续覆盖）
cargo run --release --bin vdb-server -- \
  --listen 0.0.0.0:8080 --dim 128 \
  --default-nprobe 100 --default-refine-k 5000 \
  --default-query-bits 8
```

API 示例：

```bash
# 查看统计信息
curl http://127.0.0.1:8080/stats

# 单条插入（可附带 payload）
curl -X POST http://127.0.0.1:8080/insert \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.1, 0.2, ..., 0.128], "payload": {"tag": "demo"}}'

# 向量搜索（生产推荐配置）
curl -X POST http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": [0.1, 0.2, ..., 0.128],
    "k": 10,
    "nprobe": 50,
    "refine": true,
    "refine_k": 1000
  }'

# SQL 过滤搜索
curl -X POST http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": [0.1, 0.2, ..., 0.128],
    "k": 5,
    "nprobe": 0,
    "refine": true,
    "refine_k": 100,
    "sql_filter": "score >= 0.9"
  }'

# 批量插入
curl -X POST http://127.0.0.1:8080/batch_insert \
  -H 'Content-Type: application/json' \
  -d '{"vectors": [[0.1, ...], [0.2, ...]]}'

# 请求级覆盖搜索参数（fastscan / query_bits / sq8_refine）
curl -X POST http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": [0.1, 0.2, ..., 0.128],
    "k": 10,
    "nprobe": 50,
    "fastscan": true,
    "query_bits": 8,
    "sq8_refine": false
  }'
```

`/search` 支持字段：`query`、`k`、`nprobe`、`refine`、`refine_k`、`fastscan`、`query_bits`、`sq8_refine`、`sql_filter`。未提供的字段使用服务器启动时设置的默认值。

浏览器打开 `http://127.0.0.1:8080/` 即可使用测试页面（向量搜索、性能测试、数据管理）。

程序化启动 HTTP server 的示例：

```bash
cargo run --release --example server_http
```

### 4. NNG 二进制协议（低延迟生产接口）

NNG 模式使用原始 TCP 二进制协议，解析开销极小，适合低延迟内网调用。

启动服务：

```bash
cargo run --release --bin vdb-nng-server
# 默认监听 tcp://0.0.0.0:9090，维度 64

# 自定义监听地址与维度
cargo run --release --bin vdb-nng-server -- --listen 0.0.0.0:9091 --dim 128
```

协议格式：

```text
[4 bytes: message length][1 byte: command][payload]
```

支持的命令：

| 命令 | 编码 | payload 说明 |
|------|------|--------------|
| PING | 0x01 | 空 |
| SEARCH | 0x02 | `[4B k][4B nprobe][4B dim][dim×4B f32]` |
| BATCH_SEARCH | 0x03 | `[4B k][4B nprobe][4B dim][4B num_queries][num_queries×dim×4B f32]` |
| INSERT | 0x04 | `[4B dim][dim×4B f32]` |
| IMPORT_JSON | 0x05 | JSON 数组：`[[f32, ...], ...]` |
| EXPORT_JSON | 0x06 | 空 |

响应格式：`[4B length][1B code][data]`，`code=0x00` 成功，`0xFF` 错误。SEARCH 结果每个元素为 `[8B id][4B distance]`。

Rust 客户端示例（默认连接 127.0.0.1:9090、维度 64，与 `vdb-nng-server` 默认一致）：

```bash
# 默认维度 64（与 vdb-nng-server 默认一致）
cargo run --release --example nng_client

# 自定义地址与维度
VDB_NNG_ADDR=127.0.0.1:9091 VDB_NNG_DIM=128 cargo run --release --example nng_client
```

Python 最小客户端示例：

```python
import socket, struct

s = socket.create_connection(("127.0.0.1", 9090))
dim = 64  # 与 vdb-nng-server 默认维度一致
vec = [0.1] * dim
payload = struct.pack("<I", dim) + struct.pack(f"<{dim}f", *vec)
msg = struct.pack("<I", 1 + len(payload)) + b'\x04' + payload
s.sendall(msg)

resp_len = struct.unpack("<I", s.recv(4))[0]
resp = s.recv(resp_len)
print("code", resp[0], "id", struct.unpack("<Q", resp[1:9])[0])
```

### 5. 最佳性能配置

`examples/best_performance.rs` 在随机数据上对比不同 `nprobe`、`query_bits`、`refine_k` 组合的延迟、QPS 与召回率，帮助选择生产参数。

```bash
cargo run --release --example best_performance
```

`examples/performance_matrix.rs` 在 `best_performance` 基础上增加自动调参输出，并以 CSV 格式打印多组配置结果，便于直接复制到表格做延迟-召回权衡。

```bash
# 默认 10K × 128d
cargo run --release --example performance_matrix

# 自定义规模
cargo run --release --example performance_matrix -- --n 50000 --dim 128 --k 10 --queries 200
```

`examples/performance_benchmark.sh` 是 shell 脚本批量性能测试示例，自动构建 release 二进制并串联 `vdb tune`、`vdb-benchmark` 与 `performance_matrix`：

```bash
chmod +x examples/performance_benchmark.sh
./examples/performance_benchmark.sh

# 自定义规模
DIM=128 N=50000 K=10 QUERIES=200 ./examples/performance_benchmark.sh
```

输出示例：

```text
[matrix] recommended partitions: 316
[matrix] latency:  nprobe=16 refine_k=100 query_bits=8 fastscan=true
[matrix] balanced: nprobe=50 refine_k=1000 query_bits=0 fastscan=true
[matrix] recall:   nprobe=100 refine_k=5000 query_bits=0 fastscan=true

name,nprobe,refine_k,query_bits,fastscan,recall@k,qps,p50_ms
latency,16,100,8,true,0.3200,1247.9,0.801
balanced,50,1000,0,true,1.0000,2138.9,0.468
balanced+qq8,50,1000,8,true,1.0000,527.9,1.894
high-recall,100,5000,0,true,1.0000,2194.0,0.456
exact,0,50,0,true,1.0000,2109.5,0.474
```

典型调参组合：

| 场景 | nprobe | refine_k | query_bits | fastscan | 召回目标 |
|------|--------|----------|------------|----------|----------|
| 极速（低延迟） | 16 | k×10 | 8 | true | 中等 |
| 平衡（推荐） | 50 | 1000 | 0 / 8 | true | ≥ 0.95 |
| 高召回 | 100 | 5000 | 0 | true | ≥ 0.99 |
| 离线精确比对 | 0（全分区） | k×10 | 0 | true | 1.0 |

参数说明：

- `nprobe`：扫描分区数，`0` 表示全部分区；增大可提升召回，但增加延迟。
- `refine_k`：精排候选数，对粗排 TopK 用原始向量重新计算真实距离；显著影响召回。
- `fastscan`：批量 XOR-popcount，默认开启，QPS 提升 3.5x 以上。
- `query_bits`：Query Quantization 位数，`0` 禁用，`8` 在几乎无损召回下 QPS 提升约 2 倍。
- `sq8_refine`：使用 SQ8 码进行精排，适合需要减少内存带宽的场景。

### 6. CLI 命令行与自动调参

`vdb` 二进制提供数据库创建、插入、搜索与 `tune` 自动推荐参数，避免在代码中写死性能配置。

```bash
# 创建数据库
cargo run --bin vdb -- create --dir ./data/mydb --dim 128

# 插入带 payload 的向量
cargo run --bin vdb -- insert --dir ./data/mydb \
  --vector "[0.1, 0.2, ..., 0.128]" \
  --payload '{"category":"news","score":0.95}'

# 搜索：默认 nprobe=50 refine_k=1000 fastscan=true
cargo run --bin vdb -- search --dir ./data/mydb \
  --query "[0.1, 0.2, ..., 0.128]" \
  --k 10 --nprobe 100 --refine-k 5000

# SQL 过滤搜索
cargo run --bin vdb -- search --dir ./data/mydb \
  --query "[0.1, 0.2, ..., 0.128]" \
  --sql-filter "score >= 0.9"

# 自动根据数据规模与系统资源推荐参数
cargo run --bin vdb -- tune --n 100000 --k 10
```

`tune` 会输出 CPU 核心数、物理内存、推荐 mmap 缓存预算，以及极速/平衡/高召回三套参数组合。

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
