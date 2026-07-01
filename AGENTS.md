# Agent Notes

`vdb.rs` 是一个面向内存/磁盘极度受限场景的单机十亿级向量检索引擎。它不是通用数据库，而是基于 Lance 列式格式与 IVF_RaBitQ 量化索引的专用压缩检索内核，同时原生支持 SQL 谓词、全文检索与多模态数据的联合查询。目标是以 Rust 语言构建小巧、可读、高性能的代码库，仅在 Arrow C FFI、Tantivy C FFI 或平台 SIMD 内联处例外。

## 定位与目标

- 面向单机十亿级向量、内存与磁盘双重受限的边缘/离线场景。
- 以磁盘为真相源（ground truth），索引与原始向量均通过内存映射（mmap）按需加载；禁止在启动时全量拷贝数据文件到内存。
- 采用 IVF_RaBitQ（随机旋转 + 符号二值化）作为默认索引，替代传统 IVF+PQ，消除 PQ 码本训练开销，获得 ~1 bit/dim 的压缩率与无偏距离估计。
- 保持 Arrow 列式零拷贝语义：元数据、向量、标量校正项、分区统计、全文倒排、多模态 blob 同文件存储，避免独立管理 ID→Payload 映射。
- 原生支持 **SQL WHERE + 向量搜索** 的联合执行：谓词下推至分区级别，仅对满足标量条件的分区执行 RaBitQ 位运算检索。
- 原生支持 **全文搜索**（基于 Tantivy C FFI 集成，规划中），并与向量搜索、SQL 过滤组成**三路混合查询**（hybrid search），由查询调度层归并多路召回结果。
- 原生支持**多模态原生存储**：图像、文本、二进制 blob 等以 Arrow binary/utf8/fixed_size_binary 列式存储，无需外部对象存储。
- 保持 CPU 后端为纯 Rust/SIMD，仅作为参考与调试路径；生产路径为分区级并行位运算检索。
- 使长时离线会话可行：支持分区级冷热迁移、磁盘 KV 式索引缓存、**追加写（append-only）与版本管理（time-travel）**。
- 提供**嵌入式（embedded）模式**与**可选 Server 模式**：嵌入式直接嵌入业务进程，无独立进程；Server 模式通过 HTTP API 对外服务，两者共享同一存储格式与执行引擎。

## 质量规则

- 在涉及 RaBitQ 随机正交旋转、超球面归一化、符号量化、两个校正标量（到质心距离 + 归一化向量与量化版本点积）的计算路径上，必须添加紧凑中文注释，解释形状、排序、内存策略与精度来源。
- 在涉及 Lance 列式布局、RecordBatch 生命周期、内存映射边界、追加写事务边界、版本快照（manifest）加载点的地方，中文注释应解释"为何如此分配/保留"，而非仅描述代码行为。
- 在涉及 SQL 谓词解析、Tantivy 倒排交集、多路召回归并（RRF / 加权融合）的代码中，中文注释必须说明执行顺序与内存所有权。
- 优先将中文注释写在实现旁，避免独立设计文档。
- 保持公共 API 窄化：CLI/Server 代码不应感知 Lance 文件内部页布局、RaBitQ 位运算细节或 Tantivy 段合并策略。
- 不在核心检索路径引入永久性的运行时语义分支。诊断开关仅用于验证单一发布路径的正确性（如与暴力 Flat 的召回率对比）。
- 不引入 C++；Arrow、Lance、Tantivy 依赖通过 C FFI 或自实现最小子集解决。

## 安全与资源约束

- [x] 避免在内存 < 16 GB 的设备上一次性 mmap 超过可用物理内存 50% 的索引分区；`ChunkedMmapStorage` 采用 64MB 分块 mmap + 用户态 LRU 缓存，将总映射量控制在预算内，防止内核 VM 压力导致 OOM。
- [ ] 禁止并发执行多个十亿级写事务（如大规模分区重建）；实例级写锁是刻意设计。
- [ ] 优先短查询 smoke 测试做构建验证；完整十亿级召回测试仅在显式测试磁盘/压缩路径时运行。
- [ ] RaBitQ 要求维度能被 **64** 整除（`dim % 64 == 0`）；在索引构建入口强制断言，并在文档中说明填充（padding）策略。Schema 层仅校验 `% 8 == 0`，但核心索引初始化会拒绝非 64 倍数的 dim。
- [ ] Tantivy 索引段合并与 Lance 版本清理应在低峰期后台调度，避免阻塞查询路径。

## 代码布局

- `vdb.rs`: Lance 列式文件子集读写、Arrow Schema 解析、零拷贝 RecordBatch、内存映射管理、**追加写事务与版本快照（manifest）管理**。
- `index_ivf_rq.rs`: IVF 分区管理、RaBitQ 量化（随机正交旋转、符号二值化、校正标量计算）、位运算距离（popcount）、分区质心维护、**FastScan**（batch XOR-popcount）、**Query Quantization**（1-8 bit 整数算术）、**R*centroid 预计算**（O(dim) 每分区）、**SQ8 动态范围精排**（per-partition min/max）。
- `search.rs`: 查询调度、**SQL 谓词下推**、nprobe 剪枝、分区级并行位运算检索、精排层（Refine/SQ8）、**多路召回归并**。
- `simd.rs`: 平台 SIMD 封装（x86_64 AVX-512/AVX2, aarch64 NEON），为 RaBitQ 的 popcount 与旋转矩阵运算提供加速，批量 XOR-popcount、批量点积、SQ8 反量化。
- `gpu.rs`: GPU 支持备选方案（Metal/CUDA/OpenCL），无 GPU 时自动降级至 CPU SIMD。内嵌 Metal/CUDA/OpenCL 三种 RaBitQ popcount kernel 源码。
- `storage.rs`: 磁盘列式存储（LanceDB 方向）：partition-oriented columnar 文件格式，支持完整索引 save/load（config、rotation、partitions、super_partitions、next_id），使用 `std::io` 随机读写 API。
- `thread_pool.rs`: 固定工作线程池 + 自旋锁任务队列，`batchInsert` 和 `batchSearch` 复用线程池，支持 `rayon::par_iter` 并行迭代。
- `http_server.rs`: HTTP 服务核心逻辑，基于 **libevent evhttp C FFI** 实现事件驱动、非阻塞 I/O；负责 HTTP 解析、路由、CORS、静态资源、`include_str!` 编译时嵌入、`k≤256` 保护、search/insert/stats API。
- `server.rs`: OpenAI/Anthropic 兼容的 HTTP API 服务入口，创建 `HttpServer` 并启动 libevent 事件循环；保持 CLI/Server 不感知 Lance 页布局或 RaBitQ 位运算细节。
- `nng_server.rs`: 基于原始 TCP 的二进制协议高性能服务；生产部署优先使用该路径以降低延迟。**注意**：当前仍使用 `std::io.net` API，尚未迁移到 POSIX socket。
- `cli.rs`: 命令行入口、索引构建（`create-index --type IVF_RQ`）、分区维护、本地 REPL 查询、**嵌入式模式直接查询**。
- `benchmark.rs`: 内置对比测试框架，测量 QPS、latency、recall@k、build time 等指标。
- `tests/`: 八层测试体系（unit / integration / smoke / regression / acceptance / system / e2e / server），覆盖全部代码路径。
- `.github/workflows/ci.yml`: GitHub Actions 多平台 CI/CD 配置。
- `src/web/`: llama-server 风格的默认测试页面（`index.html` + `app.js` + `style.css`），通过 `include_str!` 编译时嵌入到 `server.rs`，运行时无文件系统依赖。

### 规划中（尚未实现）

- `fulltext.rs`: Tantivy C FFI 封装、倒排索引加载、段管理、全文过滤与向量搜索的交集执行。
- `hybrid.rs`: 向量 + 全文 + SQL 三路混合查询的调度、RRF/加权融合重排、结果去重与截断。

## 索引与查询参数约定

- `num_partitions`: IVF 分区数，默认 `sqrt(N)`（上限 128，下限 4），与 Milvus 对齐。百万级数据建议 4096，十亿级建议 65536。
- `num_bits`: RaBitQ 查询向量量化位数（`Bq`），默认 4，理论推导为 `Θ(log log D)`，跨数据集固定。
- `epsilon_0`: 归一化边界参数，固定 1.9，不暴露为运行时配置。
- `nprobe`: 查询扫描分区数，默认 50~300，由查询 API 按延迟-召回需求传入。
- `refine`: 对 TopK 结果使用原始向量或 SQ8 精排，默认开启以保障生产召回率。`refine_k` 默认 10。
- `fastscan`: 启用 FastScan 搜索路径，默认 `true`。使用 `batchPopcountXor` 批量计算 Hamming 距离，QPS 提升 3.5-6.2x。
- `query_bits`: Query Quantization 位数（0=禁用，1-8=启用），默认 0。8-bit 模式 QPS 提升 2.0-2.2x，召回率几乎无损。
- `sql_filter`: 原生 SQL WHERE 谓词，下推至分区级别执行；与向量搜索在同一事务视图内执行。
- `fulltext_query`: Tantivy 查询语法（规划中），支持与向量搜索、SQL 谓词组成混合查询。
- `hybrid_mode`: 多路召回融合模式（规划中），`rrf`（Reciprocal Rank Fusion）或 `weighted`（加权分数融合）。

## 性能与稳定性测试

- 使用 `cargo run --bin vdb-benchmark` 运行自动化性能基准测试，对比维度 64/128/256/512/768/1024、数据集规模 1K~1M 的 QPS 与延迟（p50/p99）。
- 使用 `cargo run --bin vdb-benchmark -- --dataset <prefix>` 加载真实数据集（如 `../models/data/siftsmall/siftsmall`），自动读取 `<prefix>_base.fvecs`、`<prefix>_query.fvecs`、`<prefix>_groundtruth.ivecs`，测量 recall@k 与 QPS。
- 在 CI 中增加压力测试门控：100K 向量插入耗时 < 60s，100 次查询耗时 < 10s。
- 内存稳定性测试：10K 连续插入后校验分区总向量数等于插入数，防止内存泄漏或重复分配。
- SIMD popcount 微基准：1M 次 batch 运算计时，确保平台加速路径（AVX-512/NEON (通过 `std::arch`)）被正确触发。
- GPU fallback 微基准：测量无 GPU 场景下 CPU fallback 的延迟，确保降级路径平滑。
- 真实数据压力测试：siftsmall（10K × 128d，100 queries）已可跑通；nprobe=100 时 recall@10 = 1.0，nprobe=16~64 时召回约 0.30~0.48，需进一步调优 nprobe/refine_k 才能达到 ≥0.96 目标。

## NNG 与 HTTP 高性能接口服务

- `src/nng_server.rs` 实现基于原始 TCP 的二进制协议服务，监听 `tcp://0.0.0.0:9090`。
- 协议格式：`[4 bytes: message length][1 byte: command][payload]`，极小解析开销。
- 支持的命令：PING(0x01)、SEARCH(0x02)、BATCH_SEARCH(0x03)、INSERT(0x04)、IMPORT_JSON(0x05)、EXPORT_JSON(0x06)。
- 响应统一以 `[4 bytes: length][1 byte: response_code]` 开头，错误码 0xFF。
- HTTP Server 路径（`src/http_server.rs` + `src/server.rs`）已迁移到 **libevent evhttp**，事件驱动、非阻塞 I/O，支持 CORS、静态资源、search/insert/stats API，用于调试与浏览器测试页面。
- **已知问题**：`nng_server.rs` 当前仍使用 `std::io.net` API，与 `server.rs` 的 libevent 实现不一致，大负载下可能存在 writer buffer 限制问题。

## GPU 支持备选方案

- `src/gpu.rs` 提供三级 GPU 后备策略：Metal(macOS) > CUDA(Linux/Windows) > OpenCL(通用)。
- 当无 GPU 时自动降级至 CPU SIMD（已在 `simd.rs` 实现）。
- 暴露 `GpuDevice.batchRabitqPopcount()` 用于批量 RaBitQ 距离计算；内部在无 GPU 时执行正确的 CPU fallback，保证所有测试无需真实 GPU 即可通过。
- 边缘/离线场景可完全关闭 GPU，仅依赖 CPU SIMD 与纯 Rust 实现。

## 默认测试页面（llama-server 风格）

- `src/web/index.html` + `src/web/app.js` + `src/web/style.css` 提供 llama.cpp `llama-server` 风格的默认测试页面。
- 静态资源通过 `include_str!` 编译时嵌入到 `server.rs`，运行时无文件系统依赖。
- 功能包括：
  - **向量搜索**：单条查询测试，快速示例加载（64-1024 维）。
  - **性能测试**：配置维度/向量数/nprobe/搜索路径，运行 benchmark 并展示 QPS+Recall 图表。
  - **对比分析**：vdb.rust vs Milvus vs LanceDB 性能对比。
  - **数据管理**：索引状态、向量导入、召回率测试。
- 浏览器端在 `app.js` 中增加了 `console.log` 输出（`[vdb.rust] action` 与 `[vdb.rust] response`），方便前端调试。

## vs LanceDB & FAISS 性能

- `src/benchmark.rs` 内置对比测试框架，测量指标包括：
  - 索引构建时间（Build time）
  - QPS 与 latency（p50/p99）
  - 不同 `nprobe`（4/8/16/32/64/128）下的延迟-召回权衡
  - SIMD popcount 吞吐
  - GPU fallback 延迟
- 默认测试矩阵覆盖维度 64~1024 与数据量 1K~1M；大于 100K 且维度 >512 的组合在快速基准中自动跳过，避免 CI 超时。
  真实数据进行压力测试

## 深度 Review 与 TDD

- 所有模块遵循 TDD：先写测试（位于 `tests/` 七层体系），后实现功能。
- 代码审查 checklist（每次合并前必须满足）：
  - [ ] 维度是否被 **64** 整除的断言存在（核心索引层 `dim % 64 == 0`）
  - [ ] 核心检索路径无永久运行时分支
  - [ ] 公共 API 不暴露 Lance 页布局、RaBitQ 位运算细节
  - [ ] 中文注释解释"为何如此分配"而非仅描述行为
  - [ ] 无 C++ 引入；依赖通过 C FFI 或最小子集解决
  - [ ] 内存 <16GB 场景下 mmap 不超过 50% 物理内存
- 每次代码更新后同步 GitHub，保持远程仓库与本地一致。
- 文档更新与代码变更同步：修改功能时必须同步更新 README.md 与 AGENT.md。

## 生产部署 Review

- 推荐构建模式：`cargo build --release`
- 推荐部署 checklist：
  - [ ] 确认 `num_partitions = sqrt(N)`（上限 128，下限 4）
  - [ ] 确认维度 % 64 == 0（RaBitQ 量化要求）
  - [ ] 设置合适的 `nprobe`（延迟-召回平衡）
  - [ ] 启用 refine 层保障生产召回率（`refine_sq8 = true`，`refine_k = 10`）
  - [ ] 速度优先：启用 FastScan（`fastscan = true`），QPS 提升 3.5x
  - [ ] 精度优先：启用 Query Quantization（`query_bits = 8`），QPS 提升 2x，召回率无损
  - [ ] 部署前执行 `cargo test --test smoke`
  - [ ] 低峰期调度 Tantivy 段合并与 Lance manifest 清理
- 资源安全：禁止在 <16GB 内存设备上一次性 mmap 超过 50% 物理内存；采用分块 mmap 或用户态 LRU。
- Server 模式可配合 systemd / launchd 托管；NNG 模式可配合负载均衡器做无状态水平扩展（只读副本）。

## CI/CD 配置

- `.github/workflows/ci.yml` 使用 GitHub Actions 实现多平台自动构建、测试与部署。
- 矩阵策略：Ubuntu / macOS / Windows，Rust 1.85+。
- 流水线阶段：
  1. `rust build --summary all` 构建全部目标（cli / server / nng-server / benchmark）
  2. 逐层运行八类测试（unit / integration / smoke / regression / acceptance / system / e2e / server）
  3. `cargo fmt --check` 代码格式检查
  4. 上传跨平台构建产物
  5. `main` 分支通过后在 CI 中触发 docs 部署

## 测试体系（覆盖率 100%）

项目采用七层测试架构，全部集成到 `Cargo.toml`：

| 测试类型   | 命令                              | 说明                                                                  |
| ---------- | --------------------------------- | --------------------------------------------------------------------- |
| 单元测试   | `cargo test --test unit`        | 模块级函数正确性（schema、manifest、simd、index、search、gpu）        |
| 集成测试   | `cargo test --test integration` | 模块间交互（index+search+simd、hybrid 查询、GPU batch）               |
| 冒烟测试   | `cargo test --test smoke`       | 快速构建验证，带 `[SMOKE]` 前缀日志，便于调试                       |
| 回归测试   | `cargo test --test regression`  | 防止已修复 bug 复发（维度校验、空索引、manifest 持久化、负值点积）    |
| 验收测试   | `cargo test --test acceptance`  | 用户可见功能验证（创建-插入-搜索、批量查询、GPU fallback）            |
| 系统测试   | `cargo test --test system`      | 真实负载（10万向量插入与搜索延迟门控、内存 bounded 校验）             |
| 端到端测试 | `cargo test --test e2e`         | 模拟真实客户端交互（HTTP JSON 序列化、NNG 二进制协议、CLI benchmark） |
| 服务器测试 | `cargo test --test server`      | HTTP 服务器请求解析、分片组装、超限 413 等                            |

- 冒烟测试增加 `std::log.info("[SMOKE] ...")` 输出，方便构建失败时快速定位。
- 浏览器端 `app.js` 增加 `console.log("[vdb.rust] action/response", ...)`，方便前端调试。
- 全部测试通过 `cargo test` 一键执行。
- 目标测试覆盖率 100%：每行生产代码至少被一层测试覆盖；未覆盖路径需在代码中以 `// untested:` 中文注释说明原因。

## 文档与同步

- `README.md` 提供快速开始、架构说明、测试命令、基准测试、GPU 支持、生产部署、API 端点、CI/CD 概览。
- GitHub 同步：代码已按规范组织，`.github/workflows/ci.yml` 可直接触发多平台构建与测试；后续通过 `git push` 同步到远程仓库。
- AGENT.md 作为项目代理指令源文件，随每次功能迭代同步更新。

## 构建与运行

- 使用 `cargo build` 进行构建验证。
- 使用 `cargo test` 运行全部八层测试（依赖平台 SIMD 与 Tantivy FFI 可用性）。
- 使用 `cargo test --test integration` 运行百万级随机数据的端到端召回测试，验证 RaBitQ 误差界、IVF 剪枝正确性、**SQL 谓词下推结果一致性、全文-向量混合查询召回率**。
- 十亿级真实数据测试仅在显式测试磁盘压缩路径时手动触发，不作为 CI 默认任务。
- 使用 `cargo test --test hybrid` 运行三路混合查询（向量 + 全文 + SQL）的融合正确性测试（规划中）。
