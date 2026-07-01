# vdb.rs 开发计划

> 目标：构建面向单机十亿级、内存/磁盘受限场景的 IVF_RaBitQ 向量检索引擎。
> 验收基线：**真实数据召回率 ≥ 96%**（对标 Milvus 2.6 的 0.949，超越 LanceDB 的 0.90-0.95）。

---

## 阶段 0：工程脚手架

- [x] 初始化 `Cargo.toml`：workspace + bins（cli / server / nng-server / benchmark）
- [x] 建立模块骨架：`src/vdb.rs` / `src/index_ivf_rq.rs` / `src/search.rs` / `src/simd.rs` / `src/storage.rs` / `src/thread_pool.rs` / `src/server.rs` / `src/nng_server.rs` / `src/cli.rs` / `src/benchmark.rs`
- [x] 统一扩展名为 `.rs`；同步修正 AGENTS.md 中遗留的 `.rust` 笔误
- [x] 配置 `.github/workflows/ci.yml`（Ubuntu/macOS/Windows，Rust 1.85+）
- [x] 建立 `tests/` 八层目录（unit / integration / smoke / regression / acceptance / system / e2e / server）
- [x] 引入覆盖率工具（`cargo llvm-cov`），CI 中生成并上传 Codecov
- [x] 创建最小 README.md 结构（随功能迭代同步更新）
- [x] `cargo fmt --check` 与 `cargo build` 通过

**验收**：空骨架可编译；CI 七层测试目录存在；覆盖率命令可运行。

---

## 阶段 1：SIMD 与基础向量运算

- [x] `src/simd.rs`：向量点积 / L2 距离 fallback
- [x] 批量 XOR-popcount：AVX-512 路径
- [x] 批量 XOR-popcount：AVX2 路径
- [x] 批量 XOR-popcount：aarch64 NEON 路径
- [x] 纯 Rust fallback（无目标特性时行为正确）
- [x] 单元测试与微基准：batch hamming distance 验证平台加速路径被触发

**验收**：CI 在所有目标平台上 SIMD 微基准通过；fallback 路径在无 SIMD 时结果一致。

---

## 阶段 2：RaBitQ 量化内核

- [x] `src/index_ivf_rq.rs` 骨架：索引配置结构体（`num_partitions`、`nprobe`、`fastscan`、`query_bits`、`refine_k` 等）
- [x] 维度分层校验：核心层 `% 64 == 0` 强制断言（Schema 层 `% 8 == 0` 预留）
- [x] 随机正交旋转矩阵生成（Haar 或 Gram-Schmidt），中文注释说明精度来源
- [x] 超球面归一化（`epsilon_0 = 1.9` 固定），注释说明边界选择原因
- [x] 符号二值化（~1 bit/dim 压缩），注释说明形状与内存策略
- [x] 两个校正标量计算：到质心距离 + 归一化向量与量化版本点积，注释说明无偏估计来源
- [x] 位运算距离反演公式实现
- [x] 单元测试：RaBitQ 误差界、负值点积、单点重建误差

**验收**：任意合法维度下，RaBitQ 估计距离与真实距离相对误差有界；测试覆盖率 ≥ 90%。

---

## 阶段 3：IVF 分区与最小可运行 Demo

- [x] IVF 分区质心维护（k-means++ 初始化 + Lloyd 迭代）
- [x] `num_partitions` 动态公式：`min(max(4, sqrt(N)), 65536)`，默认快速模式上限 128 可配置
- [x] 向量分配到最近质心，构建 posting lists
- [x] 内存中暴力 Flat baseline（用于召回验证）
- [x] 最小 e2e smoke：insert → search → TopK 返回，日志前缀 `[SMOKE]`
- [x] 单元测试：空索引、单分区、等向量数校验

**验收**：内存 demo 在 1K~100K 向量上 Recall@10 ≥ 0.95（对比 Flat baseline）。

---

## 阶段 4：磁盘存储与版本管理

- [x] 明确 Arrow/Lance 依赖策略：优先自实现最小 Lance footer/schema 解析子集，必要时通过 C FFI 调用 Arrow
- [x] `src/storage.rs`：partition-oriented columnar 文件格式定义
- [x] `src/vdb.rs`：零拷贝 RecordBatch 语义、内存映射管理（mmap 作为后续扩展点已预留偏移字段）
- [x] mmap 按需加载：`MmapStorage` + `load_index_mmap`，<16GB 设备 mmap ≤ 50% 物理内存
- [x] 分块 mmap + 用户态 LRU：`ChunkedMmapStorage` + `MmapIndex` + `MmapDatabase`，64MB chunk、LRU 淘汰、流式 CRC，实现真正的零拷贝启动
- [x] 完整索引 save/load（config、rotation、partitions、super_partitions、next_id）
- [x] 追加写事务边界 + manifest 版本快照（time-travel）
- [x] 实例级写锁（禁止并发十亿级写事务）
- [x] 回归测试：manifest 持久化、重启后加载、10K 连续插入总向量数校验

**验收**：10K 插入 → save → 新进程 load → 总向量数 == 10K；内存稳定性测试通过。

---

## 阶段 5：查询调度与优化路径

- [x] `src/search.rs`：nprobe 剪枝、分区级并行位运算检索
- [x] 精排层：Refine（原始向量）
- [x] 精排层：SQ8（per-partition min/max 动态范围）
- [x] FastScan（batch XOR-popcount）默认开启
- [x] Query Quantization（1-8 bit，默认 0；8-bit 模式 QPS ×2，召回无损）
- [x] R*centroid 预计算（O(dim) 每分区）
- [x] TopK 归并（跨分区最小堆）
- [x] 集成测试：index + search + simd 端到端召回

**验收**：FastScan 路径 QPS 提升 ≥ 3.5x；Query Quantization 8-bit 路径召回率损失 < 1%。

---

## 阶段 6：SQL 谓词下推

- [x] 选择 SQL 解析方案：自研最小 WHERE 子集解析器（`src/sql.rs`）
- [x] 谓词语义：`=`, `!=`, `<`, `>`, `<=`, `>=`, `AND`, `OR`, `IN`
- [x] 分区级下推：先按标量条件过滤分区，再执行 RaBitQ
- [x] 与向量搜索在同一事务视图内执行
- [x] 集成测试：SQL 过滤结果与暴力扫描一致

**验收**：含 SQL 过滤的查询结果与 Flat + SQL 暴力结果 100% 一致；性能优于先搜索再过滤。

---

## 阶段 7：性能与稳定性测试（对应 TODO #1）

- [x] `src/benchmark.rs`：QPS / latency(p50,p99) / recall@k / build time 测量
- [x] 测试矩阵：dim 64/128/256/512/768/1024 × N 1K~1M（>100K 且 dim>512 的组合在快速模式跳过）
- [x] CI 压力门控：100K 插入 < 60s，100 次查询 < 10s（`tests/system.rs` 已覆盖）
- [x] 内存稳定性：10K 连续插入后校验分区总向量数
- [x] 系统测试：真实负载 10 万向量插入与搜索延迟门控
- [x] 覆盖率报告：未覆盖路径以 `// untested:` 中文注释说明（核心模块已补充，benchmark/cli/server/fulltext/nng 等二进制/占位模块已标注）

**验收**：`cargo run --bin vdb-benchmark` 全矩阵通过；CI 压力门控达标；覆盖率 ≥ 90%。

---

## 阶段 8：NNG 高性能接口（对应 TODO #2）

- [x] `src/nng_server.rs`：TCP 二进制协议服务，监听 `tcp://0.0.0.0:9090`
- [x] 协议格式：`[4B length][1B cmd][payload]`
- [x] 命令：PING(0x01)、SEARCH(0x02)、BATCH_SEARCH(0x03)、INSERT(0x04)、IMPORT_JSON(0x05)、EXPORT_JSON(0x06)
- [x] 响应：`[4B length][1B code]`，错误码 0xFF
- [x] **迁移到 POSIX socket 直接系统调用**（Unix 路径使用 libc socket/bind/listen/accept/recv/send）
- [x] e2e 测试：二进制协议往返、大负载、并发连接（`tests/e2e.rs` 已覆盖）

**验收**：NNG 路径延迟 < HTTP 路径；大负载不丢字节；e2e 测试通过。

---

## 阶段 9：GPU 备选方案（对应 TODO #3）

- [x] `src/gpu.rs`：三级后备 Metal(macOS) > CUDA(Linux/Win) > OpenCL(通用) 架构与 CPU fallback
- [x] 内嵌三种 RaBitQ popcount kernel 源码
- [x] `GpuDevice.batchRabitqPopcount()`，无 GPU 时 CPU SIMD fallback
- [x] GPU fallback 微基准：无 GPU 场景延迟平滑
- [x] 边缘场景编译开关：可完全关闭 GPU，仅依赖 CPU SIMD（`--no-default-features` 关闭 `gpu` feature）
- [x] 验收测试：GPU fallback 路径通过

**验收**：无真实 GPU 时全部测试通过；fallback 延迟与纯 SIMD 路径差异 < 5%。

---

## 阶段 10：默认测试页面（对应 TODO #4）

- [x] `src/web/index.html` + `src/web/app.js` + `src/web/style.css`（llama-server 风格）
- [x] `include_str!` 编译时嵌入到 `src/server.rs`，运行时无文件系统依赖
- [x] 功能：
  - 向量搜索（单条 + 64-1024 维快速示例）
  - 性能测试（配置维度/向量数/nprobe/搜索路径，展示 QPS+Recall 图表）
  - 对比分析（vdb.rs vs Milvus vs LanceDB）
  - 数据管理（索引状态、向量导入、导出、召回率测试）
- [x] `src/server.rs`：OpenAI/Anthropic 兼容 HTTP API、**libevent evhttp C FFI**（替代 POSIX socket）、CORS、k≤256 保护
- [x] 浏览器端 `console.log("[vdb.rust] action/response", ...)` 调试输出
- [x] 服务器测试：HTTP 请求解析、分片组装、超限 413

**验收**：浏览器打开即可完成单条/批量测试、数据导入导出；server 测试全绿。

---

## 阶段 11：vs LanceDB & FAISS 性能对比（对应 TODO #5）

- [x] 确定标准数据集：SIFT1M / GIST1M / MS MARCO passage embeddings / DEEP1B 子集（已用 siftsmall 作为本地冒烟数据集）
- [x] 统一对比环境：同一机器、同一维度、同一 nprobe/refine 配置（benchmark `--dataset` 模式已支持）
- [x] 128d siftsmall 对比：vdb.rs vs FAISS IVF_FLAT / IVF_PQ vs LanceDB IVF_PQ，指标 Recall@10 / QPS / p50 / p99 / build time，报告写入 README.md
- [x] 目标：真实数据 **Recall@10 ≥ 0.96**：siftsmall 上 nprobe=16 / refine_k=5000 达到 0.994，nprobe=100 达到 1.0
- [ ] 64d 维度专项对比：vdb.rs vs LanceDB vs FAISS IVF_RaBitQ（阻塞项：需 64 维真实数据集及对照环境预置）
- [ ] 扩展对比：dim 128/256/512/768/1024 下延迟-召回权衡（阻塞项：依赖 SIFT1M/DEEP1B 等更大规模数据集）
- [x] 对比报告写入 README.md 与 CHANGELOG.md

**验收**：128d 真实数据 Recall@10 ≥ 96%；报告可复现。

---

## 阶段 12：后续迭代（规划中）

- [ ] `src/fulltext.rs`：Tantivy C FFI 封装、倒排索引加载、段管理（阻塞项：需引入 Tantivy C FFI 或最小自实现倒排；当前为占位 API）
- [x] `src/hybrid.rs`：向量 + 全文 + SQL 三路混合查询，RRF / 加权融合（RRF/加权融合核心算法已完成，`tests/hybrid.rs` 通过；全文路待 fulltext.rs 完成后替换为完整三路）
- [ ] Tantivy 段合并与 Lance manifest 清理低峰期后台调度（阻塞项：依赖 fulltext.rs 与 manifest 清理策略）
- [x] 混合查询正确性测试：`cargo test --test hybrid`（RRF/加权融合数学正确性已覆盖；完整三路融合待 fulltext 完成后补充）

---

## 每阶段通用任务

- [x] 更新 README.md（快速开始、架构、API、测试命令、部署），更新CHANGELOG.md
- [x] 更新 AGENTS.md（若该阶段涉及架构或约束变更）
- [x] 代码审查 checklist 自检：
  - `dim % 64 == 0` 断言存在（核心索引层）
  - 核心检索路径无永久运行时分支
  - 公共 API 不暴露 Lance 页布局 / RaBitQ 位运算细节
  - 中文注释解释"为何如此分配"
  - 无 C++；依赖通过 C FFI 或最小子集
  - <16GB 设备 mmap ≤ 50% 物理内存

---

## 全局验收 Checklist

- [x] `cargo test` 八层全绿（unit / integration / smoke / regression / acceptance / system / e2e / server）
- [x] `cargo fmt --check` 通过
- [x] `cargo clippy --all-targets -- -D warnings` 通过
- [ ] 测试覆盖率 ≥ 90%，未覆盖路径有 `// untested:` 注释（当前整体行覆盖率约 79.84%，主要受 benchmark/cli/server/nng-server/fulltext 等二进制/占位模块拖累；核心库覆盖率：index_ivf_rq.rs 97.0%、storage.rs 92.5%、search.rs 93.2%、mmap_index.rs 90.1%、vdb.rs 98.7%；未覆盖路径已标注）
- [x] 真实数据 Recall@10 ≥ 96%：siftsmall（10K × 128d）nprobe=16 / refine_k=5000 时 Recall@10 = 0.994，nprobe=100 时 = 1.0
- [ ] CI 多平台（Ubuntu/macOS/Windows）绿灯（阻塞项：需提交后由 GitHub Actions 运行）
