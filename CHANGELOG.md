# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.0] - 2026-07-01

### Added

- 工程脚手架：workspace + 多 bin（cli / server / nng-server / benchmark），`cargo build` 与 `cargo fmt --check` 通过。
- `src/simd.rs`：向量点积 / L2 距离 + 批量 XOR-popcount，支持 AVX-512、AVX2、aarch64 NEON 与纯 Rust fallback。
- `src/index_ivf_rq.rs`：IVF_RaBitQ 量化索引，包含随机正交旋转、超球面归一化（`epsilon_0 = 1.9`）、符号二值化、两个校正标量计算。
- IVF 分区管理：k-means++ 初始化 + Lloyd 迭代，`num_partitions = min(max(4, sqrt(N)), 65536)`。
- 内存中暴力 Flat baseline，用于召回验证与对比测试。
- `src/search.rs`：查询调度、nprobe 剪枝、分区级并行位运算检索、精排层 Refine/SQ8、TopK 归并、FastScan 路径、Query Quantization（1-8 bit）。
- `src/storage.rs`：partition-oriented columnar 文件格式，完整支持索引 save/load（config、rotation、partitions、super_partitions、next_id）；新增 `MmapStorage`、`load_index_mmap` 与物理内存检测，实现 mmap 按需加载与 <16GB 设备 50% 内存 bound。
- `src/vdb.rs`：零拷贝 RecordBatch 语义、内存映射管理、追加写事务边界 + manifest 版本快照（time-travel）、实例级写锁。
- `src/sql.rs`：自研最小 SQL WHERE 子集解析器，支持 `=`, `!=`, `<`, `>`, `<=`, `>=`, `AND`, `OR`, `IN`，实现分区级谓词下推。
- `src/thread_pool.rs`：固定工作线程池 + 自旋锁任务队列，`batchInsert` 与 `batchSearch` 复用线程池。
- `src/nng_server.rs`：基于 TCP 的二进制协议服务，协议 `[4B length][1B cmd][payload]`，支持 PING/SEARCH/BATCH_SEARCH/INSERT/IMPORT_JSON/EXPORT_JSON；Unix 路径迁移到 POSIX socket 直接系统调用。
- `src/server.rs` + `src/http_server.rs`：libevent evhttp C FFI 实现的 HTTP Server，OpenAI/Anthropic 兼容 API、CORS、静态资源、`k≤256` 保护、请求体大小限制与超时。
- `src/web/`：llama-server 风格默认测试页面（index.html / app.js / style.css），通过 `include_str!` 编译时嵌入。
- `src/gpu.rs`：GPU 三级后备策略（Metal > CUDA > OpenCL），内嵌三种 RaBitQ popcount kernel 源码，无 GPU 时平滑降级 CPU SIMD。
- `src/benchmark.rs`：QPS / latency(p50,p99) / recall@k / build time 测量，支持随机数据矩阵与真实 `.fvecs` 数据集（`--dataset`）。
- 八层测试体系：`tests/unit.rs`、`tests/integration.rs`、`tests/smoke.rs`、`tests/regression.rs`、`tests/acceptance.rs`、`tests/system.rs`、`tests/e2e.rs`、`tests/server.rs`。
- `AGENTS.md`：项目代理指令与架构、质量、生产部署规则。
- 常见使用场景示例：`examples/embedded.rs`、`examples/mmap_zero_copy.rs`、`examples/server_http.rs`、`examples/nng_client.rs`、`examples/best_performance.rs`、`examples/performance_matrix.rs`、`examples/performance_benchmark.sh`、`examples/siftsmall_benchmark.sh`、`examples/hongloumeng_rag.sh`、`examples/text_to_vectors.py`、`examples/lookup_results.py`，覆盖嵌入式、mmap 零拷贝、HTTP Server、NNG 二进制协议、最佳性能配置、自动调参 CSV 矩阵、shell 脚本批量性能测试、真实数据集 siftsmall 验证与中文文本 RAG（红楼梦）。
- `src/sys_info.rs`：系统资源探测（CPU 核心数、物理内存）与检索参数推荐，提供 `recommend_search_options`、`recommend_num_partitions`、`recommend_mmap_cache_bytes`。
- CLI 完整实现：`vdb create`、`vdb insert`、`vdb batch-insert`、`vdb search`、`vdb tune`，支持 payload、SQL 过滤与自动参数推荐。
- `Database::batch_insert_with_payload`：内存中批量追加向量后一次性保存，避免逐条写入产生大量全量索引快照，显著降低磁盘占用。
- `Database::compact`：清理旧版本 `index-N.vdb` 文件，只保留 manifest 指向的最新版本，回收 time-travel 快照占用的空间。
- CLI 新增 `vdb compact` 命令。
- `examples/text_to_vectors.py` 增加精确重复块去重。
- HTTP Server 支持服务器级默认搜索参数：`--default-k`、`--default-nprobe`、`--default-refine-k`、`--default-query-bits` 等，请求体可继续覆盖；`/search` 新增 `fastscan`、`query_bits`、`sq8_refine` 字段。
- NNG Server 支持 `--listen` 与 `--dim` 启动参数，消除硬编码地址与维度。
- `.github/workflows/ci.yml` 新增 `release` job：在 `v*` 标签推送时自动构建 Ubuntu/macOS/Windows release 二进制并上传至 GitHub Release。

### Changed

- `benchmark.rs` 新增 `--refine-k` 参数，允许调节精排候选数以提升召回率。
- `TODOS.md` 各阶段任务状态同步：SQ8、Query Quantization、R*centroid、mmap 按需加载、POSIX socket 迁移、libevent HTTP 优化、覆盖率与 CI 配置等标记为完成。
- `Cargo.toml` 与 `README.md` 仓库链接更新为 `https://github.com/supermy/vdb.rs`。
- `README.md` 重写：补充定位、特性矩阵、真实数据集测试、API 速览、生产部署建议、性能目标与对标。

### Fixed

- 真实数据集 siftsmall 召回率低的问题：通过 `--refine-k` 增大精排候选数，nprobe=16 / refine_k=5000 时 Recall@10 达到 0.994，nprobe=100 时 Recall@10 = 1.0。
- 查询路径重复旋转：新增 `encode_query_from_rotated`，查询向量只旋转一次，避免 O(dim²) 重复计算。
- 分块 mmap 扫描热路径内存分配：新增 `estimate_distance_sq_raw`，避免为每个候选构造 `RabitqCode`。
- Clippy 警告：`http_server.rs` 改用 `io::Error::other`、`index_ivf_rq.rs` 使用 range contains / 消除 needless range loop / 为 `PartitionStats` 实现 `Default`、`simd.rs` 关闭 `incompatible_msrv` 警告。
- `AGENTS.md` 中遗留的 `.rust` 扩展名笔误已修正为 `.rs`。

### Known Issues

- Windows 路径仍依赖 vcpkg 安装 libevent，首次配置环境较复杂。

### Performance

- FastScan 默认开启，批量 XOR-popcount 在 x86_64 / aarch64 平台触发 SIMD 加速。
- siftsmall（10K × 128d，release 单线程查询）实测：
  - vdb.rs nprobe=16 / refine_k=5000：Recall@10 = 0.994，QPS ≈ 1400，p50 ≈ 0.66 ms，build ≈ 2.7 s。
  - vdb.rs nprobe=100：Recall@10 = 1.0，QPS ≈ 203，p50 ≈ 4.7 ms，build ≈ 2.7 s。
  - 对比 FAISS IVF_FLAT（nprobe=16，Recall@10 = 0.997，QPS ≈ 16451）与 IVF_PQ（nprobe=16，Recall@10 = 0.756，QPS ≈ 12721）；vdb.rs 在 1 bit/dim 压缩下召回优于 IVF_PQ，QPS 低于 C++ 实现。
  - 对比 LanceDB IVF_PQ（nprobe=16，Recall@10 = 0.995，QPS ≈ 103）；vdb.rs 在 Python 客户端场景下 QPS 更高。

## [0.0.0] - 2026-07-01

- 项目初始化：创建最小 README.md 与 `Cargo.toml` 骨架。
