#!/usr/bin/env bash
# 真实数据集（siftsmall）性能测试示例。
#
# 目标配置（来自 README 实测）：
#   vdb.rs IVF_RaBitQ, nprobe=16, refine_k=5000
#   Recall@10 ≈ 0.994, QPS ≈ 1400, p50 ≈ 0.661 ms, build ≈ 2686 ms
#
# 用法：
#   chmod +x examples/siftsmall_benchmark.sh
#   ./examples/siftsmall_benchmark.sh
#
# 如果 siftsmall 不在默认路径，可覆盖前缀：
#   DATASET_PREFIX=/path/to/siftsmall/siftsmall ./examples/siftsmall_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_PREFIX="${DATASET_PREFIX:-../models/data/siftsmall/siftsmall}"
K="${K:-10}"

echo "=== vdb.rs siftsmall real-dataset benchmark ==="
echo "dataset prefix: ${DATASET_PREFIX}"
echo "k: ${K}"
echo

# 检查数据文件是否存在
for suffix in _base.fvecs _query.fvecs _groundtruth.ivecs; do
    file="${DATASET_PREFIX}${suffix}"
    if [[ ! -f "${file}" ]]; then
        echo "ERROR: missing dataset file: ${file}"
        echo "Download siftsmall from https://corpus-texmex.irisa.fr/ and place it at:"
        echo "  ${DATASET_PREFIX}_base.fvecs"
        echo "  ${DATASET_PREFIX}_query.fvecs"
        echo "  ${DATASET_PREFIX}_groundtruth.ivecs"
        exit 1
    fi
done

echo "[1/2] building release benchmark binary..."
cargo build --release --bin vdb-benchmark

echo
echo "[2/2] running benchmark matrix..."

# 配置矩阵：nprobe,refine_k,label
configs=(
    "0,50,exact"
    "16,50,latency"
    "16,5000,balanced-high-recall"
    "50,5000,high-recall"
    "100,5000,extreme-recall"
)

for cfg in "${configs[@]}"; do
    IFS=',' read -r nprobe refine_k label <<< "${cfg}"
    echo
    echo "--- config: ${label} (nprobe=${nprobe}, refine_k=${refine_k}) ---"
    ./target/release/vdb-benchmark \
        --dataset "${DATASET_PREFIX}" \
        --k "${K}" \
        --nprobe "${nprobe}" \
        --refine-k "${refine_k}"
done

echo
echo "=== siftsmall benchmark complete ==="
echo "Recommended production config: nprobe=16, refine_k=5000"
