#!/usr/bin/env bash
# 性能测试 shell 脚本示例：编译 release 二进制后，依次运行自动调参、
# benchmark 二进制和 performance_matrix 示例，输出 CSV 与对比结果。
#
# 用法：
#   chmod +x examples/performance_benchmark.sh
#   ./examples/performance_benchmark.sh
#
# 可通过环境变量覆盖默认规模：
#   DIM=128 N=50000 K=10 QUERIES=200 ./examples/performance_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DIM="${DIM:-128}"
N="${N:-10000}"
K="${K:-10}"
QUERIES="${QUERIES:-100}"

echo "=== vdb.rs performance benchmark ==="
echo "DIM=${DIM} N=${N} K=${K} QUERIES=${QUERIES}"
echo

# 1. 构建 release 二进制
echo "[1/4] building release binaries..."
cargo build --release --bins --example performance_matrix

# 2. 自动调参
echo
./target/release/vdb tune --n "${N}" --k "${K}"

# 3. benchmark 二进制：多组配置对比
echo
./target/release/vdb-benchmark \
  --dim "${DIM}" \
  --n "${N}" \
  --k "${K}" \
  --queries "${QUERIES}" \
  --nprobe 16 \
  --refine-k "$((K * 10))"

./target/release/vdb-benchmark \
  --dim "${DIM}" \
  --n "${N}" \
  --k "${K}" \
  --queries "${QUERIES}" \
  --nprobe 50 \
  --refine-k 1000

./target/release/vdb-benchmark \
  --dim "${DIM}" \
  --n "${N}" \
  --k "${K}" \
  --queries "${QUERIES}" \
  --nprobe 100 \
  --refine-k 5000

# 4. performance_matrix 示例：CSV 风格矩阵
echo
echo "[4/4] running performance matrix..."
./target/release/examples/performance_matrix \
  --n "${N}" \
  --dim "${DIM}" \
  --k "${K}" \
  --queries "${QUERIES}"

echo
echo "=== benchmark complete ==="
