#!/usr/bin/env bash
# 《红楼梦》语义检索完整示例：文本 → 向量 → vdb.rs 索引 → 搜索。
#
# 目标：验证 vdb.rs 在真实中文文本 RAG 场景下的召回与延迟。
#
# 依赖：
#   - llama-embedding (llama.cpp)
#   - python3
#   - vdb.rs release 二进制（脚本会自动构建）
#
# 用法：
#   chmod +x examples/hongloumeng_rag.sh
#   ./examples/hongloumeng_rag.sh
#
# 环境变量覆盖：
#   INPUT_TXT=../models/data/txt/红楼梦.txt \
#   EMBED_MODEL=../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
#   CHUNK_SIZE=256 \
#   CHUNK_OVERLAP=32 \
#   LIMIT=200 \
#   ./examples/hongloumeng_rag.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

INPUT_TXT="${INPUT_TXT:-../models/data/txt/红楼梦.txt}"
EMBED_MODEL="${EMBED_MODEL:-../models/Qwen3-Embedding-0.6B-Q8_0.gguf}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-32}"
LIMIT="${LIMIT:-20}"
DB_DIR="${DB_DIR:-./data/hongloumeng_db}"
VECTOR_FILE="${VECTOR_FILE:-./data/hongloumeng_vectors.json}"
DIM=1024  # Qwen3-Embedding-0.6B 输出维度

echo "=== vdb.rs 红楼梦 RAG 示例 ==="
echo "input:    ${INPUT_TXT}"
echo "model:    ${EMBED_MODEL}"
echo "db dir:   ${DB_DIR}"
echo "limit:    ${LIMIT} chunks"
echo

# 1. 文本转向量
echo "[1/5] converting text to vectors..."
python3 "${SCRIPT_DIR}/text_to_vectors.py" \
  --input "${INPUT_TXT}" \
  --model "${EMBED_MODEL}" \
  --output "${VECTOR_FILE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-overlap "${CHUNK_OVERLAP}" \
  --batch-size 8 \
  --limit "${LIMIT}"

# 2. 构建 vdb.rs release 二进制
echo
echo "[2/5] building vdb.rs CLI..."
cargo build --release --bin vdb

# 3. 创建数据库
echo
echo "[3/5] creating vdb.rs database (dim=${DIM})..."
rm -rf "${DB_DIR}"
./target/release/vdb create --dir "${DB_DIR}" --dim "${DIM}"

# 4. 批量插入（一次性写入，避免每条记录产生一个全量索引快照）
echo
echo "[4/5] batch inserting vectors..."
./target/release/vdb batch-insert --dir "${DB_DIR}" --file "${VECTOR_FILE}"

# 5. 搜索测试
echo
echo "[5/5] searching..."
QUERIES=(
    "贾宝玉和林黛玉的关系"
    "大观园修建的缘由"
    "王熙凤的性格特点"
)

for q in "${QUERIES[@]}"; do
    echo
    echo "--- query: ${q} ---"
    # 用 llama-embedding 把查询文本转成向量
    query_vec=$(llama-embedding \
        -m "${EMBED_MODEL}" \
        -p "${q}" \
        --embd-output-format json \
        --pooling mean \
        --embd-normalize 2 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['data'][0]['embedding']))")

    result_file=$(mktemp)
    ./target/release/vdb search \
        --dir "${DB_DIR}" \
        --query "${query_vec}" \
        --k 3 \
        --nprobe 16 \
        --refine-k 100 \
        --sql-filter "text IS NOT NULL" | tee "${result_file}"

    # CLI 只返回 id 与距离；从向量文件按 id 反查原文。
    python3 "${SCRIPT_DIR}/lookup_results.py" "${VECTOR_FILE}" "${result_file}"
    rm -f "${result_file}"

done

echo
echo "=== 红楼梦 RAG 示例完成 ==="
echo "数据库位置: ${DB_DIR}"
echo "向量文件:   ${VECTOR_FILE}"
