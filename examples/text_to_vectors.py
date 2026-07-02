#!/usr/bin/env python3
"""
文本 → 向量 转换示例：以《红楼梦》为例，使用本地 Qwen3-Embedding GGUF 模型生成向量。

依赖：
  - llama-embedding (llama.cpp 嵌入命令)
  - Python 3.8+

用法：
  python3 examples/text_to_vectors.py \
    --input ../models/data/txt/红楼梦.txt \
    --model ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --output data/hongloumeng_vectors.json \
    --chunk-size 256 \
    --chunk-overlap 32

输出 JSON 格式（每行一个对象）：
  {"id": 0, "text": "...", "vector": [0.1, -0.2, ...]}
"""

import argparse
import json
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


def read_text(path: str) -> str:
    """读取文本，自动尝试 UTF-8 与 GBK 编码。"""
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码文件: {path}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """按固定长度滑窗分块。"""
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def choose_separator(texts: List[str]) -> str:
    """选择一个不会出现在文本中的分隔符。"""
    for _ in range(10):
        sep = f"<#SEP#{secrets.token_hex(8)}#>"
        if not any(sep in t for t in texts):
            return sep
    raise RuntimeError("无法找到不与文本冲突的分隔符")


def embed_texts(texts: List[str], model_path: str, batch_size: int) -> List[List[float]]:
    """调用 llama-embedding 分批生成向量（使用 -f 文件输入）。

    分批原因：
    - 单条调用会重复加载模型；
    - 一次性传入过多长文本在某些机器上推理极慢；
    - 每批 8~16 条可在模型加载次数与单次推理长度之间取得平衡。
    """
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        separator = choose_separator(batch)

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(separator.join(batch))
            prompt_file = tmp.name

        try:
            cmd = [
                "llama-embedding",
                "-m", model_path,
                "-f", prompt_file,
                "--embd-output-format", "json",
                "--embd-separator", separator,
                "--pooling", "mean",
                "--embd-normalize", "2",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"llama-embedding stdout: {result.stdout}", file=sys.stderr)
                print(f"llama-embedding stderr: {result.stderr}", file=sys.stderr)
                raise RuntimeError("embedding failed")

            json_text = extract_json(result.stdout)
            data = json.loads(json_text)
            embeddings = [item["embedding"] for item in data["data"]]
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"embedding count mismatch: expected {len(batch)}, got {len(embeddings)}"
                )
            all_embeddings.extend(embeddings)
        finally:
            Path(prompt_file).unlink(missing_ok=True)

    return all_embeddings


def extract_json(stdout: str) -> str:
    """从 llama-embedding 的输出中提取完整 JSON 对象。"""
    json_text = ""
    brace_depth = 0
    in_json = False
    for ln in stdout.splitlines():
        if not in_json:
            stripped = ln.lstrip()
            if stripped.startswith("{"):
                in_json = True
                json_text = ""
                brace_depth = 0
            else:
                continue
        json_text += ln
        brace_depth += ln.count("{") - ln.count("}")
        if brace_depth == 0:
            break

    if not json_text:
        print(f"llama-embedding stdout: {stdout}", file=sys.stderr)
        raise RuntimeError("no JSON output from embedding")
    return json_text


def main():
    parser = argparse.ArgumentParser(description="文本转向量示例")
    parser.add_argument("--input", required=True, help="输入文本文件路径")
    parser.add_argument("--model", required=True, help="GGUF embedding 模型路径")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--chunk-size", type=int, default=256, help="每块字符数")
    parser.add_argument("--chunk-overlap", type=int, default=32, help="滑窗重叠字符数")
    parser.add_argument("--batch-size", type=int, default=8, help="每批嵌入文本数")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 块（0=全部）")
    args = parser.parse_args()

    print(f"[text2vec] reading {args.input}")
    text = read_text(args.input)
    print(f"[text2vec] text length: {len(text)} chars")

    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
    # 去掉完全重复的块，避免嵌入和存储重复文本
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique_chunks.append(c)
    chunks = unique_chunks
    if args.limit > 0:
        chunks = chunks[: args.limit]
    print(f"[text2vec] unique chunks: {len(chunks)}")

    print(f"[text2vec] embedding with {args.model}, batch_size={args.batch_size}")
    embeddings = embed_texts(chunks, args.model, args.batch_size)
    dim = len(embeddings[0])
    print(f"[text2vec] embedding dim: {dim}")

    # 归一化换行符，避免 CRLF 在终端回显时覆盖前缀
    normalized = [chunk.replace("\r", "\n") for chunk in chunks]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, (chunk, vec) in enumerate(zip(normalized, embeddings)):
            record = {"id": i, "text": chunk, "vector": vec}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[text2vec] wrote {len(embeddings)} records to {args.output}")
    print(f"[text2vec] next: import into vdb.rs with dim={dim}")


if __name__ == "__main__":
    main()
