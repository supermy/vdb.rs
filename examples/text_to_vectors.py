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
    --chunk-overlap 32 \
    --batch-size 8

输出 JSON 格式（每行一个对象）：
  {"id": 0, "text": "...", "vector": [0.1, -0.2, ...]}
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


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


def embed_batches(
    texts: List[str],
    model_path: str,
    batch_size: int,
    separator: str = "<#sep#>",
) -> List[List[float]]:
    """调用 llama-embedding 批量生成向量。"""
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # 确保分隔符不会出现在文本中
        safe_sep = separator
        while any(safe_sep in t for t in batch):
            safe_sep += "#"
        prompt = safe_sep.join(batch)

        cmd = [
            "llama-embedding",
            "-m", model_path,
            "-p", prompt,
            "--embd-output-format", "json",
            "--embd-separator", safe_sep,
            "--pooling", "mean",
            "--embd-normalize", "2",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"llama-embedding stdout: {result.stdout}", file=sys.stderr)
            print(f"llama-embedding stderr: {result.stderr}", file=sys.stderr)
            raise RuntimeError("embedding failed")

        # llama-embedding 的长 JSON 可能被折行，需从 { 开始累积到匹配的 }
        json_text = ""
        brace_depth = 0
        in_json = False
        for ln in result.stdout.splitlines():
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
            print(f"llama-embedding stdout: {result.stdout}", file=sys.stderr)
            print(f"llama-embedding stderr: {result.stderr}", file=sys.stderr)
            raise RuntimeError("no JSON output from embedding")

        data = json.loads(json_text)
        embeddings = [item["embedding"] for item in data["data"]]
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"embedding count mismatch: expected {len(batch)}, got {len(embeddings)}"
            )
        all_embeddings.extend(embeddings)
    return all_embeddings


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
    if args.limit > 0:
        chunks = chunks[: args.limit]
    print(f"[text2vec] chunks: {len(chunks)}")

    print(f"[text2vec] embedding with {args.model}")
    embeddings = embed_batches(chunks, args.model, args.batch_size)
    dim = len(embeddings[0])
    print(f"[text2vec] embedding dim: {dim}")

    output_path = Path(args.output)
    # 归一化换行符，避免 CRLF 在终端回显时覆盖前缀
    normalized = [chunk.replace("\r", "\n") for chunk in chunks]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, (chunk, vec) in enumerate(zip(normalized, embeddings)):
            record = {"id": i, "text": chunk, "vector": vec}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[text2vec] wrote {len(embeddings)} records to {args.output}")
    print(f"[text2vec] next: import into vdb.rs with dim={dim}")


if __name__ == "__main__":
    main()
