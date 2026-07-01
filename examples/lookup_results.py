#!/usr/bin/env python3
"""根据 vdb.rs CLI search 结果中的 id，从向量文件反查原文。

用法：
  python3 examples/lookup_results.py <vectors.jsonl> <search_result.txt>
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: lookup_results.py <vectors.jsonl> <search_result.txt>", file=sys.stderr)
        sys.exit(1)

    vector_file = sys.argv[1]
    result_file = sys.argv[2]

    records = {}
    with open(vector_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["id"]] = rec

    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("id="):
                continue
            parts = line.split()
            rid = int(parts[0].split("=")[1])
            rec = records.get(rid)
            if rec:
                text = rec["text"].replace("\r", " ").replace("\n", " ")
                print(f"  -> [{rid}] {text[:120]}...")


if __name__ == "__main__":
    main()
