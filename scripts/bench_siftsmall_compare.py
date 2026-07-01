#!/usr/bin/env python3
"""siftsmall 真实数据对比：vdb.rs vs FAISS vs LanceDB。

读取 siftsmall_base/query/groundtruth，分别用三种引擎构建索引并搜索，
输出 recall@10 / QPS / p50 / p99 / build_time 的对比表格与 JSON。
"""
import json
import os
import struct
import time
from pathlib import Path

import faiss
import lancedb
import numpy as np
import pyarrow as pa

DATASET_PREFIX = Path(__file__).resolve().parents[2] / "models" / "data" / "siftsmall" / "siftsmall"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
K = 10
WARMUP = 5
NPROBE_LIST = [16, 50, 100]


def read_fvecs(path):
    vecs = []
    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("<I", dim_bytes)[0]
            vecs.append(np.frombuffer(f.read(dim * 4), dtype=np.float32))
    return np.stack(vecs).astype("float32")


def read_ivecs(path):
    lists = []
    with open(path, "rb") as f:
        while True:
            k_bytes = f.read(4)
            if not k_bytes:
                break
            k = struct.unpack("<I", k_bytes)[0]
            lists.append(np.frombuffer(f.read(k * 4), dtype=np.int32))
    return lists


def compute_recall(results, groundtruth, k=K):
    total = 0
    for res, gt in zip(results, groundtruth):
        total += len(set(res[:k].tolist()) & set(gt[:k].tolist()))
    return total / (len(results) * k)


def measure_search(fn, queries):
    for q in queries[:WARMUP]:
        fn(q)
    times = []
    for q in queries:
        t0 = time.perf_counter()
        fn(q)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    qps = len(queries) / times.sum() * 1000.0
    return qps, float(np.median(times)), float(np.percentile(times, 99))


def bench_faiss(name, index, base, queries, groundtruth, nprobes=NPROBE_LIST):
    faiss.omp_set_num_threads(1)
    t0 = time.perf_counter()
    if not index.is_trained:
        index.train(base)
    index.add(base)
    build_ms = (time.perf_counter() - t0) * 1000.0

    reports = []
    for nprobe in nprobes:
        index.nprobe = nprobe
        D, I = index.search(queries, K)
        recall = compute_recall(I, groundtruth)

        def fn(q):
            return index.search(q.reshape(1, -1), K)

        qps, p50, p99 = measure_search(fn, queries)
        reports.append({
            "system": name,
            "nprobe": nprobe,
            "recall_at_10": round(recall, 4),
            "qps": round(qps, 1),
            "p50_ms": round(p50, 3),
            "p99_ms": round(p99, 3),
            "build_time_ms": round(build_ms, 1),
        })
    return reports


def bench_lancedb(base, queries, groundtruth, nprobes=NPROBE_LIST):
    dim = base.shape[1]
    db = lancedb.connect(str(REPORT_DIR / "lancedb_siftsmall"))
    ids = np.arange(len(base), dtype=np.int64)
    table = db.create_table(
        "siftsmall",
        data=pa.table(
            {"id": ids, "vector": pa.FixedSizeListArray.from_arrays(base.reshape(-1), dim)},
            schema=pa.schema([
                pa.field("id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]),
        ),
        mode="overwrite",
    )

    t0 = time.perf_counter()
    table.create_index(
        metric="L2",
        num_partitions=100,
        num_sub_vectors=16,
        vector_column_name="vector",
        replace=True,
    )
    build_ms = (time.perf_counter() - t0) * 1000.0

    reports = []
    for nprobe in nprobes:
        def fn(q):
            return (
                table.search(q)
                .metric("L2")
                .limit(K)
                .nprobes(nprobe)
                .refine_factor(10)
                .to_list()
            )

        # 先跑一次完整搜索拿到 recall
        results = [fn(q) for q in queries]
        preds = [[int(r["id"]) for r in res] for res in results]
        recall = compute_recall(np.array(preds), groundtruth)
        qps, p50, p99 = measure_search(fn, queries)
        reports.append({
            "system": "LanceDB IVF_PQ",
            "nprobe": nprobe,
            "recall_at_10": round(recall, 4),
            "qps": round(qps, 1),
            "p50_ms": round(p50, 3),
            "p99_ms": round(p99, 3),
            "build_time_ms": round(build_ms, 1),
        })
    return reports


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    base = read_fvecs(str(DATASET_PREFIX) + "_base.fvecs")
    queries = read_fvecs(str(DATASET_PREFIX) + "_query.fvecs")
    groundtruth = read_ivecs(str(DATASET_PREFIX) + "_groundtruth.ivecs")
    print(f"[CMP] loaded base={len(base)} query={len(queries)} truth={len(groundtruth)} dim={base.shape[1]}")

    all_reports = []

    # FAISS IVF_FLAT
    quantizer = faiss.IndexFlatL2(base.shape[1])
    ivf_flat = faiss.IndexIVFFlat(quantizer, base.shape[1], 100, faiss.METRIC_L2)
    all_reports.extend(bench_faiss("FAISS IVF_FLAT", ivf_flat, base, queries, groundtruth))

    # FAISS IVF_PQ
    quantizer = faiss.IndexFlatL2(base.shape[1])
    ivf_pq = faiss.IndexIVFPQ(quantizer, base.shape[1], 100, 16, 8, faiss.METRIC_L2)
    all_reports.extend(bench_faiss("FAISS IVF_PQ", ivf_pq, base, queries, groundtruth))

    # LanceDB
    all_reports.extend(bench_lancedb(base, queries, groundtruth))

    out = REPORT_DIR / "siftsmall_compare.json"
    with open(out, "w") as f:
        json.dump(all_reports, f, indent=2)

    print(f"\n对比结果已保存到 {out}\n")
    print(f"{'system':<20} {'nprobe':>6} {'recall@10':>10} {'QPS':>10} {'p50_ms':>8} {'p99_ms':>8} {'build_ms':>10}")
    print("-" * 80)
    for r in all_reports:
        print(f"{r['system']:<20} {r['nprobe']:>6} {r['recall_at_10']:>10.4f} {r['qps']:>10.1f} {r['p50_ms']:>8.3f} {r['p99_ms']:>8.3f} {r['build_time_ms']:>10.1f}")


if __name__ == "__main__":
    main()
