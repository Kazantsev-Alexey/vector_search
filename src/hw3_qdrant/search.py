import argparse
import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import orjson
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--vectors", default="data/hw3/vectors.npy")
    ap.add_argument("--ground-truth", default="results/hw3/ground_truth.jsonl")
    ap.add_argument("--ef", type=int, default=50)
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=300.0, prefer_grpc=True)

    vectors = np.load(args.vectors, mmap_mode="r")
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    
    gt_lines = Path(args.ground_truth).read_bytes().splitlines()
    gt = [None] * len(gt_lines)
    for raw in gt_lines:
        obj = json.loads(raw)
        i_str, gt_neighbors = next(iter(obj.items()))
        i = int(i_str)
        gt[i] = [int(x) for x in gt_neighbors]
    gt = [x for x in gt if x is not None]

    if len(gt) != vectors.shape[0]:
        raise ValueError(f"GT rows {len(gt)} != vectors {vectors.shape[0]}")

    #  Make sure that index is built (status is green) before searching
    for col in ["single_unnamed", "multiple_named"]:
        t0 = perf_counter()
        while True:
            info = client.get_collection(col)
            if str(info.status).lower().endswith("green"):
                break
            if perf_counter() - t0 > 600:
                raise TimeoutError(f"{col} is not green")
            sleep(1)

    runs = [
        ("single_unnamed", None, "single_unnamed"),
        ("multiple_named", "clip_default", "clip_default"),
        ("multiple_named", "clip_tuned", "clip_tuned"),
    ]

    Path("results/hw3").mkdir(parents=True, exist_ok=True)

    for collection, vector_name, out_name in runs:
        p1 = p3 = p5 = p10 = 0.0

        t0 = perf_counter()
        n = vectors.shape[0]

        for i in range(n):
            if i % 10000 == 0:
                print(f"{i}/{n}")
            q = vectors[i].tolist()

            res = client.query_points(
                collection_name=collection,
                query=q,
                using=vector_name,
                limit=11,
                search_params=SearchParams(hnsw_ef=args.ef, exact=False),
                with_payload=False,
                with_vectors=False,
            )

            ids = [int(p.id) for p in res.points]
            ids = [x for x in ids if x != i][:10]

            true_ids = [x for x in gt[i] if x != i][:10]

            p1 += len(set(ids[:1]) & set(true_ids[:1])) / 1
            p3 += len(set(ids[:3]) & set(true_ids[:3])) / 3
            p5 += len(set(ids[:5]) & set(true_ids[:5])) / 5
            p10 += len(set(ids[:10]) & set(true_ids[:10])) / 10

        total = perf_counter() - t0
        qps = n / total

        info = client.get_collection(collection)
        m = info.config.hnsw_config.m
        ef_construct = info.config.hnsw_config.ef_construct

        if vector_name is not None:
            vcfg = info.config.params.vectors[vector_name]
            if getattr(vcfg, "hnsw_config", None) is not None:
                if getattr(vcfg.hnsw_config, "m", None) is not None:
                    m = vcfg.hnsw_config.m
                if getattr(vcfg.hnsw_config, "ef_construct", None) is not None:
                    ef_construct = vcfg.hnsw_config.ef_construct

        record = {
            "m": int(m),
            "ef_construct": int(ef_construct),
            "ef_search": int(args.ef),
            "Precision@1": p1 / n,
            "Precision@3": p3 / n,
            "Precision@5": p5 / n,
            "Precision@10": p10 / n,
            "total_search_time": total,
            "QPS": qps,
        }

        out_path = Path(f"results/hw3/{out_name}_ef_search_{args.ef}_results.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(f"{out_name}: P@10={record['Precision@10']:.4f} | QPS={record['QPS']:.1f}")


if __name__ == "__main__":
    main()