from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from annoy import AnnoyIndex

from .metrics import evaluate_index, load_vectors


def build_annoy_index(
    vectors: np.ndarray,
    n_trees: int,
    metric: str = "euclidean",
) -> tuple[AnnoyIndex, float]:
    """Build ANNOY index and return (index, indexing time)."""
    n, dim = vectors.shape
    index = AnnoyIndex(dim, metric=metric)

    for i in range(n):
        index.add_item(i, vectors[i].tolist())

    t0 = perf_counter()
    index.build(n_trees)
    indexing_time = perf_counter() - t0

    return index, indexing_time


def run(
    vectors_path: str = "data/hw2/vectors.npy",
    output_path: str = "results/hw2/annoy_results.jsonl",
) -> None:
    """Grid search by n_trees and search_k"""
    vectors = np.asarray(load_vectors(vectors_path), dtype=np.float32)
    n = vectors.shape[0]
    print(f"Loaded {n} vectors from {vectors_path}\n")

    n_trees_list = [10, 25, 50, 100, 200]
    search_k_list = [100, 500, 1000, 5000]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"{'n_trees':>8} {'search_k':>9} {'idx_time':>9} {'QPS':>10} {'P@1':>7} {'P@5':>7} {'P@10':>7}"
    print(header)
    print("-" * len(header))

    with out_path.open("w", encoding="utf-8") as f:
        for n_trees, search_k in product(n_trees_list, search_k_list):
            index, indexing_time = build_annoy_index(vectors, n_trees=n_trees, metric="euclidean")

            def query_fn(i: int, top_k: int) -> list[int]:
                ids = index.get_nns_by_item(i, top_k + 1, search_k=search_k)
                return [idx for idx in ids if idx != i][:top_k]

            metrics = evaluate_index(
                query_fn=query_fn,
                n_queries=n,
                ks=(1, 3, 5, 10),
                verbose=False,
            )

            record = {
                "algorithm": "annoy",
                "n_trees": n_trees,
                "search_k": search_k,
                "indexing_time": indexing_time,
                **metrics,
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            print(
                f"{n_trees:>8} {search_k:>9} {indexing_time:>9.2f} "
                f"{metrics['QPS']:>10.1f} {metrics['Precision@1']:>7.4f} "
                f"{metrics['Precision@5']:>7.4f} {metrics['Precision@10']:>7.4f}"
            )

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()