from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import hnswlib

from .metrics import evaluate_index, load_vectors


def build_hnsw_index(
    vectors: np.ndarray,
    m: int,
    ef_construction: int,
    space: str = "l2",
) -> tuple[hnswlib.Index, float]:
    """Build HNSW index and return (index, indexing time)."""
    num_elements, dim = vectors.shape

    index = hnswlib.Index(space=space, dim=dim)

    t0 = perf_counter()
    index.init_index(
        max_elements=num_elements,
        M=m,
        ef_construction=ef_construction,
    )
    index.add_items(vectors, np.arange(num_elements))
    indexing_time = perf_counter() - t0

    return index, indexing_time


def run(
    vectors_path: str = "data/hw2/vectors.npy",
    output_path: str = "results/hw2/hnsw_results.jsonl",
) -> None:
    """Grid search by parameters of HNSW."""
    vectors = np.asarray(load_vectors(vectors_path), dtype=np.float32)
    n, dim = vectors.shape
    print(f"Loaded {n} vectors of dim {dim} from {vectors_path}\n")

    m_list = [8, 16, 32, 64]
    ef_construction_list = [32, 64, 100, 128, 256]
    ef_search_list = [32, 64, 100, 128, 256]

    ks = (1, 3, 5, 10)
    max_k = max(ks)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'M':>4} {'efC':>6} {'efS':>6} "
        f"{'idx_time':>9} {'QPS':>10} {'P@1':>7} {'P@5':>7} {'P@10':>7}"
    )
    print(header)
    print("-" * len(header))

    with out_path.open("w", encoding="utf-8") as fout:
        for m, ef_c in product(m_list, ef_construction_list):
            print(f"\n=== Building HNSW index: M={m}, ef_construction={ef_c} ===")
            index, indexing_time = build_hnsw_index(
                vectors=vectors,
                m=m,
                ef_construction=ef_c,
                space="l2",
            )

            for ef_s in ef_search_list:
                index.set_ef(ef_s)

                def query_fn(i: int, top_k: int) -> list[int]:
                    labels, _ = index.knn_query(vectors[i : i + 1], k=top_k + 1)
                    ids = labels[0].tolist()
                    ids = [idx for idx in ids if idx != i]
                    return ids[:top_k]

                metrics = evaluate_index(
                    query_fn=query_fn,
                    n_queries=n,
                    ks=ks,
                    verbose=False,
                )

                record: dict[str, float | int | str] = {
                    "algorithm": "hnsw",
                    "M": m,
                    "efConstruction": ef_c,
                    "efSearch": ef_s,
                    "indexing_time": indexing_time,
                    **metrics,
                }

                fout.write(json.dumps(record) + "\n")
                fout.flush()

                print(
                    f"{m:>4} {ef_c:>6} {ef_s:>6} "
                    f"{indexing_time:>9.2f} {metrics['QPS']:>10.1f} "
                    f"{metrics['Precision@1']:>7.4f} "
                    f"{metrics['Precision@5']:>7.4f} "
                    f"{metrics['Precision@10']:>7.4f}"
                )

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()