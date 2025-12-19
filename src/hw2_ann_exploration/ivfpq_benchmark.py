from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import faiss

from .metrics import evaluate_index, load_vectors


def build_ivfpq_index(
    vectors: np.ndarray,
    nlist: int,
    m: int,
    nbits: int,
) -> tuple[faiss.IndexIVFPQ, float]:
    """Build IVFPQ index and return (index, indexing time)."""
    n, dim = vectors.shape
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    index.metric_type = faiss.METRIC_L2

    t0 = perf_counter()
    index.train(vectors)
    index.add(vectors)
    indexing_time = perf_counter() - t0

    return index, indexing_time


def run(
    vectors_path: str = "data/hw2/vectors.npy",
    output_path: str = "results/hw2/ivfpq_results.jsonl",
) -> None:
    """Grid search by parameters of IVFPQ."""
    vectors = np.asarray(load_vectors(vectors_path), dtype=np.float32)
    n, dim = vectors.shape
    print(f"Loaded {n} vectors of dim {dim} from {vectors_path}\n")

    nlist_list = [64, 128, 256, 512, 1024]
    m_list = [16, 32]
    nbits_list = [8]
    nprobe_list = [1, 2, 4, 8, 16, 32, 64, 128]

    ks = (1, 3, 5, 10)
    max_k = max(ks)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'nlist':>6} {'m':>4} {'nbits':>6} {'nprobe':>7} "
        f"{'idx_time':>9} {'QPS':>10} {'P@1':>7} {'P@5':>7} {'P@10':>7}"
    )
    print(header)
    print("-" * len(header))

    with out_path.open("w", encoding="utf-8") as fout:
        for nlist, m, nbits in product(nlist_list, m_list, nbits_list):
            print(f"\n=== Building IVFPQ index: nlist={nlist}, m={m}, nbits={nbits} ===")
            index, indexing_time = build_ivfpq_index(vectors, nlist=nlist, m=m, nbits=nbits)

            for nprobe in nprobe_list:
                index.nprobe = nprobe

                _, all_neighbors = index.search(vectors, max_k + 1)

                def query_fn(i: int, top_k: int) -> list[int]:
                    row = all_neighbors[i]
                    neighbors = [idx for idx in row if idx != i]
                    return neighbors[:top_k]

                metrics = evaluate_index(
                    query_fn=query_fn,
                    n_queries=n,
                    ks=ks,
                    verbose=False,
                )

                record: dict[str, float | int | str] = {
                    "algorithm": "ivfpq",
                    "nlist": nlist,
                    "m": m,
                    "nbits": nbits,
                    "nprobe": nprobe,
                    "indexing_time": indexing_time,
                    **metrics,
                }

                fout.write(json.dumps(record) + "\n")
                fout.flush()

                print(
                    f"{nlist:>6} {m:>4} {nbits:>6} {nprobe:>7} "
                    f"{indexing_time:>9.2f} {metrics['QPS']:>10.1f} "
                    f"{metrics['Precision@1']:>7.4f} "
                    f"{metrics['Precision@5']:>7.4f} "
                    f"{metrics['Precision@10']:>7.4f}"
                )

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()