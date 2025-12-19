from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np


def run(
    vectors_path: str = "data/hw2/vectors.npy",
    output_path: str = "results/hw2/ground_truth.jsonl",
    k: int = 10,
    batch_size: int = 256,
) -> None:
    vectors_path = Path(vectors_path)
    output_path = Path(output_path)

    print(f"Loading vectors from: {vectors_path}")
    vectors = np.load(vectors_path).astype("float32")
    n, dim = vectors.shape
    print(f"Vector matrix loaded: {n} vectors, dim {dim}")
    print(f"Saving ground truth to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    db_norms = np.sum(vectors ** 2, axis=1)

    t0 = perf_counter()
    with output_path.open("w") as f:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = vectors[start:end]

            batch_norms = np.sum(batch ** 2, axis=1)

            # евклидово расстояние на матрицах
            dots = batch @ vectors.T
            dists_sq = batch_norms[:, None] + db_norms[None, :] - 2.0 * dots
            np.maximum(dists_sq, 0.0, out=dists_sq)  # float precision fix

            for local_idx in range(len(batch)):
                i = start + local_idx
                row = dists_sq[local_idx]

                # +1 т.к. сам вектор тоже попадёт
                idx = np.argpartition(row, k + 1)[:k + 1]
                idx = idx[idx != i][:k]
                # сортируем для корректного at_k
                idx = idx[np.argsort(row[idx])]

                f.write(json.dumps({str(i): idx.tolist()}) + "\n")

            if end % 5000 == 0 or end == n:
                elapsed = perf_counter() - t0
                print(f"Processed {end}/{n} vectors, elapsed {elapsed:.1f}s")

    print(f"Done. Total time: {perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    run()
