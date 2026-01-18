from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np


def load_vectors(path: str = "data/hw2/vectors.npy") -> np.ndarray:
    # mmap, экономим RAM
    return np.load(Path(path), mmap_mode="r")


def load_ground_truth(path: str = "results/hw2/ground_truth.jsonl") -> dict[int, list[int]]:
    gt: dict[int, list[int]] = {}
    with Path(path).open() as f:
        for line in f:
            rec = json.loads(line)
            (k, v), = rec.items() # делаем анпакинг
            gt[int(k)] = v
    return gt


def precision_at_k(true_ids: Sequence[int], pred_ids: Sequence[int], k: int) -> float:
    return len(set(true_ids[:k]) & set(pred_ids[:k])) / k


def evaluate_index(
    query_fn: Callable[[int, int], list[int]],
    n_queries: int,
    ks: Sequence[int] = (1, 3, 5, 10),
    verbose: bool = True,
) -> dict[str, float]:
    gt = load_ground_truth()
    ks = sorted(ks)
    max_k = max(ks)
    sums = {k: 0.0 for k in ks}

    t0 = perf_counter()
    for i in range(n_queries):
        preds = query_fn(i, max_k)
        for k in ks:
            sums[k] += precision_at_k(gt[i], preds, k)

        if verbose and (i + 1) % 5000 == 0:
            print(f"Evaluated {i + 1}/{n_queries}")

    elapsed = perf_counter() - t0
    result: dict[str, float] = {
        "total_search_time": elapsed,
        "QPS": n_queries / elapsed,
    }
    for k in ks:
        result[f"Precision@{k}"] = sums[k] / n_queries
    return result
