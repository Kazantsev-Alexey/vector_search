import time
import numpy as np
from typing import Callable, Tuple, List

from . import distances_numpy as dn
from . import distances_python as dp


def benchmark(fn: Callable, q, m, repeats: int = 5) -> Tuple[float, float, float]:
    times: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(q, m)
        end = time.perf_counter()
        times.append(end - start)
    return min(times), sum(times) / len(times), max(times)


def run():
    configs = [
        (1, 1000, 32),
        (10, 2000, 128),
        (50, 5000, 256),
    ]

    for n_queries, n_candidates, dim in configs:
        print(f"\n=== queries={n_queries}, matrix={n_candidates}, dim={dim} ===")
        print("-" * 60)

        rng = np.random.default_rng(0)
        q = rng.random((n_queries, dim))
        m = rng.random((n_candidates, dim))

        q_list = q.tolist()
        m_list = m.tolist()

        tests: List[Tuple[str, Callable, Callable]] = [
            ("dot", dn.dot_product, dp.dot_product),
            ("cosine", dn.cosine_similarity, dp.cosine_similarity),
            ("euclid", dn.euclidean_distance, dp.euclidean_distance),
            ("manhat", dn.manhattan_distance, dp.manhattan_distance),
        ]

        for name, fn_np, fn_py in tests:
            print(f"\n{name.upper()}")

            mn, md, mx = benchmark(fn_np, q, m)
            print(f"numpy :  min={mn*1000:.3f} ms   mean={md*1000:.3f} ms   max={mx*1000:.3f} ms")

            mn, md, mx = benchmark(fn_py, q_list, m_list)
            print(f"python:  min={mn*1000:.3f} ms   mean={md*1000:.3f} ms   max={mx*1000:.3f} ms")

    print("\nDone.")


if __name__ == "__main__":
    run()