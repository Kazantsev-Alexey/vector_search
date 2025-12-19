import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from . import distances_numpy as dn
from . import distances_python as dp


def _check_pair(n_queries: int, n_candidates: int, dim: int) -> None:
    rng = np.random.default_rng(0)
    queries = rng.random((n_queries, dim))
    candidates = rng.random((n_candidates, dim))

    # ---------- numpy vs sklearn ----------
    # cosine
    cos_np = dn.cosine_similarity(queries, candidates)
    cos_sk = sk_cosine_similarity(queries, candidates)
    assert np.allclose(cos_np, cos_sk, atol=1e-6)

    # euclidean
    eu_np = dn.euclidean_distance(queries, candidates)
    eu_sk = euclidean_distances(queries, candidates)
    assert np.allclose(eu_np, eu_sk, atol=1e-6)

    # manhattan
    man_np = dn.manhattan_distance(queries, candidates)
    man_sk = manhattan_distances(queries, candidates)
    assert np.allclose(man_np, man_sk, atol=1e-6)

    # dot product — сравниваем с "родным" numpy
    dot_np = dn.dot_product(queries, candidates)
    dot_ref = queries @ candidates.T
    assert np.allclose(dot_np, dot_ref, atol=1e-6)

    # ---------- numpy vs pure python ----------
    q_list = queries.tolist()
    c_list = candidates.tolist()

    dot_py = np.array(dp.dot_product(q_list, c_list))
    cos_py = np.array(dp.cosine_similarity(q_list, c_list))
    eu_py = np.array(dp.euclidean_distance(q_list, c_list))
    man_py = np.array(dp.manhattan_distance(q_list, c_list))

    assert np.allclose(dot_np, dot_py, atol=1e-6)
    assert np.allclose(cos_np, cos_py, atol=1e-6)
    assert np.allclose(eu_np, eu_py, atol=1e-6)
    assert np.allclose(man_np, man_py, atol=1e-6)


def run_checks() -> None:
    configs = [
        (1, 5, 8),     # один query, несколько кандидатов
        (3, 10, 16),   # несколько query, несколько кандидатов
        (5, 7, 64),
    ]
    for n_q, n_c, d in configs:
        _check_pair(n_q, n_c, d)
    print("All distance checks passed!")


if __name__ == "__main__":
    run_checks()