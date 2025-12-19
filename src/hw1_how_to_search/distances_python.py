from typing import Sequence, List, Tuple

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]

def _ensure_2d_py(queries: Sequence[Sequence[float]] | Sequence[float]) -> Tuple[List[List[float]], bool]:
    if queries and isinstance(queries[0], (int, float)):
        return [list(queries)], True
    return [list(row) for row in queries], False

def _norm(v: List[float]) -> float:
    s = 0.0
    for x in v:
        s += x * x
    return s ** 0.5


def dot_product(queries: Sequence[Sequence[float]] | Sequence[float],
                candidates: Matrix) -> List[List[float]] | List[float]:

    q2d, was_single = _ensure_2d_py(queries)
    c2d = [list(v) for v in candidates]

    results: List[List[float]] = []

    for q in q2d:
        row = []
        for c in c2d:
            s = 0.0
            for qi, ci in zip(q, c):
                s += qi * ci
            row.append(s)
        results.append(row)

    return results[0] if was_single else results


def cosine_similarity(queries, candidates):
    q2d, was_single = _ensure_2d_py(queries)
    c2d = [list(v) for v in candidates]

    c_normed = []
    for c in c2d:
        norm_c = _norm(c)
        if norm_c == 0:
            raise ValueError("Candidate vector has zero norm.")
        c_normed.append([ci / norm_c for ci in c])

    results = []
    for q in q2d:
        norm_q = _norm(q)
        if norm_q == 0:
            raise ValueError("Query vector has zero norm.")
        qn = [qi / norm_q for qi in q]

        row = []
        for cn in c_normed:
            s = 0.0
            for qi, ci in zip(qn, cn):
                s += qi * ci
            row.append(s)
        results.append(row)

    return results[0] if was_single else results


def manhattan_distance(queries, candidates):
    q2d, was_single = _ensure_2d_py(queries)
    c2d = [list(v) for v in candidates]

    results = []

    for q in q2d:
        row = []
        for c in c2d:
            s = 0.0
            for qi, ci in zip(q, c):
                s += abs(qi - ci)
            row.append(s)
        results.append(row)

    return results[0] if was_single else results


def euclidean_distance(queries, candidates):
    q2d, was_single = _ensure_2d_py(queries)
    c2d = [list(v) for v in candidates]

    results = []

    for q in q2d:
        row = []
        for c in c2d:
            s = 0.0
            for qi, ci in zip(q, c):
                diff = qi - ci
                s += diff * diff
            row.append(s**0.5)
        results.append(row)

    return results[0] if was_single else results
