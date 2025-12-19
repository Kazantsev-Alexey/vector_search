import argparse
import json
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models


def build_filter(cond):
    if cond is None:
        return None
    if isinstance(cond, dict) and len(cond) == 0:
        return None

    if "and" in cond:
        must = []
        for c in cond["and"]:
            x = build_filter(c)
            if x is not None:
                must.append(x)
        if not must:
            return None
        return models.Filter(must=must)

    if "or" in cond:
        should = []
        for c in cond["or"]:
            x = build_filter(c)
            if x is not None:
                should.append(x)
        if not should:
            return None
        return models.Filter(should=should)

    if "not" in cond:
        x = build_filter(cond["not"])
        if x is None:
            return None
        return models.Filter(must_not=[x])

    if not isinstance(cond, dict) or len(cond) == 0:
        return None

    field = next(iter(cond))
    spec = cond[field]

    if "range" in spec:
        r = spec["range"]
        return models.FieldCondition(
            key=field,
            range=models.Range(gt=r.get("gt"), gte=r.get("gte"), lt=r.get("lt"), lte=r.get("lte")),
        )

    if "match" in spec:
        v = spec["match"]["value"]
        return models.FieldCondition(key=field, match=models.MatchValue(value=v))

    return None


def precision_at_k(found_ids, gt_ids, k):
    s = set(gt_ids[:k])
    hit = 0
    for x in found_ids[:k]:
        if x in s:
            hit += 1
    return hit / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="hw4")
    ap.add_argument("--tests", default="data/hw4/tests.jsonl")
    ap.add_argument("--ef", type=int, default=50)
    ap.add_argument("--out", default="results/hw4/filtered_search_results.jsonl")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=300)

    tests_path = Path(args.tests)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    p1 = p3 = p5 = p10 = 0.0

    t0 = time.time()
    with tests_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            q = t["query"]
            cond = t.get("conditions", {})
            gt = t["closest_ids"]

            qf = build_filter(cond)
            res = client.query_points(
                collection_name=args.collection,
                query=q,
                limit=10,
                query_filter=qf,
                search_params=models.SearchParams(hnsw_ef=args.ef),
                with_payload=False,
                with_vectors=False,
            )

            found = [int(p.id) for p in res.points]

            n += 1
            p1 += precision_at_k(found, gt, 1)
            p3 += precision_at_k(found, gt, 3)
            p5 += precision_at_k(found, gt, 5)
            p10 += precision_at_k(found, gt, 10)

    total = time.time() - t0
    qps = n / total if total > 0 else 0.0

    record = {
        "Precision@1": p1 / n,
        "Precision@3": p3 / n,
        "Precision@5": p5 / n,
        "Precision@10": p10 / n,
        "total_search_time": total,
        "QPS": qps,
        "ef_search": args.ef,
    }

    with out_path.open("w", encoding="utf-8") as w:
        w.write(json.dumps(record) + "\n")

    print(record)


if __name__ == "__main__":
    main()