import argparse
import json
import time
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client import models


MODEL_BY_VECTOR = {
    "bm25": "Qdrant/Bm25",
    "splade": "prithivida/Splade_PP_en_v1",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_queries(path: Path):
    yield from read_jsonl(path)


def load_ground_truth(default_path: Path) -> dict[str, set[str]]:
    gt: dict[str, set[str]] = {}
    for row in read_jsonl(default_path):
        if row.get("score", 1) != 1:
            continue
        qid = str(row["query-id"])
        gt.setdefault(qid, set()).add(str(row["corpus-id"]))
    return gt


def mrr_at_k(found_ids: list[str], relevant: set[str], k: int = 10) -> float:
    for rank, pid in enumerate(found_ids[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def run_method(
    client: QdrantClient,
    collection: str,
    queries_path: Path,
    gt: dict[str, set[str]],
    vector_name: str,
    limit: int = 10,
) -> dict:
    n = 0
    mrr_sum = 0.0

    t0 = time.time()
    for q in iter_queries(queries_path):
        qid = str(q.get("_id"))
        text = q.get("text")
        if not text:
            continue

        relevant = gt.get(qid)
        if not relevant:
            continue

        res = client.query_points(
            collection_name=collection,
            query=models.Document(text=text, model=MODEL_BY_VECTOR[vector_name]),
            using=vector_name,
            limit=limit,
            with_payload=False,
            with_vectors=False,
        )

        found = [str(p.id) for p in res.points]

        n += 1
        mrr_sum += mrr_at_k(found, relevant, k=limit)

    total = time.time() - t0
    qps = (n / total) if total > 0 else 0.0

    return {
        "MRR@10": (mrr_sum / n) if n else 0.0,
        "total_search_time": total,
        "QPS": qps,
        "queries_used": n,
        "method": vector_name,
    }


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="hw5")
    ap.add_argument("--queries", default="data/hw5/queries.jsonl")
    ap.add_argument("--default", required=True)  # например data/hw5/default_test.jsonl
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out_bm25", default="results/hw5/bm25_search_results.jsonl")
    ap.add_argument("--out_splade", default="results/hw5/splade_search_results.jsonl")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=300)

    queries_path = Path(args.queries)
    default_path = Path(args.default)

    gt = load_ground_truth(default_path)

    bm25_record = run_method(
        client, args.collection, queries_path, gt, vector_name="bm25", limit=args.limit
    )
    splade_record = run_method(
        client, args.collection, queries_path, gt, vector_name="splade", limit=args.limit
    )

    write_jsonl(Path(args.out_bm25), bm25_record)
    write_jsonl(Path(args.out_splade), splade_record)

    print("bm25:", bm25_record)
    print("splade:", splade_record)


if __name__ == "__main__":
    main()