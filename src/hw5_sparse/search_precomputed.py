import argparse
import json
import random
import time
from pathlib import Path
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models


SPLADE_MODEL = "prithivida/Splade_PP_en_v1"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sample_queries(path: Path, n: int = 100):
    result = []
    for i, item in enumerate(read_jsonl(path)):
        if i < n:
            result.append(item)
        else:
            j = random.randint(0, i)
            if j < n:
                result[j] = item
    return result


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    gt: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        qid = str(row["query-id"])
        gt.setdefault(qid, set()).add(str(row["corpus-id"]))
    return gt


def mrr_at_10(found: list[str], relevant: set[str]) -> float:
    for i, pid in enumerate(found[:10], start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="hw5")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--default", required=True)
    ap.add_argument("--out", default="results/hw5/splade_precomputed_search_results.jsonl")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=300)
    encoder = SparseTextEmbedding(model_name=SPLADE_MODEL)

    queries = sample_queries(Path(args.queries), n=100)
    gt = load_ground_truth(Path(args.default))

    t_embed_start = time.time()
    texts = [q["text"] for q in queries]
    sparse_vecs = list(encoder.embed(texts))
    embeddings = [
        (str(q["_id"]), models.SparseVector(indices=v.indices.tolist(), values=v.values.tolist()))
        for q, v in zip(queries, sparse_vecs)
    ]
    embed_time = time.time() - t_embed_start

    t_search_start = time.time()
    mrr_sum = 0.0
    used = 0

    for qid, vec in embeddings:
        if qid not in gt:
            continue

        res = client.query_points(
            collection_name=args.collection,
            query=vec,
            using="splade",
            limit=10,
            with_payload=False,
            with_vectors=False,
        )

        found = [str(p.id) for p in res.points]
        mrr_sum += mrr_at_10(found, gt[qid])
        used += 1

    search_time = time.time() - t_search_start
    qps = used / search_time if search_time > 0 else 0.0

    record = {
        "MRR@10": mrr_sum / used if used else 0.0,
        "total_search_time": search_time,
        "QPS": qps,
        "queries_used": used,
        "embedding_time": embed_time,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(record)


if __name__ == "__main__":
    main()