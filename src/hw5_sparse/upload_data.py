import argparse
import json
import time
from pathlib import Path
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def chunks(seq, n):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def to_sparse(v):
    return models.SparseVector(indices=list(v.indices), values=list(v.values))

def ensure_collection(client: QdrantClient, name: str, recreate: bool):
    exists = client.collection_exists(name)
    if exists and not recreate:
        return
    if exists and recreate:
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            "splade": models.SparseVectorParams(),
        },
    )


def process_bm25(client, collection, corpus_path, bm25, batch_size):
    total = 0
    for doc_batch in chunks(read_jsonl(corpus_path), batch_size):
        texts = [f"{d.get('title','')}\n{d.get('text','')}".strip() for d in doc_batch]
        vecs = list(bm25.embed(texts))
        points = []
        for d, v in zip(doc_batch, vecs):
            pid = int(d["_id"])
            payload = {"_id": d["_id"], "title": d.get("title", ""), "text": d.get("text", "")}
            points.append(models.PointStruct(id=pid, payload=payload, vector={"bm25": to_sparse(v)}))
        client.upsert(collection_name=collection, points=points)
        total += len(doc_batch)
    return total


def process_splade(client, collection, corpus_path, splade, batch_size):
    for doc_batch in chunks(read_jsonl(corpus_path), batch_size):
        texts = [f"{d.get('title','')}\n{d.get('text','')}".strip() for d in doc_batch]
        vecs = list(splade.embed(texts))
        points = [
            models.PointStruct(id=int(d["_id"]), vector={"splade": to_sparse(v)})
            for d, v in zip(doc_batch, vecs)
        ]
        client.upsert(collection_name=collection, points=points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="hw5")
    ap.add_argument("--corpus", default="data/hw5/corpus.jsonl")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--splade-batch", type=int, default=8)
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=300)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"no such file: {corpus_path}")

    ensure_collection(client, args.collection, recreate=args.recreate)

    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    t0 = time.time()
    total_docs = process_bm25(client, args.collection, corpus_path, bm25, args.batch)
    t_bm25 = time.time() - t0
    del bm25

    splade = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    t0 = time.time()
    process_splade(client, args.collection, corpus_path, splade, args.splade_batch)
    t_splade = time.time() - t0

    print(
        {
        "docs": total_docs,
        "bm25_s": round(t_bm25, 3),
        "splade_s": round(t_splade, 3),
        }
    )


if __name__ == "__main__":
    main()