import json
from pathlib import Path
from time import perf_counter
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SearchParams, VectorParams, QueryRequest

BATCH_SIZE = 2048
SEARCH_BATCH_SIZE = 256
TOP_K = 10


def upload_vectors(client: QdrantClient, collection: str, vectors: np.ndarray) -> None:
    n = vectors.shape[0]
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        points = [
            PointStruct(id=i, vector=vectors[i].tolist())
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection, points=points)


def build_ground_truth(
    client: QdrantClient,
    collection: str,
    vectors: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = vectors.shape[0]
    results = {}

    t0 = perf_counter()
    for start in range(0, n, SEARCH_BATCH_SIZE):
        end = min(start + SEARCH_BATCH_SIZE, n)
        requests = [
            QueryRequest(
                query=vectors[i].tolist(),
                limit=TOP_K + 1,
                params=SearchParams(exact=True),
                with_payload=False,
                with_vector=False,
            )
            for i in range(start, end)
        ]

        batch_results = client.query_batch_points(collection_name=collection, requests=requests)

        for idx, res in enumerate(batch_results):
            i = start + idx
            ids = [int(p.id) for p in res.points if int(p.id) != i][:TOP_K]
            results[i] = ids

        if end % 5000 < SEARCH_BATCH_SIZE:
            print(f"{end}/{n} ({end / (perf_counter() - t0):.0f} qps)")

    with output_path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({str(i): results[i]}) + "\n")

    print(f"Done: {perf_counter() - t0:.1f}s")


def run():
    vectors = np.load("data/hw3/vectors.npy", mmap_mode="r").astype(np.float32)
    n, dim = vectors.shape
    # я пытался запустить в локал режиме QdrantClient(":memory:")
    # и получил 11qps =)
    client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True, timeout=300)

    if client.collection_exists("hw3_vectors"):
        client.delete_collection("hw3_vectors")
    client.create_collection(
        collection_name="hw3_vectors",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    upload_vectors(client, "hw3_vectors", vectors)
    build_ground_truth(client, "hw3_vectors", vectors, Path("results/hw3/ground_truth.jsonl"))


if __name__ == "__main__":
    run()
