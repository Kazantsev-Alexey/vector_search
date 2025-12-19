import argparse
from qdrant_client import QdrantClient, models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="hw5")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=300)

    client.recreate_collection(
        collection_name=args.collection,
        vectors_config={},
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            "splade": models.SparseVectorParams(),
        },
    )
    print("collection:", args.collection)


if __name__ == "__main__":
    main()