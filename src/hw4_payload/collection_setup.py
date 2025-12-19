import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="hw4")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=60)

    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    print("created:", args.collection)


if __name__ == "__main__":
    main()