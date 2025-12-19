import argparse
from qdrant_client import QdrantClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="single_unnamed")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=300)
    
    points, _ = client.scroll(
        collection_name=args.collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    pid = points[0].id

    client.delete_payload(
        collection_name=args.collection,
        keys=["__spoiled__"],
        points=[pid],
    )
    print("payload deleted")

    snap = client.create_snapshot(collection_name=args.collection, wait=True)
    snap_name = snap.name
    print("snapshot:", snap_name)

    points, _ = client.scroll(
        collection_name=args.collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    pid = points[0].id

    client.set_payload(
        collection_name=args.collection,
        payload={"__spoiled__": True},
        points=[pid],
    )

    after = client.retrieve(
        collection_name=args.collection,
        ids=[pid],
        with_payload=True,
        with_vectors=False,
    )[0].payload
    assert after.get("__spoiled__") is True

    client.recover_snapshot(
        collection_name=args.collection,
        location=f"file:///qdrant/snapshots/{args.collection}/{snap_name}",
        wait=True,
    )

    restored = client.retrieve(
        collection_name=args.collection,
        ids=[pid],
        with_payload=True,
        with_vectors=False,
    )[0].payload
    assert restored.get("__spoiled__") is not True

    print("restored")


if __name__ == "__main__":
    main()