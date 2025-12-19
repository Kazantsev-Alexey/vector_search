import argparse
from qdrant_client import QdrantClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--single", default="single_unnamed")
    ap.add_argument("--multi", default="multiple_named")
    args = ap.parse_args()

    client = QdrantClient(url=args.url)

    points, _ = client.scroll(
        collection_name=args.single,
        limit=10,
        with_payload=False,
        with_vectors=True,
    )
    ids = [p.id for p in points]
    print("single_unnamed ids:", ids)

    payloads = client.retrieve(
        collection_name=args.single,
        ids=ids,
        with_payload=True,
        with_vectors=False,
    )
    print("single_unnamed payload[0]:", payloads[0].payload if payloads else None)

    offset = None
    got = 0
    while got < 100:
        batch, offset = client.scroll(
            collection_name=args.multi,
            limit=10,
            offset=offset,
            with_payload=False,
            with_vectors=["clip_default"],
        )
        got += len(batch)
        if not batch or offset is None:
            break
    print("multiple_named read:", got)


if __name__ == "__main__":
    main()