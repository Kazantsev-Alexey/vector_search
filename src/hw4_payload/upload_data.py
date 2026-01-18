import argparse
import json
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="hw4")
    ap.add_argument("--data-dir", default="data/hw4")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=300)

    data_dir = Path(args.data_dir)
    vectors = np.load(data_dir / "vectors.npy", mmap_mode="r")
    payload_path = data_dir / "payloads.jsonl"

    with payload_path.open("r", encoding="utf-8") as f:
        batch = []
        for i, line in enumerate(f):
            payload = json.loads(line)
            v = vectors[i].astype(np.float32).tolist()
            batch.append(PointStruct(id=i, vector=v, payload=payload))

            if len(batch) >= args.batch:
                client.upsert(collection_name=args.collection, points=batch)
                print(f"upserted {i+1}")
                batch = []

        if batch:
            client.upsert(collection_name=args.collection, points=batch)
            print(f"upserted {i+1}")

    print("done")


if __name__ == "__main__":
    main()