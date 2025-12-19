import argparse
import json
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


@dataclass(frozen=True)
class UploadConfig:
    vectors_path: str = "data/hw3/vectors.npy"
    payloads_path: str = "data/hw3/payloads.jsonl"

    single_collection: str = "single_unnamed"
    multi_collection: str = "multiple_named"
    vec1: str = "clip_default"
    vec2: str = "clip_tuned"

    batch_size: int = 512
    parallel: int = 4


def load_vectors(path: str) -> np.ndarray:
    return np.load(Path(path), mmap_mode="r")


def iter_payloads(path: str) -> Generator[dict, None, None]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def batched(it: Iterable, batch_size: int) -> Generator[list, None, None]:
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def iter_points_single(vectors: np.ndarray, payloads: Iterable[dict]) -> Generator[PointStruct, None, None]:
    for idx, payload in enumerate(payloads):
        yield PointStruct(id=idx, vector=vectors[idx].astype(np.float32).tolist(), payload=payload)


def iter_points_multi(
    vectors: np.ndarray,
    payloads: Iterable[dict],
    vec1: str,
    vec2: str,
) -> Generator[PointStruct, None, None]:
    for idx, payload in enumerate(payloads):
        v = vectors[idx].astype(np.float32).tolist()
        yield PointStruct(id=idx, vector={vec1: v, vec2: v}, payload=payload)


def qps_label(done: int, t0: float) -> str:
    dt = max(perf_counter() - t0, 1e-9)
    return f"{done:,} ({done / dt:.1f} qps)"


def upsert_batched(client: QdrantClient, collection: str, points: Iterable[PointStruct], batch_size: int) -> None:
    t0 = perf_counter()
    done = 0
    for batch in batched(points, batch_size):
        client.upsert(collection_name=collection, points=batch)
        done += len(batch)
        if done % 5000 == 0:
            print(f"{collection}: {qps_label(done, t0)}")
    print(f"{collection}: done {qps_label(done, t0)}")


def upload_points_fast(
    client: QdrantClient,
    collection: str,
    points: Iterable[PointStruct],
    batch_size: int,
    parallel: int,
) -> None:
    t0 = perf_counter()
    client.upload_points(collection_name=collection, points=points, batch_size=batch_size, parallel=parallel)
    print(f"{collection}: upload_points {perf_counter() - t0:.2f}s")


def upload_collection_fast(
    client: QdrantClient,
    collection: str,
    points: Iterable[PointStruct],
    batch_size: int,
    parallel: int,
) -> None:
    t0 = perf_counter()
    client.upload_collection(collection_name=collection, vectors=points, batch_size=batch_size, parallel=parallel)
    print(f"{collection}: upload_collection {perf_counter() - t0:.2f}s")


def make_client(url: str, prefer_grpc: bool) -> QdrantClient:
    return QdrantClient(url=url, prefer_grpc=prefer_grpc)


def run_upsert(client: QdrantClient, cfg: UploadConfig, vectors: np.ndarray) -> None:
    upsert_batched(
        client,
        cfg.single_collection,
        iter_points_single(vectors, iter_payloads(cfg.payloads_path)),
        batch_size=cfg.batch_size,
    )
    upsert_batched(
        client,
        cfg.multi_collection,
        iter_points_multi(vectors, iter_payloads(cfg.payloads_path), cfg.vec1, cfg.vec2),
        batch_size=cfg.batch_size,
    )


def run_fast(client: QdrantClient, cfg: UploadConfig, vectors: np.ndarray) -> None:
    upload_points_fast(
        client,
        cfg.single_collection,
        iter_points_single(vectors, iter_payloads(cfg.payloads_path)),
        batch_size=cfg.batch_size,
        parallel=cfg.parallel,
    )
    upload_collection_fast(
        client,
        cfg.multi_collection,
        iter_points_multi(vectors, iter_payloads(cfg.payloads_path), cfg.vec1, cfg.vec2),
        batch_size=cfg.batch_size,
        parallel=cfg.parallel,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:6333")
    p.add_argument("--grpc", action="store_true")
    p.add_argument("--mode", choices=["upsert", "fast"], default="upsert")
    p.add_argument("--batch-size", type=int, default=UploadConfig.batch_size)
    p.add_argument("--parallel", type=int, default=UploadConfig.parallel)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = UploadConfig(batch_size=args.batch_size, parallel=args.parallel)
    client = make_client(args.url, prefer_grpc=args.grpc)
    vectors = load_vectors(cfg.vectors_path)

    if args.mode == "upsert":
        run_upsert(client, cfg, vectors)
    else:
        run_fast(client, cfg, vectors)


if __name__ == "__main__":
    main()