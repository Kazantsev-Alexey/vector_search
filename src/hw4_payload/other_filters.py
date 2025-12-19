import argparse
import random
import time
import numpy as np
from qdrant_client import QdrantClient, models
from hw4_payload.payload_generator import one_random_payload_please

COLLECTION = "hw4_other_filters"
DIM = 512
N_POINTS = 200_000
BATCH = 1000


def fix_payload(p: dict) -> dict:
    dt = p.get("rand_datetime")
    if hasattr(dt, "isoformat"):
        p["rand_datetime"] = dt.isoformat()
    t = p.get("rand_tuple")
    if isinstance(t, tuple):
        p["rand_tuple"] = list(t)
    return p


def recreate_collection(client: QdrantClient, name: str):
    if client.collection_exists(name):
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    )


def create_payload_indexes(client: QdrantClient, name: str):
    idx = [
        ("rand_digit", models.PayloadSchemaType.INTEGER),
        ("rand_signed_int", models.PayloadSchemaType.INTEGER),
        ("rand_number", models.PayloadSchemaType.FLOAT),
        ("rand_bool", models.PayloadSchemaType.BOOL),
        ("words", models.PayloadSchemaType.KEYWORD),
        ("two_words", models.PayloadSchemaType.KEYWORD),
        ("id_str", models.PayloadSchemaType.KEYWORD),
        ("city.name", models.PayloadSchemaType.KEYWORD),
    ]
    for key, t in idx:
        client.create_payload_index(collection_name=name, field_name=key, field_schema=t)


def upload(client: QdrantClient, name: str):
    rng = np.random.default_rng(42)

    ids = []
    vecs = []
    pays = []

    t0 = time.time()
    for i in range(N_POINTS):
        v = rng.standard_normal(DIM).astype(np.float32)
        p = fix_payload(one_random_payload_please(i))

        ids.append(i)
        vecs.append(v.tolist())
        pays.append(p)

        if len(ids) == BATCH:
            client.upsert(
                collection_name=name,
                points=models.Batch(ids=ids, vectors=vecs, payloads=pays),
            )
            ids.clear()
            vecs.clear()
            pays.clear()

            if (i + 1) % 10_000 == 0:
                print("uploaded:", i + 1)

    if ids:
        client.upsert(
            collection_name=name,
            points=models.Batch(ids=ids, vectors=vecs, payloads=pays),
        )

    print("upload done in", round(time.time() - t0, 2), "sec")


def random_query():
    rng = np.random.default_rng(123)
    return rng.standard_normal(DIM).astype(np.float32).tolist()


def run_filters(client: QdrantClient, name: str):
    q = random_query()

    filters = [
        ("rand_digit == 3",
         models.Filter(must=[models.FieldCondition(key="rand_digit", match=models.MatchValue(value=3))])),

        ("rand_signed_int > 0",
         models.Filter(must=[models.FieldCondition(key="rand_signed_int", range=models.Range(gt=0))])),

        ("range: 0.2 <= rand_number <= 0.4",
         models.Filter(must=[models.FieldCondition(key="rand_number", range=models.Range(gte=0.2, lte=0.4))])),

        ("bool filter",
         models.Filter(must=[models.FieldCondition(key="rand_bool", match=models.MatchValue(value=True))])),

        ("words == 'cat dog'",
         models.Filter(must=[models.FieldCondition(key="words", match=models.MatchValue(value="cat dog"))])),

        ("two_words has 'cat'",
         models.Filter(must=[models.FieldCondition(key="two_words", match=models.MatchValue(value="cat"))])),

        ("id_str match '07'",
         models.Filter(must=[models.FieldCondition(key="id_str", match=models.MatchValue(value="07"))])),

        ("nested: city.name",
         models.Filter(must=[models.FieldCondition(key="city.name", match=models.MatchValue(value="Berlin"))])),

        ("should (OR)",
         models.Filter(should=[
             models.FieldCondition(key="rand_digit", match=models.MatchValue(value=1)),
             models.FieldCondition(key="rand_digit", match=models.MatchValue(value=2)),
         ])),

        ("must + must_not combo",
         models.Filter(
             must=[models.FieldCondition(key="rand_digit", match=models.MatchValue(value=5))],
             must_not=[models.FieldCondition(key="rand_bool", match=models.MatchValue(value=True))],
         )),
    ]

    for title, flt in filters:
        res = client.query_points(
            collection_name=name,
            query=q,
            limit=10,
            query_filter=flt,
            search_params=models.SearchParams(hnsw_ef=64),
        )
        print(title, "->", len(res.points))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--mode", choices=["setup", "upload", "run", "all"], default="all")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=300)

    if args.mode in ("setup", "all"):
        recreate_collection(client, COLLECTION)
        create_payload_indexes(client, COLLECTION)
        print("collection + payload indexes: done")

    if args.mode in ("upload", "all"):
        upload(client, COLLECTION)

    if args.mode in ("run", "all"):
        run_filters(client, COLLECTION)


if __name__ == "__main__":
    main()