from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff


def run(url: str = "http://localhost:6333"):
    client = QdrantClient(url=url)

    client.recreate_collection(
        collection_name="single_unnamed",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    client.recreate_collection(
        collection_name="multiple_named",
        hnsw_config=HnswConfigDiff(m=32, ef_construct=256),
        vectors_config={
            "clip_default": VectorParams(size=512, distance=Distance.COSINE),
            "clip_tuned": VectorParams(
                size=512,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=36, ef_construct=300),
            ),
        },
    )


if __name__ == "__main__":
    run()