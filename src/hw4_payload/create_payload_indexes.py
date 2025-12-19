import argparse
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType


def iter_condition_fields(cond):
    if not isinstance(cond, dict):
        return

    if "and" in cond:
        for x in cond["and"]:
            yield from iter_condition_fields(x)
        return

    if "or" in cond:
        for x in cond["or"]:
            yield from iter_condition_fields(x)
        return

    if "not" in cond:
        yield from iter_condition_fields(cond["not"])
        return

    for field, spec in cond.items():
        if not isinstance(spec, dict):
            continue
        yield field, spec


def guess_schema(spec):
    if "range" in spec:
        return PayloadSchemaType.FLOAT

    if "match" in spec and isinstance(spec["match"], dict) and "value" in spec["match"]:
        v = spec["match"]["value"]
        if isinstance(v, bool):
            return PayloadSchemaType.BOOL
        if isinstance(v, int):
            return PayloadSchemaType.INTEGER
        if isinstance(v, float):
            return PayloadSchemaType.FLOAT
        return PayloadSchemaType.KEYWORD

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="hw4")
    ap.add_argument("--tests", default="data/hw4/tests.jsonl")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=60)

    fields = {}
    p = Path(args.tests)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            cond = t.get("conditions", {})
            for field, spec in iter_condition_fields(cond):
                schema = guess_schema(spec)
                if schema is not None:
                    fields[field] = schema

    for field, schema in fields.items():
        client.create_payload_index(
            collection_name=args.collection,
            field_name=field,
            field_schema=schema,
        )
        print("index:", field, schema)

    print("done")


if __name__ == "__main__":
    main()