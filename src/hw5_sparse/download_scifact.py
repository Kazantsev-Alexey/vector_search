from pathlib import Path
import json
from datasets import load_dataset


def dump_jsonl(ds, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    out_dir = Path("data/hw5")

    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries = load_dataset("mteb/scifact", "queries", split="queries")
    default_train = load_dataset("mteb/scifact", "default", split="train")
    default_test = load_dataset("mteb/scifact", "default", split="test")

    dump_jsonl(corpus, out_dir / "corpus.jsonl")
    dump_jsonl(queries, out_dir / "queries.jsonl")
    dump_jsonl(default_train, out_dir / "default_train.jsonl")
    dump_jsonl(default_test, out_dir / "default_test.jsonl")

    print("done")


if __name__ == "__main__":
    main()