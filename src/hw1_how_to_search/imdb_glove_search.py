import re
import tarfile
import zipfile
import urllib.request
from pathlib import Path
import random
import numpy as np
from .distances_numpy import cosine_similarity

DATA_DIR = Path("data")
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    imdb_tar = DATA_DIR / "aclImdb_v1.tar.gz"
    if not imdb_tar.exists():
        urllib.request.urlretrieve(IMDB_URL, imdb_tar)

    imdb_root = DATA_DIR / "aclImdb"
    if not imdb_root.exists():
        with tarfile.open(imdb_tar, "r:gz") as tar:
            tar.extractall(DATA_DIR)

    texts: list[str] = []
    for label in ("pos", "neg"):
        for p in sorted((imdb_root / "train" / label).glob("*.txt")):
            texts.append(p.read_text(encoding="utf-8", errors="ignore").strip())
            if len(texts) >= 50_000:
                break
        if len(texts) >= 50_000:
            break

    glove_zip = DATA_DIR / "glove.6B.zip"
    if not glove_zip.exists():
        urllib.request.urlretrieve(GLOVE_URL, glove_zip)

    glove_txt = DATA_DIR / "glove.6B.50d.txt"
    if not glove_txt.exists():
        with zipfile.ZipFile(glove_zip) as zf, zf.open("glove.6B.50d.txt") as fin, glove_txt.open(
            "wb"
        ) as fout:
            fout.write(fin.read())

    glove: dict[str, np.ndarray] = {}
    with glove_txt.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            glove[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    dim = len(next(iter(glove.values())))

    def sent_vec(text: str) -> np.ndarray:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        vecs = [glove[t] for t in tokens if t in glove]
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        return np.mean(vecs, axis=0)


    matrix = np.vstack([sent_vec(t) for t in texts])
    rnd = random.Random(0)
    for idx in rnd.sample(range(len(texts)), k=10):
        query_text = texts[idx]
        query_vec = matrix[idx : idx + 1]
        sims = cosine_similarity(query_vec, matrix)[0]
        top = np.argsort(sims)[::-1][:5]

        print("\n" + "=" * 80)
        print("QUERY:\n", query_text, "\n")
        print("NEAREST NEIGHBOURS:")
        for j in top:
            print(f"\n[score={sims[j]:.4f}]\n{texts[j]}\n")


if __name__ == "__main__":
    main()