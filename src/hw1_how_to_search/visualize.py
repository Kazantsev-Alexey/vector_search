import time
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def run() -> None:
    texts, matrix = load_for_visualization()

    sample_sizes = [500, 5000, 25000]

    for size in sample_sizes:
        print(f"\n=== SAMPLE SIZE: {size} ===")

        idx = np.random.choice(len(matrix), size, replace=False)
        X = matrix[idx]

        # PCA
        t0 = time.time()
        pca = PCA(n_components=2).fit_transform(X)
        t1 = time.time()
        print(f"PCA   time: {t1 - t0:.2f} sec")
        save_scatter(pca, f"pca_{size}.png", f"PCA ({size} samples)")

        # t-SNE
        if size <= 5000:
            t0 = time.time()
            tsne = TSNE(
                n_components=2,
                perplexity=30,
                learning_rate="auto",
                init="random"
            ).fit_transform(X)
            t1 = time.time()
            print(f"t-SNE time: {t1 - t0:.2f} sec")
            save_scatter(tsne, f"tsne_{size}.png", f"t-SNE ({size} samples)")
        else:
            print("t-SNE skipped (too slow for this sample size)")

        # UMAP
        t0 = time.time()
        umap2 = UMAP(n_components=2).fit_transform(X)
        t1 = time.time()
        print(f"UMAP  time: {t1 - t0:.2f} sec")
        save_scatter(umap2, f"umap_{size}.png", f"UMAP ({size} samples)")

    print("\nDone. Saved all images into:", FIG_DIR)


def save_scatter(points, filename, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=3, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()


def load_for_visualization():
    DATA_DIR = Path("data")
    imdb_root = DATA_DIR / "aclImdb"

    texts = []
    for label in ("pos", "neg"):
        for p in sorted((imdb_root / "train" / label).glob("*.txt")):
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            if len(texts) >= 50_000:
                break
        if len(texts) >= 50_000:
            break

    glove_txt = DATA_DIR / "glove.6B.50d.txt"
    glove = {}
    with glove_txt.open("r", encoding="utf-8") as f:
        for line in f:
            w, *vals = line.split()
            glove[w] = np.asarray(vals, dtype=np.float32)

    dim = len(next(iter(glove.values())))

    def sent_vec(text: str) -> np.ndarray:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        vecs = [glove[t] for t in tokens if t in glove]
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    matrix = np.vstack([sent_vec(t) for t in texts])
    return texts, matrix


if __name__ == "__main__":
    run()