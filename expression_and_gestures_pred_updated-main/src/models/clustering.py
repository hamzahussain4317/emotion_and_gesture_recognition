from __future__ import annotations

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.data.features import load_features
from src.utils.paths import CLASS_NAMES, FIGURES, METRICS, MODELS
from src.utils.seed import set_seed


def _tsne_plot(X2: np.ndarray, labels: np.ndarray, title: str, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("tab10")
    for i in range(int(labels.max()) + 1):
        mask = labels == i
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
        ax.scatter(X2[mask, 0], X2[mask, 1], s=5, alpha=0.5, label=name, color=cmap(i))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=2, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(feature_set: str = "hog_pca", tsne_subset: int = 4000) -> dict:
    set_seed()
    X_train, y_train = load_features(feature_set, "train")

    k = len(CLASS_NAMES)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    t0 = time.perf_counter()
    clusters = km.fit_predict(X_train)
    fit_t = time.perf_counter() - t0

    sample_idx = np.random.default_rng(42).choice(len(X_train), size=min(5000, len(X_train)), replace=False)
    sil = silhouette_score(X_train[sample_idx], clusters[sample_idx])
    dbi = davies_bouldin_score(X_train, clusters)
    ari = adjusted_rand_score(y_train, clusters)

    joblib.dump(km, MODELS / "kmeans.joblib")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), size=min(tsne_subset, len(X_train)), replace=False)
    print(f"[kmeans] running t-SNE on {len(idx)} points...")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42, max_iter=1000)
    X2 = tsne.fit_transform(X_train[idx])
    _tsne_plot(X2, clusters[idx], "K-Means clusters (t-SNE)", FIGURES / "tsne_clusters.png")
    _tsne_plot(X2, y_train[idx], "True emotion labels (t-SNE)", FIGURES / "tsne_labels.png")

    report = {
        "model": "kmeans",
        "n_clusters": k,
        "silhouette": float(sil),
        "davies_bouldin": float(dbi),
        "adjusted_rand_index": float(ari),
        "train_time_s": fit_t,
    }
    with open(METRICS / "kmeans.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[kmeans] silhouette={sil:.3f}  DBI={dbi:.3f}  ARI={ari:.3f}")
    return report


if __name__ == "__main__":
    run()
