from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.data.features import load_features
from src.data.loader import load_fer2013
from src.utils.paths import CLASS_NAMES, FIGURES

plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#E5E3DC",
    "grid.alpha": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

ACCENT = "#2F5D50"


def _eda_dir():
    d = FIGURES / "eda"
    d.mkdir(parents=True, exist_ok=True)
    return d


def class_distribution():
    X, y, _ = load_fer2013("train")
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(CLASS_NAMES, counts, color=ACCENT, edgecolor="black", linewidth=0.6)
    ax.set_title("FER2013 training set — class distribution", pad=12)
    ax.set_ylabel("images")
    for i, c in enumerate(counts):
        ax.text(i, c + 60, str(c), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(_eda_dir() / "class_distribution.png", dpi=160)
    plt.close(fig)


def sample_grid(n_per_class: int = 4):
    X, y, _ = load_fer2013("train")
    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class, figsize=(n_per_class * 1.3, len(CLASS_NAMES) * 1.3))
    rng = np.random.default_rng(42)
    for r, name in enumerate(CLASS_NAMES):
        idx = np.where(y == r)[0]
        pick = rng.choice(idx, n_per_class, replace=False)
        for c, j in enumerate(pick):
            axes[r, c].imshow(X[j], cmap="gray")
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("FER2013 — sample faces per class", y=1.0)
    fig.tight_layout()
    fig.savefig(_eda_dir() / "sample_grid.png", dpi=160)
    plt.close(fig)


def mean_faces():
    X, y, _ = load_fer2013("train")
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(len(CLASS_NAMES) * 1.4, 1.8))
    for i, name in enumerate(CLASS_NAMES):
        mean = X[y == i].mean(axis=0)
        axes[i].imshow(mean, cmap="gray")
        axes[i].set_title(name, fontsize=9)
        axes[i].axis("off")
    fig.suptitle("Mean face per class", y=1.05)
    fig.tight_layout()
    fig.savefig(_eda_dir() / "mean_faces.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def pca_variance():
    import joblib
    from src.utils.paths import MODELS

    pca = joblib.load(MODELS / "pca.joblib")
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, color=ACCENT, linewidth=2)
    ax.axhline(0.9, color="#C06C4B", linestyle="--", linewidth=1, label="90% variance")
    ax.set_xlabel("components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("PCA on HOG features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_eda_dir() / "pca_variance.png", dpi=160)
    plt.close(fig)


def run_all():
    class_distribution()
    sample_grid()
    mean_faces()
    pca_variance()
    print("EDA figures written to", _eda_dir())


if __name__ == "__main__":
    run_all()
