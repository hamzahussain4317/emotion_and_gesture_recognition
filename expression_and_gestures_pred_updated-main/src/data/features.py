from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.data.loader import load_fer2013
from src.data.preprocess import flatten, scale_pixels
from src.utils.paths import FEATURES, MODELS


HOG_KWARGS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)


def raw_features(X: np.ndarray) -> np.ndarray:
    return flatten(scale_pixels(X))


def hog_features(X: np.ndarray) -> np.ndarray:
    out = []
    for img in tqdm(X, desc="hog", leave=False):
        out.append(hog(img, **HOG_KWARGS))
    return np.asarray(out, dtype=np.float32)


def build_all(n_components: int = 100, force: bool = False) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    pca_path = MODELS / "pca.joblib"

    for split in ("train", "test"):
        raw_path = FEATURES / f"raw_{split}.npz"
        hog_path = FEATURES / f"hog_{split}.npz"
        pca_feat_path = FEATURES / f"hog_pca_{split}.npz"
        paths[f"raw_{split}"] = raw_path
        paths[f"hog_{split}"] = hog_path
        paths[f"hog_pca_{split}"] = pca_feat_path

        if not force and raw_path.exists() and hog_path.exists() and pca_feat_path.exists():
            continue

        X, y, _ = load_fer2013(split)

        if force or not raw_path.exists():
            np.savez_compressed(raw_path, X=raw_features(X), y=y)

        if force or not hog_path.exists() or not pca_feat_path.exists():
            Xh = hog_features(X)
            np.savez_compressed(hog_path, X=Xh, y=y)

            if split == "train":
                pca = PCA(n_components=n_components, random_state=42)
                Xp = pca.fit_transform(Xh)
                joblib.dump(pca, pca_path)
            else:
                pca = joblib.load(pca_path)
                Xp = pca.transform(Xh)
            np.savez_compressed(pca_feat_path, X=Xp, y=y)

    return paths


def load_features(name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = FEATURES / f"{name}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run src.data.features.build_all() first")
    d = np.load(path)
    return d["X"], d["y"]


if __name__ == "__main__":
    build_all()
    print("feature files:")
    for p in sorted(FEATURES.glob("*.npz")):
        print(" ", p.name, f"{p.stat().st_size / 1e6:.1f} MB")
