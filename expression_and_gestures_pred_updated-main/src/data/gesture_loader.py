from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.features import HOG_KWARGS
from src.utils.paths import CACHE, FEATURES, GESTURE_CLASSES, GESTURE_DATA, MODELS


def _folder_to_label(folder_name: str) -> int | None:
    parts = folder_name.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        idx = int(parts[0]) - 1
    except ValueError:
        return None
    return idx if 0 <= idx < len(GESTURE_CLASSES) else None


def _resolve_root() -> Path:
    for cand in (
        GESTURE_DATA,
        GESTURE_DATA.parent / "leapGestRecog" / "leapGestRecog",
        GESTURE_DATA.parent / "leapgestrecog",
    ):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"LeapGestRecog folder not found. Looked under {GESTURE_DATA}. "
        "Extract the Kaggle zip so one of those paths exists."
    )


def _image_paths() -> list[tuple[Path, int]]:
    root = _resolve_root()
    items: list[tuple[Path, int]] = []
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for cls_dir in sorted(subject_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = _folder_to_label(cls_dir.name)
            if label is None:
                continue
            for p in cls_dir.glob("*.png"):
                items.append((p, label))
    return items


def _read_image(path: Path, size: int = 64) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size))
    return np.asarray(img, dtype=np.uint8)


def load_gestures(size: int = 64, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    cache = CACHE / f"gestures_{size}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]

    items = _image_paths()
    if not items:
        raise RuntimeError("no gesture images found")

    X = np.empty((len(items), size, size), dtype=np.uint8)
    y = np.empty(len(items), dtype=np.int64)
    for i, (p, lbl) in enumerate(tqdm(items, desc="load gestures")):
        X[i] = _read_image(p, size)
        y[i] = lbl
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def build_features(n_components: int = 80, test_size: float = 0.2, seed: int = 42) -> dict[str, Path]:
    X, y = load_gestures(size=64)
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    def to_hog(arr: np.ndarray) -> np.ndarray:
        return np.asarray([hog(img, **HOG_KWARGS) for img in tqdm(arr, desc="hog", leave=False)], dtype=np.float32)

    H_train = to_hog(X_train_img)
    H_test = to_hog(X_test_img)

    pca = PCA(n_components=n_components, random_state=seed)
    P_train = pca.fit_transform(H_train)
    P_test = pca.transform(H_test)
    joblib.dump(pca, MODELS / "gesture_pca.joblib")

    paths = {
        "gesture_hog_pca_train": FEATURES / "gesture_hog_pca_train.npz",
        "gesture_hog_pca_test": FEATURES / "gesture_hog_pca_test.npz",
    }
    np.savez_compressed(paths["gesture_hog_pca_train"], X=P_train, y=y_train)
    np.savez_compressed(paths["gesture_hog_pca_test"], X=P_test, y=y_test)
    return paths


def load_gesture_features(split: str) -> tuple[np.ndarray, np.ndarray]:
    p = FEATURES / f"gesture_hog_pca_{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run src.data.gesture_loader.build_features()")
    d = np.load(p)
    return d["X"], d["y"]


if __name__ == "__main__":
    build_features()
    for name in ("train", "test"):
        X, y = load_gesture_features(name)
        print(f"{name}: X={X.shape}  classes={np.bincount(y).tolist()}")
