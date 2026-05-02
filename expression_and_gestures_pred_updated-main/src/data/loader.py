from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.paths import CACHE, CLASS_NAMES, DATA


def _image_paths(split_dir: Path) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for label, name in enumerate(CLASS_NAMES):
        cls_dir = split_dir / name
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.glob("*.jpg"):
            items.append((p, label))
        for p in cls_dir.glob("*.png"):
            items.append((p, label))
    return items


def _read_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((48, 48))
    return np.asarray(img, dtype=np.uint8)


def load_fer2013(split: str = "train", use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    assert split in {"train", "test"}
    cache_file = CACHE / f"{split}.npz"

    if use_cache and cache_file.exists():
        data = np.load(cache_file)
        return data["X"], data["y"], CLASS_NAMES

    split_dir = DATA / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"FER2013 split folder not found: {split_dir}\n"
            "Download from Kaggle (msambare/fer2013) and extract to data/fer2013/"
        )

    items = _image_paths(split_dir)
    if not items:
        raise RuntimeError(f"No images found under {split_dir}")

    X = np.empty((len(items), 48, 48), dtype=np.uint8)
    y = np.empty(len(items), dtype=np.int64)
    for i, (path, label) in enumerate(tqdm(items, desc=f"load {split}")):
        X[i] = _read_image(path)
        y[i] = label

    np.savez_compressed(cache_file, X=X, y=y)
    return X, y, CLASS_NAMES


def load_subset(
    split: str = "train",
    per_class: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, names = load_fer2013(split)
    if per_class is None:
        return X, y, names

    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for label in range(len(names)):
        idx = np.where(y == label)[0]
        if len(idx) > per_class:
            idx = rng.choice(idx, size=per_class, replace=False)
        keep.extend(idx.tolist())
    keep_arr = np.array(sorted(keep))
    return X[keep_arr], y[keep_arr], names
