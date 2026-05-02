from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def scale_pixels(X: np.ndarray) -> np.ndarray:
    return X.astype(np.float32) / 255.0


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)
