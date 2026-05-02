from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import joblib
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data.features import load_features
from src.evaluation.metrics import full_classification_report
from src.utils.paths import MODELS
from src.utils.seed import set_seed


@dataclass
class ClassifierSpec:
    name: str
    build: Callable[[], BaseEstimator]


SPECS: list[ClassifierSpec] = [
    ClassifierSpec(
        name="knn",
        build=lambda: KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
    ),
    ClassifierSpec(
        name="svm",
        build=lambda: SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42),
    ),
    ClassifierSpec(
        name="decision_tree",
        build=lambda: DecisionTreeClassifier(max_depth=15, random_state=42),
    ),
]


def train_one(spec: ClassifierSpec, X_train, y_train, X_test, y_test) -> dict:
    model = spec.build()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    joblib.dump(model, MODELS / f"{spec.name}.joblib")
    report = full_classification_report(spec.name, y_test, y_pred, train_time, predict_time)
    report["model"] = spec.name
    return report


def run(feature_set: str = "hog_pca", svm_max_train: int | None = 12000) -> list[dict]:
    set_seed()
    X_train, y_train = load_features(feature_set, "train")
    X_test, y_test = load_features(feature_set, "test")

    reports = []
    for spec in SPECS:
        Xt, yt = X_train, y_train
        if spec.name == "svm" and svm_max_train and len(X_train) > svm_max_train:
            import numpy as np
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), svm_max_train, replace=False)
            Xt, yt = X_train[idx], y_train[idx]
        print(f"[{spec.name}] training on {len(Xt)} samples ({feature_set})")
        rep = train_one(spec, Xt, yt, X_test, y_test)
        print(
            f"  acc={rep['accuracy']:.4f}  f1={rep['f1_macro']:.4f}  "
            f"train={rep['train_time_s']:.1f}s  pred={rep['predict_time_s']:.1f}s"
        )
        reports.append(rep)
    return reports


if __name__ == "__main__":
    run()
