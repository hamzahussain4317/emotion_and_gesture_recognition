from __future__ import annotations

import time

import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data.gesture_loader import load_gesture_features
from src.evaluation.metrics import classification_metrics, save_confusion, save_metrics
from src.utils.paths import GESTURE_CLASSES, MODELS
from src.utils.seed import set_seed


BUILDERS = {
    "gesture_knn": lambda: KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
    "gesture_svm": lambda: SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42),
    "gesture_decision_tree": lambda: DecisionTreeClassifier(max_depth=20, random_state=42),
}


def _report(name, y_true, y_pred, train_time, predict_time):
    payload = classification_metrics(y_true, y_pred)
    payload["model"] = name
    payload["train_time_s"] = float(train_time)
    payload["predict_time_s"] = float(predict_time)
    save_metrics(name, payload)
    save_confusion(name, y_true, y_pred, class_names=GESTURE_CLASSES)
    return payload


def run() -> list[dict]:
    set_seed()
    X_train, y_train = load_gesture_features("train")
    X_test, y_test = load_gesture_features("test")

    reports = []
    for name, build in BUILDERS.items():
        print(f"[{name}] training on {len(X_train)} samples")
        model = build()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - t0

        joblib.dump(model, MODELS / f"{name}.joblib")
        rep = _report(name, y_test, y_pred, train_time, predict_time)
        print(
            f"  acc={rep['accuracy']:.4f}  f1={rep['f1_macro']:.4f}  "
            f"train={rep['train_time_s']:.1f}s  pred={rep['predict_time_s']:.1f}s"
        )
        reports.append(rep)
    return reports


if __name__ == "__main__":
    run()
