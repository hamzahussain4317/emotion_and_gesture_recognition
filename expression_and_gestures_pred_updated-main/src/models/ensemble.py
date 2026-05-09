from __future__ import annotations

import time

import joblib
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data.features import load_features
from src.evaluation.metrics import full_classification_report
from src.utils.paths import MODELS
from src.utils.seed import set_seed


def _bagging() -> BaggingClassifier:
    base = DecisionTreeClassifier(max_depth=15, random_state=42)
    return BaggingClassifier(
        estimator=base,
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )


def _adaboost() -> AdaBoostClassifier:
    base = DecisionTreeClassifier(max_depth=5, random_state=42)
    return AdaBoostClassifier(
        estimator=base,
        n_estimators=100,
        learning_rate=0.5,
        random_state=42,
    )


BUILDERS = {
    "bagging_dt": _bagging,
    "adaboost_dt": _adaboost,
}


def run(feature_set: str = "hog_pca") -> list[dict]:
    set_seed()
    X_train, y_train = load_features(feature_set, "train")
    X_test, y_test = load_features(feature_set, "test")

    reports = []
    for name, build in BUILDERS.items():
        model = build()
        print(f"[{name}] training...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - t0

        joblib.dump(model, MODELS / f"{name}.joblib")
        rep = full_classification_report(name, y_test, y_pred, train_time, predict_time)
        rep["model"] = name
        print(
            f"  acc={rep['accuracy']:.4f}  f1={rep['f1_macro']:.4f}  "
            f"train={rep['train_time_s']:.1f}s  pred={rep['predict_time_s']:.1f}s"
        )
        reports.append(rep)
    return reports


if __name__ == "__main__":
    run()
