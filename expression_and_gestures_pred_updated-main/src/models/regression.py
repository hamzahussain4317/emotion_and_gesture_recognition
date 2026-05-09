from __future__ import annotations

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.data.features import load_features
from src.utils.paths import CLASS_NAMES, FIGURES, METRICS, MODELS, VALENCE
from src.utils.seed import set_seed

ACCENT = "#2F5D50"


def labels_to_valence(y: np.ndarray) -> np.ndarray:
    table = np.array([VALENCE[name] for name in CLASS_NAMES], dtype=np.float32)
    return table[y]


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _scatter(name: str, y_true, y_pred, y_class):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = plt.get_cmap("tab10")
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_class == i
        ax.scatter(y_true[mask], y_pred[mask], s=6, alpha=0.5, label=cls, color=colors(i))
    lims = [-1.0, 1.0]
    ax.plot(lims, lims, color="#333333", linewidth=1, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("true valence")
    ax.set_ylabel("predicted valence")
    ax.set_title(f"Valence prediction — {name}")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"regression_{name}.png", dpi=160)
    plt.close(fig)


def run(feature_set: str = "hog_pca") -> list[dict]:
    set_seed()
    X_train, y_train = load_features(feature_set, "train")
    X_test, y_test = load_features(feature_set, "test")
    v_train = labels_to_valence(y_train)
    v_test = labels_to_valence(y_test)

    models = {
        "linear_regression": LinearRegression(),
        "polynomial_regression": Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
            ("ridge", Ridge(alpha=5.0, random_state=42)),
        ]),
    }

    reports = []
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, v_train)
        fit_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = model.predict(X_test)
        pred_t = time.perf_counter() - t0

        m = _metrics(v_test, preds)
        m["train_time_s"] = fit_t
        m["predict_time_s"] = pred_t
        m["model"] = name
        with open(METRICS / f"{name}.json", "w") as f:
            json.dump(m, f, indent=2)

        joblib.dump(model, MODELS / f"{name}.joblib")
        _scatter(name, v_test, preds, y_test)
        print(f"[{name}] RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R2={m['r2']:.3f}")
        reports.append(m)
    return reports


if __name__ == "__main__":
    run()
