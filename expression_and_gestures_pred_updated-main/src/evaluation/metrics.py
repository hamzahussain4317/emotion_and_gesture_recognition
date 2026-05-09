from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.paths import CLASS_NAMES, FIGURES, METRICS

ACCENT = "#2F5D50"


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_metrics(name: str, payload: dict) -> Path:
    path = METRICS / f"{name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def save_confusion(name: str, y_true, y_pred, class_names: list[str] | None = None) -> Path:
    class_names = class_names or CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix — {name}")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    out = FIGURES / f"cm_{name}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def full_classification_report(name: str, y_true, y_pred, train_time: float, predict_time: float) -> dict:
    payload = classification_metrics(y_true, y_pred)
    payload["train_time_s"] = float(train_time)
    payload["predict_time_s"] = float(predict_time)
    payload["per_class"] = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    save_metrics(name, payload)
    save_confusion(name, y_true, y_pred)
    return payload
