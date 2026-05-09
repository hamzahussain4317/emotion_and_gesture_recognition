from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.paths import FIGURES, METRICS

ACCENT = "#2F5D50"
ACCENT_ALT = "#C06C4B"

CLASSIFICATION_MODELS = ["knn", "svm", "decision_tree", "bagging_dt", "adaboost_dt"]
REGRESSION_MODELS = ["linear_regression", "polynomial_regression"]


def _load(name: str) -> dict | None:
    path = METRICS / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def classification_table() -> pd.DataFrame:
    rows = []
    for name in CLASSIFICATION_MODELS:
        data = _load(name)
        if not data:
            continue
        rows.append({
            "model": name,
            "accuracy": round(data["accuracy"], 4),
            "precision_macro": round(data["precision_macro"], 4),
            "recall_macro": round(data["recall_macro"], 4),
            "f1_macro": round(data["f1_macro"], 4),
            "f1_weighted": round(data["f1_weighted"], 4),
            "train_time_s": round(data["train_time_s"], 2),
            "predict_time_s": round(data["predict_time_s"], 2),
        })
    return pd.DataFrame(rows)


def regression_table() -> pd.DataFrame:
    rows = []
    for name in REGRESSION_MODELS:
        data = _load(name)
        if not data:
            continue
        rows.append({
            "model": name,
            "rmse": round(data["rmse"], 4),
            "mae": round(data["mae"], 4),
            "r2": round(data["r2"], 4),
            "train_time_s": round(data["train_time_s"], 2),
        })
    return pd.DataFrame(rows)


def clustering_table() -> pd.DataFrame:
    data = _load("kmeans")
    if not data:
        return pd.DataFrame()
    return pd.DataFrame([{
        "model": "kmeans",
        "silhouette": round(data["silhouette"], 4),
        "davies_bouldin": round(data["davies_bouldin"], 4),
        "adjusted_rand_index": round(data["adjusted_rand_index"], 4),
        "train_time_s": round(data["train_time_s"], 2),
    }])


def rl_summary() -> dict:
    data = _load("q_learning")
    return data or {}


def _bar_comparison(df: pd.DataFrame, path: Path):
    if df.empty:
        return
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2F5D50", "#6B9080", "#A4C3B2", "#C06C4B"]
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m].values, width, label=m, color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["model"].values, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Classifier comparison")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _time_vs_accuracy(df: pd.DataFrame, path: Path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["train_time_s"] + df["predict_time_s"], df["accuracy"], s=80, color=ACCENT)
    for _, row in df.iterrows():
        ax.annotate(
            row["model"],
            (row["train_time_s"] + row["predict_time_s"], row["accuracy"]),
            xytext=(6, 4), textcoords="offset points", fontsize=9,
        )
    ax.set_xlabel("total time (s, train + predict)")
    ax.set_ylabel("accuracy")
    ax.set_title("Compute cost vs. accuracy")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_markdown(out: Path):
    cls = classification_table()
    reg = regression_table()
    clu = clustering_table()
    rl = rl_summary()

    lines = ["# Model Comparison Report\n"]
    lines.append("## Classification\n")
    lines.append(cls.to_markdown(index=False) if not cls.empty else "_pending_\n")
    lines.append("\n\n## Regression (valence prediction)\n")
    lines.append(reg.to_markdown(index=False) if not reg.empty else "_pending_\n")
    lines.append("\n\n## Clustering\n")
    lines.append(clu.to_markdown(index=False) if not clu.empty else "_pending_\n")
    lines.append("\n\n## Reinforcement Learning\n")
    if rl:
        lines.append(
            f"- episodes: **{rl.get('episodes')}**\n"
            f"- final avg reward (last 100): **{rl.get('final_avg_reward'):.3f}**\n"
            f"- first avg reward (first 100): **{rl.get('first_avg_reward'):.3f}**\n"
            f"- max reward: **{rl.get('max_reward'):.3f}**\n"
            f"- train time: **{rl.get('train_time_s'):.2f}s**\n"
        )
    else:
        lines.append("_pending_\n")
    out.write_text("\n".join(lines))


def run():
    cls = classification_table()
    cls.to_csv(METRICS / "classification_summary.csv", index=False)
    _bar_comparison(cls, FIGURES / "classifier_comparison.png")
    _time_vs_accuracy(cls, FIGURES / "time_vs_accuracy.png")

    regression_table().to_csv(METRICS / "regression_summary.csv", index=False)
    clustering_table().to_csv(METRICS / "clustering_summary.csv", index=False)

    write_markdown(METRICS / "report.md")
    print("wrote:", METRICS / "report.md")


if __name__ == "__main__":
    run()
