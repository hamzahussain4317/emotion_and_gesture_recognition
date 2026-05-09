from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from src.rl.q_learning import train
from src.utils.paths import FIGURES, METRICS, MODELS
from src.utils.seed import set_seed

ACCENT = "#2F5D50"


def _plot_curve(rewards: list[float], path):
    arr = np.array(rewards)
    window = max(1, len(arr) // 40)
    smooth = np.convolve(arr, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(arr, color="#BFBFBF", linewidth=0.6, label="per-episode")
    ax.plot(np.arange(len(smooth)), smooth, color=ACCENT, linewidth=2, label=f"rolling mean ({window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative reward")
    ax.set_title("Q-Learning — convergence")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(episodes: int = 2000):
    set_seed()
    t0 = time.perf_counter()
    Q, stats = train(episodes=episodes)
    dur = time.perf_counter() - t0

    np.save(MODELS / "qtable.npy", Q)
    _plot_curve(stats["rewards"], FIGURES / "rl_reward_curve.png")

    payload = {
        "model": "q_learning",
        "train_time_s": dur,
        "episodes": stats["episodes"],
        "final_avg_reward": stats["final_avg_reward"],
        "first_avg_reward": stats["first_avg_reward"],
        "max_reward": stats["max_reward"],
        "alpha": stats["alpha"],
        "gamma": stats["gamma"],
    }
    with open(METRICS / "q_learning.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[q-learning] trained {episodes} episodes in {dur:.1f}s  final_avg={payload['final_avg_reward']:.2f}")


if __name__ == "__main__":
    main()
