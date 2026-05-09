from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.rl.env import GridConfig, HRIGridWorld, Human, MOODS
from src.rl.q_learning import rollout
from src.utils.paths import FIGURES, MODELS


ACCENT = "#2F5D50"
COLORS = {
    "empty": "#FFFFFF",
    "happy": "#9DC4A3",
    "angry": "#D88A75",
    "neutral": "#D8D2BE",
    "robot": "#1A1A1A",
    "path": "#2F5D50",
}


def _grid_figure(config: GridConfig, path: list[tuple[int, int]]):
    size = config.size
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_aspect("equal")
    ax.grid(True, color="#E5E3DC", linewidth=0.6)
    ax.set_axisbelow(True)

    for h in config.humans:
        r, c = h.pos
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=COLORS[h.mood], edgecolor="#555", linewidth=0.6))
        ax.text(c, r, h.mood[0].upper(), ha="center", va="center", fontsize=12, color="#333", fontweight="bold")

    if len(path) >= 2:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color=COLORS["path"], linewidth=2, alpha=0.8)
        ax.scatter(xs, ys, color=COLORS["path"], s=18, zorder=3)

    start_r, start_c = config.start
    ax.scatter([start_c], [start_r], marker="s", s=140, color="#BFBFBF", edgecolor="#333", zorder=2)
    end_r, end_c = path[-1]
    ax.scatter([end_c], [end_r], marker="o", s=220, color=COLORS["robot"], zorder=4)

    ax.set_title("Robot trajectory")
    return fig


def render():
    st.markdown("# Robot simulation")
    st.markdown(
        "<div class='teamline'>Q-Learning agent reacting to human moods.</div>",
        unsafe_allow_html=True,
    )

    q_path = MODELS / "qtable.npy"
    if not q_path.exists():
        st.warning("Q-table not found. Train it with `python -m src.rl.train`.")
        return
    Q = np.load(q_path)

    st.markdown(
        "<div class='card'>"
        "The robot starts top-left and chooses moves from a learned policy. "
        "It gets <span class='accent'>+10</span> for reaching a happy human, "
        "<span class='accent'>-10</span> next to an angry one, "
        "and a small penalty per step."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        mood_a = st.selectbox("Human at (4, 4)", MOODS, index=0)
    with c2:
        mood_b = st.selectbox("Human at (2, 5)", MOODS, index=1)

    run = st.button("Run policy")

    config = GridConfig()
    config.humans = [Human(pos=(4, 4), mood=mood_a), Human(pos=(2, 5), mood=mood_b)]

    if run:
        result = rollout(Q, humans=config.humans)
        fig = _grid_figure(config, result["path"])
        st.pyplot(fig)
        m1, m2, m3 = st.columns(3)
        m1.metric("total reward", f"{result['total_reward']:+.2f}")
        m2.metric("steps taken", result["steps"])
        m3.metric("final cell", str(result["path"][-1]))
    else:
        fig = _grid_figure(config, [config.start])
        st.pyplot(fig)

    curve = FIGURES / "rl_reward_curve.png"
    if curve.exists():
        st.markdown("### Training convergence")
        st.image(str(curve), use_container_width=True)
