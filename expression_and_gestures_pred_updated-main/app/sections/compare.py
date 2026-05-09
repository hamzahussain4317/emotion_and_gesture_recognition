from __future__ import annotations

import streamlit as st

from src.evaluation.report import (
    classification_table,
    clustering_table,
    regression_table,
    rl_summary,
)
from src.utils.paths import FIGURES


def render():
    st.markdown("# Model comparison")
    st.markdown(
        "<div class='teamline'>Every algorithm on the same feature set.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Classification")
    cls = classification_table()
    if cls.empty:
        st.info("No classification metrics yet — run the training scripts.")
    else:
        st.dataframe(cls, hide_index=True, use_container_width=True)
        cmp_path = FIGURES / "classifier_comparison.png"
        tva_path = FIGURES / "time_vs_accuracy.png"
        c1, c2 = st.columns(2)
        if cmp_path.exists():
            c1.image(str(cmp_path), use_container_width=True)
        if tva_path.exists():
            c2.image(str(tva_path), use_container_width=True)

    st.markdown("### Regression (valence)")
    reg = regression_table()
    if reg.empty:
        st.info("Regression metrics pending.")
    else:
        st.dataframe(reg, hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        for col, name in zip((c1, c2), ("linear_regression", "polynomial_regression")):
            p = FIGURES / f"regression_{name}.png"
            if p.exists():
                col.image(str(p), use_container_width=True)

    st.markdown("### Clustering")
    clu = clustering_table()
    if clu.empty:
        st.info("Clustering metrics pending.")
    else:
        st.dataframe(clu, hide_index=True, use_container_width=True)

    st.markdown("### Reinforcement Learning")
    rl = rl_summary()
    if not rl:
        st.info("RL metrics pending — run `python -m src.rl.train`.")
    else:
        cols = st.columns(4)
        cols[0].metric("episodes", rl.get("episodes"))
        cols[1].metric("first-100 reward", f"{rl.get('first_avg_reward'):.2f}")
        cols[2].metric("last-100 reward", f"{rl.get('final_avg_reward'):.2f}")
        cols[3].metric("max reward", f"{rl.get('max_reward'):.2f}")
        curve = FIGURES / "rl_reward_curve.png"
        if curve.exists():
            st.image(str(curve), use_container_width=True)

    st.markdown("### Confusion matrices")
    cm_names = ["knn", "svm", "decision_tree", "bagging_dt", "adaboost_dt"]
    existing = [(n, FIGURES / f"cm_{n}.png") for n in cm_names if (FIGURES / f"cm_{n}.png").exists()]
    if existing:
        chosen = st.selectbox("pick a model", [n for n, _ in existing])
        for n, p in existing:
            if n == chosen:
                st.image(str(p), width=520)
