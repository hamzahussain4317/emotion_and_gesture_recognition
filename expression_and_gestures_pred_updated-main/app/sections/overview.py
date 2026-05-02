from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from src.utils.paths import CLASS_NAMES, FIGURES


def render():
    st.markdown("# Human–Robot Interaction")
    st.markdown(
        "<div class='teamline'>Emotion recognition for social robots</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(
            "<div class='card'>"
            "<h3 style='margin-top:0'>Problem</h3>"
            "<p>Robots that work alongside people need to read emotion as well as they read "
            "obstacles. We train classical machine-learning models on the FER2013 dataset "
            "to recognise seven emotions from grayscale face images, then let a "
            "Q-Learning agent use those readings to navigate a simulated room.</p>"
            "<h3>Pipeline</h3>"
            "<p><span class='accent'>Faces → HOG features → PCA → [KNN · SVM · Decision Tree · "
            "Bagging · AdaBoost]</span>. A regressor projects onto a valence axis. "
            "K-Means surfaces structure. Q-Learning closes the loop.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Classes")
        cols = st.columns(2)
        for i, name in enumerate(CLASS_NAMES):
            cols[i % 2].markdown(f"- {name}")

    st.markdown("### Dataset")
    sample_path = FIGURES / "eda" / "sample_grid.png"
    dist_path = FIGURES / "eda" / "class_distribution.png"
    c1, c2 = st.columns(2)
    with c1:
        if sample_path.exists():
            st.image(str(sample_path), caption="FER2013 samples")
        else:
            st.info("Run EDA to generate samples: `python -m src.data.eda`")
    with c2:
        if dist_path.exists():
            st.image(str(dist_path), caption="Class distribution")
