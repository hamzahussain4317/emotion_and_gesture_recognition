from __future__ import annotations

import streamlit as st

from src.utils.paths import FIGURES


def render():
    st.markdown("# Clustering structure")
    st.markdown(
        "<div class='teamline'>Unsupervised K-Means vs. true labels.</div>",
        unsafe_allow_html=True,
    )

    clusters = FIGURES / "tsne_clusters.png"
    labels = FIGURES / "tsne_labels.png"

    c1, c2 = st.columns(2)
    if clusters.exists():
        c1.image(str(clusters), caption="K-Means cluster assignments")
    else:
        c1.info("Run clustering to generate figures.")

    if labels.exists():
        c2.image(str(labels), caption="Ground truth labels")

    st.markdown(
        "<div class='card'>"
        "Dense regions align with visually similar emotions (e.g. <span class='accent'>angry</span> "
        "and <span class='accent'>disgust</span>, or <span class='accent'>fear</span> and "
        "<span class='accent'>sad</span>) — consistent with confusion patterns in the classifiers."
        "</div>",
        unsafe_allow_html=True,
    )
