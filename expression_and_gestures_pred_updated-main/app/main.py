from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.style import inject
from app.sections import clustering_view, compare, gesture, overview, robot_sim, try_it

st.set_page_config(
    page_title="HRI Emotion Recognition",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject()

SECTIONS = {
    "Overview": overview.render,
    "Emotion (face)": try_it.render,
    "Gesture (hand)": gesture.render,
    "Model comparison": compare.render,
    "Clustering": clustering_view.render,
    "Robot simulation": robot_sim.render,
}

with st.sidebar:
    st.markdown("### HRI · Emotion")
    st.markdown(
        "<div class='teamline'>ML for Robotics · Project</div>",
        unsafe_allow_html=True,
    )
    choice = st.radio("Sections", list(SECTIONS.keys()), label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.78rem;color:#777;line-height:1.6;'>"
        "Filza Salman<br>Yasal Khan<br>Yasir Memon<br>Yousha Mehdi<br>Taha Farooque"
        "</div>",
        unsafe_allow_html=True,
    )

SECTIONS[choice]()
