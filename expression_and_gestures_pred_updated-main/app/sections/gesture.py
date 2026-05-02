from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.feature import hog

from src.data.features import HOG_KWARGS
from src.utils.paths import GESTURE_CLASSES, MODELS


GESTURE_MODELS = ["gesture_knn", "gesture_svm", "gesture_decision_tree"]


@st.cache_resource(show_spinner=False)
def _load():
    pca_path = MODELS / "gesture_pca.joblib"
    if not pca_path.exists():
        return None, {}
    pca = joblib.load(pca_path)
    models = {}
    for name in GESTURE_MODELS:
        p = MODELS / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)
    return pca, models


def _to_features(img: Image.Image, pca) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(img.convert("L").resize((64, 64)), dtype=np.uint8)
    h = hog(arr, **HOG_KWARGS).reshape(1, -1)
    return pca.transform(h), arr


def _score(img, pca, models):
    feats, g64 = _to_features(img, pca)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(img, caption="Input", width=220)
        st.image(g64, caption="64×64 grayscale", width=140, clamp=True)
    with c2:
        st.markdown("### Predictions")
        rows = []
        for name, model in models.items():
            pred = int(model.predict(feats)[0])
            conf = None
            if hasattr(model, "predict_proba"):
                try:
                    conf = float(model.predict_proba(feats)[0].max())
                except Exception:
                    conf = None
            rows.append((name.replace("gesture_", ""), GESTURE_CLASSES[pred], conf))
        df = pd.DataFrame(rows, columns=["model", "prediction", "confidence"])
        df["confidence"] = df["confidence"].apply(lambda v: f"{v:.2%}" if v is not None else "—")
        st.table(df)


def render():
    st.markdown("# Gesture recognition")
    st.markdown(
        "<div class='teamline'>LeapGestRecog · 10 hand gestures</div>",
        unsafe_allow_html=True,
    )

    pca, models = _load()
    if pca is None or not models:
        st.warning(
            "Gesture models not trained yet. Run:\n\n"
            "```\npython -m src.data.gesture_loader\n"
            "python -m src.models.gesture_classification\n```"
        )
        return

    st.markdown(
        "<div class='card'>Classes: "
        + ", ".join(f"<span class='accent'>{c}</span>" for c in GESTURE_CLASSES)
        + "</div>",
        unsafe_allow_html=True,
    )

    tab_cam, tab_upload = st.tabs(["Camera", "Upload"])
    with tab_cam:
        st.markdown(
            "<div style='font-size:0.85rem;color:#666;margin-bottom:0.6rem;'>"
            "Hold the gesture against a plain background. Point the camera close enough "
            "that the hand fills most of the frame."
            "</div>",
            unsafe_allow_html=True,
        )
        shot = st.camera_input("take a photo", label_visibility="collapsed", key="gesture_cam")
        if shot:
            _score(Image.open(shot), pca, models)

    with tab_upload:
        file = st.file_uploader(
            "Upload a hand-gesture image",
            type=["jpg", "jpeg", "png"],
            key="gesture_upload",
        )
        if file:
            _score(Image.open(io.BytesIO(file.read())), pca, models)
