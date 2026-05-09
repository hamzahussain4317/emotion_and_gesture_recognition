from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.feature import hog

from src.data.features import HOG_KWARGS
from src.utils.detectors import crop_face
from src.utils.paths import CLASS_NAMES, MODELS


CLASSIFIERS = ["knn", "svm", "decision_tree", "bagging_dt", "adaboost_dt"]


@st.cache_resource(show_spinner=False)
def _load_pca():
    path = MODELS / "pca.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def _load_models():
    out = {}
    for name in CLASSIFIERS:
        p = MODELS / f"{name}.joblib"
        if p.exists():
            out[name] = joblib.load(p)
    regs = {}
    for name in ("linear_regression", "polynomial_regression"):
        p = MODELS / f"{name}.joblib"
        if p.exists():
            regs[name] = joblib.load(p)
    return out, regs


def _image_to_features(img: Image.Image, pca) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(img.convert("L").resize((48, 48)), dtype=np.uint8)
    h = hog(arr, **HOG_KWARGS).reshape(1, -1)
    return pca.transform(h), arr


def _score_and_render(img: Image.Image, pca, clfs, regs):
    face = crop_face(img)
    if face is None:
        st.error("No face detected. Move closer to the camera or improve lighting.")
        return
    feats, face48 = _image_to_features(face, pca)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(img, caption="Input", width=220)
        st.image(face, caption="Detected face", width=180)
        st.image(face48, caption="48×48 grayscale", width=140, clamp=True)

    with c2:
        st.markdown("### Predictions")
        rows = []
        for name, model in clfs.items():
            pred = int(model.predict(feats)[0])
            conf = None
            if hasattr(model, "predict_proba"):
                try:
                    conf = float(model.predict_proba(feats)[0].max())
                except Exception:
                    conf = None
            rows.append((name, CLASS_NAMES[pred], conf))

        df = pd.DataFrame(rows, columns=["model", "prediction", "confidence"])
        df["confidence"] = df["confidence"].apply(
            lambda v: f"{v:.2%}" if v is not None else "—"
        )
        st.table(df)

        if regs:
            st.markdown("### Valence (continuous sentiment)")
            vcols = st.columns(len(regs))
            for (name, model), col in zip(regs.items(), vcols):
                v = float(model.predict(feats)[0])
                v_clipped = max(-1.0, min(1.0, v))
                col.metric(name.replace("_", " "), f"{v_clipped:+.2f}")


def render():
    st.markdown("# Try it on a face")
    st.markdown(
        "<div class='teamline'>Every model scores the image you provide.</div>",
        unsafe_allow_html=True,
    )

    pca = _load_pca()
    clfs, regs = _load_models()

    if pca is None or not clfs:
        st.warning(
            "Models not trained yet. Run `python -m src.models.classification` "
            "(and ensemble/regression) first."
        )
        return

    tab_cam, tab_upload = st.tabs(["Camera", "Upload"])

    with tab_cam:
        st.markdown(
            "<div style='font-size:0.85rem;color:#666;margin-bottom:0.6rem;'>"
            "Use your webcam. Click the shutter to capture; move close to the camera "
            "so the face fills the frame."
            "</div>",
            unsafe_allow_html=True,
        )
        shot = st.camera_input("take a photo", label_visibility="collapsed", key="camshot")
        if shot is not None:
            img = Image.open(shot)
            _score_and_render(img, pca, clfs, regs)

    with tab_upload:
        file = st.file_uploader(
            "Upload a face image (jpg / png)",
            type=["jpg", "jpeg", "png"],
            key="uploader",
        )
        if file:
            img = Image.open(io.BytesIO(file.read()))
            _score_and_render(img, pca, clfs, regs)
