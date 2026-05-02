from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def crop_face(img: Image.Image, pad_ratio: float = 0.15) -> Image.Image | None:
    arr = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = _get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    pad = int(max(w, h) * pad_ratio)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(arr.shape[1], x + w + pad)
    y1 = min(arr.shape[0], y + h + pad)
    return Image.fromarray(arr[y0:y1, x0:x1])


_hands = None


def _get_hands():
    global _hands
    if _hands is None:
        from pathlib import Path
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks import python as mp_python

        root = Path(__file__).resolve().parents[2]
        model_path = root / "artifacts" / "models" / "hand_landmarker.task"
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=2,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _hands = mp_vision.HandLandmarker.create_from_options(options)
    return _hands


def crop_hand(img: Image.Image, pad_ratio: float = 0.3) -> Image.Image | None:
    import mediapipe as mp

    arr = np.asarray(img.convert("RGB"))
    h_img, w_img = arr.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    result = _get_hands().detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    xs = np.array([p.x for p in lm]) * w_img
    ys = np.array([p.y for p in lm]) * h_img
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad = max(x1 - x0, y1 - y0) * pad_ratio
    x0 = max(0, int(x0 - pad))
    y0 = max(0, int(y0 - pad))
    x1 = min(w_img, int(x1 + pad))
    y1 = min(h_img, int(y1 + pad))
    return Image.fromarray(arr[y0:y1, x0:x1])
