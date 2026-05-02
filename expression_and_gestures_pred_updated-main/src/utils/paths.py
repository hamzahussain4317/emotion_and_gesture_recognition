from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "fer2013"
GESTURE_DATA = ROOT / "data" / "leapGestRecog"
ARTIFACTS = ROOT / "artifacts"
CACHE = ARTIFACTS / "cache"
FEATURES = ARTIFACTS / "features"
MODELS = ARTIFACTS / "models"
METRICS = ARTIFACTS / "metrics"
FIGURES = ARTIFACTS / "figures"

for _p in (CACHE, FEATURES, MODELS, METRICS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

GESTURE_CLASSES = [
    "palm", "l", "fist", "fist_moved", "thumb",
    "index", "ok", "palm_moved", "c", "down",
]

VALENCE = {
    "angry": -0.8,
    "disgust": -0.6,
    "fear": -0.5,
    "sad": -0.7,
    "neutral": 0.0,
    "happy": 0.9,
    "surprise": 0.4,
}
