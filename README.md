# Emotion and Gesture Recognition

**Course project — Fundamentals of Computer Vision**




https://github.com/user-attachments/assets/11f770ca-bb92-4087-9299-dc61bf98c192




A comparative analysis of classical ML algorithms applied to facial-emotion and hand-gesture recognition, framed as the perception module of a human–robot interaction system. A Q-Learning agent uses the detected emotion to drive a robot in a grid-world simulation.

## Team

| Name | Roll |
|---|---|
| Aazar Arnold | 22K-4277 |
| Muhammad Hamza Hussain | 22K-4317 |

## What it does

- **Emotion recognition** on FER2013 (7 classes) with KNN, SVM, Decision Tree, Bagging, AdaBoost.
- **Gesture recognition** on LeapGestRecog (10 classes) with KNN, SVM, Decision Tree.
- **Valence regression** (continuous sentiment score) with Linear + Polynomial.
- **K-Means clustering** on emotion features with t-SNE visualisation.
- **Q-Learning** grid-world where a robot approaches happy humans and avoids angry ones.
- **Live Streamlit demo** with webcam input for both emotion and gesture.

## Rubric coverage

| Rubric component | Weight | Where |
|---|---|---|
| Preprocessing & feature engineering | 20% | [src/data/](src/data) |
| Classification (KNN · SVM · Decision Tree) | 15% | [src/models/classification.py](src/models/classification.py) |
| Regression (Linear · Polynomial) | 5% | [src/models/regression.py](src/models/regression.py) |
| Clustering (K-Means) | 5% | [src/models/clustering.py](src/models/clustering.py) |
| Reinforcement Learning (Q-Learning) | 5% | [src/rl/](src/rl) |
| Ensemble (Bagging · AdaBoost) | 10% | [src/models/ensemble.py](src/models/ensemble.py) |
| Evaluation & comparative analysis | 20% | [src/evaluation/](src/evaluation) |
| Report & slides | 10% | [report/](report) |
| Demo | 10% | [app/](app) |

## Repository layout

```
MLR Project/
├── src/
│   ├── data/           data loading, preprocessing, HOG + PCA features
│   ├── models/         classification, regression, clustering, ensembles
│   ├── rl/             Q-Learning environment + agent
│   ├── evaluation/     metrics + comparative report
│   └── utils/          seeds, paths, face detector
├── app/                Streamlit demo (overview, emotion, gesture, compare, clustering, RL sim)
├── report/             report.md + slides.md
├── notebooks/          (optional) EDA notebooks
├── artifacts/          trained models, metrics, figures (gitignored)
├── data/               datasets (gitignored)
└── requirements.txt
```

## Setup

### 1. Clone and install

```bash
git clone https://github.com/taha-farooqui/MLR-project.git
cd MLR-project

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download datasets

Both are free but not bundled (they are heavy and gitignored).

**FER2013** — facial emotion dataset:

```bash
curl -L -o data/fer2013.zip https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013
cd data && tar -xf fer2013.zip && mkdir -p fer2013 && mv train test fer2013/ && rm fer2013.zip && cd ..
```

Expected layout: `data/fer2013/train/<class>/*.jpg`, `data/fer2013/test/<class>/*.jpg`.

**LeapGestRecog** — hand gesture dataset:

```bash
curl -L -o data/leapgestrecog.zip https://www.kaggle.com/api/v1/datasets/download/gti-upm/leapgestrecog
cd data && tar -xf leapgestrecog.zip && rm leapgestrecog.zip && cd ..
```

Expected layout: `data/leapGestRecog/<subject>/<class>/*.png`.

## Train every model (one-shot)

```bash
python -m src.run_all
```

This runs features → EDA → classification → regression → clustering → ensemble → RL → report. Total ~15–25 min depending on CPU.

## Or train step by step

```bash
# 1. build HOG + PCA features + EDA figures
python -m src.data.features
python -m src.data.eda

# 2. emotion pipeline
python -m src.models.classification        # KNN, SVM, Decision Tree
python -m src.models.regression            # Linear + Polynomial (valence)
python -m src.models.clustering            # K-Means + t-SNE
python -m src.models.ensemble              # Bagging + AdaBoost

# 3. gesture pipeline
python -m src.data.gesture_loader          # load + HOG + PCA on gesture images
python -m src.models.gesture_classification

# 4. reinforcement learning
python -m src.rl.train

# 5. unified comparison report
python -m src.evaluation.report
```

## Launch demo

```bash
python -m streamlit run app/main.py
```

Opens at `http://localhost:8501`. Sections:

1. **Overview** — problem statement, pipeline, dataset sample.
2. **Emotion (face)** — webcam / upload → 5 models predict emotion + 2 regressors predict valence.
3. **Gesture (hand)** — webcam / upload → 3 models predict one of 10 gestures.
4. **Model comparison** — unified metrics table + bar chart + confusion matrices.
5. **Clustering** — t-SNE visualisation of K-Means clusters vs. true labels.
6. **Robot simulation** — pick human moods → watch the learned Q-Learning policy run.

## Build the report / slides

Both written in Markdown under [report/](report). Produce PDFs with Pandoc:

```bash
cd report
pandoc report.md -o report.pdf --pdf-engine=xelatex
pandoc slides.md -o slides.pdf -t beamer
```

If Pandoc is not installed, open the files in VSCode and use the *Markdown PDF* extension.

## Reproducibility

- All random seeds fixed to `42` in [src/utils/seed.py](src/utils/seed.py).
- Feature files cached to `artifacts/features/` after first run.
- Trained models saved as `.joblib` under `artifacts/models/`.

## Dataset sources

- FER2013 — https://www.kaggle.com/datasets/msambare/fer2013
- LeapGestRecog — https://www.kaggle.com/datasets/gti-upm/leapgestrecog
