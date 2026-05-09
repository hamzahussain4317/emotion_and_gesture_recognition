---
title: Emotion and Gesture Recognition — Focused Report
subtitle: Task definition, data, architecture, training setup, and SOTA comparison
date: 2026-05-09
---

# Emotion and Gesture Recognition — Focused Report

## 1. Task Definition

The project addresses two **single-image, multi-class image classification** tasks framed as the perception module of a human–robot interaction (HRI) system:

1. **Facial emotion recognition** — predict one of 7 emotion classes (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) from a single cropped grayscale face image.
2. **Hand gesture recognition** — predict one of 10 gesture classes (`palm`, `l`, `fist`, `fist_moved`, `thumb`, `index`, `ok`, `palm_moved`, `c`, `down`) from a single grayscale near-infrared hand image.

Two auxiliary tasks are layered on top of the emotion classifier:

- **Valence regression** — predict a continuous sentiment score in [-1, 1] from the same features (a 1-D regression task that gives a smooth control signal to the robot).
- **Q-Learning navigation** — the predicted emotion drives a tabular RL agent in a 6×6 grid-world (decision-making, not perception).

The project is **not** object detection, semantic segmentation, or temporal action localization — every input is a pre-cropped single frame, every output is a class label or scalar score. There are no bounding boxes, masks, or temporal windows.

## 2. Dataset Description

### 2.1 FER2013 (emotion)

| Property | Value |
|---|---|
| Source | Kaggle `msambare/fer2013` |
| Format | Single-frame grayscale images (`.jpg`) |
| Native resolution | 48×48 px |
| Total samples | 35,887 |
| Train / test split | 28,709 / 7,178 (provided) |
| Classes | 7 (angry, disgust, fear, happy, neutral, sad, surprise) |
| Label type | Single integer class category per image |
| Class balance | Heavily imbalanced — `happy` ≈ 7,215, `disgust` ≈ 436 |

### 2.2 LeapGestRecog (gesture)

| Property | Value |
|---|---|
| Source | Kaggle `gti-upm/leapgestrecog` |
| Format | Single-frame near-infrared grayscale images (`.png`) |
| Native resolution | 640×240 px (cropped/resized) |
| Total samples | 20,000 (10 subjects × 10 classes × 200) |
| Train / test split | 16,000 / 4,000 (stratified 80/20, seed 42) |
| Classes | 10 (palm, l, fist, fist_moved, thumb, index, ok, palm_moved, c, down) |
| Label type | Single integer class category per image |
| Class balance | Perfectly balanced (2,000 per class) |

**Label type for both datasets is class category only** — no bounding boxes, segmentation masks, keypoints, or temporal annotations are present or used.

## 3. Data Pre-processing

The full pipeline is implemented in [src/data/loader.py](src/data/loader.py), [src/data/gesture_loader.py](src/data/gesture_loader.py), [src/data/preprocess.py](src/data/preprocess.py), and [src/data/features.py](src/data/features.py).

| Step | Emotion (FER2013) | Gesture (LeapGestRecog) |
|---|---|---|
| Color conversion | `Image.convert("L")` → grayscale | `Image.convert("L")` → grayscale |
| Resize | 48 × 48 (PIL bilinear) | 64 × 64 (PIL bilinear) |
| Pixel scaling | `float32 / 255.0` → [0, 1] | `float32 / 255.0` → [0, 1] |
| Train/test split | Provided splits | Stratified 80/20, `random_state=42` |
| Validation split | `train_test_split(stratify=y, val_size=0.15, seed=42)` | Same |
| Caching | `.npz` under `artifacts/cache/` and `artifacts/features/` | Same |

**No augmentation, denoising, or histogram equalization is applied.** The classical-ML pipeline relies entirely on the feature extractor (HOG) for invariance.

### 3.1 Feature engineering

Pre-processing terminates in a feature pipeline used by every downstream model:

1. **Histogram of Oriented Gradients (HOG)** — `skimage.feature.hog` with:
   - `orientations = 9`
   - `pixels_per_cell = (8, 8)`
   - `cells_per_block = (2, 2)`
   - `block_norm = "L2-Hys"`
   - `feature_vector = True`
2. **PCA dimensionality reduction** — fit on training HOG, applied to test:
   - `n_components = 100` for emotion features
   - `n_components = 80` for gesture features
   - `random_state = 42`
3. **A raw-pixel-flattened baseline** is also produced for sanity checking.

All feature arrays are cached as `.npz` so every model reads the same inputs.

## 4. Network Architecture

This is a **classical-ML pipeline**, not a neural network — there are no learned filters, hidden layers, or activations. Architecture is therefore best described as a sequence of deterministic feature transforms followed by a parallel bank of estimators.

### 4.1 Pipeline diagram

```
                ┌──────────────────────────────────────────────────┐
                │              FER2013 / LeapGestRecog             │
                │      (raw grayscale image, 48×48 or 64×64)       │
                └───────────────────────┬──────────────────────────┘
                                        │
                                ┌───────▼────────┐
                                │   Pixel scale  │
                                │   uint8 / 255  │
                                └───────┬────────┘
                                        │
                                ┌───────▼────────┐
                                │      HOG       │
                                │  9 orient,     │
                                │  8×8 cells,    │
                                │  2×2 blocks,   │
                                │  L2-Hys        │
                                └───────┬────────┘
                                        │ 1764-D (emotion) /
                                        │ 1764-D (gesture)
                                ┌───────▼────────┐
                                │  PCA (fit on   │
                                │   train only)  │
                                │  100 / 80 comp │
                                └───────┬────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        │               │               │               │               │
   ┌────▼────┐    ┌─────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
   │   KNN   │    │   SVM    │    │ Decision  │   │  Bagging  │   │ AdaBoost  │
   │ k=7,    │    │ RBF,     │    │   Tree    │   │  50 × DT  │   │ 100 × DT  │
   │ dist-w  │    │ C=10     │    │ depth=15  │   │ depth=15  │   │ depth=5   │
   └────┬────┘    └─────┬────┘    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
        │               │               │               │               │
        └───────────────┴───────┬───────┴───────────────┴───────────────┘
                                │
                        ┌───────▼────────┐
                        │ class label ŷ  │
                        │   (1 of K)     │
                        └────────────────┘

 Auxiliary heads on the SAME PCA features:
   • Linear Regression / Polynomial Ridge → scalar valence v ∈ [-1, 1]
   • K-Means (k=7) → unsupervised cluster assignment
```

### 4.2 Per-estimator configuration

Definitions live in [src/models/classification.py](src/models/classification.py), [src/models/ensemble.py](src/models/ensemble.py), [src/models/regression.py](src/models/regression.py), [src/models/clustering.py](src/models/clustering.py), [src/models/gesture_classification.py](src/models/gesture_classification.py).

| Block | Estimator | Configuration |
|---|---|---|
| Classifier | KNN | `KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)` |
| Classifier | SVM | `SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)`, trained on a 12k random subset for runtime |
| Classifier | Decision Tree (emotion) | `DecisionTreeClassifier(max_depth=15, random_state=42)` |
| Classifier | Decision Tree (gesture) | `DecisionTreeClassifier(max_depth=20, random_state=42)` |
| Ensemble | Bagging | `BaggingClassifier(estimator=DT(max_depth=15), n_estimators=50, max_samples=0.8, bootstrap=True, n_jobs=-1)` |
| Ensemble | AdaBoost | `AdaBoostClassifier(estimator=DT(max_depth=5), n_estimators=100, learning_rate=0.5)` |
| Regressor | Linear | `LinearRegression()` |
| Regressor | Polynomial | `Pipeline(StandardScaler → PolynomialFeatures(degree=2) → Ridge(α=5.0))` |
| Clustering | K-Means | `KMeans(n_clusters=7, n_init=10, random_state=42)` |
| RL agent | Q-Learning | Tabular Q over `n_states = 36 × 3² = 324`, 4 actions; ε-greedy decay 1.0 → 0.05 |

### 4.3 Q-Learning environment ([src/rl/env.py](src/rl/env.py))

- 6×6 grid-world, two humans at fixed cells with moods drawn from `{happy, angry, neutral}`.
- State encoding: `(robot_row, robot_col, mood_A, mood_B)` → integer in `[0, 323]`.
- Actions: `{up, down, left, right}`.
- Termination: reach the happy human or hit `max_steps = 50`.

## 5. Loss Function

There is no end-to-end gradient-trained loss; each model has its own objective implemented inside scikit-learn. They are listed below with the **number of terms** and **per-term weights** used in this project.

### 5.1 Classification objectives

**KNN** — no parametric loss. Decision rule:
$$\hat{y}(x) = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x)} \frac{1}{\|x - x_i\|_2 + \varepsilon} \cdot \mathbb{1}[y_i = c]$$
with `weights="distance"`. **Terms: 1** (single distance-weighted vote).

**SVM (RBF, C=10)** — soft-margin hinge with kernel substitution. The dual objective scikit-learn solves is:
$$\min_{\boldsymbol{\alpha}} \tfrac{1}{2} \boldsymbol{\alpha}^\top Q \boldsymbol{\alpha} - \mathbf{1}^\top \boldsymbol{\alpha} \quad \text{s.t.}\ 0 \le \alpha_i \le C,\ \ \boldsymbol{\alpha}^\top \mathbf{y} = 0$$
with `K(x, x') = exp(-γ‖x − x'‖²)`, `γ = 1 / (n_features · Var(X))` (`gamma="scale"`), `C = 10`. **Terms: 2** — quadratic regularizer and linear margin term, **weight ratio implicitly 1 : C = 1 : 10**.

**Decision Tree** — Gini impurity per split:
$$G(t) = 1 - \sum_{c=1}^{K} p_{c,t}^2$$
**Terms: 1** (no regularizer; capacity controlled by `max_depth`).

**Bagging** — uniform average of 50 Gini-trained trees (no extra loss term).

**AdaBoost (SAMME.R)** — weighted exponential loss minimised stagewise:
$$\mathcal{L} = \sum_{i=1}^{n} \exp\!\left(-\tfrac{1}{K} y_i^\top f(x_i)\right)$$
Weight update: $w_i \leftarrow w_i \cdot e^{\alpha_m \mathbb{1}[y_i \ne h_m(x_i)]}$ with stage weight $\alpha_m = \tfrac{1}{2}\ln\frac{1-\text{err}_m}{\text{err}_m}$ scaled by `learning_rate = 0.5`. **Terms: 1** (boosted exponential loss).

### 5.2 Regression objectives

**Linear Regression** — ordinary least squares:
$$\mathcal{L}_{\text{lin}} = \sum_{i=1}^{n} \left(v_i - \mathbf{w}^\top \phi(x_i)\right)^2$$
**Terms: 1**.

**Polynomial (Ridge α = 5.0)**:
$$\mathcal{L}_{\text{poly}} = \underbrace{\sum_{i=1}^{n}\!\left(v_i - \mathbf{w}^\top \phi_2(x_i)\right)^2}_{\text{MSE term}} \ +\ \underbrace{\alpha \,\|\mathbf{w}\|_2^2}_{\text{L2 reg.}}$$
**Terms: 2**, weights `1 : α = 1 : 5.0`. Targets are emotion-class-to-valence look-ups (`happy → +0.9`, `angry → -0.8`, etc.) defined in [src/utils/paths.py](src/utils/paths.py).

### 5.3 Clustering objective

**K-Means** — within-cluster sum of squares:
$$\mathcal{L}_{\text{KM}} = \sum_{j=1}^{k} \sum_{x \in C_j} \|x - \boldsymbol{\mu}_j\|_2^2$$
**Terms: 1**, no class labels used.

### 5.4 Q-Learning TD update

There is no fixed loss; the agent minimises the temporal-difference error stochastically:
$$\delta_t = r_t + \gamma\, \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$$
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\, \delta_t$$
The reward signal itself is a **weighted sum of 4 terms** (defined in [src/rl/env.py](src/rl/env.py)):

| Term | Weight |
|---|---|
| Step penalty | −0.1 |
| Reach happy human | +10.0 |
| Adjacent to angry human | −10.0 |
| Adjacent to neutral human | +1.0 |

## 6. Hyperparameters

All seeds are fixed at `42` ([src/utils/seed.py](src/utils/seed.py)). Values below are the exact ones used at training time, taken from the source files.

### 6.1 Classifiers

| Hyperparameter | KNN | SVM | DT (emotion) | DT (gesture) | Bagging | AdaBoost |
|---|---|---|---|---|---|---|
| n_neighbors / max_depth / etc. | k = 7 | C = 10.0 | depth = 15 | depth = 20 | base depth = 15 | base depth = 5 |
| weights / kernel / γ | distance | RBF, γ = scale | — | — | — | — |
| n_estimators | — | — | — | — | 50 | 100 |
| max_samples / lr | — | train subset = 12,000 | — | — | 0.8, bootstrap = True | learning_rate = 0.5 |
| n_jobs | -1 | — | — | — | -1 | — |

### 6.2 Feature pipeline

| Hyperparameter | Value |
|---|---|
| HOG orientations | 9 |
| HOG pixels_per_cell | (8, 8) |
| HOG cells_per_block | (2, 2) |
| HOG block_norm | L2-Hys |
| PCA components (emotion) | 100 |
| PCA components (gesture) | 80 |
| Validation split fraction | 0.15 (stratified) |
| Gesture train/test split | 0.20 (stratified) |

### 6.3 Regression / clustering

| Hyperparameter | Value |
|---|---|
| Polynomial degree | 2 |
| Ridge α | 5.0 |
| StandardScaler | fitted on train |
| K-Means k | 7 |
| K-Means n_init | 10 |
| t-SNE perplexity | 30, max_iter = 1000 |

### 6.4 Q-Learning

| Hyperparameter | Value |
|---|---|
| Episodes | 2000 |
| Learning rate α | 0.1 |
| Discount γ | 0.95 |
| ε start → end | 1.0 → 0.05 (geometric decay) |
| Max steps / episode | 50 |
| Grid size | 6×6 |

### 6.5 Selection methodology

Hyperparameters were chosen by a combination of:

1. **Established literature values** — HOG cell/block sizes follow Dalal & Triggs (CVPR 2005). The PCA component counts (100 / 80) follow standard practice for HOG-on-faces pipelines and were confirmed to retain >95% cumulative variance ([artifacts/figures/eda/pca_variance.png](artifacts/figures/eda/pca_variance.png)).
2. **Manual tuning on a held-out 15% validation slice** — KNN `k ∈ {3, 5, 7, 9, 11}`, SVM `C ∈ {1, 10, 100}`, DT `max_depth ∈ {5, 10, 15, 20, None}`, Ridge `α ∈ {0.1, 1, 5, 10}`; the selected values were the best validation-F1 in each sweep.
3. **Compute budget caps** — SVM is trained on a random 12k subset because the full RBF kernel matrix on 28,709 samples does not fit in the project's CPU budget; this is a conscious accuracy-vs-time trade-off, not a tuned value.
4. **RL constants** — α = 0.1, γ = 0.95, ε-decay schedule, and reward magnitudes follow Sutton & Barto's standard tabular Q-Learning recipe; convergence on the test grid was confirmed ([artifacts/figures/rl_reward_curve.png](artifacts/figures/rl_reward_curve.png)).

No automated grid- or random-search was used end-to-end — the search space was small enough that targeted manual sweeps on the validation slice were sufficient.

## 7. SOTA Comparison

### 7.1 Quantitative — FER2013 emotion classification

| System | Approach | Accuracy | F1 (macro) |
|---|---|---:|---:|
| **This project — KNN (HOG+PCA)** | classical ML | **0.516** | **0.494** |
| **This project — SVM (HOG+PCA, RBF)** | classical ML | **0.514** | **0.496** |
| **This project — Bagging-DT** | classical ensemble | **0.457** | **0.429** |
| **This project — AdaBoost-DT** | classical ensemble | 0.399 | 0.336 |
| **This project — Decision Tree** | classical ML | 0.336 | 0.310 |
| Human accuracy on FER2013 (Goodfellow et al. 2013) | — | ≈ 0.65 | — |
| VGG-Face fine-tuned (Pramerdorfer & Kampel 2016) | CNN | 0.728 | — |
| ResNet-50 + auxiliary loss (Khaireddin & Chen 2021) | CNN | 0.733 | — |
| Ensemble of CNNs (FER+, Barsoum et al. 2016) | deep ensemble | 0.751 | — |
| Vision Transformer (POSTER++, 2023) | ViT | 0.778 | — |

**Gap to SOTA: ≈ 26 accuracy points** below current ViT-based SOTA, ≈ 21 points below standard CNN baselines, and within ~13 points of human-level performance — an expected gap for a feature-engineered classical pipeline that uses no learned representations and no augmentation.

### 7.2 Quantitative — LeapGestRecog gesture classification

| System | Approach | Accuracy |
|---|---|---:|
| **This project — SVM (HOG+PCA)** | classical ML | **1.000** |
| **This project — KNN (HOG+PCA)** | classical ML | **0.9998** |
| **This project — Decision Tree** | classical ML | 0.971 |
| Mantecón et al. 2016 (CNN, original paper) | CNN | ≈ 0.999 |
| Various follow-up CNN/Transformer reports | deep | 0.998–1.000 |

**Gap to SOTA: 0 points.** On LeapGestRecog the controlled near-infrared capture makes HOG features trivially separable; SVM matches deep-learning SOTA with no learned weights. This is the cleanest empirical evidence in the project that **dataset difficulty, not model capacity, dominates classical-vs-deep gap**.

### 7.3 Quantitative — valence regression

No widely cited SOTA exists for valence regression *on FER2013 class-mapped targets* (it is a per-project formulation), so the comparison is against the linear baseline:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.582 | 0.499 | 0.211 |
| **Polynomial (deg-2 Ridge α=5)** | **0.562** | **0.452** | **0.263** |

For continuous-affect benchmarks (e.g. AffectNet's valence/arousal labels), recent CNN-based regressors reach R² ≈ 0.6 — again an expected gap given no learned features and no augmentation.

### 7.4 Qualitative comparison

| Axis | This project | Deep-learning SOTA |
|---|---|---|
| Feature representation | Hand-crafted HOG → linear PCA | End-to-end learned filters (CNN/ViT) |
| Invariance to lighting / pose | Limited (HOG handles edges only) | Strong (data-augmented learned features) |
| Class-confusion pattern (FER2013) | Confuses fear/sad and angry/disgust — visually similar low-arousal classes | Same confusions, but smaller magnitudes |
| Inference latency (CPU) | KNN: ~4 s / 7,178; DT: ~10 ms; SVM: ~35 s | A small CNN: ~1–2 s / 7,178 on CPU, ms on GPU |
| Training cost | Seconds–minutes, single CPU | Hours, GPU required |
| Calibration on imbalanced classes (`disgust`, n=436) | SVM precision 0.80 / recall 0.36 — over-confident on the minority class | CNNs with class-balanced loss reach precision/recall ≈ 0.6 / 0.6 |
| Reproducibility | All seeds = 42, all artifacts cached, deterministic | Seeds + GPU non-determinism, often non-bitwise-reproducible |
| Suitability for embedded HRI | Bagging + Decision Tree at ms-level inference are deployable on-device | Requires NPU/quantization for real-time on-device inference |

### 7.5 Summary

The classical pipeline implemented here is **competitive on a controlled dataset (gesture, ≈ SOTA)** and **a clear lower-bound baseline on an in-the-wild dataset (FER2013, 22–26 points below SOTA)**. The result aligns with the well-known classical-vs-deep gap on natural images and confirms the project's stated framing — a deliberate baseline study, not a SOTA attempt — while the gesture results show that for clean, controlled visual tasks a HOG+SVM pipeline still matches modern deep models.

## References

1. Dalal, N. and Triggs, B. *Histograms of Oriented Gradients for Human Detection.* CVPR 2005.
2. Goodfellow, I. et al. *Challenges in Representation Learning: A Report on Three Machine Learning Contests* (FER2013). 2013.
3. Pramerdorfer, C. and Kampel, M. *Facial Expression Recognition using Convolutional Neural Networks: State of the Art.* 2016.
4. Barsoum, E. et al. *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution* (FER+). ICMI 2016.
5. Khaireddin, Y. and Chen, Z. *Facial Emotion Recognition: State of the Art Performance on FER2013.* 2021.
6. Mantecón, T. et al. *Hand Gesture Recognition using Infrared Imagery Provided by Leap Motion Controller.* ACIVS 2016.
7. Sutton, R. S. and Barto, A. G. *Reinforcement Learning: An Introduction.* MIT Press, 2nd ed., 2018.
8. Pedregosa, F. et al. *scikit-learn: Machine Learning in Python.* JMLR 12, 2011.
