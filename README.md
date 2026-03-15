# SensorHAR-GRU

**Human Activity Recognition (HAR)** from wearable sensor data using deep learning (GRU, 1D-CNN) and classical ML (Naive Bayes, XGBoost), with data augmentation, normalization, and self-supervised transfer learning

> Built as part of a **Practical Deep Learning Workshop** — a Kaggle-style competition to classify 18 human activities from accelerometer time-series data recorded by smartwatch and Vicon sensors.

---

## Problem Statement

Given **X, Y, Z acceleration sequences** recorded from sensors on different body parts, classify which of **18 activities** the user was performing (walking, stairs up, washing hands, etc.).

The dataset consists of recordings from 8 users captured by two sensor types (**smartwatch** and **Vicon**), with varying sequence lengths, units, and sensor placements. The total dataset contains **124,992 samples**, of which **50,248 are labeled** (40%) and the rest are unlabeled (60%).

---

## Data Pipeline

### Preprocessing
- **Two sensor types** (Type 1 & Type 2) normalized separately using per-type mean/std standardization
- Sequences padded to a fixed length of **4,000 timesteps**
- Null/NaN sequences filtered out

### Augmentation (applied during training)
| Technique | Details |
|---|---|
| **Gaussian jitter** | noise with mean=0, std=0.05 |
| **Uniform scaling** | additive factor in range [-0.03, 0.1] |
| **Random cropping** | Random sub-window of the sequence (rest zero-padded) |

Augmentation + normalization proved to be a **game changer** — the model generalized significantly better, with validation accuracy exceeding training accuracy at each epoch.

### Data Split
| Set | Percentage | Count |
|-----|-----------|-------|
| Train | 72% | ~36k |
| Validation | 8% | ~4k |
| Test | 20% | ~10k |

---

## Models & Results

### Classical ML (`ml_models.ipynb`)

Hand-crafted statistical features extracted from sequences (mean, std, max, min, median per axis), with experiments on different feature engineering approaches:

1. **Subsampled** sequences → 6 features (mean + std per axis)
2. **Segment-based** features → (num_segments × 6)
3. **Moving-average smoothed** sequences → 6 features
4. Smoothed + segmented

| Model | Approach | Train Acc | Test Acc | Notes |
|---|---|---|---|---|
| **Naive Bayes** | Smoothed (3) | 0.382 | 0.378 | Most stable |
| **XGBoost** | Smoothed (3) | 0.765 | **0.575** | Best ML result |

**Key insights:**
- XGBoost overfits heavily on raw features; smoothing helps
- Naive Bayes is more stable but less powerful
- ML baseline: **~0.48 accuracy** → target to beat with deep learning

### 1D-CNN (`gru_activity_recognition.py`)
3 convolutional blocks (64→128→256 filters) + adaptive pooling + FC classifier.

| Set | Accuracy | Loss |
|-----|----------|------|
| Train | 0.980 | 0.115 |
| Val | 0.776 | 0.978 |
| **Test** | **0.761** | 0.986 |

Signs of overfitting — large gap between train/val accuracy.

### GRU Network (`gru_activity_recognition.py`)
2-layer GRU (hidden_size=100) + FC head with BatchNorm, Dropout, and Sigmoid.

| Set | Accuracy | Loss |
|-----|----------|------|
| Train | 0.770 | 0.571 |
| Val | 0.772 | 0.615 |
| **Test** | **0.771** | 0.588 |

Better generalization than 1D-CNN — train/val gap is minimal.

### GRU + Augmentation & Normalization (final model)

After applying normalization + augmentation, the model generalizes better (val > train at each epoch) but learns slower:

| Set | Accuracy | Loss |
|-----|----------|------|
| Train | 0.678 | 0.800 |
| Val | 0.706 | 0.718 |
| **Test** | **0.727** | 0.674 |

### Kaggle Competition Results

| Model | Loss on dataset | Kaggle Score |
|-------|----------------|--------------|
| GRU (no aug/norm) | ~0.458 | 2.272 |
| **GRU + aug + norm** | 0.799 | **1.778** |

**The augmented model scored 28% better on the competition leaderboard** despite higher training loss — demonstrating the value of generalization.

### Self-Supervised: LSTM Autoencoder (transfer learning)

Trained an LSTM Autoencoder on the full **unlabeled dataset** (60%), then used the 128-dim encoder embeddings as input to a classifier.

| Set | Accuracy | Loss |
|-----|----------|------|
| Train | 0.098 | 2.720 |
| Val | 0.102 | 2.720 |

Results were poor — likely needs larger embedding size and more complex downstream architecture. Implemented after the competition ended as a proof-of-concept.

---

## Training Configuration

```python
model = GRUNet(input_size=3, hidden_size=100, num_layers=2, num_classes=18)
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.005) |
| Scheduler | StepLR (step=50, γ=0.99) |
| Loss | CrossEntropyLoss |
| Batch Size | 128 |

---

## Future Work

1. **Ensemble learning** — combine predictions from GRU, 1D-CNN, and ML models via a meta-learner or feature concatenation from penultimate layers
2. **Self-supervised pretraining** — improve the LSTM Autoencoder with larger embedding dimension and more complex classifier
3. **Additional features** — skewness, kurtosis, frequency-domain features (FFT)
4. **K-fold cross-validation** — better generalization estimates (not feasible during competition due to compute constraints)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
SensorHAR-GRU/
├── gru_activity_recognition.py
├── ml_models.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```
