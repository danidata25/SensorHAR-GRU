# SensorHAR-GRU 🏃

A **Human Activity Recognition (HAR)** pipeline using both **deep learning** (GRU, LSTM, Conv1D) and **classical ML** models (Naive Bayes, XGBoost, Random Forest, Logistic Regression) on sensor time-series data, with support for **transfer learning**.

---

## Overview

This project classifies human activities from multi-axis accelerometer sensor data (X, Y, Z). It includes:
- **Deep learning**: GRU, LSTM, Conv1D architectures
- **Classical ML**: Naive Bayes, XGBoost, Random Forest, Logistic Regression
- Feature engineering (mean, std, max, min, median per axis)
- Transfer learning via a pre-trained Autoencoder (AE)
- Data augmentation (jitter, scaling, random crop)
- Sequence normalization and padding
- Submission pipeline for Kaggle-style competition

---

## Model Architectures

### GRUNet
- Multi-layer bidirectional GRU
- FC head: Linear → ReLU → BatchNorm → Dropout → Sigmoid → Linear
- Output: class logits

### LSTMNet
- Multi-layer bidirectional LSTM
- Same FC head as GRUNet

### Conv1DNet
- 3× (Conv1D → MaxPool) blocks
- AdaptiveAvgPool → Dropout → FC layers

### Transfer Learning
A pre-trained **Autoencoder (AE)** is frozen and used as a feature extractor (`EmbeddedModel`), with only the classification head fine-tuned.

### Classical ML Models (`ml_models.ipynb`)
Feature-engineered approach using handcrafted statistical features (mean, std, max, min, median per axis):

| Model | Library |
|---|---|
| **Naive Bayes** | sklearn GaussianNB (incremental `partial_fit`) |
| **XGBoost** | xgboost (incremental training) |
| **Random Forest** | sklearn RandomForestClassifier |
| **Logistic Regression** | sklearn LogisticRegression |

---

## Data

- **Type 1 & Type 2** sensor sequences (`.pkl` files)
- 50,248 total samples across 18 activity classes
- Sequences padded to fixed length of 4,000 timesteps
- 70% train / 15% validation / 15% test split

**Data augmentations:**
| Augmentation | Description |
|---|---|
| Jitter | Adds Gaussian noise |
| Scaling | Random additive scaling factor |
| Random Crop | Crops a fixed-length window |

---

## Training

```python
model = GRUNet(input_size=3, hidden_size=100, num_layers=2, num_classes=18)
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.005) |
| Scheduler | StepLR (step=50, γ=0.99) |
| Loss | CrossEntropyLoss |
| Batch Size | 128 |
| Epochs | 10 |

---

## Setup

```bash
pip install -r requirements.txt
python gru_activity_recognition.py
```

> **Note:** This project was originally developed on Google Colab with Google Drive for data access. Update data paths accordingly for local use.

---

## Project Structure

```
SensorHAR-GRU/
├── gru_activity_recognition.py   # Deep learning models (GRU, LSTM, Conv1D)
├── ml_models.ipynb               # Classical ML models (NB, XGB, RF, LR)
├── requirements.txt              # Dependencies
├── .gitignore
└── README.md
```

---

## Results

Outputs a classification report and submission CSV with per-class softmax probabilities.
