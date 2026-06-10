# EEG-TACT

Code, data, and results for **EEG-TACT** — Temporal Attention and Convolutional Tokenization for Interpretable ADHD Classification from EEG \dc{in Children

## Overview

- **Task**: subject-level binary classification (ADHD vs. Control) from 19-channel,
  128 Hz resting-state EEG, framed as Multiple Instance Learning (subject = bag of
  2 s / 50%-overlap epoch instances).
- **Model**: EEGNet-style temporal/spatial/separable
  convolutional feature extractor -> 1-layer Transformer encoder -> attention-pooling -> classifier (~7.3k parameters).
- **Baselines**: EEGNet, ShallowConvNet, CNN-LSTM, Multi-Stream Transformer,
  EEGConformer, T-GARNet, IM-CBGT.
- **Validation**: Stratified Group K-Fold (5 folds, subject-level grouping), with
  per-fold Optuna (TPE) hyperparameter search.
- **Evaluation regimes**: "best-seed" (single seed, used for McNemar) and
  "varying-seed" (10 seeds × 5 folds, used for Wilcoxon signed-rank).
- **Ablation**: 4 configurations isolating the contribution of the Transformer
  encoder and the attention-pooling head.


## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Two deep-learning frameworks are required: TensorFlow/Keras (EEG-TACT and the Keras
baselines) and PyTorch (EEGConformer).

## Data

Place the IEEE DataPort ADHD/Control resting-state EEG `.mat` files under
`data/ADHD/` and `data/Control/`. Each file is a single subject's 19-channel, 128 Hz recording.
