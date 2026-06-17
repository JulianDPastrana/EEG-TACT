# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

EEG-TACT classifies ADHD vs. Control from 19-channel, 128 Hz resting-state EEG using Multiple Instance Learning (subject = bag of 2-second, 50%-overlap epochs). The main model is a Keras/TensorFlow EEGNet-style convolutional tokenizer → Transformer encoder → attention-pooling → sigmoid classifier. Baselines include EEGNet, ShallowConvNet, CNN-LSTM, Multi-Stream Transformer, T-GARNet (Keras), and EEGConformer (PyTorch).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Two frameworks are required: **TensorFlow/Keras** (main model + most baselines) and **PyTorch** (EEGConformer). EEGConformer Optuna HPO also requires `mne` for `EEGDataset_ADHD_TF` (used in `evaluate_eegformer.py`).

## Data

Place IEEE DataPort `.mat` files under `data/ADHD/` and `data/Control/`. Each file is one subject's recording. Data files are git-ignored (`.mat`, `.pkl`, `.csv`, `.json`).

**Build the PyTorch-compatible processed dataset (pickle):**
```bash
cd src/data && python build_dataset.py
```
Output: `data/processed/eeg_dataset.pkl` — a dict with keys `epochs` (N, 19, 256), `labels` (N,), `groups` (N,), `subject_map`.

The Keras path (`evaluate_eegformer.py`, `eegformer_5fold_run.ipynb`) does not use this pickle; it re-processes `.mat` files on the fly via `EEGDataset_ADHD_TF` in `src/models/keras/eegformer.py`.

## Running scripts

All scripts use absolute imports rooted at the repo root; run them from the repo root:

```bash
# Evaluate trained EEG-TACT folds (requires results/optuna/ and results/weights/)
python scripts/evaluate_eegformer.py

# Aggregate multi-seed results (requires results/many_seed_experiments/)
python scripts/explore_many_seeds.py

# Unzip and organize baseline seed result archives into results/many_seed_experiments/
python scripts/organize_many_seed_experiments.py

# Generate per-subject prediction CSVs
python scripts/generate_subject_predictions.py

# Rebuild trial probability CSVs for EEG-TACT 5-fold run
python scripts/rebuild_eegformer_5fold_all_trial_probs.py
```

**Statistical tests (run from repo root):**
```bash
python statistical_tests/proportion_tests.py   # McNemar exact test on trial_predictions CSVs
python statistical_tests/wilcoxon_test.py      # Wilcoxon signed-rank on 10-seed fold results
```

## Architecture

### Main model — `src/models/keras/eegformer.py`

`build_model()` returns a Keras `Model` named `EEGNet_Transformer_SoftmaxFixed`:
1. **EEGNet tokenizer**: temporal Conv2D → depthwise Conv2D (spatial filter) → BN/ELU/AvgPool → separable Conv2D → Reshape to token sequence (T_tokens, F2)
2. **Optional projection**: Dense to `d_model` if `d_model ≠ F2`
3. **TransformerEncoder**: stacked `TransformerEncoderLayer` (pre-norm with `RMSNorm`, GELU FFN, MHA)
4. **AttentionPooling**: learned softmax weights over token sequence → weighted sum → scalar context vector
5. **Classifier**: Dense(1, sigmoid) with max-norm constraint

This file also contains the dataset utilities used by the Keras training path:
- `EEGDataset_ADHD_TF` — loads and preprocesses `.mat` files using `mne`
- `build_epoch_arrays` — converts dataset object to numpy arrays (X, y, groups)
- `make_ds_from_indices` — creates `tf.data.Dataset` from epoch arrays
- `ValBalancedAccuracy` — Keras callback for validation balanced accuracy

### EEGConformer (PyTorch) — `src/models/torch/eegconformer.py`

`EEGConformer`: `PatchEmbedding` (temporal conv → spatial conv → BN/ELU/AvgPool → projection) → `nn.TransformerEncoder` → `ClassificationHead` (flatten → Linear(256) → Linear(32) → Linear(K)).

Optuna objective for this model: `src/tuning/objective.py` — runs 3-fold inner CV, early stopping on subject-level accuracy, Optuna pruning via `TrialPruned`.

### Baseline Keras models — `src/models/keras/`

`eegnet.py`, `shallowconvnet.py`, `cnn_lstm_eegnet.py`, `multi_stream.py`, `tgarnet.py` — standalone Keras models.

### Training utilities — `src/training/trainer.py`

PyTorch-only. `train_one_epoch` and `evaluate`: epoch-level predictions via logit threshold at 0 (BCEWithLogitsLoss), subject-level via majority voting over epochs per group.

### Data pipeline — `src/data/`

`preprocessing.py` constants: `NUM_CHANNELS = 19`, `SAMPLE_FREQUENCY = 128`. Pipeline: load `.mat` → bandpass Butterworth 0.5–60 Hz (order 4) + notch 50 Hz (Q=30) → 2-second epochs with 1-second step → epoch-wise z-score normalization.

## Results directory layout

```
results/
  optuna/           # best_hp_fold_{i}.json — per-fold best hyperparameters
  weights/          # fold_{i}.weights.h5 — trained Keras model weights
  evaluation/       # CSVs from evaluate_eegformer.py
  many_seed_experiments/   # per-model seed result bundles (git-ignored model weights)
  many_seed_summary/       # aggregated CSV from explore_many_seeds.py
  stat_tests/       # CSVs from proportion_tests.py and wilcoxon_test.py
```

Result CSVs are git-ignored; zip archives of baseline seed results are extracted by `organize_many_seed_experiments.py` and then deleted.

## Validation protocol

- **Outer CV**: Stratified Group K-Fold (5 folds), grouping by subject so no subject leaks across folds.
- **Inner HPO**: Optuna TPE on a 3-fold inner split of each training fold.
- **Evaluation regimes**: "best-seed" (single deterministic run, for McNemar) and "varying-seed" (10 seeds × 5 folds, for Wilcoxon).
- `EXCLUDED_SUBJECTS = {"v56p"}` is hardcoded in `evaluate_eegformer.py`.
