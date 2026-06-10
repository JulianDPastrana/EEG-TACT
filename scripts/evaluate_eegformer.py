#!/usr/bin/env python3
"""Leakage-safe EEGFormer evaluation using fixed fold assignments from saved CSVs."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from src.models import EEGDataset_ADHD_TF, build_epoch_arrays, build_model


SEED = 123
N_FOLDS = 5
BATCH_SIZE = 64
THRESHOLD = 0.5
EXCLUDED_SUBJECTS = {"v56p"}

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OPTUNA_DIR = RESULTS_DIR / "optuna"
WEIGHTS_DIR = RESULTS_DIR / "weights"

# Use the original root split files (these produce the expected ~0.84 trial accuracy scale)
SPLIT_TRIAL_FILE = RESULTS_DIR / "trial_predictions_5fold.csv"
SPLIT_SUBJECT_FILE = RESULTS_DIR / "subject_trial_stats_5fold.csv"

OUTPUT_DIR = RESULTS_DIR / "evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _read_group_vars(group):
    vars_group = group["vars"]
    return [vars_group[str(index)][()] for index in range(len(vars_group))]


def load_model_from_h5_weights(model, weights_path):
    with h5py.File(weights_path, "r") as handle:
        layers_group = handle["layers"]

        layer_map = {
            "eeg_temporal": ("conv2d", None),
            "bn_t": ("batch_normalization", None),
            "eeg_depthwise": ("depthwise_conv2d", None),
            "bn_dw": ("batch_normalization_1", None),
            "eeg_separable": ("separable_conv2d", None),
            "bn_sep": ("batch_normalization_2", None),
            "attn_pool": ("attention_pooling", "attn"),
            "classifier": ("dense", None),
        }

        for layer_name, (file_layer_name, nested_name) in layer_map.items():
            layer = model.get_layer(layer_name)
            layer_group = layers_group[file_layer_name]
            weights = _read_group_vars(layer_group if nested_name is None else layer_group[nested_name])
            layer.set_weights(weights)

        encoder = model.get_layer("encoder")
        encoder_group = layers_group["transformer_encoder"]
        encoder_layer_names = ["transformer_encoder_layer", "transformer_encoder_layer_1"]

        for i, encoder_layer in enumerate(encoder.layers_):
            block_group = encoder_group["layers"][encoder_layer_names[i]]
            encoder_layer.norm1.set_weights(_read_group_vars(block_group["norm1"]))
            encoder_layer.norm2.set_weights(_read_group_vars(block_group["norm2"]))
            ffn_group = block_group["ffn"]["layers"]
            encoder_layer.ffn.layers[0].set_weights(_read_group_vars(ffn_group["dense"]))
            encoder_layer.ffn.layers[2].set_weights(_read_group_vars(ffn_group["dense_1"]))
            attn_group = block_group["self_attn"]
            encoder_layer.self_attn._query_dense.set_weights(_read_group_vars(attn_group["query_dense"]))
            encoder_layer.self_attn._key_dense.set_weights(_read_group_vars(attn_group["key_dense"]))
            encoder_layer.self_attn._value_dense.set_weights(_read_group_vars(attn_group["value_dense"]))
            encoder_layer.self_attn._output_dense.set_weights(_read_group_vars(attn_group["output_dense"]))

    return model


def load_hp_for_fold(fold_id):
    hp_path = OPTUNA_DIR / f"best_hp_fold_{fold_id}.json"
    if not hp_path.exists():
        raise FileNotFoundError(f"Missing hyperparameter file: {hp_path}")
    with open(hp_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model_from_hp(hp, n_channels, n_samples):
    d_model = None if int(hp.get("use_proj", 0)) == 0 else int(hp["d_model"])
    return build_model(
        n_channels=n_channels,
        n_samples=n_samples,
        F1=int(hp["F1"]),
        D=int(hp["D"]),
        F2=int(hp["F2"]),
        kern_length=int(hp["kern_length"]),
        pool1=int(hp["pool1"]),
        pool2=int(hp["pool2"]),
        eeg_activation=str(hp["eeg_activation"]),
        d_model=d_model,
        nhead=int(hp["nhead"]),
        dim_feedforward=int(hp["dim_feedforward"]),
        num_layers=int(hp["num_layers"]),
        do_rate_transf=float(hp["do_rate_transf"]),
        do_rate_eeg=float(hp["do_rate_eeg"]),
        do_rate_cls=float(hp["do_rate_cls"]),
    )


def load_model_for_fold(fold_id, n_channels, n_samples):
    hp = load_hp_for_fold(fold_id)
    model = build_model_from_hp(hp, n_channels, n_samples)
    _ = model(tf.zeros((1, n_channels, n_samples), dtype=tf.float32), training=False)
    weights_path = WEIGHTS_DIR / f"fold_{fold_id}.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weight file: {weights_path}")
    return load_model_from_h5_weights(model, str(weights_path)), hp


def build_fold_subjects_from_split_file(split_df):
    required_cols = {"fold", "subject", "y_true", "prob"}
    missing = required_cols - set(split_df.columns)
    if missing:
        raise ValueError(f"Split trial file is missing columns: {sorted(missing)}")

    fold_subjects = {}
    for fold_id in range(N_FOLDS):
        fold_subjects[int(fold_id)] = sorted(split_df.loc[split_df["fold"] == fold_id, "subject"].astype(str).unique())

    # Leakage guard: no subject can appear in more than one test fold
    subject_to_folds = split_df.groupby("subject")["fold"].nunique()
    leaked_subjects = subject_to_folds[subject_to_folds > 1]
    if not leaked_subjects.empty:
        raise ValueError(
            "Leakage detected in split file: some subjects appear in multiple folds: "
            f"{leaked_subjects.index.tolist()}"
        )
    return fold_subjects


def evaluate_fold_trial_level(model, x_fold, y_fold, batch_size, threshold):
    probs = model.predict(x_fold, batch_size=batch_size, verbose=0).reshape(-1)
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_fold, preds)
    bacc = balanced_accuracy_score(y_fold, preds)
    f1 = f1_score(y_fold, preds)
    auc = roc_auc_score(y_fold, probs) if len(np.unique(y_fold)) > 1 else np.nan
    cm = confusion_matrix(y_fold, preds, labels=[0, 1])
    return {
        "accuracy": float(acc),
        "balanced_acc": float(bacc),
        "f1": float(f1),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "cm": cm,
        "n_trials": int(len(y_fold)),
        "preds": preds,
        "probs": probs,
    }


def main():
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print(f"Using split trial file: {SPLIT_TRIAL_FILE}")
    print(f"Using split subject file: {SPLIT_SUBJECT_FILE}")

    split_trials_df = pd.read_csv(SPLIT_TRIAL_FILE)
    _ = pd.read_csv(SPLIT_SUBJECT_FILE)  # Existence/consistency check only

    fold_subjects = build_fold_subjects_from_split_file(split_trials_df)

    dataset = EEGDataset_ADHD_TF(
        adhd_dir=str(DATA_ROOT / "ADHD"),
        control_dir=str(DATA_ROOT / "Control"),
        lowcut=0.5,
        highcut=60.0,
        notch=50.0,
        window=2.0,
        overlap=0.5,
        default_fs=128,
    )

    dataset.samples = [
        sample for sample in dataset.samples if os.path.splitext(sample[0])[0] not in EXCLUDED_SUBJECTS
    ]
    X, y, groups = build_epoch_arrays(dataset)
    groups = groups.astype(str)

    n_channels, n_samples = int(X.shape[1]), int(X.shape[2])

    all_cm = np.zeros((2, 2), dtype=int)
    fold_rows = []

    for fold_id in range(N_FOLDS):
        tf.keras.backend.clear_session()
        model, hp = load_model_for_fold(fold_id, n_channels, n_samples)

        fold_names = np.array(fold_subjects[fold_id]).astype(str)
        fold_mask = np.isin(groups, fold_names)
        x_fold = X[fold_mask]
        y_fold = y[fold_mask].astype(int)

        metrics = evaluate_fold_trial_level(model, x_fold, y_fold, BATCH_SIZE, THRESHOLD)
        all_cm += metrics["cm"]

        fold_rows.append(
            {
                "fold": int(fold_id),
                "accuracy": metrics["accuracy"],
                "balanced_acc": metrics["balanced_acc"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
                "n_trials": metrics["n_trials"],
                "n_subjects": int(len(fold_names)),
                "tn": int(metrics["cm"][0, 0]),
                "fp": int(metrics["cm"][0, 1]),
                "fn": int(metrics["cm"][1, 0]),
                "tp": int(metrics["cm"][1, 1]),
                "num_layers": int(hp["num_layers"]),
                "F1": int(hp["F1"]),
                "D": int(hp["D"]),
                "F2": int(hp["F2"]),
            }
        )

    df_folds = pd.DataFrame(fold_rows)
    folds_path = OUTPUT_DIR / "fold_metrics.csv"
    df_folds.to_csv(folds_path, index=False)

    summary = pd.DataFrame(
        {
            "metric": ["accuracy", "balanced_acc", "f1", "auc"],
            "mean": [
                df_folds["accuracy"].mean(),
                df_folds["balanced_acc"].mean(),
                df_folds["f1"].mean(),
                df_folds["auc"].mean(),
            ],
            "variance": [
                df_folds["accuracy"].var(ddof=1),
                df_folds["balanced_acc"].var(ddof=1),
                df_folds["f1"].var(ddof=1),
                df_folds["auc"].var(ddof=1),
            ],
            "std": [
                df_folds["accuracy"].std(ddof=1),
                df_folds["balanced_acc"].std(ddof=1),
                df_folds["f1"].std(ddof=1),
                df_folds["auc"].std(ddof=1),
            ],
        }
    )
    summary_path = OUTPUT_DIR / "summary_statistics.csv"
    summary.to_csv(summary_path, index=False)

    cm_path = OUTPUT_DIR / "trial_confusion_matrix.csv"
    pd.DataFrame(all_cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(cm_path)

    print("\nPer-fold trial metrics:")
    print(df_folds[["fold", "accuracy", "balanced_acc", "f1", "auc", "n_trials"]].to_string(index=False))
    print("\nSummary (mean/variance):")
    print(summary.to_string(index=False))
    print("\nTrial confusion matrix:")
    print(all_cm)
    print(f"\nSaved: {folds_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()
