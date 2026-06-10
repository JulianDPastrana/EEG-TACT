#!/usr/bin/env python3
"""Rebuild EEGFormer 5-fold all_trial_probs.csv from saved fold weights/hyperparameters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from src.models import EEGDataset_ADHD_TF, build_epoch_arrays, build_model


SEED = 123
N_FOLDS = 5
THRESHOLD = 0.5
BATCH_SIZE = 64
EXCLUDED_SUBJECTS = {"v56p"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "eegformer_5fold_run",
        help="Directory containing optuna/, weights/, and trial_predictions_5fold.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing ADHD/ and Control/ folders with .mat files",
    )
    return parser.parse_args()


def load_hp_for_fold(fold_id: int, optuna_dir: Path) -> dict:
    path = optuna_dir / f"best_hp_fold_{fold_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing hyperparameter file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_from_hp(hp: dict, n_channels: int, n_samples: int) -> tf.keras.Model:
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


def _read_group_vars(group) -> list[np.ndarray]:
    vars_group = group["vars"]
    return [vars_group[str(index)][()] for index in range(len(vars_group))]


def load_model_from_h5_weights(model: tf.keras.Model, weights_path: Path) -> tf.keras.Model:
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


def load_model_for_fold(
    fold_id: int,
    n_channels: int,
    n_samples: int,
    optuna_dir: Path,
    weights_dir: Path,
) -> tf.keras.Model:
    hp = load_hp_for_fold(fold_id, optuna_dir)
    model = build_model_from_hp(hp, n_channels, n_samples)
    _ = model(tf.zeros((1, n_channels, n_samples), dtype=tf.float32), training=False)
    weights_path = weights_dir / f"fold_{fold_id}.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    return load_model_from_h5_weights(model, weights_path)


def load_fold_subjects(split_trial_file: Path) -> dict[int, list[str]]:
    if not split_trial_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_trial_file}")

    split_df = pd.read_csv(split_trial_file)
    required = {"fold", "subject", "y_true", "prob"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Split file missing columns: {sorted(missing)}")

    fold_subjects: dict[int, list[str]] = {}
    for fold in range(N_FOLDS):
        fold_subjects[fold] = sorted(split_df.loc[split_df["fold"] == fold, "subject"].astype(str).unique())

    # Leakage guard from split file.
    counts = split_df.groupby("subject")["fold"].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        raise ValueError(f"Leakage in split file. Subjects in multiple folds: {leaked.index.tolist()}")

    return fold_subjects


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_root = args.data_root.resolve()
    optuna_dir = run_dir / "optuna"
    weights_dir = run_dir / "weights"
    split_trial_file = run_dir / "trial_predictions_5fold.csv"
    output_file = run_dir / "all_trial_probs.csv"

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print(f"Run directory: {run_dir}")
    print(f"Data root: {data_root}")
    print(f"Using split file: {split_trial_file}")

    adhd_dir = data_root / "ADHD"
    control_dir = data_root / "Control"
    if not adhd_dir.exists() or not control_dir.exists():
        raise FileNotFoundError(
            "Missing EEG data folders. Expected ADHD/ and Control/ under data root: "
            f"{data_root}. You can pass --data-root <path>."
        )

    fold_subjects = load_fold_subjects(split_trial_file)

    dataset = EEGDataset_ADHD_TF(
        adhd_dir=str(adhd_dir),
        control_dir=str(control_dir),
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

    x_all, y_all, groups_all = build_epoch_arrays(dataset)
    groups_all = groups_all.astype(str)
    n_channels, n_samples = int(x_all.shape[1]), int(x_all.shape[2])

    rows: list[dict[str, object]] = []

    for fold in range(N_FOLDS):
        tf.keras.backend.clear_session()
        model = load_model_for_fold(fold, n_channels, n_samples, optuna_dir, weights_dir)

        subjects = np.array(fold_subjects[fold]).astype(str)
        mask = np.isin(groups_all, subjects)
        indices = np.where(mask)[0]

        x_fold = x_all[indices]
        y_fold = y_all[indices].astype(int)
        subjects_fold = groups_all[indices]

        probs = model.predict(x_fold, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        preds = (probs >= THRESHOLD).astype(int)

        fold_df = pd.DataFrame(
            {
                "seed": SEED,
                "fold": int(fold),
                "subject": subjects_fold,
                "true_label": y_fold,
                "trial_global_idx": indices.astype(int),
                "prob_adhd": probs.astype(float),
                "pred_epoch": preds.astype(int),
            }
        ).sort_values(["subject", "trial_global_idx"], kind="mergesort")

        fold_df["trial_local_idx"] = fold_df.groupby("subject").cumcount().astype(int)

        subj_vote = (
            fold_df.groupby("subject", as_index=False)
            .agg(subject_majority_pred=("pred_epoch", lambda s: int(s.mean() >= 0.5)), subject_mean_prob=("prob_adhd", "mean"))
        )

        fold_df = fold_df.merge(subj_vote, on="subject", how="left", validate="many_to_one")
        rows.append(fold_df)

        print(f"Fold {fold}: subjects={len(subjects)} trials={len(fold_df)}")

    out_df = pd.concat(rows, ignore_index=True)
    out_df = out_df[
        [
            "seed",
            "fold",
            "subject",
            "true_label",
            "trial_global_idx",
            "trial_local_idx",
            "prob_adhd",
            "pred_epoch",
            "subject_majority_pred",
            "subject_mean_prob",
        ]
    ].sort_values(["fold", "subject", "trial_local_idx"], kind="mergesort")

    out_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
