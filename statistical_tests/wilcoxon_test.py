from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "many_seed_experiments"
OUTPUT_DIR = RESULTS_DIR / "stat_tests"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().lower().replace(" ", "_").replace("-", "_") for column in df.columns]
    return df


def scale_percent(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric
    if numeric.dropna().abs().max() <= 1.5:
        return numeric * 100.0
    return numeric


def find_model_name(bundle_root: str) -> str:
    text = bundle_root.lower().replace(" ", "_")
    if "results_ourmodel" in text or "eegformer_10seeds" in text:
        return "EEGFormer"
    if "results_cnn_lstm_eegnet" in text or "cnn_lstm_eegnet_10seeds" in text:
        return "CNN-LSTM EEGNet"
    if "results_tgarnet" in text or "tgarnet_10seeds" in text:
        return "T-GARNet"
    if "results_eegnet" in text or "eegnet_10seeds" in text:
        return "EEGNet"
    if "results_shallowconvnet" in text or "shallowconvnet_10seeds" in text:
        return "ShallowConvNet"
    if "results_imcbgt" in text or "imcbgt_10seeds" in text:
        return "IM-CBGT"
    if "results_multistream" in text or "multistream_10seeds" in text:
        return "Multi-Stream"
    return bundle_root


def load_all_folds() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for folder_path in sorted(EXPERIMENTS_DIR.iterdir() if EXPERIMENTS_DIR.exists() else []):
        if not folder_path.is_dir():
            continue
        fold_path = folder_path / "all_fold_results.csv"
        if not fold_path.exists():
            continue
        df = pd.read_csv(fold_path)
        df = clean_columns(df)
        if "seed" not in df.columns or "fold" not in df.columns:
            continue
        df["model_name"] = find_model_name(folder_path.name)
        df["pair_id"] = df["seed"].astype(int).astype(str) + "_fold_" + df["fold"].astype(int).astype(str)
        for metric_name in ["accuracy", "balanced_acc", "f1", "auc"]:
            if metric_name in df.columns:
                df[metric_name] = scale_percent(df[metric_name])
        frames.append(df[[column for column in ["model_name", "pair_id", "accuracy", "balanced_acc", "f1", "auc"] if column in df.columns]])

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_folds = load_all_folds()

    rows: list[dict[str, object]] = []
    for metric_name in ["accuracy", "balanced_acc", "f1", "auc"]:
        if metric_name not in all_folds.columns:
            continue
        wide = all_folds.pivot_table(index="pair_id", columns="model_name", values=metric_name, aggfunc="first")
        wide = wide.dropna(axis=0, how="any")
        ranks = wide.rank(axis=1, ascending=False, method="average")
        avg_ranks = ranks.mean(axis=0)
        models = list(wide.columns)
        for first_index, model_a in enumerate(models):
            for model_b in models[first_index + 1 :]:
                values_a = wide[model_a]
                values_b = wide[model_b]
                try:
                    statistic, p_value = wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    statistic, p_value = 0.0, 1.0
                rows.append(
                    {
                        "metric": metric_name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_pairs": int(len(wide)),
                        "mean_a": float(values_a.mean()),
                        "mean_b": float(values_b.mean()),
                        "mean_diff": float((values_a - values_b).mean()),
                        "wilcoxon_statistic": statistic,
                        "avg_rank_a": float(avg_ranks[model_a]),
                        "avg_rank_b": float(avg_ranks[model_b]),
                        "p_value": p_value,
                    }
                )

    results = pd.DataFrame(rows).sort_values(["metric", "p_value", "model_a", "model_b"]).reset_index(drop=True)
    results.to_csv(OUTPUT_DIR / "wilcoxon_results.csv", index=False)

    print("Wilcoxon pairwise results")
    print(results.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'wilcoxon_results.csv'}")


if __name__ == "__main__":
    main()