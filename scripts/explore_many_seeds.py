from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "many_seed_experiments"
OUTPUT_DIR = RESULTS_DIR / "many_seed_summary"


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


def read_summary_from_folder(folder_path: Path) -> tuple[str, pd.DataFrame] | None:
    summary_path = folder_path / "all_seed_summaries.csv"
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    return find_model_name(folder_path.name), clean_columns(df)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    metric_names = ["accuracy", "balanced_acc", "f1", "auc"]

    for folder_path in sorted(EXPERIMENTS_DIR.iterdir() if EXPERIMENTS_DIR.exists() else []):
        if not folder_path.is_dir():
            continue
        loaded = read_summary_from_folder(folder_path)
        if loaded is None:
            continue

        model_name, df = loaded
        row: dict[str, object] = {
            "model_name": model_name,
            "experiment_folder": folder_path.name,
            "n_seeds": int(df["seed"].nunique()) if "seed" in df.columns else len(df),
        }

        for metric_name in metric_names:
            mean_column = f"mean_{metric_name}"
            if mean_column not in df.columns:
                row[metric_name] = pd.NA
                continue
            row[metric_name] = scale_percent(df[mean_column]).mean()

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
    summary.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    print("Many-seed model metrics")
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()