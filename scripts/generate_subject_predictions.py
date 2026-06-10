from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TRIAL_FILE = RESULTS_DIR / "trial_predictions_5fold.csv"
OUTPUT_DIR = RESULTS_DIR / "predictions"


def build_subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["subject", "fold"], dropna=False)
        .agg(y_true=("y_true", "first"), mean_prob=("prob", "mean"), n_trials=("prob", "size"))
        .reset_index()
    )
    grouped["pred_prob"] = (grouped["mean_prob"] >= 0.5).astype(int)
    grouped["pred_vote"] = grouped["pred_prob"]

    vote_counts = (
        df.assign(pred=(df["prob"] >= 0.5).astype(int))
        .groupby(["subject", "fold", "pred"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    renamed_cols = []
    for col in vote_counts.columns:
        if col == 0:
            renamed_cols.append("n_votes_0")
        elif col == 1:
            renamed_cols.append("n_votes_1")
        else:
            renamed_cols.append(col)
    vote_counts.columns = renamed_cols
    if "n_votes_0" not in vote_counts.columns:
        vote_counts["n_votes_0"] = 0
    if "n_votes_1" not in vote_counts.columns:
        vote_counts["n_votes_1"] = 0

    summary = grouped.merge(vote_counts[["subject", "fold", "n_votes_0", "n_votes_1"]], on=["subject", "fold"], how="left")
    summary["vote_pct_0"] = summary["n_votes_0"] / summary["n_trials"]
    summary["vote_pct_1"] = summary["n_votes_1"] / summary["n_trials"]
    summary["correct_vote"] = (summary["pred_vote"] == summary["y_true"]).astype(int)
    summary["correct_prob"] = (summary["pred_prob"] == summary["y_true"]).astype(int)
    return summary.sort_values(["fold", "subject"]).reset_index(drop=True)


def build_fold_metrics(subject_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fold in sorted(subject_summary["fold"].unique()):
        fold_df = subject_summary[subject_summary["fold"] == fold].copy()
        y_true = fold_df["y_true"].astype(int)
        y_pred = fold_df["pred_vote"].astype(int)
        y_score = fold_df["mean_prob"].astype(float)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        auc = roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else float("nan")
        rows.append(
            {
                "fold": int(fold),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred)),
                "auc": float(auc),
                "n_subjects": int(len(fold_df)),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not TRIAL_FILE.exists():
        raise FileNotFoundError(f"Missing trial predictions file: {TRIAL_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(TRIAL_FILE)
    required = {"fold", "subject", "y_true", "prob"}
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"Missing columns in {TRIAL_FILE.name}: {sorted(missing)}")

    trials = trials.sort_values(["fold", "subject"]).reset_index(drop=True)
    subject_summary = build_subject_summary(trials)
    fold_metrics = build_fold_metrics(subject_summary)
    all_cm = confusion_matrix(subject_summary["y_true"], subject_summary["pred_vote"], labels=[0, 1])

    trial_path = OUTPUT_DIR / "trial_predictions_5fold.csv"
    subject_path = OUTPUT_DIR / "subject_summary_5fold.csv"
    fold_path = OUTPUT_DIR / "fold_metrics_5fold.csv"
    cm_path = OUTPUT_DIR / "subject_confusion_matrix_5fold.csv"

    trials.to_csv(trial_path, index=False)
    subject_summary.to_csv(subject_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    pd.DataFrame(all_cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(cm_path)

    print("Saved:")
    print(f"  {trial_path}")
    print(f"  {subject_path}")
    print(f"  {fold_path}")
    print(f"  {cm_path}")
    print("Subject summary preview:")
    print(subject_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()