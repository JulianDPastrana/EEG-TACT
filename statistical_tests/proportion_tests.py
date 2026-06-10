from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "stat_tests" / "proportions"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().lower().replace(" ", "_").replace("-", "_") for column in df.columns]
    return df


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    z = 1.959963984540054
    phat = successes / total
    denominator = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denominator
    margin = z * math.sqrt((phat * (1.0 - phat) / total) + (z * z) / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_exact(correct_a: pd.Series, correct_b: pd.Series) -> tuple[int, int, float, float]:
    discordant_a = int(((correct_a == 1) & (correct_b == 0)).sum())
    discordant_b = int(((correct_a == 0) & (correct_b == 1)).sum())
    total = discordant_a + discordant_b
    if total == 0:
        return discordant_a, discordant_b, 0.0, 1.0
    statistic = ((abs(discordant_a - discordant_b) - 1.0) ** 2) / total
    p_value = binomtest(min(discordant_a, discordant_b), total, 0.5, alternative="two-sided").pvalue
    return discordant_a, discordant_b, statistic, p_value


def find_prediction_files() -> list[Path]:
    files = []
    for csv_path in sorted(RESULTS_DIR.rglob("trial_predictions*.csv")):
        if "/predictions/" in csv_path.as_posix():
            continue
        files.append(csv_path)
    return files


def model_name_from_path(csv_path: Path) -> str:
    if csv_path.parent == RESULTS_DIR:
        return "EEGFormer"
    return csv_path.parent.name.replace("_", " ")


def load_trials(csv_path: Path) -> pd.DataFrame:
    df = clean_columns(pd.read_csv(csv_path))
    if "pred" not in df.columns:
        df["pred"] = (pd.to_numeric(df["prob"], errors="coerce") >= 0.5).astype(int)
    df["trial_index"] = df.groupby(["fold", "subject"]).cumcount()
    df["sample_key"] = df["fold"].astype(str) + "|" + df["subject"].astype(str) + "|" + df["trial_index"].astype(str)
    df["correct"] = (df["pred"].astype(int) == df["y_true"].astype(int)).astype(int)
    return df


def subject_majority(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model_name", "fold", "subject"], dropna=False)
        .agg(y_true=("y_true", "first"), pred_sum=("pred", "sum"), n_trials=("pred", "size"))
        .reset_index()
    )
    grouped["pred"] = (grouped["pred_sum"] >= (grouped["n_trials"] / 2.0)).astype(int)
    grouped["correct"] = (grouped["pred"].astype(int) == grouped["y_true"].astype(int)).astype(int)
    grouped["sample_key"] = grouped["fold"].astype(str) + "|" + grouped["subject"].astype(str)
    return grouped


def make_accuracy_table(df: pd.DataFrame, level_name: str) -> pd.DataFrame:
    rows = []
    for model_name, group in df.groupby("model_name", dropna=False):
        total = int(len(group))
        correct = int(group["correct"].sum())
        ci_low, ci_high = wilson_interval(correct, total)
        rows.append(
            {
                "level": level_name,
                "model_name": model_name,
                "n_samples": total,
                "n_correct": correct,
                "accuracy": correct / total if total else math.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def make_mcnemar_table(df: pd.DataFrame, level_name: str) -> pd.DataFrame:
    wide = df.pivot_table(index="sample_key", columns="model_name", values="correct", aggfunc="first")
    wide = wide.dropna(axis=0, how="any")
    models = list(wide.columns)
    rows = []
    for first_index, model_a in enumerate(models):
        for model_b in models[first_index + 1 :]:
            discordant_a, discordant_b, statistic, p_value = mcnemar_exact(wide[model_a], wide[model_b])
            rows.append(
                {
                    "level": level_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_aligned_samples": int(len(wide)),
                    "discordant_a": discordant_a,
                    "discordant_b": discordant_b,
                    "mcnemar_statistic": statistic,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_files = find_prediction_files()
    if not prediction_files:
        message = pd.DataFrame([
            {
                "status": "no_prediction_files_found",
                "note": "Add trial_predictions CSV files to run proportion tests.",
            }
        ])
        message.to_csv(OUTPUT_DIR / "availability.csv", index=False)
        print(message.to_string(index=False))
        return

    trial_frames = []
    availability_rows = []
    for csv_path in prediction_files:
        model_name = model_name_from_path(csv_path)
        df = load_trials(csv_path)
        df["model_name"] = model_name
        trial_frames.append(df)
        availability_rows.append({"model_name": model_name, "file_path": str(csv_path)})

    all_trials = pd.concat(trial_frames, ignore_index=True)
    trial_accuracy = make_accuracy_table(all_trials, "trial")
    subject_accuracy = make_accuracy_table(subject_majority(all_trials), "subject")
    availability = pd.DataFrame(availability_rows)
    availability.to_csv(OUTPUT_DIR / "availability.csv", index=False)
    trial_accuracy.to_csv(OUTPUT_DIR / "trial_accuracy.csv", index=False)
    subject_accuracy.to_csv(OUTPUT_DIR / "subject_accuracy.csv", index=False)

    if all_trials["model_name"].nunique() >= 2:
        trial_mcnemar = make_mcnemar_table(all_trials, "trial")
        subject_mcnemar = make_mcnemar_table(subject_majority(all_trials), "subject")
        trial_mcnemar.to_csv(OUTPUT_DIR / "trial_mcnemar.csv", index=False)
        subject_mcnemar.to_csv(OUTPUT_DIR / "subject_mcnemar.csv", index=False)
        print(trial_mcnemar.to_string(index=False))
    else:
        note = pd.DataFrame([
            {
                "status": "single_model_only",
                "note": "McNemar tests need at least two models with aligned trial predictions.",
            }
        ])
        note.to_csv(OUTPUT_DIR / "pairwise_note.csv", index=False)
        print(note.to_string(index=False))

    print("\nTrial accuracy")
    print(trial_accuracy.to_string(index=False))
    print("\nSubject accuracy")
    print(subject_accuracy.to_string(index=False))


if __name__ == "__main__":
    main()