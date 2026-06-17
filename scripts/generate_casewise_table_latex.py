"""Generate a case-wise LaTeX longtable with one row per subject (N=120)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "results" / "predictions" / "subject_summary_5fold.csv"
OUT_CSV = REPO_ROOT / "results" / "evaluation" / "subject_casewise_table.csv"
OUT_TEX = REPO_ROOT / "results" / "evaluation" / "subject_casewise_table.tex"

LABEL_MAP = {0: "Control", 1: "ADHD"}


def outcome_label(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def build_casewise(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["subject"] = df["subject"].str.replace(r"\.mat$", "", regex=True)
    out["fold"] = df["fold"].astype(int)
    out["true_diagnosis"] = df["y_true"].astype(int).map(LABEL_MAP)
    out["predicted_diagnosis"] = df["pred_vote"].astype(int).map(LABEL_MAP)
    out["outcome"] = [outcome_label(int(r.y_true), int(r.pred_vote)) for r in df.itertuples()]
    out["n_trials"] = df["n_trials"].astype(int)
    out["adhd_vote_pct"] = (df["vote_pct_1"] * 100).round(1)
    return out.sort_values(["fold", "true_diagnosis", "subject"]).reset_index(drop=True)


def _tex_row(row: pd.Series) -> str:
    subject = row["subject"]
    fold = int(row["fold"])
    true_dx = row["true_diagnosis"]
    pred_dx = row["predicted_diagnosis"]
    outcome = row["outcome"]
    trials = int(row["n_trials"])
    vote = f"{row['adhd_vote_pct']:.1f}"

    correct = outcome in ("TP", "TN")
    outcome_cell = f"\\textbf{{{outcome}}}" if correct else outcome
    subject_cell = f"\\texttt{{{subject}}}"

    return f"  {subject_cell} & {fold} & {true_dx} & {pred_dx} & {outcome_cell} & {trials} & {vote} \\\\"


def build_latex(casewise: pd.DataFrame) -> str:
    caption = (
        "Subject-level classification outcomes for EEG-TACT across all five "
        "cross-validation folds ($N = 120$). Each row represents one subject. "
        "\\textbf{TP}: true positive (ADHD correctly identified); "
        "\\textbf{TN}: true negative (Control correctly identified); "
        "FP: false positive (Control misclassified as ADHD); "
        "FN: false negative (ADHD misclassified as Control). "
        "ADHD~vote~(\\%): proportion of 2-second epochs classified as ADHD."
    )

    header = (
        "  \\textbf{Subject} & \\textbf{Fold} & \\textbf{True Diagnosis} "
        "& \\textbf{Predicted Diagnosis} & \\textbf{Outcome} "
        "& \\textbf{Trials} & \\textbf{ADHD vote (\\%)} \\\\"
    )

    lines: list[str] = [
        r"\begin{longtable}{lclllrr}",
        f"  \\caption{{{caption}}} \\label{{tab:subject_casewise}} \\\\",
        r"  \toprule",
        header,
        r"  \midrule",
        r"  \endfirsthead",
        r"  \multicolumn{7}{c}{{\tablename\ \thetable{} -- continued from previous page}} \\",
        r"  \toprule",
        header,
        r"  \midrule",
        r"  \endhead",
        r"  \midrule",
        r"  \multicolumn{7}{r}{{Continued on next page}} \\",
        r"  \endfoot",
        r"  \bottomrule",
        r"  \endlastfoot",
    ]

    prev_fold = None
    for _, row in casewise.iterrows():
        fold = int(row["fold"])
        if prev_fold is not None and fold != prev_fold:
            lines.append(r"  \midrule")
        prev_fold = fold
        lines.append(_tex_row(row))

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(SUMMARY_PATH)
    casewise = build_casewise(df)

    # Verify aggregate matches expected (TN=47, FP=12, FN=3, TP=58)
    counts = casewise["outcome"].value_counts()
    assert len(casewise) == 120, f"Expected 120 rows, got {len(casewise)}"
    assert counts.get("TN", 0) == 47 and counts.get("FP", 0) == 12 \
        and counts.get("FN", 0) == 3 and counts.get("TP", 0) == 58, \
        f"Aggregate mismatch: {counts.to_dict()}"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    casewise.to_csv(OUT_CSV, index=False)

    latex = build_latex(casewise)
    OUT_TEX.write_text(latex + "\n", encoding="utf-8")

    print(latex)
    print(f"\n% Saved CSV : {OUT_CSV}")
    print(f"% Saved .tex: {OUT_TEX}")
    print(f"\n% Outcome counts: {counts.to_dict()}")


if __name__ == "__main__":
    main()
