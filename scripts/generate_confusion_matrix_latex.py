"""Generate LaTeX table for the subject-level confusion matrix (reviewer response)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CM_PATH = REPO_ROOT / "results" / "predictions" / "subject_confusion_matrix_5fold.csv"
FOLD_PATH = REPO_ROOT / "results" / "predictions" / "fold_metrics_5fold.csv"
OUT_TEX = REPO_ROOT / "results" / "evaluation" / "subject_confusion_matrix.tex"


def compute_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    n = tn + fp + fn + tp
    return {
        "accuracy":    (tp + tn) / n,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv":         tp / (tp + fp),
        "npv":         tn / (tn + fn),
        "f1":          2 * tp / (2 * tp + fp + fn),
        "bacc":        0.5 * (tp / (tp + fn) + tn / (tn + fp)),
    }


def build_latex(cm: pd.DataFrame, fold_df: pd.DataFrame) -> str:
    tn = int(cm.loc["true_0", "pred_0"])
    fp = int(cm.loc["true_0", "pred_1"])
    fn = int(cm.loc["true_1", "pred_0"])
    tp = int(cm.loc["true_1", "pred_1"])
    n_control = tn + fp
    n_adhd = fn + tp
    n_total = tn + fp + fn + tp

    m = compute_metrics(tn, fp, fn, tp)

    mean_acc  = fold_df["accuracy"].mean() * 100
    mean_bacc = fold_df["balanced_acc"].mean() * 100
    mean_f1   = fold_df["f1"].mean() * 100
    mean_auc  = fold_df["auc"].mean() * 100

    caption = (
        f"Subject-level confusion matrix for EEG-TACT (best-seed, 5-fold "
        f"cross-validation, $N = {n_total}$ subjects). "
        f"Rows: true diagnosis; columns: predicted diagnosis. "
        f"TN: true negative; FP: false positive; FN: false negative; TP: true positive. "
        f"Derived diagnostic metrics: "
        f"sensitivity $= {m['sensitivity']*100:.1f}\\%$; "
        f"specificity $= {m['specificity']*100:.1f}\\%$; "
        f"PPV $= {m['ppv']*100:.1f}\\%$; "
        f"NPV $= {m['npv']*100:.1f}\\%$; "
        f"F1 $= {m['f1']*100:.1f}\\%$. "
        f"Mean fold-level metrics (mean $\\pm$ std across 5 folds): "
        f"accuracy $= {mean_acc:.1f}\\%$; "
        f"balanced accuracy $= {mean_bacc:.1f}\\%$; "
        f"F1 $= {mean_f1:.1f}\\%$; "
        f"AUC $= {mean_auc:.1f}\\%$."
    )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\label{tab:subject_cm}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Predicted}} \\",
        r"\cmidrule(lr){3-4}",
        r"& & \textbf{Control} & \textbf{ADHD} \\",
        r"\midrule",
        r"\multirow{2}{*}{\textbf{Actual}}",
        f"  & \\textbf{{Control}} ($n = {n_control}$) & \\textbf{{{tn}}} (TN) & {fp} (FP) \\\\",
        f"  & \\textbf{{ADHD}}    ($n = {n_adhd}$)    & {fn} (FN)            & \\textbf{{{tp}}} (TP) \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    cm = pd.read_csv(CM_PATH, index_col=0)
    fold_df = pd.read_csv(FOLD_PATH)

    latex = build_latex(cm, fold_df)

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(latex + "\n", encoding="utf-8")

    print(latex)
    print(f"\n% Saved to: {OUT_TEX}")


if __name__ == "__main__":
    main()
