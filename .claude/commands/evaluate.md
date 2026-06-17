Evaluate the trained EEG-TACT model across all 5 cross-validation folds.

Before running, check that these paths exist and report which are missing:
- `results/optuna/best_hp_fold_{0..4}.json` — per-fold best hyperparameters
- `results/weights/fold_{0..4}.weights.h5` — trained Keras weights
- `results/trial_predictions_5fold.csv` — fold-subject split file
- `results/subject_trial_stats_5fold.csv`
- `data/ADHD/` and `data/Control/` with `.mat` files

If all are present, run from the repo root:
```
source .venv/bin/activate && python scripts/evaluate_eegformer.py
```

Output CSVs are saved to `results/evaluation/`:
- `fold_metrics.csv` — per-fold accuracy, balanced_acc, F1, AUC
- `summary_statistics.csv` — mean/variance/std across folds
- `trial_confusion_matrix.csv` — aggregate confusion matrix

Print the per-fold table and the summary after the script finishes.
