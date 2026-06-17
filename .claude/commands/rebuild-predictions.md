Rebuild the trial-level prediction CSVs needed for statistical tests.

Two scripts are available:

**Generate per-subject predictions** (for all models that have saved weights):
```
source .venv/bin/activate && python scripts/generate_subject_predictions.py
```

**Rebuild EEG-TACT 5-fold trial probabilities** (if `results/trial_predictions_5fold.csv` is missing or stale):
```
source .venv/bin/activate && python scripts/rebuild_eegformer_5fold_all_trial_probs.py
```

Both require `results/weights/fold_{0..4}.weights.h5` and `results/optuna/best_hp_fold_{0..4}.json`.

After running, confirm that `results/trial_predictions_5fold.csv` exists and report its row count (should be total trial count across all 5 folds, one row per epoch).

If `$ARGUMENTS` is `subjects`, run only `generate_subject_predictions.py`. If `$ARGUMENTS` is `probs`, run only the rebuild script.
