Run all statistical tests comparing EEG-TACT against baselines.

There are two test suites — run both:

**1. McNemar exact test** (best-seed, trial/subject level):
Requires `results/trial_predictions_5fold.csv` plus equivalent files for each baseline model (looked up via `results/**/trial_predictions*.csv`).
```
source .venv/bin/activate && python statistical_tests/proportion_tests.py
```
Output: `results/stat_tests/proportions/{trial_mcnemar,subject_mcnemar,trial_accuracy,subject_accuracy}.csv`

**2. Wilcoxon signed-rank test** (10-seed regime):
Requires `results/many_seed_experiments/` populated with model fold result folders (use `/organize-results` first if you have zip archives).
```
source .venv/bin/activate && python statistical_tests/wilcoxon_test.py
```
Output: `results/stat_tests/wilcoxon_results.csv`

After both finish, summarize: which pairwise comparisons are significant (p < 0.05), and the direction of the difference (EEG-TACT vs each baseline).

If `$ARGUMENTS` specifies `mcnemar` or `wilcoxon`, run only that test.
