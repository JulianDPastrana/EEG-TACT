Organize many-seed experiment result archives and aggregate metrics.

**Step 1 — Extract zip archives:**
If there are any `.zip` files in `results/`, run:
```
source .venv/bin/activate && python scripts/organize_many_seed_experiments.py
```
This extracts each archive into `results/many_seed_experiments/<model_name>/` and deletes the zip. It also moves any legacy `results/many seeds/` folder to `results/many_seed_experiments/eegformer_5fold_run/`.

Expected subdirectory names after extraction:
`eegformer_10seeds`, `cnn_lstm_eegnet_10seeds`, `tgarnet_10seeds`, `eegnet_10seeds`, `shallowconvnet_10seeds`, `imcbgt_10seeds`, `multistream_10seeds`

**Step 2 — Aggregate metrics:**
```
source .venv/bin/activate && python scripts/explore_many_seeds.py
```
Output: `results/many_seed_summary/model_metrics.csv` — mean metrics across seeds per model.

Print the aggregated table after step 2 finishes. If `$ARGUMENTS` is `aggregate-only`, skip step 1.
