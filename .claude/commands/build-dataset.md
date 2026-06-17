Build the processed EEG dataset pickle from raw .mat files.

Steps:
1. Verify that `data/ADHD/` and `data/Control/` directories exist and contain `.mat` files. If either is missing or empty, stop and tell the user what's missing.
2. Activate the virtual environment: `source .venv/bin/activate`
3. Run: `cd src/data && python build_dataset.py`
4. Confirm that `data/processed/eeg_dataset.pkl` was created and print the number of total epochs and unique subjects reported by the script.

The script applies: bandpass Butterworth filter (0.5–60 Hz, order 4), notch filter at 50 Hz (Q=30), 2-second epochs with 1-second step (50% overlap), and per-epoch z-score normalization. Output shape is (N_epochs, 19, 256).
