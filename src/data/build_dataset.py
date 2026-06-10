import os
import pickle
from pathlib import Path

import numpy as np
from preprocessing import filter_eeg_data, load_subject_mat_file, make_epochs, normalize_epochs
from tqdm import tqdm

FOLDER_LABEL = {
    "ADHD": 1,
    "Control": 0,
}


def build_dataset(
    data_dir: str = "data",
    out_path: str = "data/processed/eeg_dataset.pkl",
) -> None:
    data_dir = Path(data_dir)
    os.makedirs(data_dir / "processed", exist_ok=True)
    all_epochs, all_labels, all_groups = [], [], []
    subject_id = 0
    subject_map = {}

    for folder, label in FOLDER_LABEL.items():
        for mat_file in tqdm(
            sorted((data_dir / folder).glob("*.mat")), desc=f"Processing {folder}"
        ):
            # Load data
            raw_data = load_subject_mat_file(str(mat_file))
            # Filter data
            filtered_data = filter_eeg_data(raw_data)
            # Make epochs
            epochs = make_epochs(filtered_data, epoch_length_seconds=2, overlap_seconds=1)
            # Normalize epochs
            normalized_epochs = normalize_epochs(epochs)

            # Append to dataset
            all_epochs.append(normalized_epochs)
            all_labels.append(np.full(len(normalized_epochs), label))
            all_groups.append(np.full(len(normalized_epochs), subject_id))

            subject_map[subject_id] = {
                "filename": mat_file.name,
                "label": label,
            }

            subject_id += 1

    # Full data
    data = {
        # shape: (total_epochs, num_channels, epoch_length_samples)
        "epochs": np.concatenate(all_epochs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),  # shape: (total_epochs,)
        "groups": np.concatenate(all_groups, axis=0),  # shape: (total_epochs,)
        "subject_map": subject_map,
    }

    # Save to disk
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(
        f"Dataset saved to {out_path}\n"
        f"Total epochs: {data['epochs'].shape[0]}\n"
        f"Unique subjects: {len(subject_map)}"
    )


if __name__ == "__main__":
    build_dataset()
