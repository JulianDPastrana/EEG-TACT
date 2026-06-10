from pathlib import Path

import numpy as np
import scipy.io
from numpy import ndarray
from scipy.signal import butter, filtfilt, iirnotch

NUM_CHANNELS = 19
SAMPLE_FREQUENCY = 128  # Hz


def load_subject_mat_file(filepath: str) -> ndarray:
    """
    Load a .mat file containing EEG data for a single subject.
    """
    mat = scipy.io.loadmat(filepath)
    subject_name = Path(filepath).stem

    raw = None
    if subject_name in mat and isinstance(mat[subject_name], np.ndarray):
        raw = mat[subject_name]
    else:
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray) and value.ndim == 2:
                raw = value
                break

    if raw is None:
        raise ValueError(f"Could not find EEG array in MAT file: {filepath}")

    if raw.shape[0] == NUM_CHANNELS:
        data = raw
    elif raw.shape[1] == NUM_CHANNELS:
        data = raw.T
    else:
        raise ValueError(
            f"Expected one dimension to be {NUM_CHANNELS} channels,\n"
            f"got shape {raw.shape} in {filepath}"
        )
    return data  # shape: (num_channels, num_samples)


def make_epochs(data: ndarray, epoch_length_seconds: int, overlap_seconds: int) -> ndarray:
    """
    Split EEG data into epochs of specified length and overlap.
    """
    epoch_length_samples = epoch_length_seconds * SAMPLE_FREQUENCY
    overlap_samples = overlap_seconds * SAMPLE_FREQUENCY
    step_size_samples = epoch_length_samples - overlap_samples
    num_samples = data.shape[1]
    epochs = []
    for start in range(0, num_samples - epoch_length_samples + 1, step_size_samples):
        end = start + epoch_length_samples
        epoch = data[:, start:end]
        epochs.append(epoch)
    return np.array(epochs)  # shape: (num_epochs, num_channels, epoch_length_samples)


def filter_eeg_data(data: ndarray) -> ndarray:
    """
    Apply a bandpass butterworth filter of 0.5-60 Hz and a notch filter at 50 Hz to the EEG data.
    """
    # Bandpass filter
    b, a = butter(
        N=4,
        Wn=[0.5, 60.0],
        btype="bandpass",
        analog=False,
        fs=SAMPLE_FREQUENCY,
        output="ba",
    )
    filtered_data = filtfilt(b, a, data)

    # Notch filter at 50 Hz
    b_notch, a_notch = iirnotch(
        w0=50.0,
        Q=30.0,
        fs=SAMPLE_FREQUENCY,
    )
    filtered_data = filtfilt(b_notch, a_notch, filtered_data)

    return filtered_data


def normalize_epochs(epochs: ndarray) -> ndarray:
    """
    Normalize each epoch to have zero mean and unit variance.
    """
    mean = np.mean(epochs, axis=(1, 2), keepdims=True)
    std = np.std(epochs, axis=(1, 2), keepdims=True)
    normalized_epochs = (epochs - mean) / (std + 1e-8)
    return normalized_epochs


if __name__ == "__main__":
    data = load_subject_mat_file("data/ADHD/v1p.mat")
    eeg = make_epochs(data, epoch_length_seconds=2, overlap_seconds=1)
    eeg = normalize_epochs(eeg)
    print(eeg.shape)  # should be (num_epochs, num_channels, epoch_length_samples)
