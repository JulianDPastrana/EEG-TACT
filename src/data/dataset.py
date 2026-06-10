import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, data: dict, indices=None):
        self.epochs = data["epochs"]  # shape: (total_epochs, num_channels, epoch_length_samples)
        self.labels = data["labels"]  # shape: (total_epochs,)
        self.groups = data["groups"]  # shape: (total_epochs,)
        self.subject_map = data.get("subject_map", {})
        if indices is not None:
            self.epochs = self.epochs[indices]
            self.labels = self.labels[indices]
            self.groups = self.groups[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        epoch = torch.tensor(
            self.epochs[idx], dtype=torch.float32
        )  # shape: (num_channels, epoch_length_samples)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # shape: ()
        group = torch.tensor(self.groups[idx], dtype=torch.long)  # shape: ()
        return epoch, label, group
