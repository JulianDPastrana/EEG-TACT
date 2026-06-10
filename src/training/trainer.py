import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X, y, _ in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X).squeeze(1)
        loss = criterion(logits, y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """
    Returns epoch-level and subject-level (majority vote) metrics.
    """
    model.eval()
    all_logits = []
    all_labels = []
    all_groups = []

    with torch.no_grad():
        for X, y, groups in loader:
            X = X.to(device)
            logits = model(X).squeeze(1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
            all_groups.append(groups)

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels).astype(int)
    all_groups = np.concatenate(all_groups)

    # Epoch-level metrics
    epoch_preds = (all_logits > 0).astype(int)  # threshold at 0 for BCEWithLogitsLoss
    epoch_acc = accuracy_score(all_labels, epoch_preds)
    epoch_f1 = f1_score(all_labels, epoch_preds, zero_division=0)

    # Subject-level majority voting
    subject_preds = []
    subject_labels = []

    for subject_id in np.unique(all_groups):
        mask = all_groups == subject_id
        votes = epoch_preds[mask]
        true_label = all_labels[mask][0]  # all epochs from same subject share label
        majority = int(votes.sum() > len(votes) / 2)
        subject_preds.append(majority)
        subject_labels.append(true_label)

    subject_preds = np.array(subject_preds)
    subject_labels = np.array(subject_labels)
    subject_acc = accuracy_score(subject_labels, subject_preds)
    subject_f1 = f1_score(subject_labels, subject_preds, zero_division=0)
    subject_bacc = balanced_accuracy_score(subject_labels, subject_preds)  # handles imbalance

    return {
        "epoch_acc": epoch_acc,
        "epoch_f1": epoch_f1,
        "subject_acc": subject_acc,
        "subject_f1": subject_f1,
        "subject_bacc": subject_bacc,
    }
