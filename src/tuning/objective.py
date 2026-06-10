import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models.eegconformer import EEGConformer
from src.training.trainer import evaluate, train_one_epoch
from src.utils.seed import set_seed


def make_objective(
    X,
    y,
    groups,
    device,
    n_inner_splits=3,
    epochs=50,
    patience=7,
    seed=42,
):

    def objective(trial: optuna.Trial) -> float:

        set_seed(seed, seed_cuda=False)

        emb_size = trial.suggest_categorical("emb_size", [16, 32, 64])
        num_channels = 19  # fixed
        temporal_kernel = trial.suggest_categorical("temporal_kernel", [16, 32, 64])
        num_filters = trial.suggest_categorical("num_filters", [16, 32, 64])
        pool_kernel = trial.suggest_categorical("pool_kernel", [4, 8, 16])
        pool_stride = trial.suggest_categorical("pool_stride", [2, 4, 8])
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        dim_feedfoward = trial.suggest_categorical("dim_feedfoward", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        num_classes = 1  # binary classification
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

        lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
        batch_size = 64  # fixed

        sgkf = StratifiedGroupKFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=seed,
        )
        scores = []

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            fold_data = {
                "epochs": X,
                "labels": y,
                "groups": groups,
            }
            train_ds = EEGDataset(fold_data, indices=train_idx)
            val_ds = EEGDataset(fold_data, indices=val_idx)

            # train_labels = y[train_idx]
            # n_pos = train_labels.sum()
            # n_neg = len(train_labels) - n_pos
            # pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], device=device)

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                # num_workers=0,
                # pin_memory=(device.type == "cuda"),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                # num_workers=0,
                # pin_memory=(device.type == "cuda"),
            )

            model = EEGConformer(
                emb_size=emb_size,
                num_channels=num_channels,
                temporal_kernel=temporal_kernel,
                num_filters=num_filters,
                pool_kernel=pool_kernel,
                pool_stride=pool_stride,
                num_heads=num_heads,
                dim_feedfoward=dim_feedfoward,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.BCEWithLogitsLoss(
                # pos_weight=pos_weight
            )
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, T_max=epochs
            # )

            best_objective, patience_count = 0.0, 0

            try:
                for epoch in range(epochs):
                    train_one_epoch(model, train_loader, optimizer, criterion, device)
                    # scheduler.step()
                    metrics = evaluate(model, val_loader, device)
                    objective_score = metrics["subject_acc"]

                    if objective_score > best_objective:
                        best_objective = objective_score
                        patience_count = 0
                    else:
                        patience_count += 1
                        if patience_count >= patience:
                            break

                    trial.report(objective_score, step=fold * epochs + epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            except RuntimeError as e:
                msg = str(e).lower()
                if "illegal memory access" in msg:
                    raise RuntimeError("CUDA illegal memory access") from e
                raise optuna.exceptions.TrialPruned() from e

            scores.append(best_objective)

        return float(np.mean(scores))

    return objective
