from __future__ import annotations

import copy
from collections.abc import Sized
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    patience: int,
    device: torch.device,
    min_delta: float = 0.0,
    verbose: bool = False,
) -> tuple[float, int]:
    print(f"\nStarting Training with Early Stopping (Patience={patience})...")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_loss_sum += criterion(model(inputs), targets).item() * inputs.size(0)

        avg_val_loss = val_loss_sum / len(cast(Sized, val_loader.dataset))

        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
                model.load_state_dict(best_weights)
                return best_val_loss, epoch + 1

    model.load_state_dict(best_weights)
    return best_val_loss, num_epochs


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    test_loss_sum = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            test_loss_sum += criterion(model(inputs), targets).item() * inputs.size(0)
    return float(np.sqrt(test_loss_sum / len(cast(Sized, test_loader.dataset))))


def prepare_fold_dataloaders(
    X_train_np: np.ndarray,
    X_val_np: np.ndarray,
    y_train_np: np.ndarray,
    y_val_np: np.ndarray,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    scaler_X = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train_np)
    X_val_norm = scaler_X.transform(X_val_np)

    batch_size = min(64, len(X_train_norm) // 4)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_norm).float(),
            torch.from_numpy(y_train_np).float(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val_norm).float(),
            torch.from_numpy(y_val_np).float(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, scaler_X
