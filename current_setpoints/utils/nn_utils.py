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
    """
    Handles the PyTorch training and validation loop with early stopping.
    Receives dynamic B_tensors directly from the dataloader batches.

    Args:
        model: PyTorch model implementing the (X, B) -> torque interface.
        train_loader: DataLoader yielding (inputs, targets, b_tensors) tuples for training.
        val_loader: DataLoader yielding (inputs, targets, b_tensors) tuples for validation.
        criterion: Loss function (typically nn.MSELoss).
        optimizer: PyTorch optimizer (e.g., Adam).
        num_epochs: Maximum number of training epochs.
        patience: Number of epochs without improvement before early stopping.
        device: Torch computation device.
        min_delta: Minimum improvement in validation loss to count as progress.
        verbose: Currently unused (reserved for future per-epoch logging).

    Returns:
        Tuple[float, int]: (best validation loss, epoch at which training stopped).
    """
    print(f"\nStarting Training with Early Stopping (Patience={patience})...")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for inputs, targets, b_tensors in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            b_tensors = b_tensors.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(inputs, b_tensors)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for inputs, targets, b_tensors in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                b_tensors = b_tensors.to(device).unsqueeze(-1)

                outputs = model(inputs, b_tensors)
                val_loss_sum += criterion(outputs, targets).item() * inputs.size(0)

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
    """
    Evaluates the model on a test set and returns the RMSE.

    Note: this function assumes ``criterion`` is mean-reduced MSE (the default
    for ``nn.MSELoss``), as the accumulator un-does the per-batch mean by
    multiplying by batch size, then divides by total dataset size and takes
    the square root. Passing any other loss yields a meaningless number.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader yielding (inputs, targets, b_tensors) tuples.
        criterion: Loss function — must be mean-reduced MSE for the return
            value to be a true RMSE.
        device: Torch computation device.

    Returns:
        float: Root mean squared error over the test set.
    """
    model.eval()
    test_loss_sum = 0.0

    with torch.no_grad():
        for inputs, targets, b_tensors in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            b_tensors = b_tensors.to(device).unsqueeze(-1)

            outputs = model(inputs, b_tensors)
            test_loss_sum += criterion(outputs, targets).item() * inputs.size(0)

    return np.sqrt(test_loss_sum / len(cast(Sized, test_loader.dataset)))


def prepare_fold_dataloaders(
    X_train_np: np.ndarray,
    X_val_np: np.ndarray,
    y_train_np: np.ndarray,
    y_val_np: np.ndarray,
    B_train_np: np.ndarray,
    B_val_np: np.ndarray,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Builds DataLoaders for a single train/validation split (e.g., one CV fold).

    Fits the Scikit-Learn scaler strictly on the training portion to avoid
    information leak from the validation set, transforms both portions, and
    bundles inputs/targets/B-vectors into PyTorch DataLoaders.

    Note: despite the "fold" in the name, this function does NOT itself
    perform a k-fold split — the caller (e.g., a notebook iterating over
    sklearn's KFold) supplies one fold's indices at a time.

    Args:
        X_train_np: Training-fold input features.
        X_val_np: Validation-fold input features.
        y_train_np: Training-fold targets.
        y_val_np: Validation-fold targets.
        B_train_np: Training-fold B vectors (per-row analytical linear term).
        B_val_np: Validation-fold B vectors.

    Returns:
        Tuple[DataLoader, DataLoader, StandardScaler]:
            (training loader, validation loader, fitted input scaler).
    """
    scaler_X = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train_np)
    X_val_norm = scaler_X.transform(X_val_np)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_norm).float(),
        torch.from_numpy(y_train_np).float(),
        torch.from_numpy(B_train_np).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_norm).float(),
        torch.from_numpy(y_val_np).float(),
        torch.from_numpy(B_val_np).float(),
    )

    batch_size = min(64, len(X_train_norm) // 4)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler_X
