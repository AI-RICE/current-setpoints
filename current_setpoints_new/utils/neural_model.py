"""
Neural torque residual model and loading utilities.

Classes
-------
NeuralTorquePredictor — single-hidden-layer MLP outputting the torque residual.

Functions
---------
load_neural_model — restore a saved model + scaler from disk.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class NeuralTorquePredictor(nn.Module):
    """
    Neural torque residual model.

    Outputs only the residual to be added to the analytical torque.
    Inputs are the normalized stack [omega, i_d1, i_q1, i_d3, i_q3, ...].
    Normalization parameters are stored as registered buffers so they travel
    with the state dictionary.

    Parameters
    ----------
    input_size : int          — number of input features (1 + dim)
    hidden_size : int         — neurons in the single hidden layer
    scaler_X : StandardScaler — fitted scaler; mean_ and scale_ become buffers
    device : torch.device
    """

    x_mean: torch.Tensor
    x_std: torch.Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        scaler_X: StandardScaler,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        self.register_buffer("x_mean", torch.from_numpy(scaler_X.mean_).float().to(device))
        self.register_buffer("x_std", torch.from_numpy(scaler_X.scale_).float().to(device))

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x_normed: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x_normed))
        return self.fc2(h)


def load_neural_model(
    weights_path: str,
    scaler_path: str,
    hidden_size: int,
    input_size: int,
    device: torch.device,
) -> tuple[NeuralTorquePredictor, StandardScaler]:
    """
    Load saved scaler data and model weights, returning an initialized predictor.

    The state dictionary must match the module exactly: MLP parameters
    (fc1.*, fc2.*) and normalization buffers (x_mean, x_std).

    Parameters
    ----------
    weights_path : str       — path to the .pth state dictionary
    scaler_path  : str       — path to the .npy scaler data file
    hidden_size  : int       — hidden layer width used during training
    input_size   : int       — number of input features
    device       : torch.device

    Returns
    -------
    (NeuralTorquePredictor, StandardScaler)

    Raises
    ------
    RuntimeError if the state dictionary does not match the module.
    """
    scaler_data = np.load(scaler_path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]

    model = NeuralTorquePredictor(input_size, hidden_size, scaler, device).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


