from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class NeuralTorquePredictor(nn.Module):

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
    scaler_data = np.load(scaler_path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]

    model = NeuralTorquePredictor(input_size, hidden_size, scaler, device).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


class NeuralFluxPredictor(nn.Module):

    x_mean: torch.Tensor
    x_std: torch.Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        scaler_X: StandardScaler,
        device: torch.device,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.device = device
        self.activation = activation
        self.register_buffer("x_mean", torch.from_numpy(scaler_X.mean_).float().to(device))
        self.register_buffer("x_std", torch.from_numpy(scaler_X.scale_).float().to(device))

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = ACTIVATIONS[activation]()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x_normed: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x_normed))
        return self.fc2(h)


def load_neural_flux_model(
    weights_path: str,
    scaler_path: str,
    hidden_size: int,
    input_size: int,
    output_size: int,
    device: torch.device,
    activation: str = "gelu",
) -> tuple[NeuralFluxPredictor, StandardScaler]:
    scaler_data = np.load(scaler_path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]

    model = NeuralFluxPredictor(
        input_size, hidden_size, output_size, scaler, device, activation
    ).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


