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
    """MLP predicting a scalar co-energy residual, with an arbitrary number
    of hidden layers (`hidden_sizes`, e.g. [12] for one layer, [24, 12] for
    two). `machines.py`'s `NeuralFlux` reads `self.layers` (the hidden stack)
    and `self.out_layer` (the final linear layer) directly to compute its
    own exact analytic gradient/Hessian -- see NeuralFlux._forward_grad_hess
    for the depth-general recursion this layout is designed for."""

    x_mean: torch.Tensor
    x_std: torch.Tensor

    def __init__(
        self,
        input_size: int,
        hidden_sizes: int | list[int],
        output_size: int,
        scaler_X: StandardScaler,
        device: torch.device,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.device = device
        self.activation = activation
        self.hidden_sizes = list(hidden_sizes)
        self.register_buffer("x_mean", torch.from_numpy(scaler_X.mean_).float().to(device))
        self.register_buffer("x_std", torch.from_numpy(scaler_X.scale_).float().to(device))

        dims = [input_size] + self.hidden_sizes
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.act = ACTIVATIONS[activation]()
        self.out_layer = nn.Linear(dims[-1], output_size)

    def forward(self, x_normed: torch.Tensor) -> torch.Tensor:
        h = x_normed
        for layer in self.layers:
            h = self.act(layer(h))
        return self.out_layer(h)


def load_neural_flux_model(
    weights_path: str,
    scaler_path: str,
    hidden_sizes: int | list[int],
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
        input_size, hidden_sizes, output_size, scaler, device, activation
    ).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


