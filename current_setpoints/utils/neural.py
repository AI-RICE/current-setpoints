import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import Any, Tuple, Optional

ANALYTICAL_BIAS_TERM: float = 0.0


def get_analytical_tensors(
    machine: Any, device: torch.device
) -> Optional[torch.Tensor]:
    """
    Helper to convert machine model matrices from a machine object to PyTorch tensors.
    Note: B_tensor is no longer fetched here because it is dynamically updated.

    Args:
        machine: Machine object containing 'mat_A' numpy matrix.
        device: The target torch device (CPU/CUDA).

    Returns:
        torch.Tensor: A_tensor if machine is provided, else None.
    """
    if machine:
        A_tensor = torch.from_numpy(machine.mat_A).float().to(device)
        return A_tensor
    return None


def torq_analytical(
    x_phys_currents: torch.Tensor, A_tensor: torch.Tensor, B_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Computes the analytical component of torque using the quadratic form T = i^T A i + 2b^T i.

    Args:
        x_phys_currents: Batch of physical current vectors. Shape: [batch, features]
        A_tensor: Quadratic machine parameter tensor. Shape: [features, features]
        B_tensor: Linear machine parameter tensor (dynamically updated). Shape: [batch, features, 1]

    Returns:
        torch.Tensor: Calculated analytical torque component. Shape: [batch, 1]
    """
    # Use einsum for the linear part to safely handle batched dot products
    torq_linear = 2 * torch.einsum("bi, bi -> b", x_phys_currents, B_tensor.squeeze(-1))

    torq_quadratic = torch.einsum(
        "bi, ij, bj -> b", x_phys_currents, A_tensor, x_phys_currents
    )
    return (torq_quadratic + torq_linear + ANALYTICAL_BIAS_TERM).unsqueeze(1)


class NeuralTorquePredictor(nn.Module):
    """
    Neural torque model (NTM) for torque prediction.
    Combines an analytical quadratic motor model with a neural residual.
    Supports dynamic flux maps by accepting B_tensor at the forward pass
    for any n-phase machine.
    """

    x_mean: torch.Tensor
    x_std: torch.Tensor
    A_TENSOR: torch.Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        scaler_X: StandardScaler,
        machine: Any,
        device: torch.device,
    ) -> None:
        """
        Initializes the predictor, registering normalization parameters and static tensors.

        Args:
            input_size: Number of input features (1 for omega + N for currents).
            hidden_size: Number of neurons in the hidden layer.
            scaler_X: Fitted Scikit-Learn StandardScaler for input normalization.
            machine: Machine object for analytical grounding.
            device: Computation device.
        """
        super(NeuralTorquePredictor, self).__init__()
        self.device = device
        self.register_buffer(
            "x_mean", torch.from_numpy(scaler_X.mean_).float().to(device)
        )
        self.register_buffer(
            "x_std", torch.from_numpy(scaler_X.scale_).float().to(device)
        )

        A_tensor = get_analytical_tensors(machine, device)
        if A_tensor is not None:
            self.register_buffer("A_TENSOR", A_tensor)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x_normed: torch.Tensor, B_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: denormalizes inputs, calculates analytical torque, and adds neural residual.

        Args:
            x_normed: Normalized input tensor [omega, i_d1, i_q1, ...].
            B_tensor: Dynamic linear machine parameter tensor for the current operating point.

        Returns:
            torch.Tensor: Total predicted torque.
        """
        x_phys = x_normed * self.x_std + self.x_mean
        x_phys_currents = x_phys[:, 1:]

        torq_analytical_out = torq_analytical(x_phys_currents, self.A_TENSOR, B_tensor)

        torq_neural_residual = self.fc1(x_normed)
        torq_neural_residual = self.gelu(torq_neural_residual)
        torq_neural_residual = self.fc2(torq_neural_residual)

        return torq_analytical_out + torq_neural_residual


def load_neural_model(
    weights_path: str,
    scaler_path: str,
    hidden_size: int,
    input_size: int,
    machine: Any,
    device: torch.device,
) -> Tuple[NeuralTorquePredictor, StandardScaler]:
    """
    Loads saved scaler data and model weights, initializing the predictor.

    Args:
        weights_path: Path to the .pth state dictionary.
        scaler_path: Path to the .npy/.npz scaler data.
        hidden_size: Hidden layer dimension used during training.
        input_size: Number of input features.
        machine: Machine object containing motor parameters.
        device: Target device.

    Returns:
        Tuple: (initialized_model, scaler_object)
    """
    scaler_data = np.load(scaler_path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]

    model = NeuralTorquePredictor(input_size, hidden_size, scaler, machine, device).to(
        device
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    return model, scaler


# Inference wrapper
def predict_torque_neural(
    vec_curr_dq: np.ndarray,
    omega: float,
    neural_model: NeuralTorquePredictor,
    scaler: StandardScaler,
    device: torch.device,
    machine: Any,
) -> float:
    """
    Evaluates the neural network for a single vector.
    Acts as a bridge between SciPy (NumPy) and PyTorch.

    Args:
        vec_curr_dq: NumPy array of currents (N-dimensional).
        omega: Electrical speed (scalar).
        neural_model: Initialized NeuralTorquePredictor.
        scaler: Fitted StandardScaler.
        device: Device for calculation.
        machine: Machine object (updated dynamically) to fetch accurate vec_b.

    Returns:
        float: Predicted electromagnetic torque.
    """
    if neural_model is None or scaler is None:
        raise ValueError("Neural model/scaler not provided to prediction function.")

    machine.update_state(omega=omega, vec_curr_dq=vec_curr_dq)

    X_input = np.hstack(([omega], vec_curr_dq))
    X_input_norm = scaler.transform(X_input.reshape(1, -1))
    X_tensor = torch.from_numpy(X_input_norm).float().to(device)

    B_tensor = (
        torch.from_numpy(machine.vec_b).float().to(device).unsqueeze(0).unsqueeze(-1)
    )

    with torch.no_grad():
        torq_predicted_tensor = neural_model(X_tensor, B_tensor)

    return torq_predicted_tensor.cpu().numpy().item()
