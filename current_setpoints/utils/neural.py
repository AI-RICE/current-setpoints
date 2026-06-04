import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class NeuralTorquePredictor(nn.Module):
    """
    Neural torque residual model.

    Outputs only the residual that should be ADDED to the analytical torque
    (the analytical part lives in ``ModelAnalytical`` / ``ModelNeural``).
    Inputs are the normalized stack [omega, i_d1, i_q1, i_d3, i_q3, ...].
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
        """
        Initializes the predictor and registers normalization parameters as buffers.

        Args:
            input_size: Number of input features (1 for omega + N for currents).
            hidden_size: Number of neurons in the hidden layer.
            scaler_X: Fitted Scikit-Learn StandardScaler for input normalization.
            device: Computation device.
        """
        super().__init__()
        self.device = device
        self.register_buffer("x_mean", torch.from_numpy(scaler_X.mean_).float().to(device))
        self.register_buffer("x_std", torch.from_numpy(scaler_X.scale_).float().to(device))

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x_normed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing the neural torque residual.

        Args:
            x_normed: Normalized input tensor [omega, i_d1, i_q1, ...].

        Returns:
            torch.Tensor: Predicted neural torque residual.
        """
        h = self.fc1(x_normed)
        h = self.act(h)
        return self.fc2(h)


def load_neural_model(
    weights_path: str,
    scaler_path: str,
    hidden_size: int,
    input_size: int,
    device: torch.device,
) -> tuple[NeuralTorquePredictor, StandardScaler]:
    """
    Loads saved scaler data and model weights, initializing the predictor.

    The state dictionary must match the current module exactly: the MLP
    parameters (``fc1.*``, ``fc2.*``) and normalization buffers
    (``x_mean``, ``x_std``). Any missing or unexpected key raises.

    Args:
        weights_path: Path to the .pth state dictionary.
        scaler_path: Path to the .npy scaler data.
        hidden_size: Hidden layer dimension used during training.
        input_size: Number of input features.
        device: Target device.

    Returns:
        Tuple: (initialized_model, scaler_object).

    Raises:
        RuntimeError: If the state dictionary does not match the module.
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


def predict_torque_neural(
    curr_dq: np.ndarray,
    omega: float,
    neural_model: NeuralTorquePredictor,
    scaler: StandardScaler,
    device: torch.device,
) -> float:
    """
    Evaluates the neural residual network for a single (omega, curr_dq) pair.

    Acts as a NumPy <-> PyTorch bridge for use inside SciPy-driven optimization.
    The returned value is just the residual; callers must add the analytical
    torque themselves (which ``ModelNeural.calculate_torque`` does).

    Args:
        curr_dq: NumPy array of currents (N-dimensional).
        omega: Electrical speed [rad/s] (scalar).
        neural_model: Initialized NeuralTorquePredictor.
        scaler: Fitted StandardScaler.
        device: Device for calculation.

    Returns:
        float: Predicted neural torque residual.
    """
    X_input = np.hstack(([omega], curr_dq))
    X_input_norm = scaler.transform(X_input.reshape(1, -1))
    X_tensor = torch.from_numpy(X_input_norm).float().to(device)

    with torch.no_grad():
        torq_residual_tensor = neural_model(X_tensor)

    return torq_residual_tensor.cpu().numpy().item()
