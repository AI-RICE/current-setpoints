# TODO: (DONE) move it somewhere

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Constants needed for analytical part inside NN
ANALYTICAL_BIAS_TERM = 0.0

def get_analytical_tensors(machine, device):
    """Helper to convert IPM model matrices to tensors."""
    if machine:
        A_tensor = torch.from_numpy(machine.A).float().to(device)
        b_tensor = torch.from_numpy(machine.b).float().to(device).unsqueeze(1)
        return A_tensor, b_tensor
    return None, None

def torq_analytical(x_phys_currents, A_tensor, B_tensor):
    torq_linear = torch.matmul(x_phys_currents, B_tensor).squeeze()
    torq_quadratic = torch.einsum(
        "bi, ij, bj -> b", x_phys_currents, A_tensor, x_phys_currents
    )
    return (torq_quadratic + torq_linear + ANALYTICAL_BIAS_TERM).unsqueeze(1)

class NeuralTorquePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, scaler_X, machine, device):
        super(NeuralTorquePredictor, self).__init__()
        self.device = device
        self.register_buffer("x_mean", torch.from_numpy(scaler_X.mean_).float().to(device))
        self.register_buffer("x_std", torch.from_numpy(scaler_X.scale_).float().to(device))
        
        # Register analytical matrices as buffers with exact names 
        A_tensor, B_tensor = get_analytical_tensors(machine, device)
        if A_tensor is not None and B_tensor is not None:
            self.register_buffer('A_TENSOR', A_tensor)
            self.register_buffer('B_TENSOR', B_tensor)
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Changed activation to GELU
        self.gelu = nn.GELU() 
        
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x_normed):
        x_phys = x_normed * self.x_std + self.x_mean
        x_phys_currents = x_phys[:, 1:5]
        
        # Use the registered buffer names
        torq_analytical_out = torq_analytical(x_phys_currents, self.A_TENSOR, self.B_TENSOR)
        
        torq_neural_residual = self.fc1(x_normed)
        
        # Apply GELU in the forward pass
        torq_neural_residual = self.gelu(torq_neural_residual) 
        
        torq_neural_residual = self.fc2(torq_neural_residual)
        
        return torq_analytical_out + torq_neural_residual

def load_neural_model(weights_path, scaler_path, hidden_size, input_size, machine, device):
    """
    Loads scaler and weights, initializes the model.
    Returns: (model, scaler)
    """
    scaler_data = np.load(scaler_path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]

    model = NeuralTorquePredictor(input_size, hidden_size, scaler, machine, device).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    return model, scaler

#Inference wrapper
def predict_torque_neural(is_vec, omega, neural_model, scaler, device):
    """
    Evaluates the neural network for a single vector.
    Acts as a bridge between SciPy (NumPy) and PyTorch.
    """
    if neural_model is None or scaler is None:
        raise ValueError("Neural model/scaler not provided to prediction function.")

    X_input = np.hstack(([omega], is_vec))
    X_input_norm = scaler.transform(X_input.reshape(1, -1))
    X_tensor = torch.from_numpy(X_input_norm).float().to(device)

    with torch.no_grad():
        torq_predicted_tensor = neural_model(X_tensor)

    return torq_predicted_tensor.cpu().numpy().item() 