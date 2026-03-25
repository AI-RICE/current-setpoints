import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Constants needed for analytical part inside NN
ANALYTICAL_BIAS_TERM = 0.0

def get_analytical_tensors(ipm_model, device):
    """Helper to convert IPM model matrices to tensors."""
    if ipm_model:
        A_tensor = torch.from_numpy(ipm_model.A).float().to(device)
        b_tensor = torch.from_numpy(ipm_model.b).float().to(device).unsqueeze(1)
        return A_tensor, b_tensor
    return None, None

def T_analytical(x_phys_currents, A_tensor, B_tensor):
    T_linear = torch.matmul(x_phys_currents, B_tensor).squeeze()
    T_quadratic = torch.einsum(
        "bi, ij, bj -> b", x_phys_currents, A_tensor, x_phys_currents
    )
    return (T_quadratic + T_linear + ANALYTICAL_BIAS_TERM).unsqueeze(1)

class PIRN_TorquePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, scaler_X, ipm_model, device):
        super(PIRN_TorquePredictor, self).__init__()
        self.device = device
        self.register_buffer("x_mean", torch.from_numpy(scaler_X.mean_).float().to(device))
        self.register_buffer("x_std", torch.from_numpy(scaler_X.scale_).float().to(device))
        
        # Register analytical matrices as buffers with exact names 
        A_tensor, B_tensor = get_analytical_tensors(ipm_model, device)
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
        T_analytical_out = T_analytical(x_phys_currents, self.A_TENSOR, self.B_TENSOR)
        
        T_NN_residual = self.fc1(x_normed)
        
        # Apply GELU in the forward pass
        T_NN_residual = self.gelu(T_NN_residual) 
        
        T_NN_residual = self.fc2(T_NN_residual)
        
        return T_analytical_out + T_NN_residual

def load_pirn_model(weights_path, scaler_path, hidden_size, input_size, ipm_model, device):
    """
    Loads scaler and weights, initializes the model.
    Returns: (model, scaler)
    """
    scaler_data = np.load(scaler_path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]

    model = PIRN_TorquePredictor(input_size, hidden_size, scaler, ipm_model, device).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    return model, scaler