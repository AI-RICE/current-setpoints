import numpy as np
from scipy.optimize import minimize
import torch

# ==========================================
# 1. Constraints & Helpers
# ==========================================

def current_constraint(is_vec, IPM, W):
    """Ensures current magnitude does not exceed Imax."""
    current_max = np.abs(W.is_to_ia(is_vec))
    return IPM.Imax - current_max

def voltage_constraint(is_vec, IPM, W):
    """Ensures voltage magnitude does not exceed Umax."""
    u_phases = W.is_to_ua(is_vec)
    voltage_max = np.max(np.abs(u_phases))
    return IPM.Umax - voltage_max

def predict_torque_pirn(is_vec, omega, pirn_model, scaler, device):
    """Evaluates the neural network for a single vector."""
    if pirn_model is None or scaler is None:
        raise ValueError("PIRN model/scaler not provided to prediction function.")

    X_input = np.hstack(([omega], is_vec))
    X_input_norm = scaler.transform(X_input.reshape(1, -1))
    X_tensor = torch.from_numpy(X_input_norm).float().to(device)

    with torch.no_grad():
        T_predicted_tensor = pirn_model(X_tensor)

    return T_predicted_tensor.cpu().numpy().item()

# ==========================================
# 2. Analytical Optimization (Physics Model)
# ==========================================

def reg_maxTorque(IPM, W, is0=None, opts=None):
    """
    Finds the MAXIMUM torque possible at a given speed (Analytical).
    Uses Multi-Start to ensure global maximum is found.
    """
    if opts is None: opts = {"disp": False, "ftol": 1e-8, "maxiter": 500}

    def objective(is_vec):
        # Minimize negative torque -> Maximize positive torque
        return -(is_vec @ IPM.A @ is_vec + 2 * IPM.b @ is_vec)

    constraints = [
        {"type": "ineq", "fun": current_constraint, "args": (IPM, W)},
        {"type": "ineq", "fun": voltage_constraint, "args": (IPM, W)},
    ]

    # --- Multi-Start Candidates ---
    candidates = []
    # 1. Warm Start
    if is0 is not None: candidates.append(is0)
    # 2. MTPA Guess (High Q-axis)
    g_mtpa = np.zeros(4); g_mtpa[1] = IPM.Imax * 0.95
    candidates.append(g_mtpa)
    # 3. Flux Weakening Guess (High Negative D-axis)
    g_fw = np.zeros(4); g_fw[0] = -IPM.Imax * 0.9; g_fw[1] = IPM.Imax * 0.1
    candidates.append(g_fw)

    best_res = None
    best_val = float('inf') # Minimizing negative torque

    for start_vec in candidates:
        try:
            res = minimize(objective, start_vec, method="SLSQP", constraints=constraints, options=opts)
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_res = res
        except: continue

    if best_res is not None:
        return best_res.x, -best_res.fun, True

    # Fallback
    res = minimize(objective, g_mtpa, method="SLSQP", constraints=constraints, options=opts)
    return res.x, -res.fun, res.success


def reg_defTorque(IPM, tor, W, is0=None, opts=None):
    """
    Finds the Minimum Current vector for a TARGET torque (Analytical).
    Uses Multi-Start to eliminate 'snow' noise.
    """
    if opts is None: opts = {"disp": False, "ftol": 1e-9, "maxiter": 500}
    
    def objective(is_vec):
        return np.sum(is_vec**2)

    def mycon_eq(is_vec):
        torque_achieved = is_vec @ IPM.A @ is_vec + 2 * IPM.b @ is_vec
        return torque_achieved - tor

    constraints = [
        {"type": "eq", "fun": mycon_eq},
        {"type": "ineq", "fun": current_constraint, "args": (IPM, W)},
        {"type": "ineq", "fun": voltage_constraint, "args": (IPM, W)},
    ]

    # --- Multi-Start Candidates ---
    candidates = []
    # 1. Warm Start
    if is0 is not None: candidates.append(is0)
    else: candidates.append(np.array([1.0, 0.0, 0.0, 0.0]))
    # 2. MTPA Guess
    g_mtpa = np.zeros(4); g_mtpa[1] = IPM.Imax * 0.9
    candidates.append(g_mtpa)
    # 3. Flux Weakening Guess
    g_fw = np.zeros(4); g_fw[0] = -IPM.Imax * 0.8; g_fw[1] = IPM.Imax * 0.2
    candidates.append(g_fw)

    best_res = None
    best_cost = float('inf')

    for start_vec in candidates:
        try:
            res = minimize(objective, start_vec, method="SLSQP", constraints=constraints, options=opts)
            if res.success and res.fun < best_cost:
                best_cost = res.fun
                best_res = res
        except: continue

    if best_res is not None:
        return best_res.x, True
    
    # Fallback
    fallback = is0 if is0 is not None else candidates[1]
    res = minimize(objective, fallback, method="SLSQP", constraints=constraints, options=opts)
    return res.x, res.success

# ==========================================
# 3. Neural Network Optimization (PIRN Model)
# ==========================================

def reg_maxTorque_PIRN_Compensated(IPM, W, pirn_model, scaler, device, omega=0.0, is0=None, opts=None):
    """
    Finds the MAXIMUM torque possible at a given speed (PIRN Model).
    """
    if opts is None: opts = {"disp": False, "ftol": 1e-8, "maxiter": 500}

    def objective(is_vec):
        return -predict_torque_pirn(is_vec, omega, pirn_model, scaler, device)

    constraints = [
        {"type": "ineq", "fun": current_constraint, "args": (IPM, W)},
        {"type": "ineq", "fun": voltage_constraint, "args": (IPM, W)},
    ]

    # --- Multi-Start Candidates ---
    candidates = []
    if is0 is not None: candidates.append(is0)
    
    g_mtpa = np.zeros(4); g_mtpa[1] = IPM.Imax
    candidates.append(g_mtpa)
    
    g_fw = np.zeros(4); g_fw[0] = -IPM.Imax * 0.8; g_fw[1] = IPM.Imax * 0.2
    candidates.append(g_fw)

    best_res = None
    best_val = float('inf') 

    for start_vec in candidates:
        try:
            res = minimize(objective, start_vec, method="SLSQP", constraints=constraints, options=opts)
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_res = res
        except: continue

    if best_res is not None:
        return best_res.x, -best_res.fun, True

    # Fallback
    res = minimize(objective, g_mtpa, method="SLSQP", constraints=constraints, options=opts)
    return res.x, -res.fun, res.success


def reg_defTorque_PIRN_Compensated(IPM, tor, W, pirn_model, scaler, device, omega, is0=None, opts=None):
    """
    Finds the Minimum Current vector for a TARGET torque (PIRN Model).
    """
    if opts is None: opts = {"disp": False, "ftol": 1e-9, "maxiter": 500}

    def objective(is_vec):
        return np.sum(is_vec**2)

    def eq_cons(is_vec):
        return predict_torque_pirn(is_vec, omega, pirn_model, scaler, device) - tor

    constraints = [
        {"type": "eq", "fun": eq_cons},
        {"type": "ineq", "fun": current_constraint, "args": (IPM, W)},
        {"type": "ineq", "fun": voltage_constraint, "args": (IPM, W)},
    ]

    # --- Multi-Start Candidates ---
    candidates = []
    if is0 is not None: candidates.append(is0)
    else: candidates.append(np.array([0.0, 1.0, 0.0, 0.0]))

    g_mtpa = np.zeros(4); g_mtpa[1] = IPM.Imax * 0.9
    candidates.append(g_mtpa)

    g_fw = np.zeros(4); g_fw[0] = -IPM.Imax * 0.8; g_fw[1] = IPM.Imax * 0.2
    candidates.append(g_fw)

    best_res = None
    best_cost = float('inf')

    for start_vec in candidates:
        try:
            res = minimize(objective, start_vec, method="SLSQP", constraints=constraints, options=opts)
            if res.success and res.fun < best_cost:
                best_cost = res.fun
                best_res = res
        except: continue

    if best_res is not None:
        return best_res.x, True
    
    # Fallback
    fallback = is0 if is0 is not None else candidates[1]
    res = minimize(objective, fallback, method="SLSQP", constraints=constraints, options=opts)
    return res.x, res.success