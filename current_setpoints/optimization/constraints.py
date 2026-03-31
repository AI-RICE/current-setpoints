# TODO: (DONE) finish
import numpy as np

def current_constraint(is_vec, IPM, W):
    """Ensures current magnitude does not exceed Imax."""
    current_max = np.abs(W.is_to_ia(is_vec))
    return IPM.Imax - current_max

def voltage_constraint(is_vec, IPM, W):
    """Ensures voltage magnitude does not exceed Umax."""
    u_phases = W.is_to_ua(is_vec)
    voltage_max = np.max(np.abs(u_phases))
    return IPM.Umax - voltage_max