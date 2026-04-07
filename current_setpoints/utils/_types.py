import numpy as np
from typing import Protocol

class MachineProtocol(Protocol):
    """Core protocol for all machines used in the library."""
    n_phases: int
    n_ppairs: int
    R_stat: np.ndarray
    L_stat: np.ndarray
    flux_volt: np.ndarray
    flux_torq: np.ndarray
    vec_b: np.ndarray
    mat_crossc: np.ndarray
    curr_max: float
    volt_max: float
    omega_max: float

    def update_state(self, omega: float, vec_curr_dq: np.ndarray) -> None:
        """
        Updates the machine's internal flux vectors and dependent states 
        based on the current operating point.
        """
        ...