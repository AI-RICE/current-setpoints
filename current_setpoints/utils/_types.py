import numpy as np
from typing import Protocol

class MachineProtocol(Protocol):
    """Core protocol for all machines used in the library."""
    n_phases: int
    R_stat: np.ndarray
    L_stat: np.ndarray
    flux_volt: np.ndarray
    mat_crossc: np.ndarray
    curr_max: float
    volt_max: float