"""
current_setpoints — multiphase electric-drive setpoint optimization library.

Subpackages
-----------
models       — drive physics: DriveModel, ForwardModel, Fault
optimization — solvers, grid computation, data containers, efficiency
utils        — neural model, loss fitting, training, plotting, data loading
"""
from . import models, optimization, utils

__all__ = ["models", "optimization", "utils"]
