"""
Dense-grid voltage-residual diagnostic for the active-set solver.

``current_setpoints.optimization.optimizer.ActiveSetOptimizer`` already
implements the successive-relaxation active-set algorithm itself (ported
from this module during the library unification; see ``Dynamic.tex``
Section 4). What's left here is a diagnostic with no library equivalent:
checking the worst-case voltage residual on the *dense* plotting grid,
which can exceed the coarse SLSQP grid's residual because the static
voltage contribution drifts linearly within each segment while the
inductive (forward-Euler) term is constant.
"""

from __future__ import annotations

import numpy as np

from current_setpoints.models.forward_model import ForwardModel

from ._waveforms import dynamic_phase_waveforms


def voltage_residuals_dense(
    fwd: ForwardModel,
    omega: float,
    curr_dq_grid: np.ndarray,
    volt_max: float,
) -> tuple[float, np.ndarray]:
    """
    Worst-case dynamic voltage residual on the dense angle grid (``fwd.vec_theta``,
    matching the plotting resolution). Returns ``(max_residual, v_ph_dense)``:
    scalar worst-phase residual over all dense angles, and the dense
    per-phase voltage array of shape ``(n_phases, n_theta + 1)``.
    """
    n_grid = curr_dq_grid.shape[0]
    theta_grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    _, v_ph_dense = dynamic_phase_waveforms(fwd, omega, theta_grid, curr_dq_grid)
    return float(np.max(np.abs(v_ph_dense)) - volt_max), v_ph_dense


__all__ = ["voltage_residuals_dense"]
