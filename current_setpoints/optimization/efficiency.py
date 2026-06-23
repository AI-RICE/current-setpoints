"""
Copper-loss efficiency post-processor for multiphase IM regime-aware
setpoint grids. Reads a regular ``(T*, omega)`` grid produced by
``calculate_grid_im`` and adds per-cell stator-copper, rotor-copper,
and copper-loss efficiency arrays.

No iron loss, windage, or inverter loss is modeled here -- these
require additional measured or fitted parameters and are deliberately
out of scope. The fields added are an *upper bound* on the true
drive-train efficiency; the relative comparison between two grids
(e.g. proposed vs. ``i_3 = 0``) is unaffected by the missing terms
to the extent that those terms are controller-agnostic.

Definitions (motoring sign convention; absolute value of mechanical
power is used so the same formula reads sensibly under regen):

    P_stator = (m / 2) * R_s * ||i_s||^2
    omega_r  = (R_r^1 / L_r^1) * (i_sd^1 / i_sq^1)             FOC slip
    i_r      = k_ir(omega_r) * i_s
    P_rotor  = (m / 2) * i_r^T * R_r * i_r
    P_mech   = T * (omega_elec / p_p)
    eta_cu   = |P_mech| / (|P_mech| + P_stator + P_rotor)

For amplitude-invariant Clarke, ``(m/2) * ||i_dq||^2`` is the
mean-over-theta of ``sum_k i_phase_k^2`` (Parseval); the same
identity is used by ``drive_cycle.evaluate_drive_cycle``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..parameters.im_coupling import k_ir_matrix, slip_from_dq_foc
from ..parameters.im_machine import BaseMachineIM


def add_efficiency_map(grid: dict[str, Any], machine: BaseMachineIM) -> dict[str, Any]:
    """
    Augment ``grid`` in place with per-cell stator/rotor copper-loss and
    copper-loss efficiency arrays. Returns the same dict for chaining.

    Args:
        grid: Output of ``calculate_grid_im``. Must contain
            ``vec_torq`` (n_torq,), ``vec_omega`` (n_omega, electrical
            rad/s), and ``curr_dq_grid`` (dim, n_torq, n_omega).
        machine: ``BaseMachineIM`` instance providing ``R_s_scalar``,
            ``R_r``, ``n_phases``, ``n_ppairs``, and the coupling matrices
            needed for ``k_ir`` and the slip law.

    Returns:
        ``grid`` with three new keys:
          * ``grid_loss_stator`` (n_torq, n_omega) -- stator copper [W]
          * ``grid_loss_rotor``  (n_torq, n_omega) -- rotor copper [W]
          * ``grid_eta_cu``      (n_torq, n_omega) -- copper-only efficiency

        Cells that are infeasible in ``curr_dq_grid`` (NaN) propagate
        NaN to all three new arrays.
    """
    T_ax = np.asarray(grid["vec_torq"], dtype=float)
    om_ax = np.asarray(grid["vec_omega"], dtype=float)
    curr = np.asarray(grid["curr_dq_grid"], dtype=float)

    m = machine.n_phases
    R_s = float(machine.R_s_scalar)
    p_p = machine.n_ppairs

    n_t, n_o = len(T_ax), len(om_ax)
    P_s = np.full((n_t, n_o), np.nan)
    P_r = np.full((n_t, n_o), np.nan)
    eta = np.full((n_t, n_o), np.nan)

    for j in range(n_o):
        om_mech = om_ax[j] / p_p
        for i in range(n_t):
            i_s = curr[:, i, j]
            if np.any(np.isnan(i_s)):
                continue
            P_s_ij = 0.5 * m * R_s * float(i_s @ i_s)
            om_r = slip_from_dq_foc(i_s, machine)
            i_r = k_ir_matrix(om_r, machine) @ i_s
            P_r_ij = 0.5 * m * float(i_r @ machine.R_r @ i_r)
            P_mech = abs(T_ax[i] * om_mech)
            P_in = P_mech + P_s_ij + P_r_ij
            P_s[i, j] = P_s_ij
            P_r[i, j] = P_r_ij
            eta[i, j] = P_mech / P_in if P_in > 1e-9 else 0.0

    grid["grid_loss_stator"] = P_s
    grid["grid_loss_rotor"] = P_r
    grid["grid_eta_cu"] = eta
    return grid
