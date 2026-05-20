"""
Rotor-current coupling matrix ``k_ir(omega_r)`` for the multiphase
induction motor, per Laksar et al. (IM_TIA draft, 2025), eq. (kir_matrix).

The IM steady-state rotor current vector is

    i_r = k_ir(omega_r) i_s,

where ``omega_r`` is the rotor (slip) angular frequency and the
4x4 matrix ``k_ir`` is block-diagonal in the fundamental and
third-harmonic subspaces.

Used by ``IMTransform`` to build the speed-dependent voltage
operator and by ``ModelIMAnalytical`` to build the torque matrix.
"""
from __future__ import annotations

import numpy as np

from .im_machine import BaseMachineIM


def k_ir_block(omega_r: float, L_mu_h: float, L_r_h: float, R_r_h: float, h: int) -> np.ndarray:
    """
    Returns the 2x2 ``k_ir^h`` block for harmonic ``h`` (h = 1 or 3) at
    rotor (slip) angular frequency ``omega_r``.

    Eq. from IM_TIA draft (Laksar et al., 2025), simplified:

        k_ir^h = h * omega_r * L_mu^h / D
                 * [[-h*omega_r*L_r^h,    R_r^h        ],
                    [-R_r^h,             -h*omega_r*L_r^h]]

    where  D = (R_r^h)^2 + (h*omega_r*L_r^h)^2.

    For ``omega_r == 0`` the block degenerates to the zero matrix
    (no rotor current at synchronous speed).
    """
    h_omega_r = h * omega_r
    denom = R_r_h ** 2 + (h_omega_r * L_r_h) ** 2
    if denom == 0.0:
        return np.zeros((2, 2))
    scale = h_omega_r * L_mu_h / denom
    return scale * np.array(
        [
            [-h_omega_r * L_r_h, R_r_h],
            [-R_r_h, -h_omega_r * L_r_h],
        ]
    )


def k_ir_matrix(omega_r: float, machine: BaseMachineIM) -> np.ndarray:
    """
    Returns the full ``dim x dim`` block-diagonal ``k_ir`` matrix
    assembled from the per-harmonic blocks.

    Args:
        omega_r: Rotor (slip) electrical angular frequency [rad/s].
        machine: Any ``BaseMachineIM`` instance.

    Returns:
        ``np.ndarray`` of shape ``(machine.dim, machine.dim)``.
    """
    K = np.zeros((machine.dim, machine.dim))
    for i in range(machine.n_harmonics):
        h = 2 * i + 1
        L_mu_h = machine.L_mu[2 * i, 2 * i]
        L_r_h = machine.L_r[2 * i, 2 * i]
        R_r_h = machine.R_r[2 * i, 2 * i]
        K[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = k_ir_block(omega_r, L_mu_h, L_r_h, R_r_h, h)
    return K


def slip_from_dq_foc(curr_dq: np.ndarray, machine: BaseMachineIM, eps: float = 1e-9) -> float:
    """
    First-harmonic rotor-flux-orientation slip law:

        omega_r = (R_r^1 / L_r^1) * (i_sd^1 / i_sq^1)

    (Laksar et al., IM_TIA, eq. wr_opt). Defined for ``|i_sq^1| > eps``;
    returns 0 in the degenerate limit ``i_sq^1 -> 0`` (consistent with
    zero-torque MTPA operating at standstill flux).
    """
    i_sd_1 = curr_dq[0]
    i_sq_1 = curr_dq[1]
    if abs(i_sq_1) < eps:
        return 0.0
    R_r_1 = machine.R_r[0, 0]
    L_r_1 = machine.L_r[0, 0]
    return (R_r_1 / L_r_1) * (i_sd_1 / i_sq_1)
