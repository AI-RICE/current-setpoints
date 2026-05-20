"""
Induction-motor torque model for the multiphase setpoint
optimization framework. Implements

    T(i_s) = i_s^T A(omega_r) i_s,
    A(omega_r) = (m * p_p / 2) * J * L_mu * k_ir(omega_r),

per Laksar et al. (IM_TIA draft, 2025), eqs. (torque_def, kir_matrix,
Tmatrix_final). The slip ``omega_r`` is computed internally from
``curr_dq`` via the rotor-flux-orientation rule
``omega_r = (R_r^1 / L_r^1) * i_sd^1 / i_sq^1``, so the model exposes
the same ``calculate_torque(omega, curr_dq)`` signature as the PMSM
``ModelAnalytical`` and integrates with the unchanged optimizer.
"""
from __future__ import annotations

import numpy as np

from .models import BaseTorqueModel
from ..parameters.im_machine import BaseMachineIM
from ..parameters.im_coupling import k_ir_matrix, slip_from_dq_foc


class ModelIMAnalytical(BaseTorqueModel):
    """
    Analytical IM torque model. Mirrors the PMSM ``ModelAnalytical``
    public surface so that ``MotorOptimizer``/``PointwiseOptimizer``
    consume it without modification.
    """

    def __init__(self, machine: BaseMachineIM) -> None:
        self.machine = machine
        self.n_ppairs = machine.n_ppairs
        self.mat_crossc = machine.mat_crossc
        super().__init__(machine.curr_max, machine.n_phases)
        # Override the PMSM candidate-vector dimension (which assumes
        # dim = n_phases - 1). For a 9-phase IM with 1st + 3rd harmonics
        # only, dim = 4, not 8.
        self._candidate_dim = machine.dim

    def get_candidates(self, curr_dq_guess: np.ndarray | None = None) -> list[np.ndarray]:
        """
        Multi-start candidates of dimension ``machine.dim`` (not
        ``n_phases - 1`` as in the PMSM ``BaseTorqueModel``).
        """
        dim = self._candidate_dim
        candidates: list[np.ndarray] = []
        if curr_dq_guess is not None:
            candidates.append(curr_dq_guess.copy())
        else:
            default_guess = np.zeros(dim)
            default_guess[0] = 0.1  # small d-current to seed the slip law
            default_guess[1] = 0.1
            candidates.append(default_guess)
        g_mtpa = np.zeros(dim)
        g_mtpa[0] = self.curr_max * 0.7
        g_mtpa[1] = self.curr_max * 0.7
        candidates.append(g_mtpa)
        g_fw = np.zeros(dim)
        g_fw[0] = self.curr_max * 0.5
        g_fw[1] = self.curr_max * 0.2
        candidates.append(g_fw)
        return candidates

    def _build_A(self, omega_r: float) -> np.ndarray:
        """
        A(omega_r) = (m * p_p / 2) * J * L_mu * k_ir(omega_r).

        Note: in IM_TIA the torque is written T = i_s^T A i_s, so the
        symmetric part of A determines the quadratic form. Returning
        the full (possibly asymmetric) A is fine because
        i^T A i = i^T A_sym i.
        """
        K = k_ir_matrix(omega_r, self.machine)
        return (self.machine.n_phases * self.machine.n_ppairs / 2.0) * (
            self.machine.mat_crossc @ self.machine.L_mu @ K
        )

    def calculate_torque(self, omega: float, curr_dq: np.ndarray) -> float:
        """
        Computes the steady-state electromagnetic torque at the
        given ``curr_dq`` under rotor-flux-orientation slip law.
        ``omega`` (stator electrical speed) is accepted for API
        parity but does not enter the torque expression directly --
        it only enters the voltage limit through ``IMTransform``.
        """
        omega_r = slip_from_dq_foc(curr_dq, self.machine)
        A = self._build_A(omega_r)
        return float(curr_dq @ A @ curr_dq)
