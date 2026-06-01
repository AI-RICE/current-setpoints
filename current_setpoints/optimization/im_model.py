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

        Seven seeds: warm start (if given), small balanced
        ``(d, q)``, three magnitudes of balanced MTPA along
        ``d = q``, an FW seed with d > q, and a Type-II seed that
        includes a small third-harmonic kick. The redundancy is
        cheap (SLSQP returns quickly from each) and avoids the
        slip-law degeneracy when ``i_sq^1 -> 0``.
        """
        dim = self._candidate_dim
        candidates: list[np.ndarray] = []
        if curr_dq_guess is not None:
            candidates.append(curr_dq_guess.copy())
        I = self.curr_max

        # Small balanced seed: keeps slip law well-defined, never
        # touches any constraint.
        c = np.zeros(dim); c[0] = 0.1; c[1] = 0.1
        candidates.append(c)
        # MTPA seeds at three balanced magnitudes along d = q.
        for amp in (0.3, 0.5, 0.7):
            c = np.zeros(dim); c[0] = I * amp; c[1] = I * amp
            candidates.append(c)
        # FW seed: more d (rotor magnetisation) than q (torque current).
        c = np.zeros(dim); c[0] = I * 0.85; c[1] = I * 0.30
        candidates.append(c)
        # Type-II seed with small third-harmonic kick.
        if dim >= 4:
            c = np.zeros(dim)
            c[0] = I * 0.7; c[1] = I * 0.7
            c[2] = I * 0.1; c[3] = -I * 0.1
            candidates.append(c)
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
