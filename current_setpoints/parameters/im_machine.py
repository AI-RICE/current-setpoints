"""
Induction-machine parameter classes for the multiphase setpoint
optimization framework. Mirrors the ``BaseMachine`` / ``IEEEMachine2``
pattern in ``machines.py`` but holds equivalent-circuit parameters
suited to an induction motor (no permanent-magnet flux; rotor
parameters required for the rotor-current coupling matrix
``k_ir(omega_r)``).
"""
from __future__ import annotations

import numpy as np


class BaseMachineIM:
    """
    Abstract foundation for multiphase induction-motor models.

    Required attributes (must be set on ``self`` before calling
    ``super().__init__()``):
        n_phases       physical phase count
        n_ppairs       pole pairs
        n_harmonics    number of harmonic dq subspaces in the optimization
                       state (1 = fundamental only, 2 = first + third, ...)
        R_s_scalar     stator phase resistance [Ohm]
        L_mu           magnetising inductance, diag (dim x dim) [H]
        L_s_sigma      stator leakage, diag (dim x dim) [H]
        L_r_sigma      rotor leakage, diag (dim x dim) [H]
        R_r            rotor resistance, diag (dim x dim) [Ohm]

    Operating limits ``curr_max``, ``volt_max``, ``omega_max`` are
    declared at class level and must be set via ``set_max_pars`` before
    any consumer reads them.
    """

    n_phases: int
    n_ppairs: int
    n_harmonics: int
    R_s_scalar: float
    L_mu: np.ndarray
    L_s_sigma: np.ndarray
    L_r_sigma: np.ndarray
    R_r: np.ndarray

    curr_max: float
    volt_max: float
    omega_max: float

    def __init__(self) -> None:
        self.dim: int = 2 * self.n_harmonics
        self.k_phase: float = self.n_phases / 2

        # mat_crossc: block-diagonal J with J^h on the h-th harmonic block,
        # h in {1, 3, 5, ...}. Identical pattern to the PMSM BaseMachine.
        self.mat_crossc: np.ndarray = np.zeros((self.dim, self.dim))
        for i in range(self.n_harmonics):
            h = 2 * i + 1
            self.mat_crossc[2 * i, 2 * i + 1] = -h
            self.mat_crossc[2 * i + 1, 2 * i] = h

        # Composite inductances appearing throughout the IM model.
        self.L_s: np.ndarray = self.L_mu + self.L_s_sigma
        self.L_r: np.ndarray = self.L_mu + self.L_r_sigma

        # PMSM-style stator-frame matrices, kept for API compatibility with
        # consumers that access ``R_stat`` and ``L_stat`` (e.g. for warm
        # starts or sanity checks). The actual IM voltage equation in
        # ``IMTransform`` overrides these to include the rotor coupling.
        self.R_stat: np.ndarray = self.R_s_scalar * np.eye(self.dim)
        self.L_stat: np.ndarray = self.L_s.copy()

        self._check_shapes()

    def set_max_pars(self, curr_max: float, volt_max: float, omega_max: float) -> None:
        self.curr_max = curr_max
        self.volt_max = volt_max
        self.omega_max = omega_max

    def _check_shapes(self) -> None:
        for name in ("L_mu", "L_s_sigma", "L_r_sigma", "R_r"):
            mat = getattr(self, name)
            if mat.shape != (self.dim, self.dim):
                raise ValueError(
                    f"{name} must be ({self.dim}, {self.dim}); got {mat.shape}"
                )


class IM9Phase(BaseMachineIM):
    """
    9-phase induction machine with 1st + 3rd harmonic dq state.

    Parameters from the 15 kW laboratory prototype reported in
    Laksar et al., IM_TIA draft (2025), Table I:

        Number of phases m                              9
        Number of pole pairs p_p                        2
        Stator resistance R_s [Ohm]                     5.0
        Rotor resistance R_r           1st  / 3rd       1.54 / 1.57
        Magnetising inductance L_mu    1st  / 3rd [mH]  496  / 58.2
        Stator leakage L_s_sigma       1st  / 3rd [mH]  15.1 / 13.6
        Rotor leakage  L_r_sigma       1st  / 3rd [mH]  53.3 / 33.4

    Rated/limit values used in the paper: I_max = 3 A, V_max = 200 V,
    N_max = 1500 r/min. These are *not* set by default; call
    ``set_max_pars`` before optimization.
    """

    def __init__(self) -> None:
        self.n_phases = 9
        self.n_ppairs = 2
        self.n_harmonics = 2  # fundamental + third harmonic

        self.R_s_scalar = 5.0
        # Diagonal (4x4) matrices indexed as
        # [d^1, q^1, d^3, q^3] -- per-harmonic scalar repeated on the dq pair.
        self.R_r = np.diag([1.54, 1.54, 1.57, 1.57])
        self.L_mu = 1e-3 * np.diag([496.0, 496.0, 58.2, 58.2])
        self.L_s_sigma = 1e-3 * np.diag([15.1, 15.1, 13.6, 13.6])
        self.L_r_sigma = 1e-3 * np.diag([53.3, 53.3, 33.4, 33.4])

        super().__init__()
