from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .machines import DriveModel, IM9Phase, _build_cross_coupling


@dataclass
class IMHarmonicParams:
    """Electrical parameters of one odd-harmonic plane (h = 1, 3, ...)."""

    R_r: float
    L_mu: float  # rated (unsaturated) magnetizing inductance [H]
    L_s_sigma: float
    L_r_sigma: float
    # optional magnetizing-saturation lookup table: L_mu as a function of the
    # plane's magnetizing current; constant L_mu when absent
    i_mag_table: np.ndarray | None = None  # strictly increasing [A]
    L_mu_table: np.ndarray | None = None  # matching L_mu values [H]

    def __post_init__(self) -> None:
        if (self.i_mag_table is None) != (self.L_mu_table is None):
            raise ValueError("i_mag_table and L_mu_table must be given together.")
        if self.i_mag_table is not None:
            self.i_mag_table = np.asarray(self.i_mag_table, dtype=float)
            self.L_mu_table = np.asarray(self.L_mu_table, dtype=float)
            if self.i_mag_table.ndim != 1 or self.i_mag_table.shape != self.L_mu_table.shape:
                raise ValueError("Saturation tables must be 1-D and of equal length.")
            if np.any(np.diff(self.i_mag_table) <= 0):
                raise ValueError("i_mag_table must be strictly increasing.")

    def L_mu_at(self, i_mag: float) -> float:
        if self.i_mag_table is None:
            return self.L_mu
        return float(np.interp(abs(i_mag), self.i_mag_table, self.L_mu_table))


@dataclass
class IMLUTParams:
    """Complete machine description: identity, electrical data per harmonic
    plane, and operating limits. One instance == one physical machine."""

    n_phases: int
    n_ppairs: int
    R_s: float
    harmonics: list[IMHarmonicParams]  # index i <-> harmonic h = 2 i + 1
    curr_max: float
    volt_max: float
    omega_max: float
    k_v: float = 0.0
    k_h: float = 0.0


class IMDriveLUT(DriveModel):
    """Induction drive whose machine data arrive entirely through `params` —
    the class holds physics only; a concrete machine is an instance.

    Saturation: each plane's L_mu is read from its lookup table at the
    plane's d-axis (magnetizing) current under rotor-flux orientation;
    without a table the plane is linear and reproduces `IM9Phase` exactly.
    """

    def __init__(self, params: IMLUTParams) -> None:
        self.params = params
        self.n_phases = params.n_phases
        self.n_harmonics = len(params.harmonics)
        self.dim = 2 * self.n_harmonics
        self.n_ppairs = params.n_ppairs
        self.k_phase = params.n_phases / 2

        self.R_stat = params.R_s * np.eye(self.dim)
        self.k_v = params.k_v
        self.k_h = params.k_h
        self.set_max_pars(params.curr_max, params.volt_max, params.omega_max)

        self._cross_coupling = _build_cross_coupling(self.n_harmonics)

    def _plane_L_mu(self, i: int, curr_dq: np.ndarray | None) -> float:
        i_mag = 0.0 if curr_dq is None else float(curr_dq[2 * i])
        return self.params.harmonics[i].L_mu_at(i_mag)

    def _L_matrices(self, curr_dq: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        L_mu = np.zeros((self.dim, self.dim))
        L_s = np.zeros((self.dim, self.dim))
        L_r = np.zeros((self.dim, self.dim))
        R_r = np.zeros((self.dim, self.dim))
        eye2 = np.eye(2)
        for i, p in enumerate(self.params.harmonics):
            L_mu_h = self._plane_L_mu(i, curr_dq)
            sl = slice(2 * i, 2 * i + 2)
            L_mu[sl, sl] = L_mu_h * eye2
            L_s[sl, sl] = (L_mu_h + p.L_s_sigma) * eye2
            L_r[sl, sl] = (L_mu_h + p.L_r_sigma) * eye2
            R_r[sl, sl] = p.R_r * eye2
        return L_mu, L_s, L_r, R_r

    def k_ir(self, omega_r: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        K = np.zeros((self.dim, self.dim))
        for i, p in enumerate(self.params.harmonics):
            h = 2 * i + 1
            L_mu_h = self._plane_L_mu(i, curr_dq)
            K[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = IM9Phase._k_ir_block(
                omega_r, L_mu_h, L_mu_h + p.L_r_sigma, p.R_r, h
            )
        return K

    def slip_from_dq_foc(self, curr_dq: np.ndarray, eps: float = 1e-9) -> float:
        i_sd_1 = curr_dq[0]
        i_sq_1 = curr_dq[1]
        if abs(i_sq_1) < eps:
            return 0.0
        p1 = self.params.harmonics[0]
        L_r_1 = self._plane_L_mu(0, curr_dq) + p1.L_r_sigma
        return (p1.R_r / L_r_1) * (i_sd_1 / i_sq_1)

    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        return self.R_stat + omega * self._cross_coupling @ self.inductance(omega, curr_dq)

    def bemf_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        return np.zeros(self.dim)

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        L_mu, L_s, _, _ = self._L_matrices(curr_dq)
        if curr_dq is None:
            return L_s
        omega_r = self.slip_from_dq_foc(curr_dq)
        return L_s + L_mu @ self.k_ir(omega_r, curr_dq)

    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        omega_r = self.slip_from_dq_foc(curr_dq)
        L_mu, _, _, _ = self._L_matrices(curr_dq)
        A = (self.n_phases * self.n_ppairs / 2.0) * (self._cross_coupling @ L_mu @ self.k_ir(omega_r, curr_dq))
        return float(curr_dq @ A @ curr_dq)

    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        dim = self.dim
        i_max = self.curr_max
        candidates: list[np.ndarray] = []
        if guess is not None:
            candidates.append(guess.copy())
        c = np.zeros(dim)
        c[0] = 0.1
        c[1] = 0.1
        candidates.append(c)
        for amp in (0.3, 0.5, 0.7):
            c = np.zeros(dim)
            c[0] = i_max * amp
            c[1] = i_max * amp
            candidates.append(c)
        c = np.zeros(dim)
        c[0] = i_max * 0.85
        c[1] = i_max * 0.30
        candidates.append(c)
        if dim >= 4:
            c = np.zeros(dim)
            c[0] = i_max * 0.7
            c[1] = i_max * 0.7
            c[2] = i_max * 0.1
            c[3] = -i_max * 0.1
            candidates.append(c)
        return candidates

    def copper_loss(self, omega: float, curr_dq: np.ndarray) -> float:
        P_stator = self.k_phase * self.params.R_s * float(curr_dq @ curr_dq)
        omega_r = self.slip_from_dq_foc(curr_dq)
        _, _, _, R_r = self._L_matrices(curr_dq)
        i_r = self.k_ir(omega_r, curr_dq) @ curr_dq
        P_rotor = self.k_phase * float(i_r @ R_r @ i_r)
        return P_stator + P_rotor


def im9_prototype(
    curr_max: float = 20.0,
    volt_max: float = 200.0,
    omega_max: float = 1500.0,
) -> IMDriveLUT:
    """The 9-phase induction prototype, as an instance: identical numbers to
    `IM9Phase.__init__` (identified on the lab machine); limits default to
    the values used across the test suite."""
    params = IMLUTParams(
        n_phases=9,
        n_ppairs=2,
        R_s=5.0,
        harmonics=[
            IMHarmonicParams(R_r=1.54, L_mu=496.0e-3, L_s_sigma=15.1e-3, L_r_sigma=53.3e-3),
            IMHarmonicParams(R_r=1.57, L_mu=58.2e-3, L_s_sigma=13.6e-3, L_r_sigma=33.4e-3),
        ],
        curr_max=curr_max,
        volt_max=volt_max,
        omega_max=omega_max,
    )
    return IMDriveLUT(params)
