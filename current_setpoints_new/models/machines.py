from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch



def _build_cross_coupling(n_harmonics: int) -> np.ndarray:
    """
    Anti-symmetric block-diagonal J matrix for n_harmonics dq subspaces.
    Block h (1st, 3rd, 5th, ...) has the 2x2 form [[0, -h], [h, 0]].
    Shared by PMSM and IM — identical formula in both BaseMachine and BaseMachineIM.
    """
    dim = 2 * n_harmonics
    J = np.zeros((dim, dim))
    for i in range(n_harmonics):
        h = 2 * i + 1
        J[2 * i, 2 * i + 1] = -h
        J[2 * i + 1, 2 * i] = h
    return J


class FluxModel(ABC):
    """
    Abstract flux linkage provider for machines with permanent-magnet flux.

    Encapsulates both the flux vector λ(ω, i) and the incremental inductance
    L = ∂λ/∂i together, because L is always derived from the same source as λ
    (constant arrays, FEM table, or neural network). cross_coupling (J) is
    magnetic geometry and lives here alongside flux and L.

    Not used by IM — induction motors have no PM flux and implement the
    voltage equation directly on the drive class.
    """

    cross_coupling: np.ndarray  # J — anti-symmetric rotation/coupling matrix

    @abstractmethod
    def flux(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (flux_volt, flux_torq) for the voltage and torque equations."""

    @abstractmethod
    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        """
        Returns L = ∂λ/∂i.
        For constant models omega and curr_dq are ignored.
        For LUT / neural models both are used.
        """


class ConstantFlux(FluxModel):
    """
    Flux provider for PMSM with constant (linear) magnetics.
    All arrays are stored directly; omega and curr_dq are ignored by both methods.
    Replaces parameters/flux.py: Flux, ConstantFlux, Flux_IEEEMachine2.
    The IEEEMachine2 constants are baked into PMSM5Phase.__init__.
    """

    def __init__(
        self,
        flux_volt: np.ndarray,
        flux_torq: np.ndarray,
        L_stat: np.ndarray,
        cross_coupling: np.ndarray,
    ) -> None:
        self.flux_volt = np.asarray(flux_volt)
        self.flux_torq = np.asarray(flux_torq)
        self.L_stat = np.asarray(L_stat)
        self.cross_coupling = np.asarray(cross_coupling)

    def flux(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.flux_volt, self.flux_torq

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        return self.L_stat


class DriveModel(ABC):
    """
    Self-contained drive model. Encapsulates all physics for one machine type.

    Subclasses assign all attributes in their __init__ directly.
    Limits (curr_max, volt_max, omega_max) are not set by default;
    call set_max_pars() before any optimizer reads them.

    voltage_operator, bemf_dq, and inductance are abstract — PMSM implements
    them via a FluxModel; IM implements them directly with its own rotor coupling.
    """

    # identity
    n_phases: int
    n_harmonics: int
    dim: int  # = 2 * n_harmonics
    n_ppairs: int
    k_phase: float  # = n_phases / 2

    # limits
    curr_max: float
    volt_max: float
    omega_max: float

    # electrical
    R_stat: np.ndarray  # (dim, dim) stator resistance in dq frame
    k_v: float  # eddy-type iron-loss coefficient (ModelLossesSubstitution1)
    k_h: float  # hysteresis-type iron-loss coefficient

    
    @abstractmethod
    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        Returns the matrix U such that u_dq = U @ i_dq + bemf_dq(omega, curr_dq).
        Shape (dim, dim).
        """

    @abstractmethod
    def bemf_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        Back-EMF vector in the dq frame: ω·J·flux_volt(ω, i).
        Zero for IM (no PM flux).
        """

    @abstractmethod
    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        """L = ∂λ/∂i at the given operating point."""

    # mechanical

    @abstractmethod
    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        """Exact pointwise electromagnetic torque [Nm]."""

    @abstractmethod
    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        """Multi-start candidate current vectors for the optimizer."""


    def torque_quadratic(self, omega: float) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns (A, b, c) such that T(i) ≈ i^T A i + b^T i + c.
        Fitted by least squares from self.torque() samples.
        Exact for PMSM/IM (torque is truly quadratic in i); best-fit surrogate for neural.
        Faithful to dynamic/fourier_optimizer.extract_quadratic_torque
        (seed 0, scale 8.0, n=300, A returned symmetric).
        """
        dim = self.dim
        # Use the legacy global RNG with state save/restore to avoid numpy
        # RNG-constructor ABC recursion in older numpy versions.
        _state = np.random.get_state()
        np.random.seed(0)
        Xs = np.random.normal(scale=8.0, size=(300, dim))
        np.random.set_state(_state)
        idx = [(i, j) for i in range(dim) for j in range(i, dim)]
        feats = np.array([[x[i] * x[j] for (i, j) in idx] + list(x) + [1.0] for x in Xs])
        y = np.array([self.torque(omega, x) for x in Xs])
        coef = np.linalg.lstsq(feats, y, rcond=None)[0]
        A = np.zeros((dim, dim))
        for q, (i, j) in enumerate(idx):
            if i == j:
                A[i, i] = coef[q]
            else:
                A[i, j] = A[j, i] = coef[q] / 2.0
        b = coef[len(idx) : len(idx) + dim]
        c = float(coef[-1])
        return A, b, c

    def iron_loss(self, omega: float, volt_dq: np.ndarray) -> float:
        """
        Iron-loss torque correction M_Fe = k_v·f_v + k_h·f_h.
        ModelLossesSubstitution1 formula. Post-hoc benchmarking only.
        Faithful to ModelLossesSubstitution1.iron_loss.
        """
        from ..utils.loss_fit import substitution_loss_features  # lazy — avoids circular import
        f_v, f_h = substitution_loss_features(omega, volt_dq, self.n_ppairs)
        return float(self.k_v * f_v + self.k_h * f_h)

    def set_max_pars(self, curr_max: float, volt_max: float, omega_max: float) -> None:
        self.curr_max = curr_max
        self.volt_max = volt_max
        self.omega_max = omega_max



class PMSM5Phase(DriveModel):
    """
    5-phase PMSM — IEEEMachine2 prototype.

    Parameters:
        n_phases=5, n_ppairs=8
        R_stat = diag([0.0191, 0.0514, 0.0805, 0.0801]) Ω
        L_stat = 1e-3 * [[...]] H
        flux_volt = [0.0115, 0.0018, 0, 0] Wb
        flux_torq = [1.18255974e-2, -1.36757644e-3, 8.94095382e-5, -4.58552615e-5] Wb

    Torque (faithful to ModelAnalytical.calculate_torque):
        T = i^T A i + 2 b^T i
        A = (m·pp/4) · (J @ L + L @ J^T)
        b = (m·pp/4) · (J @ flux_torq)
        Both computed per-call — L and flux_torq will be ω- and i-dependent
        for LUT / neural flux providers.
    """

    def __init__(self) -> None:
        self.n_phases = 5
        self.n_harmonics = 2
        self.dim = 4
        self.n_ppairs = 8
        self.k_phase = 5 / 2

        self.R_stat = np.diag([0.0191, 0.0514, 0.0805, 0.0801])
        self.k_v = 0.0
        self.k_h = 0.0

        cross_coupling = _build_cross_coupling(self.n_harmonics)
        L_stat = 1e-3 * np.array(
            [
                [0.0920, -0.0286, -0.0141, 0.0010],
                [-0.0133, 0.1090, -0.0008, -0.0092],
                [-0.0088, 0.0037, 0.0725, -0.0466],
                [-0.0041, -0.0053, 0.0475, 0.0722],
            ]
        )
        flux_volt = np.array([0.0115, 0.0018, 0.0, 0.0])
        flux_torq = np.array([1.18255974e-02, -1.36757644e-03, 8.94095382e-05, -4.58552615e-05])

        self.flux: FluxModel = ConstantFlux(flux_volt, flux_torq, L_stat, cross_coupling)

    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        L = self.flux.inductance(omega, curr_dq)
        return self.R_stat + omega * self.flux.cross_coupling @ L

    def bemf_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        flux_volt, _ = self.flux.flux(omega, curr_dq)
        return omega * self.flux.cross_coupling @ flux_volt

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        return self.flux.inductance(omega, curr_dq)

    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        """
        T = i^T A i + 2 b^T i.
        A = (m·pp/4) · (J @ L + L @ J^T),  b = (m·pp/4) · (J @ flux_torq).
        Faithful to ModelAnalytical.calculate_torque.
        """
        _, flux_torq = self.flux.flux(omega, curr_dq)
        L = self.flux.inductance(omega, curr_dq)
        J = self.flux.cross_coupling
        k = self.n_phases * self.n_ppairs / 4.0
        A = k * (J @ L + L @ J.T)
        b = k * (J @ flux_torq)
        return float(curr_dq @ A @ curr_dq + 2 * b @ curr_dq)

    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        """
        PMSM multi-start seeds.
        Warm start augments rather than replaces — different points converge
        from different seeds; replacing the default can lose the only working seed.
        Faithful to BaseTorqueModel.get_candidates.
        """
        dim = self.dim
        candidates: list[np.ndarray] = []
        default = np.zeros(dim)
        default[0] = 1.0
        candidates.append(default)
        if guess is not None:
            candidates.append(guess.copy())
        g_mtpa = np.zeros(dim)
        g_mtpa[1] = self.curr_max * 0.95
        candidates.append(g_mtpa)
        g_fw = np.zeros(dim)
        g_fw[0] = -self.curr_max * 0.9
        g_fw[1] = self.curr_max * 0.1
        candidates.append(g_fw)
        return candidates


class IM9Phase(DriveModel):
    """
    9-phase induction motor — 15 kW laboratory prototype.

    Parameters from Laksar et al. (IM_TIA draft, 2025), Table I:
        n_phases=9, n_ppairs=2, n_harmonics=2
        R_s = 5.0 Ω
        R_r = diag([1.54, 1.54, 1.57, 1.57]) Ω
        L_mu = 1e-3 · diag([496, 496, 58.2, 58.2]) H
        L_s_sigma = 1e-3 · diag([15.1, 15.1, 13.6, 13.6]) H
        L_r_sigma = 1e-3 · diag([53.3, 53.3, 33.4, 33.4]) H

    No PM flux — voltage equation and torque implemented directly.
    k_ir and slip_from_dq_foc live on this class (no FluxModel).

    Torque (faithful to ModelIMAnalytical.calculate_torque):
        T = i_s^T A(omega_r) i_s
        A(omega_r) = (m·pp/2) · J · L_mu · k_ir(omega_r)
        Note coefficient /2, not /4 as in PMSM — intentional per IM_TIA.
    """

    def __init__(self) -> None:
        self.n_phases = 9
        self.n_harmonics = 2
        self.dim = 4
        self.n_ppairs = 2
        self.k_phase = 9 / 2

        R_s = 5.0
        self.R_stat = R_s * np.eye(self.dim)
        self.k_v = 0.0
        self.k_h = 0.0

        self._R_r = np.diag([1.54, 1.54, 1.57, 1.57])
        self._L_mu = 1e-3 * np.diag([496.0, 496.0, 58.2, 58.2])
        L_s_sigma = 1e-3 * np.diag([15.1, 15.1, 13.6, 13.6])
        L_r_sigma = 1e-3 * np.diag([53.3, 53.3, 33.4, 33.4])
        self._L_s = self._L_mu + L_s_sigma
        self._L_r = self._L_mu + L_r_sigma
        self._cross_coupling = _build_cross_coupling(self.n_harmonics)

    # ── IM-specific magnetics ─────────────────────────────────────────────────

    def k_ir(self, omega_r: float) -> np.ndarray:
        """
        Full dim×dim block-diagonal k_ir(omega_r) matrix assembled from
        per-harmonic 2x2 blocks.
        Per Laksar et al. (IM_TIA draft, 2025), eq. (kir_matrix).
        Faithful to parameters/im_coupling.k_ir_matrix.
        """
        K = np.zeros((self.dim, self.dim))
        for i in range(self.n_harmonics):
            h = 2 * i + 1
            L_mu_h = self._L_mu[2 * i, 2 * i]
            L_r_h = self._L_r[2 * i, 2 * i]
            R_r_h = self._R_r[2 * i, 2 * i]
            K[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = self._k_ir_block(omega_r, L_mu_h, L_r_h, R_r_h, h)
        return K

    @staticmethod
    def _k_ir_block(omega_r: float, L_mu_h: float, L_r_h: float, R_r_h: float, h: int) -> np.ndarray:
        """
        2x2 k_ir block for harmonic h at rotor slip frequency omega_r.

            k_ir^h = h·omega_r·L_mu^h / D · [[-h·omega_r·L_r^h,  R_r^h      ],
                                               [-R_r^h,           -h·omega_r·L_r^h]]
            D = (R_r^h)^2 + (h·omega_r·L_r^h)^2

        Returns zero matrix when D == 0 (synchronous speed — no rotor current).
        Faithful to parameters/im_coupling.k_ir_block.
        """
        h_omega_r = h * omega_r
        denom = R_r_h**2 + (h_omega_r * L_r_h) ** 2
        if denom == 0.0:
            return np.zeros((2, 2))
        scale = h_omega_r * L_mu_h / denom
        return scale * np.array(
            [
                [-h_omega_r * L_r_h, R_r_h],
                [-R_r_h, -h_omega_r * L_r_h],
            ]
        )

    def slip_from_dq_foc(self, curr_dq: np.ndarray, eps: float = 1e-9) -> float:
        """
        First-harmonic rotor-flux-orientation slip law:
            omega_r = (R_r^1 / L_r^1) * (i_sd^1 / i_sq^1)

        Returns 0 when |i_sq^1| < eps (zero-torque / standstill limit).
        Faithful to parameters/im_coupling.slip_from_dq_foc.
        """
        i_sd_1 = curr_dq[0]
        i_sq_1 = curr_dq[1]
        if abs(i_sq_1) < eps:
            return 0.0
        return (self._R_r[0, 0] / self._L_r[0, 0]) * (i_sd_1 / i_sq_1)

    # ── facade ───────────────────────────────────────────────────────────────

    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """R_s + ω·J·L_eff(omega_r),  L_eff = L_s + L_mu·k_ir(omega_r)."""
        return self.R_stat + omega * self._cross_coupling @ self.inductance(omega, curr_dq)

    def bemf_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """Zero — IM has no permanent-magnet flux."""
        return np.zeros(self.dim)

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        """
        L_eff = L_s + L_mu · k_ir(omega_r).
        Returns L_s when curr_dq is None (static approximation).
        """
        if curr_dq is None:
            return self._L_s
        omega_r = self.slip_from_dq_foc(curr_dq)
        return self._L_s + self._L_mu @ self.k_ir(omega_r)

    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        """
        T = i_s^T A(omega_r) i_s,  A(omega_r) = (m·pp/2) · J · L_mu · k_ir(omega_r).
        omega accepted for API parity but does not enter the torque expression directly.
        Faithful to ModelIMAnalytical.calculate_torque and _build_A.
        """
        omega_r = self.slip_from_dq_foc(curr_dq)
        A = (self.n_phases * self.n_ppairs / 2.0) * (self._cross_coupling @ self._L_mu @ self.k_ir(omega_r))
        return float(curr_dq @ A @ curr_dq)

    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        """
        IM multi-start seeds. Seven seeds to avoid slip-law degeneracy
        when i_sq^1 -> 0 and to cover balanced MTPA, FW, and 3rd-harmonic kick.
        Faithful to ModelIMAnalytical.get_candidates.
        """
        dim = self.dim
        I = self.curr_max
        candidates: list[np.ndarray] = []
        if guess is not None:
            candidates.append(guess.copy())
        c = np.zeros(dim)
        c[0] = 0.1
        c[1] = 0.1
        candidates.append(c)
        for amp in (0.3, 0.5, 0.7):
            c = np.zeros(dim)
            c[0] = I * amp
            c[1] = I * amp
            candidates.append(c)
        c = np.zeros(dim)
        c[0] = I * 0.85
        c[1] = I * 0.30
        candidates.append(c)
        if dim >= 4:
            c = np.zeros(dim)
            c[0] = I * 0.7
            c[1] = I * 0.7
            c[2] = I * 0.1
            c[3] = -I * 0.1
            candidates.append(c)
        return candidates

    def copper_loss(self, omega: float, curr_dq: np.ndarray) -> float:
        """
        Stator + rotor copper loss.
            P_stator = (m/2) · R_s · ||i_s||^2
            P_rotor  = (m/2) · i_r^T · R_r · i_r,   i_r = k_ir(omega_r) · i_s
        Faithful to efficiency.py add_efficiency_map.
        """
        R_s = self.R_stat[0, 0]
        P_stator = self.k_phase * R_s * float(curr_dq @ curr_dq)
        omega_r = self.slip_from_dq_foc(curr_dq)
        i_r = self.k_ir(omega_r) @ curr_dq
        P_rotor = self.k_phase * float(i_r @ self._R_r @ i_r)
        return P_stator + P_rotor


class NeuralPMSM5Phase(PMSM5Phase):
    """
    PMSM5Phase with neural torque residual correction.
    torque() = analytical baseline (from PMSM5Phase) + neural residual.
    Faithful to ModelNeural.calculate_torque.
    """

    def __init__(
        self,
        net: Any,  # NeuralTorquePredictor
        scaler: Any,  # sklearn StandardScaler
        device: torch.device,
    ) -> None:
        super().__init__()
        self._net = net
        self._scaler = scaler
        self._device = device

    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        return super().torque(omega, curr_dq) + self._predict(omega, curr_dq)

    def _predict(self, omega: float, curr_dq: np.ndarray) -> float:
        """NumPy ↔ PyTorch bridge for a single (omega, curr_dq) inference call."""
        X = self._scaler.transform(np.hstack(([omega], curr_dq)).reshape(1, -1))
        with torch.no_grad():
            residual = self._net(torch.from_numpy(X).float().to(self._device))
        return float(residual.cpu().numpy().item())
