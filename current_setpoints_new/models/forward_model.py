"""
ForwardModel: dq-to-phase reconstruction for static and dynamic setpoint optimization.
Fault: open-circuit fault descriptor (value object) for 5-phase PMSM.

Voltage mapping is always fault-independent (inverse-Park): an open-circuit fault
forces the faulted phase current to zero but does not change the machine's voltage
equation. Current mapping uses the reduced-Clarke map under fault.

This separation fixes the high-speed fault bug present in the fault-tolerant branch
(current_setpoints_rename), where the reduced-Clarke current map was incorrectly
reused for voltage — over-counting surviving-phase voltage once ω·L·di/dθ became
significant. Faithful to dynamic/phase_voltage.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .machines import DriveModel


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_clarke_5phase() -> np.ndarray:
    """
    5-phase Clarke transform with 2/5 normalization.
    Rows: [alpha_1, beta_1, alpha_3, beta_3, zero_seq].
    Third-harmonic rows use cos(3·a)/sin(3·a) — the physical h·p·φ form.
    At the 5 sample angles cos(3·a) = cos(2·a) numerically (3 ≡ -2 mod 5),
    so C_red is identical to the textbook Concordia form, but only the h·p·φ
    convention reconstructs cos(h·θ - h·p·φ) correctly when composed with
    the Park rotation R(h·θ).
    """
    C = np.zeros((5, 5))
    for p in range(5):
        a = p * 2 * np.pi / 5
        C[0, p] = np.cos(a)
        C[1, p] = np.sin(a)
        C[2, p] = np.cos(3 * a)
        C[3, p] = np.sin(3 * a)
        C[4, p] = 0.5
    return C * (2.0 / 5.0)


def _null_space_row(C_red: np.ndarray) -> np.ndarray:
    """
    Unit-norm row spanning the left null space of C_red (shape 4×3 for two-fault).
    Computed via SVD of C_red^T; the last right singular vector is orthogonal
    to all columns of C_red^T, i.e. in the left null space of C_red.
    """
    _, _, vh = np.linalg.svd(C_red.T)
    row = vh[-1]
    return row / np.linalg.norm(row)


def _count_peaks_at_limit(waveform: np.ndarray, limit: float, rel_tol: float) -> int:
    """
    Counts polarity-consistent local extrema that reach limit within rel_tol.

    waveform: (n_phases, n_theta+1) stacked across phases, or (n_theta+1,).
    Returns min(max_across_phases, 2):
        0 — no peak reaches the limit
        1 — Type I (one peak per polarity per phase)
        2 — Type II (two or more peaks per polarity)

    Faithful to BaseTransform._count_waveform_peaks_at_limit.
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    thresh = limit * (1.0 - rel_tol)
    n_max = 0
    for ph in range(waveform.shape[0]):
        w = waveform[ph, :-1]  # drop the wrap sample
        if w.size < 3:
            continue
        is_pmax = (w[1:-1] > w[:-2]) & (w[1:-1] > w[2:]) & (w[1:-1] > 0)
        n_pos = int(np.sum(w[1:-1][is_pmax] >= thresh))
        is_nmin = (w[1:-1] < w[:-2]) & (w[1:-1] < w[2:]) & (w[1:-1] < 0)
        n_neg = int(np.sum(-w[1:-1][is_nmin] >= thresh))
        n_max = max(n_max, n_pos, n_neg)
    return min(n_max, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Fault value object
# ─────────────────────────────────────────────────────────────────────────────

class Fault:
    """
    Open-circuit fault descriptor for a 5-phase PMSM.

    Fault configurations supported:
        Healthy:          open_phases=()
        Single fault:     open_phases=(k,)              C_red 4×4, square
        Two adjacent:     open_phases=(k, k±1 mod 5)    C_red 4×3, pseudoinverse
        Two non-adjacent: open_phases=(k, j)             C_red 4×3, pseudoinverse

    The reduced-Clarke current map and null-space row (for two-fault) are built
    lazily by ForwardModel.__init__ via _build_reduced_map(). extra_constraints
    and extra_constraints_at_theta must only be called after that.
    """

    def __init__(self, open_phases: tuple[int, ...] = ()) -> None:
        open_phases = tuple(sorted(int(p) for p in open_phases))
        if len(open_phases) > 2:
            raise ValueError("At most two open phases are supported.")
        if any(not 0 <= p <= 4 for p in open_phases):
            raise ValueError("Phase indices must lie in [0, 4].")
        if len(open_phases) == 2 and open_phases[0] == open_phases[1]:
            raise ValueError("open_phases must be two distinct indices.")
        self.open_phases: tuple[int, ...] = open_phases
        self._kept: tuple[int, ...] = tuple(p for p in range(5) if p not in open_phases)
        self._N: np.ndarray | None = None  # null-space row; set by _build_reduced_map

    @property
    def is_healthy(self) -> bool:
        return len(self.open_phases) == 0

    @property
    def n_open(self) -> int:
        return len(self.open_phases)

    @property
    def n_surviving(self) -> int:
        return 5 - self.n_open

    def _build_reduced_map(self, vec_theta: np.ndarray) -> np.ndarray:
        """
        Builds the (n_theta+1, n_surviving, dim) current map from dq to surviving
        phase currents at every rotor angle.

        Single-fault: C_red is 4×4 and square-invertible; no null-space constraint.
        Two-fault:    C_red is 4×3; uses Moore-Penrose pseudoinverse and sets self._N
                      to the left null-space row so that extra_constraints() can enforce
                      N @ curr_dq = 0.

        Faithful to dynamic/envelope_solver.fault_phase_map (extended to two-fault
        with pinv + null space, and using cos(3·a) Clarke convention).
        """
        C_full = _build_clarke_5phase()
        C_red = C_full[:4, :][:, list(self._kept)]

        if C_red.shape[0] == C_red.shape[1]:
            T_inv = np.linalg.inv(C_red)
            self._N = None
        else:
            T_inv = np.linalg.pinv(C_red)
            self._N = _null_space_row(C_red)

        n_t = vec_theta.size
        c1, s1 = np.cos(vec_theta), np.sin(vec_theta)
        c3, s3 = np.cos(3 * vec_theta), np.sin(3 * vec_theta)
        R = np.zeros((n_t, 4, 4))
        R[:, 0, 0] = c1;  R[:, 0, 1] = -s1
        R[:, 1, 0] = s1;  R[:, 1, 1] = c1
        R[:, 2, 2] = c3;  R[:, 2, 3] = -s3
        R[:, 3, 2] = s3;  R[:, 3, 3] = c3

        return np.einsum("ij, tjk -> tik", T_inv, R)  # (n_t, n_surviving, dim)

    def extra_constraints(self) -> list[dict[str, Any]]:
        """
        Static null-space equality N @ curr_dq = 0 in SLSQP format.
        Required for two-fault to prevent the optimizer from inflating
        null-space components of curr_dq that produce no physical phase current
        but non-zero dq-frame torque.
        Returns [] for healthy and single-fault (C_red is square, no null space).
        """
        if self._N is None:
            return []
        N = self._N
        return [{"type": "eq", "fun": lambda i, N=N: float(N @ i)}]

    def extra_constraints_at_theta(
        self, theta_idx: int, vec_theta: np.ndarray
    ) -> list[dict[str, Any]]:
        """
        Per-theta null-space equality N @ R(θ) @ curr_dq = 0 for the dynamic optimizer.
        Pins the αβ vector R(θ) @ curr_dq to col(C_red) at every collocation angle so
        that realized phase currents match curr_dq and dq torque equals realized torque.
        Returns [] for healthy and single-fault.
        """
        if self._N is None:
            return []
        th = float(vec_theta[theta_idx])
        c1, s1 = np.cos(th), np.sin(th)
        c3, s3 = np.cos(3 * th), np.sin(3 * th)
        R = np.array([
            [c1, -s1,  0,   0],
            [s1,  c1,  0,   0],
            [0,   0,  c3, -s3],
            [0,   0,  s3,  c3],
        ])
        NR = self._N @ R
        return [{"type": "eq", "fun": lambda i, NR=NR: float(NR @ i)}]


# ─────────────────────────────────────────────────────────────────────────────
# ForwardModel
# ─────────────────────────────────────────────────────────────────────────────

class ForwardModel:
    """
    dq-to-phase reconstruction for static and dynamic setpoint optimization.

    Builds two phase-mapping matrices at construction time:

        _mat_dq_to_ph_all  (n_theta+1, n_phases, dim) — full inverse-Park,
            fault-independent. Used for voltage in all modes.

        _mat_curr_to_ph    (n_theta+1, n_surviving, dim) — reduced-Clarke map,
            fault-dependent. Used for current. Identical to _mat_dq_to_ph_all
            for healthy machines.

    Args:
        drive:      DriveModel providing voltage_operator and bemf_dq.
        fault:      Fault descriptor. Default: Fault() (healthy).
        add_volt_0: Enable zero-sequence (SVPWM) injection. Default False.
        n_theta:    Angular resolution; rounded internally to a multiple of
                    2 * n_phases to align phase-shift samples correctly.
    """

    def __init__(
        self,
        drive: DriveModel,
        fault: Fault | None = None,
        add_volt_0: bool = False,
        n_theta: int = 700,
    ) -> None:
        self.drive = drive
        self.fault = fault if fault is not None else Fault()
        self.add_volt_0 = add_volt_0

        n_phases = drive.n_phases
        n_harmonics = drive.n_harmonics
        dim = drive.dim

        if n_theta <= 0:
            raise ValueError("n_theta must be positive.")
        n_theta = round(n_theta / (2 * n_phases)) * 2 * n_phases
        if n_theta < 2 * n_phases:
            raise ValueError(
                f"n_theta too small after rounding; increase to at least {2 * n_phases}."
            )
        self.vec_theta: np.ndarray = np.linspace(0, 2 * np.pi, n_theta + 1)
        self._phase_shift_samples: int = n_theta // n_phases

        # Full inverse-Park (n_theta+1, n_phases, dim) — fault-independent.
        # Entry [t, p, :] maps curr_dq to the instantaneous contribution of
        # harmonic content to physical phase p at rotor angle vec_theta[t].
        # Convention: cos(h·θ - h·p·φ), φ = 2π/n_phases.
        phi = 2 * np.pi / n_phases
        n_t = self.vec_theta.size
        mat_all = np.zeros((n_t, n_phases, dim))
        for i in range(n_harmonics):
            h = 2 * i + 1
            for p in range(n_phases):
                theta_arg = h * self.vec_theta - h * p * phi
                mat_all[:, p, 2 * i]     = np.cos(theta_arg)
                mat_all[:, p, 2 * i + 1] = -np.sin(theta_arg)
        self._mat_dq_to_ph_all: np.ndarray = mat_all

        # Current map — equals full inverse-Park for healthy machines;
        # reduced-Clarke for faulted (open phases carry no current).
        if self.fault.is_healthy:
            self._mat_curr_to_ph: np.ndarray = mat_all
        else:
            self._mat_curr_to_ph = self.fault._build_reduced_map(self.vec_theta)

    # ── dq domain ─────────────────────────────────────────────────────────────

    def volt_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """u_dq = U(ω, i) @ i + e_dq(ω, i)."""
        return (
            self.drive.voltage_operator(omega, curr_dq) @ curr_dq
            + self.drive.bemf_dq(omega, curr_dq)
        )

    # ── phase domain ──────────────────────────────────────────────────────────

    def curr_ph(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        Phase currents via reduced-Clarke map.
        Shape: (n_surviving, n_theta+1).
        Open phases are not in the output (they carry zero current by definition).
        omega is accepted for API parity; current map is omega-independent.
        """
        return np.einsum("tik, k -> it", self._mat_curr_to_ph, curr_dq)

    def volt_ph(
        self, omega: float, curr_dq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Phase voltages via inverse-Park (fault-independent).
        Returns (volt_leg, volt_0, volt_raw), each shape (n_phases, n_theta+1).

        volt_raw: per-phase voltage from the machine's voltage equation.
        volt_0:   zero-sequence injection signal (SVPWM) if add_volt_0=True.
                  v_0(θ) = -½ (min_k v_k(θ) + max_k v_k(θ)).
        volt_leg: volt_raw + volt_0 — actual leg voltage the inverter supplies.

        All n_phases are returned. The optimizer selects surviving phases via
        fault._kept when building voltage constraints.
        """
        v_dq = self.volt_dq(omega, curr_dq)
        volt_raw = np.einsum("tpk, k -> pt", self._mat_dq_to_ph_all, v_dq)

        if self.add_volt_0:
            volt_0_scalar = -0.5 * (
                np.min(volt_raw, axis=0) + np.max(volt_raw, axis=0)
            )
            volt_0 = np.tile(volt_0_scalar, (self.drive.n_phases, 1))
        else:
            volt_0 = np.zeros_like(volt_raw)
        return volt_raw + volt_0, volt_0, volt_raw

    # ── peak values and active-set classification ─────────────────────────────

    def peak_vals(self, omega: float, curr_dq: np.ndarray) -> tuple[float, float]:
        """
        (curr_peak, volt_peak) as scalar floats.
        curr_peak: max |i| over surviving phases and all theta.
        volt_peak: max |v_leg| over surviving phases and all theta.
                   Open phases are excluded (not inverter-constrained).
        """
        curr_peak = float(np.max(np.abs(self.curr_ph(omega, curr_dq))))
        volt_ph, _, _ = self.volt_ph(omega, curr_dq)
        volt_peak = float(np.max(np.abs(volt_ph[list(self.fault._kept), :])))
        return curr_peak, volt_peak

    def count_peaks(
        self, omega: float, curr_dq: np.ndarray, rel_tol: float = 1e-3
    ) -> tuple[int, int]:
        """
        Active-set regime classification. Returns (n_curr, n_volt) in {0, 1, 2}.
            0 — not at limit
            1 — Type I (one local extremum per polarity reaches the limit)
            2 — Type II (two or more extrema reach the limit)
        Waveform-based detector — no angle heuristic.
        Faithful to BaseTransform._count_waveform_peaks_at_limit.
        """
        curr_ph = self.curr_ph(omega, curr_dq)
        volt_ph, _, _ = self.volt_ph(omega, curr_dq)
        volt_surviving = volt_ph[list(self.fault._kept), :]
        n_curr = _count_peaks_at_limit(curr_ph, self.drive.curr_max, rel_tol)
        n_volt = _count_peaks_at_limit(volt_surviving, self.drive.volt_max, rel_tol)
        return n_curr, n_volt

    # ── dynamic optimizer interface ───────────────────────────────────────────

    def phase_map_at_theta(self, theta_idx: int) -> np.ndarray:
        """
        Current-side phase map at one rotor angle: shape (n_surviving, dim).
        Row p maps curr_dq to the instantaneous current of surviving phase p.
        Used by the dynamic optimizer to build per-theta current constraints.
        """
        return self._mat_curr_to_ph[theta_idx]

    def volt_map_at_theta(
        self, theta_idx: int, omega: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Voltage-side linear maps at one rotor angle for the dynamic optimizer.
        Returns (gU, gL, bV) such that:
            v_phase(θ) ≈ gU @ curr_dq + omega * gL @ (di/dθ) + bV

        gU: (n_phases, dim) — static voltage map: H_ph @ (R + ω·J·L).
        gL: (n_phases, dim) — inductive map: H_ph @ L (for ω·L·di/dθ term).
        bV: (n_phases,)     — BEMF offset: H_ph @ e_dq.

        Uses inverse-Park (fault-independent). For ConstantFlux the linearization
        is exact; for current-dependent flux it is evaluated at zero current.
        """
        zeros = np.zeros(self.drive.dim)
        U = self.drive.voltage_operator(omega, zeros)
        L = self.drive.inductance(omega, zeros)
        bemf_dq = self.drive.bemf_dq(omega, zeros)
        H_ph = self._mat_dq_to_ph_all[theta_idx]   # (n_phases, dim)
        return H_ph @ U, H_ph @ L, H_ph @ bemf_dq

    # ── fault constraint delegation ───────────────────────────────────────────

    def extra_constraints(self) -> list[dict[str, Any]]:
        """Static null-space equality for two-fault. Empty for healthy/single-fault."""
        return self.fault.extra_constraints()

    def extra_constraints_at_theta(self, theta_idx: int) -> list[dict[str, Any]]:
        """Per-theta null-space equality for dynamic two-fault. Empty otherwise."""
        return self.fault.extra_constraints_at_theta(theta_idx, self.vec_theta)
