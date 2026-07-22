from __future__ import annotations

from typing import Any

import numpy as np

from .machines import DriveModel


def _build_clarke_5phase() -> np.ndarray:
    """Static (theta-independent) Clarke/Concordia matrix at the 5 fixed
    phase positions a_p = p*2pi/5: harmonic 1 (rows 0-1) and harmonic 2
    (rows 2-3, per the reference formulation's Eq.(1) -- NOT harmonic 3;
    the rotation rate used to reconstruct time-domain quantities from this
    static projection is a separate, genuinely different harmonic (3), see
    _build_dq_to_phase_map)."""
    C = np.zeros((5, 5))
    for p in range(5):
        a = p * 2 * np.pi / 5
        C[0, p] = np.cos(a)
        C[1, p] = np.sin(a)
        C[2, p] = np.cos(2 * a)
        C[3, p] = np.sin(2 * a)
        C[4, p] = 0.5
    return C * (2.0 / 5.0)


def _null_space_row(C_red: np.ndarray) -> np.ndarray:
    """Row vector N such that N @ C_red == 0 (left null space of C_red, i.e.
    N is orthogonal to Range(C_red)). Used ONLY for the paper's static
    achievability constraint (Eq. 31/42: N @ i_s = 0, never rotated,
    checked once for a single constant i_s) -- NOT for the per-angle
    trajectory optimizers, which need a genuinely different, theta-
    dependent condition (see Fault.extra_constraints_at_theta)."""
    _, _, vh = np.linalg.svd(C_red.T)
    row = vh[-1]
    return row / np.linalg.norm(row)


def _build_dq_to_phase_map(kept: tuple[int, ...], vec_theta: np.ndarray) -> np.ndarray:
    """The reference formulation's two-step reconstruction (Eq.(2)/Eq.(12)):
    invert the static, theta-independent Clarke matrix once (built with
    harmonic 1 and 2), then rotate by a separate, theta-dependent R(theta)
    that uses harmonic 1 and **3** for the second block -- a genuinely
    different harmonic from the static matrix's, not a relabeling of it.
    Used identically for the always-5-phase voltage map (kept = all 5
    phases) and the fault-reduced current map (kept = surviving phases).

    Returns a (n_t, len(kept), dim) map. Achievability of a chosen i_dq for
    the *open* phases (i.e. does the current model's own full 5-phase map
    also reconstruct exactly zero current there) is NOT determined by this
    static-block pseudo-inverse -- see Fault.extra_constraints_at_theta,
    which checks the open phases directly through the full theta-dependent
    map instead of through a null vector of this static construction (the
    two are not equivalent, since this map mixes the static harmonic-2
    block with a harmonic-3 rotation; a vector that is a null direction of
    the static block only is not generally a null direction of the
    rotated map at a given theta).
    """
    dim = 4
    C_full = _build_clarke_5phase()
    C_red = C_full[:4, :][:, list(kept)]  # (dim, n_kept)
    C_inv = np.linalg.pinv(C_red) if C_red.shape[1] != dim else np.linalg.inv(C_red)

    n_t = vec_theta.size
    c1, s1 = np.cos(vec_theta), np.sin(vec_theta)
    c3, s3 = np.cos(3 * vec_theta), np.sin(3 * vec_theta)
    R = np.zeros((n_t, dim, dim))
    R[:, 0, 0] = c1
    R[:, 0, 1] = -s1
    R[:, 1, 0] = s1
    R[:, 1, 1] = c1
    R[:, 2, 2] = c3
    R[:, 2, 3] = -s3
    R[:, 3, 2] = s3
    R[:, 3, 3] = c3

    return np.einsum("ij, tjk -> tik", C_inv, R)


def count_peaks_at_limit(waveform: np.ndarray, limit: float, rel_tol: float = 1e-3) -> int:
    """How many times a reconstructed phase waveform touches ``limit`` (within
    ``rel_tol``). Shared between ``ForwardModel.count_peaks`` (constant-current
    case) and ``optimization.grid``'s dynamic-mode trajectory reconstruction —
    both just need this applied to an (n_phases, n_samples) waveform array,
    however it was produced.

    The waveform is treated as circular (theta=0 and theta=2pi are the same
    physical point) via np.roll, so a peak sitting exactly at the array
    boundary is not silently invisible to the search. A plateau of equal
    values at a peak counts once, not once per sample: a point only starts
    a new peak if it's strictly greater than its predecessor (>=1 sample
    into the rise) and at least as large as its successor (still at or
    past the top), so only the plateau's leading edge is flagged.
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    thresh = limit * (1.0 - rel_tol)
    n_max = 0
    for ph in range(waveform.shape[0]):
        w = waveform[ph, :-1]  # drop the wrap sample; the array is circular
        if w.size < 3:
            continue
        prev = np.roll(w, 1)
        nxt = np.roll(w, -1)
        is_pmax = (w > prev) & (w >= nxt) & (w > 0)
        n_pos = int(np.sum(w[is_pmax] >= thresh))
        is_nmin = (w < prev) & (w <= nxt) & (w < 0)
        n_neg = int(np.sum(-w[is_nmin] >= thresh))
        n_max = max(n_max, n_pos, n_neg)
    return min(n_max, 2)


class Fault:
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
        self._N: np.ndarray | None = None  # static null-space row; set by _build_reduced_map

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
        dim = 4
        C_full = _build_clarke_5phase()
        C_red = C_full[:4, :][:, list(self._kept)]  # (dim, n_kept)
        self._N = _null_space_row(C_red) if C_red.shape[1] < dim else None
        return _build_dq_to_phase_map(self._kept, vec_theta)  # (n_t, n_surviving, dim)

    def static_constraints(self) -> list[dict[str, Any]]:
        """The paper's own achievability constraint (Eq. 31/42): a single,
        NEVER-rotated N @ i_s = 0, for a genuinely constant i_s used across
        the whole cycle (StaticOptimizer only -- current/voltage limits are
        separately checked across the full theta range by the caller). Only
        meaningful for a 2-open-phase fault (N is None otherwise: healthy and
        single-fault have no redundant dq direction to constrain)."""
        if self._N is None:
            return []
        N = self._N
        return [{"type": "eq", "fun": lambda i, N=N: float(N @ i)}]

    def extra_constraints_at_theta(self, theta_idx: int, mat_dq_to_ph_all: np.ndarray) -> list[dict[str, Any]]:
        """Per-angle achievability for the trajectory-based optimizers (a
        genuinely different i_s(theta) is solved at each grid node, so the
        constraint only needs to hold AT that node's theta, not globally):
        an open phase carries physically zero current at every instant, so
        the constraint is simply that open phase's own row of the full
        (fault-independent) dq-to-phase map, evaluated at this theta, dotted
        with i_dq. One constraint per open phase. This is NOT the same
        object as static_constraints()'s N -- rotating that static N by
        R(theta) does not actually zero the open phase's current when
        checked against the true theta-dependent map (verified empirically:
        ~0.3-0.5A residual on a unit-norm test current), because it mixes
        the static harmonic-2 block with a harmonic-3 rotation in a way that
        only cancels for a genuinely theta-VARYING i_s(theta), not a fixed
        one rotated after the fact."""
        if not self.open_phases:
            return []
        rows = mat_dq_to_ph_all[theta_idx][list(self.open_phases)]  # (n_open, dim)
        return [{"type": "eq", "fun": lambda i, row=row: float(row @ i)} for row in rows]


class ForwardModel:
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
            raise ValueError(f"n_theta too small after rounding; increase to at least {2 * n_phases}.")
        self.vec_theta: np.ndarray = np.linspace(0, 2 * np.pi, n_theta + 1)
        self._phase_shift_samples: int = n_theta // n_phases

        # Full inverse-Park (n_theta+1, n_phases, dim) — fault-independent.
        # Entry [t, p, :] maps curr_dq to the instantaneous contribution of
        # harmonic content to physical phase p at rotor angle vec_theta[t].
        n_t = self.vec_theta.size
        if n_phases == 5:
            # Two-step reconstruction matching the reference formulation's
            # Eq.(2)/Eq.(12): invert the static (harmonic 1, 2) Clarke matrix
            # once, then rotate by harmonic (1, 3) R(theta) -- a genuinely
            # different harmonic from the static one, not a relabeling.
            mat_all = _build_dq_to_phase_map(tuple(range(5)), self.vec_theta)
        else:
            # Generic n-phase construction (used by the induction-motor
            # drives): no fault support, no reference formulation to match,
            # single combined harmonic convention throughout.
            phi = 2 * np.pi / n_phases
            mat_all = np.zeros((n_t, n_phases, dim))
            for i in range(n_harmonics):
                h = 2 * i + 1
                for p in range(n_phases):
                    theta_arg = h * self.vec_theta - h * p * phi
                    mat_all[:, p, 2 * i] = np.cos(theta_arg)
                    mat_all[:, p, 2 * i + 1] = -np.sin(theta_arg)
        self._mat_dq_to_ph_all: np.ndarray = mat_all

        # Current map — equals full inverse-Park for healthy machines;
        # reduced-Clarke for faulted (open phases carry no current).
        if self.fault.is_healthy:
            self._mat_curr_to_ph: np.ndarray = mat_all
        else:
            self._mat_curr_to_ph = self.fault._build_reduced_map(self.vec_theta)

    def volt_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        return self.drive.voltage_operator(omega, curr_dq) @ curr_dq + self.drive.bemf_dq(omega, curr_dq)

    def curr_ph(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        return np.einsum("tik, k -> it", self._mat_curr_to_ph, curr_dq)

    def volt_ph(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Fault-independent inverse-Park map: voltage across a surviving
        # phase's own winding doesn't depend on which OTHER phase is open --
        # only current genuinely redistributes under a fault. Reverted
        # 2026-07-21 after a literal per-paper-equation version (reusing the
        # fault-reduced current map for voltage) produced negative max-torque
        # and solver failures at realistic speeds; also matches this
        # research line's own prior, documented finding (docs/sota.md S3:
        # conflating the two maps over-counts surviving-phase voltage by
        # ~60% at speed). See FAULT_TOLERANT.md for the full record.
        v_dq = self.volt_dq(omega, curr_dq)
        volt_raw = np.einsum("tpk, k -> pt", self._mat_dq_to_ph_all, v_dq)

        if self.add_volt_0:
            volt_0_scalar = -0.5 * (np.min(volt_raw, axis=0) + np.max(volt_raw, axis=0))
            volt_0 = np.tile(volt_0_scalar, (self.drive.n_phases, 1))
        else:
            volt_0 = np.zeros_like(volt_raw)
        return volt_raw + volt_0, volt_0, volt_raw

    def peak_vals(self, omega: float, curr_dq: np.ndarray) -> tuple[float, float]:
        curr_peak = float(np.max(np.abs(self.curr_ph(omega, curr_dq))))
        volt_ph, _, _ = self.volt_ph(omega, curr_dq)
        volt_peak = float(np.max(np.abs(volt_ph[list(self.fault._kept), :])))
        return curr_peak, volt_peak

    def count_peaks(self, omega: float, curr_dq: np.ndarray, rel_tol: float = 1e-3) -> tuple[int, int]:
        curr_ph = self.curr_ph(omega, curr_dq)
        volt_ph, _, _ = self.volt_ph(omega, curr_dq)
        volt_surviving = volt_ph[list(self.fault._kept), :]
        n_curr = count_peaks_at_limit(curr_ph, self.drive.curr_max, rel_tol)
        n_volt = count_peaks_at_limit(volt_surviving, self.drive.volt_max, rel_tol)
        return n_curr, n_volt

    def phase_map_at_theta(self, theta_idx: int) -> np.ndarray:
        return self._mat_curr_to_ph[theta_idx]

    def volt_map_at_theta(
        self, theta_idx: int, omega: float, lin_curr: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """U/L/bemf_dq linearized around lin_curr (default zeros). Exact for
        a linear machine (ConstantFlux -- these don't depend on current, so
        any linearization point gives the same answer); an approximation
        for a genuinely current-dependent flux model (NeuralFlux) unless
        lin_curr is close to the actual operating current."""
        if lin_curr is None:
            lin_curr = np.zeros(self.drive.dim)
        U = self.drive.voltage_operator(omega, lin_curr)
        L = self.drive.inductance(omega, lin_curr)
        bemf_dq = self.drive.bemf_dq(omega, lin_curr)
        # Fault-independent map (see volt_ph) -- always all 5 phases.
        H_ph = self._mat_dq_to_ph_all[theta_idx]  # (n_phases, dim)
        return H_ph @ U, H_ph @ L, H_ph @ bemf_dq

    def extra_constraints(self) -> list[dict[str, Any]]:
        # The paper's own static achievability constraint (Eq. 31/42): a
        # single, never-rotated N @ i_s = 0 for a genuinely constant i_s.
        # StaticOptimizer is the only caller; it separately checks current
        # and voltage limits across the full theta range (peak_vals/
        # count_peaks already scan the whole waveform, not a snapshot).
        return self.fault.static_constraints()

    def extra_constraints_at_theta(self, theta_idx: int) -> list[dict[str, Any]]:
        return self.fault.extra_constraints_at_theta(theta_idx, self._mat_dq_to_ph_all)
