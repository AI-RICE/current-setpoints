from __future__ import annotations

from typing import Any

import numpy as np

from .machines import DriveModel


def _build_clarke_5phase() -> np.ndarray:
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
    _, _, vh = np.linalg.svd(C_red.T)
    row = vh[-1]
    return row / np.linalg.norm(row)


def count_peaks_at_limit(waveform: np.ndarray, limit: float, rel_tol: float = 1e-3) -> int:
    """How many times a reconstructed phase waveform touches ``limit`` (within
    ``rel_tol``). Shared between ``ForwardModel.count_peaks`` (constant-current
    case) and ``optimization.grid``'s dynamic-mode trajectory reconstruction —
    both just need this applied to an (n_phases, n_samples) waveform array,
    however it was produced."""
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
        R[:, 0, 0] = c1
        R[:, 0, 1] = -s1
        R[:, 1, 0] = s1
        R[:, 1, 1] = c1
        R[:, 2, 2] = c3
        R[:, 2, 3] = -s3
        R[:, 3, 2] = s3
        R[:, 3, 3] = c3

        return np.einsum("ij, tjk -> tik", T_inv, R)  # (n_t, n_surviving, dim)

    def extra_constraints(self) -> list[dict[str, Any]]:
        if self._N is None:
            return []
        N = self._N
        return [{"type": "eq", "fun": lambda i, N=N: float(N @ i)}]

    def extra_constraints_at_theta(self, theta_idx: int, vec_theta: np.ndarray) -> list[dict[str, Any]]:
        if self._N is None:
            return []
        th = float(vec_theta[theta_idx])
        c1, s1 = np.cos(th), np.sin(th)
        c3, s3 = np.cos(3 * th), np.sin(3 * th)
        R = np.array(
            [
                [c1, -s1, 0, 0],
                [s1, c1, 0, 0],
                [0, 0, c3, -s3],
                [0, 0, s3, c3],
            ]
        )
        NR = self._N @ R
        return [{"type": "eq", "fun": lambda i, NR=NR: float(NR @ i)}]


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
        # Convention: cos(h·θ - h·p·φ), φ = 2π/n_phases.
        phi = 2 * np.pi / n_phases
        n_t = self.vec_theta.size
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

    def volt_map_at_theta(self, theta_idx: int, omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zeros = np.zeros(self.drive.dim)
        U = self.drive.voltage_operator(omega, zeros)
        L = self.drive.inductance(omega, zeros)
        bemf_dq = self.drive.bemf_dq(omega, zeros)
        H_ph = self._mat_dq_to_ph_all[theta_idx]  # (n_phases, dim)
        return H_ph @ U, H_ph @ L, H_ph @ bemf_dq

    def extra_constraints(self) -> list[dict[str, Any]]:
        return self.fault.extra_constraints()

    def extra_constraints_at_theta(self, theta_idx: int) -> list[dict[str, Any]]:
        return self.fault.extra_constraints_at_theta(theta_idx, self.vec_theta)
