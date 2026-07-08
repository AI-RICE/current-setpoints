from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.forward_model import ForwardModel


# ─────────────────────────────────────────────────────────────────────────────
# MachineData
# ─────────────────────────────────────────────────────────────────────────────

class MachineData:

    def __init__(
        self,
        torq: np.ndarray,
        omega: np.ndarray,
        segments: np.ndarray,
        curr_dq_grid: np.ndarray,
        k_skip: int | None = None,
    ) -> None:
        self.torq: np.ndarray = np.asarray(torq).flatten()
        self.omega: np.ndarray = np.asarray(omega).flatten()
        self.segments: np.ndarray = np.asarray(segments)
        self.curr_dq_grid: np.ndarray = np.asarray(curr_dq_grid)

        self._check_dimensions()

        if k_skip is not None and k_skip > 1:
            self.select_k(k_skip)

    def _check_dimensions(self) -> None:
        n_torq = len(self.torq)
        n_omega = len(self.omega)

        if n_torq == 0 or n_omega == 0:
            raise ValueError(
                f"torq and omega must be non-empty; got {n_torq} and {n_omega}."
            )

        expected = (n_torq, n_omega)
        if self.segments.shape != expected:
            raise ValueError(
                f"segments shape {self.segments.shape} != expected {expected}."
            )
        if self.curr_dq_grid.ndim != 3 or self.curr_dq_grid.shape[1:] != expected:
            raise ValueError(
                f"curr_dq_grid shape {self.curr_dq_grid.shape} must be "
                f"(dim, {n_torq}, {n_omega})."
            )
        if np.any(self.omega < 0):
            warnings.warn(
                "Negative speeds in grid data. Ensure reverse rotation is intended.",
                UserWarning,
                stacklevel=3,
            )

    def select_k(self, k_skip: int) -> None:
        if not isinstance(k_skip, (int, np.integer)) or k_skip < 1:
            raise ValueError(f"k_skip must be a positive integer, got {k_skip!r}.")
        if k_skip == 1:
            return
        self.torq = self.torq[::k_skip]
        self.omega = self.omega[::k_skip]
        self.segments = self.segments[::k_skip, ::k_skip]
        self.curr_dq_grid = self.curr_dq_grid[:, ::k_skip, ::k_skip]

    @property
    def torq_grid(self) -> np.ndarray:
        g, _ = np.meshgrid(self.torq, self.omega, indexing="ij")
        return g

    @property
    def omega_grid(self) -> np.ndarray:
        _, g = np.meshgrid(self.torq, self.omega, indexing="ij")
        return g

    @property
    def unique_segments(self) -> np.ndarray:
        return np.unique(self.segments[~np.isnan(self.segments)]).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Waveforms
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Waveforms:
    theta: np.ndarray
    curr_ph: np.ndarray
    volt_leg: np.ndarray
    volt_raw: np.ndarray
    volt_0: np.ndarray
    curr_dq: np.ndarray
    volt_dq: np.ndarray
    curr_peak: float
    volt_peak: float


# ─────────────────────────────────────────────────────────────────────────────
# evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
) -> Waveforms:
    v_dq = fwd.volt_dq(omega, curr_dq)
    c_ph = fwd.curr_ph(omega, curr_dq)
    volt_leg, volt_0, volt_raw = fwd.volt_ph(omega, curr_dq)
    curr_peak, volt_peak = fwd.peak_vals(omega, curr_dq)
    return Waveforms(
        theta=fwd.vec_theta,
        curr_ph=c_ph,
        volt_leg=volt_leg,
        volt_raw=volt_raw,
        volt_0=volt_0,
        curr_dq=curr_dq,
        volt_dq=v_dq,
        curr_peak=curr_peak,
        volt_peak=volt_peak,
    )


# ─────────────────────────────────────────────────────────────────────────────
# grid_to_data
# ─────────────────────────────────────────────────────────────────────────────

def grid_to_data(grid: dict[str, Any], k_skip: int) -> MachineData:
    torq = grid["vec_torq"]
    omega = grid["const_mech_speed"] * grid["vec_omega"]
    return MachineData(
        torq=torq,
        omega=omega,
        segments=grid["grid_segments"],
        curr_dq_grid=grid["curr_dq_grid"],
        k_skip=k_skip,
    )
