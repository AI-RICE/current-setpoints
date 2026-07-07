"""
Data containers and assessment utilities for grid results and waveform analysis.

Classes
-------
MachineData  — structured container for a (T*, omega) grid; used for neural training.
Waveforms    — full phase-domain signal bundle at one operating point.

Functions
---------
evaluate     — compute Waveforms from a ForwardModel at one static setpoint.
grid_to_data — convert a calculate_grid output dict into a MachineData instance.
"""
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
    """
    Structured container for a (T*, omega) setpoint grid.

    Holds the torque and speed axes, per-cell dq current vectors, and
    regime segment labels. Supports optional downsampling via select_k,
    and provides meshgrid properties for plotting and training pipelines.

    Parameters
    ----------
    torq : array-like (n_torq,)
        Torque axis values [Nm].
    omega : array-like (n_omega,)
        Speed axis values [mechanical RPM]. Produced by grid_to_data which
        converts the internal electrical-rad/s representation.
    segments : ndarray (n_torq, n_omega)
        Regime label per cell: 3 * n_volt_peaks + n_curr_peaks (0–8).
        NaN for unfilled cells.
    curr_dq_grid : ndarray (dim, n_torq, n_omega)
        DQ current setpoints, ordered [d1, q1, d3, q3, ...] along axis 0.
        NaN for unfilled cells.
    k_skip : int | None
        If > 1, downsample both axes by taking every k_skip-th element.
    """

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
        """Downsample both axes by taking every k_skip-th element."""
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
        """Meshgrid of torque values, shape (n_torq, n_omega)."""
        g, _ = np.meshgrid(self.torq, self.omega, indexing="ij")
        return g

    @property
    def omega_grid(self) -> np.ndarray:
        """Meshgrid of speed values, shape (n_torq, n_omega)."""
        _, g = np.meshgrid(self.torq, self.omega, indexing="ij")
        return g

    @property
    def unique_segments(self) -> np.ndarray:
        """Sorted integer unique non-NaN segment labels present in the grid."""
        return np.unique(self.segments[~np.isnan(self.segments)]).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Waveforms
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Waveforms:
    """
    Full phase-domain signal bundle at one static operating point.
    Produced by evaluate(). Use this for post-optimization analysis and
    plotting, not inside the optimizer loop.

    Attributes
    ----------
    theta     (n_theta+1,)            — rotor angle samples [rad]
    curr_ph   (n_surviving, n_theta+1)— phase currents (surviving phases only)
    volt_leg  (n_phases, n_theta+1)   — inverter leg voltage (raw + ZSC)
    volt_raw  (n_phases, n_theta+1)   — machine-side phase voltage
    volt_0    (n_phases, n_theta+1)   — zero-sequence injection signal
    curr_dq   (dim,)                  — dq current setpoint
    volt_dq   (dim,)                  — dq voltage
    curr_peak float                   — max |i| over surviving phases and theta
    volt_peak float                   — max |v_leg| over surviving phases and theta
    """
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
    """
    Compute the full waveform bundle at one static operating point.

    Calls fwd.volt_dq, fwd.curr_ph, fwd.volt_ph, and fwd.peak_vals and
    packages the results into a Waveforms instance. Use for post-optimization
    assessment and plotting; not intended for use inside the optimizer loop.

    Parameters
    ----------
    fwd : ForwardModel
    omega : float
        Electrical speed [rad/s].
    curr_dq : ndarray (dim,)
        DQ current setpoint.

    Returns
    -------
    Waveforms
    """
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
    """
    Convert a calculate_grid output dict into a MachineData instance.

    Parameters
    ----------
    grid : dict
        Output of calculate_grid. Must contain vec_torq, vec_omega,
        const_mech_speed, grid_segments, curr_dq_grid.
    k_skip : int
        Downsampling factor forwarded to MachineData. 1 keeps every sample.

    Returns
    -------
    MachineData
    """
    torq = grid["vec_torq"]
    omega = grid["const_mech_speed"] * grid["vec_omega"]
    return MachineData(
        torq=torq,
        omega=omega,
        segments=grid["grid_segments"],
        curr_dq_grid=grid["curr_dq_grid"],
        k_skip=k_skip,
    )
