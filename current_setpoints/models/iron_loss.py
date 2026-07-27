"""Iron-loss strategies — a pluggable component, kept separate from the
magnetics so any drive (analytic EC or flux-map) can hold one, exactly as
`FluxModel` is separated in `machines.py`. Generalises the scalar `k_v`/`k_h`
substitution model on `DriveModel`.

Both strategies below are validated at a single operating speed (the speed of
whatever sweep supplied their coefficients/table). The flux/B dependence is
sound; the FREQUENCY dependence is an assumption (single-speed data cannot
separate hysteresis ~f from eddy ~f^2), surfaced via `omega_ref`/`freq_exp`.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class IronLossModel(ABC):
    @abstractmethod
    def loss(self, omega: float, curr_dq: np.ndarray, flux: np.ndarray) -> float:
        """Iron loss [W] at electrical speed `omega`, given the operating-point
        current and its flux linkage (both supplied so a strategy can key on
        whichever it needs)."""


class SteinmetzIronLoss(IronLossModel):
    """P = k1|psi_1|^2 + k3|psi_3|^2 (loss ~ B^2 at fixed frequency). Default
    (`omega_ref=None`) returns the sweep-speed value; with a reference speed it
    scales by (omega/omega_ref)^freq_exp — an ASSUMED frequency law (eddy-like
    at freq_exp=2), UNVALIDATED on single-speed data."""

    def __init__(
        self,
        k1: float,
        k3: float,
        omega_ref: float | None = None,
        freq_exp: float = 2.0,
    ) -> None:
        self.k1, self.k3, self.omega_ref, self.freq_exp = k1, k3, omega_ref, freq_exp

    def loss(self, omega: float, curr_dq: np.ndarray, flux: np.ndarray) -> float:
        p = self.k1 * (flux[0] ** 2 + flux[1] ** 2) + self.k3 * (flux[2] ** 2 + flux[3] ** 2)
        if self.omega_ref:
            p *= (abs(omega) / self.omega_ref) ** self.freq_exp
        return float(p)


class CoreLossLUT(IronLossModel):
    """Interpolates a measured/FEM core-loss column directly over the dq
    currents. Faithful at the sweep speed; does not model frequency, so it
    warns if queried far from `omega_ref` (when known)."""

    def __init__(
        self,
        currents: np.ndarray,
        p_core: np.ndarray,
        omega_ref: float | None = None,
    ) -> None:
        self.omega_ref = omega_ref
        pts = np.vstack([np.zeros((1, 4)), np.asarray(currents, float)])
        val = np.concatenate([[0.0], np.asarray(p_core, float)])
        self._lin = LinearNDInterpolator(pts, val)
        self._near = NearestNDInterpolator(pts, val)

    def loss(self, omega: float, curr_dq: np.ndarray, flux: np.ndarray) -> float:
        if self.omega_ref and abs(abs(omega) - self.omega_ref) > 0.05 * self.omega_ref:
            warnings.warn(
                "CoreLossLUT queried away from the sweep speed; it does not model "
                "frequency — use SteinmetzIronLoss for other speeds.",
                stacklevel=2,
            )
        q = np.asarray(curr_dq, float).reshape(1, 4)
        p = self._lin(q)[0]
        if np.isnan(p):
            p = self._near(q)[0]
        return float(p)
