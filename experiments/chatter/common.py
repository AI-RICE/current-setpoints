"""
Shared utilities for the chatter analysis experiments E01--E05.

* ``make_setup``: builds the canonical operating point used across all
  five experiments (n_mech = 1400 rpm, T* = 4 Nm on IEEEMachine2), so
  the comparisons are like-for-like.
* ``joule_loss``: per-period average of ``|x|^2`` (the optimiser's own
  objective, scaled).
* ``fourier_spectrum``: per-component Fourier magnitudes on the coarse
  grid.
* ``low_pass_project`` / ``chatter_amplitude``: separates the "smooth"
  low-harmonic component of the trajectory from the chatter residual.
* ``figs_dir``: ensures the ``experiments/chatter/figs/`` folder exists
  and returns its absolute path.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from current_setpoints.optimization import ModelAnalytical
from current_setpoints.optimization.models import BaseTorqueModel
from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2
from current_setpoints.simulation import Transform


@dataclass
class Setup:
    machine: IEEEMachine2
    flux: Flux_IEEEMachine2
    transform: Transform
    model: BaseTorqueModel
    omega_el: float
    torq_target: float


# 5 operating points selected from the Phase-0 survey
# (see experiments/iron/e06_operating_point_survey.py).
# Format: (T_star_Nm, n_mech_rpm, short_tag, description)
OPERATING_POINTS: list[tuple[float, float, str, str]] = [
    (1.0, 100.0, "p1_free", "free baseline (low J, low iron)"),
    (3.0, 800.0, "p2_typical", "median V-residual (typical operating point)"),
    (7.0, 1200.0, "p3_curr_edge", "high-J / current-edge (R0 barely infeasible)"),
    (1.0, 1500.0, "p4_iron_heavy", "iron-dominated extreme (P_eddy/J ~ 23)"),
    (4.5, 1500.0, "p5_corner", "max V-residual corner (active-set most needed)"),
]


def make_setup(n_mech_rpm: float = 1500.0, torq_target: float = 4.5, n_theta: int = 700) -> Setup:
    """
    Canonical operating point. Defaults to point p5 from the survey
    -- the "max V-residual corner" where active-set, iron loss, and
    current limits all interact.
    """
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    flux = Flux_IEEEMachine2()
    transform = Transform(machine=machine, flux=flux, add_volt_0=False, n_theta=n_theta)
    model = ModelAnalytical(machine=machine, flux=flux)
    omega_el = n_mech_rpm * (np.pi / 30.0) * machine.n_ppairs
    return Setup(
        machine=machine,
        flux=flux,
        transform=transform,
        model=model,
        omega_el=omega_el,
        torq_target=torq_target,
    )


def joule_loss(curr_dq_grid: np.ndarray) -> float:
    """Per-period average ``(1/N) * sum_n |x_n|^2`` (same scaling as the
    optimiser's objective divided by ``N``)."""
    return float(np.mean(np.sum(curr_dq_grid**2, axis=1)))


def fourier_spectrum(curr_dq_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided amplitude spectrum on the periodic coarse grid.

    Returns ``(h, mag)`` where ``h`` are harmonic numbers
    ``0, 1, ..., N//2`` and ``mag`` has shape ``(len(h), dim)`` giving
    the modulus of the FFT per d-q component, normalised so that the
    fundamental sinusoid of amplitude 1 reads as 1.
    """
    N = curr_dq_grid.shape[0]
    coeffs = np.fft.rfft(curr_dq_grid, axis=0) / N
    h = np.arange(coeffs.shape[0])
    mag = np.abs(coeffs)
    mag[1:-1] *= 2.0  # one-sided scaling for h > 0 (except Nyquist if even N)
    return h, mag


def low_pass_project(curr_dq_grid: np.ndarray, h_max: int = 7) -> np.ndarray:
    """
    Reconstruct the trajectory keeping only harmonics ``0, 1, ..., h_max``.
    Same shape as ``curr_dq_grid``.
    """
    N = curr_dq_grid.shape[0]
    coeffs = np.fft.rfft(curr_dq_grid, axis=0)
    coeffs[h_max + 1 :] = 0.0
    return np.fft.irfft(coeffs, n=N, axis=0)


def chatter_amplitude(curr_dq_grid: np.ndarray, h_max: int = 7) -> float:
    """
    Relative chatter amplitude: ``||x - LP(x)|| / ||x||`` where ``LP(.)``
    keeps harmonics up to ``h_max`` (default 7). Dimensionless, in
    [0, 1]; small values mean the trajectory is well-described by the
    low-harmonic projection.
    """
    smooth = low_pass_project(curr_dq_grid, h_max=h_max)
    num = float(np.linalg.norm(curr_dq_grid - smooth))
    den = float(np.linalg.norm(curr_dq_grid))
    return num / max(den, 1e-12)


def tail_mass(curr_dq_grid: np.ndarray, h_max: int = 7) -> float:
    """
    Fraction of total Fourier energy above harmonic ``h_max``. Equivalent
    metric to ``chatter_amplitude`` but expressed in energy rather than
    amplitude.
    """
    _, mag = fourier_spectrum(curr_dq_grid)
    e_total = float(np.sum(mag**2))
    e_tail = float(np.sum(mag[h_max + 1 :] ** 2))
    return e_tail / max(e_total, 1e-24)


def figs_dir() -> str:
    """Absolute path to ``experiments/chatter/figs/``, created if missing."""
    d = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(d, exist_ok=True)
    return d


__all__ = [
    "Setup",
    "make_setup",
    "OPERATING_POINTS",
    "joule_loss",
    "fourier_spectrum",
    "low_pass_project",
    "chatter_amplitude",
    "tail_mass",
    "figs_dir",
]
