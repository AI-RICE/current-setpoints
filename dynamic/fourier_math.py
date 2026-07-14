"""
Fourier-basis helpers for the periodic dynamic problem's diagnostic scripts.

``current_setpoints.optimization.optimizer.FourierOptimizer`` already
implements the full periodic Fourier-ansatz solve (ported from this module
during the library unification) -- its equivalents of the functions below
are private (``_fourier_design``, ``_n_basis``, ...) since they're solve-
internal. The diagnostics in ``experiments/fourier/diag_*.py`` need the pure
math directly (design matrices, basis size, the quadratic-torque fit), not
the solve, so those three stay here as the public entry point.
"""

from __future__ import annotations

import numpy as np

from current_setpoints.models.machines import DriveModel


def _positive_harmonics(harmonics: tuple[int, ...]) -> list[int]:
    return [int(h) for h in sorted(set(harmonics)) if int(h) > 0]


def n_basis(harmonics: tuple[int, ...]) -> int:
    """Number of Fourier basis functions (1 for DC + 2 per positive harmonic)."""
    has_dc = 0 in set(int(h) for h in harmonics)
    return (1 if has_dc else 0) + 2 * len(_positive_harmonics(harmonics))


def fourier_design(theta: np.ndarray, harmonics: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """
    Design matrices ``(Phi, dPhi)`` of shape ``(len(theta), n_basis)``.

    Columns are ordered ``[1, cos(h1 t), sin(h1 t), cos(h2 t), sin(h2 t), ...]``
    (the leading constant column is present iff ``0 in harmonics``).
    ``dPhi`` is the elementwise derivative ``d/dtheta``.
    """
    theta = np.asarray(theta, dtype=float)
    cols: list[np.ndarray] = []
    dcols: list[np.ndarray] = []
    if 0 in set(int(h) for h in harmonics):
        cols.append(np.ones_like(theta))
        dcols.append(np.zeros_like(theta))
    for h in _positive_harmonics(harmonics):
        cols.append(np.cos(h * theta))
        dcols.append(-h * np.sin(h * theta))
        cols.append(np.sin(h * theta))
        dcols.append(h * np.cos(h * theta))
    return np.stack(cols, axis=1), np.stack(dcols, axis=1)


def extract_quadratic_torque(drive: DriveModel, omega: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Recover ``(A, b, c)`` such that ``T(i) = i^T A i + b^T i + c`` exactly.
    Delegates to ``DriveModel.torque_quadratic``, which does the same
    least-squares fit against ``drive.torque`` -- a fit that recovers the
    same coefficients up to floating-point precision regardless of the
    (already-exact, quadratic) function's sample points.
    """
    return drive.torque_quadratic(omega)


__all__ = ["fourier_design", "n_basis", "extract_quadratic_torque"]
