"""
Iron-loss proxies and efficiency map utilities.

These functions are post-hoc only — they evaluate energy components of a
converged static setpoint so that copper/iron trade-offs can be studied.
They are not intended to enter the optimization loop.

Three Bertotti iron-loss components, all computed in the time domain
(no FFT required):

* eddy_loss        — proportional to mean(|dlambda/dt|²), dominant at high speed.
* hysteresis_loss  — Steinmetz proxy on per-phase peak flux B_max; non-smooth.
* excess_loss      — Bertotti excess term, mean(|dB/dt|^1.5).

Coefficients k_e, k_h, k_x default to 1.0 for relative comparisons. Absolute
values require fitting against FEM or measurement data.

The flux model is minimal: lambda_dq = L_s @ curr_dq + psi_pm, projected to
surviving phases via the ForwardModel's angle-dependent Clarke matrices. For
static setpoints (constant dq), the phase-current variation with rotor angle
arises entirely from the rotating PM flux.

Additional utilities:
* low_pass_trajectory  — project a dq trajectory onto its first h_cutoff harmonics.
* deadbeat_tracking    — ZOH model of a finite-bandwidth deadbeat controller.
* add_efficiency_map   — append iron loss fields to a calculate_grid dict.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..models.forward_model import ForwardModel


# ─────────────────────────────────────────────────────────────────────────────
# Flux waveforms
# ─────────────────────────────────────────────────────────────────────────────

def phase_flux_waveforms(
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
    n_grid: int = 64,
) -> np.ndarray:
    """
    Per-phase flux-linkage waveforms on a coarse angle grid.

    Evaluates lambda_k(theta_n) = H_ph(theta_n) @ (L_s @ curr_dq + psi_pm)
    for n = 0..n_grid-1, where H_ph is the angle-dependent dq-to-phase
    projection for the surviving phases.

    Parameters
    ----------
    fwd : ForwardModel
    omega : float  — electrical speed [rad/s]
    curr_dq : ndarray (dim,)  — static dq setpoint
    n_grid : int  — number of coarse angle samples (default 64)

    Returns
    -------
    ndarray (n_surv_phases, n_grid)
    """
    drive = fwd.drive
    zeros = np.zeros(drive.dim)
    L_s = drive.inductance(omega, zeros)                   # (dim, dim)
    psi_pm = drive.flux.flux(omega, zeros)                  # (dim,)  — PM flux at zero current
    lambda_dq = L_s @ curr_dq + psi_pm                    # (dim,)  — constant for static curr_dq

    n_theta = fwd.vec_theta.size - 1
    theta_grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    idx_grid = (np.round(theta_grid / (2 * np.pi) * n_theta).astype(int)) % n_theta

    # stack: list of (n_surv,) → (n_surv, n_grid)
    return np.stack(
        [fwd.phase_map_at_theta(int(k)) @ lambda_dq for k in idx_grid],
        axis=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Iron-loss components
# ─────────────────────────────────────────────────────────────────────────────

def eddy_loss(
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
    *,
    k_e: float = 1.0,
    n_grid: int = 64,
) -> float:
    """
    Eddy-current loss proxy.

    P_eddy = k_e * mean_n( sum_k (omega * dlambda_k/dtheta|_n)^2 )

    Quadratic in the trajectory; dominant at high speed.
    """
    lambda_ph = phase_flux_waveforms(fwd, omega, curr_dq, n_grid)
    delta_theta = 2 * np.pi / n_grid
    dlambda_dth = (np.roll(lambda_ph, -1, axis=1) - lambda_ph) / delta_theta
    dlambda_dt = omega * dlambda_dth
    return float(k_e * np.mean(np.sum(dlambda_dt ** 2, axis=0)))


def hysteresis_loss(
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
    *,
    k_h: float = 1.0,
    beta: float = 1.8,
    n_grid: int = 64,
) -> float:
    """
    Steinmetz hysteresis-loss proxy.

    P_hyst = k_h * (omega / 2pi) * sum_k B_max_k^beta

    B_max_k = max_n |lambda_k(theta_n)|. Non-smooth in the trajectory;
    do not use inside an optimizer.
    """
    lambda_ph = phase_flux_waveforms(fwd, omega, curr_dq, n_grid)
    b_max = np.max(np.abs(lambda_ph), axis=1)
    return float(k_h * (omega / (2 * np.pi)) * np.sum(b_max ** beta))


def excess_loss(
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
    *,
    k_x: float = 1.0,
    n_grid: int = 64,
) -> float:
    """
    Bertotti excess-loss proxy.

    P_excess = k_x * mean_n( sum_k |omega * dlambda_k/dtheta|^1.5 )

    Sub-quadratic dependence; weights moderately-fast flux variations more
    than the pure eddy term.
    """
    lambda_ph = phase_flux_waveforms(fwd, omega, curr_dq, n_grid)
    delta_theta = 2 * np.pi / n_grid
    dlambda_dt = omega * (np.roll(lambda_ph, -1, axis=1) - lambda_ph) / delta_theta
    return float(k_x * np.mean(np.sum(np.abs(dlambda_dt) ** 1.5, axis=0)))


def iron_loss(
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
    *,
    k_e: float = 1.0,
    k_h: float = 1.0,
    k_x: float = 0.0,
    beta: float = 1.8,
    n_grid: int = 64,
) -> dict[str, float]:
    """
    All three Bertotti iron-loss components at one operating point.

    Shares the phase_flux_waveforms call across all three terms for efficiency.

    Parameters
    ----------
    k_e, k_h, k_x : float
        Eddy, hysteresis, excess coefficients (k_x=0 disables excess term).
    beta : float  — Steinmetz exponent; typically 1.6–2.0.
    n_grid : int  — coarse angle resolution for flux waveform sampling.

    Returns
    -------
    dict with keys "eddy", "hysteresis", "excess", "total"
    """
    lambda_ph = phase_flux_waveforms(fwd, omega, curr_dq, n_grid)
    delta_theta = 2 * np.pi / n_grid
    dlambda_dth = (np.roll(lambda_ph, -1, axis=1) - lambda_ph) / delta_theta
    dlambda_dt = omega * dlambda_dth

    p_e = float(k_e * np.mean(np.sum(dlambda_dt ** 2, axis=0)))
    b_max = np.max(np.abs(lambda_ph), axis=1)
    p_h = float(k_h * (omega / (2 * np.pi)) * np.sum(b_max ** beta))
    p_x = float(k_x * np.mean(np.sum(np.abs(dlambda_dt) ** 1.5, axis=0))) if k_x != 0.0 else 0.0

    return {"eddy": p_e, "hysteresis": p_h, "excess": p_x, "total": p_e + p_h + p_x}


# ─────────────────────────────────────────────────────────────────────────────
# Controller simulation utilities (trajectory-level)
# ─────────────────────────────────────────────────────────────────────────────

def low_pass_trajectory(curr_dq_grid: np.ndarray, h_cutoff: int) -> np.ndarray:
    """
    Project a per-angle dq trajectory onto its first h_cutoff Fourier harmonics.

    Models a finite-bandwidth controller that cannot track harmonics above
    h_cutoff. Use the returned array as the iron-loss reference trajectory.

    Parameters
    ----------
    curr_dq_grid : ndarray (n_grid, dim)
    h_cutoff : int  — highest harmonic to retain (inclusive)

    Returns
    -------
    ndarray (n_grid, dim)
    """
    N = curr_dq_grid.shape[0]
    coeffs = np.fft.rfft(curr_dq_grid, axis=0)
    coeffs[h_cutoff + 1:] = 0.0
    return np.fft.irfft(coeffs, n=N, axis=0)


def deadbeat_tracking(
    curr_dq_grid: np.ndarray,
    n_samples: int,
    delay: int = 0,
) -> np.ndarray:
    """
    Zero-order-hold model of a deadbeat current controller.

    Samples the reference at n_samples uniformly spaced angles per electrical
    period and holds each sample constant until the next update. Produces sharp
    step discontinuities (unlike low_pass_trajectory's smooth attenuation).

    Setting delay=1 models one-step computational latency.

    Parameters
    ----------
    curr_dq_grid : ndarray (n_grid, dim)
    n_samples : int  — controller update rate per electrical period
    delay : int      — one-step latency (0 = ideal)

    Returns
    -------
    ndarray (n_grid, dim)
    """
    X = np.asarray(curr_dq_grid)
    N = X.shape[0]
    if n_samples >= N or n_samples <= 0:
        return X.copy()
    sample_idx = np.linspace(0, N, n_samples + 1, endpoint=True).astype(int)[:-1]
    samples = X[sample_idx]
    if delay:
        samples = np.roll(samples, delay, axis=0)
    rep_counts = np.diff(np.concatenate([sample_idx, [N]]))
    return np.repeat(samples, rep_counts, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Grid-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def add_efficiency_map(
    grid: dict[str, Any],
    fwd: ForwardModel,
    *,
    k_e: float = 1.0,
    k_h: float = 1.0,
    k_x: float = 0.0,
    beta: float = 1.8,
    n_grid: int = 64,
) -> dict[str, Any]:
    """
    Append iron loss fields to a calculate_grid output dict.

    Evaluates losses at every valid (non-NaN) cell.

    New keys added to the returned dict (all shape n_torq × n_omega):
        grid_loss_eddy
        grid_loss_hyst
        grid_loss_excess
        grid_loss_iron    — eddy + hyst + excess

    Parameters
    ----------
    grid : dict  — output of calculate_grid
    fwd  : ForwardModel
    k_e, k_h, k_x, beta, n_grid : see iron_loss()

    Returns
    -------
    dict — shallow copy of grid with loss fields added
    """
    out: dict[str, Any] = dict(grid)

    curr_dq_grid = grid["curr_dq_grid"]   # (dim, n_torq, n_omega)
    vec_omega = grid["vec_omega"]
    dim, n_torq, n_omega = curr_dq_grid.shape

    shape = (n_torq, n_omega)
    loss_eddy = np.full(shape, np.nan)
    loss_hyst = np.full(shape, np.nan)
    loss_exc = np.full(shape, np.nan)
    loss_iron_arr = np.full(shape, np.nan)

    for idx_omega in range(n_omega):
        omega = float(vec_omega[idx_omega])
        for idx_torq in range(n_torq):
            curr_dq = curr_dq_grid[:, idx_torq, idx_omega]
            if np.any(np.isnan(curr_dq)):
                continue

            il = iron_loss(
                fwd, omega, curr_dq,
                k_e=k_e, k_h=k_h, k_x=k_x, beta=beta, n_grid=n_grid,
            )
            loss_eddy[idx_torq, idx_omega] = il["eddy"]
            loss_hyst[idx_torq, idx_omega] = il["hysteresis"]
            loss_exc[idx_torq, idx_omega] = il["excess"]
            loss_iron_arr[idx_torq, idx_omega] = il["total"]

    out["grid_loss_eddy"] = loss_eddy
    out["grid_loss_hyst"] = loss_hyst
    out["grid_loss_excess"] = loss_exc
    out["grid_loss_iron"] = loss_iron_arr
    return out


__all__ = [
    "phase_flux_waveforms",
    "eddy_loss",
    "hysteresis_loss",
    "excess_loss",
    "iron_loss",
    "low_pass_trajectory",
    "deadbeat_tracking",
    "add_efficiency_map",
]
