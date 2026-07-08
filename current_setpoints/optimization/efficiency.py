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
    N = curr_dq_grid.shape[0]
    coeffs = np.fft.rfft(curr_dq_grid, axis=0)
    coeffs[h_cutoff + 1:] = 0.0
    return np.fft.irfft(coeffs, n=N, axis=0)


def deadbeat_tracking(
    curr_dq_grid: np.ndarray,
    n_samples: int,
    delay: int = 0,
) -> np.ndarray:
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
