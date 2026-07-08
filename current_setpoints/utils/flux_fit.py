from __future__ import annotations

import numpy as np
import pandas as pd

CURR_COLS: tuple[str, ...] = ("id1", "iq1", "id3", "iq3")
VOLT_COLS: tuple[str, ...] = ("ud1", "uq1", "ud3", "uq3")
TORQ_COL: str = "torq"
OMEGA_COL: str = "omega"


def fit_pm_flux(
    csv_path: str,
    R_stat: np.ndarray,
    L_stat: np.ndarray,
    cross_coupling: np.ndarray,
    n_phases: int,
    n_ppairs: int,
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    J = cross_coupling
    dim = J.shape[0]
    k = n_phases * n_ppairs / 4.0

    omega = df[OMEGA_COL].to_numpy(dtype=np.float64)
    curr = df[list(CURR_COLS)].to_numpy(dtype=np.float64)
    volt = df[list(VOLT_COLS)].to_numpy(dtype=np.float64)
    torq = df[TORQ_COL].to_numpy(dtype=np.float64)
    n = len(df)

    A = k * (J @ L_stat + L_stat @ J.T)
    rows = np.empty((n * (dim + 1), dim))
    targets = np.empty(n * (dim + 1))
    for s in range(n):
        w, i = omega[s], curr[s]
        rows[s * dim : (s + 1) * dim, :] = w * J
        targets[s * dim : (s + 1) * dim] = volt[s] - R_stat @ i - w * (J @ L_stat) @ i
    base = n * dim
    for s in range(n):
        i = curr[s]
        rows[base + s, :] = 2.0 * k * (J.T @ i)
        targets[base + s] = torq[s] - i @ A @ i

    lambda_pm, *_ = np.linalg.lstsq(rows, targets, rcond=None)
    return lambda_pm
