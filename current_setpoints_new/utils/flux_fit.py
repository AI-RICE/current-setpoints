"""
Identification of a single PM flux vector from measured voltage and torque.

PMSM5Phase previously carried two independently-identified flux vectors,
``flux_volt`` (back-EMF) and ``flux_torq`` (torque), even though both enter
the same underlying physics as one flux linkage ``lambda_pm``:

    u_dq = R_stat @ i + omega * J @ (L_stat @ i + lambda_pm)          (voltage)
    T    = i^T A i + 2k * (J @ lambda_pm) @ i,  A = k*(J@L+L@J^T)     (torque)

Both equations are LINEAR in ``lambda_pm`` given fixed R_stat, L_stat and
cross_coupling (J), so ``lambda_pm`` is identified by ordinary least squares
directly against measured voltage and torque, stacked into one design matrix.
"""
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
    """
    Identifies a single PM flux vector lambda_pm (dim,) by ordinary least
    squares, jointly fitting the voltage and torque equations against the
    measured operating points in csv_path.

    Each measured sample contributes dim voltage-residual rows and one
    torque-residual row to a single stacked least-squares system:

        voltage row:  omega * J @ lambda_pm = u_meas - R_stat@i - omega*(J@L_stat)@i
        torque row:   2k * (J^T @ i) @ lambda_pm = T_meas - i^T A i

    Args:
        csv_path: Path to the aggregated measurement CSV (columns: omega,
            id1, iq1, id3, iq3, ud1, uq1, ud3, uq3, torq).
        R_stat, L_stat, cross_coupling: Fixed machine parameters (dim, dim).
        n_phases, n_ppairs: Machine constants used in the torque coefficient
            k = n_phases * n_ppairs / 4.

    Returns:
        lambda_pm: ndarray (dim,) — least-squares flux vector.
    """
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
