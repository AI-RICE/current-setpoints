"""
Fit and evaluate the substitution iron-loss torque model (Sub1).

The model adds an iron-loss correction M_Fe to the analytical baseline torque:

    M_Fe = k_v * (p² * omega_m / (4π²)) * (U1² + 9 U3²)   # eddy-type
         + k_h * (p / (2π))             * (U1² + 3 U3²)    # hysteresis-type

with U1² = U_d1² + U_q1², U3² = U_d3² + U_q3², p the pole-pair count, and
omega_m = omega_elec / p the mechanical speed. The dq voltages are measured
values from the aggregated dataset (columns ud1, uq1, ud3, uq3).

M_Fe is linear in (k_v, k_h), so coefficients are identified by ordinary least
squares against the NTM residual: target = T_measured - T_base. The test split
is reproduced identically to the NTM's (test_size=0.15, random_state=42) so
the resulting RMSE is directly comparable to the NTM's test RMSE.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE: float = 0.15
RANDOM_STATE: int = 42

CURR_COLS: tuple[str, ...] = ("id1", "iq1", "id3", "iq3")
VOLT_COLS: tuple[str, ...] = ("ud1", "uq1", "ud3", "uq3")
TORQ_COL: str = "torq"
OMEGA_COL: str = "omega"


class _DriveModel(Protocol):
    """Minimal interface required of the injected baseline torque model."""

    def torque(self, omega: float, curr_dq: np.ndarray) -> float: ...


def substitution_loss_features(
    omega: Any,
    volt_dq: Any,
    n_ppairs: int,
) -> tuple[Any, Any]:
    """
    Build the two per-coefficient basis terms (f_v, f_h) of M_Fe.

    M_Fe = k_v * f_v + k_h * f_h. Works for a single operating point
    (scalar omega, length-4 volt_dq) or vectorised over rows (omega shape (N,),
    volt_dq shape (N, 4)).

    Parameters
    ----------
    omega    : float or (N,)   — electrical speed [rad/s]
    volt_dq  : (4,) or (N, 4) — dq voltages [U_d1, U_q1, U_d3, U_q3]
    n_ppairs : int             — pole-pair count p

    Returns
    -------
    (f_v, f_h) — eddy-type and hysteresis-type basis terms, scalar or (N,)
    """
    omega = np.asarray(omega, dtype=np.float64)
    volt_dq = np.asarray(volt_dq, dtype=np.float64)

    if volt_dq.ndim == 1:
        ud1, uq1, ud3, uq3 = volt_dq
    else:
        ud1, uq1, ud3, uq3 = volt_dq[:, 0], volt_dq[:, 1], volt_dq[:, 2], volt_dq[:, 3]

    u1_sq = ud1 ** 2 + uq1 ** 2
    u3_sq = ud3 ** 2 + uq3 ** 2
    p = float(n_ppairs)
    omega_m = omega / p

    f_v = (p ** 2 * omega_m / (4.0 * np.pi ** 2)) * (u1_sq + 9.0 * u3_sq)
    f_h = (p / (2.0 * np.pi)) * (u1_sq + 3.0 * u3_sq)
    return f_v, f_h


def split_like_neural(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/test split with the same row partition as the NTM holdout.

    train_test_split partitions by row index from (n_samples, test_size,
    random_state) alone, so splitting here assigns the same operating points
    to train/test as the NTM, provided rows are in the same order.

    Parameters
    ----------
    df           : pd.DataFrame — aggregated dataset, one row per operating point
    test_size    : float        — held-out fraction (leave at 0.15 to match NTM)
    random_state : int          — split seed (leave at 42 to match NTM)

    Returns
    -------
    (df_train, df_test)
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)


def _baseline_torque(df: pd.DataFrame, drive: _DriveModel) -> np.ndarray:
    """Analytical baseline torque T_base per row."""
    omega = df[OMEGA_COL].to_numpy(dtype=np.float64)
    curr = df[list(CURR_COLS)].to_numpy(dtype=np.float64)
    return np.array(
        [drive.torque(float(w), c) for w, c in zip(omega, curr)],
        dtype=np.float64,
    )


def fit_substitution_loss(
    df_train: pd.DataFrame,
    drive: _DriveModel,
    n_ppairs: int,
) -> dict[str, Any]:
    """
    Identify k_v, k_h by ordinary least squares on the residual torque.

    Fits against target = T_measured - T_base using measured dq voltages, so
    the fitted coefficients minimise (T_base + M_Fe) - T_measured.

    Parameters
    ----------
    df_train : pd.DataFrame — training rows (85% complement of the holdout);
                              must contain omega, current columns, voltage columns, torq
    drive    : _DriveModel  — baseline model exposing torque(omega, curr_dq)
    n_ppairs : int          — pole-pair count p

    Returns
    -------
    dict with k_v, k_h, coef (ndarray), rmse [Nm], r2, n_points
    """
    omega = df_train[OMEGA_COL].to_numpy(dtype=np.float64)
    volt = df_train[list(VOLT_COLS)].to_numpy(dtype=np.float64)
    t_meas = df_train[TORQ_COL].to_numpy(dtype=np.float64)

    t_base = _baseline_torque(df_train, drive)
    target = t_meas - t_base

    f_v, f_h = substitution_loss_features(omega, volt, n_ppairs)
    phi = np.column_stack([f_v, f_h])
    coef, *_ = np.linalg.lstsq(phi, target, rcond=None)

    pred = phi @ coef
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "k_v": float(coef[0]),
        "k_h": float(coef[1]),
        "coef": coef,
        "rmse": rmse,
        "r2": r2,
        "n_points": int(len(target)),
    }


def substitution_loss_rmse(
    coeffs: dict[str, Any],
    df: pd.DataFrame,
    drive: _DriveModel,
    n_ppairs: int,
) -> float:
    """
    RMSE of the full substitution model (T_base + M_Fe) vs measured torque.

    On the holdout split this is directly comparable to the NTM's test RMSE.

    Parameters
    ----------
    coeffs   : dict         — output of fit_substitution_loss (uses k_v, k_h or coef)
    df       : pd.DataFrame — rows to evaluate (e.g. the test split)
    drive    : _DriveModel  — baseline model exposing torque(omega, curr_dq)
    n_ppairs : int          — pole-pair count p

    Returns
    -------
    float — RMSE [Nm]
    """
    omega = df[OMEGA_COL].to_numpy(dtype=np.float64)
    volt = df[list(VOLT_COLS)].to_numpy(dtype=np.float64)
    t_meas = df[TORQ_COL].to_numpy(dtype=np.float64)

    t_base = _baseline_torque(df, drive)
    f_v, f_h = substitution_loss_features(omega, volt, n_ppairs)

    coef = coeffs.get("coef")
    if coef is None:
        coef = np.array([coeffs["k_v"], coeffs["k_h"]], dtype=np.float64)
    m_fe = np.column_stack([f_v, f_h]) @ np.asarray(coef, dtype=np.float64)

    return float(np.sqrt(np.mean((t_base + m_fe - t_meas) ** 2)))
