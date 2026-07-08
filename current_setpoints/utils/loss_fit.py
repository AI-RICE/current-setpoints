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

    def torque(self, omega: float, curr_dq: np.ndarray) -> float: ...


def substitution_loss_features(
    omega: Any,
    volt_dq: Any,
    n_ppairs: int,
) -> tuple[Any, Any]:
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
    return train_test_split(df, test_size=test_size, random_state=random_state)


def _baseline_torque(df: pd.DataFrame, drive: _DriveModel) -> np.ndarray:
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
