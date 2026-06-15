"""
Fit and evaluate the substitution iron-loss torque model.

The model adds an iron-loss correction ``M_Fe`` to the analytical baseline
torque ``T_base``:

    M_Fe = k_v * (p^2 * omega_m / (4 pi^2)) * (U1^2 + 9 U3^2)   # eddy-type
         + k_h * (p / (2 pi))              * (U1^2 + 3 U3^2)    # hysteresis-type

with ``U1^2 = U_d1^2 + U_q1^2``, ``U3^2 = U_d3^2 + U_q3^2``, ``p`` the pole-pair
count, and ``omega_m = omega_elec / p`` the mechanical speed. The dq voltages
are the *measured* values from the aggregated dataset (columns
``ud1, uq1, ud3, uq3``).

``M_Fe`` is LINEAR in ``(k_v, k_h)``, so the coefficients are identified by
ordinary least squares against the same residual the neural torque model (NTM)
is trained on, ``target = T_measured - T_base``. Because the reported error is
the RMSE of ``(T_base + M_Fe) - T_measured`` on the held-out test split -- and
the test split is reproduced identically to the NTM's (``test_size=0.15``,
``random_state=42``) -- this RMSE is directly comparable to the NTM's test RMSE.

The analytical baseline model is passed in (not imported) to avoid a
utils -> optimization import cycle.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Holdout split shared with the neural torque model. Leave untouched so the
# substitution baseline is evaluated on the identical test points.
TEST_SIZE: float = 0.15
RANDOM_STATE: int = 42

# Column layout in the aggregated dataset.
CURR_COLS: tuple[str, ...] = ("id1", "iq1", "id3", "iq3")
VOLT_COLS: tuple[str, ...] = ("ud1", "uq1", "ud3", "uq3")
TORQ_COL: str = "torq"
OMEGA_COL: str = "omega"


class _AnalyticalModel(Protocol):
    """Minimal interface required of the injected baseline torque model."""

    def calculate_torque(self, omega: float, curr_dq: np.ndarray) -> float: ...


def substitution_loss_features(
    omega: Any, volt_dq: Any, n_ppairs: int
) -> tuple[Any, Any]:
    """
    Builds the two per-coefficient basis terms ``(f_v, f_h)`` of ``M_Fe``.

    ``M_Fe = k_v * f_v + k_h * f_h``. Works for a single operating point
    (scalar ``omega``, length-4 ``volt_dq``) or vectorised over rows
    (``omega`` shape ``(N,)``, ``volt_dq`` shape ``(N, 4)``).

    Args:
        omega: Electrical speed [rad/s], scalar or shape ``(N,)``.
        volt_dq: dq voltages ordered ``[U_d1, U_q1, U_d3, U_q3]``, shape
            ``(4,)`` or ``(N, 4)``.
        n_ppairs: Pole-pair count ``p``. Mechanical speed is ``omega / p``.

    Returns:
        ``(f_v, f_h)``: the eddy-type and hysteresis-type basis terms, each a
        scalar or shape ``(N,)`` matching the input.
    """
    omega = np.asarray(omega, dtype=np.float64)
    volt_dq = np.asarray(volt_dq, dtype=np.float64)

    if volt_dq.ndim == 1:
        ud1, uq1, ud3, uq3 = volt_dq
    else:
        ud1, uq1, ud3, uq3 = volt_dq[:, 0], volt_dq[:, 1], volt_dq[:, 2], volt_dq[:, 3]

    u1_sq = ud1**2 + uq1**2
    u3_sq = ud3**2 + uq3**2

    p = float(n_ppairs)
    omega_m = omega / p

    f_v = (p**2 * omega_m / (4.0 * np.pi**2)) * (u1_sq + 9.0 * u3_sq)
    f_h = (p / (2.0 * np.pi)) * (u1_sq + 3.0 * u3_sq)
    return f_v, f_h


def substitution2_loss_features(
    omega: Any, volt_dq: Any, n_ppairs: int
) -> tuple[Any, Any]:
    """
    Alternative ``M_Fe`` basis (model 2):

        M_Fe = k_v / omega_m              * (U1^2 + U3^2)
             + k_h * 2 pi / (p * omega_m^2) * (U1^2 + (1/3) U3^2)

    with ``omega_m = omega / p``. Same shape/broadcasting rules and arguments
    as ``substitution_loss_features``.

    Note: both basis terms diverge as ``omega_m -> 0``; this model is only
    valid away from standstill (the dataset contains no zero-speed points).
    """
    omega = np.asarray(omega, dtype=np.float64)
    volt_dq = np.asarray(volt_dq, dtype=np.float64)

    if volt_dq.ndim == 1:
        ud1, uq1, ud3, uq3 = volt_dq
    else:
        ud1, uq1, ud3, uq3 = volt_dq[:, 0], volt_dq[:, 1], volt_dq[:, 2], volt_dq[:, 3]

    u1_sq = ud1**2 + uq1**2
    u3_sq = ud3**2 + uq3**2

    p = float(n_ppairs)
    omega_m = omega / p

    f_v = (u1_sq + u3_sq) / omega_m
    f_h = (2.0 * np.pi / (p * omega_m**2)) * (u1_sq + u3_sq / 3.0)
    return f_v, f_h


def substitution_loss_features_excess(
    omega: Any, volt_dq: Any, n_ppairs: int
) -> tuple[Any, Any, Any]:
    """
    Three-term Bertotti basis for substitution 1 (B == U): the two terms of
    ``substitution_loss_features`` plus the excess (anomalous) term

        f_e = (p^1.5 * sqrt(omega_m) / (2 pi)^1.5) * (U1^1.5 + 3*sqrt(3) U3^1.5)

    where ``U1^1.5 = (U_d1^2 + U_q1^2)^0.75`` and the ``3*sqrt(3) = 3^1.5`` factor
    is the 3rd-harmonic ``(3 f)^1.5`` scaling. Returns ``(f_v, f_h, f_e)``.
    """
    omega = np.asarray(omega, dtype=np.float64)
    volt_dq = np.asarray(volt_dq, dtype=np.float64)

    if volt_dq.ndim == 1:
        ud1, uq1, ud3, uq3 = volt_dq
    else:
        ud1, uq1, ud3, uq3 = volt_dq[:, 0], volt_dq[:, 1], volt_dq[:, 2], volt_dq[:, 3]

    u1_sq = ud1**2 + uq1**2
    u3_sq = ud3**2 + uq3**2

    p = float(n_ppairs)
    omega_m = omega / p

    f_v = (p**2 * omega_m / (4.0 * np.pi**2)) * (u1_sq + 9.0 * u3_sq)
    f_h = (p / (2.0 * np.pi)) * (u1_sq + 3.0 * u3_sq)
    f_e = (p**1.5 * np.sqrt(omega_m) / (2.0 * np.pi) ** 1.5) * (
        u1_sq**0.75 + 3.0 * np.sqrt(3.0) * u3_sq**0.75
    )
    return f_v, f_h, f_e


def substitution2_loss_features_excess(
    omega: Any, volt_dq: Any, n_ppairs: int
) -> tuple[Any, Any, Any]:
    """
    Three-term Bertotti basis for substitution 2 (B == U/f): the two terms of
    ``substitution2_loss_features`` plus the excess (anomalous) term

        f_e = (U1^1.5 + U3^1.5) / omega_m

    (the harmonic weight is 1 because frequency cancels in ``f^1.5 (U/f)^1.5``).
    Diverges at standstill. Returns ``(f_v, f_h, f_e)``.
    """
    omega = np.asarray(omega, dtype=np.float64)
    volt_dq = np.asarray(volt_dq, dtype=np.float64)

    if volt_dq.ndim == 1:
        ud1, uq1, ud3, uq3 = volt_dq
    else:
        ud1, uq1, ud3, uq3 = volt_dq[:, 0], volt_dq[:, 1], volt_dq[:, 2], volt_dq[:, 3]

    u1_sq = ud1**2 + uq1**2
    u3_sq = ud3**2 + uq3**2

    p = float(n_ppairs)
    omega_m = omega / p

    f_v = (u1_sq + u3_sq) / omega_m
    f_h = (2.0 * np.pi / (p * omega_m**2)) * (u1_sq + u3_sq / 3.0)
    f_e = (u1_sq**0.75 + u3_sq**0.75) / omega_m
    return f_v, f_h, f_e


def split_like_neural(
    df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/test split with the same row partition as the NTM holdout.

    ``train_test_split`` partitions by row index from ``(n_samples, test_size,
    random_state)`` alone, so splitting the dataframe here assigns the same
    operating points to train/test as the NTM's ``(X, residual)`` split, as
    long as the defaults match and the rows are in the same order the NTM was
    fed (i.e. the dataframe loaded by ``load_aggregated_csv_data``).

    Args:
        df: Aggregated dataset (one row per operating point).
        test_size: Held-out fraction (leave at 0.15 to match the NTM).
        random_state: Split seed (leave at 42 to match the NTM).

    Returns:
        ``(df_train, df_test)``.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)


def _baseline_torque(df: pd.DataFrame, analytical_model: _AnalyticalModel) -> np.ndarray:
    """Analytical baseline torque ``T_base`` per row, matching the NTM baseline."""
    omega = df[OMEGA_COL].to_numpy(dtype=np.float64)
    curr = df[list(CURR_COLS)].to_numpy(dtype=np.float64)
    return np.array(
        [analytical_model.calculate_torque(float(w), c) for w, c in zip(omega, curr)],
        dtype=np.float64,
    )


def fit_substitution_loss(
    df_train: pd.DataFrame,
    analytical_model: _AnalyticalModel,
    n_ppairs: int,
    features_fn: Any = substitution_loss_features,
) -> dict[str, Any]:
    """
    Identifies ``k_v, k_h`` by ordinary least squares on the residual torque.

    Fits against ``target = T_measured - T_base`` using the measured dq
    voltages, so the fitted coefficients minimise the full-model torque error
    ``(T_base + M_Fe) - T_measured``.

    Args:
        df_train: Training rows (the 85% complement of the holdout). Must
            contain ``omega``, the current columns, the voltage columns, and
            ``torq``.
        analytical_model: Baseline model exposing
            ``calculate_torque(omega, curr_dq) -> float``.
        n_ppairs: Pole-pair count ``p``.
        features_fn: Basis builder ``(omega, volt_dq, n_ppairs) -> (f_v, f_h)``;
            selects the loss model (e.g. ``substitution_loss_features`` or
            ``substitution2_loss_features``).

    Returns:
        Dict with ``k_v``, ``k_h``, training ``rmse`` [Nm] (equal to the
        residual RMSE, hence comparable to the NTM), training ``r2``, and
        ``n_points``.
    """
    omega = df_train[OMEGA_COL].to_numpy(dtype=np.float64)
    volt = df_train[list(VOLT_COLS)].to_numpy(dtype=np.float64)
    t_meas = df_train[TORQ_COL].to_numpy(dtype=np.float64)

    t_base = _baseline_torque(df_train, analytical_model)
    target = t_meas - t_base  # residual the loss term must explain

    feats = features_fn(omega, volt, n_ppairs)
    phi = np.column_stack(feats)

    coef, *_ = np.linalg.lstsq(phi, target, rcond=None)

    pred = phi @ coef
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Name the first coefficients k_v, k_h, k_e (eddy, hysteresis, excess);
    # ``coef`` holds the full vector so evaluation is agnostic to the count.
    names = ("k_v", "k_h", "k_e", "k_4", "k_5")[: len(coef)]
    result: dict[str, Any] = {name: float(c) for name, c in zip(names, coef)}
    result["coef"] = coef
    result["rmse"] = rmse
    result["r2"] = r2
    result["n_points"] = int(len(target))
    return result


def substitution_loss_rmse(
    coeffs: dict[str, float],
    df: pd.DataFrame,
    analytical_model: _AnalyticalModel,
    n_ppairs: int,
    features_fn: Any = substitution_loss_features,
) -> float:
    """
    RMSE of the full substitution model ``(T_base + M_Fe)`` vs measured torque.

    Equal to the residual RMSE ``M_Fe - (T_measured - T_base)``, so on the
    holdout split it is directly comparable to the NTM's reported test RMSE.

    Args:
        coeffs: Output of ``fit_substitution_loss`` (uses ``k_v``, ``k_h``).
        df: Rows to evaluate (e.g. the test split).
        analytical_model: Baseline model exposing ``calculate_torque``.
        n_ppairs: Pole-pair count ``p``.
        features_fn: Basis builder; must match the one used to fit ``coeffs``.

    Returns:
        RMSE [Nm].
    """
    omega = df[OMEGA_COL].to_numpy(dtype=np.float64)
    volt = df[list(VOLT_COLS)].to_numpy(dtype=np.float64)
    t_meas = df[TORQ_COL].to_numpy(dtype=np.float64)

    t_base = _baseline_torque(df, analytical_model)
    feats = features_fn(omega, volt, n_ppairs)

    coef = coeffs.get("coef")
    if coef is None:
        names = ("k_v", "k_h", "k_e", "k_4", "k_5")[: len(feats)]
        coef = np.array([coeffs[name] for name in names], dtype=np.float64)
    m_fe = np.column_stack(feats) @ np.asarray(coef, dtype=np.float64)
    t_pred = t_base + m_fe

    return float(np.sqrt(np.mean((t_pred - t_meas) ** 2)))
