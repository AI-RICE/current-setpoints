"""
Identification of the Morimoto core-loss resistances (``R_c1``, ``R_c3``) for
``ModelLossParametric``, plus the train/test split used to fit them.

The iron-loss torque model (see ``ModelLossParametric``) is

    T_Rc = T_base - p_p * omega * (||lam_1||^2 / R_c1 + 9 ||lam_3||^2 / R_c3)

with the back-EMF flux ``lam = Psi_PM + L_s i_s`` ordered ``[d1, q1, d3, q3]``.
Writing the loss torque ``L = T_base - T_measured`` and the per-plane
conductances ``g1 = 1/R_c1``, ``g3 = 1/R_c3``, the model is LINEAR in
``(g1, g3)``:

    L = phi_1 * g1 + phi_3 * g3,
        phi_1 =     p_p * omega * ||lam_1||^2,
        phi_3 = 9 * p_p * omega * ||lam_3||^2.

so the resistances are identified by ordinary least squares on the measured
operating points. A conductance is inverted to its resistance ``R_c = 1/g``
(``+inf`` if a conductance is exactly zero). On a poor fit OLS may return a
negative conductance -> a non-physical negative ``R_c``, which is left as-is so
the inadequacy is visible rather than hidden.

``split_like_neural`` reproduces the exact 85 / 15 holdout the NTM was trained
on (``test_size=0.15``, ``random_state=42``), so a baseline identified here is
evaluated on the identical test points and its RMSE is directly comparable to
the NTM's. This two-plane form assumes the 5-phase dq layout
``[d1, q1, d3, q3]``, matching ``ModelLossParametric``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from ..parameters import BaseMachine, Flux

# Holdout split shared with the neural torque model (NTM). These mirror the
# train_test_split call in the trainer notebook and build_residual_test_set;
# leave them untouched to keep the parametric baseline comparable to the NTM.
NTM_TEST_SIZE: float = 0.15
NTM_RANDOM_STATE: int = 42


def split_like_neural(
    X: np.ndarray,
    y_measured: np.ndarray,
    test_size: float = NTM_TEST_SIZE,
    random_state: int = NTM_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train/test split identical (same rows) to the NTM holdout split.

    sklearn's ``train_test_split`` partitions by row index from
    ``(n_samples, test_size, random_state)`` alone -- independent of the target
    values -- so splitting ``(X, y_measured)`` here assigns the same operating
    points to train/test as the NTM's ``(X, residual)`` split, *provided* the
    defaults are left untouched and ``X`` holds the same rows in the same order
    the NTM was fed. Targets are returned as RAW MEASURED torque (what
    ``fit_core_loss_resistances`` expects), not the NTM residual.

    Args:
        X: Inputs, columns ``[omega, i_d1, i_q1, i_d3, i_q3]``.
        y_measured: Measured torque, shape ``(N,)`` or ``(N, 1)``.
        test_size: Held-out fraction (leave at 0.15 to match the NTM).
        random_state: Split seed (leave at 42 to match the NTM).

    Returns:
        ``(X_train, X_test, y_train, y_test)`` with measured-torque targets
        shaped ``(n, 1)``.
    """
    X = np.asarray(X, dtype=np.float64)
    y_measured = np.asarray(y_measured, dtype=np.float64).reshape(-1, 1)
    return train_test_split(X, y_measured, test_size=test_size, random_state=random_state)


def _core_loss_design(
    X: np.ndarray,
    analytical_model: Any,
    flux: Flux,
    machine: BaseMachine,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds the iron-loss design matrix and the analytical baseline torque.

    Args:
        X: Inputs ``[omega, i_d1, i_q1, i_d3, i_q3]``.
        analytical_model: Baseline torque model exposing
            ``calculate_torque(omega, curr_dq) -> float``.
        flux: Flux provider; its voltage-equation flux (``flux_volt``) defines
            the back-EMF flux, matching ``ModelLossParametric``.
        machine: Machine object supplying ``L_stat`` and ``n_ppairs`` (p_p).

    Returns:
        ``(Phi, T_base)`` where ``Phi`` has columns ``[phi_1, phi_3]`` (shape
        ``(N, 2)``) and ``T_base`` is the analytical torque per row (shape
        ``(N,)``).
    """
    X = np.asarray(X, dtype=np.float64)
    L_stat = machine.L_stat
    p_p = machine.n_ppairs

    phi1 = np.empty(len(X))
    phi3 = np.empty(len(X))
    t_base = np.empty(len(X))
    for k, row in enumerate(X):
        w = float(row[0])
        curr_dq = row[1:]
        flux_volt, _ = flux.get_flux(w, curr_dq)
        lam = flux_volt + L_stat @ curr_dq
        norm_sq_1 = lam[0] ** 2 + lam[1] ** 2
        norm_sq_3 = lam[2] ** 2 + lam[3] ** 2
        phi1[k] = p_p * w * norm_sq_1
        phi3[k] = 9.0 * p_p * w * norm_sq_3
        t_base[k] = analytical_model.calculate_torque(w, curr_dq)

    return np.column_stack([phi1, phi3]), t_base


def fit_core_loss_resistances(
    X_train: np.ndarray,
    y_measured_train: np.ndarray,
    analytical_model: Any,
    flux: Flux,
    machine: BaseMachine,
) -> dict[str, Any]:
    """
    Identifies the per-plane core-loss resistances ``R_c1``, ``R_c3`` by
    ordinary least squares on the loss torque ``L = T_base - T_measured``.

    Args:
        X_train: Training inputs ``[omega, i_d1, i_q1, i_d3, i_q3]``.
        y_measured_train: Measured torque on the same rows, shape ``(N,)`` or
            ``(N, 1)``.
        analytical_model: Baseline torque model exposing
            ``calculate_torque(omega, curr_dq) -> float`` (e.g.
            ``ModelAnalytical``). Passed in (not imported) to avoid a
            utils -> optimization import cycle.
        flux: Flux provider; its voltage-equation flux (``flux_volt``) defines
            the back-EMF flux, matching ``ModelLossParametric``.
        machine: Machine object supplying ``L_stat`` and ``n_ppairs`` (p_p).

    Returns:
        Dict with:
            ``R_c1``, ``R_c3`` : identified resistances [Ohm] (``np.inf`` when a
                plane's conductance is exactly zero; negative if OLS returns a
                negative conductance, flagging an inadequate fit),
            ``g_c1``, ``g_c3`` : the fitted conductances ``1/R_c`` [1/Ohm],
            ``r2``       : coefficient of determination on the training loss torque,
            ``rmse``     : training RMSE on the loss torque [Nm] (equal to the
                residual RMSE, hence comparable to the NTM),
            ``n_points`` : number of training points.
    """
    phi, t_base = _core_loss_design(X_train, analytical_model, flux, machine)
    y_meas = np.asarray(y_measured_train, dtype=np.float64).reshape(-1)
    loss_torque = t_base - y_meas  # analytical over-prediction

    g, *_ = np.linalg.lstsq(phi, loss_torque, rcond=None)
    g_c1, g_c3 = float(g[0]), float(g[1])

    r_c1 = 1.0 / g_c1 if g_c1 != 0.0 else np.inf
    r_c3 = 1.0 / g_c3 if g_c3 != 0.0 else np.inf

    loss_pred = phi @ g
    ss_res = float(np.sum((loss_torque - loss_pred) ** 2))
    ss_tot = float(np.sum((loss_torque - loss_torque.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((loss_torque - loss_pred) ** 2)))

    return {
        "R_c1": r_c1,
        "R_c3": r_c3,
        "g_c1": g_c1,
        "g_c3": g_c3,
        "r2": r2,
        "rmse": rmse,
        "n_points": int(len(loss_torque)),
    }


def core_loss_rmse(
    model: dict[str, Any],
    X: np.ndarray,
    y_measured: np.ndarray,
    analytical_model: Any,
    flux: Flux,
    machine: BaseMachine,
) -> float:
    """
    Residual-torque RMSE of an identified core-loss model on ``(X, y_measured)``.

    Equivalent to the RMSE of ``ModelLossParametric.calculate_torque`` against
    the measured torque, but computed without importing the optimization
    package -- directly comparable to the NTM's reported test RMSE.

    Args:
        model: Output of ``fit_core_loss_resistances`` (uses ``g_c1``, ``g_c3``).
        X: Inputs ``[omega, i_d1, i_q1, i_d3, i_q3]``.
        y_measured: Measured torque, shape ``(N,)`` or ``(N, 1)``.
        analytical_model: As in ``fit_core_loss_resistances``.
        flux: As in ``fit_core_loss_resistances``.
        machine: As in ``fit_core_loss_resistances``.

    Returns:
        RMSE in Nm.
    """
    phi, t_base = _core_loss_design(X, analytical_model, flux, machine)
    y_meas = np.asarray(y_measured, dtype=np.float64).reshape(-1)
    loss_torque = t_base - y_meas
    g = np.array([model["g_c1"], model["g_c3"]])
    loss_pred = phi @ g
    return float(np.sqrt(np.mean((loss_torque - loss_pred) ** 2)))
