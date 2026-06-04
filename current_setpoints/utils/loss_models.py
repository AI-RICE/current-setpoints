"""
Parametric loss-torque baselines for benchmarking against the neural torque
model (NTM).

Includes classical physics-based terms and an Empirical Polynomial Loss Model.
The Empirical Polynomial baseline maps macroscopic vehicle powertrain losses 
into a torque equivalent:

    T_loss(w, i_s) = T_c + B*w + k_w*w**2 + k_c*||i_s||**2

    T_c              : constant (friction stiction / Coulomb)
    B * w            : viscous friction 
    k_w * w**2       : aerodynamic windage & high-order speed losses
    k_c * ||i_s||**2 : macroscopic load-dependent losses (proxy for T^2)

``w`` is the electrical speed and ``||i_s||`` the full stator-current magnitude
(3rd harmonic included). All are linear in their coefficients and fit by 
ordinary least squares.

Models are compared against the NTM on the *same* residual target
(``measured - analytical``), so the reported RMSEs are directly comparable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Physics and Empirical feature terms (T_c is always fit as the intercept).
SPEED_ONLY_TERMS: tuple[str, ...] = ("w", "w2")
FULL_TERMS: tuple[str, ...] = ("w", "w2", "w_i", "w2_i2")
IRON_LOSS_TERMS: tuple[str, ...] = ("w", "is2", "w_i2")

# The new macroscopic empirical polynomial baseline
EMPIRICAL_POLYNOMIAL_TERMS: tuple[str, ...] = ("w", "w2", "is2")

# Coefficient name + unit per term, for reporting.
_TERM_INFO: dict[str, tuple[str, str]] = {
    "w": ("B", "Nm/(rad/s)"),
    "w2": ("k_w", "Nm/(rad/s)^2"),
    "w_i": ("a", "Nm/((rad/s)*A)"),
    "w2_i2": ("b", "Nm/((rad/s)^2*A^2)"),
    "is2": ("k_c", "Nm/A^2"),  # Mapped to k_c for the empirical polynomial
    "w_i2": ("k_e", "Nm/((rad/s)*A^2)"),
}


def parametric_loss_features(X: np.ndarray, terms: tuple[str, ...]) -> np.ndarray:
    """
    Builds the (intercept-free) physics feature matrix for the loss model.

    Args:
        X: Inputs, columns ``[omega, i_d1, i_q1, i_d3, i_q3]``.
        terms: Subset of ``{"w", "w2", "w_i", "w2_i2", "is2", "w_i2"}`` to include.

    Returns:
        Feature matrix of shape ``(N, len(terms))``. ``||i_s||`` is the norm of
        all four dq current components (3rd harmonic included).
    """
    X = np.asarray(X, dtype=np.float64)
    omega = X[:, 0]
    i_norm = np.linalg.norm(X[:, 1:5], axis=1)
    
    library = {
        "w": omega,
        "w2": omega**2,
        "w_i": omega * i_norm,
        "w2_i2": omega**2 * i_norm**2,
        "is2": i_norm**2,
        "w_i2": omega * i_norm**2,
    }
    return np.column_stack([library[t] for t in terms])


def fit_parametric_loss(
    X_train: np.ndarray,
    T_loss_train: np.ndarray,
    terms: tuple[str, ...] = EMPIRICAL_POLYNOMIAL_TERMS,
) -> dict[str, Any]:
    """
    Fits the parametric loss-torque model by ordinary least squares.

    Feature columns are standardized internally for conditioning (speed and
    speed**2 are strongly collinear); coefficients are returned in physical
    units. ``T_c`` is fit as the intercept.

    Args:
        X_train: Training inputs ``[omega, i_d1, i_q1, i_d3, i_q3]``.
        T_loss_train: Training loss-torque target (``analytical - measured``),
            so fitted coefficients are positive and physically interpretable.
        terms: Which physics terms to include.

    Returns:
        Dict with ``terms``, ``intercept`` (T_c), ``coef`` (raw-unit array
        aligned with ``terms``), ``named`` (``{coeff_name: value}``), and the
        training ``r2``.
    """
    Phi = parametric_loss_features(X_train, terms)
    T = np.asarray(T_loss_train, dtype=np.float64).ravel()

    mu = Phi.mean(axis=0)
    sd = Phi.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    Phi_s = (Phi - mu) / sd

    A = np.column_stack([np.ones(len(Phi_s)), Phi_s])
    beta, *_ = np.linalg.lstsq(A, T, rcond=None)

    coef = beta[1:] / sd
    intercept = float(beta[0] - np.sum(beta[1:] * mu / sd))

    named = {"T_c": intercept}
    for term, c in zip(terms, coef):
        named[_TERM_INFO[term][0]] = float(c)

    model = {"terms": terms, "intercept": intercept, "coef": coef, "named": named}
    model["r2"] = parametric_loss_r2(model, X_train, T)
    return model


def predict_parametric_loss(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predicts loss torque for inputs ``X`` from a fitted model."""
    Phi = parametric_loss_features(X, model["terms"])
    return Phi @ model["coef"] + model["intercept"]


def parametric_loss_r2(model: dict[str, Any], X: np.ndarray, T_loss: np.ndarray) -> float:
    """Coefficient of determination of a fitted model on ``(X, T_loss)``."""
    T = np.asarray(T_loss, dtype=np.float64).ravel()
    pred = predict_parametric_loss(model, X)
    ss_res = float(np.sum((T - pred) ** 2))
    ss_tot = float(np.sum((T - T.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def loss_model_rmse(model: dict[str, Any], X: np.ndarray, T_loss: np.ndarray) -> float:
    """RMSE of a fitted loss model on ``(X, T_loss)`` (Nm)."""
    T = np.asarray(T_loss, dtype=np.float64).ravel()
    pred = predict_parametric_loss(model, X)
    return float(np.sqrt(np.mean((pred - T) ** 2)))