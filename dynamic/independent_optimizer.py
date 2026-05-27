"""
Independent per-angle optimizer -- relaxation R0 from ``Dynamic.tex``.

At each grid angle ``theta_n`` the dynamic problem decouples into a
single static MTPA-like problem with the for-all-theta constraints
replaced by their pointwise evaluation at theta_n:

    min   x_n^T x_n
    s.t.  T(x_n) = T*
          | h(theta_n)   x_n            | <= I_max
          | g(theta_n)   x_n + bemf(n)  | <= V_max

where ``h(theta_n) = mat_dq_to_ph[n_idx, :]`` and
``g(theta_n) = mat_curr_dq_to_volt_ph[n_idx, :]`` are the rows of the
existing Transform matrices, and ``bemf(n)`` is the phase-A BEMF sample
at theta_n. This reduces to the existing static problem in the limit
where the worst-case-over-theta is replaced by a single angle.

The result is a per-angle trajectory ``curr_dq_grid`` of shape
``(N, dim)``, periodic by construction (each angle is independent so
the trajectory is also evaluable at the wrap-point).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from current_setpoints.optimization.models import BaseTorqueModel
from current_setpoints.simulation import Transform


def _angle_to_index(theta: float, transform: Transform) -> int:
    """Snaps theta to the nearest sample of ``transform.vec_theta``."""
    n_theta = transform.vec_theta.size - 1
    return int(np.round((theta % (2 * np.pi)) / (2 * np.pi) * n_theta)) % n_theta


def _per_angle_constraints(
    transform: Transform,
    omega: float,
    n_idx: int,
    curr_max: float,
    volt_max: float,
    bemf_ph_sample: float,
) -> list[dict[str, object]]:
    """
    Builds 4 SLSQP inequality constraints (two-sided current + two-sided
    voltage) evaluated only at angle index ``n_idx``.
    """
    h_row = transform.mat_dq_to_ph[n_idx, :].copy()
    g_row = transform.mat_curr_dq_to_volt_ph[n_idx, :].copy()

    def cur_upper(x: np.ndarray) -> float:
        return curr_max - h_row @ x

    def cur_lower(x: np.ndarray) -> float:
        return curr_max + h_row @ x

    def volt_upper(x: np.ndarray) -> float:
        return volt_max - (g_row @ x + bemf_ph_sample)

    def volt_lower(x: np.ndarray) -> float:
        return volt_max + (g_row @ x + bemf_ph_sample)

    return [
        {"type": "ineq", "fun": cur_upper},
        {"type": "ineq", "fun": cur_lower},
        {"type": "ineq", "fun": volt_upper},
        {"type": "ineq", "fun": volt_lower},
    ]


def run_independent_per_angle(
    *,
    model: BaseTorqueModel,
    transform: Transform,
    omega: float,
    torq_target: float,
    curr_max: float,
    volt_max: float,
    n_grid: int = 64,
    warm_start: np.ndarray | None = None,
    opts: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the independent-per-angle (R0) relaxation of the dynamic
    problem at a fixed (T*, omega) operating point.

    Args:
        model: Torque model exposing ``calculate_torque(omega, curr_dq)``.
        transform: ``Transform`` instance for the machine. Must have been
            initialised consistently with the machine and flux providers.
        omega: Electrical speed [rad/s] -- low values in this experiment.
        torq_target: T* [Nm].
        curr_max, volt_max: physical limits [A], [V] (machine.curr_max etc.).
        n_grid: number of independent angles on ``[0, 2*pi)``.
        warm_start: optional ``(dim,)`` seed for the first angle; subsequent
            angles use the previous successful solution as a warm-start.
        opts: SLSQP options. Defaults to ``{"ftol": 1e-8, "maxiter": 500}``.

    Returns:
        ``(theta_grid, curr_dq_grid, ok_grid)`` of shapes
        ``(N,)``, ``(N, dim)``, ``(N,)`` (bool). Entries where the solver
        failed are filled with ``np.nan`` in ``curr_dq_grid``.
    """
    if opts is None:
        opts = {"disp": False, "ftol": 1e-8, "maxiter": 500, "eps": 1e-8}

    transform._set_omega(omega)
    dim = transform.dim

    flux_volt, _ = transform.flux.get_flux(omega, np.zeros(dim))
    bemf_dq = omega * transform.machine.mat_crossc @ flux_volt
    bemf_ph_all = transform.mat_dq_to_ph @ bemf_dq

    theta_grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    n_theta = transform.vec_theta.size - 1
    idx_grid = (np.round(theta_grid / (2 * np.pi) * n_theta).astype(int)) % n_theta

    curr_dq_grid = np.full((n_grid, dim), np.nan)
    ok_grid = np.zeros(n_grid, dtype=bool)

    seed = warm_start.copy() if warm_start is not None else np.array([1.0] + [0.0] * (dim - 1))
    if seed.shape[0] != dim:
        raise ValueError(f"warm_start must have length {dim}, got {seed.shape[0]}.")

    for i, n_idx in enumerate(idx_grid):
        constraints = _per_angle_constraints(
            transform, omega, int(n_idx), curr_max, volt_max, float(bemf_ph_all[n_idx])
        )

        def eq_torq(x: np.ndarray, _T=torq_target) -> float:
            return model.calculate_torque(omega, x) - _T

        constraints.append({"type": "eq", "fun": eq_torq})

        def objective(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        candidates = model.get_candidates(seed) if hasattr(model, "get_candidates") else [seed]

        best_fun = float("inf")
        best_x: np.ndarray | None = None
        for x0 in candidates:
            res = minimize(objective, x0, method="SLSQP", constraints=constraints, options=opts)
            if res.success and res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x

        if best_x is not None:
            curr_dq_grid[i] = best_x
            ok_grid[i] = True
            seed = best_x

    return theta_grid, curr_dq_grid, ok_grid


__all__ = ["run_independent_per_angle"]
