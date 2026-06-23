from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..simulation import BaseTransform
from .constraints import current_constraint, voltage_constraint
from .models import BaseTorqueModel


class MotorOptimizer:
    """
    Static-i_s solver for the paper's optimization (Eq. 10 / 20 / 31 /
    42). Finds the DQ current vector that either maximises torque or
    delivers a target torque with minimum ``i_sᵀ·i_s``, subject to the
    per-phase current and voltage envelopes evaluated at the discretized
    rotor angles (paper Eq. 10c / 10d etc.) and any transform-specific
    extras.

    Note: in the stateless API, ``omega`` is passed explicitly to every
    public method; neither the torque model nor the machine carries any
    operating-point state. The transform internally caches matrices for
    the last omega it saw (see ``BaseTransform._set_omega``), so within
    a single SLSQP run that holds omega fixed, repeated
    constraint/objective evaluations short-circuit the matrix
    recomputation.

    Two-fault transforms (``TransformFault2Adjacent``,
    ``TransformFault2NonAdjacent``) contribute the null-space equality
    ``N @ i_s = 0`` (paper Eq. 23 / 34) via
    ``transform.get_extra_constraints``; the optimizer picks it up
    automatically. Healthy and single-fault transforms return an empty
    list there, so the extra constraints are a no-op for them.
    """

    def __init__(self, model: BaseTorqueModel, opts: dict[str, Any] | None = None) -> None:
        """
        Initializes the optimizer with a torque model and solver settings.

        Args:
            model: The torque model instance (Analytical or Neural).
            opts: Dictionary of SLSQP solver options.
                  Defaults to {"disp": False, "ftol": 1e-8, "maxiter": 500}.
        """
        self.model = model
        self.opts = opts if opts is not None else {"disp": False, "ftol": 1e-8, "maxiter": 500}

    def _get_base_constraints(self, omega: float, transform: BaseTransform) -> list[dict[str, Any]]:
        """
        Constructs the physical inequality constraints (paper Eq. 10c/d,
        20c/d, 31d/e, 42d/e — phase-current and phase-voltage envelopes
        evaluated over a discretized rotor angle grid) as closures bound
        to the current operating speed, plus any transform-specific
        extras — in particular the null-space equality ``N @ i_s = 0``
        contributed by two-fault transforms (paper Eq. 23 / 34).

        Args:
            omega: Electrical speed [rad/s] for which constraints will be evaluated.
            transform: The transform instance providing the speed-dependent matrices.

        Returns:
            List[Dict]: SciPy-compatible constraint definitions.
        """
        machine = transform.machine
        curr_max = machine.curr_max
        volt_max = machine.volt_max

        def current_cons(vec_curr_dq: np.ndarray) -> float:
            return current_constraint(omega, vec_curr_dq, curr_max, transform)

        def voltage_cons(vec_curr_dq: np.ndarray) -> float:
            return voltage_constraint(omega, vec_curr_dq, volt_max, transform)

        return [
            {"type": "ineq", "fun": current_cons},
            {"type": "ineq", "fun": voltage_cons},
            *transform.get_extra_constraints(),
        ]

    def _run_optimization(
        self,
        objective_fun: Callable[[np.ndarray], float],
        constraints: list[dict[str, Any]],
        candidates: list[np.ndarray],
        opts: dict[str, Any],
    ) -> tuple[np.ndarray, float, bool]:
        """
        Internal multi-start solver engine. Executes SLSQP from multiple
        starting points to avoid local minima.

        Returns:
            Tuple: (optimal_vector, best_cost_value, success_flag).
        """
        best_res = None
        best_val = float("inf")

        for vec_start in candidates:
            res = minimize(
                objective_fun,
                vec_start,
                method="SLSQP",
                constraints=constraints,
                options=opts,
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_res = res

        if best_res is not None:
            return best_res.x, best_res.fun, True

        nan_vector = np.full(candidates[0].shape, np.nan)
        return nan_vector, float("inf"), False

    def maximize_torque(
        self,
        omega: float,
        transform: BaseTransform,
        vec_curr_guess: np.ndarray | None = None,
        opts: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, float, bool]:
        """
        Finds the current vector that produces the maximum possible torque
        under current and voltage limits at the given speed.

        Args:
            omega: Electrical speed [rad/s].
            transform: The transform instance providing speed-dependent matrices.
            vec_curr_guess: Optional warm-start added to the multi-start candidates.
            opts: Solver options (overrides class defaults).

        Returns:
            Tuple: (optimal_dq_currents, max_torque_value, success).
        """
        opts = opts if opts is not None else self.opts
        constraints = self._get_base_constraints(omega, transform)
        candidates = self.model.get_candidates(vec_curr_guess)

        def objective(vec_curr_dq: np.ndarray) -> float:
            return -self.model.calculate_torque(omega, vec_curr_dq)

        vec_curr_dq_best, best_val, success = self._run_optimization(objective, constraints, candidates, opts)

        return vec_curr_dq_best, -best_val, success

    def minimize_current(
        self,
        torq_target: float,
        omega: float,
        transform: BaseTransform,
        vec_curr_guess: np.ndarray | None = None,
        opts: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Finds the minimum-current vector (MTPA logic) that produces a specific
        target torque at the given speed.

        Args:
            torq_target: The required torque [Nm].
            omega: Electrical speed [rad/s].
            transform: The transform instance providing speed-dependent matrices.
            vec_curr_guess: Optional warm-start (typically previous grid step).
            opts: Solver options.

        Returns:
            Tuple: (optimal_dq_currents, success).
        """
        opts = opts if opts is not None else self.opts
        constraints = self._get_base_constraints(omega, transform)
        candidates = self.model.get_candidates(vec_curr_guess)

        def eq_cons(vec_curr_dq: np.ndarray) -> float:
            return self.model.calculate_torque(omega, vec_curr_dq) - torq_target

        constraints.append({"type": "eq", "fun": eq_cons})

        def objective(vec_curr_dq: np.ndarray) -> float:
            return float(np.sum(vec_curr_dq**2))

        vec_curr_dq_best, _, success = self._run_optimization(objective, constraints, candidates, opts)

        return vec_curr_dq_best, success
