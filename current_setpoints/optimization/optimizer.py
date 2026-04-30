from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..model import Transform
from .constraints import current_constraint, voltage_constraint
from .models import BaseTorqueModel


class MotorOptimizer:
    """
    Unified solver class for motor control optimization problems.
    Provides methods to find maximum torque or minimum current operating points
    for any n-phase machine.

    Note: in the new stateless API, ``omega`` is passed explicitly to every
    public method; neither the torque model nor the machine carries any
    operating-point state. The transform internally caches matrices for the
    last omega it saw (see ``Transform._set_omega``), so within a single
    SLSQP run that holds omega fixed, repeated constraint/objective
    evaluations short-circuit the matrix recomputation.
    """

    def __init__(
        self, model: BaseTorqueModel, opts: dict[str, Any] | None = None
    ) -> None:
        """
        Initializes the optimizer with a torque model and solver settings.

        Args:
            model: The torque model instance (Analytical or Neural).
            opts: Dictionary of SLSQP solver options.
                  Defaults to {"disp": False, "ftol": 1e-8, "maxiter": 500}.
        """
        self.model = model
        self.opts = (
            opts if opts is not None else {"disp": False, "ftol": 1e-8, "maxiter": 500}
        )

    def _get_base_constraints(
        self, omega: float, transform: Transform
    ) -> list[dict[str, Any]]:
        """
        Constructs the physical inequality constraints (current and voltage limits)
        as closures bound to the current operating speed.

        Args:
            omega: Electrical speed [rad/s] for which constraints will be evaluated.
            transform: The Transform instance providing the speed-dependent matrices.

        Returns:
            List[Dict]: SciPy-compatible constraint definitions.
        """
        machine = self.model.machine
        curr_max = machine.curr_max
        volt_max = machine.volt_max

        def current_cons(vec_curr_dq: np.ndarray) -> float:
            return current_constraint(omega, vec_curr_dq, curr_max, transform)

        def voltage_cons(vec_curr_dq: np.ndarray) -> float:
            return voltage_constraint(omega, vec_curr_dq, volt_max, transform)

        return [
            {"type": "ineq", "fun": current_cons},
            {"type": "ineq", "fun": voltage_cons},
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
        transform: Transform,
        vec_curr_guess: np.ndarray | None = None,
        opts: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, float, bool]:
        """
        Finds the current vector that produces the maximum possible torque
        under current and voltage limits at the given speed.

        Args:
            omega: Electrical speed [rad/s].
            transform: The Transform instance providing speed-dependent matrices.
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

        vec_curr_dq_best, best_val, success = self._run_optimization(
            objective, constraints, candidates, opts
        )

        return vec_curr_dq_best, -best_val, success

    def minimize_current(
        self,
        torq_target: float,
        omega: float,
        transform: Transform,
        vec_curr_guess: np.ndarray | None = None,
        opts: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Finds the minimum-current vector (MTPA logic) that produces a specific
        target torque at the given speed.

        Args:
            torq_target: The required torque [Nm].
            omega: Electrical speed [rad/s].
            transform: The Transform instance providing speed-dependent matrices.
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

        vec_curr_dq_best, _, success = self._run_optimization(
            objective, constraints, candidates, opts
        )

        return vec_curr_dq_best, success
