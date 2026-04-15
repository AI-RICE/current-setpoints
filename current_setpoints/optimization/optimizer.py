from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .constraints import current_constraint, voltage_constraint


class MotorOptimizer:
    """
    Unified solver class for motor control optimization problems.
    Provides methods to find maximum torque or minimum current operating points
    for any n-phase machine.
    """

    def __init__(self, model: Any, opts: Optional[Dict[str, Any]] = None) -> None:
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

    def _get_base_constraints(self, transform: Any) -> List[Dict[str, Any]]:
        """
        Constructs the physical inequality constraints (Current and Voltage limits).

        Args:
            transform: The Transform instance providing current speed/voltage matrices.

        Returns:
            List[Dict]: Scipy-compatible constraint definitions.
        """

        def safe_current_constraint(vec_curr_dq):
            self.model.machine.update_state(transform.omega, vec_curr_dq)
            return current_constraint(vec_curr_dq, self.model.machine, transform)

        def safe_voltage_constraint(vec_curr_dq):
            self.model.machine.update_state(transform.omega, vec_curr_dq)
            return voltage_constraint(vec_curr_dq, self.model.machine, transform)

        return [
            {
                "type": "ineq",
                "fun": safe_current_constraint,
            },
            {
                "type": "ineq",
                "fun": safe_voltage_constraint,
            },
        ]

    def _run_optimization(
        self,
        objective_fun: Callable[[np.ndarray], float],
        constraints: List[Dict[str, Any]],
        candidates: List[np.ndarray],
        opts: Dict[str, Any],
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Internal multi-start solver engine. Executes SLSQP from multiple starting points
        to avoid local minima and find the global optimum.

        Args:
            objective_fun: The scalar function to minimize.
            constraints: List of scipy constraints.
            candidates: List of starting N-dimensional DQ vectors.
            opts: Solver options for minimize().

        Returns:
            Tuple: (optimal_vector, best_cost_value, success_flag)
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
        transform: Any,
        vec_curr_guess: Optional[np.ndarray] = None,
        opts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Finds the current vector that produces the maximum possible torque
        under current and voltage limits.

        Args:
            transform: The Transform instance for the current operating speed.
            vec_curr_guess: Optional external guess to add to multi-start candidates.
            opts: Method-specific solver options (overrides class defaults).

        Returns:
            Tuple: (optimal_dq_currents, max_torque_value, success)
        """
        opts = opts if opts is not None else self.opts
        constraints = self._get_base_constraints(transform)
        candidates = self.model.get_candidates(vec_curr_guess)

        def objective(vec_curr_dq: np.ndarray) -> float:
            self.model.machine.update_state(transform.omega, vec_curr_dq)

            return -self.model.calculate_torque(vec_curr_dq)

        vec_curr_dq_best, best_val, success = self._run_optimization(
            objective, constraints, candidates, opts
        )

        return vec_curr_dq_best, -best_val, success

    def minimize_current(
        self,
        torq_target: float,
        transform: Any,
        vec_curr_guess: Optional[np.ndarray] = None,
        opts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Finds the minimum current magnitude (Maximum Torque Per Ampere logic)
        required to produce a specific target torque.

        Args:
            torq_target: The required torque in Nm.
            transform: The Transform instance for the current operating speed.
            vec_curr_guess: Optional external guess (usually from previous grid step).
            opts: Method-specific solver options.

        Returns:
            Tuple: (optimal_dq_currents, success)
        """
        opts = opts if opts is not None else self.opts
        constraints = self._get_base_constraints(transform)
        candidates = self.model.get_candidates(vec_curr_guess)

        def eq_cons(vec_curr_dq: np.ndarray) -> float:
            self.model.machine.update_state(transform.omega, vec_curr_dq)

            return self.model.calculate_torque(vec_curr_dq) - torq_target

        constraints.append({"type": "eq", "fun": eq_cons})

        def objective(vec_curr_dq: np.ndarray) -> float:
            return np.sum(vec_curr_dq**2)

        vec_curr_dq_best, _, success = self._run_optimization(
            objective, constraints, candidates, opts
        )

        return vec_curr_dq_best, success
