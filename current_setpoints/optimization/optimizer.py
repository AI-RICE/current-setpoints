import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Optional, Tuple, List, Callable
from constraints import current_constraint, voltage_constraint


class MotorOptimizer:
    """
    Unified solver class for motor control optimization problems.
    Provides methods to find maximum torque or minimum current operating points.
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
        return [
            {
                "type": "ineq",
                "fun": current_constraint,
                "args": (self.model.machine, transform),
            },
            {
                "type": "ineq",
                "fun": voltage_constraint,
                "args": (self.model.machine, transform),
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
            candidates: List of starting DQ vectors [id1, iq1, id3, iq3].
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
            # Select the result with the lowest cost among successful runs
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_res = res

        # If at least one start was successful, return the best one
        if best_res is not None:
            return best_res.x, best_res.fun, True

        # Fallback
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
            # Negated because minimize() finds the minimum
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

        # Append equality constraint: produced torque must equal target torque
        def eq_cons(vec_curr_dq: np.ndarray) -> float:
            return self.model.calculate_torque(vec_curr_dq) - torq_target

        constraints.append({"type": "eq", "fun": eq_cons})

        def objective(vec_curr_dq: np.ndarray) -> float:
            # Minimize squared magnitude of current vector
            return np.sum(vec_curr_dq**2)

        vec_curr_dq_best, _, success = self._run_optimization(
            objective, constraints, candidates, opts
        )

        return vec_curr_dq_best, success
